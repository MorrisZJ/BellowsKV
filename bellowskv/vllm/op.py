"""`torch.ops.vllm.bellows_attention`: one BellowsKV-A layer's cache writes + attention, as one opaque op.

Inside the op, for layer l (position p in its group of n):
  1. write this step's k/v (RoPE'd) into the group's window cache, head slots [p*Hkv, (p+1)*Hkv);
  2. accumulate this layer's latent share (h_l @ [WdK_l | WdV_l], computed in the model code) into the shared fp32
     accumulator (p == 0 starts it);
  3. attention: decode rows (q_len 1) -> attn_ops.decode_attention (paged latent kernels + exact window incl. the
     current token, one softmax); multi-token rows -> attn_ops.chunk_attention per request (fresh prompt: plain causal
     attention; continuation chunk: latent + sinks + last W + the chunk, causal inside the chunk);
  4. p == n-1: rotate, quantize and store the group's completed latents of this step's tokens (attn_ops.latent_store).
The current step's tokens are always attended exactly (their latent is only complete at the group's last layer).

Registered as a splitting op (appended to CompilationConfig._attention_ops by `bellowskv.vllm.register`). The decode
branch uses only fixed-shape ops on persistent buffers, so it can be captured in a FULL CUDA graph.
"""
from __future__ import annotations

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from . import attn_ops

OP_NAME = "bellows_attention"
QUALIFIED_OP_NAME = f"vllm::{OP_NAME}"


def _exact_past(win_kv, block_row, block_size, heads, hd, C, sink, window):
    """Exact past tokens of one request: positions [0, min(s, C)) U [max(s, C - W), C) -> K, V (Hkv, P, hd)."""
    dev = win_kv.device
    if C <= 0:
        z = torch.zeros(heads.shape[0], 0, hd, device=dev, dtype=win_kv.dtype)
        return z, z
    rs = max(sink, C - window)
    pos = torch.cat([torch.arange(min(sink, C), device=dev), torch.arange(rs, C, device=dev)])
    slots = block_row[pos // block_size].long() * block_size + pos % block_size
    g = win_kv[(slots // block_size)[:, None], heads[None, :], (slots % block_size)[:, None], :]   # (P, Hkv, 2hd)
    return g[..., :hd].transpose(0, 1), g[..., hd:].transpose(0, 1)


def run_layer(layer, q, k, v, pkv, positions, output) -> None:
    md = get_forward_context().attn_metadata
    if isinstance(md, list):
        md = md[0]
    lat_owner, win_owner = layer.bkv_latent, layer.bkv_window
    if not isinstance(md, dict) or lat_owner.pages is None or win_owner.kv_cache.numel() == 0:
        output.zero_()                                        # profiling / dummy runs
        return
    lat_md = md[lat_owner.prefix]
    win_md = md[win_owner.prefix]
    T = q.shape[0]
    Hq, Hkv, hd = layer.num_heads, layer.num_kv_heads, layer.head_dim
    q3 = q.view(T, Hq, hd)
    k3 = k.view(T, Hkv, hd)
    v3 = v.view(T, Hkv, hd)
    out3 = output.view(T, Hq, hd)
    win_kv = win_owner.kv_cache                               # (nb, n*Hkv, bs, 2hd)
    bs = win_md.block_size
    heads = layer.bkv_heads                                   # (Hkv,) head slots of this layer
    sink, window = layer.bkv_sink, layer.bkv_window_len
    layout = lat_owner.layout
    pages = lat_owner.pages

    # 1. window write (fixed shape: slots of -1 are redirected to the null block 0)
    ws = win_md.slot_mapping[:T]
    ws = torch.where(ws >= 0, ws, torch.zeros_like(ws))
    win_kv[(ws // bs)[:, None], heads[None, :], (ws % bs)[:, None], :] = torch.cat([k3, v3], -1)

    # 2. latent accumulation (fp32, shared by all groups; sized to max tokens)
    acc = layer.bkv_acc[:T]
    if layer.bkv_pos == 0:
        acc.copy_(pkv)
    else:
        acc.add_(pkv)

    # 3. attention
    nd = lat_md.num_decodes
    nreq = lat_md.num_reqs
    if nd > 0:
        slots = win_md.win_slots[:nd]
        g = win_kv[(slots // bs)[..., None], heads, (slots % bs)[..., None], :]                     # (nd, X, Hkv, 2hd)
        Kw = g[..., :hd].permute(0, 2, 1, 3)
        Vw = g[..., hd:].permute(0, 2, 1, 3)
        from .paged_kernels import pick_nsplit
        o = attn_ops.decode_attention(
            q3[:nd], pages, layout, lat_md.block_table[:nd], lat_md.lat_len[:nd], sink, Kw, Vw,
            win_md.win_mask[:nd], layer.WuK_p, layer.WuV_p, layer.idxK, layer.idxV, layer.scaling,
            nsplit=pick_nsplit(nd, layout.R_V), ws=lat_owner.ws, lat_end=lat_md.lat_end[:nd],
            soff=lat_md.soff[:nd + 1])
        out3[:nd] = o.to(out3.dtype)
    if nd < nreq:
        qsl = lat_md.query_start_loc_cpu.tolist()
        seq = lat_md.seq_lens_cpu.tolist()
        for b in range(nd, nreq):
            a, e = qsl[b], qsl[b + 1]
            S = e - a
            if S <= 0:
                continue
            C = seq[b] - S
            Kp, Vp = _exact_past(win_kv, win_md.block_table[b], bs, heads, hd, C, sink, window)
            o = attn_ops.chunk_attention(
                q3[a:e], k3[a:e], v3[a:e], Kp, Vp, pages, layout, lat_md.block_table[b],
                sink, max(sink, C - window), layer.WuK_p, layer.WuV_p, layer.idxK, layer.idxV, layer.scaling)
            out3[a:e] = o.to(out3.dtype)
        end = qsl[nreq]
        if end < T:
            out3[end:].zero_()                                # padding tokens

    # 4. the group's last layer stores the completed latents of this step's tokens
    if layer.bkv_pos == layer.bkv_group - 1:
        cs = layer.rotary_emb.cos_sin_cache.index_select(0, positions[:T].long())
        half = cs.shape[-1] // 2
        attn_ops.latent_store(acc, cs[:, :half], cs[:, half:], lat_md.slot_mapping[:T], pages, layout,
                              (layer.rot_band, layer.rot_is_re, layer.rot_partner))


@eager_break_during_capture
def bellows_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pkv: torch.Tensor,
                      positions: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    run_layer(layer, q, k, v, pkv, positions, output)


def bellows_attention_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pkv: torch.Tensor,
                           positions: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    return


_REGISTERED = False


def register_op() -> None:
    global _REGISTERED
    if _REGISTERED or hasattr(torch.ops.vllm, OP_NAME):
        _REGISTERED = True
        return
    direct_register_custom_op(op_name=OP_NAME, op_func=bellows_attention, mutates_args=["output"],
                              fake_impl=bellows_attention_fake)
    _REGISTERED = True


register_op()
