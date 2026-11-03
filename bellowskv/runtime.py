"""BellowsKV-A at inference: the compressed cache and the attention that reads it.

Per group of n layers the cache holds ONE rotated INT4 key latent (R_K) and ONE INT4 value latent (R_V) per
token; every layer of the group reads both through its own read-heads. Per layer it also holds an exact fp16
window: the first `sink` tokens and the most recent tokens (`recent`, plus up to `decode_block` generated tokens
waiting to be compressed).

Forward of one layer l (position p in its group):
  1. its input h_l adds h_l @ W_down^(l) to the group's latent accumulator (the latent is additive over layers);
  2. its exact K/V for the new tokens join its window;
  3. attention = softmax over [compressed tokens (in the latent) + window tokens], merged in one softmax:
       one new token and a non-empty latent -> the INT4 Triton kernels (scores and values in the latent);
       several new tokens with a non-empty latent -> the same math in PyTorch (chunked prefill, multi-turn);
       empty latent -> SDPA over the window (a fresh prompt: plain causal attention on the original K/V,
       as in the paper, where prefill attention never sees the compressed cache);
  4. at the group's last layer the new tokens' latents are complete: rotated by their RoPE angles, and the
     oldest window tokens beyond `recent` are quantized and packed into the latent (after a prompt: all but
     the sink and the last `recent`; during generation: in blocks of `decode_block`).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from transformers.cache_utils import Cache, CacheLayerMixin

from .codec import BandRotation, dequantize, pack_int4, quantize, unpack_int4
from .kernels import DEFAULT_CFG, eq_absorbed_decode, permute_index, prepare_heads
from .calibration import layer_groups

BT = DEFAULT_CFG["BT"]


def _round_up(x, m):
    return -(-x // m) * m


class LayerWeights:
    __slots__ = ("WdK", "WdV", "WuK_p", "WuV_p")


class Runtime:
    """Model-level constants: per-layer codec weights, group layout, rotations."""

    def __init__(self, model, cfg, factors, meta):
        mc = model.config
        self.cfg = cfg; self.meta = meta
        self.L = mc.num_hidden_layers; self.n = cfg.group_size
        self.d = mc.hidden_size; self.Hq = mc.num_attention_heads; self.Hkv = mc.num_key_value_heads
        self.hd = getattr(mc, "head_dim", None) or self.d // self.Hq; self.kv = self.Hkv * self.hd
        self.G = self.Hq // self.Hkv
        self.R_K, self.R_V = cfg.r_k, cfg.r_v
        self.groups = layer_groups(self.L, self.n)
        self.group_of = {l: gi for gi, g in enumerate(self.groups) for l in g}
        self.pos_of = {l: p for g in self.groups for p, l in enumerate(g)}
        dev = next(model.parameters()).device; self.device = dev
        self.scale = model.model.layers[0].self_attn.scaling
        self.layers = {}
        self.rot = {}
        for gi, g in enumerate(self.groups):
            f = factors[g[0]]
            self.rot[gi] = BandRotation(f["ranks"].tolist(), dev)
            for p, l in enumerate(g):
                w = LayerWeights()
                w.WdK = f["WdK"][p * self.d:(p + 1) * self.d].to(dev, torch.float16).contiguous()
                w.WdV = f["WdV"][p * self.d:(p + 1) * self.d].to(dev, torch.float16).contiguous()
                wk = f["WuK"][:, p * self.kv:(p + 1) * self.kv].reshape(self.R_K, self.Hkv, self.hd).permute(1, 2, 0)
                wv = f["WuV"][:, p * self.kv:(p + 1) * self.kv].reshape(self.R_V, self.Hkv, self.hd).permute(1, 0, 2)
                w.WuK_p, w.WuV_p = prepare_heads(wk.to(dev, torch.float16).contiguous(),
                                                 wv.to(dev, torch.float16).contiguous(), DEFAULT_CFG, cfg.qgroup)
                self.layers[l] = w
        self.idxK = permute_index(self.R_K, cfg.qgroup).to(dev)            # kernel channel order, K side
        self.idxV = permute_index(self.R_V, DEFAULT_CFG["BKV"]).to(dev)    # kernel channel order, V side
        self.force_torch = False                                           # testing: PyTorch path for decode too

    def latent_bytes_per_token(self):
        side = lambda r: r // 2 + 2 * 2 * (r // self.cfg.qgroup)
        return len(self.groups) * (side(self.R_K) + side(self.R_V))

    def dense_bytes_per_token(self):
        return self.L * 2 * self.kv * 2


class Window:
    """One layer's exact tokens, (B, Hkv, cap, hd) fp16: [sink | tail], in position order."""

    def __init__(self, B, Hkv, hd, device):
        self.B, self.Hkv, self.hd, self.device = B, Hkv, hd, device
        self.K = self.V = None; self.len = 0

    def _alloc(self, cap):
        K = torch.empty(self.B, self.Hkv, cap, self.hd, device=self.device, dtype=torch.float16)
        V = torch.empty_like(K)
        if self.len:
            K[:, :, :self.len] = self.K[:, :, :self.len]; V[:, :, :self.len] = self.V[:, :, :self.len]
        self.K, self.V = K, V

    def append(self, k, v):
        S = k.shape[2]
        if self.K is None or self.len + S > self.K.shape[2]:
            self._alloc(_round_up(self.len + S + 64, 64))
        self.K[:, :, self.len:self.len + S] = k; self.V[:, :, self.len:self.len + S] = v
        self.len += S

    def drop(self, start, m, keep_cap):
        """Remove m tokens at [start, start + m); shrink the buffer if it is far larger than needed."""
        if m:
            rest = self.len - start - m
            if rest:
                self.K[:, :, start:start + rest] = self.K[:, :, start + m:self.len].clone()
                self.V[:, :, start:start + rest] = self.V[:, :, start + m:self.len].clone()
            self.len -= m
        if self.K is not None:
            cap = max(keep_cap, _round_up(self.len + 64, 64))     # never below what the window still holds
            if self.K.shape[2] > 2 * cap:
                self._alloc(cap)

    def view(self):
        return self.K[:, :, :self.len], self.V[:, :, :self.len]

    def nbytes(self):
        return 0 if self.K is None else 2 * self.K.numel() * 2


class Group:
    """The packed latents of one layer group, plus the latents of window tokens awaiting compression."""

    def __init__(self, rt, gi, B):
        self.rt, self.gi, self.B = rt, gi, B
        self.n_lat = 0; self.cap = 0
        self.cK = self.sK = self.zK = self.cV = self.sV = self.zV = None
        self.tK = None; self.tV = None                 # (B, n_tail, R) fp32: rotated K latent / V latent of tail tokens
        self.acc = None                                # (accK, accV) of the tokens in flight
        self.deq = None                                # dequantized latent for the PyTorch path, valid for one forward

    # -- packed storage
    def _ensure(self, need):
        if need <= self.cap:
            return
        cap = _round_up(max(need, int(self.cap * 1.25)) + 512, 512)
        rt, B, dev, qg = self.rt, self.B, self.rt.device, self.rt.cfg.qgroup
        new = dict(cK=torch.zeros(B, cap, rt.R_K // 2, device=dev, dtype=torch.uint8),
                   sK=torch.zeros(B, cap, rt.R_K // qg, device=dev, dtype=torch.float16),
                   zK=torch.zeros(B, cap, rt.R_K // qg, device=dev, dtype=torch.float16),
                   cV=torch.zeros(B, cap // BT, rt.R_V // 4, BT, device=dev, dtype=torch.int16),
                   sV=torch.zeros(B, cap, rt.R_V // qg, device=dev, dtype=torch.float16),
                   zV=torch.zeros(B, cap, rt.R_V // qg, device=dev, dtype=torch.float16))
        if self.cap:
            for k in ("cK", "sK", "zK", "sV", "zV"):
                new[k][:, :self.cap] = getattr(self, k)
            new["cV"][:, :self.cap // BT] = self.cV
        for k, v in new.items():
            setattr(self, k, v)
        self.cap = cap

    def _write(self, cK_rot, cV):
        """Quantize, pack and append m tokens' latents (fp32 (B, m, R_K) rotated, (B, m, R_V))."""
        cfg = self.rt.cfg
        m = cK_rot.shape[1]
        self._ensure(self.n_lat + m)
        for c0 in range(0, m, 8192):
            c1 = min(m, c0 + 8192); a, b = self.n_lat + c0, self.n_lat + c1
            q, s, z = quantize(cK_rot[:, c0:c1], cfg.bits, cfg.qgroup, cfg.clip)
            self.cK[:, a:b] = pack_int4(q); self.sK[:, a:b] = s; self.zK[:, a:b] = z
            q, s, z = quantize(cV[:, c0:c1], cfg.bits, cfg.qgroup, cfg.clip)
            words = pack_int4(q).view(torch.int16)                                  # (B, k, R_V/4)
            idx = torch.arange(a, b, device=words.device)
            self.cV.transpose(2, 3)[:, idx // BT, idx % BT] = words                 # token-contiguous tiles
            self.sV[:, a:b] = s; self.zV[:, a:b] = z
        self.n_lat += m

    def dequant(self):
        """(K latent in kernel channel order, V latent in natural order), fp16, for the PyTorch path."""
        if self.deq is None:
            rt, n, qg = self.rt, self.n_lat, self.rt.cfg.qgroup
            K = dequantize(unpack_int4(self.cK[:, :n]), self.sK[:, :n], self.zK[:, :n], qg)
            words = self.cV.transpose(2, 3).reshape(self.B, -1, rt.R_V // 4)[:, :n].contiguous()
            V = dequantize(unpack_int4(words.view(torch.uint8)), self.sV[:, :n], self.zV[:, :n], qg)
            self.deq = (K.index_select(-1, rt.idxK), V)
        return self.deq

    def nbytes(self):
        if not self.cap:
            return 0, 0
        used = self.n_lat * self.B * (self.rt.latent_bytes_per_token() // len(self.rt.groups))
        alloc = sum(getattr(self, k).numel() * getattr(self, k).element_size() for k in ("cK", "sK", "zK", "cV", "sV", "zV"))
        return used, alloc


class BellowsLayer(CacheLayerMixin):
    """Per-layer bookkeeping HuggingFace sees (lengths for masks/positions); the data live in BellowsCache."""
    is_sliding = False
    is_compileable = False
    is_croppable = False
    supports_early_init = False

    def __init__(self):
        super().__init__()
        self.seen = 0

    def lazy_initialization(self, key_states, value_states):
        pass

    def update(self, key_states, value_states, *args, **kwargs):
        raise RuntimeError("BellowsKV cache layers are written by the patched attention, not by Cache.update")

    def get_mask_sizes(self, query_length):
        return self.seen + query_length, 0

    def get_seq_length(self):
        return self.seen

    def get_max_length(self):
        return -1

    def reset(self):
        raise NotImplementedError("create a new BellowsCache instead of resetting one")

    def _unsupported(self, *a, **k):
        if self.seen:
            raise NotImplementedError("BellowsKV v1 does not support beam search / cache reordering")

    reorder_cache = batch_repeat_interleave = batch_select_indices = _unsupported

    def crop(self, *a, **k):
        raise NotImplementedError("BellowsKV v1 does not support cache cropping (assisted decoding)")


class BellowsCache(Cache):
    def __init__(self, runtime):
        super().__init__(layers=[BellowsLayer() for _ in range(runtime.L)])
        self.rt = runtime
        self.B = None
        self.groups = None; self.windows = None
        self.n_sink = 0; self.n_tail = {}                # n_tail per group index (dense leftover layers: -1)

    def _init(self, B):
        rt = self.rt; self.B = B
        self.groups = [Group(rt, gi, B) for gi in range(len(rt.groups))]
        self.windows = [Window(B, rt.Hkv, rt.hd, rt.device) for _ in range(rt.L)]
        self.n_tail = {gi: 0 for gi in range(len(rt.groups))}
        self.n_sink_g = {gi: 0 for gi in range(len(rt.groups))}

    # -- the flush policy, run at each group's last layer
    def commit(self, gi, S, cos, sin):
        rt, cfg, g = self.rt, self.rt.cfg, self.groups[gi]
        accK, accV = g.acc; g.acc = None; g.deq = None
        ns = self.n_sink_g[gi]
        a = min(cfg.sink - ns, S) if ns < cfg.sink else 0          # new tokens that fill the sink
        self.n_sink_g[gi] = ns + a
        new = S - a
        old = self.n_tail[gi]
        tail = old + new
        if S > 1:
            m = max(0, tail - cfg.recent)                           # a prompt / chunk: compress all but the recent
        elif cfg.decode_block and tail - cfg.recent >= cfg.decode_block:
            m = tail - cfg.recent
        else:
            m = 0
        kN = rt.rot[gi](accK[:, a:], cos[:, a:], sin[:, a:]) if new else None   # rotated latents of new tail tokens
        vN = accV[:, a:] if new else None
        m1 = min(m, old); m2 = m - m1                               # from the old tail, then from the new tokens
        if m:
            parts_k = ([g.tK[:, :m1]] if m1 else []) + ([kN[:, :m2]] if m2 else [])
            parts_v = ([g.tV[:, :m1]] if m1 else []) + ([vN[:, :m2]] if m2 else [])
            g._write(torch.cat(parts_k, 1) if len(parts_k) > 1 else parts_k[0],
                     torch.cat(parts_v, 1) if len(parts_v) > 1 else parts_v[0])
        keep_k = ([g.tK[:, m1:]] if g.tK is not None and old > m1 else []) + ([kN[:, m2:]] if new > m2 else [])
        keep_v = ([g.tV[:, m1:]] if g.tV is not None and old > m1 else []) + ([vN[:, m2:]] if new > m2 else [])
        g.tK = torch.cat(keep_k, 1) if keep_k else None
        g.tV = torch.cat(keep_v, 1) if keep_v else None
        self.n_tail[gi] = tail - m
        keep_cap = _round_up(cfg.sink + cfg.recent + (cfg.decode_block or 0) + 64, 64)
        for l in rt.groups[gi]:
            self.windows[l].drop(self.n_sink_g[gi], m, keep_cap)

    def clone(self):
        """Deep copy of the cache state (shares the runtime)."""
        c = BellowsCache(self.rt)
        for a, b in zip(c.layers, self.layers):
            a.seen = b.seen
        if self.groups is None:
            return c
        cp = lambda t: t.clone() if torch.is_tensor(t) else t
        c.B = self.B; c.n_tail = dict(self.n_tail); c.n_sink_g = dict(self.n_sink_g)
        c.groups = []
        for g in self.groups:
            g2 = Group(self.rt, g.gi, g.B)
            for k in ("n_lat", "cap", "cK", "sK", "zK", "cV", "sV", "zV", "tK", "tV"):
                setattr(g2, k, cp(getattr(g, k)))
            c.groups.append(g2)
        c.windows = []
        for w in self.windows:
            w2 = Window(w.B, w.Hkv, w.hd, w.device); w2.K, w2.V, w2.len = cp(w.K), cp(w.V), w.len
            c.windows.append(w2)
        return c

    def memory(self):
        """Bytes held: compressed latents (used / allocated), exact windows, and the dense fp16 K/V equivalent."""
        if self.groups is None:
            return {}
        used = sum(g.nbytes()[0] for g in self.groups); alloc = sum(g.nbytes()[1] for g in self.groups)
        win = sum(w.nbytes() for w in self.windows)
        seen = self.layers[0].seen
        dense = self.B * seen * self.rt.dense_bytes_per_token()
        return {"tokens": seen, "batch": self.B, "latent_used": used, "latent_allocated": alloc, "windows": win,
                "total_allocated": alloc + win, "dense_equivalent": dense,
                "ratio_vs_dense": dense / max(1, alloc + win)}


# ----------------------------------------------------------------------------------------------- attention
def _has_padding(mask):
    if mask is None or not torch.is_tensor(mask) or mask.dim() != 4:
        return False
    last = mask[:, :, -1, :]
    return bool((~last).any()) if mask.dtype == torch.bool else bool((last < -1).any())


def _window_sdpa(q, Kw, Vw, S):
    """q (B,Hq,S,hd) over the window (B,Hkv,T,hd) whose last S tokens are the queries themselves."""
    T = Kw.shape[2]
    if S == 1:
        return F.scaled_dot_product_attention(q, Kw, Vw, enable_gqa=True)
    if T == S:
        return F.scaled_dot_product_attention(q, Kw, Vw, is_causal=True, enable_gqa=True)
    i = torch.arange(S, device=q.device)[:, None]; j = torch.arange(T, device=q.device)[None, :]
    return F.scaled_dot_product_attention(q, Kw, Vw, attn_mask=j <= (T - S) + i, enable_gqa=True)


def latent_attention_torch(rt, w, g, q, Kw, Vw, S):
    """The decode kernel's math in PyTorch, for several queries at once: q (B,Hq,S,hd) fp16 over the group's latent
    and this layer's window (whose last S tokens are the queries). Returns (B,Hq,S,hd) fp16."""
    B, Hq, _, hd = q.shape; Hkv, G = rt.Hkv, rt.G
    CK, CV = g.dequant(); CKf, CVf = CK.float(), CV.float()
    T = Kw.shape[2]; n = g.n_lat
    qg = (q * rt.scale).view(B, Hkv, G, S, hd)
    out = torch.empty(B, Hkv, G, S, hd, device=q.device, dtype=torch.float32)
    blk = max(1, min(256, (1 << 27) // max(1, Hq * (n + T) * B)))
    j = torch.arange(T, device=q.device)
    for s0 in range(0, S, blk):
        s1 = min(S, s0 + blk); qb = qg[:, :, :, s0:s1]
        qt = torch.einsum("bkgsd,kdr->bkgsr", qb, w.WuK_p).float()                      # absorbed query
        s_lat = torch.einsum("bkgsr,btr->bkgst", qt, CKf)
        s_win = torch.einsum("bkgsd,bktd->bkgst", qb.float(), Kw.float())
        i = torch.arange(s0, s1, device=q.device)[:, None]
        s_win = s_win.masked_fill(j[None, :] > (T - S) + i, float("-inf"))
        m = torch.maximum(s_lat.amax(-1), s_win.amax(-1))[..., None]
        p_lat = torch.exp(s_lat - m); p_win = torch.exp(s_win - m)
        Ls = (p_lat.sum(-1) + p_win.sum(-1))[..., None]
        o_lat = torch.einsum("bkgst,btr->bkgsr", p_lat / Ls, CVf).index_select(-1, rt.idxV)
        o = torch.einsum("bkgsr,krd->bkgsd", o_lat, w.WuV_p.float())
        o = o + torch.einsum("bkgst,bktd->bkgsd", p_win / Ls, Vw.float())
        out[:, :, :, s0:s1] = o
    return out.view(B, Hq, S, hd).half()


def make_forward(rt, attn, l, orig_forward, apply_rope):
    grouped = l in rt.group_of
    gi = rt.group_of.get(l); p = rt.pos_of.get(l); last = grouped and p == rt.n - 1
    w = rt.layers.get(l)
    Hq, Hkv, hd = rt.Hq, rt.Hkv, rt.hd

    def forward(hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        cache = past_key_values if past_key_values is not None else kwargs.get("past_key_value")
        if not isinstance(cache, BellowsCache):
            return orig_forward(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
                                past_key_values=past_key_values, **kwargs)
        B, S, _ = hidden_states.shape
        if cache.B is None:
            cache._init(B)
        elif cache.B != B:
            raise ValueError(f"batch size changed from {cache.B} to {B}; use a new BellowsCache")
        if S > 1 and cache.layers[l].seen == 0 and _has_padding(attention_mask):
            raise NotImplementedError("BellowsKV v1 needs unpadded batches (equal-length prompts or batch size 1)")
        out_dtype = hidden_states.dtype
        h = hidden_states
        q = attn.q_proj(h).view(B, S, Hq, hd).transpose(1, 2)
        k = attn.k_proj(h).view(B, S, Hkv, hd).transpose(1, 2)
        v = attn.v_proj(h).view(B, S, Hkv, hd).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rope(q, k, cos, sin)
        q = q.half()
        win = cache.windows[l]
        win.append(k.half(), v.half())
        Kw, Vw = win.view()
        if grouped:
            g = cache.groups[gi]
            h16 = h.reshape(B * S, -1).half()
            pk = (h16 @ w.WdK).float().view(B, S, -1); pv = (h16 @ w.WdV).float().view(B, S, -1)
            if p == 0:
                g.acc = (pk, pv)
            else:
                g.acc[0].add_(pk); g.acc[1].add_(pv)
            if g.n_lat == 0:
                o = _window_sdpa(q, Kw, Vw, S)
            elif S == 1 and not rt.force_torch:
                o = eq_absorbed_decode(q[:, :, 0].contiguous(), (g.cK, g.sK, g.zK), (g.cV, g.sV, g.zV),
                                       w.WuK_p, w.WuV_p, rt.scale, window=(Kw, Vw), qgroup=rt.cfg.qgroup,
                                       S=g.n_lat).unsqueeze(2)
            else:
                o = latent_attention_torch(rt, w, g, q, Kw, Vw, S)
        else:                                                         # leftover layers outside any group: dense
            o = _window_sdpa(q, Kw, Vw, S)
        o = o.transpose(1, 2).reshape(B, S, Hq * hd).to(out_dtype)
        out = attn.o_proj(o)
        cache.layers[l].seen += S
        if last:
            cache.commit(gi, S, cos, sin)
        return out, None

    return forward
