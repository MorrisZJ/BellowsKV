"""BellowsKV-A attention math for the vLLM backend: the contract between the vLLM integration and the kernels.

Everything here is pure tensor code (no vLLM import). Each operation has a PyTorch reference (`*_ref`, correct,
not fast, not CUDA-graph-safe) and, where it matters, a Triton version with the same signature in
`paged_kernels.py`. The integration calls the dispatchers at the bottom (`latent_store`, `decode_attention`,
`chunk_attention`), which pick the Triton path when available.

Latent page (one KV block of the latent cache, `page_tokens` tokens, struct-of-arrays, natural channel order):
    K codes  [page_tokens][R_K/2]  uint8   byte j = channel 2j | channel 2j+1 << 4
    K scale  [page_tokens][R_K/qg] fp16
    K zero   [page_tokens][R_K/qg] fp16
    V codes  [page_tokens/BT][R_V/4][BT] int16   tokens contiguous inside each BT-token tile (word k = ch 4k..4k+3)
    V scale  [page_tokens][R_V/qg] fp16
    V zero   [page_tokens][R_V/qg] fp16
    pad up to page_bytes
A token at position t of a request lives in page block_table[b, t // page_tokens], row t % page_tokens. The K latent
is stored rotated (RoPE angle of its own position, band by band); quantization per token per 128 channels with
range clip, scale/zero rounded to fp16 first (identical to bellowskv.codec.quantize).

Token sets for a request with C cached tokens before this step (W = exact window, s = sink):
    latent : [s, max(s, C - W))                     read from the latent cache
    exact  : [0, min(s, C)) U [max(s, C - W), C)    read from this layer's window cache
    fresh  : this step's tokens, attended with their own k/v (causal inside the chunk)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..codec import quantize, pack_int4, unpack_int4, dequantize


@dataclass(frozen=True)
class PageLayout:
    R_K: int
    R_V: int
    page_tokens: int = 128
    qgroup: int = 128
    BT: int = 128
    page_bytes: int | None = None          # padded page size; None = payload size

    @property
    def nbk(self):
        return self.R_K // self.qgroup

    @property
    def nbv(self):
        return self.R_V // self.qgroup

    @property
    def off_kc(self):
        return 0

    @property
    def off_ks(self):
        return self.page_tokens * self.R_K // 2

    @property
    def off_kz(self):
        return self.off_ks + self.page_tokens * self.nbk * 2

    @property
    def off_vc(self):
        return self.off_kz + self.page_tokens * self.nbk * 2

    @property
    def off_vs(self):
        return self.off_vc + self.page_tokens * self.R_V // 2

    @property
    def off_vz(self):
        return self.off_vs + self.page_tokens * self.nbv * 2

    @property
    def payload_bytes(self):
        return self.off_vz + self.page_tokens * self.nbv * 2

    @property
    def bytes(self):
        return self.page_bytes or self.payload_bytes

    def __post_init__(self):
        assert self.page_tokens % self.BT == 0 and self.page_tokens % 64 == 0
        assert self.page_bytes is None or self.page_bytes >= self.payload_bytes

    # typed views of one page (uint8 tensor of >= payload bytes) -----------------------------------------------
    def views(self, page: torch.Tensor):
        P = self.page_tokens
        kc = page[self.off_kc:self.off_ks].view(P, self.R_K // 2)
        ks = page[self.off_ks:self.off_kz].view(torch.float16).view(P, self.nbk)
        kz = page[self.off_kz:self.off_vc].view(torch.float16).view(P, self.nbk)
        vc = page[self.off_vc:self.off_vs].view(torch.int16).view(P // self.BT, self.R_V // 4, self.BT)
        vs = page[self.off_vs:self.off_vz].view(torch.float16).view(P, self.nbv)
        vz = page[self.off_vz:self.payload_bytes].view(torch.float16).view(P, self.nbv)
        return kc, ks, kz, vc, vs, vz


def rotate_latent(cK: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, band, is_re, partner):
    """cK (T, R_K) fp32 unrotated; cos/sin (T, hd/2) per rotary frequency. Same as bellowskv.codec.BandRotation."""
    ct = cos.float().index_select(-1, band); st = sin.float().index_select(-1, band)
    other = cK.index_select(-1, partner)
    return torch.where(is_re, cK * ct - other * st, cK * ct + other * st)


# ----------------------------------------------------------------------------------------------- store
def latent_store_ref(acc, cos, sin, slot_mapping, pages, layout: PageLayout, rot, bits=4, clip=0.95):
    """Write the latents of T tokens into their slots.

    acc (T, R_K + R_V) fp32 = sum over the group's layers of h_l @ [WdK_l | WdV_l]; cos/sin (T, hd/2);
    slot_mapping (T,) int64 = block * page_tokens + row (-1 = skip); pages (num_blocks, page_bytes) uint8.
    rot = (band, is_re, partner) index tensors of the group.
    """
    keep = slot_mapping >= 0
    if not bool(keep.any()):
        return
    acc, cos, sin, slots = acc[keep], cos[keep], sin[keep], slot_mapping[keep]
    R_K, qg = layout.R_K, layout.qgroup
    cK = rotate_latent(acc[:, :R_K].float(), cos, sin, *rot)
    qk, sk, zk = quantize(cK, bits, qg, clip)
    qv, sv, zv = quantize(acc[:, R_K:].float(), bits, qg, clip)
    pk = pack_int4(qk); wv = pack_int4(qv).view(torch.int16)
    for i in range(slots.shape[0]):                       # reference: one token at a time
        blk, row = divmod(int(slots[i]), layout.page_tokens)
        kc, ks, kz, vc, vs, vz = layout.views(pages[blk])
        kc[row] = pk[i]; ks[row] = sk[i]; kz[row] = zk[i]
        vc[row // layout.BT, :, row % layout.BT] = wv[i]; vs[row] = sv[i]; vz[row] = zv[i]


def dequant_latent_ref(pages, layout: PageLayout, block_row, t0, t1):
    """Latent tokens [t0, t1) of one request -> (K_rot (n, R_K), V (n, R_V)) fp16, natural channel order."""
    Ks, Vs = [], []
    for t in range(t0, t1):
        blk, row = int(block_row[t // layout.page_tokens]), t % layout.page_tokens
        kc, ks, kz, vc, vs, vz = layout.views(pages[blk])
        Ks.append(dequantize(unpack_int4(kc[row]), ks[row], kz[row], layout.qgroup))
        w = vc[row // layout.BT, :, row % layout.BT].contiguous().view(torch.uint8)
        Vs.append(dequantize(unpack_int4(w), vs[row], vz[row], layout.qgroup))
    if not Ks:
        z = pages.new_zeros(0)
        return (torch.zeros(0, layout.R_K, device=pages.device, dtype=torch.float16),
                torch.zeros(0, layout.R_V, device=pages.device, dtype=torch.float16))
    return torch.stack(Ks), torch.stack(Vs)


# ----------------------------------------------------------------------------------------------- attention
def _attend(q, lat, Kx, Vx, xmask, WuK_p, WuV_p, idxK, idxV, scale):
    """One request. q (S, Hq, hd); lat = (K_rot (n, R_K), V (n, R_V)) or None; Kx/Vx (Hkv, X, hd) exact keys/values
    with xmask (S, X) bool (True = visible). Returns (S, Hq, hd) fp32. Latent in natural order; read-heads permuted."""
    S, Hq, hd = q.shape; Hkv = WuK_p.shape[0]; G = Hq // Hkv
    qg = (q.float() * scale).view(S, Hkv, G, hd)
    s_x = torch.einsum("skgd,kxd->skgx", qg, Kx.float()).masked_fill(~xmask[:, None, None, :], float("-inf"))
    m = s_x.amax(-1)
    if lat is not None and lat[0].shape[0]:
        CK = lat[0].float().index_select(-1, idxK); CV = lat[1].float()
        qt = torch.einsum("skgd,kdr->skgr", qg, WuK_p.float())
        s_l = torch.einsum("skgr,nr->skgn", qt, CK)
        m = torch.maximum(m, s_l.amax(-1))
    p_x = torch.exp(s_x - m[..., None]); L = p_x.sum(-1)
    o = torch.einsum("skgx,kxd->skgd", p_x, Vx.float())
    if lat is not None and lat[0].shape[0]:
        p_l = torch.exp(s_l - m[..., None]); L = L + p_l.sum(-1)
        o_l = torch.einsum("skgn,nr->skgr", p_l, CV).index_select(-1, idxV)
        o = o + torch.einsum("skgr,krd->skgd", o_l, WuV_p.float())
    return (o / L[..., None]).reshape(S, Hq, hd)


def decode_attention_ref(q, pages, layout, block_table, lat_len, sink, Kw, Vw, wmask, WuK_p, WuV_p, idxK, idxV, scale):
    """q (B, Hq, hd) fp16 (RoPE'd); latent tokens [sink, sink + lat_len[b]); Kw/Vw (B, Hkv, X, hd) this layer's exact
    tokens INCLUDING the current one, wmask (B, X) bool. Returns (B, Hq, hd) fp16."""
    out = []
    for b in range(q.shape[0]):
        n = int(lat_len[b])
        lat = dequant_latent_ref(pages, layout, block_table[b], sink, sink + n) if n else None
        out.append(_attend(q[b:b + 1], lat, Kw[b], Vw[b], wmask[b:b + 1], WuK_p, WuV_p, idxK, idxV, scale))
    return torch.cat(out).half()


def chunk_attention_ref(q, k, v, Kp, Vp, lat, WuK_p, WuV_p, idxK, idxV, scale):
    """One request's multi-token chunk. q (S, Hq, hd), k/v (S, Hkv, hd) the chunk's own (RoPE'd) keys/values;
    Kp/Vp (Hkv, P, hd) its exact past tokens (sinks + recent window, P may be 0); lat = (K_rot, V) of its latent range
    or None. Causal inside the chunk. Returns (S, Hq, hd) fp16."""
    S = q.shape[0]; P = Kp.shape[1]
    Kx = torch.cat([Kp, k.transpose(0, 1)], 1); Vx = torch.cat([Vp, v.transpose(0, 1)], 1)
    i = torch.arange(S, device=q.device)[:, None]; j = torch.arange(P + S, device=q.device)[None, :]
    xmask = j <= P + i
    if lat is None and P == 0:                                      # a fresh prompt: plain causal attention
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                           is_causal=True, enable_gqa=True, scale=scale)
        return o[0].transpose(0, 1).half()
    return _attend(q, lat, Kx, Vx, xmask, WuK_p, WuV_p, idxK, idxV, scale).half()


# ----------------------------------------------------------------------------------------------- dispatch
def latent_store(acc, cos, sin, slot_mapping, pages, layout, rot, bits=4, clip=0.95):
    """Graph-safe store (fixed-shape torch ops; slots of -1 land in the null block 0)."""
    from .paged_kernels import latent_store_triton
    return latent_store_triton(acc, cos, sin, slot_mapping, pages, layout, rot, bits, clip)


def decode_attention(q, pages, layout, block_table, lat_len, sink, Kw, Vw, wmask, WuK_p, WuV_p, idxK, idxV, scale,
                     nsplit=None, ws=None, lat_end=None, soff=None):
    """Graph-safe paged decode (Triton). Same contract as decode_attention_ref. For CUDA-graph capture pass a
    persistent score workspace `ws` (Hq, capacity >= sum_b round_up(sink + lat_len[b], 64)) and a fixed `nsplit`
    (paged_kernels.pick_nsplit(B_pad, R_V)); lat_end/soff may be precomputed with paged_kernels.score_offsets."""
    from .paged_kernels import decode_attention_paged
    return decode_attention_paged(q, pages, layout, block_table, lat_len, sink, Kw, Vw, wmask, WuK_p, WuV_p,
                                  idxK, idxV, scale, nsplit=nsplit, ws=ws, lat_end=lat_end, soff=soff)


def chunk_attention(q, k, v, Kp, Vp, pages, layout, block_row, t0, t1, WuK_p, WuV_p, idxK, idxV, scale):
    """One request's multi-token chunk (eager only: t0/t1 are host ints). Latent range [t0, t1) of that request
    (t1 <= t0: none). Other arguments as chunk_attention_ref. Query-blocked to bound the score memory."""
    from .paged_kernels import dequant_latent_paged
    lat = dequant_latent_paged(pages, layout, block_row, t0, t1) if t1 > t0 else None
    S = q.shape[0]; P = Kp.shape[1]
    if lat is None and P == 0:
        return chunk_attention_ref(q, k, v, Kp, Vp, None, WuK_p, WuV_p, idxK, idxV, scale)
    Kx = torch.cat([Kp, k.transpose(0, 1)], 1); Vx = torch.cat([Vp, v.transpose(0, 1)], 1)
    n = 0 if lat is None else lat[0].shape[0]
    blk = max(1, min(S, (1 << 27) // max(1, q.shape[1] * (n + P + S))))
    j = torch.arange(P + S, device=q.device)[None, :]
    out = torch.empty(S, q.shape[1], q.shape[2], device=q.device, dtype=torch.float16)
    for s0 in range(0, S, blk):
        s1 = min(S, s0 + blk)
        i = torch.arange(s0, s1, device=q.device)[:, None]
        out[s0:s1] = _attend(q[s0:s1], lat, Kx, Vx, j <= P + i, WuK_p, WuV_p, idxK, idxV, scale).half()
    return out
