"""Paged, CUDA-graph-safe BellowsKV-A kernels for the vLLM backend.

Same arithmetic as bellowskv/kernels.py (the paper's measured decode kernels: nibble-to-fp16 unpack by exponent
OR, mod-4 channel order, per-block affine applied after the dot on the scores side), with three changes:
  * addressing goes through the request's block table into struct-of-arrays latent pages (attn_ops.PageLayout);
  * every length is read inside the kernels (no host-side sizes): grids depend only on the batch size and a
    split count chosen from it, so a captured graph replays correctly for any context lengths;
  * scores live in a ragged workspace (Hq, sum_b round_up(end_b, 64)) indexed by a per-request offset.
The store (rotate, quantize, pack, scatter) is written with fixed-shape torch ops: graph-safe, slots of -1 are
redirected to block 0 (vLLM's null block, never handed to a request).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..codec import quantize, pack_int4
from .attn_ops import PageLayout, rotate_latent

BM_S = 64          # scores token tile (always inside one page: pages are multiples of 128 tokens)
BKV = 256          # values: latent channels per program


def _round_up(x, m):
    return (x + m - 1) // m * m


# ----------------------------------------------------------------------------------------------- store
def latent_store_triton(acc, cos, sin, slot_mapping, pages, layout: PageLayout, rot, bits=4, clip=0.95):
    """Graph-safe version of attn_ops.latent_store_ref (fixed shapes, no host sync). pages: (num_blocks, >=bytes)
    uint8 view with unit stride along dim 1."""
    T = acc.shape[0]
    if T == 0:
        return
    R_K, R_V, qg, P, BT = layout.R_K, layout.R_V, layout.qgroup, layout.page_tokens, layout.BT
    cK = rotate_latent(acc[:, :R_K].float(), cos, sin, *rot)
    qk, sk, zk = quantize(cK, bits, qg, clip)
    qv, sv, zv = quantize(acc[:, R_K:].float(), bits, qg, clip)
    pk = pack_int4(qk)                                            # (T, R_K/2) u8
    pv = pack_int4(qv).view(torch.int16).view(torch.uint8).view(T, R_V // 4, 2)   # words as byte pairs
    slots = torch.where(slot_mapping >= 0, slot_mapping, torch.zeros_like(slot_mapping))
    blk = (slots // P).long(); row = (slots % P).long()
    dev = acc.device
    b2 = blk[:, None]

    def put(off_bytes, vals):                                     # vals (T, n) u8 at byte offsets (T, n)
        pages.index_put_((b2.expand_as(off_bytes), off_bytes), vals)

    ak = torch.arange(R_K // 2, device=dev)
    put(layout.off_kc + row[:, None] * (R_K // 2) + ak[None, :], pk)
    an = torch.arange(layout.nbk * 2, device=dev)
    put(layout.off_ks + row[:, None] * (layout.nbk * 2) + an[None, :], sk.contiguous().view(torch.uint8).view(T, -1))
    put(layout.off_kz + row[:, None] * (layout.nbk * 2) + an[None, :], zk.contiguous().view(torch.uint8).view(T, -1))
    wk = torch.arange(R_V // 4, device=dev)
    base = layout.off_vc + ((row // BT)[:, None] * (R_V // 4) + wk[None, :]) * (BT * 2) + (row % BT)[:, None] * 2
    put(base, pv[:, :, 0]); put(base + 1, pv[:, :, 1])
    anv = torch.arange(layout.nbv * 2, device=dev)
    put(layout.off_vs + row[:, None] * (layout.nbv * 2) + anv[None, :], sv.contiguous().view(torch.uint8).view(T, -1))
    put(layout.off_vz + row[:, None] * (layout.nbv * 2) + anv[None, :], zv.contiguous().view(torch.uint8).view(T, -1))


def dequant_latent_paged(pages, layout: PageLayout, block_row, t0, t1):
    """Vectorized dequantization of latent tokens [t0, t1) of one request (eager paths only: t0/t1 are host ints).
    Returns (K_rot (n, R_K), V (n, R_V)) fp16, natural channel order."""
    from ..codec import unpack_int4, dequantize
    P, BT, qg = layout.page_tokens, layout.BT, layout.qgroup
    n = t1 - t0
    if n <= 0:
        dev = pages.device
        return (torch.zeros(0, layout.R_K, device=dev, dtype=torch.float16), torch.zeros(0, layout.R_V, device=dev, dtype=torch.float16))
    p0, p1 = t0 // P, (t1 - 1) // P + 1
    pg = pages[block_row[p0:p1].long()][:, :layout.payload_bytes]            # (np, payload)
    npg = pg.shape[0]
    kc = pg[:, layout.off_kc:layout.off_ks].reshape(npg * P, layout.R_K // 2)
    ks = pg[:, layout.off_ks:layout.off_kz].contiguous().view(torch.float16).reshape(npg * P, layout.nbk)
    kz = pg[:, layout.off_kz:layout.off_vc].contiguous().view(torch.float16).reshape(npg * P, layout.nbk)
    vc = pg[:, layout.off_vc:layout.off_vs].contiguous().view(torch.int16).reshape(npg, P // BT, layout.R_V // 4, BT)
    vc = vc.transpose(2, 3).reshape(npg * P, layout.R_V // 4).contiguous().view(torch.uint8)
    vs = pg[:, layout.off_vs:layout.off_vz].contiguous().view(torch.float16).reshape(npg * P, layout.nbv)
    vz = pg[:, layout.off_vz:layout.payload_bytes].contiguous().view(torch.float16).reshape(npg * P, layout.nbv)
    a, b = t0 - p0 * P, t1 - p0 * P
    K = dequantize(unpack_int4(kc[a:b]), ks[a:b], kz[a:b], qg)
    V = dequantize(unpack_int4(vc[a:b]), vs[a:b], vz[a:b], qg)
    return K, V


# ----------------------------------------------------------------------------------------------- kernels
@triton.jit
def _nib4(w):
    k1024 = tl.full([1, 1], 1024.0, tl.float16)
    n0 = ((w & 0x000F) | 0x6400).to(tl.float16, bitcast=True) - k1024
    n1 = (((w >> 4) & 0x000F) | 0x6400).to(tl.float16, bitcast=True) - k1024
    n2 = (((w >> 8) & 0x000F) | 0x6400).to(tl.float16, bitcast=True) - k1024
    n3 = (((w >> 12) & 0x000F) | 0x6400).to(tl.float16, bitcast=True) - k1024
    return n0, n1, n2, n3


@triton.jit
def _paged_scores_kernel(pw, ph, page_stride_w, page_stride_h, bt, bt_stride, lat_end, sink, qt, qsum,
                         ws, ws_stride, soff, NSPLIT,
                         RW: tl.constexpr, NB: tl.constexpr, GW: tl.constexpr, HQ: tl.constexpr, BM: tl.constexpr,
                         PT: tl.constexpr, OFF_KC_W: tl.constexpr, OFF_KS_H: tl.constexpr, OFF_KZ_H: tl.constexpr):
    """Program (b, j): latent tiles tau = j, j + NSPLIT, ... of request b. Scores of positions [sink, end) are
    written to ws[h, soff[b] + position]."""
    b = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    end = tl.load(lat_end + b)
    off = tl.load(soff + b).to(tl.int64)
    ntile = tl.cdiv(end, BM)
    hh = tl.arange(0, HQ)
    jj = tl.arange(0, GW)
    kk = tl.arange(0, GW)[:, None] * HQ
    R: tl.constexpr = 4 * RW
    qb = qt + b * (R * HQ) + hh[None, :]
    for tau in range(j, ntile, NSPLIT):
        t0 = tau * BM
        m = t0 + tl.arange(0, BM)
        valid = (m >= sink) & (m < end)
        page = tl.load(bt + b * bt_stride + t0 // PT).to(tl.int64)
        row = (t0 % PT) + tl.arange(0, BM)
        cb = pw + page * page_stride_w + OFF_KC_W + row[:, None] * RW + jj[None, :]
        sbp = ph + page * page_stride_h + OFF_KS_H + row * NB
        zbp = ph + page * page_stride_h + OFF_KZ_H + row * NB
        acc = tl.zeros((BM, HQ), dtype=tl.float32)
        for blk in range(0, NB):
            w = tl.load(cb + blk * GW, mask=valid[:, None], other=0)
            n0, n1, n2, n3 = _nib4(w)
            k0 = blk * (4 * GW)
            part = tl.dot(n0, tl.load(qb + (k0 + 0 * GW) * HQ + kk))
            part += tl.dot(n1, tl.load(qb + (k0 + 1 * GW) * HQ + kk))
            part += tl.dot(n2, tl.load(qb + (k0 + 2 * GW) * HQ + kk))
            part += tl.dot(n3, tl.load(qb + (k0 + 3 * GW) * HQ + kk))
            s_b = tl.load(sbp + blk, mask=valid, other=0.0).to(tl.float32)
            z_b = tl.load(zbp + blk, mask=valid, other=0.0).to(tl.float32)
            qs = tl.load(qsum + (b * NB + blk) * HQ + hh)
            acc += part * s_b[:, None] + z_b[:, None] * qs[None, :]
        tl.store(ws + hh[None, :].to(tl.int64) * ws_stride + off + m[:, None], acc, mask=valid[:, None])


@triton.jit
def _paged_values_kernel(pw, ph, page_stride_w, page_stride_h, bt, bt_stride, lat_end, sink, ws, ws_stride, soff,
                         acc_ptr, m_ptr, l_ptr, NSPLIT,
                         RW: tl.constexpr, NBV: tl.constexpr, GH: tl.constexpr, HQ: tl.constexpr, BM: tl.constexpr,
                         BKV_: tl.constexpr, BT: tl.constexpr, PT: tl.constexpr,
                         OFF_VC_W: tl.constexpr, OFF_VS_H: tl.constexpr, OFF_VZ_H: tl.constexpr):
    """Program (channel chunk c, split j, request b): online softmax over the split's contiguous range of BM-token
    tiles, values read from the page's token-contiguous V tile."""
    cb_ = tl.program_id(0)
    j = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    end = tl.load(lat_end + b)
    off = tl.load(soff + b).to(tl.int64)
    ntile = tl.cdiv(end, BM)
    per = tl.cdiv(ntile, NSPLIT)
    t_lo = j * per
    t_hi = tl.minimum(t_lo + per, ntile)
    hh = tl.arange(0, HQ)
    R: tl.constexpr = 4 * RW
    W: tl.constexpr = BKV_ // 4
    NBLK: tl.constexpr = BKV_ // (2 * GH)
    m_i = tl.full((HQ,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((HQ,), dtype=tl.float32)
    a0 = tl.zeros((W, HQ), dtype=tl.float32)
    a1 = tl.zeros((W, HQ), dtype=tl.float32)
    a2 = tl.zeros((W, HQ), dtype=tl.float32)
    a3 = tl.zeros((W, HQ), dtype=tl.float32)
    wi = cb_ * W + tl.arange(0, W)
    bb = cb_ * NBLK + tl.arange(0, NBLK)
    for tau in range(t_lo, t_hi):
        t0 = tau * BM
        m = t0 + tl.arange(0, BM)
        valid = (m >= sink) & (m < end)
        sv = tl.load(ws + hh[:, None].to(tl.int64) * ws_stride + off + m[None, :], mask=valid[None, :], other=float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sv, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(sv - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        p16 = p.to(tl.float16)
        m_i = m_new
        page = tl.load(bt + b * bt_stride + t0 // PT).to(tl.int64)
        tin = t0 % PT
        t_in = (tin % BT) + tl.arange(0, BM)
        rows = tin + tl.arange(0, BM)
        w = tl.load(pw + page * page_stride_w + OFF_VC_W + ((tin // BT) * RW + wi[:, None]) * BT + t_in[None, :],
                    mask=valid[None, :], other=0)                                          # (W, BM) int16
        s2 = tl.load(ph + page * page_stride_h + OFF_VS_H + rows[None, :] * NBV + bb[:, None], mask=valid[None, :], other=0.0)
        z2 = tl.load(ph + page * page_stride_h + OFF_VZ_H + rows[None, :] * NBV + bb[:, None], mask=valid[None, :], other=0.0)
        if NBLK == 1:
            sb = s2
            zb = z2
        else:
            sb = tl.reshape(tl.broadcast_to(s2[:, None, :], (NBLK, W // NBLK, BM)), (W, BM))
            zb = tl.reshape(tl.broadcast_to(z2[:, None, :], (NBLK, W // NBLK, BM)), (W, BM))
        n0, n1, n2, n3 = _nib4(w)
        n0 = n0 * sb + zb
        n1 = n1 * sb + zb
        n2 = n2 * sb + zb
        n3 = n3 * sb + zb
        pT = tl.trans(p16)
        a0 = a0 * alpha[None, :] + tl.dot(n0, pT)
        a1 = a1 * alpha[None, :] + tl.dot(n1, pT)
        a2 = a2 * alpha[None, :] + tl.dot(n2, pT)
        a3 = a3 * alpha[None, :] + tl.dot(n3, pT)
    base = (b * NSPLIT + j) * HQ + hh
    col = cb_ * BKV_ + tl.arange(0, W)
    tl.store(acc_ptr + base[None, :] * R + (col + 0 * W)[:, None], a0)          # mod-4 permuted columns
    tl.store(acc_ptr + base[None, :] * R + (col + 1 * W)[:, None], a1)
    tl.store(acc_ptr + base[None, :] * R + (col + 2 * W)[:, None], a2)
    tl.store(acc_ptr + base[None, :] * R + (col + 3 * W)[:, None], a3)
    if cb_ == 0:
        tl.store(m_ptr + base, m_i)
        tl.store(l_ptr + base, l_i)


# ----------------------------------------------------------------------------------------------- decode
def pick_nsplit(B: int, R_V: int, programs: int = 1024, max_split: int = 64) -> int:
    """Splits per request from the (padded) batch size only, so a captured graph serves every context length."""
    return max(1, min(max_split, programs // max(1, B * (R_V // BKV))))


def score_offsets(lat_len: torch.Tensor, sink: int):
    """(lat_end (B,) int32, soff (B+1,) int32): latent positions [sink, lat_end) per request (lat_end = 0 when the
    request has no latent tokens), offsets of each request's rows in the score workspace. Fixed-shape torch ops."""
    lat_end = torch.where(lat_len > 0, lat_len + sink, torch.zeros_like(lat_len)).to(torch.int32)
    padded = (lat_end + BM_S - 1) // BM_S * BM_S
    soff = torch.zeros(lat_len.shape[0] + 1, dtype=torch.int32, device=lat_len.device)
    soff[1:] = torch.cumsum(padded, 0)
    return lat_end, soff


def decode_attention_paged(q, pages, layout: PageLayout, block_table, lat_len, sink, Kw, Vw, wmask, WuK_p, WuV_p,
                           idxK=None, idxV=None, scale=1.0, nsplit=None, ws=None, lat_end=None, soff=None):
    """Graph-safe decode attention. Same contract as attn_ops.decode_attention_ref, plus:
      ws     (Hq, capacity) fp32 score workspace, capacity >= sum_b round_up(sink + lat_len[b], 64); allocated here
             (with a host sync) when None, which is fine eagerly but not under graph capture;
      lat_end/soff from score_offsets(lat_len, sink) (computed here when None; fixed-shape ops, graph-safe);
      nsplit from pick_nsplit(B, R_V) when None.
    idxK/idxV are unused (the kernels consume the permuted read-heads directly)."""
    B, Hq, hd = q.shape
    Hkv = WuK_p.shape[0]; G = Hq // Hkv
    R_K, R_V, qg = layout.R_K, layout.R_V, layout.qgroup
    dev = q.device
    if lat_end is None or soff is None:
        lat_end, soff = score_offsets(lat_len, sink)
    if ws is None:
        ws = torch.empty(Hq, max(int(soff[-1]), BM_S), device=dev, dtype=torch.float32)
    J = nsplit or pick_nsplit(B, R_V)
    qg_ = (q * scale).view(B, Hkv, G, hd)
    qt = torch.einsum("bkgd,kdr->brkg", qg_, WuK_p).reshape(B, R_K, Hq).contiguous()
    qsum = qt.view(B, R_K // qg, qg, Hq).float().sum(2).contiguous()
    pw = pages.view(torch.int16); ph = pages.view(torch.float16)
    assert pages.stride(1) == 1 and pages.stride(0) % 2 == 0
    _paged_scores_kernel[(B, J)](pw, ph, pw.stride(0), ph.stride(0), block_table, block_table.stride(0), lat_end, sink,
                                 qt, qsum, ws, ws.stride(0), soff, J,
                                 RW=R_K // 4, NB=R_K // qg, GW=qg // 4, HQ=Hq, BM=BM_S, PT=layout.page_tokens,
                                 OFF_KC_W=layout.off_kc // 2, OFF_KS_H=layout.off_ks // 2, OFF_KZ_H=layout.off_kz // 2,
                                 num_warps=4, num_stages=3)
    acc = torch.empty(B, J, Hq, R_V, device=dev, dtype=torch.float32)
    mm = torch.empty(B, J, Hq, device=dev, dtype=torch.float32)
    ll = torch.empty(B, J, Hq, device=dev, dtype=torch.float32)
    _paged_values_kernel[(R_V // BKV, J, B)](pw, ph, pw.stride(0), ph.stride(0), block_table, block_table.stride(0),
                                             lat_end, sink, ws, ws.stride(0), soff, acc, mm, ll, J,
                                             RW=R_V // 4, NBV=R_V // qg, GH=qg // 2, HQ=Hq, BM=layout.BT, BKV_=BKV,
                                             BT=layout.BT, PT=layout.page_tokens, OFF_VC_W=layout.off_vc // 2,
                                             OFF_VS_H=layout.off_vs // 2, OFF_VZ_H=layout.off_vz // 2,
                                             num_warps=4, num_stages=3)
    # merge the latent splits with the exact window (one softmax), normalize before the fp16 cast
    Kw = Kw.masked_fill(~wmask[:, None, :, None], 0); Vw = Vw.masked_fill(~wmask[:, None, :, None], 0)
    s_w = torch.einsum("bkgd,bktd->bkgt", qg_, Kw).float().reshape(B, Hq, -1)
    s_w = s_w.masked_fill(~wmask[:, None, :], float("-inf"))
    m_g = torch.maximum(mm.amax(1), s_w.amax(-1))
    p_w = torch.exp(s_w - m_g[..., None])
    w = torch.exp(mm - m_g[:, None, :])
    L = (w * ll).sum(1) + p_w.sum(-1)
    o_lat = (w[..., None] * acc).sum(1) / L[..., None]
    o = torch.einsum("bkgr,krd->bkgd", o_lat.view(B, Hkv, G, R_V).half(), WuV_p).reshape(B, Hq, hd).float()
    o_w = torch.einsum("bkgt,bktd->bkgd", (p_w / L[..., None]).half().view(B, Hkv, G, -1), Vw)
    return (o + o_w.reshape(B, Hq, hd).float()).half()
