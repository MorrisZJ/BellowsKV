"""INT4-resident absorbed decode kernels for BellowsKV-A.

Taken from the research code's measured kernels with the timing-only helpers and the per-kv-head variant
removed; the kernels themselves are unchanged.

At decode nothing position-dependent happens: the K latent is stored *rotated* (the RoPE-equivariant read-head
lets the latent's frequency pairs be rotated exactly at write time), the query is absorbed through the read-head
(one (Hq, hd) x (hd, R_K) product per layer per step), and the score is a dot product in the latent. Two kernels
per layer:

  scores : s[b,h,t] = sum_r dq(c_K[b,t,r]) * qt[b,r,h]       reads the packed K latent once
  values : o[b,h,:] = sum_t softmax_t(s)[b,h,t] * dq(c_V[b,t,:])
                                                              reads the packed V latent once (online softmax
                                                              over token splits, partials merged on the host)

Layouts: K codes (B, S, R_K/2) uint8, byte j = channels 2j | 2j+1 << 4, one fp16 (scale, zero) per 128 channels;
V codes in token-contiguous tiles (B, S/BT, R_V/4, BT) int16 (see `to_v_tiles`). Read-head rows are permuted to
the kernels' mod-4 channel order once at load (`prepare_heads`).
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:                                    # pragma: no cover
    HAS_TRITON = False

__all__ = ["eq_scores_int4", "eq_values_int4", "eq_absorbed_decode", "permute_mod4", "permute_index", "to_v_tiles",
           "prepare_heads", "pick_nsplit", "DEFAULT_CFG", "HAS_TRITON"]


# BM_s/BK_s: scores token tile / channel chunk; BM_v/BKV: values token tile / channels per program;
# BT: V storage tile (tokens contiguous). BM_v must divide BT.
DEFAULT_CFG = dict(BM_s=64, warps_s=4, BM_v=128, BKV=256, warps_v=4, BT=128, stages=3, programs=1024)


def permute_mod4(W: torch.Tensor, dim: int, block: int) -> torch.Tensor:
    """Reorder channels along `dim` so each `block`-wide chunk lists channels = 0, 1, 2, 3 (mod 4) in turn."""
    R = W.shape[dim]
    idx = torch.cat([torch.arange(k0 + r, k0 + block, 4) for k0 in range(0, R, block) for r in range(4)])
    return W.index_select(dim, idx.to(W.device)).contiguous()


def prepare_heads(WuK: torch.Tensor, WuV: torch.Tensor, cfg=None, qgroup: int = 128):
    """WuK (Hkv, hd, R_K), WuV (Hkv, R_V, hd) -> the permuted copies the kernels consume: mod-4 order inside
    every quant block on the K side (the scores kernel iterates blocks), inside every BKV chunk on the V side."""
    cfg = cfg or DEFAULT_CFG
    return permute_mod4(WuK, 2, qgroup), permute_mod4(WuV, 1, cfg["BKV"])


def to_v_tiles(codes: torch.Tensor, BT: int) -> torch.Tensor:
    """(B, S, R/2) uint8 packed codes -> (B, ceil(S/BT), R/4, BT) int16, tokens contiguous, zero-padded."""
    B, S, RH = codes.shape
    nt = -(-S // BT)
    w = codes.view(torch.int16)                                          # (B, S, R/4) words: channels 4k..4k+3
    if nt * BT != S:
        w = torch.cat([w, w.new_zeros(B, nt * BT - S, RH // 2)], 1)
    return w.view(B, nt, BT, RH // 2).transpose(2, 3).contiguous()


def permute_index(R: int, block: int) -> torch.Tensor:
    """The channel order `permute_mod4` applies along a dimension of size R."""
    return torch.cat([torch.arange(k0 + r, k0 + block, 4) for k0 in range(0, R, block) for r in range(4)])


def pick_nsplit(S: int, B: int, R_V: int, BKV: int, BM: int, programs: int = 1024, qgroup: int = 128) -> int:
    """Token splits for the values pass: enough programs to fill the GPU, capped so the fp32 partials
    (Hq x R_V per split per sequence) stay under ~25% of the V-latent bytes."""
    nt = R_V // BKV
    by_fill = -(-programs // (nt * B))
    by_traffic = max(1, int(S * (R_V // 2 + 4 * R_V // qgroup) / (4 * 32 * R_V * 4)))
    return max(1, min(by_fill, by_traffic, -(-S // BM)))


if HAS_TRITON:

    @triton.jit
    def _nib4(w):
        """int16 words -> four fp16 tiles holding the nibble values (channels 4k, 4k+1, 4k+2, 4k+3), exact."""
        k1024 = tl.full([1, 1], 1024.0, tl.float16)
        n0 = ((w & 0x000F) | 0x6400).to(tl.float16, bitcast=True) - k1024
        n1 = (((w >> 4) & 0x000F) | 0x6400).to(tl.float16, bitcast=True) - k1024
        n2 = (((w >> 8) & 0x000F) | 0x6400).to(tl.float16, bitcast=True) - k1024
        n3 = (((w >> 12) & 0x000F) | 0x6400).to(tl.float16, bitcast=True) - k1024
        return n0, n1, n2, n3

    @triton.jit
    def _eq_scores_kernel(codes, sc, zr, qt, qsum, out, smax, S, SS, NTILE,
                          RW: tl.constexpr, NB: tl.constexpr, GW: tl.constexpr, HQ: tl.constexpr,
                          BM: tl.constexpr, MODE: tl.constexpr, PF: tl.constexpr):
        """One program = one (sequence, BM-token tile); all HQ heads; loops over the R_K quant blocks.
        GW = words per block (qgroup/4). MODE 0 = real; 1 = no latent read (dots on zeros); 4 = loads only."""
        b = tl.program_id(0).to(tl.int64)
        mb = tl.program_id(1)
        m = mb * BM + tl.arange(0, BM)
        valid = m < S
        row = b * SS + m
        hh = tl.arange(0, HQ)
        R: tl.constexpr = 4 * RW
        acc = tl.zeros((BM, HQ), dtype=tl.float32)
        jj = tl.arange(0, GW)
        kk = tl.arange(0, GW)[:, None] * HQ
        qb = qt + b * (R * HQ) + hh[None, :]
        cb = codes + row[:, None] * RW + jj[None, :]
        if PF >= 1:
            w = tl.load(cb, mask=valid[:, None], other=0)
        if PF == 2:
            w1 = tl.load(cb + GW, mask=valid[:, None], other=0)
        for blk in range(0, NB):
            if MODE == 4:
                w = tl.load(cb + blk * GW, mask=valid[:, None], other=0)
                acc += tl.sum(w.to(tl.float32), 1)[:, None]
            else:
                if MODE == 1:
                    w = tl.zeros((BM, GW), dtype=tl.int16)
                if PF == 0 and MODE != 1:
                    w = tl.load(cb + blk * GW, mask=valid[:, None], other=0)
                if PF == 1:
                    wn = tl.load(cb + (blk + 1) * GW, mask=valid[:, None] & (blk + 1 < NB), other=0)
                if PF == 2:
                    wn = tl.load(cb + (blk + 2) * GW, mask=valid[:, None] & (blk + 2 < NB), other=0)
                n0, n1, n2, n3 = _nib4(w)                                                       # (BM, GW) each
                k0 = blk * (4 * GW)
                part = tl.dot(n0, tl.load(qb + (k0 + 0 * GW) * HQ + kk))
                part += tl.dot(n1, tl.load(qb + (k0 + 1 * GW) * HQ + kk))
                part += tl.dot(n2, tl.load(qb + (k0 + 2 * GW) * HQ + kk))
                part += tl.dot(n3, tl.load(qb + (k0 + 3 * GW) * HQ + kk))                      # (BM, HQ) fp32
                s_b = tl.load(sc + row * NB + blk, mask=valid, other=0.0).to(tl.float32)
                z_b = tl.load(zr + row * NB + blk, mask=valid, other=0.0).to(tl.float32)
                qs = tl.load(qsum + (b * NB + blk) * HQ + hh)                                   # (HQ,) fp32
                acc += part * s_b[:, None] + z_b[:, None] * qs[None, :]
                if PF == 1:
                    w = wn
                if PF == 2:
                    w = w1
                    w1 = wn
        o = out + (b * HQ + hh)[None, :] * SS + m[:, None]
        tl.store(o, acc, mask=valid[:, None])
        tmax = tl.max(tl.where(valid[:, None], acc, float("-inf")), 0)                          # (HQ,)
        tl.store(smax + (b * HQ + hh) * NTILE + mb, tmax)


    @triton.jit
    def _eq_values_kernel(codes, sc, zr, s_ptr, acc_ptr, m_ptr, l_ptr, S, SS, CH, NT, NSPLIT,
                          RW: tl.constexpr, NB: tl.constexpr, GH: tl.constexpr, HQ: tl.constexpr,
                          BM: tl.constexpr, BKV: tl.constexpr, BT: tl.constexpr,
                          MODE: tl.constexpr):
        """One program = one (BKV-channel chunk, token split, sequence): online softmax over its tokens,
        V read from token-contiguous tiles. MODE 0 = real; 1 = no latent read/dequant; 2 = no softmax."""
        cb_ = tl.program_id(0)
        j = tl.program_id(1)
        b = tl.program_id(2).to(tl.int64)
        hh = tl.arange(0, HQ)
        R: tl.constexpr = 4 * RW
        W: tl.constexpr = BKV // 4
        NBLK: tl.constexpr = BKV // (2 * GH)
        m_i = tl.full((HQ,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((HQ,), dtype=tl.float32)
        a0 = tl.zeros((W, HQ), dtype=tl.float32)
        a1 = tl.zeros((W, HQ), dtype=tl.float32)
        a2 = tl.zeros((W, HQ), dtype=tl.float32)
        a3 = tl.zeros((W, HQ), dtype=tl.float32)
        wi = cb_ * W + tl.arange(0, W)
        bb = cb_ * NBLK + tl.arange(0, NBLK)
        t_lo = j * CH
        t_hi = tl.minimum(t_lo + CH, S)
        for t0 in range(t_lo, t_hi, BM):
            t0 = tl.multiple_of(t0, BM)
            m = t0 + tl.arange(0, BM)
            valid = m < t_hi
            sv = tl.load(s_ptr + (b * HQ + hh)[:, None] * SS + m[None, :],
                         mask=valid[None, :], other=float("-inf"))                 # (HQ, BM) fp32
            if MODE == 2:
                p16 = tl.where(valid[None, :], sv, 0.0).to(tl.float16)
                alpha = tl.full((HQ,), 1.0, dtype=tl.float32)
            else:
                m_new = tl.maximum(m_i, tl.max(sv, 1))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(sv - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                p16 = p.to(tl.float16)
                m_i = m_new
            if MODE == 1 or MODE == 4:
                n0 = tl.zeros((W, BM), dtype=tl.float16)
                n1 = n0
                n2 = n0
                n3 = n0
            else:
                tile = t0 // BT
                t_in = tl.multiple_of(t0 - tile * BT, BM) + tl.arange(0, BM)
                w = tl.load(codes + ((b * NT + tile) * RW + wi[:, None]) * BT + t_in[None, :],
                            mask=valid[None, :], other=0)                          # (W, BM) int16
                s2 = tl.load(sc + (b * SS + m)[None, :] * NB + bb[:, None], mask=valid[None, :], other=0.0)   # (NBLK, BM)
                z2 = tl.load(zr + (b * SS + m)[None, :] * NB + bb[:, None], mask=valid[None, :], other=0.0)
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
            if MODE >= 3:
                if MODE == 4:
                    tile = t0 // BT
                    t_in = tl.multiple_of(t0 - tile * BT, BM) + tl.arange(0, BM)
                    ww = tl.load(codes + ((b * NT + tile) * RW + wi[:, None]) * BT + t_in[None, :], mask=valid[None, :], other=0)
                    a0 += tl.sum(ww.to(tl.float32), 1)[:, None]
                else:
                    a0 += tl.sum((n0 + n1 + n2 + n3).to(tl.float32), 1)[:, None]
            else:
                pT = tl.trans(p16)                                                    # (BM, HQ)
                a0 = a0 * alpha[None, :] + tl.dot(n0, pT)
                a1 = a1 * alpha[None, :] + tl.dot(n1, pT)
                a2 = a2 * alpha[None, :] + tl.dot(n2, pT)
                a3 = a3 * alpha[None, :] + tl.dot(n3, pT)
        base = (b * NSPLIT + j) * HQ + hh                                         # (HQ,)
        col = cb_ * BKV + tl.arange(0, W)
        tl.store(acc_ptr + base[None, :] * R + (col + 0 * W)[:, None], a0)        # mod-4 permuted columns
        tl.store(acc_ptr + base[None, :] * R + (col + 1 * W)[:, None], a1)
        tl.store(acc_ptr + base[None, :] * R + (col + 2 * W)[:, None], a2)
        tl.store(acc_ptr + base[None, :] * R + (col + 3 * W)[:, None], a3)
        if cb_ == 0:
            tl.store(m_ptr + base, m_i)
            tl.store(l_ptr + base, l_i)




def eq_scores_int4(codes, scale, zero, qt_p, qsum, out, smax, qgroup=128, BM=64, num_warps=4, num_stages=3, mode=0, S=None, pf=1):
    """codes (B,S,R/2) u8 (fold.codec packing), scale/zero (B,S,R/qgroup) f16, qt_p (B,R,HQ) f16 with rows
    mod-4-permuted per block (prepare_heads), qsum (B,R/qgroup,HQ) f32 block column sums of qt, out (B,HQ,S) f32,
    smax (B,HQ,ntiles) f32 per-tile row maxima."""
    B, SS, RH = codes.shape                                     # SS = buffer stride (rows); S = valid tokens
    S = S or SS
    HQ = qt_p.shape[-1]
    assert qt_p.is_contiguous() and out.is_contiguous() and codes.is_contiguous() and qsum.is_contiguous()
    assert out.shape[-1] == SS and S <= SS and SS % 16 == 0, "pad the latent buffers to a multiple of 16 rows (Triton specializes on divisibility)"
    ntile = triton.cdiv(S, BM)
    assert smax.shape == (B, HQ, ntile)
    w = codes.view(torch.int16)
    _eq_scores_kernel[(B, ntile)](w, scale, zero, qt_p, qsum, out, smax, S, SS, ntile, RW=RH // 2, NB=2 * RH // qgroup,
                                  GW=qgroup // 4, HQ=HQ, BM=BM, MODE=mode, PF=pf, num_warps=num_warps, num_stages=num_stages)
    return out



def eq_values_int4(codes_t, scale, zero, s, nsplit, qgroup=128, BM=64, BKV=256, num_warps=4, num_stages=3, mode=0, S=None):
    """codes_t (B, NT, R/4, BT) int16 token-contiguous tiles (to_v_tiles), scale/zero (B,S,R/qgroup) f16,
    s (B,HQ,S) f32. Returns (acc (B,J,HQ,R) f32, mod-4-permuted columns per BKV chunk; m (B,J,HQ); l (B,J,HQ))."""
    B, NT, RW, BT = codes_t.shape
    R = 4 * RW
    HQ, SS = s.shape[1], s.shape[2]
    S = S or SS
    assert R % BKV == 0 and BT % BM == 0 and NT * BT >= S and S <= SS and SS % 16 == 0
    CH = -(-S // max(1, nsplit))
    CH = -(-CH // BM) * BM
    J = -(-S // CH)
    acc = torch.empty(B, J, HQ, R, device=s.device, dtype=torch.float32)
    mm = torch.empty(B, J, HQ, device=s.device, dtype=torch.float32)
    ll = torch.empty(B, J, HQ, device=s.device, dtype=torch.float32)
    grid = (R // BKV, J, B)
    _eq_values_kernel[grid](codes_t, scale, zero, s, acc, mm, ll, S, SS, CH, NT, J, RW=RW, NB=R // qgroup,
                            GH=qgroup // 2, HQ=HQ, BM=BM, BKV=BKV, BT=BT, MODE=mode,
                            num_warps=num_warps, num_stages=num_stages)
    return acc, mm, ll


def eq_absorbed_decode(q_rot, latK, latV, WuK_p, WuV_p, scale, window=None, qgroup=128, cfg=None,
                       nsplit=None, stats=None, S=None):
    """One layer's decode step for a batch of sequences, all in the latent.

    q_rot   (B, Hq, hd) f16      the layer's RoPE'd queries
    latK    (codes, scale, zero) packed rotated K latent, R_K channels
    latV    (codes_t, scale, zero) packed V latent in token-contiguous tiles (to_v_tiles), R_V channels
    WuK_p   (Hkv, hd, R_K) f16   read-head, transposed, mod-4-permuted per block (prepare_heads)
    WuV_p   (Hkv, R_V, hd) f16   value read-head, mod-4-permuted per block (prepare_heads)
    window  optional (K_w (B,Hkv,Tw,hd), V_w (B,Hkv,Tw,hd)) fp16 tokens kept exact (sink, recent, fresh)
    S       valid latent tokens (the buffers may be longer; their row stride must be a multiple of 16)
    Returns o (B, Hq, hd) f16 (before o_proj).

    Change from the paper's benchmark copy: the latent-side partial sum is divided by the softmax denominator
    before the fp16 cast (unnormalized sums can pass fp16's range on long, diffuse-attention contexts).
    """
    cfg = cfg or DEFAULT_CFG
    B, Hq, hd = q_rot.shape
    Hkv = WuK_p.shape[0]
    G = Hq // Hkv
    cK, sK, zK = latK
    cV, sV, zV = latV
    SS = cK.shape[1]; S = S or SS                                   # SS = padded stride, S = valid tokens
    R_K = 2 * cK.shape[2]
    R_V = 4 * cV.shape[2]
    dev = q_rot.device
    qg = (q_rot * scale).view(B, Hkv, G, hd)
    qt = torch.einsum("bkgd,kdr->brkg", qg, WuK_p).reshape(B, R_K, Hq).contiguous()   # einsum's layout is shape-dependent
    qsum = qt.view(B, R_K // qgroup, qgroup, Hq).float().sum(2).contiguous()          # (B, NB_K, Hq)
    ntile = -(-S // cfg["BM_s"])
    s = torch.empty(B, Hq, SS, device=dev, dtype=torch.float32)
    smax = torch.empty(B, Hq, ntile, device=dev, dtype=torch.float32)
    eq_scores_int4(cK, sK, zK, qt, qsum, s, smax, qgroup, BM=cfg["BM_s"], num_warps=cfg["warps_s"],
                   num_stages=cfg["stages"], S=S, pf=cfg.get("pf", 1))
    J = nsplit or pick_nsplit(S, B, R_V, cfg["BKV"], cfg["BM_v"], cfg.get("programs", 1024), qgroup)
    acc, mm, ll = eq_values_int4(cV, sV, zV, s, J, qgroup, BM=cfg["BM_v"], BKV=cfg["BKV"], num_warps=cfg["warps_v"],
                                 num_stages=cfg["stages"], S=S)
    m_g = mm.amax(1)                                                                # (B, Hq)
    if window is not None:
        K_w, V_w = window
        Tw = K_w.shape[2]
        s_w = torch.einsum("bkgd,bktd->bkgt", qg, K_w).float().reshape(B, Hq, Tw)
        m_g = torch.maximum(m_g, s_w.amax(-1))
        p_w = torch.exp(s_w - m_g[..., None])
        l_w = p_w.sum(-1)
    w = torch.exp(mm - m_g[:, None, :])                                             # (B, J, Hq)
    L = (w * ll).sum(1)
    if window is not None:
        L = L + l_w
    o_lat = (w[..., None] * acc).sum(1) / L[..., None]                              # (B, Hq, R_V) permuted cols
    o = torch.einsum("bkgr,krd->bkgd", o_lat.view(B, Hkv, G, R_V).half(), WuV_p).reshape(B, Hq, hd).float()
    if window is not None:
        o_w = torch.einsum("bkgt,bktd->bkgd", (p_w / L[..., None]).half().view(B, Hkv, G, Tw), V_w)
        o = o + o_w.reshape(B, Hq, hd).float()
    if stats is not None:
        stats["J"] = J
    return o.half()
