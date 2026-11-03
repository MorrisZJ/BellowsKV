"""Fitting the BellowsKV-A codec of one layer group from the calibration statistics.

Values: the paper's joint G-weighted reduced-rank regression, square-root (balanced) split, R_V = kv:
    W_down = G^{-1/2} U_R Sigma_R^{1/2},   W_up = Sigma_R^{1/2} V_R^T,   SVD of G^{1/2} blockdiag(W_V^(l)).
Keys: the RoPE-equivariant variant. Each rotary pair (dims i, i+hd/2 of every head) is one complex number; the
key map is fit band by band as a complex reduced-rank regression on the same joint Gram, so the read-head maps
latent band i only to key band i and commutes with the rotation. Ranks per band are allocated greedily on the
bands' singular values weighted by the calibration query energy of the band ("qw" in the paper's code).
Same math as the research code behind the paper's numbers.

Latent layout (keys): per band i, [Re(r_i) | Im(r_i)] columns, bands in frequency order. Wu columns follow the
model's K layout: layer position p, kv head h, dim j -> p*kv + h*hd + j.
"""
from __future__ import annotations

import json

import torch

from .calibration import joint_gram


def _sqrt_and_inv_sqrt(M, ridge):
    """Symmetric square root and inverse square root of M + ridge * mean(diag) * I."""
    M = .5 * (M + M.T); rr = M.diagonal().mean().clamp_min(1e-8)
    w, Q = torch.linalg.eigh(M + torch.eye(M.shape[0], device=M.device, dtype=M.dtype) * ridge * rr)
    ws = w.clamp_min(1e-8).sqrt()
    return (Q * ws) @ Q.T, (Q * (1 / ws)) @ Q.T


def _greedy_ranks(spectra, budget, min_r=1):
    """Complex rank per band from the largest (weighted) singular values overall; total = budget."""
    ranks = [min_r] * len(spectra); cands = []
    for i, s2 in enumerate(spectra):
        for k in range(min_r, len(s2)):
            cands.append((float(s2[k]), i, k))
    cands.sort(reverse=True)
    left = budget - min_r * len(spectra)
    for _, i, k in cands:
        if left == 0:
            break
        if k == ranks[i]:                       # only contiguous prefixes are valid
            ranks[i] += 1; left -= 1
    if left:
        raise ValueError(f"r_k too large for this model: {left} complex ranks could not be placed")
    return ranks


@torch.no_grad()
def fit_group(model, stats, grp, r_k, r_v, ridge=1e-3, log=print):
    """Returns {"WdK": (n*d, r_k), "WuK": (r_k, n*kv), "WdV": (n*d, r_v), "WuV": (r_v, n*kv), "ranks": (hd/2,)}."""
    cfg = model.config; layers = model.model.layers
    n = len(grp); d = cfg.hidden_size; Hq = cfg.num_attention_heads; Hkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or d // Hq; kv = Hkv * hd; half = hd // 2
    GJ = joint_gram(stats, grp); dev = GJ.device
    Gs, Gi = _sqrt_and_inv_sqrt(GJ, ridge); del GJ
    WK = [layers[l].self_attn.k_proj.weight.t().float().to(dev) for l in grp]          # (d, kv)
    WV = [layers[l].self_attn.v_proj.weight.t().float().to(dev) for l in grp]

    # ---- values: balanced RRR, R_V = r_v
    U, S, Vh = torch.linalg.svd(Gs @ torch.block_diag(*WV), full_matrices=False)
    sr = S[:r_v].clamp_min(1e-8).sqrt()
    WdV = (Gi @ U[:, :r_v]) * sr[None, :]; WuV = sr[:, None] * Vh[:r_v]
    del U, S, Vh

    # ---- keys: one complex RRR per rotary band
    Wc = torch.zeros(n * d, n * Hkv, half, dtype=torch.complex64, device=dev)
    for pos, W in enumerate(WK):
        W = W.view(d, Hkv, hd)
        Wc[pos * d:(pos + 1) * d, pos * Hkv:(pos + 1) * Hkv, :] = torch.complex(W[:, :, :half], W[:, :, half:])
    Gsc = Gs.to(torch.complex64); del Gs
    svds = []
    for i in range(half):
        svds.append(torch.linalg.svd(Gsc @ Wc[:, :, i], full_matrices=False))
    del Gsc, Wc
    spectra = [(s ** 2).cpu() for _, s, _ in svds]
    # band weight = calibration query energy in the band over the group's layers and all query heads
    w = torch.zeros(half, dtype=torch.float64)
    for l in grp:
        e = stats["QE"][l].cpu().view(Hq, hd)
        w += (e[:, :half] + e[:, half:]).sum(0)
    w = (w / w.mean()).tolist()
    ranks = _greedy_ranks([s * w[i] for i, s in enumerate(spectra)], r_k // 2)

    Gic = Gi.to(torch.complex64); del Gi
    WdK = torch.zeros(n * d, r_k, device=dev, dtype=torch.float32)
    WuK = torch.zeros(r_k, n * kv, device=dev, dtype=torch.float32)
    col = 0
    for i, (U, S, Vh) in enumerate(svds):
        r = ranks[i]
        if r == 0:
            continue
        sr = S[:r].sqrt()
        Dc = (Gic @ U[:, :r]) * sr[None, :]                     # (n*d, r) complex
        Uc = sr[:, None] * Vh[:r]                                # (r, n*Hkv) complex
        WdK[:, col:col + r] = Dc.real; WdK[:, col + r:col + 2 * r] = Dc.imag
        for pos in range(n):
            for h in range(Hkv):
                m = pos * Hkv + h; cr = pos * kv + h * hd + i; ci = cr + half
                WuK[col:col + r, cr] = Uc[:, m].real;         WuK[col:col + r, ci] = Uc[:, m].imag
                WuK[col + r:col + 2 * r, cr] = -Uc[:, m].imag; WuK[col + r:col + 2 * r, ci] = Uc[:, m].real
        col += 2 * r
    assert col == r_k
    dt = torch.float16
    return {"WdK": WdK.to(dt).cpu(), "WuK": WuK.to(dt).cpu(), "WdV": WdV.to(dt).cpu(), "WuV": WuV.to(dt).cpu(),
            "ranks": torch.tensor(ranks, dtype=torch.int32)}


def save_factors(path, factors, meta):
    from safetensors.torch import save_file
    flat = {f"g{g0}.{k}": v.contiguous() for g0, f in factors.items() for k, v in f.items()}
    save_file(flat, path, metadata={"bellowskv": json.dumps(meta)})


def load_factors(path):
    from safetensors import safe_open
    out = {}
    with safe_open(path, framework="pt") as f:
        meta = json.loads(f.metadata()["bellowskv"])
        for key in f.keys():
            g, k = key.split(".", 1)
            out.setdefault(int(g[1:]), {})[k] = f.get_tensor(key)
    return out, meta
