"""Calibration: one forward pass over a fixed corpus, accumulating the second moments the fit needs.

Mirrors the research code behind the paper's numbers: windows are cut from WikiText-2 train and from
Python source, every k_proj input h_l is captured, and per group of n layers we keep the per-layer Grams
E[h_l^T h_l] and the cross terms E[h_i^T h_j] (i < j in the group). The query energy per output dimension of
q_proj weights the per-band rank allocation of the key latent.
"""
from __future__ import annotations

import torch

# Code corpus: the paper used bigcode/the-stack-smol (Python, first 800 files), which is gated on the Hub.
# Without access, the open codeparrot validation split (Python) stands in.
CODE_SOURCES = (
    ("bigcode/the-stack-smol", dict(data_dir="data/python", split="train"), "content", 800),
    ("codeparrot/codeparrot-clean-valid", dict(split="train"), "content", 800),
)


def calibration_windows(tok, seq=2048, nwin=32, code_frac=0.5, log=print, sources=CODE_SOURCES):
    from datasets import load_dataset
    nl = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip())
    code, src = "", None
    for name, kw, col, k in sources:
        try:
            ds = load_dataset(name, **kw)
            code = "\n\n".join(str(x) for x in ds[:k][col]); src = name
            break
        except Exception as e:                                           # gated / offline: try the next source
            log(f"[bellowskv] code corpus {name} unavailable ({type(e).__name__}); trying the next one")
    nl_ids = tok(nl, return_tensors="pt").input_ids[0]
    code_ids = tok(code, return_tensors="pt").input_ids[0] if code else nl_ids
    nc = int(nwin * code_frac); nn = nwin - nc
    wins = [nl_ids[i * seq:(i + 1) * seq] for i in range(nn)] + [code_ids[i * seq:(i + 1) * seq] for i in range(nc)]
    wins = [w for w in wins if w.numel() == seq]
    return wins, {"text": "Salesforce/wikitext wikitext-2-raw-v1 train", "code": src or "none (text only)",
                  "windows": len(wins), "seq": seq}


def layer_groups(num_layers: int, n: int) -> list[list[int]]:
    """Consecutive blocks of n layers; a trailing remainder (< n layers) stays dense."""
    return [list(range(i, i + n)) for i in range(0, num_layers, n) if i + n <= num_layers]


@torch.no_grad()
def collect_statistics(model, windows, n, stats_device=None, log=print):
    """Returns {"G": {l: (d,d)}, "X": {(i,j): (d,d)}, "QE": {l: (d_q,)}, "tokens": int}, fp32 on stats_device."""
    layers = model.model.layers; L = len(layers)
    dev = next(model.parameters()).device; sd = stats_device or dev
    groups = layer_groups(L, n)
    grouped = [l for g in groups for l in g]
    d = model.config.hidden_size
    H = {}
    QE = {l: torch.zeros(layers[l].self_attn.q_proj.out_features, device=sd, dtype=torch.float64) for l in grouped}
    hooks = []
    for l in grouped:
        attn = layers[l].self_attn
        hooks.append(attn.k_proj.register_forward_pre_hook(lambda m, a, l=l: H.__setitem__(l, a[0].detach())))

        def qhook(m, i, o, l=l):                       # must return None (a returned value replaces the output)
            QE[l].add_((o.detach().float() ** 2).reshape(-1, o.shape[-1]).sum(0).to(sd, torch.float64))
        hooks.append(attn.q_proj.register_forward_hook(qhook))
    G = {l: torch.zeros(d, d, device=sd, dtype=torch.float32) for l in grouped}
    X = {(g[i], g[j]): torch.zeros(d, d, device=sd, dtype=torch.float32)
         for g in groups for i in range(n) for j in range(i + 1, n)}
    nt = 0
    try:
        for wi, w in enumerate(windows):
            H.clear()
            model(w.unsqueeze(0).to(dev), use_cache=False)
            h = {l: H[l].reshape(-1, d).float().to(sd) for l in grouped}
            for l in grouped:
                G[l] += h[l].T @ h[l]
            for (i, j) in X:
                X[(i, j)] += h[i].T @ h[j]
            nt += h[grouped[0]].shape[0]
            if (wi + 1) % 8 == 0:
                log(f"[bellowskv] calibration window {wi + 1}/{len(windows)}")
    finally:
        for hk in hooks:
            hk.remove()
        H.clear()
    for M in list(G.values()) + list(X.values()):
        M /= nt
    return {"G": G, "X": X, "QE": QE, "tokens": nt}


def joint_gram(stats, grp):
    """The (n*d, n*d) joint second moment of the group's stacked inputs [h_g0 | h_g0+1 | ...]."""
    G, X = stats["G"], stats["X"]
    rows = []
    for a in grp:
        row = []
        for b in grp:
            row.append(G[a] if a == b else (X[(a, b)] if a < b else X[(b, a)].T))
        rows.append(torch.cat(row, 1))
    return torch.cat(rows, 0)
