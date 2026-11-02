#!/usr/bin/env python3
"""Unit tests for the paged vLLM kernels against the PyTorch references (GPU, no vLLM needed).

  store  : latent_store_triton writes the same bytes as latent_store_ref
  decode : decode_attention_paged == decode_attention_ref on ragged requests with shuffled block tables
  graph  : a CUDA graph captured once replays correctly after lengths and contents change
  chunk  : dequant_latent_paged == dequant_latent_ref
"""
import os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from bellowskv.codec import BandRotation  # noqa: E402
from bellowskv.kernels import prepare_heads, permute_index, DEFAULT_CFG  # noqa: E402
from bellowskv.vllm.attn_ops import PageLayout, latent_store_ref, decode_attention_ref, dequant_latent_ref  # noqa: E402
from bellowskv.vllm.paged_kernels import (latent_store_triton, decode_attention_paged, dequant_latent_paged,  # noqa: E402
                                           score_offsets, pick_nsplit)

torch.manual_seed(0)
dev = "cuda"
R_K, R_V, Hq, Hkv, hd, sink = 2560, 1024, 32, 8, 128, 4
lay = PageLayout(R_K, R_V, page_tokens=128, page_bytes=262144)
ranks = [20] * 64                                                   # 64 bands x 20 complex = 1280 -> R_K 2560
rot_obj = BandRotation(ranks, dev); rot = (rot_obj.band, rot_obj.is_re, rot_obj.partner)
NBLK = 256
pages_a = torch.zeros(NBLK, lay.bytes, dtype=torch.uint8, device=dev)
pages_b = torch.zeros_like(pages_a)

# requests: lengths, shuffled block tables
lens = [37, 700, 5000]
perm = torch.randperm(NBLK - 1, device=dev) + 1                    # block 0 = null block
nb = max(-(-L // 128) for L in lens)
bt = torch.zeros(len(lens), nb, dtype=torch.int32, device=dev); k = 0
for b, L in enumerate(lens):
    n = -(-L // 128); bt[b, :n] = perm[k:k + n].int(); k += n


def cos_sin(pos):
    inv = 1.0 / (500000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    ang = pos.float()[:, None] * inv[None, :]
    return ang.cos(), ang.sin()


# ---- store
ok_store = True
for b, L in enumerate(lens):
    pos = torch.arange(L, device=dev)
    acc = torch.randn(L, R_K + R_V, device=dev) * 0.5
    slots = (bt[b, pos // 128].long() * 128 + pos % 128)
    slots[:sink] = -1                                                # sinks are never read from the latent
    cos, sin = cos_sin(pos)
    latent_store_ref(acc, cos, sin, slots, pages_a, lay, rot)
    latent_store_triton(acc, cos, sin, slots, pages_b, lay, rot)
same = torch.equal(pages_a[1:], pages_b[1:])
print(f"[store] bytes identical outside the null block: {same}")

# ---- decode
W = 64
lat_len = torch.tensor([max(0, L - 1 - W - sink) for L in lens], device=dev, dtype=torch.int32)   # C = L-1 cached
X = sink + W + 1
Kw = torch.randn(len(lens), Hkv, X, hd, device=dev, dtype=torch.float16)
Vw = torch.randn_like(Kw)
wmask = torch.ones(len(lens), X, dtype=torch.bool, device=dev); wmask[0, 30:] = False
q = torch.randn(len(lens), Hq, hd, device=dev, dtype=torch.float16)
wk = (torch.randn(Hkv, hd, R_K, device=dev) * 0.03).half(); wv = (torch.randn(Hkv, R_V, hd, device=dev) * 0.03).half()
WuK_p, WuV_p = prepare_heads(wk, wv, DEFAULT_CFG, 128)
idxK = permute_index(R_K, 128).to(dev); idxV = permute_index(R_V, 256).to(dev)
scale = hd ** -0.5
ref = decode_attention_ref(q, pages_b, lay, bt, lat_len, sink, Kw, Vw, wmask, WuK_p, WuV_p, idxK, idxV, scale)
out = decode_attention_paged(q, pages_b, lay, bt, lat_len, sink, Kw, Vw, wmask, WuK_p, WuV_p, scale=scale)
err = (out.float() - ref.float()).abs().max().item(); mag = ref.float().abs().max().item()
print(f"[decode] max abs err {err:.2e} (output max {mag:.2e})")

# ---- graph: capture with one set of lengths, replay with another
B = len(lens); J = pick_nsplit(B, R_V)
ws = torch.empty(Hq, 16384, device=dev)
lat_len_buf = lat_len.clone()
qb = q.clone(); Kwb, Vwb, wmb = Kw.clone(), Vw.clone(), wmask.clone()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        decode_attention_paged(qb, pages_b, lay, bt, lat_len_buf, sink, Kwb, Vwb, wmb, WuK_p, WuV_p, scale=scale, nsplit=J, ws=ws)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    gout = decode_attention_paged(qb, pages_b, lay, bt, lat_len_buf, sink, Kwb, Vwb, wmb, WuK_p, WuV_p, scale=scale, nsplit=J, ws=ws)
new_len = torch.tensor([0, 650, 4800], device=dev, dtype=torch.int32)
lat_len_buf.copy_(new_len); qb.copy_(torch.randn_like(qb))
g.replay(); torch.cuda.synchronize()
ref2 = decode_attention_ref(qb, pages_b, lay, bt, new_len, sink, Kwb, Vwb, wmb, WuK_p, WuV_p, idxK, idxV, scale)
gerr = (gout.float() - ref2.float()).abs().max().item()
print(f"[graph] replay after length change: max abs err {gerr:.2e}")

# ---- chunk dequant
Kd, Vd = dequant_latent_paged(pages_b, lay, bt[2], 100, 1500)
Kr, Vr = dequant_latent_ref(pages_b, lay, bt[2], 100, 1500)
print(f"[dequant] K equal {torch.equal(Kd, Kr)}  V equal {torch.equal(Vd, Vr)}")
assert same and err < 2e-2 and gerr < 2e-2 and torch.equal(Kd, Kr) and torch.equal(Vd, Vr), "FAILED"
print("ALL OK")
