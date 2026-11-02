#!/usr/bin/env python3
"""Consistency checks for BellowsKV-A (needs a GPU).

Teacher-forced over a WikiText-2 continuation (Llama-3.1-8B-Instruct by default):
  1. BellowsKV-A vs dense: NLL, KL and top-1 agreement on the same tokens.
  2. kernel vs PyTorch: one decode step from the same cache state through the INT4 Triton kernels and through the
     PyTorch implementation of the same math. Must agree to fp16 noise.
  3. decode-time compression (decode_block=128) vs keeping every generated token exact.

    python tests/check_equivalence.py [--model NousResearch/Meta-Llama-3.1-8B-Instruct]
"""
import argparse, json, os, sys, time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import bellowskv  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
ap.add_argument("--prefix", type=int, default=6144); ap.add_argument("--cont", type=int, default=256)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_grad_enabled(False)
dev = "cuda"
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
res = {"model": a.model, "prefix": a.prefix, "cont": a.cont}

from datasets import load_dataset  # noqa: E402
txt = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(txt, return_tensors="pt").input_ids[:, :a.prefix + a.cont].to(dev)
P, C = a.prefix, a.cont


def teacher_forced(cache):
    """Logits (C, V) of the continuation tokens given the prefix, one decode step at a time."""
    out = model(ids[:, :P], past_key_values=cache, use_cache=True)
    lg = [out.logits[0, -1].float()]
    for t in range(P, P + C - 1):
        lg.append(model(ids[:, t:t + 1], past_key_values=cache, use_cache=True).logits[0, -1].float())
    return torch.stack(lg)


def compare(x, y):
    lx, ly = F.log_softmax(x, -1), F.log_softmax(y, -1)
    kl = (lx.exp() * (lx - ly)).sum(-1)
    return {"kl_mean": kl.mean().item(), "top1_agree": (x.argmax(-1) == y.argmax(-1)).float().mean().item(),
            "max_abs_logit": (x - y).abs().max().item()}


def nll(lg):
    return F.cross_entropy(lg, ids[0, P:P + C]).item()


# ---------------------------------------------------------------------------------------------- dense
t0 = time.time(); dense = teacher_forced(DynamicCache()); res["dense_nll"] = nll(dense)
print(f"[check] dense NLL {res['dense_nll']:.4f} ({time.time() - t0:.0f}s)", flush=True)

# ---------------------------------------------------------------------------------------------- ours
cfg = bellowskv.BellowsConfig(decode_block=None)          # the paper's evaluation setting: generated tokens exact
bellowskv.enable(model, tok, cfg)
rt = model.bellowskv
cache = rt.new_cache()
ours = teacher_forced(cache); res["ours_nll"] = nll(ours); res["ours_vs_dense"] = compare(dense, ours)
res["memory"] = cache.memory()
print(f"[check] BellowsKV-A NLL {res['ours_nll']:.4f}  vs dense {res['ours_vs_dense']}", flush=True)
print(f"[check] memory {json.dumps(res['memory'])}", flush=True)

# 1. kernel vs PyTorch on the same state
c1 = rt.new_cache(); model(ids[:, :P], past_key_values=c1, use_cache=True)
diffs = []
for t in range(P, P + 16):
    c2 = c1.clone()
    rt.force_torch = True;  lt = model(ids[:, t:t + 1], past_key_values=c2, use_cache=True).logits[0, -1].float()
    rt.force_torch = False; lk = model(ids[:, t:t + 1], past_key_values=c1, use_cache=True).logits[0, -1].float()
    diffs.append(((lk - lt).abs().max().item(), int(lk.argmax() == lt.argmax())))
res["kernel_vs_torch"] = {"max_abs_logit": max(d for d, _ in diffs), "top1_agree": sum(s for _, s in diffs) / len(diffs)}
print(f"[check] kernel vs torch: {res['kernel_vs_torch']}", flush=True)

# 1b. generation with decode-time compression (default config) runs and matches the paper setting closely
cache_g = rt.new_cache(); rt.cfg.decode_block = 128
g_ours = teacher_forced(cache_g); rt.cfg.decode_block = None
res["decode_block128_vs_paper_setting"] = compare(ours, g_ours); res["decode_block128_nll"] = nll(g_ours)
print(f"[check] decode_block=128 NLL {res['decode_block128_nll']:.4f} vs exact-generated {res['decode_block128_vs_paper_setting']}", flush=True)

if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
