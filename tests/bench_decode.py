#!/usr/bin/env python3
"""Decode step time and cache bytes, dense HF cache vs BellowsKV-A, at a given context and batch.

One sequence of T random tokens is prefilled and its cache replicated B times, then `steps` greedy decode steps are timed (whole model:
32 layers, MLPs, lm_head) with a plain forward loop. Reports ms/step, tokens/s and cache bytes per sequence, so
the equal-memory batch of each arm follows from the bytes.

    python tests/bench_decode.py --seq 32768 --batches 1,4,8
"""
import argparse, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import bellowskv  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
ap.add_argument("--seq", type=int, default=32768)
ap.add_argument("--batches", default="1,4,8,16")
ap.add_argument("--steps", type=int, default=32); ap.add_argument("--warm", type=int, default=4)
ap.add_argument("--arms", default="dense,bellowskv")
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
V = model.config.vocab_size
res = {"model": a.model, "seq": a.seq, "rows": []}


def prefill_one(arm):
    """One sequence of T random tokens (only the last position's logits are computed)."""
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, V - 1000, (1, a.seq), generator=g).cuda()
    cache = model.bellowskv.new_cache() if arm == "bellowskv" else DynamicCache()
    torch.cuda.synchronize(); t0 = time.time()
    step = 8192 if arm == "dense" else a.seq                   # dense: chunked so the T x T mask stays small
    for s0 in range(0, a.seq, step):
        out = model(ids[:, s0:s0 + step], past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    return cache, out.logits[:, -1].argmax(-1, keepdim=True), time.time() - t0


def expand(cache, B):
    """Replicate a batch-1 cache B times (decode cost and memory do not depend on the values)."""
    if isinstance(cache, DynamicCache):
        for l in cache.layers:
            l.keys = l.keys.repeat(B, 1, 1, 1); l.values = l.values.repeat(B, 1, 1, 1)
        return cache
    c = cache
    rep = lambda t: None if t is None else t.repeat(B, *([1] * (t.dim() - 1)))
    for g in c.groups:
        for k in ("cK", "sK", "zK", "cV", "sV", "zV", "tK", "tV"):
            setattr(g, k, rep(getattr(g, k)))
        g.B = B
    for w in c.windows:
        w.K, w.V, w.B = rep(w.K), rep(w.V), B
    c.B = B
    return c


def run(arm, B):
    cache, nxt, t_pre = prefill_one(arm)
    cache = expand(cache, B); nxt = nxt.repeat(B, 1)
    times = []
    for i in range(a.warm + a.steps):
        torch.cuda.synchronize(); t0 = time.time()
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        if i >= a.warm:
            times.append(time.time() - t0)
    step = sum(times) / len(times)
    if arm == "bellowskv":
        cache_bytes = cache.memory()["total_allocated"]
    else:
        cache_bytes = sum(l.keys.numel() * 2 + l.values.numel() * 2 for l in cache.layers)
    row = {"arm": arm, "B": B, "prefill_1seq_s": round(t_pre, 2), "step_ms": round(step * 1e3, 2), "tok_s": round(B / step, 1),
           "cache_bytes_per_seq": int(cache_bytes / B), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    print(json.dumps(row), flush=True)
    del cache, out; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return row


for arm in a.arms.split(","):
    if arm == "bellowskv":
        bellowskv.enable(model, tok)
    for B in [int(x) for x in a.batches.split(",")]:
        try:
            res["rows"].append(run(arm, B))
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({"arm": arm, "B": B, "oom": True}), flush=True); torch.cuda.empty_cache()
    if arm == "bellowskv":
        bellowskv.disable(model)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
