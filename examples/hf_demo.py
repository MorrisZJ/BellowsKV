#!/usr/bin/env python3
"""Long-context question answering with BellowsKV-A under HuggingFace transformers.

A needle (a passcode) is hidden in a long document; the model is asked for it. The script prints the answer,
the KV-cache memory held (compressed latents + exact windows) against what a dense fp16 cache would hold, and the
decode speed. `--compare-dense` runs the same prompt with the stock dense cache first.

    python examples/hf_demo.py --model NousResearch/Meta-Llama-3.1-8B-Instruct --context 32000 --compare-dense
"""
import argparse, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bellowskv  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
ap.add_argument("--context", type=int, default=32000, help="approximate document length in tokens")
ap.add_argument("--depth", type=float, default=0.5, help="where the needle sits (0 = start, 1 = end)")
ap.add_argument("--max-new-tokens", type=int, default=64)
ap.add_argument("--compare-dense", action="store_true")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()

# ---- the document: WikiText-2 test text, with the needle inserted at the requested depth
from datasets import load_dataset  # noqa: E402
text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, add_special_tokens=False).input_ids
while len(ids) < a.context:
    ids = ids + ids
ids = ids[:a.context]
cut = int(len(ids) * a.depth)
needle = " The secret passcode for the archive is 48213. Remember it. "
doc = tok.decode(ids[:cut]) + needle + tok.decode(ids[cut:])
messages = [{"role": "user", "content": doc + "\n\nWhat is the secret passcode for the archive? Answer with the number only."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
n_prompt = inputs["input_ids"].shape[1]
print(f"prompt: {n_prompt} tokens, needle at {a.depth:.0%}")


def run(label, **kw):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False, return_dict_in_generate=True, **kw)
    torch.cuda.synchronize(); dt = time.time() - t0
    new = out.sequences[0, n_prompt:]
    print(f"\n[{label}] answer: {tok.decode(new, skip_special_tokens=True).strip()!r}")
    print(f"[{label}] {len(new)} tokens in {dt:.1f}s (prefill included); peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    return out


if a.compare_dense:
    run("dense")

bellowskv.enable(model, tok)                       # first run fits the codec (a few minutes), later runs load it
cache = model.bellowskv.new_cache()
run("BellowsKV-A", past_key_values=cache)
m = cache.memory()
gib = lambda b: b / 2**30
print(f"[BellowsKV-A] KV cache: {gib(m['total_allocated']):.2f} GiB held (latents {gib(m['latent_allocated']):.2f} + "
      f"exact windows {gib(m['windows']):.2f}) vs {gib(m['dense_equivalent']):.2f} GiB dense fp16 "
      f"-> {m['ratio_vs_dense']:.1f}x smaller at {m['tokens']} tokens")
