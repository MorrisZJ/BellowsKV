#!/usr/bin/env python3
"""Exercise the cache paths the equivalence check does not: batches, multi-turn chunks, long generations.

  batch     : two equal-length prompts in one batch give the same greedy tokens as each run alone
  multiturn : prompt -> generate -> a second multi-token turn (PyTorch latent path) -> generate; compared with
              dense on the same token stream (teacher-forced NLL of the second answer)
  longgen   : 1,000 teacher-forced decode steps with decode_block=128 (7 block compressions); NLL vs dense
"""
import argparse, json, os, sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import bellowskv  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
from datasets import load_dataset  # noqa: E402
txt = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(txt, return_tensors="pt").input_ids.cuda()
res = {}


def tf_nll(cache, stream, start):
    """Feed stream[:, :start] in one go, then teacher-force the rest one token at a time; NLL of stream[start:]."""
    out = model(stream[:, :start], past_key_values=cache, use_cache=True)
    lg = [out.logits[0, -1].float()]
    for t in range(start, stream.shape[1] - 1):
        lg.append(model(stream[:, t:t + 1], past_key_values=cache, use_cache=True).logits[0, -1].float())
    return F.cross_entropy(torch.stack(lg), stream[0, start:]).item()


def chunked_nll(cache, stream, cuts):
    """Multi-turn: feed chunks [cuts[i], cuts[i+1]) as multi-token forwards; NLL of the tokens after the last cut,
    teacher-forced one at a time."""
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        out = model(stream[:, c0:c1], past_key_values=cache, use_cache=True)
    lg = [out.logits[0, -1].float()]
    for t in range(cuts[-1], stream.shape[1] - 1):
        lg.append(model(stream[:, t:t + 1], past_key_values=cache, use_cache=True).logits[0, -1].float())
    return F.cross_entropy(torch.stack(lg), stream[0, cuts[-1]:]).item()


# dense references first (BellowsKV not enabled yet)
stream_mt = ids[:, :6000 + 64 + 800 + 128]
cuts = [0, 6000, 6064, 6864]                       # prompt | 64 "answer" tokens | 800-token second turn | answer
res["multiturn_dense_nll"] = chunked_nll(DynamicCache(), stream_mt, cuts)
stream_lg = ids[:, 20000:20000 + 4096 + 1000]
res["longgen_dense_nll"] = tf_nll(DynamicCache(), stream_lg, 4096)

bellowskv.enable(model, tok)
rt = model.bellowskv

# batch
p = ids[:, 30000:30000 + 3000]; q = ids[:, 40000:40000 + 3000]
gk = dict(max_new_tokens=48, do_sample=False)
o_b = model.generate(input_ids=torch.cat([p, q]), attention_mask=torch.ones(2, 3000, dtype=torch.long, device="cuda"), **gk)
o_p = model.generate(input_ids=p, attention_mask=torch.ones_like(p), **gk)
o_q = model.generate(input_ids=q, attention_mask=torch.ones_like(q), **gk)
agree = lambda x, y: (x[3000:] == y[3000:]).float().mean().item()
res["batch_token_agreement"] = [agree(o_b[0], o_p[0]), agree(o_b[1], o_q[0])]
print("[paths] batch agreement", res["batch_token_agreement"], flush=True)

# multi-turn (second and later chunks go through the PyTorch latent path)
res["multiturn_bkv_nll"] = chunked_nll(rt.new_cache(), stream_mt, cuts)
print(f"[paths] multiturn NLL dense {res['multiturn_dense_nll']:.4f} bellowskv {res['multiturn_bkv_nll']:.4f}", flush=True)

# long generation with block compression of generated tokens
c = rt.new_cache()
res["longgen_bkv_nll"] = tf_nll(c, stream_lg, 4096)
res["longgen_memory"] = c.memory()
print(f"[paths] longgen NLL dense {res['longgen_dense_nll']:.4f} bellowskv {res['longgen_bkv_nll']:.4f}; memory {json.dumps(res['longgen_memory'])}", flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
