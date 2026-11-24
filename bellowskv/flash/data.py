"""Long-context instruction prompts for the Turbo stage (no evaluation-benchmark data).

Text prompts: several WikiText-103 train articles concatenated to 8K-31K tokens, then an instruction about one of
them (summarize an article, describe a section, find which article holds a section). Code prompts: codeparrot
Python files concatenated, then a completion or a question about a function in the context. Raw prompts (no chat
template), as LongBench-style harnesses feed the model. No answers: the teacher supplies the targets.
Deterministic for a given tokenizer and seed (the reference set: Llama-3.1 tokenizer, n 4000, seed 0).

    python -m bellowskv.flash.data --model meta-llama/Llama-3.1-8B-Instruct --out turbo_prompts.pt
"""
from __future__ import annotations

import argparse
import random
import re

import torch

KINDS = [("summ", 0.25), ("sec", 0.25), ("find", 0.15), ("code_cont", 0.20), ("code_qa", 0.15)]


def build(tok, n=4000, seed=0, lo=8192, hi=31000, log=print):
    from datasets import load_dataset
    rng = random.Random(seed)
    TITLE = re.compile(r"^ = ([^=].*[^=]) = $"); SEC = re.compile(r"^ = = ([^=].*[^=]) = = $")
    arts, cur = [], None
    for line in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]:
        s = line.rstrip("\n")
        m = TITLE.match(s)
        if m:
            if cur and len(cur["text"]) > 2000:
                arts.append(cur)
            cur = dict(title=m.group(1).strip(), secs=[], text="")
            if len(arts) >= 8000:
                break
            continue
        if cur is None:
            continue
        m = SEC.match(s)
        if m:
            cur["secs"].append(m.group(1).strip())
        cur["text"] += line
    for a in arts:
        a["body"] = f"Article: {a['title']}\n{a['text'].strip()}"
        a["ntok"] = len(tok(a["body"]).input_ids)
    log(f"[data] {len(arts)} articles")
    code = [str(x) for x in load_dataset("codeparrot/codeparrot-clean-valid", split="train")["content"][:20000]]
    code = [c for c in code if 800 < len(c) < 60000]
    code_ntok = [len(tok(c).input_ids) for c in code]
    log(f"[data] {len(code)} code files")
    art_ntok = [a["ntok"] for a in arts]

    def fill(pool_ntok, L):
        idx, tot = [], 0
        while tot < L:
            i = rng.randrange(len(pool_ntok))
            if i in idx or tot + pool_ntok[i] > hi - 400:
                if tot > lo:
                    break
                continue
            idx.append(i); tot += pool_ntok[i]
        return idx

    prompts, kinds = [], []
    while len(prompts) < n:
        kind = rng.choices([k for k, _ in KINDS], [w for _, w in KINDS])[0]
        L = rng.randint(lo, hi - 1000)
        if kind in ("summ", "sec", "find"):
            idx = fill(art_ntok, L)
            cand = [i for i in idx if arts[i]["secs"]] if kind != "summ" else idx
            if not cand:
                continue
            t = arts[rng.choice(cand)]
            ctx = "\n\n".join(arts[i]["body"] for i in idx)
            head = "Below are several encyclopedia articles.\n\n"
            if kind == "summ":
                q = f"\n\nWrite a short summary of the article titled \"{t['title']}\".\n\nSummary:"
            elif kind == "sec":
                q = (f"\n\nQuestion: What is described in the section \"{rng.choice(t['secs'])}\" of the article "
                     f"\"{t['title']}\"? Answer in one or two sentences.\nAnswer:")
            else:
                q = f"\n\nQuestion: Which article has a section titled \"{rng.choice(t['secs'])}\"? Answer with the article title only.\nAnswer:"
            text = head + ctx + q
        else:
            idx = fill(code_ntok, L)
            ctx = "\n\n".join(code[i] for i in idx)
            if kind == "code_cont":
                lines = ctx.split("\n")
                cut = rng.randint(int(len(lines) * 0.9), len(lines) - 1)
                text = "Complete the next lines of the following code.\n\n" + "\n".join(lines[:cut]) + "\n"
            else:
                fns = re.findall(r"^\s*def ([A-Za-z_][A-Za-z0-9_]*)\(", ctx, flags=re.M)
                if not fns:
                    continue
                text = ctx + f"\n\nQuestion: What does the function `{rng.choice(fns)}` do? Answer briefly.\nAnswer:"
        ids = tok(text).input_ids
        if not (lo <= len(ids) <= hi):
            continue
        prompts.append(torch.tensor(ids, dtype=torch.int32)); kinds.append(kind)
        if len(prompts) % 500 == 0:
            log(f"[data] {len(prompts)} prompts")
    return prompts, kinds


def main(argv=None):
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description="Turbo-stage prompt set (WikiText-103 train + codeparrot)")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    prompts, kinds = build(AutoTokenizer.from_pretrained(a.model), a.n, a.seed)
    torch.save(dict(ids=prompts, kinds=kinds, seed=a.seed,
                    source="WikiText-103 train articles (first 8000 with >2000 chars) + codeparrot-clean-valid Python"), a.out)
    L = [p.numel() for p in prompts]
    print(f"[data] wrote {a.out}: {len(prompts)} prompts, tokens {min(L)}..{max(L)} (mean {sum(L) // len(L)}), "
          + ", ".join(f"{k}={kinds.count(k)}" for k, _ in KINDS), flush=True)


if __name__ == "__main__":
    main()
