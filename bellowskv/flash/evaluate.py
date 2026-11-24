"""`bellowskv-flash eval`: LongBench Protocol-B (32K) through a BellowsKV-Flash backend and the dense model on the
same backend, then the scores and the retention table.

  bellowskv-flash eval --params <dir|repo> --out runs/pb --gpus 0,1,2,3          # engine: dense, then flash8
  bellowskv-flash eval --params <dir|repo> --out runs/pb --backend hf --variants flash8 --ref engine-dense
  bellowskv-flash eval --out runs/pb --score-only

Backends:
  engine : the decode engine (vLLM kernels, one CUDA graph per decode step). The release numbers come from it.
  hf     : HF Transformers. flash = `bellowskv.flash.enable(model, params)` (fast backend), dense = the plain model.
  vllm   : vLLM 0.29 serving the samples in batches. flash8 = the plugin (bellowskv.flash.vllm, --hf-overrides),
           dense = stock vLLM; both with exact prefill and no prefix caching.
Every sample: the whole prompt in one prefill (exact attention), greedy decoding (longbench.py has the protocol);
engine and hf run batch 1, vllm batches the samples (so its greedy tokens depend slightly on the batching). Each run is <out>/<backend>-<variant>/ and is resumable: finished samples are skipped. The retention of a
run is relative to --ref (default: the dense run of the same backend).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

from . import longbench as LB

# One engine cache size for every run, so the results do not depend on how the samples are split over the GPUs.
TCAP = LB.DATALEN + max(LB.GEN_LEN.values()) + 64
VARIANTS = {"engine": ("dense", "flash16", "flash8"), "hf": ("dense", "flash16", "flash8"),
            "vllm": ("dense", "flash8", "flash8-chunked")}
# Measured seconds per sample (engine, H100). Only used to balance the samples over the GPUs.
COST = {"gov_report": 3.6, "qmsum": 1.5, "repobench-p": 1.0, "narrativeqa": 1.0, "musique": 0.8, "lcc": 0.8,
        "hotpotqa": 0.7, "passage_retrieval_en": 0.6, "multifieldqa_en": 0.5, "qasper": 0.4, "2wikimqa": 0.4}


# ----------------------------------------------------------------------------------------------- prompts
def prompts(out: str, model: str, tasks, log=print) -> dict:
    """<out>/prompts.pt: the tokenized prompt set (built once per output dir) and the model's EOS tokens."""
    import torch
    path = os.path.join(out, "prompts.pt")
    P = torch.load(path) if os.path.exists(path) else dict(model=model, revision=LB.REVISION, tasks={})
    need = [t for t in tasks if t not in P["tasks"]]
    if need or "eos" not in P:
        from transformers import AutoTokenizer, GenerationConfig
        if need:
            P["tasks"].update(LB.build(AutoTokenizer.from_pretrained(model), need, log=log))
        eos = GenerationConfig.from_pretrained(model).eos_token_id
        P["eos"] = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        torch.save(P, path)
    if P["model"] != model:
        log(f"[eval] note: {path} was built with the tokenizer of {P['model']!r}")
    return P


def n_samples(P: dict, task: str, limit: int) -> int:
    n = len(P["tasks"][task]["ids"])
    return min(n, limit) if limit else n


def finished(rundir: str, task: str) -> dict:
    """{i: row} over the run's part files (a line cut by a crash is skipped; that sample runs again)."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(rundir, f"{task}.part*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[r["i"]] = r
    return rows


def split(items, P: dict, n: int):
    """Longest expected sample first, each to the least loaded worker."""
    cost = lambda ti: COST.get(ti[0], 1.0) + P["tasks"][ti[0]]["ids"][ti[1]].numel() / 1e5
    load, parts = [0.0] * n, [[] for _ in range(n)]
    for ti in sorted(items, key=cost, reverse=True):
        k = min(range(n), key=load.__getitem__)
        parts[k].append(ti); load[k] += cost(ti)
    return [sorted(p) for p in parts]


# ----------------------------------------------------------------------------------------------- generation
def generator(backend: str, variant: str, model: str, params: str, eos: list[int]):
    """gen(ids (T,) on cuda, max_new) -> generated token ids (the EOS token included when it is generated)."""
    import torch
    flash = variant.startswith("flash")
    if backend == "engine":
        from . import engine as EN
        M = EN.Model(model, params if flash else None, "cuda")
        E = EN.Engine(M, EN.Cache(M, variant, 1, TCAP))
        es = set(eos)
        return lambda ids, G: E.generate(ids, G, eos=es)
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if flash:
        from .hf import enable
        enable(m, params, backend="fast", fp8=variant == "flash8")

    def gen(ids, G):
        out = m.generate(ids[None], max_new_tokens=G, do_sample=False, eos_token_id=eos, pad_token_id=eos[0],
                         temperature=None, top_p=None)
        return out[0, ids.numel():].tolist()
    return gen


def vllm_worker(S: dict, P: dict) -> int:
    """vLLM backend: one engine per worker, its samples batched by vLLM's scheduler (written in chunks, so a stopped
    run resumes). dense = stock vLLM; flash8 = the plugin (--hf-overrides). Both: bf16, exact prefill (no chunked
    prefill, the whole prompt in one step), no prefix caching."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    t0 = time.time()
    maxlen = LB.DATALEN + max(LB.GEN_LEN.values()) + 64
    chunked = S["variant"].endswith("-chunked")             # chunked prefill, 8K-token chunks (the approximation)
    kw = dict(model=S["model"], dtype="bfloat16", enable_prefix_caching=False, enable_chunked_prefill=chunked,
              max_model_len=maxlen, max_num_batched_tokens=8192 if chunked else maxlen, gpu_memory_utilization=0.9,
              seed=0)
    if S["variant"].startswith("flash"):
        kw["hf_overrides"] = {"architectures": ["BellowsKVFlashLlamaForCausalLM"], "bellowskv_flash": S["params"]}
    llm = LLM(**kw)
    print(f"[worker {S['rank']}] vllm-{S['variant']}: {len(S['items'])} samples, engine up in {time.time() - t0:.0f}s",
          flush=True)
    t0, files, items = time.time(), {}, S["items"]
    for c0 in range(0, len(items), 256):
        chunk = items[c0:c0 + 256]
        prompts = [TokensPrompt(prompt_token_ids=P["tasks"][t]["ids"][i].tolist()) for t, i in chunk]
        sps = [SamplingParams(temperature=0.0, max_tokens=P["tasks"][t]["gen_len"], stop_token_ids=S["eos"])
               for t, _ in chunk]
        for (t, i), o in zip(chunk, llm.generate(prompts, sps, use_tqdm=False)):
            if t not in files:
                files[t] = open(os.path.join(S["rundir"], f"{t}.part{S['rank']}.jsonl"), "a")
            files[t].write(json.dumps(dict(i=i, T=P["tasks"][t]["ids"][i].numel(), toks=list(o.outputs[0].token_ids)))
                           + "\n")
            files[t].flush()
        print(f"[worker {S['rank']}] {c0 + len(chunk)}/{len(items)} in {time.time() - t0:.0f}s", flush=True)
    print(f"[worker {S['rank']}] WORKER_DONE {time.time() - t0:.0f}s", flush=True)
    return 0


def worker(spec_path: str) -> int:
    import torch
    S = json.load(open(spec_path))
    P = torch.load(S["prompts"])
    if S["backend"] == "vllm":
        return vllm_worker(S, P)
    files, t0 = {}, time.time()
    with torch.inference_mode():
        gen = generator(S["backend"], S["variant"], S["model"], S["params"], S["eos"])
        print(f"[worker {S['rank']}] {S['backend']}-{S['variant']}: {len(S['items'])} samples, model loaded in "
              f"{time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        for k, (t, i) in enumerate(S["items"]):
            ids = P["tasks"][t]["ids"][i].long().cuda()
            toks = gen(ids, P["tasks"][t]["gen_len"])
            if t not in files:
                files[t] = open(os.path.join(S["rundir"], f"{t}.part{S['rank']}.jsonl"), "a")
            files[t].write(json.dumps(dict(i=i, T=ids.numel(), toks=toks)) + "\n")
            files[t].flush()
            if (k + 1) % 100 == 0:
                print(f"[worker {S['rank']}] {k + 1}/{len(S['items'])} in {time.time() - t0:.0f}s", flush=True)
    print(f"[worker {S['rank']}] WORKER_DONE {time.time() - t0:.0f}s", flush=True)
    return 0


def run(a, P: dict, variant: str, tasks, gpus, log=print) -> bool:
    name = f"{a.backend}-{variant}"
    rundir = os.path.join(a.out, name)
    os.makedirs(rundir, exist_ok=True)
    items = []
    for t in tasks:
        done = finished(rundir, t)
        items += [(t, i) for i in range(n_samples(P, t, a.limit)) if i not in done]
    meta = dict(backend=a.backend, variant=variant, model=a.model, params=a.params if variant.startswith("flash") else "",
                limit=a.limit, tasks=list(tasks), tcap=TCAP if a.backend == "engine" else None)
    json.dump(meta, open(os.path.join(rundir, "run.json"), "w"), indent=1)
    if not items:
        log(f"[eval] {name}: complete")
        return True
    total, t0 = len(items), time.time()
    log(f"[eval] {name}: {total} samples on GPUs {','.join(gpus)}")
    procs = {}
    for r, (g, part) in enumerate(zip(gpus, split(items, P, len(gpus)))):
        if not part:
            continue
        spec = os.path.join(rundir, f"worker{r}.json")
        json.dump(dict(backend=a.backend, variant=variant, model=a.model, params=a.params, rank=r, rundir=rundir,
                       prompts=os.path.join(a.out, "prompts.pt"), eos=P["eos"], items=part), open(spec, "w"))
        logf = open(os.path.join(rundir, f"worker{r}.log"), "a")
        cmd = [sys.executable, "-m", "bellowskv.flash.evaluate", "--worker", spec]
        procs[r] = subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": g}, stdout=logf,
                                    stderr=subprocess.STDOUT)
    want = {t: {i for tt, i in items if tt == t} for t in tasks}
    last = 0.0
    while any(p.poll() is None for p in procs.values()):
        time.sleep(10)
        if time.time() - last >= 300:                  # progress every 5 min
            last = time.time()
            left = sum(len(want[t] - finished(rundir, t).keys()) for t in tasks)
            log(f"[eval] {name}: {total - left}/{total} samples, {(time.time() - t0) / 60:.1f} min")
    bad = {r: p.returncode for r, p in procs.items() if p.returncode != 0}
    for r, code in bad.items():
        tail = open(os.path.join(rundir, f"worker{r}.log")).read().splitlines()[-15:]
        log(f"[eval] {name}: worker {r} failed (exit {code}):\n  " + "\n  ".join(tail))
    log(f"[eval] {name}: {'FAILED' if bad else 'done'} in {(time.time() - t0) / 60:.1f} min")
    return not bad


# ----------------------------------------------------------------------------------------------- scoring
def score(out: str, ref: str | None = None, log=print) -> dict:
    """Score every finished run under `out`; writes <run>/<task>.scores.jsonl, results.json and results.md."""
    import torch
    from transformers import AutoTokenizer
    P = torch.load(os.path.join(out, "prompts.pt"))
    tok = AutoTokenizer.from_pretrained(P["model"])
    res = {}
    for meta_f in sorted(glob.glob(os.path.join(out, "*", "run.json"))):
        rundir = os.path.dirname(meta_f); name = os.path.basename(rundir)
        meta = json.load(open(meta_f)); R = {}
        for t in [t for t in LB.TASKS if t in P["tasks"] and t in meta["tasks"]]:
            n, task = n_samples(P, t, meta.get("limit", 0)), P["tasks"][t]
            rows = finished(rundir, t)
            if any(i not in rows for i in range(n)):
                R[t] = dict(n=n, done=sum(1 for i in range(n) if i in rows))
                continue
            bad = [i for i in range(n) if rows[i]["T"] != task["ids"][i].numel()]
            if bad:
                raise ValueError(f"{name} {t}: the prompt of sample {bad[0]} differs from prompts.pt")
            preds = [LB.decode(tok, rows[i]["toks"]) for i in range(n)]
            avg, per = LB.score_task(t, preds, task["answers"][:n], task["classes"][:n])
            R[t] = dict(n=n, score=avg)
            with open(os.path.join(rundir, f"{t}.scores.jsonl"), "w") as f:
                for i, (p, s) in enumerate(zip(preds, per)):
                    f.write(json.dumps(dict(i=i, score=s, pred=p)) + "\n")
        res[name] = dict(meta=meta, tasks=R)
    md = table(res, ref)
    json.dump(dict(ref=ref, runs=res), open(os.path.join(out, "results.json"), "w"), indent=1)
    open(os.path.join(out, "results.md"), "w").write(md)
    log(md)
    return res


def table(res: dict, ref: str | None = None) -> str:
    tasks = [t for t in LB.TASKS if any(t in r["tasks"] for r in res.values())]
    head = " | ".join(LB.SHORT[t] for t in tasks) + " | Avg |"
    sep = "|---|" + "---|" * (len(tasks) + 1)

    def avg_cell(ok, bold=False):
        if not ok:
            return "-"
        s = f"{sum(ok) / len(ok):.{1 if bold else 2}f}"
        return (f"**{s}**" if bold else s) + ("" if len(ok) == len(tasks) else f" ({len(ok)}/{len(tasks)} tasks)")

    out = ["Scores (x100), LongBench Protocol-B at 32K, greedy:\n", "| run | " + head, sep]
    for name, r in res.items():
        S = r["tasks"]
        cell = lambda t: (f"{100 * S[t]['score']:.2f}" if "score" in S[t] else f"{S[t]['done']}/{S[t]['n']}") if t in S else "-"
        out.append(f"| {name} | " + " | ".join(cell(t) for t in tasks) + " | "
                   + avg_cell([100 * S[t]["score"] for t in tasks if "score" in S.get(t, {})]) + " |")
    rows = []
    for name, r in res.items():
        base = ref or f"{r['meta']['backend']}-dense"
        if name == base or base not in res:
            continue
        S, B = r["tasks"], res[base]["tasks"]
        ok = lambda t: "score" in S.get(t, {}) and "score" in B.get(t, {}) and S[t]["n"] == B[t]["n"] and B[t]["score"]
        vals = [100 * S[t]["score"] / B[t]["score"] if ok(t) else None for t in tasks]
        rows.append(f"| {name} vs {base} | " + " | ".join(f"{v:.1f}" if v is not None else "-" for v in vals) + " | "
                    + avg_cell([v for v in vals if v is not None], bold=True) + " |")
    if rows:
        out += ["\nRetention (% of the reference run per task; Avg = mean over the tasks):\n", "| run | " + head, sep] + rows
    limited = sorted({r["meta"].get("limit") for r in res.values()} - {0, None})
    if limited:
        out.append(f"\nNote: runs limited to the first {limited} prompts per task are not the full protocol.")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bellowskv-flash eval", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="base model (local dir or HF Hub id)")
    ap.add_argument("--params", default="", help="BellowsKV-Flash params dir or HF Hub repo id (flash variants)")
    ap.add_argument("--backend", default="engine", choices=sorted(VARIANTS))
    ap.add_argument("--variants", default="dense,flash8", help="comma-separated: dense, flash16, flash8")
    ap.add_argument("--tasks", default=",".join(LB.TASKS))
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids, one worker each")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts per task (smoke test; 0 = the full set)")
    ap.add_argument("--ref", default="", help="reference run for the retention (default: <backend>-dense)")
    ap.add_argument("--out", default="", help="output dir (required)")
    ap.add_argument("--score-only", action="store_true", help="only (re)score the finished runs under --out")
    ap.add_argument("--worker", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker:
        return worker(a.worker)
    if not a.out:
        ap.error("--out is required")
    os.makedirs(a.out, exist_ok=True)
    if not a.score_only:
        tasks, variants = a.tasks.split(","), a.variants.split(",")
        bad = [t for t in tasks if t not in LB.TASKS] + [v for v in variants if v not in VARIANTS[a.backend]]
        if bad:
            ap.error(f"unknown task / variant {bad}; tasks: {LB.TASKS}; variants: {VARIANTS[a.backend]}")
        if any(v.startswith("flash") for v in variants) and not a.params:
            ap.error("flash variants need --params")
        P = prompts(a.out, a.model, tasks)
        for v in variants:
            if not run(a, P, v, tasks, a.gpus.split(",")):
                return 1
    score(a.out, a.ref or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
