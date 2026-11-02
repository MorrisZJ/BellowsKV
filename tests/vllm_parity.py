#!/usr/bin/env python3
"""BellowsKV-A on vLLM vs the HuggingFace reference (greedy decoding, same factors, same window).

Each mode runs in its own process (HF and vLLM do not share a GPU budget) and writes a JSON file; `compare` reads two.

    # HF BellowsKV-A reference: decode_block=1 == fixed exact window W=recent (after a prompt the last W prompt
    # tokens stay exact, then every step compresses exactly one token)
    python tests/vllm_parity.py --mode hf     --suite parity --out hf_parity.json
    python tests/vllm_parity.py --mode vllm   --suite parity --out vllm_parity.json     # needs the plugin importable
    python tests/vllm_parity.py --mode dense  --suite parity --out dense_parity.json    # stock vLLM
    python tests/vllm_parity.py --mode hfdense --suite parity --out hfdense_parity.json # stock HF
    python tests/vllm_parity.py --mode compare --a hf_parity.json --b vllm_parity.json

Suites: parity (B=1, ~6k-token WikiText prompt, 64 tokens), batch (B=4, different lengths), chunk (the parity prompt
with max_num_batched_tokens=2048 in vLLM and fed to HF in the same 2048-token chunks), needle (32k document,
passcode at 50%), replay (B=4 with different max_tokens so the batch shrinks mid-run; for graph-vs-eager checks).
`--graphs` runs vLLM with cudagraph_mode=FULL_DECODE_ONLY instead of enforce_eager.
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["hf", "hfdense", "vllm", "dense", "compare"])
ap.add_argument("--suite", default="parity",
                choices=["parity", "batch", "chunk", "needle", "replay", "batch8k", "replay8k"])
ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
ap.add_argument("--factors", default=os.environ.get("BELLOWSKV_FACTORS"))
ap.add_argument("--window", type=int, default=int(os.environ.get("BELLOWSKV_W", "64")))
ap.add_argument("--sink", type=int, default=int(os.environ.get("BELLOWSKV_SINK", "4")))
ap.add_argument("--new-tokens", type=int, default=64)
ap.add_argument("--gpu-util", type=float, default=0.6)
ap.add_argument("--max-batched", type=int, default=None, help="vLLM max_num_batched_tokens override")
ap.add_argument("--blocks", type=int, default=None, help="vLLM num_gpu_blocks_override (identical scheduling across runs)")
ap.add_argument("--graphs", action="store_true", help="vLLM: FULL_DECODE_ONLY cuda graphs instead of eager")
ap.add_argument("--compile", action="store_true", help="with --graphs: keep torch.compile (VLLM_COMPILE) on")
ap.add_argument("--default-cc", action="store_true",
                help="vLLM defaults (torch.compile + FULL_AND_PIECEWISE graphs, capture sizes 1,2,4,8)")
ap.add_argument("--out", default=None)
ap.add_argument("--force", default=None, help="HF modes: teacher-force the tokens of this result file (e.g. a vLLM run)")
ap.add_argument("--a"); ap.add_argument("--b")
a = ap.parse_args()

CHUNK = 2048
SUITE = {
    "parity": dict(lengths=[6000], offsets=[0]),
    "chunk": dict(lengths=[6000], offsets=[0], chunk=CHUNK),
    "batch": dict(lengths=[1536, 2900, 4321, 6000], offsets=[0, 9000, 18000, 27000]),
    "replay": dict(lengths=[1536, 2900, 4321, 6000], offsets=[0, 9000, 18000, 27000], max_tokens=[16, 40, 64, 28]),
    "needle": dict(context=32000, depth=0.5),
    # all four prompts fit one 8192-token step: no chunk-schedule dependence (eager vs graphs, vLLM vs HF)
    "batch8k": dict(lengths=[1024, 1700, 2300, 3000], offsets=[0, 9000, 18000, 27000]),
    "replay8k": dict(lengths=[1024, 1700, 2300, 3000], offsets=[0, 9000, 18000, 27000], max_tokens=[16, 40, 64, 28]),
}[a.suite]


# ----------------------------------------------------------------------------------------------- prompts
def build_prompts(tok):
    from datasets import load_dataset
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
                       if t.strip())
    ids = tok(text, add_special_tokens=False).input_ids
    if a.suite == "needle":
        n = SUITE["context"]
        while len(ids) < n:
            ids = ids + ids
        ids = ids[:n]
        cut = int(len(ids) * SUITE["depth"])
        doc = tok.decode(ids[:cut]) + " The secret passcode for the archive is 48213. Remember it. " + tok.decode(ids[cut:])
        msgs = [{"role": "user", "content": doc + "\n\nWhat is the secret passcode for the archive? Answer with the number only."}]
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if hasattr(p, "keys"):                              # BatchEncoding (transformers 5)
            p = p["input_ids"]
        if len(p) and isinstance(p[0], list):
            p = p[0]
        return [[int(t) for t in p]]
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    return [bos + ids[o:o + n - len(bos)] for n, o in zip(SUITE["lengths"], SUITE["offsets"])]


def max_tokens_list(n):
    mt = SUITE.get("max_tokens")
    return list(mt) if mt else [a.new_tokens] * n


# ----------------------------------------------------------------------------------------------- HF
def run_hf(dense: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = build_prompts(tok)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    if not dense:
        import bellowskv
        cfg = bellowskv.BellowsConfig(decode_block=1, recent=a.window, sink=a.sink)
        bellowskv.enable(model, tok, config=cfg, factors_path=a.factors)
    chunk = SUITE.get("chunk")
    forced = [r["tokens"] for r in json.load(open(a.force))["results"]] if a.force else None
    res = []
    for pi, (ids, mt) in enumerate(zip(prompts, max_tokens_list(len(prompts)))):
        cache = DynamicCache() if dense else model.bellowskv.new_cache()
        x = torch.tensor([ids], device="cuda")
        t0 = time.time()
        step = chunk or x.shape[1]
        for s0 in range(0, x.shape[1], step):
            out = model(x[:, s0:s0 + step], past_key_values=cache, use_cache=True, logits_to_keep=1)
        toks, lps, amax, top2 = [], [], [], []
        logits = out.logits[0, -1].float()
        steps = len(forced[pi]) if forced else mt
        for i in range(steps):
            lp = torch.log_softmax(logits, -1)
            v2, i2 = lp.topk(2)
            t = forced[pi][i] if forced else int(i2[0])
            toks.append(t); lps.append(float(lp[t])); amax.append(int(i2[0])); top2.append([float(v2[0]), float(v2[1])])
            if i == steps - 1:
                break
            out = model(torch.tensor([[t]], device="cuda"), past_key_values=cache, use_cache=True)
            logits = out.logits[0, -1].float()
        res.append(dict(prompt_len=len(ids), tokens=toks, logprobs=lps, argmax=amax, top2=top2,
                        forced=bool(forced), text=tok.decode(toks), seconds=round(time.time() - t0, 2)))
        print(f"[hf{'dense' if dense else ''}] len {len(ids)}: {tok.decode(toks)[:120]!r}", flush=True)
    return res


# ----------------------------------------------------------------------------------------------- vLLM
def run_vllm(bellows: bool):
    if bellows:
        os.environ["BELLOWSKV_VLLM"] = "1"
        os.environ.setdefault("BELLOWSKV_W", str(a.window)); os.environ.setdefault("BELLOWSKV_SINK", str(a.sink))
        if a.factors:
            os.environ["BELLOWSKV_FACTORS"] = a.factors
    else:
        os.environ["BELLOWSKV_VLLM"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    if bellows:
        import bellowskv.vllm
        bellowskv.vllm.register()
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = build_prompts(tok)
    longest = max(len(p) for p in prompts) + max(max_tokens_list(len(prompts))) + 16
    kw = dict(model=a.model, dtype="float16", enable_prefix_caching=False, gpu_memory_utilization=a.gpu_util,
              max_model_len=max(8192, longest), max_num_seqs=8, seed=0,
              max_num_batched_tokens=a.max_batched or SUITE.get("chunk") or max(8192, longest))
    if a.blocks:
        kw["num_gpu_blocks_override"] = a.blocks
    if a.default_cc:
        kw.update(enforce_eager=False, compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]})
    elif a.graphs:
        cc = {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8]}
        if not a.compile:
            cc["mode"] = 0                                   # no torch.compile: graphs around the eager forward
        kw.update(enforce_eager=False, compilation_config=cc)
    else:
        kw.update(enforce_eager=True)
    t0 = time.time()
    llm = LLM(**kw)
    print(f"[vllm] engine up in {time.time() - t0:.0f}s", flush=True)
    sps = [SamplingParams(temperature=0.0, max_tokens=mt, logprobs=1, ignore_eos=True)
           for mt in max_tokens_list(len(prompts))]
    t0 = time.time()
    try:
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sps)
    except BaseException:
        llm.llm_engine.engine_core.shutdown(timeout=30)      # do not leave a busy engine core behind
        raise
    dt = time.time() - t0
    res = []
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        toks = list(c.token_ids)
        lps = [float(d[t].logprob) if (d is not None and t in d) else None for t, d in zip(toks, c.logprobs or [])]
        res.append(dict(prompt_len=len(p), tokens=toks, logprobs=lps, text=tok.decode(toks)))
        print(f"[vllm{'' if bellows else '-dense'}] len {len(p)}: {tok.decode(toks)[:120]!r}", flush=True)
    print(f"[vllm] generate {dt:.1f}s", flush=True)
    return res


# ----------------------------------------------------------------------------------------------- compare
def compare(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    rows = []
    if any(r.get("forced") for r in B["results"]):      # B teacher-forced on A's tokens
        for x, y in zip(A["results"], B["results"]):
            n = min(len(x["tokens"]), len(y["tokens"]))
            agree = [y["argmax"][i] == x["tokens"][i] for i in range(n)]
            d = [abs(x["logprobs"][i] - y["logprobs"][i]) for i in range(n) if x["logprobs"][i] is not None]
            dis = [dict(step=i, a_tok=x["tokens"][i], b_argmax=y["argmax"][i],
                        b_margin=round(y["top2"][i][0] - y["top2"][i][1], 4),
                        b_lp_of_a_tok=round(y["logprobs"][i], 4)) for i in range(n) if not agree[i]]
            rows.append(dict(prompt_len=x["prompt_len"], n=n, teacher_forced_agreement=round(sum(agree) / n, 4),
                             mean_abs_dlogprob=round(sum(d) / len(d), 4), max_abs_dlogprob=round(max(d), 4),
                             disagreements=dis))
        print(json.dumps(dict(a=fa, b=fb, teacher_forced=True, rows=rows), indent=1))
        return
    for x, y in zip(A["results"], B["results"]):
        tx, ty = x["tokens"], y["tokens"]
        n = min(len(tx), len(ty))
        first = next((i for i in range(n) if tx[i] != ty[i]), n)
        eq = sum(tx[i] == ty[i] for i in range(n)) / max(1, n)
        d = [abs(x["logprobs"][i] - y["logprobs"][i]) for i in range(first)
             if x["logprobs"][i] is not None and y["logprobs"][i] is not None]
        rows.append(dict(prompt_len=x["prompt_len"], n=n, first_divergence=first, token_agreement=round(eq, 4),
                         mean_abs_dlogprob_prefix=round(sum(d) / len(d), 4) if d else None,
                         max_abs_dlogprob_prefix=round(max(d), 4) if d else None))
    print(json.dumps(dict(a=fa, b=fb, rows=rows), indent=1))
    if A.get("suite") == "needle" or B.get("suite") == "needle":
        for name, R in (("a", A), ("b", B)):
            print(name, "needle answer:", R["results"][0]["text"][:80].strip(), "->",
                  "PASS" if "48213" in R["results"][0]["text"] else "FAIL")


def main():
    if a.mode == "compare":
        compare(a.a, a.b)
        return
    res = run_hf(dense=a.mode == "hfdense") if a.mode in ("hf", "hfdense") else run_vllm(bellows=a.mode == "vllm")
    if a.suite == "needle":
        print("needle:", "PASS" if "48213" in res[0]["text"] else "FAIL", flush=True)
    if a.out:
        json.dump(dict(mode=a.mode, suite=a.suite, graphs=a.graphs, compile=a.compile, default_cc=a.default_cc, window=a.window, sink=a.sink,
                       results=res), open(a.out, "w"), indent=1)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)                                               # never hang at interpreter exit after a failure
