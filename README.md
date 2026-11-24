<div align="center">

# BellowsKV

**One latent for many layers, expanded on demand.**

Cross-layer KV-cache compression for 🤗 Transformers and vLLM.

[![arXiv](https://img.shields.io/badge/arXiv-2610.35023-b31b1b.svg)](https://arxiv.org/abs/2610.35023)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20models-BellowsKV--Flash-ffcc4d.svg)](#released-models)
[![vLLM](https://img.shields.io/badge/vLLM-0.29-30a2ff.svg)](#serving-with-vllm)
[![Transformers](https://img.shields.io/badge/transformers-%E2%89%A55.0-ffcc4d.svg)](#huggingface-transformers)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776ab.svg)](#installation)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[Paper](https://arxiv.org/abs/2610.35023) · [BellowsKV-Flash](#bellowskv-flash) · [Results](#results) · [BellowsKV-A](#bellowskv-a) · [TODO](#todo) · [Citation](#citation)

</div>

---

Every layer of an LLM keeps its own key–value cache and re-reads it at every decoding step, so KV-cache memory is
often what limits how many requests a GPU can serve. Neighbouring layers' keys and values are quite different, but
the hidden states that produce them are nearly the same. **BellowsKV** therefore maps the hidden states of
neighbouring layers to **one shared latent per token**, and each layer reads its own keys and values back through
a layer-specific projection. The model's weights are not changed.

<p align="center">
  <img src="assets/teaser.png" width="92%" alt="Cross-layer KV-cache compression: share by retraining, factorize the caches, or share the stream (BellowsKV)">
</p>
<p align="center"><sub>(c) is BellowsKV. The 26–28× in the figure is the paper's highest-ratio configuration
(BellowsKV-Quant + BellowsKV-Evict). This repo ships BellowsKV-Flash (13.4×) and BellowsKV-A (8.6×); the other
configurations are on the <a href="#todo">TODO list</a>.</sub></p>

## News

- **2026-11** — v0.2: **BellowsKV-Flash**, 13.4× smaller KV cache and faster decode than dense. Two converted
  models for Llama-3.1-8B-Instruct (Lite and Turbo), for 🤗 Transformers and vLLM 0.29.
- **2026-11** — v0.1: BellowsKV-A for 🤗 Transformers and vLLM 0.29.

## What is in this repo

| | KV cache | Decode speed vs dense | Setup |
|---|---|---|---|
| [**BellowsKV-Flash**](#bellowskv-flash) (v0.2) | 13.4× smaller | 1.0× at batch 1, up to 4.6× throughput | download the converted params (1.5 GB) |
| [BellowsKV-A](#bellowskv-a) (v0.1) | 8.6× smaller | 0.94–0.97× throughput | none: fitted at load in about 1 min |

## Installation

```bash
git clone https://github.com/MorrisZJ/BellowsKV.git
cd BellowsKV
pip install -e .
pip install vllm==0.29.0          # needed for BellowsKV-Flash (its kernels) and for serving
pip install -e '.[eval]'          # optional: LongBench evaluation (rouge, rapidfuzz)
```

You need an NVIDIA GPU, Python ≥ 3.10, PyTorch ≥ 2.5, Triton ≥ 3.0 and transformers ≥ 5.0. BellowsKV-Flash needs
a Hopper GPU (H100 / H200) for FlashMLA.

---

## BellowsKV-Flash

BellowsKV-Flash is the configuration of BellowsKV built for decode speed. Each pair of layers shares one record per
token, in the layout that FlashMLA reads natively (512 latent + 64 RoPE dims, fp8). At decode, attention reads the
fp8 record through FlashMLA plus a small exact window (the first 4 and the latest 123 tokens), and merges the two in
one softmax. Nothing is reconstructed.

- **13.4× smaller KV cache.** At 32K tokens, one token takes 9.6 KiB instead of 128 KiB.
- **Faster than dense at decode.** On one H100, a decode step is as fast as dense at batch 1 and 2.35× faster at
  batch 16 (32K context). The freed memory holds 8× the batch, so the best throughput is 4.6× dense (5.2× at 128K).
- **95.3% of dense accuracy** (Turbo) on LongBench at 32K ([Results](#results)).
- **The base model is not changed.** The release is a small delta: only the record maps (1.5 GB). The base weights
  come from Meta.

### Released models

| Model | Conversion | LongBench retention (engine / vLLM) |
|---|---|---|
| [Llama-3.1-8B-Instruct-BellowsKV-Flash-Lite](https://huggingface.co/morriszjm/Llama-3.1-8B-Instruct-BellowsKV-Flash-Lite) | closed-form init + block-wise distillation, 1.8 H100 GPU hours | 93.5 / 93.2 |
| [Llama-3.1-8B-Instruct-BellowsKV-Flash-Turbo](https://huggingface.co/morriszjm/Llama-3.1-8B-Instruct-BellowsKV-Flash-Turbo) | Lite + on-policy distillation, 1.2 more GPU hours | 95.3 / 95.1 |

Both use the base model [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
(accept its license on the Hub first) and the same deploy config, so they have the same speed. Built with Llama.

### HuggingFace Transformers

```python
import torch, bellowskv.flash
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda()

bellowskv.flash.enable(model, "morriszjm/Llama-3.1-8B-Instruct-BellowsKV-Flash-Turbo")

messages = [{"role": "user", "content": "Summarize the following report: ..."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to("cuda")
out = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

bellowskv.flash.disable(model)   # back to the stock attention
```

### Serving with vLLM

The plugin is installed with the package. Select BellowsKV-Flash per model with `--hf-overrides`:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 32768 \
    --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-batched-tokens 32768 \
    --hf-overrides '{"architectures": ["BellowsKVFlashLlamaForCausalLM"],
                     "bellowskv_flash": "morriszjm/Llama-3.1-8B-Instruct-BellowsKV-Flash-Turbo"}'
```

Then query it with any OpenAI client. The default vLLM compilation settings work (torch.compile, CUDA graphs).

**Prefill: use exact prefill.** With `--no-enable-chunked-prefill` and `--max-num-batched-tokens` at least the
longest prompt, each prompt is prefilled in one step with exact attention, which is the evaluated setting. With
chunked prefill (vLLM's default), a later chunk sees the earlier chunks only through the compressed record. It runs,
but costs accuracy, mostly on retrieval tasks:

| Prefill mode | LongBench retention (Turbo, vLLM) |
|---|---|
| exact (recommended) | 95.1 |
| chunked, 8K tokens per chunk | 87.6 |

### Reproduce

```bash
P=morriszjm/Llama-3.1-8B-Instruct-BellowsKV-Flash-Turbo

# LongBench Protocol-B at 32K: dense and Flash on one backend, then scores and retention (18 min on 4 x H100)
bellowskv-flash eval --backend engine --params $P --gpus 0,1,2,3 --out runs/turbo_engine   # 95.3
bellowskv-flash eval --backend vllm   --params $P --gpus 0,1,2,3 --out runs/turbo_vllm     # 95.1

# decode speed (a 32K needle check runs first), and vLLM serving at one memory budget
bellowskv-flash bench decode --params $P
bellowskv-flash bench serve  --params $P
```

`eval` rebuilds the Protocol-B prompts from LongBench and checks them against fingerprints of the paper's prompt
set. `--backend hf` runs the 🤗 Transformers path.

Convert the base model yourself (both releases were made this way):

```bash
# Lite: 27 min on 4 x H100 NVL
bellowskv-flash convert --model meta-llama/Llama-3.1-8B-Instruct --out flash-lite --gpus 0,1,2,3
# Turbo: a prompt set (no benchmark data), then on-policy distillation from Lite (18 min on 4 x H100 NVL)
bellowskv-flash data --out turbo_prompts.pt
torchrun --nproc_per_node 4 -m bellowskv.flash.turbo --init flash-lite --prompts turbo_prompts.pt --out flash-turbo
```

### Limits (v0.2)

- Llama-3.1-8B-Instruct; bfloat16; one GPU (vLLM tensor parallel size 1); Hopper GPUs.
- vLLM: no prefix caching, no speculative decoding, no KV connectors.
- 🤗 Transformers: batches must be unpadded (batch size 1 or equal-length prompts).
- At batch 1 and contexts well below 32K, decode is somewhat slower than dense.

---

## Results

Llama-3.1-8B-Instruct.

**Speed**, relative to dense on the same GPU (higher is faster):

| | KV cache per token | Decode step, batch 1 | Max decode throughput | vLLM serving throughput |
|---|---|---|---|---|
| Dense | 128 KiB | 1.00× | 1.00× | 1.00× |
| BellowsKV (paper) | 8.5 KiB | 0.18× | 0.08× | 0.26–0.70× |
| **BellowsKV-Flash** | **9.6 KiB** | **1.01×** | **4.6×** | **1.22–2.51×** |

32K context. Max decode throughput: each method at the largest batch its cache fits. vLLM serving: all requests
submitted at once at one memory budget, 32K prompts + 256 output tokens and 8K prompts + 2K output tokens
(BellowsKV: 8K + 256). BellowsKV-Flash: H100 NVL, CUDA graphs. BellowsKV: the paper's decode path, which rebuilds
keys and values from the latent at every step, on H200.

**Accuracy** on LongBench Protocol-B at 32K (11 tasks, all 2,028 prompts, greedy), average per-task score as % of
dense:

| Method | KV compression | LongBench retention (%) |
|---|---|---|
| Dense (bf16) | 1× | 100 |
| CommonKV | 4.0× | 94.8 |
| MiniCache | 4.8× | 73.1 |
| KVSharer | 5.0× | 87.9 |
| SnapKV-style eviction | 8.0× | 94.7 |
| SnapKV-style eviction | 16.0× | 89.5 |
| BellowsKV (paper) | 13.8× | 98.0 |
| **BellowsKV-Flash-Lite** | **13.4×** | **93.5** |
| **BellowsKV-Flash-Turbo** | **13.4×** | **95.3** |

Baselines and BellowsKV run in the xKV harness; BellowsKV-Flash runs in its deployed engine. Each is divided by the
dense run of its own harness (the two dense runs agree to 99.6%). SnapKV-style eviction is our implementation of the
SnapKV criterion.

---

## BellowsKV-A

BellowsKV-A (v0.1) needs no download: the codec is fitted when the model loads. Its key read-head is constrained to
commute with RoPE, so attention at decode is computed directly on the compressed cache and nothing is
reconstructed.

- **8.6× smaller KV cache.** On Llama-3.1-8B, one token in one group of 4 layers takes 1,904 bytes instead of
  16,384 (an INT4 key latent of width 2,560 and an INT4 value latent of width 1,024, with an fp16 scale and zero
  per 128 channels).
- **About 99% of dense accuracy.** 99.2% of the dense average on the paper's 8-task LongBench suite, and the
  same on our re-run of four of those tasks with this package.
- **8.6× the sequences at about dense throughput.** At equal GPU memory it holds 190 sequences of 32k tokens
  where the dense cache holds 22, at 0.94–0.97× dense decode throughput (one H200).
- **No training and nothing to download.** The codec is fitted when the model loads, in about 70 s for an 8B model
  on one H100, and cached on disk.

**Relation to the paper.** BellowsKV-A uses the paper's calibration, joint regression and shared INT4 latent. Its
decode path differs:
- **The paper's shipped decode path** (Appendix B.7, "Mode A") absorbs the value read-head into attention but
  still reconstructs every key at read time.
- **BellowsKV-A absorbs the keys too.** Its key read-head is fitted to commute with RoPE and made wider
  (R<sub>K</sub> = 2.5 d<sub>kv</sub>) to pay for that constraint.

### Quick start (🤗 Transformers)

```python
import torch, bellowskv
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).cuda()

bellowskv.enable(model, tok)   # first call fits the codec (~1 min) and caches it; later calls load it
# model.generate(...) works unchanged
bellowskv.disable(model)       # back to the stock dense cache
```

`enable` runs one forward pass over 32 windows of 2,048 tokens (half WikiText-2, half Python source), then solves
one regression per group of layers. The fitted factors are written to `~/.cache/bellowskv` (set `BELLOWSKV_CACHE`
to change this) and reused on every later load.

**Demo: a passcode hidden in a 32k-token document.**

```bash
python examples/hf_demo.py --context 32000 --compare-dense
```

```text
prompt: 32042 tokens, needle at 50%
[dense] answer: '48213'
[bellowskv] BellowsKV-A on: 8 groups of 4 layers, R_K=2560, R_V=1024, INT4, sink 4 + recent 64 exact, 8.61x on the compressed tokens
[BellowsKV-A] answer: '48213'
[BellowsKV-A] KV cache: 0.50 GiB held (latents 0.46 + exact windows 0.04) vs 3.91 GiB dense fp16 -> 7.8x smaller at 32044 tokens
```

(Llama-3.1-8B-Instruct on one H100. The overall ratio is 7.8× rather than 8.6× because the first 4 and the
latest 64 tokens stay exact.)

### Serving with vLLM

The BellowsKV-A plugin turns on when `BELLOWSKV_VLLM=1` is set. A stock checkpoint loads unchanged.

```bash
BELLOWSKV_VLLM=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 --no-enable-prefix-caching
```

The plugin uses the same factor cache as the 🤗 Transformers package. If no factor file exists for the model, it
fits one on first launch. During that fit a second fp16 copy of the model sits on the GPU (about 16 GB for 8B).
To avoid that, run `bellowskv.enable` once in 🤗 Transformers beforehand: it writes the same file.

### Settings

| Environment variable | Default | Meaning |
|---|---|---|
| `BELLOWSKV_CACHE` | `~/.cache/bellowskv` | where fitted factors are stored |
| `BELLOWSKV_VLLM` | `0` | `1` turns the BellowsKV-A vLLM plugin on |
| `BELLOWSKV_FACTORS` | — | load this factor file instead of looking it up in the cache (vLLM) |
| `BELLOWSKV_W` | `64` | recent tokens kept exact (vLLM) |
| `BELLOWSKV_SINK` | `4` | first tokens kept exact (vLLM) |

In 🤗 Transformers the same knobs are fields of `bellowskv.BellowsConfig`, passed as
`bellowskv.enable(model, tok, config=BellowsConfig(...))`. Generated tokens move into the compressed region in
blocks of 128 (`decode_block`), so long generations stay bounded. `decode_block=None` keeps every generated token
exact instead, as in the paper's evaluation.

### Results

**Accuracy** on LongBench, using the paper's prompts and scorer on every sample:

| | qasper | trec | samsum | lcc | avg |
|---|---|---|---|---|---|
| dense (paper / our re-run) | 46.57 / 46.56 | 72.5 / 73.0 | 43.47 / 43.70 | 63.38 / 63.52 | 56.48 / 56.70 |
| BellowsKV-A, reference evaluation code | 44.55 | 72.5 | 42.89 | 64.08 | 56.00 |
| **BellowsKV-A, this package** | **45.06** | **73.0** | **43.86** | **64.07** | **56.50** |

**Memory and speed.** One H200 under 🤗 Transformers. Each method gets the memory that 22 dense sequences of 32k
tokens need; dense runs FlashAttention-3:

| | KV-cache ratio | sequences of 32k tokens | decode throughput vs dense (32k / 128k) |
|---|---|---|---|
| dense fp16 | 1.0× | 22 | 1.00× / 1.00× |
| **BellowsKV-A** | **8.6×** | **190** | **0.94× / 0.97×** |

### Limits (v0.1)

- Llama 3.x (🤗 Transformers and vLLM) and Mistral (🤗 Transformers only).
- fp16 only; `enable` casts the model to fp16, the paper's setting.
- 🤗 Transformers: batches must be unpadded; no beam search or cache cropping.
- vLLM: tensor parallel size 1, prefix caching off, no speculative decoding, no KV connectors.
- vLLM prefill of long prompts is slow: a 32k-token prompt takes about 95 s on an H100, against about 4 s in
  🤗 Transformers. Decode is not affected.

---

## Tests

```bash
# BellowsKV-Flash
python tests/test_flash_kernels.py        # Triton kernels vs PyTorch
python tests/test_longbench_metrics.py    # LongBench metrics on hand-computed cases
python tests/test_flash_student.py        # conversion: student attention vs exact attention
python tests/check_flash_short.py --params morriszjm/Llama-3.1-8B-Instruct-BellowsKV-Flash-Turbo   # short prompts vs dense
# BellowsKV-A
python tests/check_equivalence.py         # Triton kernels vs the PyTorch reference of the same math
python tests/check_paths.py               # batches, multi-turn chats, 1,000-token generations
python tests/test_paged_kernels.py        # vLLM paged kernels, including CUDA-graph replay
```

## TODO

- ✅ BellowsKV-A in 🤗 Transformers and vLLM 0.29
- ✅ BellowsKV-Flash: conversion (Lite, Turbo), 🤗 Transformers path, vLLM 0.29 plugin, LongBench and speed scripts
- ✅ Converted models for Llama-3.1-8B-Instruct on the Hugging Face Hub
- ⬜ BellowsKV-Flash for more models (Mistral, Qwen2.5, Llama 70B-class)
- ⬜ BellowsKV-Flash with tensor parallelism and prefix caching in vLLM; padded batches in 🤗 Transformers
- ⬜ Faster BellowsKV-A prefill in vLLM
- ⬜ BellowsKV-A for Qwen2.5, Qwen3 and other QK-norm models; bf16
- ⬜ The paper's other configurations: BellowsKV-Quant, BellowsKV-Evict, and the two combined (up to 26–28×)
- ⬜ PyPI package

## Citation

```bibtex
@article{zhang2026bellowskv,
  title   = {BellowsKV: One Latent for Many Layers, Expanded on Demand},
  author  = {Zhang, Jiamu and Chen, Hanjie},
  journal = {arXiv preprint arXiv:2610.35023},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.35023}
}
```

## License

BellowsKV is released under the [Apache License 2.0](LICENSE). The BellowsKV-Flash params are derived from
Llama 3.1 and are distributed under the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/).

## Acknowledgements

BellowsKV builds on [🤗 Transformers](https://github.com/huggingface/transformers),
[vLLM](https://github.com/vllm-project/vllm), [FlashMLA](https://github.com/deepseek-ai/FlashMLA) and
[Triton](https://github.com/triton-lang/triton). Calibration uses
[WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) and Python code from
[the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol) or
[codeparrot-clean-valid](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid); the Turbo prompt set
uses WikiText-103.
