"""Public entry point: `bellowskv.enable(model, tokenizer)`."""
from __future__ import annotations

import os
import time

import torch

from .calibration import calibration_windows, collect_statistics, layer_groups
from .config import BellowsConfig
from .fit import fit_group, load_factors, save_factors
from .runtime import BellowsCache, Runtime, make_forward

SUPPORTED = {"llama", "mistral"}


def _model_id(model):
    return getattr(model.config, "_name_or_path", None) or getattr(model, "name_or_path", "model")


def _check_model(model):
    mt = getattr(model.config, "model_type", "")
    if mt not in SUPPORTED:
        raise NotImplementedError(f"BellowsKV-A v1 supports {sorted(SUPPORTED)} models, got '{mt}'")
    for layer in model.model.layers:
        a = layer.self_attn
        if a.k_proj.bias is not None or a.v_proj.bias is not None:
            raise NotImplementedError("k/v projection biases are not supported in v1 (Qwen2-style models)")
        if getattr(a, "k_norm", None) is not None:
            raise NotImplementedError("QK-norm models cannot use the absorbable key latent")


def fit(model, tokenizer, config=None, log=print):
    """Calibrate and fit BellowsKV-A factors for `model` (no caching). Returns (factors, meta)."""
    cfg = (config or BellowsConfig()).resolve(model.config)
    t0 = time.time()
    wins, corpus = calibration_windows(tokenizer, cfg.calib_seq, cfg.calib_windows, cfg.calib_code_frac, log=log)
    log(f"[bellowskv] calibrating on {len(wins)} windows x {cfg.calib_seq} tokens ({corpus['code']} for code)")
    stats = collect_statistics(model, wins, cfg.group_size, log=log)
    groups = layer_groups(model.config.num_hidden_layers, cfg.group_size)
    factors = {}
    for gi, g in enumerate(groups):
        factors[g[0]] = fit_group(model, stats, g, cfg.r_k, cfg.r_v, cfg.ridge, log=log)
        log(f"[bellowskv] fitted group {gi + 1}/{len(groups)} (layers {g[0]}-{g[-1]})")
        torch.cuda.empty_cache()
    del stats
    meta = {"model": _model_id(model), "config": cfg.fit_key(), "calibration": corpus,
            "fit_seconds": round(time.time() - t0, 1), "torch": torch.__version__}
    try:
        import transformers
        meta["transformers"] = transformers.__version__
    except Exception:
        pass
    return factors, meta


def enable(model, tokenizer=None, config: BellowsConfig | None = None, factors_path: str | None = None, log=print):
    """Turn on BellowsKV-A for a HuggingFace Llama / Mistral model.

    Factors are loaded from `factors_path` if given; otherwise from the local cache (keyed by model and fit
    settings); otherwise fitted now (one calibration pass, a few minutes on one GPU; needs `tokenizer`) and cached.
    After this call `model.generate(...)` uses the compressed cache automatically; `model.bellowskv.new_cache()`
    gives one for manual forward loops. The model is moved to float16 (the paper's setting; the INT4 kernels
    run in fp16).
    """
    _check_model(model)
    cfg = (config or BellowsConfig()).resolve(model.config)
    if next(model.parameters()).dtype != torch.float16:
        log("[bellowskv] casting the model to float16 (BellowsKV-A runs attention in fp16)")
        model.half()
    model.eval()
    mid = _model_id(model)
    path = factors_path
    if path is None:
        cdir = os.path.expanduser(cfg.cache_dir); os.makedirs(cdir, exist_ok=True)
        path = os.path.join(cdir, f"{mid.strip('/').replace('/', '--')}.A.{cfg.fit_hash(mid)}.safetensors")
    if os.path.exists(path):
        factors, meta = load_factors(path)
        log(f"[bellowskv] loaded factors from {path}")
    else:
        if tokenizer is None:
            raise ValueError("no cached factors for this model: pass `tokenizer` so they can be fitted")
        factors, meta = fit(model, tokenizer, cfg, log=log)
        save_factors(path, factors, meta)
        log(f"[bellowskv] fitted in {meta['fit_seconds']} s, saved to {path}")
    rt = Runtime(model, cfg, factors, meta)
    rt.factors_path = path

    from importlib import import_module
    mod = import_module(type(model.model.layers[0].self_attn).__module__)
    apply_rope = getattr(mod, "apply_rotary_pos_emb")
    for l, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if not hasattr(attn, "_bellowskv_orig_forward"):
            attn._bellowskv_orig_forward = attn.forward
        attn.forward = make_forward(rt, attn, l, attn._bellowskv_orig_forward, apply_rope)

    rt.new_cache = lambda: BellowsCache(rt)
    model.bellowskv = rt
    if not hasattr(model, "_bellowskv_orig_generate"):
        model._bellowskv_orig_generate = model.generate

        def generate(*args, **kwargs):
            if kwargs.get("past_key_values") is None and kwargs.get("use_cache", True) and getattr(model, "bellowskv", None):
                kwargs["past_key_values"] = model.bellowskv.new_cache()
            return model._bellowskv_orig_generate(*args, **kwargs)
        model.generate = generate
    log(f"[bellowskv] BellowsKV-A on: {len(rt.groups)} groups of {cfg.group_size} layers, R_K={cfg.r_k}, R_V={cfg.r_v}, "
        f"INT{cfg.bits}, sink {cfg.sink} + recent {cfg.recent} exact, {cfg.compression(model.config):.2f}x on the compressed tokens")
    return model


def disable(model):
    """Restore the original attention and generate."""
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "_bellowskv_orig_forward"):
            layer.self_attn.forward = layer.self_attn._bellowskv_orig_forward
    if hasattr(model, "_bellowskv_orig_generate"):
        model.generate = model._bellowskv_orig_generate
        del model._bellowskv_orig_generate
    model.bellowskv = None
    return model
