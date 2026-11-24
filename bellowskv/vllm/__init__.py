"""BellowsKV-A for vLLM 0.29 (out-of-tree plugin).

Enable with the `vllm.general_plugins` entry point (`bellowskv = bellowskv.vllm:plugin`, declared in pyproject.toml)
plus `BELLOWSKV_VLLM=1` in the environment of every vLLM process; or call `bellowskv.vllm.register()` yourself in a
single-process setup. Knobs (environment): BELLOWSKV_W (exact window, default 64), BELLOWSKV_SINK (4),
BELLOWSKV_CACHE (factor cache dir, as the HF package), BELLOWSKV_FACTORS (explicit factor file).

Run with dtype=float16, enable_prefix_caching=False, no speculative decoding, no KV connectors, TP=1.

`register()` does three things, idempotently:
  * appends `vllm::bellows_attention` to CompilationConfig._attention_ops (so it is a piecewise splitting op);
  * registers SinkWindowSpec -> SinkWindowManager in vLLM's KVCacheSpecRegistry (after the built-ins);
  * rebinds the LlamaForCausalLM architecture to BellowsLlamaForCausalLM (a stock checkpoint loads unchanged).
"""
from __future__ import annotations

import os

_REGISTERED = False

ARCHS = {"LlamaForCausalLM": "bellowskv.vllm.model:BellowsLlamaForCausalLM"}


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.config.compilation import CompilationConfig
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.v1.kv_cache_interface import SlidingWindowSpec
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    from .op import QUALIFIED_OP_NAME, register_op
    from .specs import SinkWindowManager, SinkWindowSpec

    register_op()
    if QUALIFIED_OP_NAME not in CompilationConfig._attention_ops:
        CompilationConfig._attention_ops.append(QUALIFIED_OP_NAME)
    # Built-ins first: _ensure_registered() skips them once anything is in the registry.
    KVCacheSpecRegistry._ensure_registered()
    KVCacheSpecRegistry.register(SinkWindowSpec, SinkWindowManager, uniform_type_base_spec=SlidingWindowSpec)
    for arch, path in ARCHS.items():
        ModelRegistry.register_model(arch, path)
    _REGISTERED = True


def plugin() -> None:
    """vllm.general_plugins entry point. Always registers the BellowsKV-Flash architecture (it only adds a name,
    selected with --hf-overrides); BellowsKV-A (which rebinds LlamaForCausalLM) only with BELLOWSKV_VLLM=1."""
    try:
        from bellowskv.flash.vllm import register as register_flash
        register_flash()
    except Exception as e:  # noqa: BLE001  never break vLLM for other models
        import logging
        logging.getLogger("vllm.bellowskv").warning("BellowsKV-Flash plugin not registered: %r", e)
    if os.environ.get("BELLOWSKV_VLLM", "0") == "1":
        register()
