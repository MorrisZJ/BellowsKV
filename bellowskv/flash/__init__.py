"""BellowsKV-Flash: a FlashMLA-native shared record per block of n = 2 layers (512 latent + 64 RoPE dims, fp8).

Post-training conversion: the base model stays frozen; the record maps are initialized in closed form and fitted
by block-wise distillation on calibration text (WikiText-2 train + Python code), about 1.7 H100 hours for an 8B
model. Decode then reads one fp8 record per block through FlashMLA, plus a small exact window (4 sink + 123 recent).

    python -m bellowskv.flash.convert --model meta-llama/Llama-3.1-8B-Instruct --out flash-llama31-8b --gpus 0,1,2,3
"""
from .config import FlashConfig
from . import params


def enable(model, params, **kwargs):
    """`bellowskv.flash.hf.enable`: BellowsKV-Flash attention for a HuggingFace Llama model (see its docstring)."""
    from .hf import enable as _enable
    return _enable(model, params, **kwargs)


def disable(model):
    from .hf import disable as _disable
    return _disable(model)


__all__ = ["FlashConfig", "params", "enable", "disable"]
