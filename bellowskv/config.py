"""BellowsKV-A settings. Defaults are the BellowsKV-A reference operating point (README, "Results")."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass

# Key-latent width per architecture, as a multiple of the per-layer kv width (num_key_value_heads * head_dim).
# Reference settings: Llama-3.1-8B R_K = 2560 (2.5 x 1024), Mistral-7B R_K = 2048 (2 x 1024).
R_K_MULT = {"llama": 2.5, "mistral": 2.0}
DEFAULT_R_K_MULT = 2.5


@dataclass
class BellowsConfig:
    group_size: int = 4            # n: layers sharing one latent
    r_k: int | None = None         # key latent width; None = R_K_MULT[model_type] * kv, rounded to 128
    r_v: int | None = None         # value latent width; None = kv
    bits: int = 4                  # latent bit-width (both sides)
    qgroup: int = 128              # channels per (scale, zero)
    clip: float = 0.95             # min-max range shrink before quantizing
    sink: int = 4                  # first tokens kept exact
    recent: int = 64               # most recent tokens kept exact
    # Generated tokens: the paper's evaluation keeps every generated token exact. For long generations that grows
    # without bound, so by default the oldest exact tokens beyond `recent` are compressed in blocks of this many.
    # None reproduces the paper's evaluation setting exactly.
    decode_block: int | None = 128
    # calibration (paper: 32 windows of 2048 tokens, half WikiText-2 train, half Python code)
    calib_windows: int = 32
    calib_seq: int = 2048
    calib_code_frac: float = 0.5
    ridge: float = 1e-3            # relative ridge on the joint Gram
    cache_dir: str = dataclasses.field(default_factory=lambda: os.environ.get("BELLOWSKV_CACHE", "~/.cache/bellowskv"))

    def resolve(self, model_config) -> "BellowsConfig":
        """Fill r_k / r_v from the model's shapes."""
        hd = getattr(model_config, "head_dim", None) or model_config.hidden_size // model_config.num_attention_heads
        kv = model_config.num_key_value_heads * hd
        c = dataclasses.replace(self)
        if c.r_v is None:
            c.r_v = kv
        if c.r_k is None:
            mult = R_K_MULT.get(getattr(model_config, "model_type", ""), DEFAULT_R_K_MULT)
            c.r_k = int(round(mult * kv / 128)) * 128
        if c.r_k % 128 or c.r_v % 256:
            raise ValueError(f"r_k must be a multiple of 128 and r_v of 256 (kernel tiling); got {c.r_k}, {c.r_v}")
        return c

    def fit_key(self) -> dict:
        """Fields that change the fitted factors (runtime-only fields excluded)."""
        return {k: getattr(self, k) for k in ("group_size", "r_k", "r_v", "calib_windows", "calib_seq",
                                               "calib_code_frac", "ridge")}

    def fit_hash(self, model_id: str) -> str:
        blob = json.dumps({"model": model_id, **self.fit_key(), "fit_version": 1}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def compression(self, model_config) -> float:
        """Accounted ratio of the compressed region: dense fp16 K/V of n layers over the two INT4 latents with
        their fp16 (scale, zero) per 128 channels."""
        hd = getattr(model_config, "head_dim", None) or model_config.hidden_size // model_config.num_attention_heads
        kv = model_config.num_key_value_heads * hd
        side = lambda r: r * self.bits + (r // self.qgroup) * 32
        return self.group_size * 2 * kv * 16 / (side(self.r_k) + side(self.r_v))
