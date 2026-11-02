"""Llama with BellowsKV-A attention for vLLM 0.29.

`BellowsLlamaAttention` keeps LlamaAttention's projections and rotary table (so a stock checkpoint loads unchanged)
and replaces its `Attention` module by the `bellows_attention` op plus, at each group's first layer, the group's two
cache owners (bellowskv.vllm.caches). The codec factors (down-projections Wd, read-heads WuK/WuV, band rotation)
are loaded into non-persistent buffers by `load_weights`, from the same cache the HF package uses; if no factors
exist for the model they are fitted first with the HF package (one calibration pass), before vLLM profiles memory.
"""
from __future__ import annotations

import gc
import os
import re

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM
from vllm.model_executor.models.utils import extract_layer_index

from ..config import BellowsConfig
from ..kernels import DEFAULT_CFG, permute_index
from .caches import BellowsLatentCache, BellowsWindowCache, page_layout, settings
from .op import register_op

logger = init_logger("vllm.bellowskv")
register_op()

_SHARED: dict = {}          # per-device shared buffers (latent accumulator)


def bellows_config(hf_config) -> BellowsConfig:
    s = settings()
    return BellowsConfig(group_size=s["group"], sink=s["sink"], recent=s["window"], decode_block=1).resolve(hf_config)


def factors_path(model_id: str, cfg: BellowsConfig) -> str:
    explicit = os.environ.get("BELLOWSKV_FACTORS")
    if explicit:
        return explicit
    cdir = os.path.expanduser(cfg.cache_dir)
    return os.path.join(cdir, f"{model_id.strip('/').replace('/', '--')}.A.{cfg.fit_hash(model_id)}.safetensors")


class BellowsAttnMixin:
    """Turns a LlamaAttention-like module into a BellowsKV-A layer. Call `_bellows_init(prefix)` after the base
    __init__ (which built qkv_proj / o_proj / rotary_emb and an `Attention` we remove)."""

    def _bellows_init(self, prefix: str) -> None:
        vllm_config = get_current_vllm_config()
        hf = vllm_config.model_config.hf_config
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("BellowsKV-A v1 on vLLM supports tensor_parallel_size=1 only")
        cfg = bellows_config(hf)
        s = settings()
        n, L = cfg.group_size, hf.num_hidden_layers
        l = extract_layer_index(prefix)
        if (l // n + 1) * n > L:
            raise NotImplementedError(f"layer {l} is outside a full group of {n} (num_hidden_layers={L})")
        ctx = vllm_config.compilation_config.static_forward_context
        ctx.pop(f"{prefix}.attn", None)                         # drop the stock Attention built by the base class
        del self.attn
        self.layer_name = f"{prefix}.attn"
        ctx[self.layer_name] = self                             # op lookup only (not an AttentionLayerBase)

        g, p = l // n, l % n
        self.bkv_group, self.bkv_pos, self.bkv_layer = n, p, l
        self.bkv_sink, self.bkv_window_len = s["sink"], s["window"]
        self.R_K, self.R_V, self.qgroup = cfg.r_k, cfg.r_v, cfg.qgroup
        dt = vllm_config.model_config.dtype
        if dt != torch.float16:
            raise NotImplementedError("BellowsKV-A runs attention in fp16: start vLLM with dtype=float16")
        layout = page_layout(cfg.r_k, cfg.r_v, cfg.qgroup)
        first = re.sub(rf"\.{l}\.", f".{g * n}.", prefix, count=1)
        if p == 0:
            self.bellows_latent = BellowsLatentCache(f"{prefix}.bellows_latent", layout, self.num_heads, dt)
            self.bellows_window = BellowsWindowCache(f"{prefix}.bellows_window", self.num_kv_heads, self.head_dim,
                                                     n, s["window"], s["sink"], dt)
            lat, win = self.bellows_latent, self.bellows_window
        else:
            lat, win = ctx[f"{first}.bellows_latent"], ctx[f"{first}.bellows_window"]
        object.__setattr__(self, "bkv_latent", lat)             # plain references (not submodules)
        object.__setattr__(self, "bkv_window", win)

        d, Hkv, hd = self.hidden_size, self.num_kv_heads, self.head_dim
        self.register_buffer("Wd", torch.zeros(d, cfg.r_k + cfg.r_v, dtype=dt), persistent=False)
        self.register_buffer("WuK_p", torch.zeros(Hkv, hd, cfg.r_k, dtype=dt), persistent=False)
        self.register_buffer("WuV_p", torch.zeros(Hkv, cfg.r_v, hd, dtype=dt), persistent=False)
        self.register_buffer("idxK", permute_index(cfg.r_k, cfg.qgroup), persistent=False)
        self.register_buffer("idxV", permute_index(cfg.r_v, DEFAULT_CFG["BKV"]), persistent=False)
        self.register_buffer("bkv_heads", torch.arange(p * Hkv, (p + 1) * Hkv), persistent=False)
        self.register_buffer("rot_band", torch.zeros(cfg.r_k, dtype=torch.long), persistent=False)
        self.register_buffer("rot_is_re", torch.zeros(cfg.r_k, dtype=torch.bool), persistent=False)
        self.register_buffer("rot_partner", torch.zeros(cfg.r_k, dtype=torch.long), persistent=False)
        max_tokens = max(vllm_config.scheduler_config.max_num_batched_tokens,
                         vllm_config.compilation_config.max_cudagraph_capture_size or 0)
        key = ("acc", str(self.Wd.device))
        acc = _SHARED.get(key)
        if acc is None or acc.shape[0] < max_tokens:
            acc = torch.zeros(max_tokens, cfg.r_k + cfg.r_v, dtype=torch.float32, device=self.Wd.device)
            _SHARED[key] = acc
        object.__setattr__(self, "bkv_acc", acc)
        self.bkv_loaded = False

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        pkv = torch.mm(hidden_states, self.Wd)                  # this layer's latent share (fp16 GEMM)
        out = torch.empty(q.shape[0], self.num_heads * self.head_dim, dtype=q.dtype, device=q.device)
        torch.ops.vllm.bellows_attention(q.contiguous(), k.contiguous(), v.contiguous(), pkv, positions, out,
                                         self.layer_name)
        output, _ = self.o_proj(out)
        return output


class BellowsLlamaAttention(BellowsAttnMixin, LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prefix = kwargs.get("prefix")
        if prefix is None:
            raise ValueError("BellowsLlamaAttention needs prefix=")
        self._bellows_init(prefix)


class BellowsLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "", config=None, attn_layer_type=None):
        super().__init__(vllm_config=vllm_config, prefix=prefix, config=config,
                         attn_layer_type=BellowsLlamaAttention)


# ----------------------------------------------------------------------------------------------- factors
def _fit_factors(model_id: str, cfg: BellowsConfig, path: str) -> None:
    """Fit BellowsKV-A factors with the HF package (one calibration pass on this GPU) and save them to `path`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..api import fit
    from ..fit import save_factors
    logger.info("[bellowskv] no factors at %s: fitting with the HF package (one calibration pass)", path)
    tok = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16,
                                                    attn_implementation="sdpa").cuda().eval()
    with torch.no_grad():
        factors, meta = fit(hf_model, tok, cfg, log=logger.info)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_factors(path, factors, meta)
    del hf_model, factors
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def load_bellows_factors(model: nn.Module, model_id: str) -> str:
    from ..codec import BandRotation
    from ..fit import load_factors
    from ..kernels import prepare_heads
    hf = model.config
    cfg = bellows_config(hf)
    path = factors_path(model_id, cfg)
    if not os.path.exists(path):
        _fit_factors(model_id, cfg, path)
    factors, meta = load_factors(path)
    mcfg = meta.get("config", {})
    for key in ("group_size", "r_k", "r_v"):
        if key in mcfg and mcfg[key] != getattr(cfg, key):
            raise ValueError(f"factors at {path} were fitted with {key}={mcfg[key]}, runtime wants {getattr(cfg, key)}")
    n = cfg.group_size
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if not isinstance(attn, BellowsAttnMixin):
            continue
        l, p = attn.bkv_layer, attn.bkv_pos
        f = factors[(l // n) * n]
        d, Hkv, hd = attn.hidden_size, attn.num_kv_heads, attn.head_dim
        kv = Hkv * hd
        dev, dt = attn.Wd.device, attn.Wd.dtype
        wd = torch.cat([f["WdK"][p * d:(p + 1) * d], f["WdV"][p * d:(p + 1) * d]], 1)
        attn.Wd.copy_(wd.to(dev, dt))
        wk = f["WuK"][:, p * kv:(p + 1) * kv].reshape(cfg.r_k, Hkv, hd).permute(1, 2, 0)
        wv = f["WuV"][:, p * kv:(p + 1) * kv].reshape(cfg.r_v, Hkv, hd).permute(1, 0, 2)
        wk_p, wv_p = prepare_heads(wk.to(dev, dt).contiguous(), wv.to(dev, dt).contiguous(), DEFAULT_CFG, cfg.qgroup)
        attn.WuK_p.copy_(wk_p)
        attn.WuV_p.copy_(wv_p)
        rot = BandRotation(f["ranks"].tolist(), dev)
        attn.rot_band.copy_(rot.band)
        attn.rot_is_re.copy_(rot.is_re)
        attn.rot_partner.copy_(rot.partner)
        attn.bkv_loaded = True
    logger.info("[bellowskv] BellowsKV-A factors loaded from %s (R_K=%d, R_V=%d, group %d, W=%d, sink %d)",
                path, cfg.r_k, cfg.r_v, n, cfg.recent, cfg.sink)
    return path


def _check_config(vllm_config: VllmConfig) -> None:
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("BellowsKV-A v1 needs enable_prefix_caching=False")
    if vllm_config.speculative_config is not None:
        raise NotImplementedError("BellowsKV-A v1 does not support speculative decoding")
    if vllm_config.kv_transfer_config is not None:
        raise NotImplementedError("BellowsKV-A v1 does not support KV connectors")


class BellowsLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", layer_type=None):
        _check_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=BellowsLlamaDecoderLayer)
        self._bkv_model_id = vllm_config.model_config.model

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        load_bellows_factors(self, self._bkv_model_id)
        return loaded
