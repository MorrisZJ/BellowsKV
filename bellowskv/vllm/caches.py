"""The two cache owners of a BellowsKV-A group, their (dummy) backends and metadata builders.

Per group of n consecutive decoder layers:
  * BellowsLatentCache  - FullAttentionSpec, 128-token blocks, one head slot, 2048 B per token: one struct-of-arrays
                          latent page per block (attn_ops.PageLayout, payload 1904 B/token, padded to 262,144 B/page);
  * BellowsWindowCache  - SinkWindowSpec, fp16 K|V of the group's n layers (layer p -> head slots p*Hkv .. p*Hkv+Hkv-1),
                          block size chosen so its page equals the latent page (16 tokens for Llama-3.1-8B), keeps
                          the sink block and the last W tokens.
Both are `AttentionLayerBase` side caches registered in `static_forward_context` (the QSA / Inkling / DSv4 pattern):
the runner allocates them, builds their metadata with the builders below, and binds the tensors; the per-layer op
(`bellowskv.vllm.op`) reads them from the forward context. Their backends have no impl.

All per-step metadata the decode path reads lives in persistent GPU buffers owned by the builders (CUDA-graph safe).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (AttentionBackend, AttentionCGSupport, AttentionMetadata,
                                       AttentionMetadataBuilder, CommonAttentionMetadata, MultipleOf)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from .attn_ops import PageLayout
from .specs import SinkWindowSpec

logger = init_logger("vllm.bellowskv")
_LOGGED: set = set()

LATENT_PAGE_TOKENS = 128
LATENT_TOKEN_BYTES = 2048          # 1904 B payload for Llama-3.1-8B, padded so the page is 262,144 B


def settings():
    """Runtime knobs (environment, read in every process): exact window W, sinks, group size."""
    return dict(window=int(os.environ.get("BELLOWSKV_W", "64")),
                sink=int(os.environ.get("BELLOWSKV_SINK", "4")),
                group=int(os.environ.get("BELLOWSKV_GROUP", "4")))


def page_layout(r_k: int, r_v: int, qgroup: int = 128) -> PageLayout:
    return PageLayout(R_K=r_k, R_V=r_v, page_tokens=LATENT_PAGE_TOKENS, qgroup=qgroup, BT=128,
                      page_bytes=LATENT_PAGE_TOKENS * LATENT_TOKEN_BYTES)


# per-device shared scratch (score workspace), sized at bind time from the number of latent blocks
_SHARED: dict = {}


def score_workspace(device, hq: int, capacity: int) -> torch.Tensor:
    key = ("ws", str(device), hq)
    ws = _SHARED.get(key)
    if ws is None or ws.shape[1] < capacity:
        ws = torch.empty(hq, capacity, device=device, dtype=torch.float32)
        _SHARED[key] = ws
    return ws


def _register(module: nn.Module, prefix: str, vllm_config: VllmConfig) -> None:
    ctx = vllm_config.compilation_config.static_forward_context
    if prefix in ctx:
        raise ValueError(f"Duplicate layer name: {prefix}")
    ctx[prefix] = module


# ----------------------------------------------------------------------------------------------- metadata
@dataclass
class BellowsLatentMetadata(AttentionMetadata):
    block_table: torch.Tensor            # (B, max_blocks) latent group, 128-token blocks
    slot_mapping: torch.Tensor           # (T,) latent slots of this step's tokens (-1 = skip)
    seq_lens: torch.Tensor               # (B,) incl. this step's tokens
    query_start_loc: torch.Tensor        # (B+1,)
    lat_len: torch.Tensor                # (B,) int32 latent tokens [sink, sink+lat_len) per request (persistent)
    lat_end: torch.Tensor                # (B,) int32 from paged_kernels.score_offsets (persistent)
    soff: torch.Tensor                   # (B+1,) int32 score-workspace offsets (persistent)
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    num_decodes: int
    num_decode_tokens: int
    query_start_loc_cpu: torch.Tensor    # (B+1,) CPU
    seq_lens_cpu: torch.Tensor | None    # (B,) CPU (exact for prefill rows)


@dataclass
class BellowsWindowMetadata(AttentionMetadata):
    block_table: torch.Tensor            # (B, max_blocks) window group, block_size tokens
    slot_mapping: torch.Tensor           # (T,) window slots of this step's tokens (-1 = skip)
    block_size: int
    win_slots: torch.Tensor              # (B, sink + W + 1) int64 decode exact-token slots incl. current (persistent)
    win_mask: torch.Tensor               # (B, sink + W + 1) bool (persistent)
    num_reqs: int
    num_decodes: int


def _cpu_lens(cam: CommonAttentionMetadata):
    seq_cpu = cam.seq_lens_cpu_upper_bound
    if seq_cpu is None:                                        # async spec decode only (unsupported anyway)
        seq_cpu = cam.seq_lens[:cam.num_reqs].cpu()
    return cam.query_start_loc_cpu[:cam.num_reqs + 1], seq_cpu[:cam.num_reqs]


class BellowsLatentBuilder(AttentionMetadataBuilder[BellowsLatentMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec: KVCacheSpec, layer_names: list[str], vllm_config: VllmConfig,
                 device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1)
        s = settings()
        self.sink, self.window = s["sink"], s["window"]
        n = vllm_config.scheduler_config.max_num_seqs
        self.lat_len = torch.zeros(n, dtype=torch.int32, device=device)
        self.lat_end = torch.zeros(n, dtype=torch.int32, device=device)
        self.soff = torch.zeros(n + 1, dtype=torch.int32, device=device)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> BellowsLatentMetadata:
        from .paged_kernels import score_offsets
        cam = common_attn_metadata
        B = cam.num_reqs
        nd, _, ndt, _ = split_decodes_and_prefills(cam, decode_threshold=1)
        qsl = cam.query_start_loc[:B + 1]
        ctx = cam.seq_lens[:B] - (qsl[1:] - qsl[:-1])                       # cached tokens before this step
        lat = (torch.clamp(ctx - self.window, min=self.sink) - self.sink).clamp_(min=0)
        self.lat_len[:B].copy_(lat)
        lat_end, soff = score_offsets(self.lat_len[:B], self.sink)
        self.lat_end[:B].copy_(lat_end)
        self.soff[:B + 1].copy_(soff)
        qsl_cpu, seq_cpu = _cpu_lens(cam)
        return BellowsLatentMetadata(
            block_table=cam.block_table_tensor, slot_mapping=cam.slot_mapping, seq_lens=cam.seq_lens,
            query_start_loc=cam.query_start_loc, lat_len=self.lat_len[:B], lat_end=self.lat_end[:B],
            soff=self.soff[:B + 1], num_reqs=B, num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len, num_decodes=nd, num_decode_tokens=ndt,
            query_start_loc_cpu=qsl_cpu, seq_lens_cpu=seq_cpu)

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata) -> BellowsLatentMetadata:
        common_attn_metadata.seq_lens.fill_(1)                 # cheap capture; real lengths arrive at replay
        return self.build(0, common_attn_metadata)


class BellowsWindowBuilder(AttentionMetadataBuilder[BellowsWindowMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec: KVCacheSpec, layer_names: list[str], vllm_config: VllmConfig,
                 device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1)
        s = settings()
        self.sink, self.window = s["sink"], s["window"]
        self.block_size = kv_cache_spec.block_size
        n = vllm_config.scheduler_config.max_num_seqs
        X = self.sink + self.window + 1
        self.win_slots = torch.zeros(n, X, dtype=torch.int64, device=device)
        self.win_mask = torch.zeros(n, X, dtype=torch.bool, device=device)
        self.col_sink = torch.arange(self.sink, device=device)
        self.col_rec = torch.arange(self.window + 1, device=device)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> BellowsWindowMetadata:
        cam = common_attn_metadata
        B = cam.num_reqs
        nd, _, _, _ = split_decodes_and_prefills(cam, decode_threshold=1)
        # decode rows: exact tokens = [0, min(s, L)) U [max(s, L-1-W), L), L = seq_len incl. the current token
        L = cam.seq_lens[:B].long()
        rs = torch.clamp(L - 1 - self.window, min=self.sink)
        pos = torch.cat([self.col_sink[None, :].expand(B, -1), rs[:, None] + self.col_rec[None, :]], 1)
        valid = torch.cat([self.col_sink[None, :] < L[:, None], (rs[:, None] + self.col_rec[None, :]) < L[:, None]], 1)
        bt = cam.block_table_tensor[:B]
        bidx = (pos // self.block_size).clamp_(max=bt.shape[1] - 1)
        slots = bt.gather(1, bidx).long() * self.block_size + pos % self.block_size
        slots = torch.where(valid, slots, torch.zeros_like(slots))
        valid[:, 0] = True                                     # padding rows: one (null-block) column -> no NaN
        self.win_slots[:B].copy_(slots)
        self.win_mask[:B].copy_(valid)
        return BellowsWindowMetadata(
            block_table=cam.block_table_tensor, slot_mapping=cam.slot_mapping, block_size=self.block_size,
            win_slots=self.win_slots[:B], win_mask=self.win_mask[:B], num_reqs=B, num_decodes=nd)

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata) -> BellowsWindowMetadata:
        common_attn_metadata.seq_lens.fill_(1)
        return self.build(0, common_attn_metadata)


# ----------------------------------------------------------------------------------------------- backends
class _NoImplBackend(AttentionBackend):
    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("BellowsKV side caches have no attention impl; the bellows_attention op runs them")

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class BellowsLatentBackend(_NoImplBackend):
    @staticmethod
    def get_name() -> str:
        return "BELLOWS_LATENT"

    @staticmethod
    def get_builder_cls():
        return BellowsLatentBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [LATENT_PAGE_TOKENS]


class BellowsWindowBackend(_NoImplBackend):
    @staticmethod
    def get_name() -> str:
        return "BELLOWS_WINDOW"

    @staticmethod
    def get_builder_cls():
        return BellowsWindowBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(8)]


# ----------------------------------------------------------------------------------------------- cache owners
class BellowsLatentCache(nn.Module, AttentionLayerBase):
    """Owner of one group's latent cache. `pages` = (num_blocks, page_bytes) uint8 view after binding."""

    supports_dcp = False

    def __init__(self, prefix: str, layout: PageLayout, num_q_heads: int, dtype: torch.dtype):
        super().__init__()
        vllm_config = get_current_vllm_config()
        self.prefix = prefix
        self.layout = layout
        self.num_q_heads = num_q_heads
        self.dtype = dtype
        self.kv_cache = torch.tensor([])
        self.pages = None
        self.ws = None
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        assert layout.bytes == LATENT_PAGE_TOKENS * LATENT_TOKEN_BYTES, layout
        _register(self, prefix, vllm_config)

    def forward(self):  # pragma: no cover - never called
        ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return BellowsLatentBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(block_size=LATENT_PAGE_TOKENS, num_kv_heads=1, head_size=128, head_size_v=128,
                                 dtype=self.dtype, num_head_slots=1, state_content_bytes=LATENT_TOKEN_BYTES)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache
        if kv_cache.numel() == 0:
            self.pages = None
            return
        u8 = kv_cache.view(torch.uint8)                       # (num_blocks, 1, 128, 2048)
        nb = u8.shape[0]
        assert u8.shape[1:] == (1, LATENT_PAGE_TOKENS, LATENT_TOKEN_BYTES), u8.shape
        assert u8.stride(3) == 1 and u8.stride(2) == LATENT_TOKEN_BYTES, u8.stride()   # page bytes contiguous
        self.pages = torch.as_strided(u8, (nb, self.layout.bytes), (u8.stride(0), 1), u8.storage_offset())
        assert self.pages.stride(0) % 2 == 0
        cap = nb * LATENT_PAGE_TOKENS + 64 * self.max_num_seqs
        self.ws = score_workspace(kv_cache.device, self.num_q_heads, cap)
        if "lat" not in _LOGGED:
            _LOGGED.add("lat")
            logger.info("[bellowskv] latent cache %s: view %s strides %s, %d blocks x %d tokens, page %d B "
                        "(payload %d B), score workspace %.0f MiB", self.prefix, tuple(kv_cache.shape),
                        tuple(u8.stride()), nb, LATENT_PAGE_TOKENS, self.layout.bytes, self.layout.payload_bytes,
                        self.ws.numel() * 4 / 2**20)


class BellowsWindowCache(nn.Module, AttentionLayerBase):
    """Owner of one group's exact windows: fp16 [K|V] for the group's n layers (n*Hkv head slots)."""

    supports_dcp = False

    def __init__(self, prefix: str, num_kv_heads: int, head_size: int, group: int, window: int, sink: int,
                 dtype: torch.dtype):
        super().__init__()
        vllm_config = get_current_vllm_config()
        self.prefix = prefix
        self.num_kv_heads, self.head_size, self.group = num_kv_heads, head_size, group
        self.window, self.sink, self.dtype = window, sink, dtype
        token_bytes = group * num_kv_heads * 2 * head_size * torch.finfo(dtype).bits // 8
        page = LATENT_PAGE_TOKENS * LATENT_TOKEN_BYTES
        assert page % token_bytes == 0, (page, token_bytes)
        self.block_size = page // token_bytes                 # 16 for Llama-3.1-8B (equal pages, 2 groups)
        self.kv_cache = torch.tensor([])
        _register(self, prefix, vllm_config)

    def forward(self):  # pragma: no cover
        ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return BellowsWindowBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SinkWindowSpec(block_size=self.block_size, num_kv_heads=self.num_kv_heads, head_size=self.head_size,
                              head_size_v=self.head_size, dtype=self.dtype,
                              num_head_slots=self.group * self.num_kv_heads,
                              sliding_window=self.window + 1, sink_tokens=self.sink)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # (num_blocks, n*Hkv, block_size, 2*hd) logical [B, H, N, C]
        self.kv_cache = kv_cache
        if kv_cache.numel() and "win" not in _LOGGED:
            _LOGGED.add("win")
            logger.info("[bellowskv] window cache %s: view %s strides %s dtype %s, block %d tokens, W=%d sink=%d",
                        self.prefix, tuple(kv_cache.shape), tuple(kv_cache.stride()), kv_cache.dtype,
                        self.block_size, self.window, self.sink)
