"""KV-cache spec + manager for the exact window: a sliding window that never frees the request's first block(s).

`SinkWindowSpec` is a `SlidingWindowSpec` (same grouping, same block recycling) that additionally keeps the blocks
holding the first `sink_tokens` positions for the whole life of the request. Registered with vLLM's
`KVCacheSpecRegistry` (see `bellowskv.vllm.register`). Prefix caching is not supported (the hit logic would also have
to find the sink blocks), so the spec reports itself as not prefix-cacheable and the model refuses to start with
prefix caching on.
"""
from __future__ import annotations

from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec


@dataclass(frozen=True, kw_only=True)
class SinkWindowSpec(SlidingWindowSpec):
    """Exact window of one BellowsKV group: positions [0, sink) plus the last `sliding_window - 1` computed tokens."""

    sink_tokens: int = 4

    def max_admission_blocks_per_request(self, max_in_flight_tokens: int, max_model_len: int) -> int:
        return (super().max_admission_blocks_per_request(max_in_flight_tokens, max_model_len)
                + cdiv(self.sink_tokens, self.block_size))

    @property
    def prefix_cacheable(self) -> bool:
        return False


class SinkWindowManager(SlidingWindowManager):
    """SlidingWindowManager that frees [n_sink_blocks, skipped) instead of [0, skipped)."""

    def __init__(self, kv_cache_spec: SinkWindowSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.num_sink_blocks = cdiv(kv_cache_spec.sink_tokens, self.block_size)

    def remove_skipped_blocks(self, request_id: str, processed_computed_tokens: int,
                              num_prompt_tokens: int | None = None) -> None:
        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
        if num_skipped_tokens <= 0:
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = min(num_skipped_tokens // self.block_size, len(blocks))
        self._remove_blocks_in_range(request_id, self.num_sink_blocks, num_skipped_blocks)
