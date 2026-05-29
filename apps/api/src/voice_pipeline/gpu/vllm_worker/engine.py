from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VLLMEngineConfig:
    model_name: str = ""
    model_path: str = ""
    cache_dir: str = ""
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    enable_streaming: bool = True
    max_num_seqs: int = 1
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = True
    max_num_batched_tokens: int = 128
    max_model_len: int = 2048
    max_inflight_tokens_per_request: int = 256
    max_dispatch_queue_depth: int = 8
    max_backpressure_propagation_delay_ms: float = 10.0
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 128


@dataclass(frozen=True, slots=True)
class PrefixCacheStats:
    hits: int
    misses: int

    @property
    def hit_ratio(self) -> float:
        total = int(self.hits) + int(self.misses)
        if total <= 0:
            return 0.0
        return float(self.hits) / float(total)


def build_prompt_cache_key(*, system_prompt: str, context_prefix: str, stable_prefix: str) -> str:
    parts = [
        " ".join(str(system_prompt or "").strip().split()),
        " ".join(str(context_prefix or "").strip().split()),
        " ".join(str(stable_prefix or "").strip().split()),
    ]
    return "|".join(parts)


class VLLMEngine:
    """GPU0 streaming token runtime backed by vLLM when available."""

    def __init__(self, model_name: str, *, config: VLLMEngineConfig | None = None) -> None:
        self.model_name = str(model_name or "").strip()
        self.config = config or VLLMEngineConfig(model_name=self.model_name)
        self._prefix_cache: set[str] = set()
        self._prefix_cache_hits = 0
        self._prefix_cache_misses = 0
        self._dispatch_queue_depth = 0
        self._engine: Any | None = None
        self._sampling_params_cls: Any | None = None
        self._warm = False

    @property
    def is_warm(self) -> bool:
        return bool(self._warm)

    @property
    def prefix_cache_ready(self) -> bool:
        return bool(self._warm and self.config.enable_prefix_caching and len(self._prefix_cache) > 0)

    def prewarm_prefix_cache(self, *cache_keys: str) -> None:
        if not self.config.enable_prefix_caching:
            return
        for cache_key in cache_keys:
            normalized = str(cache_key or "").strip()
            if normalized:
                self._prefix_cache.add(normalized)

    def warm(self, *, strict: bool | None = None) -> None:
        strict_mode = True if strict is None else bool(strict)
        if not strict_mode:
            raise RuntimeError("vllm_strict_mode_required")
        resolved_model = str(self.config.model_path or self.config.model_name or self.model_name).strip()
        if not resolved_model:
            raise RuntimeError("vllm model name is required for strict warm start")
        if not Path(resolved_model).exists():
            raise RuntimeError(f"vllm model path does not exist: {resolved_model}")

        try:
            from vllm import AsyncLLMEngine, SamplingParams  # type: ignore
            try:
                from vllm import AsyncEngineArgs  # type: ignore
            except Exception:  # pragma: no cover - compatibility path
                from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("vllm runtime is unavailable") from exc

        engine_kwargs: dict[str, object] = {
            "model": resolved_model,
            "trust_remote_code": True,
            "gpu_memory_utilization": float(self.config.gpu_memory_utilization),
            "max_model_len": int(self.config.max_model_len),
            "max_num_seqs": int(self.config.max_num_seqs),
            "max_num_batched_tokens": int(self.config.max_num_batched_tokens),
            "enforce_eager": bool(self.config.enforce_eager),
            "enable_prefix_caching": bool(self.config.enable_prefix_caching),
            "enable_chunked_prefill": bool(self.config.enable_chunked_prefill),
        }
        if str(self.config.cache_dir).strip():
            engine_kwargs["download_dir"] = str(self.config.cache_dir).strip()
        try:
            engine_args = AsyncEngineArgs(**engine_kwargs)
        except TypeError:
            engine_kwargs.pop("enable_chunked_prefill", None)
            engine_args = AsyncEngineArgs(**engine_kwargs)
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._sampling_params_cls = SamplingParams
        self._warm = True

    def cache_stats(self) -> PrefixCacheStats:
        return PrefixCacheStats(hits=self._prefix_cache_hits, misses=self._prefix_cache_misses)

    async def cancel_request(self, request_id: str) -> None:
        resolved_request_id = str(request_id or "").strip()
        if not resolved_request_id:
            return
        if self._engine is None:
            raise RuntimeError("vllm_streaming_engine_unavailable")
        abort = getattr(self._engine, "abort", None)
        if not callable(abort):
            raise RuntimeError("vllm_abort_unavailable")
        outcome = abort(resolved_request_id)
        if inspect.isawaitable(outcome):
            await outcome

    async def stream_tokens(
        self,
        prompt: str,
        *,
        cache_key: str = "",
        request_id: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        if self._dispatch_queue_depth >= int(self.config.max_dispatch_queue_depth):
            raise RuntimeError("vllm_dispatch_queue_overflow")
        resolved_request_id = str(request_id or "").strip()
        if not resolved_request_id:
            raise RuntimeError("vllm_request_id_required")
        self._dispatch_queue_depth += 1
        try:
            resolved_cache_key = str(cache_key or prompt).strip()
            if self.config.enable_prefix_caching and resolved_cache_key:
                if resolved_cache_key in self._prefix_cache:
                    self._prefix_cache_hits += 1
                else:
                    self._prefix_cache_misses += 1
                    self._prefix_cache.add(resolved_cache_key)

            if self._engine is None or self._sampling_params_cls is None:
                raise RuntimeError("vllm_streaming_engine_unavailable")

            sampling_params = self._sampling_params_cls(
                temperature=float(self.config.temperature if temperature is None else temperature),
                top_p=float(self.config.top_p if top_p is None else top_p),
                max_tokens=int(self.config.max_tokens if max_tokens is None else max_tokens),
            )
            previous_text = ""
            async for request_output in self._engine.generate(str(prompt), sampling_params, request_id=resolved_request_id):
                text = str(request_output.outputs[0].text)
                if not text:
                    continue
                if text.startswith(previous_text):
                    delta = text[len(previous_text) :]
                else:
                    delta = text
                previous_text = text
                if delta:
                    yield delta
        finally:
            self._dispatch_queue_depth = max(0, self._dispatch_queue_depth - 1)


__all__ = [
    "PrefixCacheStats",
    "VLLMEngine",
    "VLLMEngineConfig",
    "build_prompt_cache_key",
]
