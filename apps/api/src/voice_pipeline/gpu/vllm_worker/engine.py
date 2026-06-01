from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
from pathlib import Path
import re
import sys
from typing import Any


@dataclass(frozen=True, slots=True)
class VLLMEngineConfig:
    model_name: str = ""
    model_path: str = ""
    cache_dir: str = ""
    required_device: str = "cuda:0"
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    enable_streaming: bool = True
    max_num_seqs: int = 1
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = True
    compilation_config_mode: int = 0
    max_num_batched_tokens: int = 128
    max_model_len: int = 2048
    offload_backend: str = "auto"
    cpu_offload_gb: float = 0.0
    kv_offloading_size: float = 0.0
    kv_offloading_backend: str = "native"
    kv_cache_dtype: str = "auto"
    kv_cache_memory_bytes: int = 0
    num_gpu_blocks_override: int = 0
    attention_backend: str = "auto"
    safetensors_load_strategy: str = "prefetch"
    max_inflight_tokens_per_request: int = 256
    max_dispatch_queue_depth: int = 8
    max_backpressure_propagation_delay_ms: float = 10.0
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 128
    system_prompt: str = (
        "You are the low-latency voice assistant for a real-time conversation. "
        "Answer directly, naturally, and briefly unless the user explicitly asks for detail. "
        "Output exactly one short spoken reply in plain conversational text only. "
        "Do not add notes, explanations, markdown, repetition, bullets, role tags, "
        "stage directions, or follow-up formatting. Never say 'Note' or append a second answer."
    )


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
        self._tokenizer: Any | None = None
        self._tokenizer_load_attempted = False
        self._warm = False

    @property
    def is_warm(self) -> bool:
        return bool(self._warm)

    @property
    def prefix_cache_ready(self) -> bool:
        return bool(self._warm and self.config.enable_prefix_caching and len(self._prefix_cache) > 0)

    def shutdown(self, *, timeout: float | None = None) -> None:
        engine = self._engine
        self._engine = None
        self._sampling_params_cls = None
        self._tokenizer = None
        self._tokenizer_load_attempted = False
        self._prefix_cache.clear()
        self._prefix_cache_hits = 0
        self._prefix_cache_misses = 0
        self._dispatch_queue_depth = 0
        self._warm = False
        if engine is not None:
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown(timeout=timeout)
        try:
            import torch  # type: ignore
        except Exception:
            return
        if not torch.cuda.is_available():
            return
        try:
            with _cuda_device_context(str(self.config.required_device or "cuda:0")):
                torch.cuda.empty_cache()
        except Exception:
            # Shutdown is best-effort; keep the original caller outcome.
            pass

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
        if not bool(self.config.enable_prefix_caching):
            raise RuntimeError("vllm_prefix_caching_required")
        resolved_model = str(self.config.model_path or self.config.model_name or self.model_name).strip()
        if not resolved_model:
            raise RuntimeError("vllm model name is required for strict warm start")
        if resolved_model.startswith("http://") or resolved_model.startswith("https://"):
            raise RuntimeError(f"vllm model path must be local/offline, got {resolved_model}")
        model_path = Path(resolved_model)
        if not model_path.exists():
            raise RuntimeError(f"vllm model path does not exist: {resolved_model}")
        if not model_path.is_dir():
            raise RuntimeError(f"vllm model path is not a directory: {resolved_model}")
        resolved_cache_dir = str(self.config.cache_dir).strip()
        if not resolved_cache_dir:
            raise RuntimeError("vllm cache_dir is required for strict warm start")
        if resolved_cache_dir.startswith("http://") or resolved_cache_dir.startswith("https://"):
            raise RuntimeError(f"vllm cache_dir must be local/offline, got {resolved_cache_dir}")
        cache_dir = Path(resolved_cache_dir)
        if not cache_dir.exists() or not cache_dir.is_dir():
            raise RuntimeError(f"vllm cache_dir does not exist or is not a directory: {resolved_cache_dir}")
        _assert_cuda_device_binding(str(self.config.required_device or "cuda:0"))
        try:
            from vllm import AsyncLLMEngine, SamplingParams  # type: ignore
            try:
                from vllm import AsyncEngineArgs  # type: ignore
            except Exception:  # pragma: no cover - compatibility path
                from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(f"vllm runtime is unavailable: {exc}") from exc

        engine_kwargs: dict[str, object] = {
            "model": resolved_model,
            "trust_remote_code": True,
            "device": str(self.config.required_device or "cuda:0"),
            "gpu_memory_utilization": float(self.config.gpu_memory_utilization),
            "compilation_config": {"mode": int(self.config.compilation_config_mode)},
            "max_model_len": int(self.config.max_model_len),
            "max_num_seqs": int(self.config.max_num_seqs),
            "max_num_batched_tokens": int(self.config.max_num_batched_tokens),
            "offload_backend": str(self.config.offload_backend or "auto"),
            "cpu_offload_gb": float(self.config.cpu_offload_gb),
            "kv_offloading_size": (
                float(self.config.kv_offloading_size)
                if float(self.config.kv_offloading_size) > 0
                else None
            ),
            "kv_offloading_backend": str(self.config.kv_offloading_backend or "native"),
            "kv_cache_dtype": str(self.config.kv_cache_dtype or "auto"),
            "kv_cache_memory_bytes": (
                int(self.config.kv_cache_memory_bytes)
                if int(self.config.kv_cache_memory_bytes) > 0
                else None
            ),
            "num_gpu_blocks_override": (
                int(self.config.num_gpu_blocks_override)
                if int(self.config.num_gpu_blocks_override) > 0
                else None
            ),
            "safetensors_load_strategy": str(self.config.safetensors_load_strategy or "").strip() or None,
            "attention_backend": (
                None
                if str(self.config.attention_backend or "auto").strip().lower() == "auto"
                else str(self.config.attention_backend).strip()
            ),
            "enforce_eager": bool(self.config.enforce_eager),
            "enable_prefix_caching": bool(self.config.enable_prefix_caching),
            "enable_chunked_prefill": bool(self.config.enable_chunked_prefill),
            "worker_use_ray": False,
            "engine_use_ray": False,
            "distributed_executor_backend": "uni",
        }
        engine_kwargs["download_dir"] = resolved_cache_dir
        try:
            supported_engine_args = set(inspect.signature(AsyncEngineArgs).parameters)
        except (TypeError, ValueError):
            supported_engine_args = set(engine_kwargs)
        filtered_engine_kwargs = {
            key: value for key, value in engine_kwargs.items() if key in supported_engine_args
        }
        try:
            engine_args = AsyncEngineArgs(**filtered_engine_kwargs)
        except TypeError:
            filtered_engine_kwargs.pop("enable_chunked_prefill", None)
            filtered_engine_kwargs.pop("distributed_executor_backend", None)
            filtered_engine_kwargs.pop("device", None)
            engine_args = AsyncEngineArgs(**filtered_engine_kwargs)
        print(
            (
                "VLLM_WARM_MARKER before_from_engine_args "
                f"model={resolved_model} device={str(self.config.required_device or 'cuda:0')} "
                f"max_model_len={int(self.config.max_model_len)} "
                f"max_num_batched_tokens={int(self.config.max_num_batched_tokens)}"
            ),
            file=sys.stderr,
            flush=True,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("VLLM_WARM_MARKER after_from_engine_args", file=sys.stderr, flush=True)
        self._sampling_params_cls = SamplingParams
        self._warm = True
        print("VLLM_WARM_MARKER warm_complete", file=sys.stderr, flush=True)

    def _maybe_load_tokenizer(self) -> Any | None:
        if self._tokenizer_load_attempted:
            return self._tokenizer
        self._tokenizer_load_attempted = True
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception:
            return None
        resolved_model = str(self.config.model_path or self.config.model_name or self.model_name).strip()
        if not resolved_model:
            return None
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                resolved_model,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            self._tokenizer = None
        return self._tokenizer

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

    def render_prompt(
        self,
        *,
        user_text: str,
        stable_session_summary: str = "",
    ) -> str:
        normalized_user_text = " ".join(str(user_text or "").strip().split())
        if "/no_think" not in normalized_user_text:
            normalized_user_text = f"{normalized_user_text} /no_think"
        if not normalized_user_text:
            raise RuntimeError("vllm_user_text_required")
        normalized_summary = " ".join(str(stable_session_summary or "").strip().split())
        max_model_len = int(self.config.max_model_len)
        if max_model_len > 0 and max_model_len <= 24:
            # The constrained live profile uses a tiny KV budget to keep GPU0
            # alive. In that mode the chat template overhead is larger than the
            # real spoken turn itself, so use the smallest prompt shape for the
            # live request path instead of faulting the turn.
            return self.render_minimal_probe_prompt(user_text=normalized_user_text)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": str(self.config.system_prompt).strip()},
        ]
        if normalized_summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Stable session summary: {normalized_summary}",
                }
            )
        messages.append({"role": "user", "content": normalized_user_text})
        tokenizer = self._tokenizer or self._maybe_load_tokenizer()

        def _render_messages(render_messages: list[dict[str, str]]) -> str:
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                return str(
                    tokenizer.apply_chat_template(
                        render_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            rendered = []
            for message in render_messages:
                rendered.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
            rendered.append("<|im_start|>assistant\n")
            return "\n".join(rendered)

        prompt = _render_messages(messages)
        prompt_token_count = self.prompt_token_count(prompt)
        if prompt_token_count is None or max_model_len <= 0 or prompt_token_count < max_model_len:
            return prompt

        # Under very small live-memory profiles, the chat template wrapper can
        # exceed the token budget even for tiny turns like "hello there". Fall
        # back to progressively smaller low-latency prompt shapes rather than
        # faulting the real request.
        compact_messages = [
            {"role": "system", "content": "Reply briefly."},
            {"role": "user", "content": normalized_user_text},
        ]
        compact_prompt = _render_messages(compact_messages)
        compact_token_count = self.prompt_token_count(compact_prompt)
        if compact_token_count is None or compact_token_count < max_model_len:
            return compact_prompt
        return self.render_minimal_probe_prompt(user_text=normalized_user_text)

    def render_minimal_probe_prompt(self, *, user_text: str) -> str:
        normalized_user_text = " ".join(str(user_text or "").strip().split())
        if "/no_think" not in normalized_user_text:
            normalized_user_text = f"{normalized_user_text} /no_think"
        if not normalized_user_text:
            raise RuntimeError("vllm_user_text_required")
        # Keep the constrained live profile under the tiny token budget while
        # making it explicit that the text is quoted user speech that needs an
        # assistant reply, not a continuation seed for the model to extend.
        minimal_user_text = f"User said: {normalized_user_text}. Assistant reply only."
        return (
            "<|im_start|>user\n"
            f"{minimal_user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def prompt_token_count(self, prompt: str) -> int | None:
        tokenizer = self._tokenizer or self._maybe_load_tokenizer()
        if tokenizer is None:
            return None
        try:
            token_ids = tokenizer.encode(str(prompt), add_special_tokens=False)
        except Exception:
            return None
        return len(token_ids)

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
                stop=["**", "<|im_end|>", "? Note", ". Note", "! Note", "? Note:", ". Note:", "! Note:"],
            )
            previous_text = ""
            previous_spoken_text = ""
            pending_spoken_delta = ""
            generator = self._engine.generate(str(prompt), sampling_params, request_id=resolved_request_id)
            async for request_output in generator:
                text = str(request_output.outputs[0].text)
                if not text:
                    continue
                previous_text = text
                spoken_text = _normalize_spoken_output(_strip_reasoning_sections(text))
                if spoken_text.startswith(previous_spoken_text):
                    delta = spoken_text[len(previous_spoken_text) :]
                else:
                    delta = spoken_text
                previous_spoken_text = spoken_text
                if not delta:
                    continue
                pending_spoken_delta = _append_spoken_delta(pending_spoken_delta, delta)
                flush_text, pending_spoken_delta = _split_flushable_spoken_prefix(pending_spoken_delta)
                if flush_text:
                    yield flush_text
            final_flush = _normalize_spoken_output(pending_spoken_delta)
            if final_flush:
                yield final_flush
        finally:
            self._dispatch_queue_depth = max(0, self._dispatch_queue_depth - 1)


__all__ = [
    "PrefixCacheStats",
    "VLLMEngine",
    "VLLMEngineConfig",
    "build_prompt_cache_key",
]


def _assert_cuda_device_binding(required_device: str) -> None:
    resolved_required = str(required_device or "").strip().lower()
    if not resolved_required.startswith("cuda:"):
        raise RuntimeError(f"vllm required device must be cuda:N, got {required_device}")
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("torch unavailable for vllm device validation") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("cuda unavailable for vllm")
    expected_index = int(resolved_required.split(":", 1)[1])
    if expected_index >= int(torch.cuda.device_count()):
        raise RuntimeError(f"vllm required device missing: {required_device}")
    with _cuda_device_context(resolved_required):
        current_index = int(torch.cuda.current_device())
        if current_index != expected_index:
            raise RuntimeError(
                f"vllm device binding mismatch: expected {resolved_required}, current cuda:{current_index}"
            )


def _strip_reasoning_sections(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    output: list[str] = []
    in_thinking = False
    index = 0
    open_tag = "<think>"
    close_tag = "</think>"
    while index < len(raw):
        if raw.startswith(open_tag, index):
            in_thinking = True
            index += len(open_tag)
            continue
        if raw.startswith(close_tag, index):
            in_thinking = False
            index += len(close_tag)
            continue
        remaining = raw[index:]
        if open_tag.startswith(remaining) or close_tag.startswith(remaining):
            break
        if not in_thinking:
            output.append(raw[index])
        index += 1
    return "".join(output)


def _normalize_spoken_output(text: str) -> str:
    spoken = str(text or "")
    if not spoken:
        return ""
    spoken = re.sub(r"[*_`#]+", " ", spoken)
    spoken = re.sub(r"\s+([,.;:!?])", r"\1", spoken)
    spoken = re.sub(r"([(\[{])\s+", r"\1", spoken)
    spoken = re.sub(r"\s+([)\]}])", r"\1", spoken)
    spoken = re.sub(r"\s+", " ", spoken)
    return spoken.strip()


def _append_spoken_delta(prefix: str, delta: str) -> str:
    resolved_prefix = str(prefix or "")
    resolved_delta = str(delta or "")
    if not resolved_prefix:
        return resolved_delta
    if not resolved_delta:
        return resolved_prefix
    return f"{resolved_prefix}{resolved_delta}"


def _split_flushable_spoken_prefix(text: str) -> tuple[str, str]:
    normalized = str(text or "")
    if not normalized:
        return "", ""
    collapsed = re.sub(r"\s+", " ", normalized).lstrip()
    if not collapsed:
        return "", ""
    if re.search(r"[.!?。！？…]['\")\]]*\s*$", collapsed):
        return collapsed.strip(), ""
    last_space = collapsed.rfind(" ")
    if last_space <= 0:
        return "", collapsed
    flush = collapsed[:last_space].strip()
    remain = collapsed[last_space + 1 :].strip()
    if not flush:
        return "", collapsed
    if not re.search(r"[A-Za-z0-9]", flush):
        return "", collapsed
    return flush, remain


@contextmanager
def _cuda_device_context(required_device: str):
    resolved_required = str(required_device or "").strip().lower()
    if not resolved_required.startswith("cuda:"):
        raise RuntimeError(f"vllm required device must be cuda:N, got {required_device}")
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("torch unavailable for vllm device validation") from exc
    expected_index = int(resolved_required.split(":", 1)[1])
    previous_index = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    torch.cuda.set_device(expected_index)
    try:
        yield
    finally:
        torch.cuda.set_device(previous_index)
