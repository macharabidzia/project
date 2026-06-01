from __future__ import annotations

import asyncio
import importlib.util
import inspect
import re
from collections.abc import AsyncIterable, AsyncIterator, Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np


@dataclass(slots=True)
class CosyVoiceBiStreamSession:
    epoch_id: str = ""
    warmed: bool = False
    prompt_text: str = ""
    prompt_speech_path: str = ""
    native_prompt_cache_key: str = ""
    native_prompt_kwargs: dict[str, object] | None = None
    _engine_ref: object | None = None


class TTSEngine:
    def __init__(
        self,
        model_name: str,
        *,
        sample_rate: int = 24_000,
        max_fragment_tokens: int = 6,
        max_lookahead_ms: int = 120,
        max_queue_depth: int = 8,
        prompt_text: str = "",
        prompt_speech_path: str = "",
        text_frontend: bool = False,
        required_device: str = "cuda:1",
        cache_dir: str = "",
    ) -> None:
        self.model_name = str(model_name or "").strip()
        self.sample_rate = int(sample_rate)
        self.max_fragment_tokens = max(2, int(max_fragment_tokens))
        self.max_lookahead_ms = max(20, int(max_lookahead_ms))
        self.max_queue_depth = max(1, int(max_queue_depth))
        self.prompt_text = str(prompt_text or "").strip()
        self.prompt_speech_path = str(prompt_speech_path or "").strip()
        self.text_frontend = bool(text_frontend)
        self.required_device = str(required_device or "cuda:1").strip()
        self.cache_dir = str(cache_dir or "").strip()
        self._session = CosyVoiceBiStreamSession(prompt_text=self.prompt_text, prompt_speech_path=self.prompt_speech_path)
        self._session._engine_ref = self
        self._active_queue_depth = 0
        self._model: Any | None = None
        self._warm = False
        self._cancelled_epochs: set[str] = set()
        self._cancelled_request_ids: set[str] = set()
        self._generator_idle_timeout_s = max(0.5, float(self.max_lookahead_ms) / 1000.0 * 8.0)
        self._generator_resume_gap_s = max(0.25, float(self.max_lookahead_ms) / 1000.0 * 3.0)
        self._generator_weak_resume_rms = 0.04
        self._generator_weak_resume_peak = 0.12
        self._short_reply_tail_token_limit = 3
        self._short_reply_weak_resume_rms = 0.05
        self._short_reply_weak_resume_peak = 0.20
        self._generator_fragment_trace: list[dict[str, object]] = []
        self._generator_fragment_thresholds: dict[str, int] = {}
        self._last_native_mode: str = ""
        self._last_backend_path: str = ""
        self._last_native_text: str = ""

    @property
    def is_warm(self) -> bool:
        return bool(self._warm and self._session.warmed and self._model is not None)

    def warm(self, *, model_dir: str | None = None, strict: bool | None = None) -> None:
        strict_mode = True if strict is None else bool(strict)
        if not strict_mode:
            raise RuntimeError("tts_strict_mode_required")
        resolved_model_dir = str(model_dir or self.model_name).strip()
        if not resolved_model_dir:
            raise RuntimeError("cosyvoice model_dir is required for strict warm start")
        if resolved_model_dir.startswith("http://") or resolved_model_dir.startswith("https://"):
            raise RuntimeError(f"cosyvoice model_dir must be local/offline, got {resolved_model_dir}")

        model_path = Path(resolved_model_dir)
        if not model_path.exists():
            raise RuntimeError(f"cosyvoice model_dir does not exist: {resolved_model_dir}")
        if not model_path.is_dir():
            raise RuntimeError(f"cosyvoice model_dir is not a directory: {resolved_model_dir}")
        if not self.cache_dir:
            raise RuntimeError("cosyvoice cache_dir is required for strict warm start")
        if self.cache_dir.startswith("http://") or self.cache_dir.startswith("https://"):
            raise RuntimeError(f"cosyvoice cache_dir must be local/offline, got {self.cache_dir}")
        cache_path = Path(self.cache_dir)
        if not cache_path.exists() or not cache_path.is_dir():
            raise RuntimeError(f"cosyvoice cache_dir does not exist or is not a directory: {self.cache_dir}")
        _assert_cuda_device_binding(self.required_device, lane_label="tts")

        try:
            _ensure_cosyvoice_runtime_paths()
            from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("CosyVoice runtime is unavailable") from exc

        with _cuda_device_context(self.required_device, lane_label="tts"):
            self._model = AutoModel(model_dir=resolved_model_dir)
        self.sample_rate = int(getattr(self._model, "sample_rate", self.sample_rate) or self.sample_rate)
        self._session.warmed = True
        self._session.epoch_id = str(self._session.epoch_id).strip()
        self._warm = True

    def start_persistent_session(self, *, epoch_id: str = "", prompt_text: str = "", prompt_speech_path: str = "") -> None:
        if self._model is None:
            raise RuntimeError("cosyvoice_native_bistream_unavailable")
        self._session.warmed = True
        self._session.epoch_id = str(epoch_id or self._session.epoch_id).strip()
        if prompt_text:
            self._session.prompt_text = str(prompt_text).strip()
        if prompt_speech_path:
            self._session.prompt_speech_path = str(prompt_speech_path).strip()
        if self._session.prompt_speech_path:
            resolved_prompt_path = Path(self._session.prompt_speech_path)
            if not resolved_prompt_path.exists() or not resolved_prompt_path.is_file():
                raise RuntimeError(f"cosyvoice speaker asset missing: {self._session.prompt_speech_path}")
        if self._session.epoch_id:
            self._cancelled_epochs.discard(self._session.epoch_id)
        self._session.native_prompt_cache_key = ""
        self._session.native_prompt_kwargs = None
        with _cuda_device_context(self.required_device, lane_label="tts"):
            _prime_native_prompt_cache(self._model, self._session)
        self._warm = True

    def reset(self, *, epoch_id: str) -> None:
        self._session.warmed = True
        self._session.epoch_id = str(epoch_id or "").strip()
        if self._session.epoch_id:
            self._cancelled_epochs.discard(self._session.epoch_id)
        self._generator_fragment_trace.clear()
        self._generator_fragment_thresholds = {}
        self._last_native_mode = ""
        self._last_backend_path = ""
        self._last_native_text = ""

    def cancel(self, *, request_id: str = "", epoch_id: str = "") -> None:
        resolved_epoch = str(epoch_id or self._session.epoch_id).strip()
        resolved_request_id = str(request_id or "").strip()
        if resolved_epoch:
            self._cancelled_epochs.add(resolved_epoch)
        if resolved_request_id:
            self._cancelled_request_ids.add(resolved_request_id)

    def shutdown(self) -> None:
        self._active_queue_depth = 0
        self._warm = False
        self._model = None
        self._cancelled_epochs.clear()
        self._cancelled_request_ids.clear()
        self._generator_fragment_trace.clear()
        self._generator_fragment_thresholds = {}
        self._last_native_mode = ""
        self._last_backend_path = ""
        self._last_native_text = ""
        self._session = CosyVoiceBiStreamSession(
            prompt_text=self.prompt_text,
            prompt_speech_path=self.prompt_speech_path,
        )
        self._session._engine_ref = self
        try:
            import torch  # type: ignore
        except Exception:
            return
        if not torch.cuda.is_available():
            return
        try:
            with _cuda_device_context(self.required_device, lane_label="tts"):
                torch.cuda.empty_cache()
        except Exception:
            # Teardown should not mask the original shutdown path.
            pass

    def debug_metrics(self) -> dict[str, object]:
        return {
            "generator_fragment_thresholds": dict(self._generator_fragment_thresholds),
            "generator_fragment_trace": [dict(item) for item in self._generator_fragment_trace],
            "last_native_mode": str(self._last_native_mode),
            "last_backend_path": str(self._last_backend_path),
            "last_native_text": str(self._last_native_text),
        }

    def _record_generator_fragment(
        self,
        item: dict[str, object],
    ) -> None:
        record = dict(item)
        self._last_native_mode = "native_bistream"
        self._last_backend_path = "native_bistream"
        fragment_text = " ".join(str(record.get("text", "") or "").strip().split())
        if fragment_text:
            self._last_native_text = fragment_text
        record["emitted_index"] = int(len(self._generator_fragment_trace) + 1)
        if len(self._generator_fragment_trace) >= 12:
            self._generator_fragment_trace.pop(0)
        self._generator_fragment_trace.append(record)

    async def stream_pcm(
        self,
        text: str | Iterable[str] | AsyncIterable[str],
        *,
        request_id: str = "",
        epoch_id: str = "",
        prompt_text: str = "",
        prompt_speech_path: str = "",
    ) -> AsyncIterator[tuple[bytes, int, bool]]:
        if self._active_queue_depth >= self.max_queue_depth:
            raise RuntimeError("tts_queue_overflow")
        self._active_queue_depth += 1
        resolved_request_id = str(request_id or "").strip()
        resolved_epoch = str(epoch_id or self._session.epoch_id).strip()
        resolved_prompt_text = str(prompt_text or self._session.prompt_text or self.prompt_text).strip()
        resolved_prompt_speech_path = str(prompt_speech_path or self._session.prompt_speech_path or self.prompt_speech_path).strip()
        if not self.is_warm:
            raise RuntimeError("tts_streaming_engine_not_warm")
        if resolved_request_id and resolved_request_id in self._cancelled_request_ids:
            return
        if resolved_epoch and resolved_epoch in self._cancelled_epochs:
            return

        if self._session.epoch_id and resolved_epoch and self._session.epoch_id != resolved_epoch:
            raise ValueError("stale_epoch_fragment")

        if resolved_epoch and not self._session.epoch_id:
            self._session.epoch_id = resolved_epoch
        if resolved_prompt_text and self._session.prompt_text != resolved_prompt_text:
            self._session.prompt_text = resolved_prompt_text
            self._session.native_prompt_cache_key = ""
            self._session.native_prompt_kwargs = None
        if resolved_prompt_speech_path and self._session.prompt_speech_path != resolved_prompt_speech_path:
            self._session.prompt_speech_path = resolved_prompt_speech_path
            self._session.native_prompt_cache_key = ""
            self._session.native_prompt_kwargs = None

        try:
            if self._model is None:
                raise RuntimeError("cosyvoice_native_bistream_unavailable")
            inference_kwargs: dict[str, object] = {"text_frontend": bool(self.text_frontend)}
            if isinstance(text, Generator):
                async for pcm, sample_rate, is_final in self._stream_generator_pcm_threaded(
                    text,
                    request_id=resolved_request_id,
                    epoch_id=resolved_epoch,
                    prompt_text=resolved_prompt_text,
                    prompt_speech_path=resolved_prompt_speech_path,
                    inference_kwargs=inference_kwargs,
                ):
                    yield pcm, sample_rate, is_final
                return
            saw_fragment = False
            async for fragment_text, is_last_fragment in _iter_fragments_with_last(text):
                saw_fragment = True
                if resolved_request_id and resolved_request_id in self._cancelled_request_ids:
                    break
                if resolved_epoch and resolved_epoch in self._cancelled_epochs:
                    break
                emitted_pcm = False
                last_emit_monotonic = 0.0
                self._last_native_mode = "native_bistream"
                self._last_backend_path = "native_bistream"
                self._last_native_text = str(fragment_text)
                with _cuda_device_context(self.required_device, lane_label="tts"):
                    native_iterable = _call_cosyvoice_native_stream(
                        self._model,
                        fragment_text,
                        resolved_prompt_text,
                        resolved_prompt_speech_path,
                        session=self._session,
                        **inference_kwargs,
                    )
                    for item in native_iterable:
                        if resolved_request_id and resolved_request_id in self._cancelled_request_ids:
                            break
                        if resolved_epoch and resolved_epoch in self._cancelled_epochs:
                            break
                        pcm = _tts_speech_to_pcm_bytes(item.get("tts_speech"))
                        if pcm is None:
                            continue
                        now_monotonic = time.monotonic()
                        if self._should_drop_resumed_tail(
                            pcm=pcm,
                            emitted_pcm=emitted_pcm,
                            last_emit_monotonic=last_emit_monotonic,
                            fragment_text=fragment_text,
                        ):
                            break
                        emitted_pcm = True
                        last_emit_monotonic = now_monotonic
                        await asyncio.sleep(0)
                        yield pcm, int(self.sample_rate), False
                if emitted_pcm:
                    await asyncio.sleep(0)
                    yield b"", int(self.sample_rate), bool(is_last_fragment)
            if not saw_fragment:
                return
        finally:
            if resolved_epoch and resolved_epoch in self._cancelled_epochs:
                self._cancelled_epochs.discard(resolved_epoch)
            if resolved_request_id and resolved_request_id in self._cancelled_request_ids:
                self._cancelled_request_ids.discard(resolved_request_id)
            self._active_queue_depth = max(0, self._active_queue_depth - 1)

    async def _stream_generator_pcm_threaded(
        self,
        text: Generator[str, None, None],
        *,
        request_id: str,
        epoch_id: str,
        prompt_text: str,
        prompt_speech_path: str,
        inference_kwargs: dict[str, object],
    ) -> AsyncIterator[tuple[bytes, int, bool]]:
        loop = asyncio.get_running_loop()
        results: asyncio.Queue[object] = asyncio.Queue()
        sentinel = object()
        emitted_any = False
        final_sent = False

        def _run() -> None:
            emitted_pcm = False
            sample_rate = int(self.sample_rate)
            last_emit_monotonic = 0.0
            try:
                self._last_native_mode = "native_bistream"
                self._last_backend_path = "native_bistream"
                with _cuda_device_context(self.required_device, lane_label="tts"):
                    generator = _call_cosyvoice_native_stream(
                        self._model,
                        text,
                        prompt_text,
                        prompt_speech_path,
                        session=self._session,
                        **inference_kwargs,
                    )
                    for item in generator:
                        if request_id and request_id in self._cancelled_request_ids:
                            break
                        if epoch_id and epoch_id in self._cancelled_epochs:
                            break
                        pcm = _tts_speech_to_pcm_bytes(item.get("tts_speech"))
                        if pcm is None:
                            continue
                        now_monotonic = time.monotonic()
                        if self._should_drop_resumed_tail(
                            pcm=pcm,
                            emitted_pcm=emitted_pcm,
                            last_emit_monotonic=last_emit_monotonic,
                            fragment_text="",
                        ):
                            break
                        emitted_pcm = True
                        last_emit_monotonic = now_monotonic
                        loop.call_soon_threadsafe(
                            results.put_nowait,
                            (pcm, sample_rate, False),
                        )
                if emitted_pcm:
                    loop.call_soon_threadsafe(
                        results.put_nowait,
                        (b"", sample_rate, True),
                    )
            except Exception as exc:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(results.put_nowait, exc)
            finally:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(results.put_nowait, sentinel)

        threading.Thread(
            target=_run,
            name=f"tts-generator:{request_id or epoch_id or 'session'}",
            daemon=True,
        ).start()

        while True:
            if emitted_any:
                try:
                    item = await asyncio.wait_for(results.get(), timeout=float(self._generator_idle_timeout_s))
                except asyncio.TimeoutError:
                    self.cancel(request_id=request_id, epoch_id=epoch_id)
                    if not final_sent:
                        final_sent = True
                        yield b"", int(self.sample_rate), True
                    break
            else:
                item = await results.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            pcm, sample_rate, is_final = item
            if pcm:
                emitted_any = True
            if is_final:
                final_sent = True
            await asyncio.sleep(0)
            yield pcm, sample_rate, is_final

    def _should_drop_resumed_tail(
        self,
        *,
        pcm: bytes,
        emitted_pcm: bool,
        last_emit_monotonic: float,
        fragment_text: str,
    ) -> bool:
        if (
            not emitted_pcm
            or last_emit_monotonic <= 0.0
            or (time.monotonic() - last_emit_monotonic) < float(self._generator_resume_gap_s)
        ):
            return False
        resumed_rms, resumed_peak = _pcm_bytes_rms_peak(pcm)
        if (
            resumed_rms <= float(self._generator_weak_resume_rms)
            and resumed_peak <= float(self._generator_weak_resume_peak)
        ):
            return True
        if _token_count(fragment_text) <= int(self._short_reply_tail_token_limit):
            if (
                resumed_rms <= float(self._short_reply_weak_resume_rms)
                and resumed_peak <= float(self._short_reply_weak_resume_peak)
            ):
                return True
        return False


def _pcm_bytes_rms_peak(pcm: bytes) -> tuple[float, float]:
    if not pcm:
        return 0.0, 0.0
    samples = np.frombuffer(bytes(pcm), dtype="<i2").astype(np.float32)
    if samples.size == 0:
        return 0.0, 0.0
    normalized = samples / 32768.0
    rms = float(np.sqrt(np.mean(np.square(normalized))))
    peak = float(np.max(np.abs(normalized)))
    return rms, peak


def _normalize_fragment(fragment: str) -> str:
    return " ".join(str(fragment).strip().split())


async def _iter_fragments_with_last(
    text: str | Iterable[str] | AsyncIterable[str],
) -> AsyncIterator[tuple[str, bool]]:
    if isinstance(text, str):
        normalized = _normalize_fragment(text)
        if normalized:
            yield normalized, True
        return
    if isinstance(text, AsyncIterable):
        pending: str | None = None
        async for fragment in text:
            normalized = _normalize_fragment(fragment)
            if not normalized:
                continue
            if pending is not None:
                yield pending, False
            pending = normalized
        if pending is not None:
            yield pending, True
        return
    pending: str | None = None
    for fragment in text:
        normalized = _normalize_fragment(fragment)
        if not normalized:
            continue
        if pending is not None:
            yield pending, False
        pending = normalized
    if pending is not None:
        yield pending, True


def _iter_text_tokens(text: str) -> Iterator[str]:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return
    for token in normalized.split(" "):
        if token:
            yield token


def _resolve_native_stream_emit_thresholds(model: Any) -> tuple[int, int]:
    token_min_hop = max(1, int(getattr(model, "token_min_hop_len", 0) or 0))
    mix_text_tokens = max(1, int(os.getenv("COSYVOICE_BISTREAM_TEXT_MIX_TOKENS", "8")))
    default_emit_windows = max(2, int(os.getenv("VOICE_PIPELINE_TTS_NATIVE_STREAM_EMIT_WINDOWS", "2")))
    default_emit_tokens = max(token_min_hop * 2, mix_text_tokens * default_emit_windows, 8)
    min_emit_tokens = max(
        1,
        int(os.getenv("VOICE_PIPELINE_TTS_NATIVE_STREAM_MIN_EMIT_TOKENS", str(default_emit_tokens))),
    )
    default_sentence_tokens = max(mix_text_tokens, token_min_hop * 2, 6)
    min_sentence_tokens = max(
        1,
        int(os.getenv("VOICE_PIPELINE_TTS_NATIVE_STREAM_MIN_SENTENCE_TOKENS", str(default_sentence_tokens))),
    )
    return min_emit_tokens, min_sentence_tokens


def _resolve_native_max_prompt_speech_tokens(*, generator_stream: bool = False) -> int:
    if generator_stream:
        raw = str(
            os.getenv(
                "VOICE_PIPELINE_TTS_NATIVE_MAX_GENERATOR_PROMPT_SPEECH_TOKENS",
                os.getenv("VOICE_PIPELINE_TTS_NATIVE_MAX_PROMPT_SPEECH_TOKENS", "0"),
            )
        ).strip()
    else:
        raw = str(os.getenv("VOICE_PIPELINE_TTS_NATIVE_MAX_PROMPT_SPEECH_TOKENS", "0")).strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _force_zero_shot_native_stream() -> bool:
    value = str(os.getenv("VOICE_PIPELINE_TTS_FORCE_ZERO_SHOT_STREAM", "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _ensure_endofprompt(prompt_text: str) -> str:
    resolved_prompt_text = str(prompt_text or "").strip()
    if "<|endofprompt|>" in resolved_prompt_text:
        return resolved_prompt_text
    return (
        f"You are a helpful assistant.<|endofprompt|>{resolved_prompt_text}"
        if resolved_prompt_text
        else "You are a helpful assistant.<|endofprompt|>"
    )


def _resolve_native_stream_inference(model: Any):
    direct_bistream = getattr(model, "inference_bistream", None)
    if callable(direct_bistream):
        return ("direct", direct_bistream)
    model_impl = getattr(model, "model", None)
    frontend = getattr(model, "frontend", None)
    synthesize = getattr(model_impl, "tts", None)
    if model_impl is None or frontend is None or not callable(synthesize):
        return None
    return ("frontend", (frontend, synthesize))


def _call_cosyvoice_native_stream(
    model: Any,
    text_input: str | Iterable[str],
    prompt_text: str,
    prompt_speech_path: str,
    session: CosyVoiceBiStreamSession | None = None,
    **kwargs: object,
):
    if model is None:
        raise RuntimeError("CosyVoice model is not warm")
    resolved_stream = _resolve_native_stream_inference(model)
    if resolved_stream is None:
        raise RuntimeError("CosyVoice runtime does not expose native stream inference")
    stream_kind, stream_target = resolved_stream
    resolved_prompt_path = str(prompt_speech_path or "").strip()
    resolved_prompt_text = str(prompt_text or "").strip()
    if stream_kind == "direct":
        direct_bistream = stream_target
        return direct_bistream(text_input, resolved_prompt_text, resolved_prompt_path, stream=True)

    frontend, synthesize = stream_target
    generator_stream = isinstance(text_input, Generator)
    if isinstance(text_input, str):
        normalized_text = " ".join(str(text_input).strip().split())
    else:
        normalized_text = "" if generator_stream else " ".join(_iter_fragments(text_input))
    if generator_stream:
        cross_lingual_mode = bool(resolved_prompt_path) and _contains_cjk(resolved_prompt_text)
    else:
        cross_lingual_mode = bool(resolved_prompt_path) and _should_use_cross_lingual_mode(
            text=normalized_text,
            prompt_text=resolved_prompt_text,
        )
    if cross_lingual_mode and _force_zero_shot_native_stream():
        cross_lingual_mode = False
        if cross_lingual_mode:
            normalized_text = _ensure_english_lang_tag(normalized_text)

    text_stream: str | Iterator[str]
    if isinstance(text_input, str):
        text_stream = normalized_text
    elif generator_stream:
        min_emit_tokens, min_sentence_tokens = _resolve_native_stream_emit_thresholds(model)
        if session is not None:
            session_engine = getattr(session, "_engine_ref", None)
            if session_engine is not None:
                session_engine._generator_fragment_trace.clear()
                session_engine._generator_fragment_thresholds = {
                    "min_emit_tokens": int(min_emit_tokens),
                    "min_sentence_tokens": int(min_sentence_tokens),
                }
        text_stream = _streaming_text_fragments(
            text_input,
            add_english_lang_tag=bool(cross_lingual_mode),
            min_emit_tokens=min_emit_tokens,
            min_sentence_tokens=min_sentence_tokens,
            on_emit=(
                session_engine._record_generator_fragment
                if session is not None and getattr(session, "_engine_ref", None) is not None
                else None
            ),
        )
    else:
        text_stream = normalized_text

    if resolved_prompt_path:
        cached_prompt_kwargs: dict[str, object] | None = None
        if not cross_lingual_mode:
            resolved_prompt_text = _ensure_endofprompt(resolved_prompt_text)
        prompt_trim_limit = _resolve_native_max_prompt_speech_tokens(generator_stream=generator_stream)
        cache_key = "|".join(
            [
                "generator" if generator_stream else "string",
                "cross" if cross_lingual_mode else "zero",
                resolved_prompt_text,
                resolved_prompt_path,
                f"trim:{prompt_trim_limit}",
            ]
        )
        if session is not None and session.native_prompt_cache_key == cache_key and session.native_prompt_kwargs:
            cached_prompt_kwargs = dict(session.native_prompt_kwargs)
        if cached_prompt_kwargs is None:
            if cross_lingual_mode:
                cached_prompt_kwargs = frontend.frontend_cross_lingual(
                    "",
                    resolved_prompt_path,
                    int(getattr(model, "sample_rate", 24000) or 24000),
                    "",
                )
            else:
                cached_prompt_kwargs = frontend.frontend_zero_shot(
                    "",
                    resolved_prompt_text,
                    resolved_prompt_path,
                    int(getattr(model, "sample_rate", 24000) or 24000),
                    "",
                )
            cached_prompt_kwargs.pop("text", None)
            cached_prompt_kwargs.pop("text_len", None)
            cached_prompt_kwargs = _trim_native_prompt_kwargs(
                cached_prompt_kwargs,
                max_prompt_tokens=prompt_trim_limit,
                generator_stream=generator_stream,
            )
            if session is not None:
                session.native_prompt_cache_key = cache_key
                session.native_prompt_kwargs = dict(cached_prompt_kwargs)
        text_token, text_token_len = frontend._extract_text_token(text_stream)
        model_input = {
            **cached_prompt_kwargs,
            "text": text_token,
            "text_len": text_token_len,
        }
    else:
        list_available_spks = getattr(model, "list_available_spks", None)
        available_spks = list(list_available_spks()) if callable(list_available_spks) else []
        if not available_spks:
            raise RuntimeError("cosyvoice_builtin_speaker_unavailable")
        model_input = frontend.frontend_sft(text_stream, str(available_spks[0]))

    signature = inspect.signature(synthesize)
    call_kwargs = {"stream": True}
    if "speed" in signature.parameters:
        call_kwargs["speed"] = float(kwargs.get("speed", 1.0) or 1.0)
    return synthesize(**model_input, **call_kwargs)


def _prime_native_prompt_cache(model: Any, session: CosyVoiceBiStreamSession) -> None:
    if model is None or session is None:
        return
    resolved_prompt_path = str(session.prompt_speech_path or "").strip()
    resolved_prompt_text = str(session.prompt_text or "").strip()
    if not resolved_prompt_path:
        return
    resolved_stream = _resolve_native_stream_inference(model)
    if resolved_stream is None:
        return
    stream_kind, stream_target = resolved_stream
    if stream_kind == "direct":
        return
    frontend, _ = stream_target
    cross_lingual_mode = bool(resolved_prompt_path) and _should_use_cross_lingual_mode(
        text="<|en|>warm",
        prompt_text=resolved_prompt_text,
    )
    if cross_lingual_mode and _force_zero_shot_native_stream():
        cross_lingual_mode = False
    prompt_trim_limit = _resolve_native_max_prompt_speech_tokens(generator_stream=True)
    cache_key = "|".join(
        [
            "generator",
            "cross" if cross_lingual_mode else "zero",
            _ensure_endofprompt(resolved_prompt_text) if not cross_lingual_mode else resolved_prompt_text,
            resolved_prompt_path,
            f"trim:{prompt_trim_limit}",
        ]
    )
    if session.native_prompt_cache_key == cache_key and session.native_prompt_kwargs:
        return
    if cross_lingual_mode:
        prompt_kwargs = frontend.frontend_cross_lingual(
            "",
            resolved_prompt_path,
            int(getattr(model, "sample_rate", 24000) or 24000),
            "",
        )
    else:
        prompt_kwargs = frontend.frontend_zero_shot(
            "",
            _ensure_endofprompt(resolved_prompt_text),
            resolved_prompt_path,
            int(getattr(model, "sample_rate", 24000) or 24000),
            "",
    )
    prompt_kwargs.pop("text", None)
    prompt_kwargs.pop("text_len", None)
    session.native_prompt_cache_key = cache_key
    session.native_prompt_kwargs = _trim_native_prompt_kwargs(
        prompt_kwargs,
        max_prompt_tokens=prompt_trim_limit,
        generator_stream=True,
    )


def _trim_native_prompt_kwargs(
    prompt_kwargs: dict[str, object],
    *,
    max_prompt_tokens: int | None = None,
    generator_stream: bool = False,
) -> dict[str, object]:
    if max_prompt_tokens is None:
        max_prompt_tokens = _resolve_native_max_prompt_speech_tokens(generator_stream=generator_stream)
    if max_prompt_tokens <= 0:
        return dict(prompt_kwargs)
    trimmed = dict(prompt_kwargs)
    # Keep prompt-conditioning aligned to the original prompt onset. The real
    # room traces showed generator tail-slicing weakens the first native burst
    # and expands the native chunk schedule without improving continuity.
    keep_tail = False

    def _trim_token_tensor(key: str, len_key: str) -> int:
        tensor = trimmed.get(key)
        if tensor is None or not hasattr(tensor, "shape"):
            return 0
        token_count = int(tensor.shape[1]) if len(tensor.shape) >= 2 else 0
        if token_count <= 0 or token_count <= max_prompt_tokens:
            return token_count
        if keep_tail:
            trimmed[key] = tensor[:, token_count - max_prompt_tokens :]
        else:
            trimmed[key] = tensor[:, :max_prompt_tokens]
        if len_key in trimmed:
            len_tensor = trimmed[len_key]
            if hasattr(len_tensor, "new_tensor"):
                trimmed[len_key] = len_tensor.new_tensor([max_prompt_tokens])
        return max_prompt_tokens

    llm_tokens = _trim_token_tensor("llm_prompt_speech_token", "llm_prompt_speech_token_len")
    flow_tokens_before = 0
    flow_tensor = trimmed.get("flow_prompt_speech_token")
    if flow_tensor is not None and hasattr(flow_tensor, "shape") and len(flow_tensor.shape) >= 2:
        flow_tokens_before = int(flow_tensor.shape[1])
    flow_tokens_after = _trim_token_tensor("flow_prompt_speech_token", "flow_prompt_speech_token_len")
    prompt_feat = trimmed.get("prompt_speech_feat")
    if (
        prompt_feat is not None
        and hasattr(prompt_feat, "shape")
        and len(prompt_feat.shape) >= 3
        and flow_tokens_before > 0
        and flow_tokens_after > 0
        and flow_tokens_after < flow_tokens_before
    ):
        feat_frames = int(prompt_feat.shape[1])
        trimmed_frames = max(1, int(round(float(feat_frames) * float(flow_tokens_after) / float(flow_tokens_before))))
        if keep_tail:
            trimmed["prompt_speech_feat"] = prompt_feat[:, feat_frames - trimmed_frames :, :]
        else:
            trimmed["prompt_speech_feat"] = prompt_feat[:, :trimmed_frames, :]
        feat_len = trimmed.get("prompt_speech_feat_len")
        if hasattr(feat_len, "new_tensor"):
            trimmed["prompt_speech_feat_len"] = feat_len.new_tensor([trimmed_frames])
    return trimmed


_SENTENCE_END_RE = re.compile(r"[.!?。！？…]['\")\\]]*$")
_FOLLOWUP_NOTE_PREFIX_RE = re.compile(
    r"^[\s\?\!\.,;:)\]\"']*(?:\*\*\s*)?note(?:\s*:\s*|\s+)",
    re.IGNORECASE,
)


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(str(text or "").strip()))


def _token_count(text: str) -> int:
    normalized = " ".join(str(text or "").strip().split())
    return len(normalized.split(" ")) if normalized else 0


def _strip_followup_metadata_tail(fragment: str) -> str:
    normalized = _normalize_fragment(fragment)
    if not normalized:
        return ""
    if _FOLLOWUP_NOTE_PREFIX_RE.match(normalized):
        return ""
    return normalized


def _streaming_text_fragments(
    text_input: Generator[str, None, None],
    *,
    add_english_lang_tag: bool,
    min_emit_tokens: int,
    min_sentence_tokens: int = 3,
    on_emit: callable | None = None,
) -> Iterator[str]:
    prefixed = not add_english_lang_tag
    emitted_any = False
    suppress_followup_metadata = False
    buffered: list[str] = []
    buffered_tokens = 0
    for fragment in text_input:
        normalized = _normalize_fragment(fragment)
        if emitted_any:
            if suppress_followup_metadata:
                continue
            if _FOLLOWUP_NOTE_PREFIX_RE.match(normalized):
                suppress_followup_metadata = True
                continue
            normalized = _strip_followup_metadata_tail(normalized)
        if not normalized:
            continue
        buffered.append(normalized)
        buffered_tokens += _token_count(normalized)
        combined = " ".join(buffered)
        should_emit = False
        if buffered_tokens >= max(1, int(min_emit_tokens)):
            should_emit = True
        elif _ends_sentence(normalized) and buffered_tokens >= max(1, int(min_sentence_tokens)):
            should_emit = True
        if not should_emit:
            continue
        if add_english_lang_tag and not prefixed:
            combined = _ensure_english_lang_tag(combined)
            prefixed = True
        if callable(on_emit):
            on_emit(
                {
                    "text": str(combined),
                    "token_count": int(_token_count(combined)),
                    "buffered_source_fragments": int(len(buffered)),
                    "min_emit_tokens": int(min_emit_tokens),
                    "min_sentence_tokens": int(min_sentence_tokens),
                    "sentence_boundary": bool(_ends_sentence(normalized)),
                    "emitted_index": 0,
                }
            )
        yield combined
        emitted_any = True
        buffered.clear()
        buffered_tokens = 0
    if buffered:
        combined = " ".join(buffered)
        if add_english_lang_tag and not prefixed:
            combined = _ensure_english_lang_tag(combined)
        if callable(on_emit):
            on_emit(
                {
                    "text": str(combined),
                    "token_count": int(_token_count(combined)),
                    "buffered_source_fragments": int(len(buffered)),
                    "min_emit_tokens": int(min_emit_tokens),
                    "min_sentence_tokens": int(min_sentence_tokens),
                    "sentence_boundary": False,
                    "emitted_index": 0,
                }
            )
        yield combined


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(str(text or "")))


def _latin_ratio(text: str) -> float:
    normalized = str(text or "")
    letters = [ch for ch in normalized if ch.isalpha()]
    if not letters:
        return 0.0
    latin_letters = sum(1 for ch in letters if _LATIN_RE.fullmatch(ch))
    return float(latin_letters) / float(len(letters))


def _should_use_cross_lingual_mode(*, text: str, prompt_text: str) -> bool:
    normalized_text = " ".join(str(text or "").strip().split())
    if not normalized_text:
        return False
    if normalized_text.startswith("<|"):
        return True
    if _contains_cjk(normalized_text):
        return False
    if _latin_ratio(normalized_text) < 0.6:
        return False
    # The current reference prompt is Chinese. Route Latin-only output through the
    # native cross-lingual frontend instead of the zero-shot prompt-text path.
    return _contains_cjk(prompt_text)


def _ensure_english_lang_tag(text: str) -> str:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return normalized
    if normalized.startswith("<|"):
        return normalized
    return f"<|en|>{normalized}"


def _tts_speech_to_pcm_bytes(tts_speech: object) -> bytes | None:
    if tts_speech is None:
        return None
    array = np.asarray(tts_speech)
    if array.size == 0:
        return None
    if np.issubdtype(array.dtype, np.floating):
        max_abs = float(np.max(np.abs(array))) if array.size else 0.0
        if max_abs <= 1.0:
            array = np.clip(array, -1.0, 1.0)
            array = (array * 32767.0).astype("<i2")
        else:
            array = np.clip(array, -32768.0, 32767.0).astype("<i2")
    else:
        array = np.asarray(array, dtype="<i2")
    return np.ascontiguousarray(array.reshape(-1)).tobytes()


__all__ = ["CosyVoiceBiStreamSession", "TTSEngine"]


def _prepend_sys_path(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return
    resolved = str(path.resolve())
    if resolved in sys.path:
        return
    sys.path.insert(0, resolved)


def _candidate_cosyvoice_roots() -> tuple[Path, ...]:
    configured = str(os.getenv("COSYVOICE_REPO_DIR", "")).strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    cwd_candidate = Path.cwd() / ".models" / "CosyVoice-runtime"
    candidates.append(cwd_candidate)
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / ".models" / "CosyVoice-runtime")
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return tuple(deduped)


def _ensure_cosyvoice_runtime_paths() -> None:
    if importlib.util.find_spec("cosyvoice") is not None:
        return
    for root in _candidate_cosyvoice_roots():
        _prepend_sys_path(root)
        _prepend_sys_path(root / "third_party" / "Matcha-TTS")
        if importlib.util.find_spec("cosyvoice") is not None:
            return


def _assert_cuda_device_binding(required_device: str, *, lane_label: str) -> None:
    resolved_required = str(required_device or "").strip().lower()
    if not resolved_required.startswith("cuda:"):
        raise RuntimeError(f"{lane_label} required device must be cuda:N, got {required_device}")
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError(f"torch unavailable for {lane_label} device validation") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(f"cuda unavailable for {lane_label}")
    expected_index = int(resolved_required.split(":", 1)[1])
    if expected_index >= int(torch.cuda.device_count()):
        raise RuntimeError(f"{lane_label} required device missing: {required_device}")
    with _cuda_device_context(resolved_required, lane_label=lane_label):
        current_index = int(torch.cuda.current_device())
        if current_index != expected_index:
            raise RuntimeError(
                f"{lane_label} device binding mismatch: expected {resolved_required}, current cuda:{current_index}"
            )


@contextmanager
def _cuda_device_context(required_device: str, *, lane_label: str):
    resolved_required = str(required_device or "").strip().lower()
    if not resolved_required.startswith("cuda:"):
        raise RuntimeError(f"{lane_label} required device must be cuda:N, got {required_device}")
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError(f"torch unavailable for {lane_label} device validation") from exc
    expected_index = int(resolved_required.split(":", 1)[1])
    previous_index = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    torch.cuda.set_device(expected_index)
    try:
        yield
    finally:
        try:
            torch.cuda.set_device(previous_index)
        except Exception:
            # Probe cancellation can surface latent CUDA OOM state during teardown.
            # Restoring the previous device is best-effort and should not mask
            # the already-handled generation result.
            pass
