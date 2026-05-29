from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class CosyVoiceBiStreamSession:
    epoch_id: str = ""
    warmed: bool = False
    prompt_text: str = ""
    prompt_speech_path: str = ""


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
    ) -> None:
        self.model_name = str(model_name or "").strip()
        self.sample_rate = int(sample_rate)
        self.max_fragment_tokens = max(2, int(max_fragment_tokens))
        self.max_lookahead_ms = max(20, int(max_lookahead_ms))
        self.max_queue_depth = max(1, int(max_queue_depth))
        self.prompt_text = str(prompt_text or "").strip()
        self.prompt_speech_path = str(prompt_speech_path or "").strip()
        self.text_frontend = bool(text_frontend)
        self._session = CosyVoiceBiStreamSession(prompt_text=self.prompt_text, prompt_speech_path=self.prompt_speech_path)
        self._active_queue_depth = 0
        self._model: Any | None = None
        self._warm = False
        self._cancelled_epochs: set[str] = set()
        self._cancelled_request_ids: set[str] = set()

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

        if not Path(resolved_model_dir).exists():
            raise RuntimeError(f"cosyvoice model_dir does not exist: {resolved_model_dir}")

        try:
            from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("CosyVoice runtime is unavailable") from exc

        self._model = AutoModel(model_dir=resolved_model_dir)
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
        if self._session.epoch_id:
            self._cancelled_epochs.discard(self._session.epoch_id)
        self._warm = True

    def reset(self, *, epoch_id: str) -> None:
        self._session.warmed = True
        self._session.epoch_id = str(epoch_id or "").strip()
        if self._session.epoch_id:
            self._cancelled_epochs.discard(self._session.epoch_id)

    def cancel(self, *, request_id: str = "", epoch_id: str = "") -> None:
        resolved_epoch = str(epoch_id or self._session.epoch_id).strip()
        resolved_request_id = str(request_id or "").strip()
        if resolved_epoch:
            self._cancelled_epochs.add(resolved_epoch)
        if resolved_request_id:
            self._cancelled_request_ids.add(resolved_request_id)

    async def stream_pcm(
        self,
        text: str | Iterable[str],
        *,
        epoch_id: str = "",
        prompt_text: str = "",
        prompt_speech_path: str = "",
    ) -> AsyncIterator[tuple[bytes, int, bool]]:
        if self._active_queue_depth >= self.max_queue_depth:
            raise RuntimeError("tts_queue_overflow")
        self._active_queue_depth += 1
        resolved_epoch = str(epoch_id or self._session.epoch_id).strip()
        resolved_prompt_text = str(prompt_text or self._session.prompt_text or self.prompt_text).strip()
        resolved_prompt_speech_path = str(prompt_speech_path or self._session.prompt_speech_path or self.prompt_speech_path).strip()
        if not self.is_warm:
            raise RuntimeError("tts_streaming_engine_not_warm")
        if resolved_epoch and resolved_epoch in self._cancelled_epochs:
            return

        if self._session.epoch_id and resolved_epoch and self._session.epoch_id != resolved_epoch:
            raise ValueError("stale_epoch_fragment")

        if resolved_epoch and not self._session.epoch_id:
            self._session.epoch_id = resolved_epoch
        if resolved_prompt_text:
            self._session.prompt_text = resolved_prompt_text
        if resolved_prompt_speech_path:
            self._session.prompt_speech_path = resolved_prompt_speech_path

        fragments = tuple(_iter_fragments(text))
        if not fragments:
            return

        try:
            if self._model is None:
                raise RuntimeError("cosyvoice_native_bistream_unavailable")
            inference_kwargs: dict[str, object] = {
                "stream": True,
                "text_frontend": bool(self.text_frontend),
            }
            final_fragment_index = len(fragments) - 1
            for fragment_index, fragment in enumerate(fragments):
                if resolved_epoch and resolved_epoch in self._cancelled_epochs:
                    break
                generator = _call_cosyvoice_native_stream(
                    self._model,
                    fragment,
                    resolved_prompt_text,
                    resolved_prompt_speech_path,
                    **inference_kwargs,
                )
                for item in generator:
                    if resolved_epoch and resolved_epoch in self._cancelled_epochs:
                        break
                    pcm = _tts_speech_to_pcm_bytes(item.get("tts_speech"))
                    if pcm is None:
                        continue
                    is_final = bool(item.get("is_final", False)) and fragment_index == final_fragment_index
                    await asyncio.sleep(0)
                    yield pcm, int(self.sample_rate), bool(is_final)
            if resolved_epoch and resolved_epoch in self._cancelled_epochs:
                self._cancelled_epochs.discard(resolved_epoch)
        finally:
            self._active_queue_depth = max(0, self._active_queue_depth - 1)


def _iter_fragments(text: str | Iterable[str]) -> Iterable[str]:
    if isinstance(text, str):
        normalized = " ".join(str(text).strip().split())
        if normalized:
            yield normalized
        return
    for fragment in text:
        normalized = " ".join(str(fragment).strip().split())
        if normalized:
            yield normalized


def _resolve_native_stream_inference(model: Any):
    inference = getattr(model, "inference_bistream", None)
    if callable(inference):
        return inference
    return None


def _call_cosyvoice_native_stream(
    model: Any,
    text_input: str | Iterable[str],
    prompt_text: str,
    prompt_speech_path: str,
    **kwargs: object,
):
    if model is None:
        raise RuntimeError("CosyVoice model is not warm")
    inference = _resolve_native_stream_inference(model)
    if inference is None:
        raise RuntimeError("CosyVoice runtime does not expose native stream inference")
    signature = inspect.signature(inference)
    call_kwargs = dict(kwargs)
    if "text_frontend" in signature.parameters:
        call_kwargs["text_frontend"] = bool(call_kwargs.get("text_frontend", False))
    else:
        call_kwargs.pop("text_frontend", None)
    return inference(text_input, prompt_text, prompt_speech_path, **call_kwargs)


def _tts_speech_to_pcm_bytes(tts_speech: object) -> bytes | None:
    if tts_speech is None:
        return None
    array = np.asarray(tts_speech)
    if array.size == 0:
        return None
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, -1.0, 1.0)
        array = (array * 32767.0).astype("<i2")
    else:
        array = np.asarray(array, dtype="<i2")
    return np.ascontiguousarray(array.reshape(-1)).tobytes()


__all__ = ["CosyVoiceBiStreamSession", "TTSEngine"]
