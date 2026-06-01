from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from voice_pipeline.shared.audio_resample import StreamingAudioResampler
from voice_pipeline.shared.time import now_ns


@dataclass(frozen=True, slots=True)
class ASREvent:
    event_type: str
    text: str
    lineage_id: str
    emitted_at_ns: int = 0
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ASRRuntimeConfig:
    model_path: str = ""
    sample_rate: int = 16_000
    input_sample_rate: int = 16_000


class ASREngine:
    """CPU streaming ASR runtime backed by Vosk when available."""

    def __init__(self, *, config: ASRRuntimeConfig | None = None) -> None:
        self.config = config or ASRRuntimeConfig(
            model_path=os.getenv("VOSK_MODEL_PATH", ""),
            sample_rate=int(os.getenv("VOICE_PIPELINE_ASR_SAMPLE_RATE", "16000")),
            input_sample_rate=int(os.getenv("VOICE_PIPELINE_INPUT_SAMPLE_RATE", "16000")),
        )
        self._resampler = StreamingAudioResampler(target_rate=int(self.config.sample_rate))
        self._model: Any | None = None
        self._recognizer: Any | None = None
        self._warm = False
        self._session_lineage_id = ""
        self._partial_text = ""
        self._final_text = ""

    @property
    def is_warm(self) -> bool:
        return bool(self._warm and self._model is not None and self._recognizer is not None)

    @property
    def sample_rate(self) -> int:
        return int(self.config.sample_rate)

    def warm(self, *, model_path: str | None = None, strict: bool | None = None) -> None:
        resolved_model_path = str(model_path or self.config.model_path or os.getenv("VOSK_MODEL_PATH", "")).strip()
        strict_mode = True if strict is None else bool(strict)
        if not strict_mode:
            raise RuntimeError("asr_strict_mode_required")
        if not resolved_model_path:
            raise RuntimeError("vosk model path is required for strict ASR warm start")
        if not os.path.exists(resolved_model_path):
            raise RuntimeError(f"vosk model path does not exist: {resolved_model_path}")

        try:
            from vosk import Model, KaldiRecognizer
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("vosk runtime is unavailable") from exc

        model = Model(resolved_model_path)
        recognizer = KaldiRecognizer(model, int(self.config.sample_rate))
        if hasattr(recognizer, "SetWords"):
            recognizer.SetWords(True)
        self._model = model
        self._recognizer = recognizer
        self._warm = True

    def start_session(self, *, lineage_id: str = "") -> None:
        self._session_lineage_id = str(lineage_id or "").strip()
        self._partial_text = ""
        self._final_text = ""
        self._resampler = StreamingAudioResampler(target_rate=int(self.config.sample_rate))
        if self._model is not None:
            from vosk import KaldiRecognizer  # pragma: no cover - imported only when model is live

            self._recognizer = KaldiRecognizer(self._model, int(self.config.sample_rate))
            if hasattr(self._recognizer, "SetWords"):
                self._recognizer.SetWords(True)

    def _resample_input_audio(self, source_audio: np.ndarray) -> np.ndarray:
        source_rate = int(self.config.input_sample_rate)
        target_rate = int(self.config.sample_rate)
        if source_audio.size == 0:
            return np.empty(0, dtype=np.float32)
        if source_rate == target_rate:
            return np.ascontiguousarray(np.asarray(source_audio, dtype=np.float32).reshape(-1))

        return self._resampler.resample(source_audio, source_rate)

    def ingest_partial(self, *, text: str, lineage_id: str) -> ASREvent:
        self._partial_text = str(text).strip()
        self._session_lineage_id = str(lineage_id or self._session_lineage_id).strip()
        return ASREvent(
            event_type="ASRPartialReceived",
            text=self._partial_text,
            lineage_id=self._session_lineage_id,
            emitted_at_ns=now_ns(),
        )

    def ingest_final(self, *, text: str, lineage_id: str) -> ASREvent:
        self._final_text = str(text).strip()
        self._partial_text = ""
        self._session_lineage_id = str(lineage_id or self._session_lineage_id).strip()
        return ASREvent(
            event_type="ASRFinalReceived",
            text=self._final_text,
            lineage_id=self._session_lineage_id,
            emitted_at_ns=now_ns(),
        )

    def ingest_audio(self, pcm_chunk: bytes, *, lineage_id: str = "") -> tuple[ASREvent, ...]:
        """Process PCM16 mono audio and emit partial/final transcript events."""

        if not self.is_warm:
            raise RuntimeError("asr_streaming_engine_not_warm")

        self._session_lineage_id = str(lineage_id or self._session_lineage_id).strip()

        if self._recognizer is None:
            raise RuntimeError("vosk_streaming_recognizer_unavailable")

        source_audio = np.frombuffer(bytes(pcm_chunk), dtype=np.int16).astype(np.float32) / 32768.0
        resampled = self._resample_input_audio(source_audio)
        if resampled.size == 0:
            return ()
        resampled_pcm = np.clip(resampled * 32767.0, -32768.0, 32767.0).astype("<i2").tobytes()

        emitted: list[ASREvent] = []
        if self._recognizer.AcceptWaveform(resampled_pcm):
            payload = _safe_json_loads(self._recognizer.Result())
            text = str(payload.get("text", "")).strip()
            if text:
                emitted.append(self.ingest_final(text=text, lineage_id=self._session_lineage_id))
        else:
            payload = _safe_json_loads(self._recognizer.PartialResult())
            text = str(payload.get("partial", "")).strip()
            if text and text != self._partial_text:
                emitted.append(self.ingest_partial(text=text, lineage_id=self._session_lineage_id))
        return tuple(emitted)

    def finalize(self, *, lineage_id: str = "") -> ASREvent | None:
        if self._recognizer is None:
            text = self._partial_text
            if text:
                event = self.ingest_final(text=text, lineage_id=lineage_id or self._session_lineage_id)
                self._final_text = ""
                return event
            return None

        payload = _safe_json_loads(self._recognizer.FinalResult())
        text = str(payload.get("text", "")).strip()
        if not text:
            text = self._partial_text
        if not text:
            return None
        event = self.ingest_final(text=text, lineage_id=lineage_id or self._session_lineage_id)
        self._final_text = ""
        return event

    def shutdown(self) -> None:
        self._recognizer = None
        self._model = None
        self._warm = False
        self._session_lineage_id = ""
        self._partial_text = ""
        self._final_text = ""
        self._resampler = StreamingAudioResampler(target_rate=int(self.config.sample_rate))


def _safe_json_loads(data: str) -> dict[str, object]:
    try:
        parsed = json.loads(data)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = ["ASREngine", "ASREvent", "ASRRuntimeConfig"]
