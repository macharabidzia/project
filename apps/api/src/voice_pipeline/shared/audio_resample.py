from __future__ import annotations

import numpy as np
import soxr


class StreamingAudioResampler:
    def __init__(self, *, target_rate: int, quality: str = "LQ") -> None:
        if target_rate <= 0:
            raise ValueError("target_rate must be > 0")
        self._target_rate = int(target_rate)
        self._quality = str(quality)
        self._source_rate = 0
        self._stream: soxr.ResampleStream | None = None
        self._supports_stream_api = hasattr(soxr, "ResampleStream")

    def _reset_stream(self, source_rate: int) -> None:
        self._source_rate = int(source_rate)
        if self._supports_stream_api:
            self._stream = soxr.ResampleStream(
                self._source_rate,
                self._target_rate,
                num_channels=1,
                dtype="float32",
                quality=self._quality,
            )
        else:
            self._stream = None

    def resample(self, pcm: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate <= 0:
            raise ValueError("source_rate must be > 0")

        audio = np.ascontiguousarray(np.asarray(pcm, dtype=np.float32).reshape(-1))
        if audio.size == 0:
            return np.empty(0, dtype=np.float32)
        if int(source_rate) == self._target_rate:
            return audio

        if int(source_rate) != self._source_rate or self._stream is None:
            self._reset_stream(int(source_rate))

        if self._stream is not None:
            output = self._stream.resample_chunk(audio, last=False)
            return np.ascontiguousarray(np.asarray(output, dtype=np.float32).reshape(-1))

        output = soxr.resample(audio, int(source_rate), self._target_rate, quality=self._quality)
        return np.ascontiguousarray(np.asarray(output, dtype=np.float32).reshape(-1))

    def flush(self) -> np.ndarray:
        if self._stream is None:
            return np.empty(0, dtype=np.float32)
        output = self._stream.resample_chunk(np.empty(0, dtype=np.float32), last=True)
        return np.ascontiguousarray(np.asarray(output, dtype=np.float32).reshape(-1))


__all__ = ["StreamingAudioResampler"]
