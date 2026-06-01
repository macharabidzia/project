from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from voice_pipeline.gpu.tts_worker.engine import TTSEngine


class TTSAudioStreamer:
    def __init__(self, engine: TTSEngine) -> None:
        self._engine = engine

    async def stream(
        self,
        text: str | list[str] | tuple[str, ...] | AsyncIterable[str],
        *,
        request_id: str = "",
        epoch_id: str = "",
        prompt_text: str = "",
        prompt_speech_path: str = "",
    ) -> AsyncIterator[tuple[bytes, int, bool]]:
        async for frame in self._engine.stream_pcm(
            text,
            request_id=request_id,
            epoch_id=epoch_id,
            prompt_text=prompt_text,
            prompt_speech_path=prompt_speech_path,
        ):
            yield frame


__all__ = ["TTSAudioStreamer"]
