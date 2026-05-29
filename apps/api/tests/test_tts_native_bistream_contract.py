from __future__ import annotations

import asyncio
import pytest

from voice_pipeline.gpu.tts_worker.engine import TTSEngine, _resolve_native_stream_inference


class _NoNativeStreamModel:
    pass


def test_native_bistream_resolver_returns_none_when_api_missing() -> None:
    assert _resolve_native_stream_inference(_NoNativeStreamModel()) is None


def test_stream_pcm_fails_fast_when_native_bistream_api_is_unavailable() -> None:
    engine = TTSEngine("D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512")
    engine._model = _NoNativeStreamModel()
    engine._session.warmed = True
    engine._warm = True

    async def _run() -> None:
        async for _ in engine.stream_pcm("hello", epoch_id="tts:epoch:1"):
            pass

    with pytest.raises(RuntimeError, match="native stream inference"):
        asyncio.run(_run())
