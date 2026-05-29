from __future__ import annotations

import asyncio

import numpy as np
import pytest

from voice_pipeline.gpu.tts_worker.engine import TTSEngine


class _FakeCosyVoiceModel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str, str, dict[str, object]]] = []

    def inference_bistream(self, text_input, prompt_text, prompt_speech_path, **kwargs):
        self.calls.append((text_input, prompt_text, prompt_speech_path, dict(kwargs)))
        yield {"tts_speech": np.array([0.0, 0.1], dtype=np.float32), "is_final": True}


async def _collect_frames(engine: TTSEngine, text) -> list[tuple[bytes, int, bool]]:
    frames: list[tuple[bytes, int, bool]] = []
    async for frame in engine.stream_pcm(text, epoch_id="session:epoch:1"):
        frames.append(frame)
    return frames


def test_tts_stream_pcm_passes_committed_fragment_without_local_token_chunking() -> None:
    model = _FakeCosyVoiceModel()
    engine = TTSEngine(
        "D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512",
    )
    engine._model = model
    engine.start_persistent_session(epoch_id="session:epoch:1")

    frames = asyncio.run(_collect_frames(engine, "alpha beta gamma"))

    assert len(frames) == 1
    assert model.calls[0][0] == "alpha beta gamma"
    assert bool(model.calls[0][3].get("stream", False)) is True


def test_tts_stream_pcm_keeps_fragment_boundaries_from_kernel() -> None:
    model = _FakeCosyVoiceModel()
    engine = TTSEngine(
        "D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512",
    )
    engine._model = model
    engine.start_persistent_session(epoch_id="session:epoch:1")

    frames = asyncio.run(_collect_frames(engine, ["first fragment", "second fragment"]))

    assert len(frames) == 2
    assert [call[0] for call in model.calls] == ["first fragment", "second fragment"]


def test_tts_stream_pcm_has_no_non_streaming_toggle_surface() -> None:
    model = _FakeCosyVoiceModel()
    engine = TTSEngine(
        "D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512",
    )
    engine._model = model
    engine.start_persistent_session(epoch_id="session:epoch:1")

    with pytest.raises(TypeError):
        _ = engine.stream_pcm("alpha beta gamma", epoch_id="session:epoch:1", stream=False)  # type: ignore[call-arg]
