from __future__ import annotations

import asyncio
import time
import threading

import numpy as np
import pytest
import torch

from voice_pipeline.gpu.tts_worker.engine import (
    TTSEngine,
    _streaming_text_fragments,
)


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

    assert len(frames) == 2
    assert frames[0][0]
    assert frames[0][2] is False
    assert frames[1][0] == b""
    assert frames[1][2] is True
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

    assert len(frames) == 4
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


def test_tts_stream_pcm_keeps_short_final_text_on_native_bistream_path() -> None:
    model = _FakeCosyVoiceModel()
    engine = TTSEngine(
        "D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512",
    )
    engine._model = model
    engine.start_persistent_session(epoch_id="session:epoch:1")

    frames = asyncio.run(_collect_frames(engine, "hello there"))

    assert len(frames) == 2
    assert frames[0][0]
    assert frames[1][0] == b""
    assert model.calls[0][0] == "hello there"
    assert bool(model.calls[0][3].get("stream", False)) is True


def test_streaming_text_fragments_coalesces_tiny_generator_updates() -> None:
    def _fragments():
        yield "That's"
        yield "a bit unclear."
        yield "Are you referring to something specific?"

    fragments = list(
        _streaming_text_fragments(
            _fragments(),
            add_english_lang_tag=True,
            min_emit_tokens=4,
        )
    )

    assert fragments == [
        "<|en|>That's a bit unclear.",
        "Are you referring to something specific?",
    ]


def test_tts_stream_pcm_drops_weak_resumed_tail_after_long_generator_gap() -> None:
    class _FakeGapTailModel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str, str, dict[str, object]]] = []

        def inference_bistream(self, text_input, prompt_text, prompt_speech_path, **kwargs):
            self.calls.append((text_input, prompt_text, prompt_speech_path, dict(kwargs)))
            strong = np.array([0.0, 3000.0, -3000.0, 2000.0], dtype=np.float32)
            weak = np.array([0.0, 200.0, -200.0, 120.0], dtype=np.float32)
            yield {"tts_speech": strong, "is_final": False}
            time.sleep(0.5)
            yield {"tts_speech": weak, "is_final": False}

    def _generator():
        yield "hi"
        yield "there"

    model = _FakeGapTailModel()
    engine = TTSEngine("D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512", max_lookahead_ms=120)
    engine._model = model
    engine.start_persistent_session(epoch_id="session:epoch:1")

    frames = asyncio.run(_collect_frames(engine, _generator()))

    assert len(frames) == 2
    assert frames[0][0]
    assert frames[0][2] is False
    assert frames[1][0] == b""
    assert frames[1][2] is True


def test_tts_stream_pcm_drops_weak_resumed_tail_after_long_string_gap() -> None:
    class _FakeGapTailModel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str, str, dict[str, object]]] = []

        def inference_bistream(self, text_input, prompt_text, prompt_speech_path, **kwargs):
            self.calls.append((text_input, prompt_text, prompt_speech_path, dict(kwargs)))
            strong = np.array([0.0, 3000.0, -3000.0, 2000.0], dtype=np.float32)
            weak = np.array([0.0, 200.0, -200.0, 120.0], dtype=np.float32)
            yield {"tts_speech": strong, "is_final": False}
            time.sleep(0.5)
            yield {"tts_speech": weak, "is_final": False}

    model = _FakeGapTailModel()
    engine = TTSEngine("D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512", max_lookahead_ms=120)
    engine._model = model
    engine.start_persistent_session(epoch_id="session:epoch:1")

    frames = asyncio.run(_collect_frames(engine, "hi there"))

    assert len(frames) == 2
    assert frames[0][0]
    assert frames[0][2] is False
    assert frames[1][0] == b""
    assert frames[1][2] is True


def test_tts_stream_pcm_drops_short_reply_resumed_tail_with_higher_peak() -> None:
    class _FakeGapTailModel:
        def inference_bistream(self, text_input, prompt_text, prompt_speech_path, **kwargs):
            strong = np.array([0.0, 0.20, -0.20, 0.12], dtype=np.float32)
            weak_but_peaky = np.concatenate(
                [
                    np.zeros(32, dtype=np.float32),
                    np.array([0.18, -0.18, 0.04, -0.04], dtype=np.float32),
                    np.zeros(32, dtype=np.float32),
                ]
            )
            yield {"tts_speech": strong, "is_final": False}
            time.sleep(0.5)
            yield {"tts_speech": weak_but_peaky, "is_final": False}

    model = _FakeGapTailModel()
    engine = TTSEngine("D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512", max_lookahead_ms=120)
    engine._model = model
    engine.start_persistent_session(epoch_id="session:epoch:1")

    frames = asyncio.run(_collect_frames(engine, "hi there"))

    assert len(frames) == 2
    assert frames[0][0]
    assert frames[0][2] is False
    assert frames[1][0] == b""
    assert frames[1][2] is True


def test_native_streaming_scheduler_uses_second_hop_override_before_steady_state() -> None:
    import sys

    cosyvoice_root = "/workspace/project/.models/CosyVoice-runtime"
    if cosyvoice_root not in sys.path:
        sys.path.insert(0, cosyvoice_root)
    from cosyvoice.cli.model import CosyVoice3Model

    class _Flow:
        pre_lookahead_len = 0

    model = CosyVoice3Model.__new__(CosyVoice3Model)
    model.lock = threading.Lock()
    model.tts_speech_token_dict = {}
    model.llm_end_dict = {}
    model.hift_cache_dict = {}
    model.flow = _Flow()
    model.token_hop_len = 5
    model.first_stream_token_hop_len = 5
    model.second_stream_token_hop_len = 4
    model.token_max_hop_len = 8
    model.stream_scale_factor = 2
    model.stream_poll_interval_s = 0.0

    observed_token_offsets: list[int] = []

    def _token2wav(*, token_offset, **_kwargs):
        observed_token_offsets.append(int(token_offset))
        return torch.zeros(1, 1)

    def _llm_job(_text, _prompt_text, _llm_prompt_speech_token, _llm_embedding, this_uuid):
        model.tts_speech_token_dict[this_uuid] = list(range(20))
        model.llm_end_dict[this_uuid] = True

    model.token2wav = _token2wav  # type: ignore[assignment]
    model.llm_job = _llm_job  # type: ignore[assignment]

    outputs = list(
        model.tts(
            text=torch.zeros(1, 0, dtype=torch.int32),
            flow_embedding=torch.zeros(0, 192),
            llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80),
            source_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            stream=True,
        )
    )

    assert len(outputs) == 4
    assert observed_token_offsets == [0, 5, 9, 17]
