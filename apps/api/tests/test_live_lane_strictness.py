from __future__ import annotations

import asyncio

import pytest

from voice_pipeline.gpu.tts_worker.engine import TTSEngine
from voice_pipeline.gpu.vllm_worker.engine import VLLMEngine, VLLMEngineConfig
from voice_pipeline.stt.asr_engine import ASREngine, ASRRuntimeConfig


def test_asr_ingest_audio_requires_warm_engine() -> None:
    engine = ASREngine(config=ASRRuntimeConfig(model_path=""))
    with pytest.raises(RuntimeError, match="asr_streaming_engine_not_warm"):
        engine.ingest_audio(b"\x00\x00")


def test_tts_start_persistent_session_requires_warm_model() -> None:
    engine = TTSEngine(
        "D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512",
    )
    with pytest.raises(RuntimeError, match="cosyvoice_native_bistream_unavailable"):
        engine.start_persistent_session(epoch_id="session:epoch:1")


def test_tts_stream_pcm_requires_warm_model() -> None:
    engine = TTSEngine(
        "D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512",
    )

    async def _run() -> None:
        async for _ in engine.stream_pcm("alpha beta gamma", epoch_id="session:epoch:1"):
            pass

    with pytest.raises(RuntimeError, match="tts_streaming_engine_not_warm"):
        asyncio.run(_run())


def test_vllm_stream_tokens_requires_authoritative_request_id() -> None:
    engine = VLLMEngine(
        model_name="D:/models/vllm/Qwen3-8B",
        config=VLLMEngineConfig(
            model_name="D:/models/vllm/Qwen3-8B",
            model_path="D:/models/vllm/Qwen3-8B",
        ),
    )

    async def _run() -> None:
        async for _ in engine.stream_tokens("hello world"):
            pass

    with pytest.raises(RuntimeError, match="vllm_request_id_required"):
        asyncio.run(_run())
