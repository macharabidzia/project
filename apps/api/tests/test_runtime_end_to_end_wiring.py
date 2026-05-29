from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from voice_pipeline.runtime.bootstrap import bootstrap_runtime
from voice_pipeline.runtime.config import RuntimeConfig
from voice_pipeline.stt.asr_engine import ASREvent
from voice_pipeline.transport.pcm_clock import PCMFrame


class _FakeVLLMStreamer:
    async def stream(self, prompt: str, *, cache_key: str = "", request_id: str = ""):
        _ = (prompt, cache_key, request_id)
        yield "hello"
        yield "world"


class _FakeTTSStreamer:
    def __init__(self, *, sample_rate: int = 24_000) -> None:
        self.sample_rate = int(sample_rate)
        self.chunk = b"\xff\x7f" * int(self.sample_rate * 20 / 1000)

    async def stream(self, text: str, *, epoch_id: str = "", prompt_text: str = "", prompt_speech_path: str = ""):
        _ = (text, epoch_id, prompt_text, prompt_speech_path)
        yield self.chunk, int(self.sample_rate), True


def _build_runtime(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap.hardware_admission_check", lambda config: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._bind_cuda_device", lambda device_name: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_asr_engine", lambda asr, config, session_id: True)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_vllm_engine", lambda vllm, config: True)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_tts_engine", lambda tts, config, session_id: True)
    runtime = bootstrap_runtime(
        session_id="runtime-e2e-wiring",
        config=RuntimeConfig(
            asr_model_path=str(tmp_path),
            vllm_model_path=str(tmp_path),
            vllm_cache_dir=str(tmp_path / "vllm-cache"),
            cosyvoice3_model_path=str(tmp_path),
            cosyvoice3_cache_dir=str(tmp_path / "cosy-cache"),
        ),
    )
    runtime.worker_status.transport = "READY"
    return runtime


def test_runtime_process_pcm_frame_runs_authority_chain_to_pcm(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.asr.ingest_audio = lambda _pcm, *, lineage_id="": (
        ASREvent(
            event_type="ASRFinalReceived",
            text="hello world",
            lineage_id=str(lineage_id or runtime.kernel.current_lease().epoch_id),
            emitted_at_ns=123,
        ),
    )
    runtime._vllm_streamer = _FakeVLLMStreamer()
    runtime._tts_streamer = _FakeTTSStreamer()

    frames = asyncio.run(runtime.process_pcm_frame(b"\x00\x01" * 960))
    frame_bytes = runtime._output_frame_bytes()

    assert frames
    assert any(len(frame.pcm) == frame_bytes for frame in frames)
    event_types = [record["type"] for record in runtime.event_log.as_records() if "type" in record]
    assert "ASRFinalReceived" in event_types
    assert "VLLMChunkReceived" in event_types
    assert "VLLMCompleted" in event_types
    assert "TTSChunkReceived" in event_types
    assert "TTSCompleted" in event_types
    assert runtime.rings.kernel_stream_ring.depth == 0

    sent: list[tuple[bytes, int]] = []

    async def _send(pcm: bytes, sample_rate: int) -> None:
        sent.append((bytes(pcm), int(sample_rate)))

    asyncio.run(runtime.send_pcm_once(_send))
    assert sent
    assert len(sent[0][0]) == frame_bytes
    assert sent[0][1] == int(runtime.config.output_sample_rate)
    assert any(byte != 0 for byte in sent[0][0])


def test_barge_in_rotates_epoch_and_suppresses_stale_pcm(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    initial_epoch = runtime.kernel.current_lease().epoch_id
    initial_output_version = int(runtime.kernel.state.output.version)
    runtime.pcm_clock.enqueue(
        PCMFrame(
            pcm=b"\x01\x02" * (runtime._output_frame_bytes() // 2),
            sample_rate=int(runtime.config.output_sample_rate),
            epoch_id=initial_epoch,
            output_version=initial_output_version,
        )
    )
    runtime.kernel._state = replace(runtime.kernel.state, phase="playing", active_tts_request_id="tts-active")
    runtime.asr.ingest_audio = lambda _pcm, *, lineage_id="": (
        ASREvent(
            event_type="ASRPartialReceived",
            text="barge now",
            lineage_id=str(lineage_id or runtime.kernel.current_lease().epoch_id),
            emitted_at_ns=456,
        ),
    )

    _ = asyncio.run(runtime.process_pcm_frame(b"\x00\x01" * 960))
    assert runtime.kernel.state.output.version > initial_output_version
    assert runtime.kernel.current_lease().epoch_id != initial_epoch
    assert runtime.rings.kernel_stream_ring.depth == 0

    sent: list[tuple[bytes, int]] = []

    async def _send(pcm: bytes, sample_rate: int) -> None:
        sent.append((bytes(pcm), int(sample_rate)))

    asyncio.run(runtime.send_pcm_once(_send))
    assert sent
    assert sent[0][0] == (b"\x00" * runtime._output_frame_bytes())
    assert runtime.pcm_clock.dropped_stale_frames >= 1
