from __future__ import annotations

import asyncio
from pathlib import Path

from voice_pipeline.runtime.bootstrap import bootstrap_runtime
from voice_pipeline.runtime.config import RuntimeConfig


def _build_runtime(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap.hardware_admission_check", lambda config: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._bind_cuda_device", lambda device_name: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_asr_engine", lambda asr, config, session_id: True)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_vllm_engine", lambda vllm, config: True)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_tts_engine", lambda tts, config, session_id: True)
    return bootstrap_runtime(
        session_id="tts-frame-align",
        config=RuntimeConfig(
            asr_model_path=str(tmp_path),
            vllm_model_path=str(tmp_path),
            vllm_cache_dir=str(tmp_path / "vllm-cache"),
            cosyvoice3_model_path=str(tmp_path),
            cosyvoice3_cache_dir=str(tmp_path / "cosy-cache"),
        ),
    )


def test_runtime_chunks_tts_output_to_exact_20ms_frames(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    frame_bytes = runtime._output_frame_bytes()

    payload = b"\x01\x02" * ((frame_bytes + (frame_bytes // 2)) // 2)
    frames = runtime._chunk_output_pcm(
        payload,
        epoch_id="session:epoch:1",
        output_version=1,
        flush=False,
    )

    assert len(frames) == 1
    assert len(frames[0]) == frame_bytes
    assert len(runtime._tts_frame_carry) == frame_bytes // 2

    flushed = runtime._chunk_output_pcm(
        b"",
        epoch_id="session:epoch:1",
        output_version=1,
        flush=True,
    )
    assert len(flushed) == 1
    assert len(flushed[0]) == frame_bytes
    assert runtime._tts_frame_carry == b""


def test_runtime_pcm_clock_emits_silence_frame_at_output_cadence(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    sent: list[tuple[int, int]] = []

    async def _send(pcm: bytes, sample_rate: int) -> None:
        sent.append((len(pcm), int(sample_rate)))

    asyncio.run(runtime.send_pcm_once(_send))

    assert sent == [(runtime._output_frame_bytes(), int(runtime.config.output_sample_rate))]
