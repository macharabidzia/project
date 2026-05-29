from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from voice_pipeline.runtime.bootstrap import bootstrap_runtime
from voice_pipeline.runtime.config import RuntimeConfig
from voice_pipeline.stt.asr_engine import ASREvent


def _build_runtime(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap.hardware_admission_check", lambda config: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._bind_cuda_device", lambda device_name: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_asr_engine", lambda asr, config, session_id: True)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_vllm_engine", lambda vllm, config: True)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_tts_engine", lambda tts, config, session_id: True)
    runtime = bootstrap_runtime(
        session_id="barge-runtime",
        config=RuntimeConfig(
            asr_model_path=str(tmp_path),
            vllm_model_path=str(tmp_path),
            vllm_cache_dir=str(tmp_path / "vllm-cache"),
            cosyvoice3_model_path=str(tmp_path),
            cosyvoice3_cache_dir=str(tmp_path / "cosy-cache"),
        ),
    )
    asyncio.run(runtime.start())
    runtime.worker_status.transport = "READY"
    return runtime


def test_asr_partial_during_tts_enqueues_soft_interrupt(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.kernel._state = replace(runtime.kernel.state, phase="playing", active_tts_request_id="tts-1")
    runtime.asr.ingest_audio = lambda _pcm, *, lineage_id="": (
        ASREvent(
            event_type="ASRPartialReceived",
            text="hello",
            lineage_id=str(lineage_id or runtime.kernel.current_lease().epoch_id),
            emitted_at_ns=1,
        ),
    )
    asyncio.run(runtime.process_pcm_frame(b"\x00\x01" * 960))
    asyncio.run(runtime.stop())

    reasons = [str(event.payload.get("reason", "")) for event in runtime.kernel.event_log if event.event_type == "InterruptRequested"]
    assert "SOFT_PRE_INTERRUPT" in reasons


def test_asr_final_during_generation_enqueues_hard_interrupt(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.kernel._state = replace(runtime.kernel.state, phase="generating", active_vllm_request_id="vllm-1")
    runtime.asr.ingest_audio = lambda _pcm, *, lineage_id="": (
        ASREvent(
            event_type="ASRFinalReceived",
            text="interrupt now",
            lineage_id=str(lineage_id or runtime.kernel.current_lease().epoch_id),
            emitted_at_ns=2,
        ),
    )
    asyncio.run(runtime.process_pcm_frame(b"\x00\x01" * 960))
    asyncio.run(runtime.stop())

    reasons = [str(event.payload.get("reason", "")) for event in runtime.kernel.event_log if event.event_type == "CancelRequested"]
    assert "HARD_INTERRUPT" in reasons
