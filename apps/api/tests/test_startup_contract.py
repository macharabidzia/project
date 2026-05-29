from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from voice_pipeline.runtime.admission_gate import AdmissionError, _require_path
from voice_pipeline.runtime.bootstrap import bootstrap_runtime
from voice_pipeline.runtime.config import RuntimeConfig


def test_runtime_config_reads_explicit_model_and_cache_keys(monkeypatch) -> None:
    monkeypatch.setenv("VOSK_MODEL_PATH", "/tmp/vosk")
    monkeypatch.setenv("VLLM_MODEL_PATH", "/tmp/vllm-model")
    monkeypatch.setenv("VLLM_CACHE_DIR", "/tmp/vllm-cache")
    monkeypatch.setenv("COSYVOICE3_MODEL_PATH", "/tmp/cosy-model")
    monkeypatch.setenv("COSYVOICE3_CACHE_DIR", "/tmp/cosy-cache")
    monkeypatch.setenv("COSYVOICE3_SPEAKER_PATH", "/tmp/speaker.wav")
    cfg = RuntimeConfig.from_env()

    assert cfg.asr_model_path == "/tmp/vosk"
    assert cfg.resolved_vllm_model_path() == "/tmp/vllm-model"
    assert cfg.vllm_cache_dir == "/tmp/vllm-cache"
    assert cfg.resolved_cosyvoice3_model_path() == "/tmp/cosy-model"
    assert cfg.cosyvoice3_cache_dir == "/tmp/cosy-cache"
    assert cfg.cosyvoice3_speaker_path == "/tmp/speaker.wav"


def test_admission_rejects_remote_model_paths() -> None:
    with pytest.raises(AdmissionError, match="must be local/offline"):
        _require_path("vllm model path", "https://example.com/model")


def test_runtime_rejects_live_audio_before_transport_ready(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap.hardware_admission_check", lambda config: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._bind_cuda_device", lambda device_name: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_asr_engine", lambda asr, config, session_id: True)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_vllm_engine", lambda vllm, config: True)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_tts_engine", lambda tts, config, session_id: True)

    runtime = bootstrap_runtime(
        session_id="startup-contract",
        config=RuntimeConfig(
            asr_model_path=str(tmp_path),
            vllm_model_path=str(tmp_path),
            vllm_cache_dir=str(tmp_path / "vllm-cache"),
            cosyvoice3_model_path=str(tmp_path),
            cosyvoice3_cache_dir=str(tmp_path / "cosy-cache"),
        ),
    )

    assert runtime.model_cache_identity["model_cache_hash"]
    with pytest.raises(RuntimeError, match="runtime_not_ready_for_live_audio"):
        asyncio.run(runtime.process_pcm_frame(b"\x00\x00" * 320))


def test_runtime_config_removes_non_strict_warmup_toggles() -> None:
    with pytest.raises(TypeError):
        _ = RuntimeConfig(warm_strict=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _ = RuntimeConfig(warmup_required=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _ = RuntimeConfig(cosyvoice_stream=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _ = RuntimeConfig(asr_input_sample_rate=16_000)  # type: ignore[arg-type]
