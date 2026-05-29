from __future__ import annotations

from pathlib import Path

from voice_pipeline.runtime.bootstrap import bootstrap_runtime
from voice_pipeline.runtime.config import RuntimeConfig


def test_runtime_bootstrap_returns_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap.hardware_admission_check", lambda config: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._bind_cuda_device", lambda device_name: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_asr_engine", lambda asr, config, session_id: True)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_vllm_engine", lambda vllm, config: True)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_tts_engine", lambda tts, config, session_id: True)

    runtime = bootstrap_runtime(
        session_id="integration-speech-runtime",
        config=RuntimeConfig(
            asr_model_path=str(tmp_path),
            vllm_model_path=str(tmp_path),
            vllm_cache_dir=str(tmp_path / "vllm-cache"),
            cosyvoice3_model_path=str(tmp_path),
            cosyvoice3_cache_dir=str(tmp_path / "cosy-cache"),
        ),
    )

    assert hasattr(runtime, "tick")
    assert runtime.kernel is not None
