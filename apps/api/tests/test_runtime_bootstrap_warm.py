from __future__ import annotations

from pathlib import Path

from voice_pipeline.runtime.bootstrap import bootstrap_runtime
from voice_pipeline.runtime.config import RuntimeConfig


def test_bootstrap_warms_workers_in_architecture_order(monkeypatch, tmp_path: Path) -> None:
    order: list[str] = []
    bound_devices: list[str] = []

    def _warm_asr(_asr, _config, _session_id):
        order.append("asr")
        return True

    def _warm_vllm(_vllm, _config):
        order.append("vllm")
        bound_devices.append("cuda:0")
        return True

    def _warm_tts(_tts, _config, _session_id):
        order.append("tts")
        bound_devices.append("cuda:1")
        return True

    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_asr_engine", _warm_asr)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_vllm_engine", _warm_vllm)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_tts_engine", _warm_tts)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap.hardware_admission_check", lambda config: None)

    config = RuntimeConfig(
        asr_model_path=str(tmp_path),
        vllm_model_path=str(tmp_path),
        vllm_cache_dir=str(tmp_path / "vllm-cache"),
        cosyvoice3_model_path=str(tmp_path),
        cosyvoice3_cache_dir=str(tmp_path / "cosy-cache"),
    )
    runtime = bootstrap_runtime(session_id="warm-order-test", config=config)

    assert order == ["asr", "vllm", "tts"]
    assert bound_devices == ["cuda:0", "cuda:1"]
    assert runtime.dry_run_report() == "CPU ASR, GPU0 vLLM, GPU1 CosyVoice3"
    assert runtime.topology.asr.device == "cpu"
    assert runtime.topology.vllm.device == "cuda:0"
    assert runtime.topology.tts.device == "cuda:1"
    assert runtime.warm_report.asr_warm is True
    assert runtime.warm_report.vllm_warm is True
    assert runtime.warm_report.tts_warm is True
    assert runtime.worker_status.kernel == "READY"


def test_bootstrap_shutdowns_vllm_after_warm_failure(monkeypatch, tmp_path: Path) -> None:
    shutdown_calls: list[str] = []

    monkeypatch.setattr("voice_pipeline.runtime.bootstrap.hardware_admission_check", lambda config: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_asr_engine", lambda asr, config, session_id: True)
    monkeypatch.setattr(
        "voice_pipeline.runtime.bootstrap._warm_vllm_engine",
        lambda vllm, config: (_ for _ in ()).throw(RuntimeError("vllm warm failed")),
    )
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_tts_engine", lambda tts, config, session_id: True)
    monkeypatch.setattr(
        "voice_pipeline.gpu.vllm_worker.engine.VLLMEngine.shutdown",
        lambda self, timeout=None: shutdown_calls.append("vllm"),
    )

    runtime = bootstrap_runtime(
        session_id="warm-failure-shutdown",
        config=RuntimeConfig(
            asr_model_path=str(tmp_path),
            vllm_model_path=str(tmp_path),
            vllm_cache_dir=str(tmp_path / "vllm-cache"),
            cosyvoice3_model_path=str(tmp_path),
            cosyvoice3_cache_dir=str(tmp_path / "cosy-cache"),
        ),
    )

    assert runtime.warm_report.vllm_warm is False
    assert runtime.worker_status.vllm == "FAILED"
    assert shutdown_calls == ["vllm"]
