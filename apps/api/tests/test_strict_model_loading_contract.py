from __future__ import annotations

from pathlib import Path

import pytest

from voice_pipeline.gpu.tts_worker.engine import TTSEngine
from voice_pipeline.gpu.vllm_worker.engine import VLLMEngine, VLLMEngineConfig
from voice_pipeline.stt.asr_engine import ASREngine, ASRRuntimeConfig


def test_asr_warm_requires_strict_mode() -> None:
    engine = ASREngine(config=ASRRuntimeConfig(model_path=""))
    with pytest.raises(RuntimeError, match="asr_strict_mode_required"):
        engine.warm(strict=False)


def test_asr_warm_requires_model_path_when_strict(monkeypatch) -> None:
    monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
    engine = ASREngine(config=ASRRuntimeConfig(model_path=""))
    with pytest.raises(RuntimeError, match="vosk model path is required"):
        engine.warm(strict=True)


def test_asr_warm_requires_existing_model_path_when_strict(tmp_path: Path) -> None:
    missing_model_path = tmp_path / "missing-vosk-model"
    engine = ASREngine(config=ASRRuntimeConfig(model_path=str(missing_model_path)))
    with pytest.raises(RuntimeError, match="vosk model path does not exist"):
        engine.warm(strict=True)


def test_vllm_warm_requires_strict_mode() -> None:
    engine = VLLMEngine(
        model_name="",
        config=VLLMEngineConfig(model_name="", model_path=""),
    )
    with pytest.raises(RuntimeError, match="vllm_strict_mode_required"):
        engine.warm(strict=False)


def test_vllm_warm_requires_model_name_when_strict() -> None:
    engine = VLLMEngine(
        model_name="",
        config=VLLMEngineConfig(model_name="", model_path=""),
    )
    with pytest.raises(RuntimeError, match="vllm model name is required"):
        engine.warm(strict=True)


def test_vllm_warm_requires_existing_model_path_when_strict(tmp_path: Path) -> None:
    missing_model_path = tmp_path / "missing-vllm-model"
    engine = VLLMEngine(
        model_name=str(missing_model_path),
        config=VLLMEngineConfig(
            model_name=str(missing_model_path),
            model_path=str(missing_model_path),
        ),
    )
    with pytest.raises(RuntimeError, match="vllm model path does not exist"):
        engine.warm(strict=True)


def test_tts_warm_requires_strict_mode() -> None:
    engine = TTSEngine(model_name="")
    with pytest.raises(RuntimeError, match="tts_strict_mode_required"):
        engine.warm(strict=False)


def test_tts_warm_requires_existing_model_dir_when_strict(tmp_path: Path) -> None:
    missing_model_dir = tmp_path / "missing-cosyvoice3-model"
    engine = TTSEngine(model_name=str(missing_model_dir))
    with pytest.raises(RuntimeError, match="cosyvoice model_dir does not exist"):
        engine.warm(strict=True)


def test_strict_loading_toggle_surface_removed() -> None:
    with pytest.raises(TypeError):
        _ = ASRRuntimeConfig(strict_model_loading=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        _ = VLLMEngineConfig(strict_model_loading=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        _ = TTSEngine(model_name="", strict_model_loading=True)  # type: ignore[call-arg]
