from __future__ import annotations

import sys
from pathlib import Path

from runtime_entrypoint import ensure_runtime_library_path, ensure_runtime_python

REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "apps" / "api" / "src"

ensure_runtime_python()
ensure_runtime_library_path()
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from voice_pipeline.runtime.admission_gate import AdmissionError
from voice_pipeline.runtime.config import RuntimeConfig
from voice_pipeline.stt.asr_engine import ASREngine, ASRRuntimeConfig


def _validate_asr_inputs(config: RuntimeConfig) -> None:
    if str(config.asr_device).strip().lower() != "cpu":
        raise AdmissionError(f"ASR device must be cpu, got {config.asr_device}")
    model_path = Path(str(config.asr_model_path or "").strip())
    if not str(model_path):
        raise AdmissionError("vosk model path is not configured")
    if not model_path.exists():
        raise AdmissionError(f"vosk model path does not exist: {model_path}")


def main() -> int:
    try:
        config = RuntimeConfig.from_env()
        _validate_asr_inputs(config)
        engine = ASREngine(
            config=ASRRuntimeConfig(
                model_path=config.asr_model_path,
                sample_rate=int(config.asr_sample_rate),
                input_sample_rate=int(config.input_sample_rate),
            )
        )
        engine.warm(strict=True)
        engine.start_session(lineage_id="real-vosk-smoke:epoch:1")
        silence_frame = b"\x00\x00" * max(1, int(config.input_sample_rate * config.frame_ms / 1000))
        _ = engine.ingest_audio(silence_frame, lineage_id="real-vosk-smoke:epoch:1")
        print("real-vosk-smoke: READY")
        return 0
    except Exception as exc:
        print(f"real-vosk-smoke: FAILED ({exc})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
