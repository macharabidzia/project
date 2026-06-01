from __future__ import annotations

import sys
from pathlib import Path

from runtime_entrypoint import ensure_runtime_library_path, ensure_runtime_python


REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "apps" / "api" / "src"
COSYVOICE_RUNTIME = REPO_ROOT / ".models" / "CosyVoice-runtime"
MATCHA_TTS = COSYVOICE_RUNTIME / "third_party" / "Matcha-TTS"

ensure_runtime_python()
ensure_runtime_library_path()
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))
if COSYVOICE_RUNTIME.exists() and str(COSYVOICE_RUNTIME) not in sys.path:
    sys.path.insert(0, str(COSYVOICE_RUNTIME))
if MATCHA_TTS.exists() and str(MATCHA_TTS) not in sys.path:
    sys.path.insert(0, str(MATCHA_TTS))

from voice_pipeline.runtime.bootstrap import bootstrap_runtime
from voice_pipeline.runtime.config import RuntimeConfig


def main() -> int:
    try:
        runtime = bootstrap_runtime(session_id="warmup-check", config=RuntimeConfig.from_env())
    except Exception as exc:
        print(f"warmup-check: FAILED ({exc})")
        return 1

    if not (runtime.warm_report.asr_warm and runtime.warm_report.vllm_warm and runtime.warm_report.tts_warm):
        print(
            "warmup-check: FAILED",
            {
                "asr": runtime.worker_status.asr,
                "asr_reason": runtime.worker_failure_reason.asr,
                "vllm": runtime.worker_status.vllm,
                "vllm_reason": runtime.worker_failure_reason.vllm,
                "tts": runtime.worker_status.tts,
                "tts_reason": runtime.worker_failure_reason.tts,
                "kernel": runtime.worker_status.kernel,
                "kernel_reason": runtime.worker_failure_reason.kernel,
            },
        )
        return 1

    print(
        "warmup-check:",
        {
            "asr": runtime.worker_status.asr,
            "vllm": runtime.worker_status.vllm,
            "tts": runtime.worker_status.tts,
            "cache": runtime.model_cache_identity.get("model_cache_hash", ""),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
