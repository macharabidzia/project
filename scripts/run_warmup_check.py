from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "apps" / "api" / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from voice_pipeline.runtime.bootstrap import bootstrap_runtime
from voice_pipeline.runtime.config import RuntimeConfig


def main() -> int:
    try:
        runtime = bootstrap_runtime(session_id="warmup-check", config=RuntimeConfig.from_env())
    except Exception as exc:
        print(f"warmup-check: FAILED ({exc})")
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

