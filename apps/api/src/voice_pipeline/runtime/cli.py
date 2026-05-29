from __future__ import annotations

import argparse
import os
from pathlib import Path
from uuid import uuid4

from voice_pipeline.runtime.config import RuntimeConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_env_file() -> Path:
    return _repo_root() / ".env.voice_pipeline"


def _configure_runtime_env(*, env_file: str) -> None:
    if str(env_file).strip():
        os.environ["VOICE_PIPELINE_ENV_FILE"] = str(Path(env_file).expanduser().resolve())
        return
    if not str(os.getenv("VOICE_PIPELINE_ENV_FILE", "")).strip():
        os.environ["VOICE_PIPELINE_ENV_FILE"] = str(_default_env_file())


def _assert_runtime_contract(config: RuntimeConfig) -> None:
    if str(config.asr_device).strip().lower() != "cpu":
        raise RuntimeError("contract_violation: asr device must be cpu")
    if str(config.llm_device).strip().lower() != "cuda:0":
        raise RuntimeError("contract_violation: vllm device must be cuda:0")
    if str(config.tts_device).strip().lower() != "cuda:1":
        raise RuntimeError("contract_violation: tts device must be cuda:1")
    if int(config.frame_ms) != 20:
        raise RuntimeError("contract_violation: frame_ms must be 20")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-process Voice OS runtime.")
    parser.add_argument("--session-id", default=f"voice-runtime-{uuid4().hex[:8]}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env-file", default="", help="Optional explicit runtime env file path.")
    parser.add_argument("--dry-run", action="store_true", help="Print CPU/GPU topology and exit.")
    parser.add_argument(
        "--reload-dir",
        action="append",
        default=[],
        help="Directory to watch for backend reload (may be specified multiple times).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _configure_runtime_env(env_file=str(args.env_file))
    os.environ["VOICE_PIPELINE_SESSION_ID"] = str(args.session_id)
    if args.dry_run:
        config = RuntimeConfig.from_env()
        _assert_runtime_contract(config)
        print("CPU ASR, GPU0 vLLM, GPU1 CosyVoice3")
        return 0
    default_reload_dir = Path(__file__).resolve().parents[1]
    reload_dirs = [str(Path(item).resolve()) for item in args.reload_dir if str(item).strip()]
    if not reload_dirs:
        reload_dirs = [str(default_reload_dir)]
    import uvicorn

    uvicorn.run(
        "voice_pipeline.runtime.server:create_app",
        host=str(args.host),
        port=int(args.port),
        factory=True,
        reload=True,
        reload_dirs=reload_dirs,
    )
    return 0


__all__ = ["main"]
