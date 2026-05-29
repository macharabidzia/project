from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "apps" / "api" / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from voice_pipeline.gpu.vllm_worker.engine import VLLMEngine, VLLMEngineConfig
from voice_pipeline.runtime.admission_gate import AdmissionError
from voice_pipeline.runtime.bootstrap import _bind_cuda_device
from voice_pipeline.runtime.config import RuntimeConfig


def _validate_vllm_inputs(config: RuntimeConfig) -> None:
    model_path = Path(str(config.resolved_vllm_model_path()).strip())
    if not str(model_path):
        raise AdmissionError("vLLM model path is not configured")
    if not model_path.exists():
        raise AdmissionError(f"vLLM model path does not exist: {model_path}")
    cache_dir = Path(str(config.vllm_cache_dir or "").strip())
    if not str(cache_dir):
        raise AdmissionError("vLLM cache directory is not configured")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not cache_dir.is_dir():
        raise AdmissionError(f"vLLM cache directory is not a directory: {cache_dir}")


async def _run() -> int:
    config = RuntimeConfig.from_env()
    _validate_vllm_inputs(config)
    _bind_cuda_device(config.llm_device)

    engine = VLLMEngine(
        config.resolved_vllm_model_path(),
        config=VLLMEngineConfig(
            model_name=config.resolved_vllm_model_path(),
            model_path=config.resolved_vllm_model_path(),
            cache_dir=config.vllm_cache_dir,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_tokens=min(16, int(config.vllm_max_tokens)),
            temperature=float(config.vllm_temperature),
            top_p=float(config.vllm_top_p),
        ),
    )
    engine.warm(strict=True)
    engine.prewarm_prefix_cache("voice_pipeline_system_prefix", "real-vllm-smoke")

    first_token = ""
    async for token in engine.stream_tokens(
        "Respond with exactly: ok",
        cache_key="real-vllm-smoke",
        request_id="real-vllm-smoke",
        max_tokens=8,
    ):
        first_token = str(token)
        if first_token:
            break

    if not first_token:
        raise RuntimeError("no token emitted from vLLM stream probe")

    print("real-vllm-smoke: READY")
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except Exception as exc:
        print(f"real-vllm-smoke: FAILED ({exc})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
