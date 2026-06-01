from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from runtime_entrypoint import ensure_runtime_library_path, ensure_runtime_python

REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "apps" / "api" / "src"

ensure_runtime_python()
ensure_runtime_library_path()
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from voice_pipeline.gpu.vllm_worker.engine import (
    VLLMEngine,
    VLLMEngineConfig,
    build_prompt_cache_key,
)
from voice_pipeline.runtime.admission_gate import AdmissionError
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
    print("real-vllm-smoke: starting", flush=True)

    engine = VLLMEngine(
        config.resolved_vllm_model_path(),
        config=VLLMEngineConfig(
            model_name=config.resolved_vllm_model_path(),
            model_path=config.resolved_vllm_model_path(),
            cache_dir=config.vllm_cache_dir,
            required_device=config.llm_device,
            max_num_seqs=int(config.vllm_max_num_seqs),
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            gpu_memory_utilization=float(config.vllm_gpu_memory_utilization),
            max_model_len=int(config.vllm_max_model_len),
            max_num_batched_tokens=int(config.vllm_max_num_batched_tokens),
            offload_backend=str(config.vllm_offload_backend),
            cpu_offload_gb=float(config.vllm_cpu_offload_gb),
            kv_offloading_size=float(config.vllm_kv_offloading_size),
            kv_offloading_backend=str(config.vllm_kv_offloading_backend),
            kv_cache_dtype=str(config.vllm_kv_cache_dtype),
            attention_backend=str(config.vllm_attention_backend),
            safetensors_load_strategy=str(config.vllm_safetensors_load_strategy),
            max_tokens=min(16, int(config.vllm_max_tokens)),
            temperature=float(config.vllm_temperature),
            top_p=float(config.vllm_top_p),
            system_prompt=str(config.vllm_system_prompt),
        ),
    )
    print("real-vllm-smoke: warming engine", flush=True)
    engine.warm(strict=True)
    print("real-vllm-smoke: engine warm complete", flush=True)
    cache_key = build_prompt_cache_key(
        system_prompt=str(config.vllm_system_prompt),
        context_prefix="real vllm smoke",
        stable_prefix="Respond with exactly: ok",
    )
    engine.prewarm_prefix_cache(cache_key)
    print("real-vllm-smoke: rendering prompt", flush=True)
    prompt = engine.render_prompt(
        user_text="Respond with exactly: ok",
        stable_session_summary="real vllm smoke",
    )
    print("real-vllm-smoke: waiting for first token", flush=True)

    first_token = ""
    async for token in engine.stream_tokens(
        prompt,
        cache_key=cache_key,
        request_id="real-vllm-smoke",
        max_tokens=8,
    ):
        first_token = str(token)
        if first_token:
            break

    if not first_token:
        raise RuntimeError("no token emitted from vLLM stream probe")

    print(f"real-vllm-smoke: READY ({first_token!r})", flush=True)
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except Exception as exc:
        print(f"real-vllm-smoke: FAILED ({exc})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
