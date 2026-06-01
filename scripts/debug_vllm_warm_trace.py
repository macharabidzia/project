from __future__ import annotations

import importlib.util
import os
import site
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "apps" / "api" / "src"


def _resolved_runtime_library_path() -> str:
    paths: list[str] = []
    for site_dir in site.getsitepackages():
        root = Path(site_dir)
        torch_lib = root / "torch" / "lib"
        if torch_lib.is_dir():
            paths.append(str(torch_lib))
        nvidia_root = root / "nvidia"
        if nvidia_root.is_dir():
            for lib_dir in sorted(nvidia_root.glob("*/lib")):
                if lib_dir.is_dir():
                    paths.append(str(lib_dir))
    seen: set[str] = set()
    ordered: list[str] = []
    for item in paths:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return os.pathsep.join(ordered)


def _ensure_runtime_library_path() -> None:
    marker = "VOICE_PIPELINE_LD_LIBRARY_PATH_READY"
    if os.getenv(marker) == "1":
        return
    resolved = _resolved_runtime_library_path()
    if not resolved:
        return
    existing = [entry for entry in str(os.getenv("LD_LIBRARY_PATH", "")).split(os.pathsep) if entry]
    needed = [entry for entry in resolved.split(os.pathsep) if entry]
    if all(entry in existing for entry in needed):
        os.environ[marker] = "1"
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([*needed, *existing])
    os.environ[marker] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ))


def _load_repo_sitecustomize() -> None:
    delegated_path = API_SRC / "sitecustomize.py"
    if not delegated_path.is_file():
        return
    spec = importlib.util.spec_from_file_location(
        "_voice_pipeline_repo_sitecustomize",
        delegated_path,
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)


_ensure_runtime_library_path()
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))
_load_repo_sitecustomize()

from voice_pipeline.gpu.vllm_worker.engine import VLLMEngine, VLLMEngineConfig
from voice_pipeline.runtime.config import RuntimeConfig


def main() -> None:
    cfg = RuntimeConfig.from_env()
    engine = VLLMEngine(
        cfg.resolved_vllm_model_path(),
        config=VLLMEngineConfig(
            model_name=cfg.resolved_vllm_model_path(),
            model_path=cfg.resolved_vllm_model_path(),
            cache_dir=cfg.vllm_cache_dir,
            required_device=cfg.llm_device,
            max_num_seqs=int(cfg.vllm_max_num_seqs),
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            gpu_memory_utilization=float(cfg.vllm_gpu_memory_utilization),
            max_model_len=int(cfg.vllm_max_model_len),
            max_num_batched_tokens=int(cfg.vllm_max_num_batched_tokens),
            offload_backend=str(cfg.vllm_offload_backend),
            cpu_offload_gb=float(cfg.vllm_cpu_offload_gb),
            kv_offloading_size=float(cfg.vllm_kv_offloading_size),
            kv_offloading_backend=str(cfg.vllm_kv_offloading_backend),
            kv_cache_dtype=str(cfg.vllm_kv_cache_dtype),
            kv_cache_memory_bytes=int(cfg.vllm_kv_cache_memory_bytes),
            num_gpu_blocks_override=int(cfg.vllm_num_gpu_blocks_override),
            attention_backend=str(cfg.vllm_attention_backend),
            safetensors_load_strategy=str(cfg.vllm_safetensors_load_strategy),
            max_tokens=min(16, int(cfg.vllm_max_tokens)),
            temperature=float(cfg.vllm_temperature),
            top_p=float(cfg.vllm_top_p),
            system_prompt=str(cfg.vllm_system_prompt),
        ),
    )
    print("debug-vllm-warm-trace: warming", flush=True)
    engine.warm(strict=True)
    print("debug-vllm-warm-trace: OK", flush=True)


if __name__ == "__main__":
    main()
