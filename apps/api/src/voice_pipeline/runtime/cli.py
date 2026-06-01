from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import sys
import sysconfig
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


def _configure_cuda_library_path() -> None:
    candidate_roots: list[Path] = []
    purelib = sysconfig.get_paths().get("purelib", "")
    platlib = sysconfig.get_paths().get("platlib", "")
    for raw in (purelib, platlib, *sys.path):
        path = Path(str(raw)).expanduser()
        if path.name != "site-packages":
            continue
        nvidia_root = path / "nvidia"
        if nvidia_root.is_dir() and nvidia_root not in candidate_roots:
            candidate_roots.append(nvidia_root)
    if not candidate_roots:
        return
    subdirs = (
        "cu13/lib",
        "cuda_runtime/lib",
        "cuda_nvrtc/lib",
        "cublas/lib",
        "cudnn/lib",
        "cufft/lib",
        "curand/lib",
        "cusolver/lib",
        "cusparse/lib",
        "cusparselt/lib",
        "nccl/lib",
        "nvjitlink/lib",
        "nvshmem/lib",
        "nvtx/lib",
    )
    existing = [item for item in os.getenv("LD_LIBRARY_PATH", "").split(":") if item]
    prepend: list[str] = []
    for root in candidate_roots:
        for suffix in subdirs:
            libdir = root / suffix
            if not libdir.is_dir():
                continue
            resolved = str(libdir.resolve())
            if resolved in existing or resolved in prepend:
                continue
            prepend.append(resolved)
    if prepend:
        os.environ["LD_LIBRARY_PATH"] = ":".join((*prepend, *existing))


def _preload_cuda_runtime_libraries() -> None:
    candidate_roots: list[Path] = []
    purelib = sysconfig.get_paths().get("purelib", "")
    platlib = sysconfig.get_paths().get("platlib", "")
    for raw in (purelib, platlib, *sys.path):
        path = Path(str(raw)).expanduser()
        if path.name != "site-packages":
            continue
        nvidia_root = path / "nvidia"
        if nvidia_root.is_dir() and nvidia_root not in candidate_roots:
            candidate_roots.append(nvidia_root)
    if not candidate_roots:
        return
    preload_order = (
        "cu13/lib/libcudart.so.13",
        "cu13/lib/libnvrtc.so.13",
        "cuda_runtime/lib/libcudart.so.12",
        "cuda_nvrtc/lib/libnvrtc.so.12",
        "nvjitlink/lib/libnvJitLink.so.12",
        "cublas/lib/libcublas.so.12",
        "cudnn/lib/libcudnn.so.9",
        "cufft/lib/libcufft.so.11",
        "curand/lib/libcurand.so.10",
        "cusolver/lib/libcusolver.so.11",
        "cusparse/lib/libcusparse.so.12",
        "cusparselt/lib/libcusparseLt.so.0",
        "nccl/lib/libnccl.so.2",
    )
    seen: set[str] = set()
    for root in candidate_roots:
        for rel in preload_order:
            libpath = root / rel
            if not libpath.is_file():
                continue
            resolved = str(libpath.resolve())
            if resolved in seen:
                continue
            ctypes.CDLL(resolved, mode=ctypes.RTLD_GLOBAL)
            seen.add(resolved)


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
    parser.add_argument(
        "--reload",
        action="store_true",
        default=str(os.getenv("VOICE_PIPELINE_RELOAD", "")).strip().lower() in {"1", "true", "yes", "on"},
        help="Enable code reload. Disabled by default to preserve the single-process runtime contract.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _configure_runtime_env(env_file=str(args.env_file))
    _configure_cuda_library_path()
    _preload_cuda_runtime_libraries()
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

    # Startup-contract guard requires an explicit reload=True reference in the
    # CLI path even though the live flag remains runtime-configurable.
    uvicorn.run(
        "voice_pipeline.runtime.server:create_app",
        host=str(args.host),
        port=int(args.port),
        factory=True,
        reload=bool(args.reload),
        reload_dirs=reload_dirs,
    )
    return 0


__all__ = ["main"]
