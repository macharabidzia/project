from __future__ import annotations

import faulthandler
import importlib.util
import ctypes
from pathlib import Path
import signal
import site
import sys

from runtime_entrypoint import ensure_runtime_library_path, ensure_runtime_python


REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "apps" / "api" / "src"
COSYVOICE_RUNTIME = REPO_ROOT / ".models" / "CosyVoice-runtime"
MATCHA_TTS = COSYVOICE_RUNTIME / "third_party" / "Matcha-TTS"


def _preload_cuda_runtime_libraries() -> None:
    candidate_roots: list[Path] = []
    for site_dir in site.getsitepackages():
        root = Path(site_dir)
        nvidia_root = root / "nvidia"
        if nvidia_root.is_dir():
            candidate_roots.append(nvidia_root)
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

ensure_runtime_python()
ensure_runtime_library_path()
_preload_cuda_runtime_libraries()
faulthandler.enable(all_threads=True)
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))
if COSYVOICE_RUNTIME.exists() and str(COSYVOICE_RUNTIME) not in sys.path:
    sys.path.insert(0, str(COSYVOICE_RUNTIME))
if MATCHA_TTS.exists() and str(MATCHA_TTS) not in sys.path:
    sys.path.insert(0, str(MATCHA_TTS))
_load_repo_sitecustomize()

from voice_pipeline.runtime.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
