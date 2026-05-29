from __future__ import annotations

from typing import Iterable


def module_runtime_owner(module_name: str) -> str:
    module = str(module_name or "")
    if ".shared." in module or module.endswith(".shared") or module == "voice_pipeline.shared":
        return "shared"
    if ".runtime." in module or module.endswith(".runtime"):
        return "runtime"
    if ".kernel." in module or module.endswith(".kernel"):
        return "kernel"
    if ".transport." in module or module.endswith(".transport"):
        return "transport"
    if ".gpu." in module or module.endswith(".gpu"):
        return "gpu"
    if ".stt." in module or module.endswith(".stt"):
        return "stt"
    return "shared"


def assert_valid_runtime_import(module_name: str, imports: Iterable[str]) -> None:
    owner = module_runtime_owner(module_name)
    for target in imports:
        target_text = str(target or "")
        if not target_text.startswith("voice_pipeline."):
            continue
        if owner in {"gpu", "stt", "transport"} and ".runtime." in target_text:
            raise RuntimeError(f"runtime import guard: {module_name} must not import runtime module {target_text}")
        if owner in {"gpu", "stt", "transport"} and ".kernel." in target_text:
            raise RuntimeError(f"runtime import guard: {module_name} must not import kernel authority module {target_text}")
