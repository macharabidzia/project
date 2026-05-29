from __future__ import annotations

from pathlib import Path

import voice_pipeline.kernel as kernel


REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_ROOT = REPO_ROOT / "src" / "voice_pipeline" / "execution"


def test_removed_execution_namespace_has_no_python_modules() -> None:
    if not EXECUTION_ROOT.exists():
        return
    assert not any(path.is_file() for path in EXECUTION_ROOT.rglob("*.py")), (
        "voice_pipeline/execution must be removed after the worker/kernel collapse."
    )


def test_session_reducer_loop_not_exported_from_runtime_core() -> None:
    removed_loop_symbol = "Session" + "ReducerLoop"
    assert not hasattr(kernel, removed_loop_symbol)


def test_kernel_runtime_has_no_event_alias_compatibility_shim() -> None:
    source = (REPO_ROOT / "src" / "voice_pipeline" / "kernel" / "kernel_runtime.py").read_text(encoding="utf-8")
    assert "_event" + "_alias" not in source
