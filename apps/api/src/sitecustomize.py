from __future__ import annotations

import os
import sys
from pathlib import Path


_ORIGINAL_RGLOB = Path.rglob


def _append_runtime_sys_path_entries() -> None:
    raw = os.getenv("VOICE_PIPELINE_APPEND_SYS_PATH", "").strip()
    if not raw:
        return

    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        if not entry or entry in sys.path:
            continue
        sys.path.append(entry)


def _prepend_runtime_sys_path_entries() -> None:
    raw = os.getenv("VOICE_PIPELINE_PREPEND_SYS_PATH", "").strip()
    if not raw:
        return

    entries = [entry.strip() for entry in raw.split(os.pathsep) if entry.strip()]
    for entry in reversed(entries):
        if entry in sys.path:
            continue
        sys.path.insert(0, entry)


def _should_skip_transformers_image_processor_scan(self: Path, pattern: str) -> bool:
    if os.getenv("TRANSFORMERS_SKIP_IMAGE_PROCESSOR_ALIAS_SCAN", "").strip() not in {"1", "true", "yes", "on"}:
        return False
    return (
        pattern == "image_processing_*.py"
        and self.name == "models"
        and self.parent.name == "transformers"
    )


def _patched_rglob(self: Path, pattern: str):
    if _should_skip_transformers_image_processor_scan(self, pattern):
        return iter(())
    return _ORIGINAL_RGLOB(self, pattern)


Path.rglob = _patched_rglob
_prepend_runtime_sys_path_entries()
_append_runtime_sys_path_entries()
