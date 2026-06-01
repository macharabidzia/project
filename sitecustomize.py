from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_repo_sitecustomize() -> None:
    repo_root = Path(__file__).resolve().parent
    delegated_path = repo_root / "apps" / "api" / "src" / "sitecustomize.py"
    if not delegated_path.exists() or not delegated_path.is_file():
        return
    spec = importlib.util.spec_from_file_location("_voice_pipeline_repo_sitecustomize", delegated_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)


_load_repo_sitecustomize()
