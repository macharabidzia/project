from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "apps" / "api" / "src"


def _exit_code_from(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return int(code)
    return 1


def _run_script(path: Path) -> int:
    added_api_src = False
    if str(API_SRC) not in sys.path:
        sys.path.insert(0, str(API_SRC))
        added_api_src = True

    previous_argv = list(sys.argv)
    previous_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        sys.argv = [str(path)]
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as exc:
        return _exit_code_from(exc)
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
        if added_api_src:
            try:
                sys.path.remove(str(API_SRC))
            except ValueError:
                pass


def main() -> int:
    steps = [
        REPO_ROOT / "apps" / "api" / "scripts" / "check_runtime_imports.py",
        REPO_ROOT / "apps" / "api" / "scripts" / "check_startup_contract.py",
        REPO_ROOT / "apps" / "api" / "scripts" / "check_drift_guards.py",
        REPO_ROOT / "scripts" / "check_backend_install_contract.py",
        REPO_ROOT / "scripts" / "check_real_smoke_contract.py",
        REPO_ROOT / "scripts" / "check_frontend_contract.py",
        REPO_ROOT / "scripts" / "audit_voice_pipeline_writers.py",
    ]
    for step in steps:
        code = _run_script(step)
        if code != 0:
            return code
    print("drift-audit: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
