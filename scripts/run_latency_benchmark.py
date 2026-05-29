from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
before = REPO_ROOT / "metrics" / "latency" / "before.json"
after = REPO_ROOT / "metrics" / "latency" / "after.json"


def _exit_code_from(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return int(code)
    return 1


def _run_script(path: Path, argv: list[str]) -> int:
    previous_argv = list(sys.argv)
    previous_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        sys.argv = argv
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as exc:
        return _exit_code_from(exc)
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)


def main() -> int:
    if before.exists() and after.exists():
        script = REPO_ROOT / "scripts" / "check_latency_regression.py"
        return _run_script(
            script,
            [
                str(script),
                "--baseline",
                str(before),
                "--candidate",
                str(after),
                "--report",
                "docs/v2/phase2-latency-report.md",
            ],
        )

    print("latency-benchmark: missing metrics/latency/before.json or after.json")
    print("latency-benchmark: real benchmark inputs are required; mocked fallback has been removed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
