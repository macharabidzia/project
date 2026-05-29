from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / ".run"
API_SRC = REPO_ROOT / "apps" / "api" / "src"
COSYVOICE_REPO_DIR = REPO_ROOT / ".vendor" / "CosyVoice"
MATCHA_DIR = COSYVOICE_REPO_DIR / "third_party" / "Matcha-TTS"


def _read_hint_path(name: str) -> Path | None:
    hint_path = RUN_DIR / name
    if not hint_path.exists() or not hint_path.is_file():
        return None
    resolved = Path(hint_path.read_text(encoding="utf-8").strip())
    if not str(resolved).strip():
        return None
    return resolved


def _resolve_python(root: Path | None) -> Path | None:
    if root is None:
        return None
    python_path = root / "bin" / "python"
    if python_path.exists():
        return python_path
    return None


def _role_python_paths() -> dict[str, Path]:
    api_root = _read_hint_path("runpod-api-venv") or (REPO_ROOT / ".venv-runpod")
    worker_root = _read_hint_path("runpod-worker-venv") or (REPO_ROOT / ".venv-runpod-worker")
    cosy_root = _read_hint_path("runpod-cosyvoice-venv") or (REPO_ROOT / ".venv-runpod-cosyvoice")

    api_python = _resolve_python(api_root)
    worker_python = _resolve_python(worker_root) or api_python
    cosy_python = _resolve_python(cosy_root) or worker_python or api_python

    resolved: dict[str, Path] = {}
    if api_python is not None:
        resolved["api"] = api_python
    if worker_python is not None:
        resolved["worker"] = worker_python
    if cosy_python is not None:
        resolved["cosyvoice"] = cosy_python
    return resolved


def _run_import_check(python_path: Path, script: str) -> tuple[int, str]:
    env = dict(os.environ)
    existing_path = env.get("PYTHONPATH", "")
    prefixes = [str(API_SRC)]
    if COSYVOICE_REPO_DIR.exists():
        prefixes.append(str(COSYVOICE_REPO_DIR))
    if MATCHA_DIR.exists():
        prefixes.append(str(MATCHA_DIR))
    env["PYTHONPATH"] = os.pathsep.join(prefixes + ([existing_path] if existing_path else []))
    try:
        completed = subprocess.run(
            [str(python_path), "-c", script],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return 124, "import check timed out after 10s"
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode, output


def _check_python(python_path: Path, label: str, script: str) -> list[str]:
    code, output = _run_import_check(python_path, script)
    if code == 0:
        return [f"{label}: OK ({python_path})"]
    detail = output or "import check failed without output"
    return [f"{label}: FAILED ({python_path})", detail]


def _cosyvoice_repo_ready() -> tuple[bool, str]:
    if not COSYVOICE_REPO_DIR.exists() or not COSYVOICE_REPO_DIR.is_dir():
        return False, f"CosyVoice repo directory missing: {COSYVOICE_REPO_DIR}"
    entries = [item for item in COSYVOICE_REPO_DIR.iterdir()]
    if not entries:
        return False, f"CosyVoice repo directory is empty: {COSYVOICE_REPO_DIR}"
    return True, ""


def main() -> int:
    roles = _role_python_paths()
    if not roles:
        print("runtime-backend-check: FAILED")
        print("No provisioned backend Python environments found under .run hints or .venv-runpod* directories.")
        return 1

    role_checks: dict[str, tuple[tuple[str, str], ...]] = {
        "api": (
            ("api-runtime-imports", "import soxr, livekit, numpy; import voice_pipeline.runtime.bootstrap"),
        ),
        "worker": (
            ("asr-runtime-imports", "import soxr, numpy, vosk; import voice_pipeline.stt.asr_engine"),
            ("llm-runtime-imports", "import torch, numpy, vllm; import voice_pipeline.gpu.vllm_worker.engine"),
        ),
        "cosyvoice": (
            ("tts-runtime-imports", "import torch, numpy, cosyvoice; import voice_pipeline.gpu.tts_worker.engine"),
        ),
    }

    failures = False
    printed_paths: set[Path] = set()
    for role, python_path in roles.items():
        if python_path not in printed_paths:
            print(f"backend-python: {python_path}")
            printed_paths.add(python_path)
        print(f"backend-role: {role}")
        if role == "cosyvoice":
            repo_ready, repo_detail = _cosyvoice_repo_ready()
            if not repo_ready:
                failures = True
                print(f"tts-runtime-imports: FAILED ({python_path})")
                print(repo_detail)
                continue
        for label, script in role_checks.get(role, ()):
            lines = _check_python(python_path, label, script)
            if any("FAILED" in line for line in lines):
                failures = True
            for line in lines:
                print(line)

    if failures:
        print("runtime-backend-check: FAILED")
        return 1

    print("runtime-backend-check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
