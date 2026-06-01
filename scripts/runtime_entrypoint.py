from __future__ import annotations

import os
from pathlib import Path
import shutil
import site
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / ".run"
RUNTIME_PYTHON_READY_ENV = "VOICE_PIPELINE_RUNTIME_PYTHON_READY"
LD_LIBRARY_PATH_READY_ENV = "VOICE_PIPELINE_LD_LIBRARY_PATH_READY"


def _read_hint_path(name: str) -> Path | None:
    hint_path = RUN_DIR / name
    if not hint_path.exists() or not hint_path.is_file():
        return None
    resolved = Path(hint_path.read_text(encoding="utf-8").strip()).expanduser()
    if not str(resolved).strip():
        return None
    return resolved


def _resolve_python(root: Path | None) -> Path | None:
    if root is None:
        return None
    python_path = root / "bin" / "python"
    if python_path.exists() and python_path.is_file():
        return python_path.absolute()
    return None


def _resolve_command_path(command: str) -> Path | None:
    resolved = shutil.which(command)
    if resolved is None:
        return None
    return Path(resolved).absolute()


def _absolute_path(path: Path | str) -> Path:
    return Path(path).expanduser().absolute()


def resolve_runtime_python() -> Path:
    worker_root = _read_hint_path("runpod-worker-venv") or (REPO_ROOT / ".venv-runpod-worker")
    api_root = _read_hint_path("runpod-api-venv") or (REPO_ROOT / ".venv-runpod")

    worker_python = _resolve_python(worker_root)
    if worker_python is not None:
        return worker_python

    api_python = _resolve_python(api_root)
    if api_python is not None:
        return api_python

    for command in ("python3", "python"):
        resolved = _resolve_command_path(command)
        if resolved is not None:
            return resolved
    return _absolute_path(sys.executable)


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


def resolve_runtime_library_path(python_exec: Path | None = None) -> str:
    target = _absolute_path(python_exec or sys.executable)
    current = _absolute_path(sys.executable)
    if target == current:
        return _resolved_runtime_library_path()

    probe = (
        "import os, site\n"
        "from pathlib import Path\n"
        "paths=[]\n"
        "for site_dir in site.getsitepackages():\n"
        "    root=Path(site_dir)\n"
        "    torch_lib=root/'torch'/'lib'\n"
        "    if torch_lib.is_dir():\n"
        "        paths.append(str(torch_lib))\n"
        "    nvidia_root=root/'nvidia'\n"
        "    if nvidia_root.is_dir():\n"
        "        for lib_dir in sorted(nvidia_root.glob('*/lib')):\n"
        "            if lib_dir.is_dir():\n"
        "                paths.append(str(lib_dir))\n"
        "seen=set()\n"
        "ordered=[]\n"
        "for item in paths:\n"
        "    if item in seen:\n"
        "        continue\n"
        "    seen.add(item)\n"
        "    ordered.append(item)\n"
        "print(os.pathsep.join(ordered))\n"
    )
    completed = subprocess.run(
        [str(target), "-c", probe],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"failed to resolve runtime library path via {target}: {detail or 'unknown error'}"
        )
    return completed.stdout.strip()


def _prepend_library_path(resolved: str) -> bool:
    if not resolved:
        return False
    existing = [entry for entry in str(os.getenv("LD_LIBRARY_PATH", "")).split(os.pathsep) if entry]
    needed = [entry for entry in resolved.split(os.pathsep) if entry]
    if all(entry in existing for entry in needed):
        os.environ[LD_LIBRARY_PATH_READY_ENV] = "1"
        return False
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([*needed, *existing])
    os.environ[LD_LIBRARY_PATH_READY_ENV] = "1"
    return True


def ensure_runtime_python(argv: list[str] | None = None) -> Path:
    resolved = resolve_runtime_python()
    if os.getenv(RUNTIME_PYTHON_READY_ENV) == "1":
        return resolved
    current = _absolute_path(sys.executable)
    if current == resolved:
        return resolved
    env = dict(os.environ)
    env[RUNTIME_PYTHON_READY_ENV] = "1"
    runtime_library_path = resolve_runtime_library_path(resolved)
    if runtime_library_path:
        existing = [entry for entry in str(env.get("LD_LIBRARY_PATH", "")).split(os.pathsep) if entry]
        needed = [entry for entry in runtime_library_path.split(os.pathsep) if entry]
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*needed, *existing])
        env[LD_LIBRARY_PATH_READY_ENV] = "1"
    exec_argv = [str(resolved), *(argv or sys.argv)]
    os.execve(str(resolved), exec_argv, env)
    return resolved


def ensure_runtime_library_path() -> None:
    resolved = resolve_runtime_library_path()
    if resolved:
        os.environ["VOICE_PIPELINE_RESOLVED_LD_LIBRARY_PATH"] = resolved
    if os.getenv(LD_LIBRARY_PATH_READY_ENV) == "1":
        return
    if not resolved:
        return
    if _prepend_library_path(resolved):
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ))
