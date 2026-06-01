#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  DEFAULT_HOST="0.0.0.0"
else
  DEFAULT_HOST="127.0.0.1"
fi
HOST="${HOST:-${DEFAULT_HOST}}"
PORT="${PORT:-8000}"
RUN_DIR="${ROOT_DIR}/.run"
WORKER_VENV_HINT_FILE="${RUN_DIR}/runpod-worker-venv"
API_VENV_HINT_FILE="${RUN_DIR}/runpod-api-venv"
PYTHON_BIN="${PYTHON_BIN:-}"

resolve_runtime_python() {
  local candidate=""

  if [ -n "${PYTHON_BIN}" ] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf '%s' "${PYTHON_BIN}"
    return 0
  fi

  if [ -f "${WORKER_VENV_HINT_FILE}" ]; then
    candidate="$(<"${WORKER_VENV_HINT_FILE}")"
    if [ -x "${candidate}/bin/python" ]; then
      printf '%s' "${candidate}/bin/python"
      return 0
    fi
  fi

  if [ -f "${API_VENV_HINT_FILE}" ]; then
    candidate="$(<"${API_VENV_HINT_FILE}")"
    if [ -x "${candidate}/bin/python" ]; then
      printf '%s' "${candidate}/bin/python"
      return 0
    fi
  fi

  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "python3"
    return 0
  fi

  printf '%s' "python"
}

resolve_runtime_library_path() {
  local python_exec="$1"

  "${python_exec}" - <<'PY'
import site
from pathlib import Path

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
seen = set()
ordered = []
for item in paths:
    if item in seen:
        continue
    seen.add(item)
    ordered.append(item)
print(":".join(ordered))
PY
}

export VOICE_PIPELINE_ENV_FILE="${VOICE_PIPELINE_ENV_FILE:-${ROOT_DIR}/.env.voice_pipeline}"
export VOICE_PIPELINE_OFFLINE="${VOICE_PIPELINE_OFFLINE:-1}"
export COSYVOICE_TEXT_FRONTEND="${COSYVOICE_TEXT_FRONTEND:-off}"
RUNTIME_PYTHON="$(resolve_runtime_python)"
RUNTIME_LIBRARY_PATH="$(resolve_runtime_library_path "${RUNTIME_PYTHON}")"

if [ -n "${RUNTIME_LIBRARY_PATH}" ]; then
  export LD_LIBRARY_PATH="${RUNTIME_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${ROOT_DIR}"
exec "${RUNTIME_PYTHON}" scripts/run_voice_runtime.py --host "${HOST}" --port "${PORT}" "$@"
