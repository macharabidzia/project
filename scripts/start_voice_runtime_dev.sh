#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

export VOICE_PIPELINE_ENV_FILE="${VOICE_PIPELINE_ENV_FILE:-${ROOT_DIR}/.env.voice_pipeline}"

cd "${ROOT_DIR}"
exec python scripts/run_voice_runtime.py --host "${HOST}" --port "${PORT}" "$@"
