#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${VOICE_PIPELINE_ENV_FILE:-${ROOT_DIR}/.env.voice_pipeline}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

if [ -n "${RUNPOD_POD_ID:-}" ]; then
  DEFAULT_BIND_HOST="0.0.0.0"
else
  DEFAULT_BIND_HOST="127.0.0.1"
fi

BIND_HOST="${LIVEKIT_BIND_HOST:-${DEFAULT_BIND_HOST}}"
LIVEKIT_URL_VALUE="${LIVEKIT_URL:-ws://127.0.0.1:7880}"
API_KEY="${LIVEKIT_API_KEY:-}"
API_SECRET="${LIVEKIT_API_SECRET:-}"
NODE_IP="${LIVEKIT_NODE_IP:-${RUNPOD_PUBLIC_IP:-}}"
UDP_PORT="${LIVEKIT_UDP_PORT:-}"

if [ -z "${API_KEY}" ] || [ -z "${API_SECRET}" ]; then
  echo "LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be configured" >&2
  exit 1
fi

args=(
  --dev
  --bind "${BIND_HOST}"
  --keys "${API_KEY}: ${API_SECRET}"
)

if [ -n "${NODE_IP}" ]; then
  args+=(--node-ip "${NODE_IP}")
fi

if [ -n "${UDP_PORT}" ]; then
  args+=(--udp-port "${UDP_PORT}")
fi

exec livekit-server "${args[@]}"
