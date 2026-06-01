#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_ROOT="${MODELS_ROOT:-${ROOT_DIR}/.models}"
DOWNLOAD_PROVIDER="${DOWNLOAD_PROVIDER:-huggingface}"
HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

DOWNLOAD_RESPONSE_MODEL="${DOWNLOAD_RESPONSE_MODEL:-1}"
DOWNLOAD_TTS_MODEL="${DOWNLOAD_TTS_MODEL:-1}"
DOWNLOAD_ASR_MODEL="${DOWNLOAD_ASR_MODEL:-1}"
DOWNLOAD_COSYVOICE_REPO="${DOWNLOAD_COSYVOICE_REPO:-0}"

RESPONSE_MODEL_ID="${RESPONSE_MODEL_ID:-Qwen/Qwen3-8B}"
TTS_MODEL_ID="${TTS_MODEL_ID:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
VOSK_MODEL_NAME="${VOSK_MODEL_NAME:-vosk-model-small-en-us-0.15}"
VOSK_MODEL_PATH="${VOSK_MODEL_PATH:-${MODELS_ROOT}/${VOSK_MODEL_NAME}}"
VOSK_MODEL_URL="${VOSK_MODEL_URL:-https://alphacephei.com/vosk/models/${VOSK_MODEL_NAME}.zip}"
COSYVOICE_REPO_URL="${COSYVOICE_REPO_URL:-https://github.com/FunAudioLLM/CosyVoice.git}"
COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR:-${ROOT_DIR}/.models/CosyVoice-runtime}"
COSYVOICE_REPO_REF="${COSYVOICE_REPO_REF:-}"

as_model_dir_name() {
  local model_id="$1"
  printf '%s' "${model_id##*/}"
}

resolve_download_python() {
  if [ -x "${ROOT_DIR}/.venv-runpod-worker/bin/python" ]; then
    printf '%s' "${ROOT_DIR}/.venv-runpod-worker/bin/python"
    return 0
  fi

  if [ -x "${ROOT_DIR}/.venv-runpod/bin/python" ]; then
    printf '%s' "${ROOT_DIR}/.venv-runpod/bin/python"
    return 0
  fi

  printf '%s' "python3"
}

DOWNLOAD_PYTHON="$(resolve_download_python)"

ensure_huggingface_cli() {
  if "${DOWNLOAD_PYTHON}" - <<'PY' >/dev/null 2>&1; then
import importlib.metadata as m
import sys

try:
    version = m.version("huggingface_hub")
except m.PackageNotFoundError:
    sys.exit(1)

major = int(version.split(".", 1)[0])
sys.exit(0 if major < 1 else 1)
PY
    return 0
  fi

  "${DOWNLOAD_PYTHON}" -m pip install -U "huggingface_hub>=0.34,<1.0"
}

ensure_modelscope_cli() {
  if command -v modelscope >/dev/null 2>&1; then
    return 0
  fi

  "${DOWNLOAD_PYTHON}" -m pip install -U modelscope
}

ensure_cosyvoice_checkout() {
  if [ "${DOWNLOAD_COSYVOICE_REPO}" != "1" ]; then
    return 0
  fi

  mkdir -p "$(dirname "${COSYVOICE_REPO_DIR}")"
  if [ ! -d "${COSYVOICE_REPO_DIR}/.git" ]; then
    echo "cloning CosyVoice runtime into ${COSYVOICE_REPO_DIR}"
    git clone --recursive "${COSYVOICE_REPO_URL}" "${COSYVOICE_REPO_DIR}"
  fi

  if [ -n "${COSYVOICE_REPO_REF}" ]; then
    git -C "${COSYVOICE_REPO_DIR}" fetch --tags origin
    git -C "${COSYVOICE_REPO_DIR}" checkout "${COSYVOICE_REPO_REF}"
  fi
  git -C "${COSYVOICE_REPO_DIR}" submodule update --init --recursive
}

download_with_huggingface() {
  local model_id="$1"
  local target_dir="$2"
  local revision="${3:-}"
  HF_MODEL_ID="${model_id}" HF_TARGET_DIR="${target_dir}" HF_MODEL_REVISION="${revision}" "${DOWNLOAD_PYTHON}" - <<'PY'
import os
from huggingface_hub import snapshot_download

revision = os.environ.get("HF_MODEL_REVISION") or None

snapshot_download(
    repo_id=os.environ["HF_MODEL_ID"],
    local_dir=os.environ["HF_TARGET_DIR"],
    local_dir_use_symlinks=False,
    resume_download=True,
    revision=revision,
)
PY
}

download_with_modelscope() {
  local model_id="$1"
  local target_dir="$2"
  modelscope download --model "${model_id}" --local_dir "${target_dir}"
}

download_if_missing() {
  local model_id="$1"
  local target_dir="$2"
  local revision="${3:-}"

  if [ -d "${target_dir}" ] && [ -n "$(ls -A "${target_dir}" 2>/dev/null || true)" ]; then
    echo "model cache present: ${model_id} -> ${target_dir}"
    return 0
  fi

  mkdir -p "${target_dir}"
  echo "downloading ${model_id} -> ${target_dir}"
  if [ "${DOWNLOAD_PROVIDER}" = "modelscope" ]; then
    download_with_modelscope "${model_id}" "${target_dir}"
  else
    download_with_huggingface "${model_id}" "${target_dir}" "${revision}"
  fi
}

download_vosk_if_missing() {
  local model_name="$1"
  local model_path="$2"
  local model_url="$3"

  if [ -d "${model_path}" ] && [ -n "$(ls -A "${model_path}" 2>/dev/null || true)" ]; then
    echo "vosk cache present: ${model_name} -> ${model_path}"
    return 0
  fi

  echo "downloading vosk ${model_name} -> ${model_path}"
  VOSK_DOWNLOAD_URL="${model_url}" VOSK_DOWNLOAD_PATH="${model_path}" "${DOWNLOAD_PYTHON}" - <<'PY'
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

model_url = Path(".")
_ = model_url
download_url = __import__("os").environ["VOSK_DOWNLOAD_URL"]
target_path = Path(__import__("os").environ["VOSK_DOWNLOAD_PATH"]).resolve()
target_path.parent.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as temp_dir:
    archive_path = Path(temp_dir) / "vosk-model.zip"
    extract_root = Path(temp_dir) / "extract"
    with urllib.request.urlopen(download_url) as response, archive_path.open("wb") as archive_file:
        shutil.copyfileobj(response, archive_file)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_root)

    extracted_dirs = [path for path in extract_root.iterdir() if path.is_dir()]
    if len(extracted_dirs) != 1:
        raise RuntimeError(f"Unexpected Vosk archive layout from {download_url}: {extracted_dirs}")

    extracted_model_dir = extracted_dirs[0]
    if target_path.exists():
        shutil.rmtree(target_path)
    shutil.move(str(extracted_model_dir), str(target_path))
PY
}

mkdir -p "${MODELS_ROOT}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"
export HF_HOME
export TRANSFORMERS_CACHE

if [ "${DOWNLOAD_PROVIDER}" = "modelscope" ]; then
  ensure_modelscope_cli
else
  ensure_huggingface_cli
fi

ensure_cosyvoice_checkout

if [ "${DOWNLOAD_RESPONSE_MODEL}" = "1" ]; then
  download_if_missing "${RESPONSE_MODEL_ID}" "${MODELS_ROOT}/$(as_model_dir_name "${RESPONSE_MODEL_ID}")"
fi
if [ "${DOWNLOAD_TTS_MODEL}" = "1" ]; then
  download_if_missing "${TTS_MODEL_ID}" "${MODELS_ROOT}/$(as_model_dir_name "${TTS_MODEL_ID}")"
fi
if [ "${DOWNLOAD_ASR_MODEL}" = "1" ]; then
  download_vosk_if_missing "${VOSK_MODEL_NAME}" "${VOSK_MODEL_PATH}" "${VOSK_MODEL_URL}"
fi

echo "model download phase complete"
echo "models root: ${MODELS_ROOT}"
