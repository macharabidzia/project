#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "${ROOT_DIR}/scripts/lib/apt.sh"
API_DIR="${ROOT_DIR}/apps/api"
WEB_DIR="${ROOT_DIR}/apps/web"
RUN_DIR="${ROOT_DIR}/.run"
API_VENV_DIR="${API_VENV_DIR:-${ROOT_DIR}/.venv-runpod}"
WORKER_VENV_DIR="${WORKER_VENV_DIR:-${ROOT_DIR}/.venv-runpod-worker}"
RESPONSE_MODEL_VENV_DIR="${RESPONSE_MODEL_VENV_DIR:-${ROOT_DIR}/.venv-runpod-response-model}"
COSYVOICE_VENV_DIR="${COSYVOICE_VENV_DIR:-${ROOT_DIR}/.venv-runpod-cosyvoice}"
API_VENV_HINT_FILE="${RUN_DIR}/runpod-api-venv"
WORKER_VENV_HINT_FILE="${RUN_DIR}/runpod-worker-venv"
RESPONSE_MODEL_VENV_HINT_FILE="${RUN_DIR}/runpod-response-model-venv"
COSYVOICE_VENV_HINT_FILE="${RUN_DIR}/runpod-cosyvoice-venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_GPU="${INSTALL_GPU:-1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-${INSTALL_GPU}}"
INSTALL_DEV="${INSTALL_DEV:-0}"
INSTALL_FRONTEND="${INSTALL_FRONTEND:-1}"
BUILD_FRONTEND="${BUILD_FRONTEND:-1}"
NODE_MAJOR="${NODE_MAJOR:-20}"
INSTALL_LIVEKIT_SERVER="${INSTALL_LIVEKIT_SERVER:-1}"
LIVEKIT_SERVER_VERSION="${LIVEKIT_SERVER_VERSION:-1.11.0}"
LIVEKIT_SERVER_INSTALL_DIR="${LIVEKIT_SERVER_INSTALL_DIR:-/usr/local/bin}"
SEPARATE_RESPONSE_MODEL_VENV="${SEPARATE_RESPONSE_MODEL_VENV:-0}"
SEPARATE_WORKER_VENV="${SEPARATE_WORKER_VENV:-1}"
SEPARATE_COSYVOICE_VENV="${SEPARATE_COSYVOICE_VENV:-1}"
CLEAN_INSTALL="${CLEAN_INSTALL:-0}"
UPGRADE_PYTHON_TOOLING="${UPGRADE_PYTHON_TOOLING:-0}"
FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn==2.8.3}"
FLASH_ATTN_FIND_LINKS="${FLASH_ATTN_FIND_LINKS:-}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-}"
FLASH_ATTN_ALLOW_SOURCE_BUILD="${FLASH_ATTN_ALLOW_SOURCE_BUILD:-0}"
FLASH_ATTN_SOURCE_BUILD_TIMEOUT_SEC="${FLASH_ATTN_SOURCE_BUILD_TIMEOUT_SEC:-0}"
FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-4}"
FLASH_ATTN_NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS:-1}"
# Worker torch defaults are computed below. Override these only when you
# intentionally need a custom worker torch build.
WORKER_TORCH_SPEC="${WORKER_TORCH_SPEC:-}"
WORKER_TORCHAUDIO_SPEC="${WORKER_TORCHAUDIO_SPEC:-}"
WORKER_TORCHVISION_SPEC="${WORKER_TORCHVISION_SPEC:-}"
WORKER_TORCH_INDEX_URL="${WORKER_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
RESPONSE_MODEL_FLASHINFER_PYTHON_SPEC="${RESPONSE_MODEL_FLASHINFER_PYTHON_SPEC:-flashinfer-python==0.6.11.post3}"
RESPONSE_MODEL_FLASHINFER_CUBIN_SPEC="${RESPONSE_MODEL_FLASHINFER_CUBIN_SPEC:-flashinfer-cubin==0.6.11.post3}"
RESPONSE_MODEL_FLASHINFER_JIT_CACHE_SPEC="${RESPONSE_MODEL_FLASHINFER_JIT_CACHE_SPEC:-flashinfer-jit-cache==0.6.11.post3+cu130}"
RESPONSE_MODEL_FLASHINFER_JIT_CACHE_INDEX_URL="${RESPONSE_MODEL_FLASHINFER_JIT_CACHE_INDEX_URL:-https://flashinfer.ai/whl/cu130}"
COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR:-${ROOT_DIR}/.models/CosyVoice-runtime}"
COSYVOICE_REPO_URL="${COSYVOICE_REPO_URL:-https://github.com/FunAudioLLM/CosyVoice.git}"
COSYVOICE_REPO_REF="${COSYVOICE_REPO_REF:-}"
COSYVOICE_ONNXRUNTIME_SPEC="${COSYVOICE_ONNXRUNTIME_SPEC:-onnxruntime-gpu==1.21.0}"

# Keep the worker on the same torch tuple required by the in-process vLLM
# runtime so bootstrap never downgrades a compatible install underneath it.
# The selected index URL still controls the CUDA wheel flavor.
: "${WORKER_TORCH_SPEC:=torch==2.11.0}"
: "${WORKER_TORCHAUDIO_SPEC:=torchaudio==2.11.0}"
: "${WORKER_TORCHVISION_SPEC:=torchvision==0.26.0}"

ensure_venv() {
  local venv_dir="$1"
  local venv_python
  local expected_python_mm
  local actual_python_mm
  local site_packages_dir

  expected_python_mm="$("${PYTHON_BIN}" - <<'PY'
import sys

print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

  if [ -d "${venv_dir}" ] && [ -x "${venv_dir}/bin/python" ]; then
    venv_python="${venv_dir}/bin/python"
    actual_python_mm="$("${venv_python}" - <<'PY'
import sys

print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
    site_packages_dir="${venv_dir}/lib/python${actual_python_mm}/site-packages"

    if [ "${actual_python_mm}" != "${expected_python_mm}" ] \
      || { [ ! -d "${site_packages_dir}" ] && find "${venv_dir}/lib" -maxdepth 1 -type d -name 'python*' | grep -q .; }; then
      echo "Recreating stale virtualenv at ${venv_dir} (expected python${expected_python_mm}, found python${actual_python_mm})"
      safe_remove_dir "${venv_dir}"
    fi
  fi

  if [ ! -d "${venv_dir}" ]; then
    "${PYTHON_BIN}" -m venv --without-pip "${venv_dir}"
  fi

  venv_python="${venv_dir}/bin/python"
  if ! "${venv_python}" -m pip --version >/dev/null 2>&1; then
    "${venv_python}" -m ensurepip --upgrade
    PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
      "${venv_python}" -m pip install --upgrade pip setuptools wheel
  fi
}

run_pip() {
  local python_exec="$1"
  shift
  PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_PROGRESS_BAR=off "${python_exec}" -m pip "$@"
}

remove_conflicting_runtime_packages() {
  local python_exec="$1"
  shift || true

  if [ "$#" -eq 0 ]; then
    return
  fi

  run_pip "${python_exec}" uninstall -y "$@" >/dev/null 2>&1 || true
}

scrub_stale_pip_uninstall_artifacts() {
  local python_exec="$1"

  "${python_exec}" - <<'PY'
import shutil
import site
from pathlib import Path


def should_remove(name: str) -> bool:
  # Pip leaves invalid placeholder directories like "~ransformers" behind when
  # an uninstall is interrupted. Those break future resolver runs until they
  # are removed. Keep the cleanup narrow to obviously-invalid top-level entries.
  return name.startswith("~")


removed = []
for site_dir in site.getsitepackages():
  root = Path(site_dir)
  if not root.exists():
    continue
  for entry in root.iterdir():
    if not should_remove(entry.name):
      continue
    if entry.is_dir() and not entry.is_symlink():
      shutil.rmtree(entry)
    else:
      entry.unlink()
    removed.append(str(entry))

if removed:
  print("Removed stale pip uninstall artifacts:")
  for entry in removed:
    print(f"  {entry}")
PY
}

safe_remove_dir() {
  local target_dir="$1"
  local trash_dir

  if [ -z "${target_dir}" ] || [ ! -e "${target_dir}" ]; then
    return
  fi

  case "${target_dir}" in
    "${ROOT_DIR}"/*) ;;
    *)
      echo "Refusing to remove path outside repo root: ${target_dir}" >&2
      exit 1
      ;;
  esac

  trash_dir="${target_dir}.trash.$$.$RANDOM"
  mv "${target_dir}" "${trash_dir}"
  rm -rf "${trash_dir}" >/dev/null 2>&1 &
}

install_python_tooling() {
  local python_exec="$1"

  # On this RunPod image, self-upgrading pip/setuptools inside a brand-new venv
  # can drop into hidden uninstall prompts when stdout is piped through tee.
  # The project installs use build isolation, so the bundled venv tooling is
  # sufficient for normal bootstrap runs.
  if [ "${UPGRADE_PYTHON_TOOLING}" = "1" ]; then
    run_pip "${python_exec}" install --upgrade --no-input pip setuptools wheel
    return
  fi

  "${python_exec}" -m pip --version >/dev/null
  if ! "${python_exec}" -c "import wheel" >/dev/null 2>&1; then
    run_pip "${python_exec}" install wheel
  fi
}

install_project() {
  local python_exec="$1"
  local extras="$2"

  cd "${API_DIR}"
  if [ -n "${extras}" ]; then
    run_pip "${python_exec}" install -e ".[${extras}]"
  else
    run_pip "${python_exec}" install -e .
  fi
}

install_response_model_runtime() {
  local python_exec="$1"
  local vllm_spec="${VLLM_SPEC:-vllm>=0.11,<1.0}"

  run_pip "${python_exec}" install "${vllm_spec}"
  run_pip "${python_exec}" uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache >/dev/null 2>&1 || true
  run_pip "${python_exec}" install "${RESPONSE_MODEL_FLASHINFER_CUBIN_SPEC}" "${RESPONSE_MODEL_FLASHINFER_PYTHON_SPEC}"
  run_pip "${python_exec}" install --index-url "${RESPONSE_MODEL_FLASHINFER_JIT_CACHE_INDEX_URL}" "${RESPONSE_MODEL_FLASHINFER_JIT_CACHE_SPEC}"
}

install_runtime_torch_stack() {
  local runtime_name="$1"
  local python_exec="$2"
  local torch_spec="$3"
  local torchaudio_spec="$4"
  local torchvision_spec="$5"
  local index_url="$6"
  local -a packages=()

  if [ -n "${torch_spec}" ]; then
    packages+=("${torch_spec}")
  fi
  if [ -n "${torchaudio_spec}" ]; then
    packages+=("${torchaudio_spec}")
  fi
  if [ -n "${torchvision_spec}" ]; then
    packages+=("${torchvision_spec}")
  fi

  if [ ${#packages[@]} -eq 0 ]; then
    return
  fi

  scrub_stale_pip_uninstall_artifacts "${python_exec}"
  echo "Preinstalling ${runtime_name} torch stack: ${packages[*]}"
  if [ -n "${index_url}" ]; then
    run_pip "${python_exec}" install --index-url "${index_url}" "${packages[@]}"
  else
    run_pip "${python_exec}" install "${packages[@]}"
  fi
}

ensure_legacy_pkg_resources() {
  local python_exec="$1"

  if "${python_exec}" - <<'PY' >/dev/null 2>&1
import pkg_resources
PY
  then
    return
  fi

  echo "Installing setuptools<81 so legacy setup.py packages can import pkg_resources"
  run_pip "${python_exec}" install "setuptools<81"
}

ensure_cosyvoice_checkout() {
  mkdir -p "$(dirname "${COSYVOICE_REPO_DIR}")"
  if [ ! -d "${COSYVOICE_REPO_DIR}/.git" ]; then
    git clone --recursive "${COSYVOICE_REPO_URL}" "${COSYVOICE_REPO_DIR}"
  fi

  if [ -n "${COSYVOICE_REPO_REF}" ]; then
    git -C "${COSYVOICE_REPO_DIR}" fetch --tags origin
    git -C "${COSYVOICE_REPO_DIR}" checkout "${COSYVOICE_REPO_REF}"
  fi

  git -C "${COSYVOICE_REPO_DIR}" submodule update --init --recursive
}

install_cosyvoice_runtime() {
  local python_exec="$1"
  local requirements_file
  local whisper_spec="openai-whisper==20250625"

  ensure_cosyvoice_checkout
  remove_conflicting_runtime_packages "${python_exec}" openai-whisper matplotlib

  requirements_file="$(mktemp)"
  trap 'rm -f "${requirements_file}"' RETURN
  COSYVOICE_REQUIREMENTS_SOURCE="${COSYVOICE_REPO_DIR}/requirements.txt" "${python_exec}" - <<'PY' > "${requirements_file}"
import os
import re
from pathlib import Path

source = Path(os.environ["COSYVOICE_REQUIREMENTS_SOURCE"])
required_packages = {"s3tokenizer", "tiktoken"}
skipped_packages = {
    # These upstream pins are older than the worker/runtime contract and
    # downgrade the already-provisioned API/vLLM environment when installed
    # into the shared worker venv.
    "fastapi",
    "numpy",
    "protobuf",
    "pydantic",
    "rich",
    "transformers",
    "deepspeed",
    "fastapi-cli",
    "gradio",
    "grpcio",
    "grpcio-tools",
    "matplotlib",
    "onnxruntime",
    "onnxruntime-gpu",
    "openai-whisper",
    "tensorboard",
    "tensorrt-cu12",
    "tensorrt-cu12-bindings",
    "tensorrt-cu12-libs",
    "torch",
    "torchaudio",
    "torchvision",
    "uvicorn",
}
seen = set()
for line in source.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "-")):
        print(line)
        continue

    name = re.split(r"[\s\[<>=!~;#]", stripped, maxsplit=1)[0].strip().lower()
    seen.add(name)
    if name in skipped_packages:
        continue
    print(line)

# Upstream CosyVoice runtime imports s3tokenizer, but the top-level requirements file
# does not currently include it. Whisper is installed separately without dependency
# resolution so it cannot downgrade the preinstalled torch tuple.
for package_name in sorted(required_packages - seen):
    print(package_name)
PY
  ensure_legacy_pkg_resources "${python_exec}"
  run_pip "${python_exec}" install -r "${requirements_file}"
  run_pip "${python_exec}" install "matplotlib>=3.10,<4"
  if [ -n "${COSYVOICE_ONNXRUNTIME_SPEC}" ]; then
    run_pip "${python_exec}" install --upgrade "${COSYVOICE_ONNXRUNTIME_SPEC}"
  fi
  run_pip "${python_exec}" install --no-deps --no-build-isolation "${whisper_spec}"
}

verify_flash_attn() {
  local python_exec="$1"
  "${python_exec}" - <<'PY'
import flash_attn

print(f"flash-attn import verified from {getattr(flash_attn, '__file__', 'unknown')}")
PY
}

verify_worker_runtime() {
  local python_exec="$1"
  "${python_exec}" - <<'PY'

import torch

print(f"worker torch import verified: {torch.__version__} ({getattr(torch, '__file__', 'unknown')})")
PY
}

verify_cosyvoice_runtime() {
  local python_exec="$1"
  COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR}" VOICE_PIPELINE_API_SRC="${API_DIR}/src" "${python_exec}" - <<'PY'
import importlib
import os
import sys

repo_dir = os.environ["COSYVOICE_REPO_DIR"]
api_src = os.environ.get("VOICE_PIPELINE_API_SRC", "")
if api_src and api_src not in sys.path:
  sys.path.insert(0, api_src)
if repo_dir and repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
matcha_dir = os.path.join(repo_dir, "third_party", "Matcha-TTS")
if os.path.isdir(matcha_dir) and matcha_dir not in sys.path:
    sys.path.insert(0, matcha_dir)

for module_name in (
  "cosyvoice",
  "hyperpyyaml",
  "modelscope",
  "s3tokenizer",
  "whisper",
  "voice_pipeline.gpu.tts_worker.engine",
):
    module = importlib.import_module(module_name)
    print(f"{module_name} import verified from {getattr(module, '__file__', 'built-in')}")
PY
}

verify_response_model_runtime() {
  local python_exec="$1"
  "${python_exec}" - <<'PY'
import importlib
import importlib.metadata as md


def _print_version(name: str, *, required: bool = True) -> None:
    try:
        version = md.version(name)
    except md.PackageNotFoundError:
        if required:
            raise
        print(f"{name} version: not installed")
        return
    print(f"{name} version: {version}")


vllm_module = importlib.import_module("vllm")
flashinfer_module = importlib.import_module("flashinfer")
print(f"vllm import verified from {getattr(vllm_module, '__file__', 'built-in')}")
print(f"flashinfer import verified from {getattr(flashinfer_module, '__file__', 'built-in')}")
_print_version("flashinfer-python")
_print_version("flashinfer-cubin")
_print_version("flashinfer-jit-cache", required=False)
PY
}

verify_asr_runtime() {
  local python_exec="$1"
  "${python_exec}" - <<'PY'
import importlib

for module_name in ("vosk",):
    module = importlib.import_module(module_name)
    print(f"{module_name} import verified from {getattr(module, '__file__', 'built-in')}")
PY
}

flash_attn_runtime_tuple() {
  local python_exec="$1"
  "${python_exec}" - <<'PY'
import platform
import sys

import torch

python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
machine = platform.machine().lower()
machine = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "arm64": "aarch64",
    "aarch64": "aarch64",
}.get(machine, machine)
torch_version = getattr(torch, "__version__", "").split("+", 1)[0]
torch_mm = ".".join(torch_version.split(".")[:2])
cxx11abi = "TRUE" if torch.compiled_with_cxx11_abi() else "FALSE"
cuda_version = getattr(torch.version, "cuda", "") or "unknown"
print(
    f"python={python_tag} torch={torch_mm} cuda={cuda_version} "
    f"cxx11abi={cxx11abi} platform=linux_{machine}"
)
PY
}

resolve_flash_attn_auto_wheel_url() {
  local python_exec="$1"
  local flash_attn_version

  case "${FLASH_ATTN_SPEC}" in
    flash-attn==*)
      flash_attn_version="${FLASH_ATTN_SPEC#flash-attn==}"
      ;;
    *)
      return 1
      ;;
  esac

  FLASH_ATTN_VERSION="${flash_attn_version}" "${python_exec}" - <<'PY'
import os
import platform
import sys

import torch

version = os.environ["FLASH_ATTN_VERSION"]
python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
machine = platform.machine().lower()
machine = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "arm64": "aarch64",
    "aarch64": "aarch64",
}.get(machine, machine)
torch_version = getattr(torch, "__version__", "").split("+", 1)[0]
torch_mm = ".".join(torch_version.split(".")[:2])
cxx11abi = "TRUE" if torch.compiled_with_cxx11_abi() else "FALSE"

print(
    f"https://github.com/Dao-AILab/flash-attention/releases/download/v{version}/"
    f"flash_attn-{version}+cu12torch{torch_mm}cxx11abi{cxx11abi}-"
    f"{python_tag}-{python_tag}-linux_{machine}.whl"
)
PY
}

resolve_flash_attn_community_wheel_url() {
  local python_exec="$1"
  local flash_attn_version

  case "${FLASH_ATTN_SPEC}" in
    flash-attn==*)
      flash_attn_version="${FLASH_ATTN_SPEC#flash-attn==}"
      ;;
    *)
      return 1
      ;;
  esac

  FLASH_ATTN_VERSION="${flash_attn_version}" "${python_exec}" - <<'PY'
import os
import platform
import sys

import torch

version = os.environ["FLASH_ATTN_VERSION"]
python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
machine = platform.machine().lower()
machine = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
}.get(machine, machine)
torch_version = getattr(torch, "__version__", "").split("+", 1)[0]
torch_mm = ".".join(torch_version.split(".")[:2])
cxx11abi = "TRUE" if torch.compiled_with_cxx11_abi() else "FALSE"

if version != "2.8.3" or torch_mm != "2.11" or python_tag not in {"cp310", "cp311", "cp312", "cp313"} or machine != "x86_64":
    raise SystemExit(1)

print(
    "https://github.com/lesj0610/flash-attention/releases/download/"
    f"v{version}-cu12-torch{torch_mm}/"
    f"flash_attn-{version}%2Bcu12torch{torch_mm}cxx11abi{cxx11abi}-"
    f"{python_tag}-{python_tag}-linux_{machine}.whl"
)
PY
}

install_flash_attn_from_source() {
  local python_exec="$1"

  if [ "${FLASH_ATTN_SOURCE_BUILD_TIMEOUT_SEC}" != "0" ] && command -v timeout >/dev/null 2>&1; then
    MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS}" \
      timeout --preserve-status "${FLASH_ATTN_SOURCE_BUILD_TIMEOUT_SEC}" \
      env PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
      "${python_exec}" -m pip install --no-build-isolation "${FLASH_ATTN_SPEC}"
    return
  fi

  MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS}" \
    PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 "${python_exec}" -m pip install --no-build-isolation "${FLASH_ATTN_SPEC}"
}

ensure_node() {
  if command -v npm >/dev/null 2>&1; then
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "npm is not available and apt-get is not present to install it." >&2
    exit 1
  fi

  echo "Installing Node.js ${NODE_MAJOR}.x for the RunPod web bootstrap"
  apt_get_update
  apt-get install -y ca-certificates curl gnupg
  install -d -m 0755 /etc/apt/keyrings
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor --yes -o /etc/apt/keyrings/nodesource.gpg
  echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" > /etc/apt/sources.list.d/nodesource.list
  apt_get_update
  apt-get install -y nodejs
}

install_livekit_server() {
  if [ "${INSTALL_LIVEKIT_SERVER}" != "1" ]; then
    return
  fi

  if command -v livekit-server >/dev/null 2>&1; then
    local installed_ver
    installed_ver="$(livekit-server --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    if [ "${installed_ver}" = "${LIVEKIT_SERVER_VERSION#v}" ]; then
      echo "livekit-server ${installed_ver} already installed at $(command -v livekit-server)"
      return
    fi
    echo "livekit-server ${installed_ver} found but want ${LIVEKIT_SERVER_VERSION}, upgrading..."
  fi

  local machine arch version tarball url tmpdir
  machine="$(uname -m)"
  case "${machine}" in
    x86_64|amd64) arch="amd64" ;;
    aarch64|arm64) arch="arm64" ;;
    *)
      echo "Unsupported architecture for livekit-server: ${machine}" >&2
      return 1
      ;;
  esac

  version="${LIVEKIT_SERVER_VERSION#v}"
  tarball="livekit_${version}_linux_${arch}.tar.gz"
  url="https://github.com/livekit/livekit/releases/download/v${version}/${tarball}"

  echo "Installing livekit-server v${version} from ${url}"
  tmpdir="$(mktemp -d)"
  if ! curl -fsSL "${url}" -o "${tmpdir}/${tarball}"; then
    echo "Failed to download livekit-server from ${url}" >&2
    rm -rf "${tmpdir}"
    return 1
  fi
  tar -xzf "${tmpdir}/${tarball}" -C "${tmpdir}"
  install -d -m 0755 "${LIVEKIT_SERVER_INSTALL_DIR}"
  install -m 0755 "${tmpdir}/livekit-server" "${LIVEKIT_SERVER_INSTALL_DIR}/livekit-server"
  rm -rf "${tmpdir}"
  echo "Installed livekit-server: $(${LIVEKIT_SERVER_INSTALL_DIR}/livekit-server --version 2>&1 | head -n1)"
}

install_flash_attn() {
  local python_exec="$1"
  local runtime_tuple=""
  local auto_wheel_url=""
  local community_wheel_url=""

  if [ "${INSTALL_GPU}" != "1" ]; then
    return
  fi

  if [ "${INSTALL_FLASH_ATTN}" != "1" ]; then
    echo "Skipping flash-attn install (INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN})"
    return 0
  fi

  if run_pip "${python_exec}" show flash-attn >/dev/null 2>&1; then
    if verify_flash_attn "${python_exec}"; then
      return
    fi
    echo "Reinstalling flash-attn because the existing install is not importable."
    run_pip "${python_exec}" uninstall -y flash-attn >/dev/null 2>&1 || true
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1 && ! command -v nvcc >/dev/null 2>&1; then
    echo "flash-attn is required, but no CUDA runtime or toolkit was detected." >&2
    return 1
  fi

  echo "Installing ${FLASH_ATTN_SPEC} (prebuilt wheel, no source build)"
  run_pip "${python_exec}" install "ninja>=1.11.1" "packaging>=24.1"

  runtime_tuple="$(flash_attn_runtime_tuple "${python_exec}" 2>/dev/null || true)"
  auto_wheel_url="$(resolve_flash_attn_auto_wheel_url "${python_exec}" 2>/dev/null || true)"
  community_wheel_url="$(resolve_flash_attn_community_wheel_url "${python_exec}" 2>/dev/null || true)"

  if [ -n "${FLASH_ATTN_WHEEL_URL}" ]; then
    echo "Trying direct wheel URL: ${FLASH_ATTN_WHEEL_URL}"
    if run_pip "${python_exec}" install --no-build-isolation --no-deps "${FLASH_ATTN_WHEEL_URL}"; then
      verify_flash_attn "${python_exec}"
      return
    fi
    echo "Direct flash-attn wheel install failed; trying --find-links."
  fi
  if [ -n "${auto_wheel_url}" ]; then
    echo "Trying auto-detected release wheel: ${auto_wheel_url}"
    if run_pip "${python_exec}" install --no-build-isolation --no-deps "${auto_wheel_url}"; then
      verify_flash_attn "${python_exec}"
      return
    fi
    echo "No matching auto-detected flash-attn wheel was available for ${runtime_tuple}."
  fi
  if [ -n "${community_wheel_url}" ]; then
    echo "Trying community flash-attn wheel for ${runtime_tuple}: ${community_wheel_url}"
    if run_pip "${python_exec}" install --no-build-isolation --no-deps "${community_wheel_url}"; then
      verify_flash_attn "${python_exec}"
      return
    fi
    echo "Community flash-attn wheel install failed for ${runtime_tuple}."
  fi
  if [ -n "${FLASH_ATTN_FIND_LINKS}" ]; then
    if MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS}" \
        PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 "${python_exec}" -m pip install --no-build-isolation --only-binary=flash-attn \
        --find-links "${FLASH_ATTN_FIND_LINKS}" "${FLASH_ATTN_SPEC}"; then
      verify_flash_attn "${python_exec}"
      return
    fi
    echo "Prebuilt flash-attn wheel not found at ${FLASH_ATTN_FIND_LINKS}."
  fi

  if [ "${FLASH_ATTN_ALLOW_SOURCE_BUILD}" != "1" ]; then
    cat >&2 <<EOF
No compatible prebuilt flash-attn wheel was found for:
  ${runtime_tuple:-unknown runtime tuple}
Requested spec:
  ${FLASH_ATTN_SPEC}

Bootstrap will not fall back to a source build unless FLASH_ATTN_ALLOW_SOURCE_BUILD=1.
That source build is the step that stalls at "Building wheel for flash-attn".

Options:
  1. Set FLASH_ATTN_WHEEL_URL to an exact compatible wheel.
  2. Switch the worker runtime to a Python/torch tuple that has a published wheel.
  3. Set FLASH_ATTN_ALLOW_SOURCE_BUILD=1 to explicitly allow a long native build.
EOF
    return 1
  fi

  echo "Falling back to a source build for flash-attn because FLASH_ATTN_ALLOW_SOURCE_BUILD=1"
  install_flash_attn_from_source "${python_exec}"
  verify_flash_attn "${python_exec}"
}

if command -v apt-get >/dev/null 2>&1; then
  echo "Installing system dependencies: git and sox"
  apt_get_update
  apt-get install -y git sox

  if [ "${INSTALL_GPU}" = "1" ] && [ "${INSTALL_FLASH_ATTN}" = "1" ]; then
    echo "Installing flash-attn build prerequisites"
    apt-get install -y build-essential
  fi
fi

mkdir -p "${RUN_DIR}"

if [ "${SEPARATE_RESPONSE_MODEL_VENV}" = "1" ]; then
  echo "SEPARATE_RESPONSE_MODEL_VENV=1 is not supported by the in-process LLM runtime; using the worker environment instead"
  SEPARATE_RESPONSE_MODEL_VENV="0"
fi

if [ "${CLEAN_INSTALL}" = "1" ]; then
  echo "CLEAN_INSTALL=1 -> recreating RunPod virtualenvs"
  safe_remove_dir "${API_VENV_DIR}"
  rm -f "${API_VENV_HINT_FILE}"

  if [ "${SEPARATE_WORKER_VENV}" = "1" ] && [ "${WORKER_VENV_DIR}" != "${API_VENV_DIR}" ]; then
    safe_remove_dir "${WORKER_VENV_DIR}"
  fi
  rm -f "${WORKER_VENV_HINT_FILE}"

  if [ "${SEPARATE_RESPONSE_MODEL_VENV}" = "1" ] \
    && [ "${RESPONSE_MODEL_VENV_DIR}" != "${API_VENV_DIR}" ] \
    && [ "${RESPONSE_MODEL_VENV_DIR}" != "${WORKER_VENV_DIR}" ]; then
    safe_remove_dir "${RESPONSE_MODEL_VENV_DIR}"
  fi
  rm -f "${RESPONSE_MODEL_VENV_HINT_FILE}"

  if [ "${SEPARATE_COSYVOICE_VENV}" = "1" ] \
    && [ "${COSYVOICE_VENV_DIR}" != "${API_VENV_DIR}" ] \
    && [ "${COSYVOICE_VENV_DIR}" != "${WORKER_VENV_DIR}" ] \
    && [ "${COSYVOICE_VENV_DIR}" != "${RESPONSE_MODEL_VENV_DIR}" ]; then
    safe_remove_dir "${COSYVOICE_VENV_DIR}"
  fi
  rm -f "${COSYVOICE_VENV_HINT_FILE}"
fi

cd "${API_DIR}"

ensure_venv "${API_VENV_DIR}"
API_PYTHON="${API_VENV_DIR}/bin/python"
ACTIVE_WORKER_VENV_DIR="${API_VENV_DIR}"
ACTIVE_WORKER_PYTHON="${API_PYTHON}"
ACTIVE_RESPONSE_MODEL_VENV_DIR="${API_VENV_DIR}"
ACTIVE_RESPONSE_MODEL_PYTHON="${API_PYTHON}"
ACTIVE_COSYVOICE_VENV_DIR="${API_VENV_DIR}"
ACTIVE_COSYVOICE_PYTHON="${API_PYTHON}"

install_python_tooling "${API_PYTHON}"

API_EXTRAS=""
if [ "${INSTALL_DEV}" = "1" ]; then
  API_EXTRAS="dev"
fi

API_EXTRAS="${API_EXTRAS:+${API_EXTRAS},}asr"

if [ "${INSTALL_GPU}" = "1" ] && [ "${SEPARATE_WORKER_VENV}" != "1" ]; then
  install_runtime_torch_stack \
    "worker" \
    "${API_PYTHON}" \
    "${WORKER_TORCH_SPEC}" \
    "${WORKER_TORCHAUDIO_SPEC}" \
    "${WORKER_TORCHVISION_SPEC}" \
    "${WORKER_TORCH_INDEX_URL}"
  API_EXTRAS="${API_EXTRAS:+${API_EXTRAS},}gpu"
fi

if [ "${INSTALL_GPU}" = "1" ] && [ "${SEPARATE_RESPONSE_MODEL_VENV}" != "1" ]; then
  API_EXTRAS="${API_EXTRAS:+${API_EXTRAS},}response-model"
fi

install_project "${API_PYTHON}" "${API_EXTRAS}"
printf '%s\n' "${API_VENV_DIR}" > "${API_VENV_HINT_FILE}"

if [ "${INSTALL_GPU}" = "1" ] && [ "${SEPARATE_WORKER_VENV}" = "1" ]; then
  ensure_venv "${WORKER_VENV_DIR}"
  ACTIVE_WORKER_VENV_DIR="${WORKER_VENV_DIR}"
  ACTIVE_WORKER_PYTHON="${WORKER_VENV_DIR}/bin/python"

  install_python_tooling "${ACTIVE_WORKER_PYTHON}"
  remove_conflicting_runtime_packages "${ACTIVE_WORKER_PYTHON}" openai-whisper matplotlib
  install_runtime_torch_stack \
    "worker" \
    "${ACTIVE_WORKER_PYTHON}" \
    "${WORKER_TORCH_SPEC}" \
    "${WORKER_TORCHAUDIO_SPEC}" \
    "${WORKER_TORCHVISION_SPEC}" \
    "${WORKER_TORCH_INDEX_URL}"

  WORKER_EXTRAS="asr,gpu,response-model"
  if [ "${INSTALL_DEV}" = "1" ]; then
    WORKER_EXTRAS="dev,asr,gpu,response-model"
  fi
  install_project "${ACTIVE_WORKER_PYTHON}" "${WORKER_EXTRAS}"

  # Re-apply explicit worker torch overrides only when the caller requested a
  # custom worker torch stack.
  install_runtime_torch_stack \
    "worker" \
    "${ACTIVE_WORKER_PYTHON}" \
    "${WORKER_TORCH_SPEC}" \
    "${WORKER_TORCHAUDIO_SPEC}" \
    "${WORKER_TORCHVISION_SPEC}" \
    "${WORKER_TORCH_INDEX_URL}"
fi

if [ "${INSTALL_GPU}" = "1" ]; then
  ACTIVE_RESPONSE_MODEL_VENV_DIR="${ACTIVE_WORKER_VENV_DIR}"
  ACTIVE_RESPONSE_MODEL_PYTHON="${ACTIVE_WORKER_PYTHON}"
fi

if [ "${INSTALL_GPU}" = "1" ] && [ "${SEPARATE_COSYVOICE_VENV}" = "1" ]; then
  ensure_venv "${COSYVOICE_VENV_DIR}"
  ACTIVE_COSYVOICE_VENV_DIR="${COSYVOICE_VENV_DIR}"
  ACTIVE_COSYVOICE_PYTHON="${COSYVOICE_VENV_DIR}/bin/python"

  install_python_tooling "${ACTIVE_COSYVOICE_PYTHON}"
  install_runtime_torch_stack \
    "cosyvoice" \
    "${ACTIVE_COSYVOICE_PYTHON}" \
    "${WORKER_TORCH_SPEC}" \
    "${WORKER_TORCHAUDIO_SPEC}" \
    "${WORKER_TORCHVISION_SPEC}" \
    "${WORKER_TORCH_INDEX_URL}"
  install_cosyvoice_runtime "${ACTIVE_COSYVOICE_PYTHON}"
elif [ "${INSTALL_GPU}" = "1" ]; then
  install_cosyvoice_runtime "${ACTIVE_WORKER_PYTHON}"
  ACTIVE_COSYVOICE_VENV_DIR="${ACTIVE_WORKER_VENV_DIR}"
  ACTIVE_COSYVOICE_PYTHON="${ACTIVE_WORKER_PYTHON}"
fi

if [ "${INSTALL_GPU}" = "1" ] && [ "${ACTIVE_COSYVOICE_PYTHON}" != "${ACTIVE_WORKER_PYTHON}" ]; then
  # The worker owns the in-process TTS runtime, so it must be able to import
  # the CosyVoice stack even when we also maintain a dedicated CosyVoice venv.
  install_cosyvoice_runtime "${ACTIVE_WORKER_PYTHON}"
fi

printf '%s\n' "${ACTIVE_WORKER_VENV_DIR}" > "${WORKER_VENV_HINT_FILE}"
printf '%s\n' "${ACTIVE_RESPONSE_MODEL_VENV_DIR}" > "${RESPONSE_MODEL_VENV_HINT_FILE}"
printf '%s\n' "${ACTIVE_COSYVOICE_VENV_DIR}" > "${COSYVOICE_VENV_HINT_FILE}"

if [ "${INSTALL_GPU}" = "1" ]; then
  verify_response_model_runtime "${ACTIVE_RESPONSE_MODEL_PYTHON}"
fi

install_flash_attn "${ACTIVE_WORKER_PYTHON}"
if [ "${INSTALL_GPU}" = "1" ] && [ "${ACTIVE_COSYVOICE_PYTHON}" != "${ACTIVE_WORKER_PYTHON}" ]; then
  install_flash_attn "${ACTIVE_COSYVOICE_PYTHON}"
fi

if [ "${INSTALL_GPU}" = "1" ]; then
  verify_flash_attn "${ACTIVE_WORKER_PYTHON}"
  if [ "${ACTIVE_COSYVOICE_PYTHON}" != "${ACTIVE_WORKER_PYTHON}" ]; then
    verify_flash_attn "${ACTIVE_COSYVOICE_PYTHON}"
  fi
  verify_worker_runtime "${ACTIVE_WORKER_PYTHON}"
  verify_cosyvoice_runtime "${ACTIVE_WORKER_PYTHON}"
  verify_cosyvoice_runtime "${ACTIVE_COSYVOICE_PYTHON}"
  verify_asr_runtime "${ACTIVE_WORKER_PYTHON}"
fi

if [ "${INSTALL_FRONTEND}" = "1" ]; then
  ensure_node
  cd "${WEB_DIR}"
  if [ -f package-lock.json ]; then
    npm ci
  else
    npm install
  fi

  if [ "${BUILD_FRONTEND}" = "1" ]; then
    npm run build
  fi
fi

install_livekit_server

MODEL_DOWNLOAD_PROVIDER="${MODEL_DOWNLOAD_PROVIDER:-huggingface}"
DOWNLOAD_RESPONSE_MODEL="${DOWNLOAD_RESPONSE_MODEL:-1}"
DOWNLOAD_TTS_MODEL="${DOWNLOAD_TTS_MODEL:-1}"
DOWNLOAD_ASR_MODEL="${DOWNLOAD_ASR_MODEL:-1}"
DOWNLOAD_COSYVOICE_REPO="${DOWNLOAD_COSYVOICE_REPO:-1}"
VOSK_MODEL_NAME="${VOSK_MODEL_NAME:-vosk-model-small-en-us-0.15}"
VOSK_MODEL_PATH="${VOSK_MODEL_PATH:-${ROOT_DIR}/.models/${VOSK_MODEL_NAME}}"
VOSK_MODEL_URL="${VOSK_MODEL_URL:-https://alphacephei.com/vosk/models/${VOSK_MODEL_NAME}.zip}"
REFERENCE_AUDIO_PATH="${COSYVOICE3_SPEAKER_PATH:-}"

echo "Downloading cached model weights into ${ROOT_DIR}/.models"
DOWNLOAD_PROVIDER="${MODEL_DOWNLOAD_PROVIDER}" \
MODELS_ROOT="${ROOT_DIR}/.models" \
DOWNLOAD_RESPONSE_MODEL="${DOWNLOAD_RESPONSE_MODEL}" \
DOWNLOAD_TTS_MODEL="${DOWNLOAD_TTS_MODEL}" \
DOWNLOAD_ASR_MODEL="${DOWNLOAD_ASR_MODEL}" \
DOWNLOAD_COSYVOICE_REPO="${DOWNLOAD_COSYVOICE_REPO}" \
VOSK_MODEL_NAME="${VOSK_MODEL_NAME}" \
VOSK_MODEL_PATH="${VOSK_MODEL_PATH}" \
VOSK_MODEL_URL="${VOSK_MODEL_URL}" \
COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR}" \
COSYVOICE_REPO_URL="${COSYVOICE_REPO_URL}" \
COSYVOICE_REPO_REF="${COSYVOICE_REPO_REF}" \
bash "${ROOT_DIR}/scripts/download_model_weights.sh"

if [ -n "${REFERENCE_AUDIO_PATH}" ] && [ ! -f "${REFERENCE_AUDIO_PATH}" ]; then
  COSYVOICE_PROMPT_WAV="${COSYVOICE_REPO_DIR}/asset/zero_shot_prompt.wav"
  if [ -f "${COSYVOICE_PROMPT_WAV}" ]; then
    echo "Copying CosyVoice prompt wav to missing TTS reference path: ${REFERENCE_AUDIO_PATH}"
    mkdir -p "$(dirname "${REFERENCE_AUDIO_PATH}")"
    cp -f "${COSYVOICE_PROMPT_WAV}" "${REFERENCE_AUDIO_PATH}"
  else
    echo "Generating missing TTS reference wav at ${REFERENCE_AUDIO_PATH}"
    "${ACTIVE_WORKER_PYTHON}" "${ROOT_DIR}/scripts/generate_reference_wav.py" "${REFERENCE_AUDIO_PATH}"
  fi
fi

cat <<EOF
RunPod stack environment is ready.
API environment:
  source "${API_VENV_DIR}/bin/activate"
Single-process runtime environment:
  source "${ACTIVE_WORKER_VENV_DIR}/bin/activate"
ASR environment:
  source "${ACTIVE_WORKER_VENV_DIR}/bin/activate"
CosyVoice environment:
  source "${ACTIVE_COSYVOICE_VENV_DIR}/bin/activate"
LLM runtime:
  loaded in-process by the worker from "${ACTIVE_WORKER_VENV_DIR}"
Start single-process runtime with:
  ${ROOT_DIR}/scripts/start_voice_runtime_dev.sh
  launcher auto-resolves ${WORKER_VENV_HINT_FILE} when present
Isolation:
  SEPARATE_RESPONSE_MODEL_VENV=${SEPARATE_RESPONSE_MODEL_VENV}
  SEPARATE_WORKER_VENV=${SEPARATE_WORKER_VENV}
  SEPARATE_COSYVOICE_VENV=${SEPARATE_COSYVOICE_VENV}
  Worker launcher reads ${WORKER_VENV_HINT_FILE}
  CosyVoice launcher reads ${COSYVOICE_VENV_HINT_FILE}
HTTP UI and API:
  https://<pod-id>-8000.proxy.runpod.net
Optional Vite dev server:
  cd "${WEB_DIR}" && npm run dev -- --host 0.0.0.0 --port 5173
clean install:
  CLEAN_INSTALL=${CLEAN_INSTALL}
python tooling upgrade:
  UPGRADE_PYTHON_TOOLING=${UPGRADE_PYTHON_TOOLING}
flash-attn tuning:
  FLASH_ATTN_MAX_JOBS=${FLASH_ATTN_MAX_JOBS}
  FLASH_ATTN_NVCC_THREADS=${FLASH_ATTN_NVCC_THREADS}
EOF
