from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import time
from dataclasses import dataclass

from voice_pipeline.runtime.config import RuntimeConfig


class AdmissionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AdmissionConfig:
    require_avx2: bool = True
    max_socket_buffer_bytes: int = 262_144
    required_clock_source: str = "CLOCK_MONOTONIC"


@dataclass(frozen=True, slots=True)
class _GpuInventoryEntry:
    index: int
    total_memory_mib: int
    free_memory_mib: int
    uuid: str


@dataclass(frozen=True, slots=True)
class _GpuProcessEntry:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mib: int


_TRUTHY = {"1", "true", "yes", "on"}
_FORBIDDEN_DISTRIBUTED_ENV = (
    "RAY_ADDRESS",
    "RAY_RUNTIME_ENV",
    "VLLM_ALLOW_ENGINE_USE_RAY",
    "VLLM_USE_RAY",
    "VLLM_WORKER_USE_RAY",
    "VLLM_ENGINE_USE_RAY",
)


def _cpu_flags() -> str:
    if platform.system().lower() != "linux":
        return ""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            return handle.read().lower()
    except OSError:
        return ""


def _check_avx2(required: bool) -> None:
    if not required:
        return
    if platform.system().lower() == "linux" and "avx2" not in _cpu_flags():
        raise AdmissionError("hardware admission failed: avx2 unavailable")


def _check_clock(required_clock: str) -> None:
    resolved = str(required_clock).strip()
    if resolved == "CLOCK_MONOTONIC_RAW" and hasattr(time, "CLOCK_MONOTONIC_RAW"):
        _ = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        return
    if resolved == "CLOCK_MONOTONIC" and hasattr(time, "CLOCK_MONOTONIC"):
        _ = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        return
    if hasattr(time, "perf_counter_ns"):
        _ = time.perf_counter_ns()
        return
    raise AdmissionError(f"hardware admission failed: unsupported clock source {resolved}")


def _check_socket_limits(max_socket_buffer_bytes: int) -> None:
    configured = int(os.getenv("VOICE_PIPELINE_WS_WRITE_BUFFER_BYTES", "262144"))
    if configured > int(max_socket_buffer_bytes):
        raise AdmissionError("hardware admission failed: socket buffer too large")


def _check_forbidden_distributed_env() -> None:
    for key in _FORBIDDEN_DISTRIBUTED_ENV:
        value = str(os.getenv(key, "")).strip()
        if key in {"RAY_ADDRESS", "RAY_RUNTIME_ENV"} and value:
            raise AdmissionError(f"hardware admission failed: distributed env {key} must not be set")
        if value.lower() in _TRUTHY:
            raise AdmissionError(f"hardware admission failed: distributed env {key}={value} is forbidden")


def _require_path(label: str, path_value: str) -> None:
    resolved = str(path_value or "").strip()
    if not resolved:
        raise AdmissionError(f"hardware admission failed: {label} is not configured")
    if resolved.startswith("http://") or resolved.startswith("https://"):
        raise AdmissionError(f"hardware admission failed: {label} must be local/offline, got {resolved}")
    if not Path(resolved).exists():
        raise AdmissionError(f"hardware admission failed: {label} does not exist: {resolved}")


def _require_cache_dir(label: str, path_value: str) -> None:
    resolved = str(path_value or "").strip()
    if not resolved:
        raise AdmissionError(f"hardware admission failed: {label} is not configured")
    path = Path(resolved)
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise AdmissionError(f"hardware admission failed: {label} is not a directory: {resolved}")


def _require_value(label: str, value: str) -> None:
    resolved = str(value or "").strip()
    if not resolved:
        raise AdmissionError(f"hardware admission failed: {label} is not configured")


def _require_file(path: Path, *, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise AdmissionError(f"hardware admission failed: missing {label}: {path}")


def _require_any_glob(root: Path, *, patterns: tuple[str, ...], label: str) -> None:
    for pattern in patterns:
        if any(root.glob(pattern)):
            return
    raise AdmissionError(f"hardware admission failed: missing {label} under {root}")


def _check_vosk_artifacts(model_path: str) -> None:
    root = Path(str(model_path).strip())
    _require_file(root / "am" / "final.mdl", label="Vosk acoustic model")
    _require_file(root / "conf" / "model.conf", label="Vosk model config")
    _require_any_glob(
        root,
        patterns=("graph/HCLr.fst", "graph/HCLG.fst"),
        label="Vosk graph fst",
    )


def _check_vllm_artifacts(model_path: str) -> None:
    root = Path(str(model_path).strip())
    _require_file(root / "config.json", label="vLLM model config.json")
    _require_any_glob(
        root,
        patterns=(
            "model.safetensors",
            "model-*.safetensors",
            "pytorch_model.bin",
            "pytorch_model-*.bin",
        ),
        label="vLLM model weight shards",
    )
    _require_any_glob(
        root,
        patterns=("tokenizer.json", "tokenizer.model", "vocab.json"),
        label="vLLM tokenizer artifacts",
    )


def _check_cosyvoice3_artifacts(model_path: str) -> None:
    root = Path(str(model_path).strip())
    _require_file(root / "cosyvoice3.yaml", label="CosyVoice3 cosyvoice3.yaml")
    _require_file(root / "llm.pt", label="CosyVoice3 llm.pt")
    _require_file(root / "flow.pt", label="CosyVoice3 flow.pt")
    _require_file(root / "hift.pt", label="CosyVoice3 hift.pt")


def _check_optional_speaker_asset(path_value: str) -> None:
    resolved = str(path_value or "").strip()
    if not resolved:
        return
    path = Path(resolved)
    if not path.exists() or not path.is_file():
        raise AdmissionError(f"hardware admission failed: CosyVoice3 speaker asset missing: {resolved}")


def _check_cuda_device(label: str, device_name: str) -> None:
    device_text = str(device_name or "").strip().lower()
    if not device_text.startswith("cuda:"):
        raise AdmissionError(f"hardware admission failed: {label} must bind to cuda device, got {device_name}")
    device_index = int(device_text.split(":", 1)[1])
    if device_index >= _visible_cuda_device_count():
        raise AdmissionError(f"hardware admission failed: {label} device {device_name} missing")
    inventory = _gpu_inventory()
    device = inventory.get(device_index)
    if device is None:
        raise AdmissionError(
            f"hardware admission failed: {label} device {device_name} missing from nvidia-smi inventory"
        )
    active_processes = _gpu_compute_processes().get(device.uuid, ())
    if active_processes:
        process_summary = ", ".join(
            f"pid={entry.pid} name={entry.process_name} mem={entry.used_memory_mib}MiB"
            for entry in active_processes
        )
        raise AdmissionError(
            "hardware admission failed: "
            f"{label} device {device_name} busy "
            f"(free={device.free_memory_mib}MiB total={device.total_memory_mib}MiB, active_compute={process_summary})"
        )


def _visible_cuda_device_count() -> int:
    output = _run_nvidia_smi(["--list-gpus"])
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return len(lines)


def _gpu_inventory() -> dict[int, _GpuInventoryEntry]:
    output = _run_nvidia_smi(
        ["--query-gpu=index,memory.total,memory.free,gpu_uuid", "--format=csv,noheader,nounits"]
    )
    inventory: dict[int, _GpuInventoryEntry] = {}
    for raw_line in output.splitlines():
        line = str(raw_line).strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 4:
            raise AdmissionError(f"hardware admission failed: unexpected nvidia-smi gpu row: {line}")
        entry = _GpuInventoryEntry(
            index=int(parts[0]),
            total_memory_mib=int(parts[1]),
            free_memory_mib=int(parts[2]),
            uuid=str(parts[3]),
        )
        inventory[entry.index] = entry
    return inventory


def _gpu_compute_processes() -> dict[str, tuple[_GpuProcessEntry, ...]]:
    try:
        output = _run_nvidia_smi(
            [
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except AdmissionError as exc:
        message = str(exc)
        if "No running processes found" in message:
            return {}
        raise
    grouped: dict[str, list[_GpuProcessEntry]] = {}
    for raw_line in output.splitlines():
        line = str(raw_line).strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 4:
            raise AdmissionError(f"hardware admission failed: unexpected nvidia-smi compute row: {line}")
        entry = _GpuProcessEntry(
            gpu_uuid=str(parts[0]),
            pid=int(parts[1]),
            process_name=str(parts[2]),
            used_memory_mib=int(parts[3]),
        )
        grouped.setdefault(entry.gpu_uuid, []).append(entry)
    return {gpu_uuid: tuple(entries) for gpu_uuid, entries in grouped.items()}


def _run_nvidia_smi(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except FileNotFoundError as exc:
        raise AdmissionError("hardware admission failed: nvidia-smi is unavailable") from exc
    except subprocess.TimeoutExpired as exc:
        raise AdmissionError("hardware admission failed: nvidia-smi timed out") from exc
    except subprocess.CalledProcessError as exc:
        stderr = str(exc.stderr or "").strip()
        raise AdmissionError(
            "hardware admission failed: nvidia-smi failed"
            + (f": {stderr}" if stderr else "")
        ) from exc
    stdout = str(completed.stdout or "").strip()
    stderr = str(completed.stderr or "").strip()
    if stdout:
        return stdout
    if stderr:
        return stderr
    return ""


def _check_livekit_config(config: RuntimeConfig) -> None:
    _require_value("LiveKit URL", config.livekit_url)
    _require_value("LiveKit API key", config.livekit_api_key)
    _require_value("LiveKit API secret", config.livekit_api_secret)
    _require_value("LiveKit room name", config.livekit_room_name)
    _require_value("LiveKit runtime identity", config.livekit_runtime_identity)
    resolved_url = str(config.livekit_url).strip().lower()
    if not (resolved_url.startswith("ws://") or resolved_url.startswith("wss://")):
        raise AdmissionError(f"hardware admission failed: LiveKit URL must be ws:// or wss://, got {config.livekit_url}")


def hardware_admission_check(
    config: RuntimeConfig,
    admission: AdmissionConfig | None = None,
) -> None:
    cfg = admission or AdmissionConfig()
    _check_avx2(required=cfg.require_avx2)
    _check_clock(required_clock=cfg.required_clock_source)
    _check_socket_limits(max_socket_buffer_bytes=cfg.max_socket_buffer_bytes)
    _check_forbidden_distributed_env()
    _require_path("vosk model path", config.asr_model_path)
    _require_path("vLLM model path", config.resolved_vllm_model_path())
    _require_path("CosyVoice3 model path", config.resolved_cosyvoice3_model_path())
    _check_vosk_artifacts(config.asr_model_path)
    _check_vllm_artifacts(config.resolved_vllm_model_path())
    _check_cosyvoice3_artifacts(config.resolved_cosyvoice3_model_path())
    _check_optional_speaker_asset(config.cosyvoice3_speaker_path)
    _require_cache_dir("vLLM cache directory", config.vllm_cache_dir)
    _require_cache_dir("CosyVoice3 cache directory", config.cosyvoice3_cache_dir)
    if str(config.asr_device).strip().lower() != "cpu":
        raise AdmissionError(f"hardware admission failed: ASR device must be cpu, got {config.asr_device}")
    if str(config.llm_device).strip().lower() != "cuda:0":
        raise AdmissionError(f"hardware admission failed: vLLM device must be cuda:0, got {config.llm_device}")
    if str(config.tts_device).strip().lower() != "cuda:1":
        raise AdmissionError(f"hardware admission failed: CosyVoice3 device must be cuda:1, got {config.tts_device}")
    if _visible_cuda_device_count() < 2:
        raise AdmissionError("hardware admission failed: fewer than 2 CUDA devices are visible")
    _check_cuda_device("vLLM", config.llm_device)
    _check_cuda_device("CosyVoice3", config.tts_device)
    _check_livekit_config(config)


__all__ = ["AdmissionConfig", "AdmissionError", "hardware_admission_check"]
