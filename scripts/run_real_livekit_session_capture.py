from __future__ import annotations

import argparse
import asyncio
import json
import os
import site
import sys
import time
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path
from urllib.error import HTTPError

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "apps" / "api" / "src"
_ENV_LOADED = False


def _load_env_file_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_file = str(os.getenv("VOICE_PIPELINE_ENV_FILE", str(REPO_ROOT / ".env.voice_pipeline"))).strip()
    if env_file:
        path = Path(env_file)
        if path.exists() and path.is_file():
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                os.environ.setdefault(key, value.strip().strip("\"'"))
    _ENV_LOADED = True


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


def _ensure_runtime_library_path() -> None:
    marker = "VOICE_PIPELINE_LD_LIBRARY_PATH_READY"
    if os.getenv(marker) == "1":
        return
    resolved = _resolved_runtime_library_path()
    if not resolved:
        return
    existing = [entry for entry in str(os.getenv("LD_LIBRARY_PATH", "")).split(os.pathsep) if entry]
    needed = [entry for entry in resolved.split(os.pathsep) if entry]
    if all(entry in existing for entry in needed):
        os.environ[marker] = "1"
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([*needed, *existing])
    os.environ[marker] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ))


_ensure_runtime_library_path()
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from livekit import rtc  # type: ignore
from livekit.rtc._proto import room_pb2  # type: ignore

from voice_pipeline.shared.audio_resample import StreamingAudioResampler


def _env(key: str, default: str) -> str:
    _load_env_file_once()
    return str(os.getenv(key, default)).strip() or default


def _track_source_from_env(raw_value: str) -> int:
    source_name = str(raw_value or "").strip().upper()
    if not source_name:
        return int(rtc.TrackSource.SOURCE_UNKNOWN)
    if not source_name.startswith("SOURCE_"):
        source_name = f"SOURCE_{source_name}"
    try:
        return int(rtc.TrackSource.Value(source_name))
    except Exception as exc:
        raise RuntimeError(f"unsupported VOICE_PIPELINE_INPUT_TRACK_SOURCE: {raw_value}") from exc


def _audio_encoding_from_env(raw_value: str) -> room_pb2.AudioEncoding | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    max_bitrate = int(value)
    if max_bitrate <= 0:
        return None
    encoding = room_pb2.AudioEncoding()
    encoding.max_bitrate = int(max_bitrate)
    return encoding


def _queue_size_ms_from_env(raw_value: str, *, default: int) -> int:
    value = str(raw_value or "").strip()
    if not value:
        return int(default)
    resolved = int(value)
    return max(0, int(resolved))


def _preroll_mode_from_env(raw_value: str) -> str:
    value = str(raw_value or "").strip().lower()
    if not value:
        return "silence"
    if value not in {"silence", "noise"}:
        raise RuntimeError(
            f"unsupported VOICE_PIPELINE_PUBLISH_PREROLL_MODE: {raw_value}"
        )
    return value


def _preroll_noise_i16_from_env(raw_value: str, *, default: int) -> int:
    value = str(raw_value or "").strip()
    if not value:
        return int(default)
    resolved = abs(int(value))
    return min(32767, max(0, resolved))


def _bool_from_env(raw_value: str, *, default: bool) -> bool:
    value = str(raw_value or "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}


def _optional_bool_from_env(raw_value: str) -> bool | None:
    value = str(raw_value or "").strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"unsupported boolean env value: {raw_value}")


def _bool_publish_option_from_env(raw_value: str, *, default: bool) -> bool:
    value = str(raw_value or "").strip().lower()
    if not value:
        return bool(default)
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"unsupported boolean publish option: {raw_value}")


def _required_path(env_key: str, default_path: Path) -> Path:
    path = Path(str(os.getenv(env_key, str(default_path))).strip()).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"{env_key} does not exist: {path}")
    return path


def _http_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def _http_post_json(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    body = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


async def _wait_for_runtime_ready(base_url: str, *, timeout_seconds: float) -> dict[str, object]:
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    last_payload: dict[str, object] = {}
    consecutive_ready = 0
    while time.monotonic() < deadline:
        try:
            payload = _http_json(f"{base_url.rstrip('/')}/v1/system/readiness")
        except Exception:
            consecutive_ready = 0
            await asyncio.sleep(1.0)
            continue
        last_payload = payload
        if bool(payload.get("ready")):
            consecutive_ready += 1
            if consecutive_ready >= 2:
                return payload
        else:
            consecutive_ready = 0
        await asyncio.sleep(1.0)
    raise RuntimeError(f"runtime did not become ready within {int(timeout_seconds)}s: {last_payload}")


async def _wait_for_runtime_available(base_url: str, *, timeout_seconds: float) -> dict[str, object]:
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    last_payload: dict[str, object] = {}
    consecutive_available = 0
    while time.monotonic() < deadline:
        try:
            payload = _http_json(f"{base_url.rstrip('/')}/v1/system/runtime?limit=5")
        except Exception:
            consecutive_available = 0
            await asyncio.sleep(1.0)
            continue
        last_payload = payload
        if bool(payload.get("available")):
            consecutive_available += 1
            if consecutive_available >= 2:
                return payload
        else:
            consecutive_available = 0
        await asyncio.sleep(1.0)
    raise RuntimeError(f"runtime did not become available within {int(timeout_seconds)}s: {last_payload}")


async def _post_json_when_runtime_ready(
    base_url: str,
    path: str,
    *,
    payload: dict[str, object] | None = None,
    timeout_seconds: float,
) -> dict[str, object]:
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    last_error = ""
    while time.monotonic() < deadline:
        try:
            return _http_post_json(f"{base_url.rstrip('/')}{path}", payload)
        except HTTPError as exc:
            if int(exc.code) not in {503, 500}:
                raise
            last_error = f"http_{exc.code}"
        except Exception as exc:
            last_error = str(exc)
        await _wait_for_runtime_ready(base_url, timeout_seconds=min(30.0, max(1.0, deadline - time.monotonic())))
        await _wait_for_runtime_available(base_url, timeout_seconds=min(30.0, max(1.0, deadline - time.monotonic())))
        await asyncio.sleep(0.25)
    raise RuntimeError(f"runtime endpoint {path} did not succeed within {int(timeout_seconds)}s: {last_error}")


async def _wait_for_runtime_ingress_lock(
    base_url: str,
    *,
    timeout_seconds: float,
    participant_identity: str,
    track_name: str,
) -> dict[str, object]:
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    last_payload: dict[str, object] = {}
    expected_identity = str(participant_identity or "").strip()
    expected_track_name = str(track_name or "").strip()
    while time.monotonic() < deadline:
        try:
            payload = _http_json(f"{base_url.rstrip('/')}/v1/system/runtime?limit=20")
        except Exception:
            await asyncio.sleep(0.1)
            continue
        last_payload = payload
        bridge = dict(payload.get("bridge", {}))
        if (
            bool(bridge.get("ingress_lock_active"))
            and str(bridge.get("ingress_participant_identity", "")).strip() == expected_identity
            and str(bridge.get("ingress_track_name", "")).strip() == expected_track_name
        ):
            return payload
        if (
            str(bridge.get("last_lock_participant_identity", "")).strip() == expected_identity
            and str(bridge.get("last_lock_track_name", "")).strip() == expected_track_name
            and int(bridge.get("last_lock_acquired_ns", 0) or 0) > 0
        ):
            return payload
        await asyncio.sleep(0.1)
    raise RuntimeError(
        "runtime ingress lock did not match input track within "
        f"{timeout_seconds:.1f}s: {last_payload.get('bridge', {})}"
    )


def _fetch_livekit_session(base_url: str, *, identity: str) -> dict[str, object]:
    query = urllib.parse.urlencode({"identity": identity})
    return _http_json(f"{base_url.rstrip('/')}/v1/livekit/token?{query}")


def _load_publish_frames(*, wav_path: Path, target_rate: int, channels: int, frame_ms: int) -> tuple[bytes, ...]:
    with wave.open(str(wav_path), "rb") as handle:
        source_channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        source_rate = int(handle.getframerate())
        raw = handle.readframes(int(handle.getnframes()))
    if source_channels != 1:
        raise RuntimeError(f"publish wav must be mono, got channels={source_channels}")
    if sample_width != 2:
        raise RuntimeError(f"publish wav must be PCM16, got sample_width={sample_width}")
    source = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if source_rate != int(target_rate):
        source = StreamingAudioResampler(target_rate=int(target_rate)).resample(source, int(source_rate))
    pcm = np.clip(source * 32767.0, -32768.0, 32767.0).astype("<i2").tobytes()
    frame_bytes = max(2, int(int(target_rate) * int(frame_ms) / 1000) * int(channels) * 2)
    frames: list[bytes] = []
    for offset in range(0, len(pcm), frame_bytes):
        chunk = pcm[offset : offset + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + (b"\x00" * (frame_bytes - len(chunk)))
        frames.append(chunk)
    if not frames:
        raise RuntimeError("publish wav produced zero frames")
    return tuple(frames)


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frame_rate = int(handle.getframerate())
        frame_count = int(handle.getnframes())
    if frame_rate <= 0:
        return 0.0
    return float(frame_count) / float(frame_rate)


def _write_pcm16_wav(*, path: Path, pcm_bytes: bytes, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(channels))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm_bytes)


def _runtime_snapshot_path_for_capture(capture_wav: Path) -> Path:
    stem = capture_wav.stem.strip() or "capture"
    return capture_wav.with_name(f"runtime_after_{stem}.json")


async def _capture_frame_realtime(
    *,
    source: object,
    payload: bytes,
    sample_rate: int,
    channels: int,
    samples_per_channel: int,
    next_deadline_monotonic: float,
    frame_duration_s: float,
) -> float:
    sleep_for = float(next_deadline_monotonic) - float(time.monotonic())
    if sleep_for > 0.0:
        await asyncio.sleep(sleep_for)
    frame = rtc.AudioFrame(
        data=payload,
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=samples_per_channel,
    )
    await source.capture_frame(frame)
    return float(next_deadline_monotonic) + float(frame_duration_s)


async def _capture_frame_realtime_direct10ms(
    *,
    source: object,
    payload: bytes,
    sample_rate: int,
    channels: int,
    next_deadline_monotonic: float,
    frame_duration_s: float,
) -> float:
    target_samples_per_channel = max(1, int(round(float(sample_rate) * 0.01)))
    frame_bytes = max(2 * channels, int(target_samples_per_channel) * channels * 2)
    deadline = float(next_deadline_monotonic)
    for offset in range(0, len(payload), frame_bytes):
        sleep_for = float(deadline) - float(time.monotonic())
        if sleep_for > 0.0:
            await asyncio.sleep(sleep_for)
        chunk = bytes(payload[offset : offset + frame_bytes])
        if len(chunk) < frame_bytes:
            chunk = chunk + (b"\x00" * (frame_bytes - len(chunk)))
        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=sample_rate,
            num_channels=channels,
            samples_per_channel=target_samples_per_channel,
        )
        await source.capture_frame(frame)
        deadline += 0.01
    return deadline


def _noise_payload(*, byte_length: int, amplitude_i16: int, rng: np.random.Generator) -> bytes:
    if byte_length <= 0 or amplitude_i16 <= 0:
        return b""
    sample_count = int(byte_length // 2)
    if sample_count <= 0:
        return b""
    noise = rng.integers(
        low=-int(amplitude_i16),
        high=int(amplitude_i16) + 1,
        size=sample_count,
        dtype=np.int16,
    )
    return noise.astype("<i2", copy=False).tobytes()


def _audio_metrics(*, pcm_bytes: bytes, sample_rate: int) -> dict[str, float]:
    if not pcm_bytes:
        return {"duration_s": 0.0, "rms": 0.0, "peak": 0.0, "silence_ratio": 1.0}
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    duration_s = float(samples.shape[0]) / float(sample_rate)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    silence_ratio = float(np.mean(np.abs(samples) < 0.005)) if samples.size else 1.0
    return {
        "duration_s": duration_s,
        "rms": rms,
        "peak": peak,
        "silence_ratio": silence_ratio,
    }


def _last_voiced_frame_index(*, frames: tuple[bytes, ...], rms_threshold: float) -> int:
    threshold = max(0.0, float(rms_threshold))
    last_index = -1
    for index, frame_bytes in enumerate(frames):
        if _frame_rms(frame_bytes) >= threshold:
            last_index = int(index)
    if last_index >= 0:
        return int(last_index)
    return max(0, len(frames) - 1)


def _frame_rms(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def _runtime_turn_settled(snapshot: dict[str, object]) -> bool:
    kernel_state = dict(snapshot.get("kernel_state", {}) or {})
    phase = str(kernel_state.get("phase", "")).strip().lower()
    active_vllm_request_id = str(snapshot.get("active_vllm_request_id", "") or "").strip()
    active_tts_request_id = str(snapshot.get("active_tts_request_id", "") or "").strip()
    return phase == "idle" and not active_vllm_request_id and not active_tts_request_id


def _frame_bytes_and_stats(frame: object) -> tuple[bytes, dict[str, float | int]]:
    raw_data = getattr(frame, "_data", None)
    data_obj = getattr(frame, "data", None)

    payload_variants: dict[str, bytes] = {}
    if raw_data is not None:
        payload_variants["raw_data"] = bytes(raw_data)
    if data_obj is not None:
        payload_variants["data_cast_b"] = bytes(memoryview(data_obj).cast("B"))
        payload_variants["data_np_i2"] = np.asarray(data_obj, dtype=np.int16).astype("<i2", copy=False).tobytes()

    canonical = next((variant for variant in payload_variants.values() if variant), b"")
    stats: dict[str, float | int] = {
        "sample_rate": int(getattr(frame, "sample_rate", 0) or 0),
        "num_channels": int(getattr(frame, "num_channels", 0) or 0),
        "samples_per_channel": int(getattr(frame, "samples_per_channel", 0) or 0),
        "payload_bytes": int(len(canonical)),
        "variant_count": int(len(payload_variants)),
    }
    for name, variant in payload_variants.items():
        stats[f"{name}_bytes"] = int(len(variant))
        stats[f"{name}_rms"] = float(_frame_rms(variant))
        if variant:
            samples = np.frombuffer(variant, dtype="<i2").astype(np.float32) / 32768.0
            stats[f"{name}_peak"] = float(np.max(np.abs(samples))) if samples.size else 0.0
    return canonical, stats


async def _run() -> int:
    base_url = _env("VOICE_PIPELINE_RUNTIME_BASE_URL", "http://127.0.0.1:8000")
    configured_identity = _env("VOICE_PIPELINE_SESSION_CLIENT_IDENTITY", "")
    run_suffix = uuid.uuid4().hex[:8]
    identity = configured_identity or f"voice-test-client-{run_suffix}"
    input_track_name = _env("VOICE_PIPELINE_INPUT_TRACK_NAME", "") or f"voice-test-input-{run_suffix}"
    publish_wav = _required_path(
        "VOICE_PIPELINE_REAL_SMOKE_WAV",
        REPO_ROOT / ".run" / "flite_hello_there.wav",
    )
    capture_wav = Path(
        str(
            os.getenv(
                "VOICE_PIPELINE_CAPTURE_WAV",
                str(REPO_ROOT / ".run" / "livekit_session_capture.wav"),
            )
        ).strip()
    ).expanduser().resolve()
    ready_timeout_seconds = float(_env("VOICE_PIPELINE_READY_TIMEOUT_SECONDS", "900"))
    session_timeout_seconds = float(_env("VOICE_PIPELINE_SESSION_TIMEOUT_SECONDS", "120"))
    sample_rate = int(_env("VOICE_PIPELINE_SESSION_SAMPLE_RATE", "48000"))
    channels = int(_env("VOICE_PIPELINE_SESSION_CHANNELS", "1"))
    frame_ms = int(_env("VOICE_PIPELINE_SESSION_FRAME_MS", "20"))
    expected_remote_identity = _env("VOICE_PIPELINE_EXPECTED_REMOTE_IDENTITY", "voice-runtime-backend")
    expected_remote_track_name = _env("VOICE_PIPELINE_EXPECTED_REMOTE_TRACK_NAME", "voice-runtime-out")
    max_capture_seconds = float(_env("VOICE_PIPELINE_MAX_CAPTURE_SECONDS", "12"))
    idle_after_first_audible_seconds = float(_env("VOICE_PIPELINE_IDLE_AFTER_AUDIBLE_SECONDS", "0.75"))
    runtime_settled_grace_seconds = float(_env("VOICE_PIPELINE_RUNTIME_SETTLED_GRACE_SECONDS", "0.75"))
    runtime_settled_poll_seconds = float(_env("VOICE_PIPELINE_RUNTIME_SETTLED_POLL_SECONDS", "0.25"))
    post_capture_settle_ms = float(_env("VOICE_PIPELINE_POST_CAPTURE_SETTLE_MS", "500"))
    publish_trailing_silence_ms = float(_env("VOICE_PIPELINE_PUBLISH_TRAILING_SILENCE_MS", "300"))
    publish_preroll_ms = float(_env("VOICE_PIPELINE_PUBLISH_PREROLL_MS", "0"))
    input_voiced_rms_threshold = float(_env("VOICE_PIPELINE_INPUT_VOICED_RMS_THRESHOLD", "0.01"))
    publish_preroll_mode = _preroll_mode_from_env(_env("VOICE_PIPELINE_PUBLISH_PREROLL_MODE", "silence"))
    publish_preroll_noise_i16 = _preroll_noise_i16_from_env(
        _env("VOICE_PIPELINE_PUBLISH_PREROLL_NOISE_I16", "3"),
        default=3,
    )
    wait_for_input_subscription = _bool_from_env(
        _env("VOICE_PIPELINE_WAIT_FOR_INPUT_SUBSCRIPTION", "0"),
        default=False,
    )
    wait_for_runtime_ingress_lock = _bool_from_env(
        _env("VOICE_PIPELINE_WAIT_FOR_RUNTIME_INGRESS_LOCK", "1"),
        default=True,
    )
    input_subscription_timeout_seconds = float(
        _env("VOICE_PIPELINE_INPUT_SUBSCRIPTION_TIMEOUT_SECONDS", "5")
    )
    runtime_ingress_lock_timeout_seconds = float(
        _env("VOICE_PIPELINE_RUNTIME_INGRESS_LOCK_TIMEOUT_SECONDS", "5")
    )
    input_track_source = _track_source_from_env(_env("VOICE_PIPELINE_INPUT_TRACK_SOURCE", "MICROPHONE"))
    input_audio_encoding = _audio_encoding_from_env(_env("VOICE_PIPELINE_INPUT_AUDIO_MAX_BITRATE", "0"))
    input_queue_size_ms = _queue_size_ms_from_env(
        _env("VOICE_PIPELINE_INPUT_QUEUE_SIZE_MS", _env("LIVEKIT_INPUT_QUEUE_SIZE_MS", "40")),
        default=40,
    )
    input_preconnect_buffer = _bool_publish_option_from_env(
        _env("VOICE_PIPELINE_INPUT_PRECONNECT_BUFFER", "0"),
        default=False,
    )
    single_peer_connection = _optional_bool_from_env(os.getenv("VOICE_PIPELINE_SINGLE_PEER_CONNECTION", ""))
    capture_silence_rms_threshold = float(_env("VOICE_PIPELINE_CAPTURE_SILENCE_RMS_THRESHOLD", "0.0015"))
    close_source_after_publish = _env("VOICE_PIPELINE_CLOSE_SOURCE_AFTER_PUBLISH", "1").lower() in {"1", "true", "yes", "on"}
    input_playout_timeout_seconds = float(_env("VOICE_PIPELINE_INPUT_PLAYOUT_TIMEOUT_SECONDS", "5"))
    run_runtime_warmup = _env("VOICE_PIPELINE_RUN_RUNTIME_WARMUP", "1").lower() in {"1", "true", "yes", "on"}
    subscribe_native_stream = _env("VOICE_PIPELINE_SUBSCRIBE_NATIVE_STREAM", "0").lower() in {"1", "true", "yes", "on"}
    debug_ingress_wav = str(os.getenv("VOICE_PIPELINE_DEBUG_SAVE_INGRESS_WAV", "")).strip()
    debug_ingress_seconds = float(str(os.getenv("VOICE_PIPELINE_DEBUG_SAVE_INGRESS_SECONDS", "0")).strip() or "0")

    print("livekit-session-capture: waiting for runtime readiness", flush=True)
    readiness = await _wait_for_runtime_ready(base_url, timeout_seconds=ready_timeout_seconds)
    await _wait_for_runtime_available(base_url, timeout_seconds=ready_timeout_seconds)
    print(
        "livekit-session-capture: runtime ready",
        {
            "summary": readiness.get("summary"),
            "transport_status": readiness.get("transport_status"),
        },
        flush=True,
    )
    print(
        "livekit-session-capture: configuring runtime ingress filter",
        {"identity": identity, "track_name": input_track_name},
        flush=True,
    )
    ingress_filter = await _post_json_when_runtime_ready(
        base_url,
        "/v1/system/ingress-filter",
        payload={"identity": identity, "track_name": input_track_name},
        timeout_seconds=ready_timeout_seconds,
    )
    print("livekit-session-capture: ingress filter configured", ingress_filter, flush=True)
    print("livekit-session-capture: resetting runtime session state", flush=True)
    reset = await _post_json_when_runtime_ready(
        base_url,
        "/v1/system/reset",
        timeout_seconds=ready_timeout_seconds,
    )
    print("livekit-session-capture: runtime reset complete", reset, flush=True)
    readiness = await _wait_for_runtime_ready(base_url, timeout_seconds=ready_timeout_seconds)
    await _wait_for_runtime_available(base_url, timeout_seconds=ready_timeout_seconds)
    if debug_ingress_wav and debug_ingress_seconds > 0.0:
        print(
            "livekit-session-capture: enabling ingress debug capture",
            {"path": debug_ingress_wav, "seconds": debug_ingress_seconds},
            flush=True,
        )
        ingress_debug_url = (
            f"{base_url.rstrip('/')}/v1/system/ingress-debug?"
            f"path={urllib.parse.quote(debug_ingress_wav, safe='')}&seconds={debug_ingress_seconds}"
        )
        ingress_debug = _http_post_json(ingress_debug_url)
        print("livekit-session-capture: ingress debug capture configured", ingress_debug, flush=True)
    if run_runtime_warmup:
        print("livekit-session-capture: running runtime warmup", flush=True)
        try:
            warmup = await _post_json_when_runtime_ready(
                base_url,
                "/v1/system/warmup",
                timeout_seconds=ready_timeout_seconds,
            )
        except HTTPError as exc:
            raise RuntimeError(f"runtime warmup request failed: http {exc.code}") from exc
        print("livekit-session-capture: runtime warmup complete", warmup, flush=True)

    session = _fetch_livekit_session(base_url, identity=identity)
    livekit_url = str(session.get("url", "")).strip()
    token = str(session.get("token", "")).strip()
    room_name = str(session.get("room_name", "")).strip()
    if not livekit_url or not token or not room_name:
        raise RuntimeError(f"invalid livekit session payload: {session}")

    frames = _load_publish_frames(
        wav_path=publish_wav,
        target_rate=sample_rate,
        channels=channels,
        frame_ms=frame_ms,
    )
    last_voiced_input_frame_index = _last_voiced_frame_index(
        frames=frames,
        rms_threshold=input_voiced_rms_threshold,
    )
    input_duration_s = _wav_duration_seconds(publish_wav)

    room = rtc.Room()
    output_pcm = bytearray()
    output_frames = 0
    output_sample_rate = sample_rate
    output_channels = channels
    first_publish_started_monotonic = 0.0
    input_speech_end_monotonic = 0.0
    first_output_monotonic = 0.0
    first_payload_monotonic = 0.0
    last_payload_monotonic = 0.0
    last_audible_monotonic = 0.0
    first_output_event = asyncio.Event()
    capture_done_event = asyncio.Event()
    subscribed_tracks: list[dict[str, object]] = []
    matched_track_seen = False
    max_received_frame_rms = 0.0
    max_received_frame_peak = 0.0
    first_nonzero_frame_stats: dict[str, float | int] | None = None
    runtime_settled_since_monotonic = 0.0
    last_runtime_poll_monotonic = 0.0

    async def _consume_remote_audio(track: rtc.Track) -> None:
        nonlocal first_output_monotonic, first_payload_monotonic, last_payload_monotonic, last_audible_monotonic
        nonlocal output_frames, output_sample_rate, output_channels
        nonlocal max_received_frame_rms, max_received_frame_peak, first_nonzero_frame_stats
        stream_kwargs: dict[str, object] = {"track": track}
        if not subscribe_native_stream:
            stream_kwargs.update(
                sample_rate=sample_rate,
                num_channels=channels,
                frame_size_ms=frame_ms,
            )
        stream = rtc.AudioStream(**stream_kwargs)
        remote_frame_duration_s = 0.0
        async for frame_event in stream:
            frame = getattr(frame_event, "frame", frame_event)
            payload, frame_stats = _frame_bytes_and_stats(frame)
            if not payload:
                continue
            if output_frames == 0:
                if int(frame_stats.get("sample_rate", 0) or 0) > 0:
                    output_sample_rate = int(frame_stats["sample_rate"])
                if int(frame_stats.get("num_channels", 0) or 0) > 0:
                    output_channels = int(frame_stats["num_channels"])
                remote_sample_rate = int(frame_stats.get("sample_rate", 0) or 0)
                remote_samples_per_channel = int(frame_stats.get("samples_per_channel", 0) or 0)
                if remote_sample_rate > 0 and remote_samples_per_channel > 0:
                    remote_frame_duration_s = float(remote_samples_per_channel) / float(remote_sample_rate)
                print("livekit-session-capture: first remote frame", frame_stats, flush=True)
            frame_rms = _frame_rms(payload)
            samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
            frame_peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            max_received_frame_rms = max(float(max_received_frame_rms), float(frame_rms))
            max_received_frame_peak = max(float(max_received_frame_peak), float(frame_peak))
            if first_nonzero_frame_stats is None and frame_peak > 0.0:
                first_nonzero_frame_stats = {**frame_stats, "frame_rms": float(frame_rms), "frame_peak": float(frame_peak)}
                print("livekit-session-capture: first nonzero remote frame", first_nonzero_frame_stats, flush=True)
            if first_payload_monotonic <= 0.0:
                first_payload_monotonic = time.monotonic()
                first_output_event.set()
            last_payload_monotonic = time.monotonic()
            if first_output_monotonic <= 0.0 and frame_rms > capture_silence_rms_threshold:
                first_output_monotonic = time.monotonic()
                last_audible_monotonic = float(first_output_monotonic)
                print(
                    "livekit-session-capture: detected first output frame",
                    {"frame_rms": round(frame_rms, 6), "frame_peak": round(frame_peak, 6), "output_frames": output_frames},
                    flush=True,
                )
                first_output_event.set()
            elif frame_rms > capture_silence_rms_threshold:
                last_audible_monotonic = time.monotonic()
            output_pcm.extend(payload)
            output_frames += 1
            if remote_frame_duration_s > 0.0:
                captured_seconds = float(output_frames) * float(remote_frame_duration_s)
            else:
                captured_seconds = float(output_frames) * (float(frame_ms) / 1000.0)
            if captured_seconds >= max_capture_seconds:
                capture_done_event.set()
                break

    @room.on("track_subscribed")
    def _on_track_subscribed(track: rtc.Track, publication: object, participant: object) -> None:
        nonlocal matched_track_seen
        if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        participant_identity = str(getattr(participant, "identity", "")).strip()
        publication_name = str(getattr(publication, "name", "") or getattr(track, "name", "")).strip()
        publication_sid = str(getattr(publication, "sid", "")).strip()
        track_info = {
            "participant_identity": participant_identity,
            "publication_name": publication_name,
            "publication_sid": publication_sid,
        }
        subscribed_tracks.append(track_info)
        print("livekit-session-capture: track subscribed", track_info, flush=True)
        if participant_identity == identity:
            return
        if participant_identity != expected_remote_identity:
            return
        if expected_remote_track_name and publication_name and publication_name != expected_remote_track_name:
            return
        matched_track_seen = True
        asyncio.create_task(_consume_remote_audio(track))

    room_options = rtc.RoomOptions(auto_subscribe=True)
    if single_peer_connection is not None:
        room_options.single_peer_connection = bool(single_peer_connection)
    print(
        "livekit-session-capture: connecting room",
        {"url": livekit_url, "room": room_name, "single_peer_connection": single_peer_connection},
        flush=True,
    )
    await room.connect(livekit_url, token, options=room_options)

    source = rtc.AudioSource(sample_rate, channels, queue_size_ms=input_queue_size_ms)
    track = rtc.LocalAudioTrack.create_audio_track(input_track_name, source)
    publish_options = rtc.TrackPublishOptions(source=input_track_source)
    if input_audio_encoding is not None:
        publish_options.audio_encoding.CopyFrom(input_audio_encoding)
    publish_options.dtx = False
    publish_options.red = False
    publish_options.preconnect_buffer = bool(input_preconnect_buffer)
    print(
        "livekit-session-capture: publishing input track",
        {
            "track_source": int(input_track_source),
            "queue_size_ms": int(input_queue_size_ms),
            "preconnect_buffer": bool(input_preconnect_buffer),
            "dtx": bool(publish_options.dtx),
            "red": bool(publish_options.red),
        },
        flush=True,
    )
    input_publication = await room.local_participant.publish_track(track, publish_options)
    if wait_for_input_subscription:
        print(
            "livekit-session-capture: waiting for local input track subscription",
            {
                "track_sid": str(getattr(input_publication, "sid", "") or ""),
                "timeout_s": round(input_subscription_timeout_seconds, 2),
            },
            flush=True,
        )
        try:
            await asyncio.wait_for(
                input_publication.wait_for_subscription(),
                timeout=max(0.1, float(input_subscription_timeout_seconds)),
            )
            print("livekit-session-capture: local input track subscribed", flush=True)
        except asyncio.TimeoutError:
            print("livekit-session-capture: local input track subscription wait timed out", flush=True)
    if wait_for_runtime_ingress_lock:
        print(
            "livekit-session-capture: waiting for runtime ingress lock",
            {
                "identity": identity,
                "track_name": input_track_name,
                "timeout_s": round(runtime_ingress_lock_timeout_seconds, 2),
            },
            flush=True,
        )
        ingress_snapshot = await _wait_for_runtime_ingress_lock(
            base_url,
            timeout_seconds=max(0.1, float(runtime_ingress_lock_timeout_seconds)),
            participant_identity=identity,
            track_name=input_track_name,
        )
        print(
            "livekit-session-capture: runtime ingress lock ready",
            dict(ingress_snapshot.get("bridge", {})),
            flush=True,
        )
    frame_duration_s = float(frame_ms) / 1000.0
    frame_samples_per_channel = max(1, int(len(frames[0]) / (2 * channels)))
    next_capture_deadline = time.monotonic()
    preroll_rng = np.random.default_rng(0)
    silence_payload = b"\x00" * len(frames[0])
    direct_capture = int(input_queue_size_ms) <= 0

    preroll_silence_frames = max(0, int(round(publish_preroll_ms / max(1.0, float(frame_ms)))))
    if preroll_silence_frames:
        if publish_preroll_mode == "noise":
            preroll_payload = _noise_payload(
                byte_length=len(frames[0]),
                amplitude_i16=publish_preroll_noise_i16,
                rng=preroll_rng,
            )
        else:
            preroll_payload = b"\x00" * len(frames[0])
        print(
            "livekit-session-capture: publishing preroll silence",
            {
                "frames": preroll_silence_frames,
                "ms": round(publish_preroll_ms, 2),
                "mode": publish_preroll_mode,
                "noise_i16": int(publish_preroll_noise_i16) if publish_preroll_mode == "noise" else 0,
            },
            flush=True,
        )
        for _ in range(preroll_silence_frames):
            if direct_capture:
                next_capture_deadline = await _capture_frame_realtime_direct10ms(
                    source=source,
                    payload=preroll_payload,
                    sample_rate=sample_rate,
                    channels=channels,
                    next_deadline_monotonic=next_capture_deadline,
                    frame_duration_s=frame_duration_s,
                )
            else:
                next_capture_deadline = await _capture_frame_realtime(
                    source=source,
                    payload=preroll_payload,
                    sample_rate=sample_rate,
                    channels=channels,
                    samples_per_channel=frame_samples_per_channel,
                    next_deadline_monotonic=next_capture_deadline,
                    frame_duration_s=frame_duration_s,
                )

    print("livekit-session-capture: publishing input audio", {"frames": len(frames)}, flush=True)
    first_publish_started_monotonic = time.monotonic()
    input_speech_end_monotonic = (
        float(first_publish_started_monotonic)
        + (float(frame_duration_s) * float(last_voiced_input_frame_index + 1))
    )
    next_capture_deadline = float(first_publish_started_monotonic)
    for frame_bytes in frames:
        if direct_capture:
            next_capture_deadline = await _capture_frame_realtime_direct10ms(
                source=source,
                payload=frame_bytes,
                sample_rate=sample_rate,
                channels=channels,
                next_deadline_monotonic=next_capture_deadline,
                frame_duration_s=frame_duration_s,
            )
        else:
            next_capture_deadline = await _capture_frame_realtime(
                source=source,
                payload=frame_bytes,
                sample_rate=sample_rate,
                channels=channels,
                samples_per_channel=frame_samples_per_channel,
                next_deadline_monotonic=next_capture_deadline,
                frame_duration_s=frame_duration_s,
            )
    trailing_silence_frames = max(0, int(round(publish_trailing_silence_ms / max(1.0, float(frame_ms)))))
    if trailing_silence_frames:
        for _ in range(trailing_silence_frames):
            if direct_capture:
                next_capture_deadline = await _capture_frame_realtime_direct10ms(
                    source=source,
                    payload=silence_payload,
                    sample_rate=sample_rate,
                    channels=channels,
                    next_deadline_monotonic=next_capture_deadline,
                    frame_duration_s=frame_duration_s,
                )
            else:
                next_capture_deadline = await _capture_frame_realtime(
                    source=source,
                    payload=silence_payload,
                    sample_rate=sample_rate,
                    channels=channels,
                    samples_per_channel=frame_samples_per_channel,
                    next_deadline_monotonic=next_capture_deadline,
                    frame_duration_s=frame_duration_s,
                )
    wait_for_playout_fn = getattr(source, "wait_for_playout", None)
    if callable(wait_for_playout_fn):
        queued_duration_seconds = 0.0
        queued_duration_attr = getattr(source, "queued_duration", None)
        if queued_duration_attr is not None:
            try:
                queued_duration_seconds = float(queued_duration_attr)
            except Exception:
                queued_duration_seconds = 0.0
        print(
            "livekit-session-capture: waiting for input playout",
            {
                "queued_duration_s": round(max(0.0, queued_duration_seconds), 6),
                "timeout_s": round(max(0.1, input_playout_timeout_seconds), 2),
            },
            flush=True,
        )
        try:
            await asyncio.wait_for(
                wait_for_playout_fn(),
                timeout=max(0.1, float(input_playout_timeout_seconds)),
            )
            print("livekit-session-capture: input playout complete", flush=True)
        except asyncio.TimeoutError:
            print("livekit-session-capture: input playout wait timed out", flush=True)
    if close_source_after_publish:
        close_fn = getattr(source, "aclose", None)
        if callable(close_fn):
            await close_fn()
    publication_sid = str(getattr(input_publication, "sid", "") or "").strip()
    if publication_sid:
        try:
            await room.local_participant.unpublish_track(publication_sid)
            print(
                "livekit-session-capture: unpublished input track",
                {"track_sid": publication_sid},
                flush=True,
            )
        except Exception as exc:
            print(
                "livekit-session-capture: input track unpublish failed",
                {"track_sid": publication_sid, "error": str(exc)},
                flush=True,
            )

    print("livekit-session-capture: waiting for output audio", flush=True)
    try:
        await asyncio.wait_for(first_output_event.wait(), timeout=session_timeout_seconds)
    except TimeoutError as exc:
        raise RuntimeError(
            "timed out waiting for first output audio frame "
            f"(matched_track_seen={matched_track_seen}, subscribed_tracks={subscribed_tracks})"
        ) from exc

    capture_started_monotonic = time.monotonic()
    while True:
        if capture_done_event.is_set():
            break
        now_monotonic = time.monotonic()
        elapsed_capture_seconds = now_monotonic - capture_started_monotonic
        if elapsed_capture_seconds >= max_capture_seconds:
            break
        if last_audible_monotonic > 0.0 and output_frames > 0:
            if now_monotonic - last_audible_monotonic >= idle_after_first_audible_seconds:
                break
        should_poll_runtime = (
            first_output_monotonic > 0.0
            and output_frames > 0
            and (now_monotonic - last_runtime_poll_monotonic) >= runtime_settled_poll_seconds
        )
        if should_poll_runtime:
            last_runtime_poll_monotonic = now_monotonic
            try:
                live_runtime_snapshot = _http_json(f"{base_url.rstrip('/')}/v1/system/runtime?limit=20")
            except Exception:
                live_runtime_snapshot = {}
            if live_runtime_snapshot and _runtime_turn_settled(live_runtime_snapshot):
                if runtime_settled_since_monotonic <= 0.0:
                    runtime_settled_since_monotonic = now_monotonic
                if last_audible_monotonic > 0.0 and (
                    now_monotonic - last_audible_monotonic >= min(0.35, idle_after_first_audible_seconds)
                ):
                    break
                if now_monotonic - runtime_settled_since_monotonic >= runtime_settled_grace_seconds:
                    break
            else:
                runtime_settled_since_monotonic = 0.0
        await asyncio.sleep(0.1)

    if post_capture_settle_ms > 0.0:
        await asyncio.sleep(max(0.0, float(post_capture_settle_ms) / 1000.0))

    _write_pcm16_wav(
        path=capture_wav,
        pcm_bytes=bytes(output_pcm),
        sample_rate=output_sample_rate,
        channels=output_channels,
    )
    metrics = _audio_metrics(pcm_bytes=bytes(output_pcm), sample_rate=output_sample_rate)
    runtime_snapshot = _http_json(f"{base_url.rstrip('/')}/v1/system/runtime?limit=500")
    runtime_snapshot_path = _runtime_snapshot_path_for_capture(capture_wav)
    runtime_snapshot_path.write_text(json.dumps(runtime_snapshot, indent=2, sort_keys=True), encoding="utf-8")
    runtime_timestamps = dict(runtime_snapshot.get("timestamps", {}))
    runtime_metrics = dict(runtime_snapshot.get("metrics", {}))
    runtime_dispatch_to_first_token_ms = float(runtime_metrics.get("dispatch_to_first_token_ms", 0.0) or 0.0)
    runtime_turn_to_first_pcm_ms = 0.0
    runtime_first_token_to_first_pcm_ms = 0.0
    kernel_decision_ns = int(runtime_timestamps.get("kernel_decision_ns", 0) or 0)
    vllm_first_token_ns = int(runtime_timestamps.get("vllm_first_token_ns", 0) or 0)
    tts_first_pcm_ns = int(runtime_timestamps.get("tts_first_pcm_ns", 0) or 0)
    if kernel_decision_ns > 0 and tts_first_pcm_ns >= kernel_decision_ns:
        runtime_turn_to_first_pcm_ms = float(tts_first_pcm_ns - kernel_decision_ns) / 1_000_000.0
    if vllm_first_token_ns > 0 and tts_first_pcm_ns >= vllm_first_token_ns:
        runtime_first_token_to_first_pcm_ms = float(tts_first_pcm_ns - vllm_first_token_ns) / 1_000_000.0
    first_output_latency_ms = (
        (first_payload_monotonic - first_publish_started_monotonic) * 1000.0
        if first_publish_started_monotonic > 0.0 and first_payload_monotonic > 0.0
        else 0.0
    )
    user_stop_to_first_output_latency_ms = (
        (first_payload_monotonic - input_speech_end_monotonic) * 1000.0
        if input_speech_end_monotonic > 0.0 and first_payload_monotonic > 0.0
        else 0.0
    )
    first_audible_latency_ms = (
        (first_output_monotonic - first_publish_started_monotonic) * 1000.0
        if first_publish_started_monotonic > 0.0 and first_output_monotonic > 0.0
        else 0.0
    )
    user_stop_to_first_audible_latency_ms = (
        (first_output_monotonic - input_speech_end_monotonic) * 1000.0
        if input_speech_end_monotonic > 0.0 and first_output_monotonic > 0.0
        else 0.0
    )

    await room.disconnect()
    close_fn = getattr(source, "aclose", None)
    if callable(close_fn):
        await close_fn()

    print(
        "livekit-session-capture: READY",
        {
            "capture_wav": str(capture_wav),
            "runtime_snapshot": str(runtime_snapshot_path),
            "output_frames": output_frames,
            "output_sample_rate": output_sample_rate,
            "output_channels": output_channels,
            "matched_track_seen": matched_track_seen,
            "max_received_frame_rms": round(max_received_frame_rms, 6),
            "max_received_frame_peak": round(max_received_frame_peak, 6),
            "first_nonzero_frame_stats": first_nonzero_frame_stats,
            "subscribed_tracks": subscribed_tracks,
            "input_wav": str(publish_wav),
            "input_duration_s": round(input_duration_s, 6),
            "input_last_voiced_frame_index": int(last_voiced_input_frame_index),
            "input_speech_end_offset_ms": round(
                float(last_voiced_input_frame_index + 1) * float(frame_duration_s) * 1000.0,
                2,
            ),
            "first_output_latency_ms": round(first_output_latency_ms, 2),
            "user_stop_to_first_output_latency_ms": round(user_stop_to_first_output_latency_ms, 2),
            "first_audible_latency_ms": round(first_audible_latency_ms, 2),
            "user_stop_to_first_audible_latency_ms": round(user_stop_to_first_audible_latency_ms, 2),
            "runtime_dispatch_to_first_token_ms": round(runtime_dispatch_to_first_token_ms, 2),
            "runtime_turn_to_first_pcm_ms": round(runtime_turn_to_first_pcm_ms, 2),
            "runtime_first_token_to_first_pcm_ms": round(runtime_first_token_to_first_pcm_ms, 2),
            **{key: round(value, 6) for key, value in metrics.items()},
        },
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-audio", dest="input_audio")
    parser.add_argument("--output-wav", dest="output_wav")
    args = parser.parse_args()
    if args.input_audio:
        os.environ["VOICE_PIPELINE_REAL_SMOKE_WAV"] = args.input_audio
    if args.output_wav:
        os.environ["VOICE_PIPELINE_CAPTURE_WAV"] = args.output_wav
    try:
        return asyncio.run(_run())
    except Exception as exc:
        print(f"livekit-session-capture: FAILED ({exc})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
