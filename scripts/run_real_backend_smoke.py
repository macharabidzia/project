from __future__ import annotations

import asyncio
import os
import runpy
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "apps" / "api" / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from voice_pipeline.runtime.livekit_bridge import LiveKitRuntimeBridge
from voice_pipeline.runtime.bootstrap import bootstrap_runtime
from voice_pipeline.runtime.config import RuntimeConfig
from voice_pipeline.shared.audio_resample import StreamingAudioResampler
from voice_pipeline.shared.time import now_ns


def _exit_code_from(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return int(code)
    return 1


def _run_script(path: Path) -> int:
    previous_argv = list(sys.argv)
    previous_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        sys.argv = [str(path)]
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as exc:
        return _exit_code_from(exc)
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)


def _require_real_smoke_inputs(config: RuntimeConfig) -> Path:
    wav_path = Path(str(os.getenv("VOICE_PIPELINE_REAL_SMOKE_WAV", "")).strip())
    if not str(wav_path):
        raise RuntimeError("VOICE_PIPELINE_REAL_SMOKE_WAV is required for full-chain real smoke")
    if not wav_path.exists() or not wav_path.is_file():
        raise RuntimeError(f"VOICE_PIPELINE_REAL_SMOKE_WAV does not exist: {wav_path}")
    if not str(config.livekit_api_key).strip():
        raise RuntimeError("LIVEKIT_API_KEY is required for full-chain real smoke")
    if not str(config.livekit_api_secret).strip():
        raise RuntimeError("LIVEKIT_API_SECRET is required for full-chain real smoke")
    if not str(config.livekit_url).strip():
        raise RuntimeError("LIVEKIT_URL is required for full-chain real smoke")
    return wav_path


def _wav_to_runtime_pcm_frames(*, wav_path: Path, runtime_sample_rate: int, frame_ms: int) -> tuple[bytes, ...]:
    with wave.open(str(wav_path), "rb") as handle:
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        source_rate = int(handle.getframerate())
        raw = handle.readframes(int(handle.getnframes()))
    if channels != 1:
        raise RuntimeError(f"real smoke wav must be mono, got channels={channels}")
    if sample_width != 2:
        raise RuntimeError(f"real smoke wav must be PCM16, got sample_width={sample_width}")
    source = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    resampler = StreamingAudioResampler(target_rate=int(runtime_sample_rate))
    if source_rate != int(runtime_sample_rate):
        source = resampler.resample(source, int(source_rate))
    pcm = np.clip(source * 32767.0, -32768.0, 32767.0).astype("<i2").tobytes()
    frame_bytes = max(2, int(int(runtime_sample_rate) * int(frame_ms) / 1000) * 2)
    frames: list[bytes] = []
    for offset in range(0, len(pcm), frame_bytes):
        chunk = pcm[offset : offset + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + (b"\x00" * (frame_bytes - len(chunk)))
        frames.append(chunk)
    if not frames:
        raise RuntimeError("real smoke wav produced zero frames")
    return tuple(frames)


async def _run_runtime_full_chain_probe() -> int:
    config = RuntimeConfig.from_env()
    wav_path = _require_real_smoke_inputs(config)
    runtime = bootstrap_runtime(session_id="real-backend-smoke", config=RuntimeConfig.from_env())
    bridge = LiveKitRuntimeBridge(runtime=runtime, transport=runtime.transport)
    runtime.worker_status.transport = "WARMING"
    await bridge.start()
    await runtime.start()
    try:
        if not runtime.global_ready():
            raise RuntimeError("runtime not globally READY after LiveKit bridge start")

        ingress_frames = _wav_to_runtime_pcm_frames(
            wav_path=wav_path,
            runtime_sample_rate=int(config.input_sample_rate),
            frame_ms=int(config.frame_ms),
        )
        for frame in ingress_frames:
            await runtime.process_pcm_frame(frame)

        asr_final = runtime.asr.finalize(lineage_id=runtime.kernel.current_lease().epoch_id)
        if asr_final is not None:
            for authority_event in runtime._asr_events_to_authority((asr_final,), ingress_received_ns=now_ns()):
                runtime._append_event(authority_event)
            await runtime.run_tick_and_dispatch()

        event_types = [record.get("type", "") for record in runtime.event_log.as_records() if isinstance(record, dict)]
        required = (
            "ASRFinalReceived",
            "VLLMChunkReceived",
            "VLLMCompleted",
            "TTSChunkReceived",
            "TTSCompleted",
        )
        missing = [name for name in required if name not in event_types]
        if missing:
            raise RuntimeError(f"full chain missing required authority events: {missing}")

        transport_metrics = runtime.transport.ingress_metrics()
        if int(transport_metrics.get("transport_egress_frames", 0)) <= 0:
            raise RuntimeError("LiveKit egress emitted zero frames")

        print(
            "real-backend-smoke: READY",
            {
                "events": {name: event_types.count(name) for name in required},
                "transport_egress_frames": int(transport_metrics.get("transport_egress_frames", 0)),
                "pcm_stale_drops": int(runtime.pcm_clock.dropped_stale_frames),
            },
        )
        return 0
    finally:
        await bridge.stop()
        await runtime.stop()


def main() -> int:
    try:
        lane_smokes = (
            REPO_ROOT / "scripts" / "run_real_vosk_smoke.py",
            REPO_ROOT / "scripts" / "run_real_vllm_smoke.py",
            REPO_ROOT / "scripts" / "run_real_cosyvoice3_smoke.py",
        )
        for script_path in lane_smokes:
            code = _run_script(script_path)
            if code != 0:
                return int(code)
        return asyncio.run(_run_runtime_full_chain_probe())
    except Exception as exc:
        print(f"real-backend-smoke: FAILED ({exc})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
