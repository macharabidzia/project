from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from voice_pipeline.runtime.livekit_bridge import LiveKitRuntimeBridge, create_livekit_access_token
from voice_pipeline.runtime.bootstrap import VoicePipelineRuntime, bootstrap_runtime
from voice_pipeline.runtime.config import RuntimeConfig
from voice_pipeline.shared.time import now_ns


def _client_livekit_url(runtime: VoicePipelineRuntime) -> str:
    public_url = str(getattr(runtime.config, "livekit_public_url", "")).strip()
    if public_url:
        return public_url
    return str(runtime.transport.config.url)


def _allowed_cors_origins() -> list[str]:
    configured = str(os.getenv("VOICE_PIPELINE_ALLOWED_ORIGINS", "")).strip()
    if configured:
        origins = [item.strip() for item in configured.split(",") if item.strip()]
        if origins:
            return origins
    return [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]


def _runtime_readiness(runtime: VoicePipelineRuntime) -> dict[str, object]:
    cache_stats = runtime.vllm.cache_stats()
    checks = {
        "asr_cpu": {
            "ready": bool(runtime.warm_report.asr_warm),
            "target": "cpu",
            "detail": runtime.worker_status.asr,
            "status": "ready" if runtime.warm_report.asr_warm else "failed",
        },
        "vllm_gpu0": {
            "ready": bool(runtime.warm_report.vllm_warm),
            "target": "cuda:0",
            "detail": runtime.worker_status.vllm,
            "status": "ready" if runtime.warm_report.vllm_warm else "failed",
        },
        "tts_gpu1": {
            "ready": bool(runtime.warm_report.tts_warm),
            "target": "cuda:1",
            "detail": runtime.worker_status.tts,
            "status": "ready" if runtime.warm_report.tts_warm else "failed",
        },
        "kernel_authority": {
            "ready": str(getattr(runtime.worker_status, "kernel", "READY")) == "READY",
            "target": "KernelRuntime",
            "detail": str(getattr(runtime.worker_status, "kernel", "READY")),
            "status": "ready" if str(getattr(runtime.worker_status, "kernel", "READY")) == "READY" else "failed",
        },
        "transport": {
            "ready": str(runtime.worker_status.transport) == "READY",
            "target": "livekit",
            "detail": str(runtime.worker_status.transport),
            "status": "ready" if str(runtime.worker_status.transport) == "READY" else "failed",
        },
    }
    ready_count = sum(1 for check in checks.values() if bool(check["ready"]))
    total_checks = max(1, len(checks))
    progress = int((float(ready_count) / float(total_checks)) * 100.0)
    ready = bool(runtime.global_ready() and ready_count == total_checks)
    cache_identity = dict(runtime.model_cache_identity)
    return {
        "ready": ready,
        "session_eligible": ready,
        "status": "ready" if ready else "failed",
        "progress": 100 if ready else progress,
        "summary": runtime.dry_run_report(),
        "checks": checks,
        "asr_status": runtime.worker_status.asr,
        "asr_device": runtime.config.asr_device,
        "asr_model_path": runtime.config.asr_model_path,
        "asr_sample_rate": int(runtime.config.asr_sample_rate),
        "asr_failure_reason": str(runtime.worker_failure_reason.asr or ""),
        "llm_status": runtime.worker_status.vllm,
        "llm_device": runtime.config.llm_device,
        "llm_model_path": runtime.config.resolved_vllm_model_path(),
        "llm_cache_dir": runtime.config.vllm_cache_dir,
        "llm_prefix_cache_ready": bool(runtime.vllm.prefix_cache_ready),
        "llm_prefix_cache_hits": int(cache_stats.hits),
        "llm_prefix_cache_misses": int(cache_stats.misses),
        "llm_prefix_cache_hit_ratio": float(cache_stats.hit_ratio),
        "llm_failure_reason": str(runtime.worker_failure_reason.vllm or ""),
        "tts_status": runtime.worker_status.tts,
        "tts_device": runtime.config.tts_device,
        "tts_model_path": runtime.config.resolved_cosyvoice3_model_path(),
        "tts_cache_dir": runtime.config.cosyvoice3_cache_dir,
        "tts_native_bistream_ready": bool(runtime.warm_report.tts_warm),
        "tts_failure_reason": str(runtime.worker_failure_reason.tts or ""),
        "kernel_status": str(getattr(runtime.worker_status, "kernel", "READY")),
        "transport_status": runtime.worker_status.transport,
        "transport_failure_reason": str(runtime.worker_failure_reason.transport or ""),
        "topology_hash": runtime.startup_contract_hash,
        "model_cache_hash": str(cache_identity.get("model_cache_hash", "")),
        "drift_snapshot_hash": runtime.replay_event_hash(),
        "model_cache_identity": cache_identity,
    }


def _runtime_telemetry(runtime: VoicePipelineRuntime) -> dict[str, object]:
    latency = {name: summary.__dict__ for name, summary in runtime.latency_summary().items()}
    ingress = runtime.transport.ingress_metrics()
    kernel_metrics = runtime.kernel.runtime_metrics()
    timestamps = runtime.last_timestamps()
    cache_stats = runtime.vllm.cache_stats()
    return {
        "available": True,
        "summary": runtime.dry_run_report(),
        "session_id": runtime.kernel.session_id,
        "updated_at": int(now_ns()),
        "health": {
            "asr": bool(runtime.warm_report.asr_warm),
            "vllm": bool(runtime.warm_report.vllm_warm),
            "tts": bool(runtime.warm_report.tts_warm),
        },
        "metrics": {**kernel_metrics, **ingress},
        "stats": {
            "pcm_queue_depth": int(runtime.pcm_clock.depth),
            "kernel_queue_depth": int(runtime.kernel.queued_event_count),
        },
        "scheduler": {
            "queued": int(runtime.kernel.queued_event_count),
            "running": 1 if runtime.worker_status.transport == "READY" else 0,
            "completed": len(runtime.event_log.events),
            "cancelled": 0,
            "devices": {
                "cpu": {"active": True, "current_work_unit_id": runtime.worker_status.asr, "queued": 0},
                "cuda:0": {"active": True, "current_work_unit_id": runtime.worker_status.vllm, "queued": 0},
                "cuda:1": {"active": True, "current_work_unit_id": runtime.worker_status.tts, "queued": 0},
            },
            "recent": [],
        },
        "playout": {
            "target_buffer_ms": int(runtime.config.frame_ms * 2),
            "buffered_ms": int(runtime.pcm_clock.depth * runtime.config.frame_ms),
            "max_buffered_ms": int(runtime.pcm_clock.max_buffer_frames * runtime.config.frame_ms),
            "primed": runtime.pcm_clock.depth > 0,
            "underruns": 0,
            "stale_chunk_drops": int(runtime.kernel.runtime_metrics()["stale_pcm_drop_count"]),
        },
        "backpressure": {
            "text_queue_depth": int(runtime.kernel.queued_event_count),
            "text_queue_max_depth": int(runtime.config.ingress_max_items),
            "text_oldest_age_ms": float(kernel_metrics.get("ingress_queue_oldest_age_ms", 0.0)),
            "text_drops": int(kernel_metrics.get("ingress_drop_count", 0)),
            "backpressure_action": str(kernel_metrics.get("backpressure_action", "none")),
            "audio_queue_depth": int(runtime.pcm_clock.depth),
            "audio_oldest_age_ms": float(runtime.pcm_clock.oldest_age_ms),
            "audio_queue_max_depth": int(runtime.pcm_clock.max_buffer_frames),
            "audio_overflows": int(runtime.pcm_clock.dropped_overflow_frames),
            "audio_stale_drops": int(runtime.pcm_clock.dropped_stale_frames),
        },
        "latency": latency,
        "timestamps": {
            "ingress_received_ns": int(timestamps.get("ingress_received_ns", 0)),
            "asr_event_ns": int(timestamps.get("asr_event_ns", 0)),
            "kernel_decision_ns": int(timestamps.get("kernel_decision_ns", 0)),
            "vllm_first_token_ns": int(timestamps.get("vllm_first_token_ns", 0)),
            "tts_first_pcm_ns": int(timestamps.get("tts_first_pcm_ns", 0)),
            "transport_emit_ns": int(timestamps.get("transport_emit_ns", 0)),
        },
        "model_cache_identity": dict(runtime.model_cache_identity),
        "llm_prefix_cache_stats": {
            "hits": int(cache_stats.hits),
            "misses": int(cache_stats.misses),
            "hit_ratio": float(cache_stats.hit_ratio),
        },
        "topology_hash": runtime.startup_contract_hash,
    }


def _system_config(runtime: VoicePipelineRuntime) -> dict[str, object]:
    return {
        "app_name": "Voice OS",
        "app_env": "runtime",
        "transport": {
            "kind": "livekit",
            "livekit_url": _client_livekit_url(runtime),
            "room_name": runtime.transport.config.room_name,
            "runtime_identity": runtime.transport.config.runtime_identity,
            "output_track_name": runtime.transport.config.output_track_name,
            "turn_enabled": bool(runtime.transport.config.turn_enabled),
        },
        "layers": [
            {"name": "ASR", "purpose": "CPU Vosk streaming decode", "backend": "vosk", "status": "active"},
            {"name": "Kernel", "purpose": "single authority reducer", "backend": "KernelRuntime", "status": "active"},
            {"name": "LLM", "purpose": "GPU0 token streaming", "backend": "vllm", "status": "active"},
            {"name": "TTS", "purpose": "GPU1 native bi-stream", "backend": "CosyVoice3", "status": "active"},
            {"name": "Transport", "purpose": "LiveKit WebRTC media ingress/egress", "backend": "livekit", "status": "active"},
        ],
    }


def create_app() -> FastAPI:
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        session_id = os.getenv("VOICE_PIPELINE_SESSION_ID", "voice-runtime")
        runtime = bootstrap_runtime(session_id=session_id, config=RuntimeConfig.from_env())
        bridge = LiveKitRuntimeBridge(runtime=runtime, transport=runtime.transport)
        runtime.worker_status.transport = "WARMING"
        try:
            await bridge.start()
            await runtime.start()
        except Exception:
            await bridge.stop()
            await runtime.stop()
            raise
        app.state.runtime = runtime
        app.state.livekit_bridge = bridge
        try:
            yield
        finally:
            active_bridge: LiveKitRuntimeBridge | None = getattr(app.state, "livekit_bridge", None)
            if active_bridge is not None:
                await active_bridge.stop()
            active_runtime: VoicePipelineRuntime | None = getattr(app.state, "runtime", None)
            if active_runtime is not None:
                await active_runtime.stop()

    app = FastAPI(title="Voice OS Runtime", version="1.0", lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_cors_origins(),
        allow_origin_regex=r"https://[a-z0-9-]+-\d+\.proxy\.runpod\.net",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, object]:
        runtime: VoicePipelineRuntime = app.state.runtime
        return {"ok": True, "summary": runtime.dry_run_report()}

    @app.get("/v1/system/config")
    async def system_config() -> dict[str, object]:
        runtime: VoicePipelineRuntime = app.state.runtime
        return _system_config(runtime)

    @app.get("/config")
    async def config_alias() -> dict[str, object]:
        runtime: VoicePipelineRuntime = app.state.runtime
        return _system_config(runtime)

    @app.get("/v1/system/readiness")
    async def system_readiness() -> dict[str, object]:
        runtime: VoicePipelineRuntime = app.state.runtime
        return _runtime_readiness(runtime)

    @app.get("/ready")
    async def ready_alias() -> dict[str, object]:
        runtime: VoicePipelineRuntime = app.state.runtime
        return _runtime_readiness(runtime)

    @app.get("/v1/system/runtime")
    async def system_runtime() -> dict[str, object]:
        runtime: VoicePipelineRuntime = app.state.runtime
        return _runtime_telemetry(runtime)

    @app.get("/telemetry")
    async def telemetry_alias() -> dict[str, object]:
        runtime: VoicePipelineRuntime = app.state.runtime
        return _runtime_telemetry(runtime)

    @app.get("/v1/livekit/token")
    async def livekit_token(identity: str = "voice-web-client") -> dict[str, object]:
        runtime: VoicePipelineRuntime = app.state.runtime
        token = create_livekit_access_token(
            api_key=runtime.transport.config.api_key,
            api_secret=runtime.transport.config.api_secret,
            identity=str(identity),
            room_name=runtime.transport.config.room_name,
            can_publish=True,
            can_subscribe=True,
            ttl_seconds=int(runtime.transport.config.token_ttl_seconds),
            name=str(identity),
        )
        return {
            "url": _client_livekit_url(runtime),
            "room_name": runtime.transport.config.room_name,
            "identity": str(identity),
            "token": token,
            "turn_enabled": bool(runtime.transport.config.turn_enabled),
        }

    @app.get("/livekit/token")
    async def livekit_token_alias(identity: str = "voice-web-client") -> dict[str, object]:
        return await livekit_token(identity=identity)

    return app


__all__ = ["create_app"]
