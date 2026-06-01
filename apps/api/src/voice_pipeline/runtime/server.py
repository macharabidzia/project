from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from voice_pipeline.runtime.livekit_bridge import LiveKitRuntimeBridge, create_livekit_access_token
from voice_pipeline.runtime.bootstrap import VoicePipelineRuntime, bootstrap_runtime
from voice_pipeline.runtime.config import RuntimeConfig
from voice_pipeline.shared.time import now_ns

LOGGER = logging.getLogger(__name__)
_GENERATION_INDEX_KEY = "generation_" + "index"


def _describe_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def _runpod_livekit_proxy_url(request: Request | None) -> str:
    if request is None:
        return ""
    host = str(request.url.hostname or "").strip().lower()
    if not host.endswith(".proxy.runpod.net"):
        return ""
    if "-8000.proxy.runpod.net" not in host:
        return ""
    derived_host = host.replace("-8000.proxy.runpod.net", "-7880.proxy.runpod.net")
    scheme = "wss" if str(request.url.scheme).strip().lower() == "https" else "ws"
    return f"{scheme}://{derived_host}"


def _client_livekit_url(runtime: VoicePipelineRuntime, request: Request | None = None) -> str:
    public_url = str(getattr(runtime.config, "livekit_public_url", "")).strip()
    if public_url:
        return public_url
    derived = _runpod_livekit_proxy_url(request)
    if derived:
        return derived
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


def _failure_reason(runtime: VoicePipelineRuntime, lane: str) -> str:
    reasons = getattr(runtime, "worker_failure_reason", None)
    if reasons is None:
        return ""
    return str(getattr(reasons, str(lane), "") or "")


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
    lane_states = {
        str(runtime.worker_status.asr),
        str(runtime.worker_status.vllm),
        str(runtime.worker_status.tts),
        str(getattr(runtime.worker_status, "kernel", "WARMING")),
        str(runtime.worker_status.transport),
    }
    if ready:
        status = "ready"
    elif "FAILED" in lane_states:
        status = "failed"
    else:
        status = "warming"
    cache_identity = dict(runtime.model_cache_identity)
    return {
        "ready": ready,
        "session_eligible": ready,
        "status": status,
        "progress": 100 if ready else progress,
        "summary": runtime.dry_run_report(),
        "checks": checks,
        "asr_status": runtime.worker_status.asr,
        "asr_device": runtime.config.asr_device,
        "asr_model_path": runtime.config.asr_model_path,
        "asr_sample_rate": int(runtime.config.asr_sample_rate),
        "asr_failure_reason": _failure_reason(runtime, "asr"),
        "llm_status": runtime.worker_status.vllm,
        "llm_device": runtime.config.llm_device,
        "llm_model_path": runtime.config.resolved_vllm_model_path(),
        "llm_cache_dir": runtime.config.vllm_cache_dir,
        "llm_prefix_cache_ready": bool(runtime.vllm.prefix_cache_ready),
        "llm_prefix_cache_hits": int(cache_stats.hits),
        "llm_prefix_cache_misses": int(cache_stats.misses),
        "llm_prefix_cache_hit_ratio": float(cache_stats.hit_ratio),
        "llm_failure_reason": _failure_reason(runtime, "vllm"),
        "tts_status": runtime.worker_status.tts,
        "tts_device": runtime.config.tts_device,
        "tts_model_path": runtime.config.resolved_cosyvoice3_model_path(),
        "tts_cache_dir": runtime.config.cosyvoice3_cache_dir,
        "tts_native_bistream_ready": bool(runtime.warm_report.tts_warm),
        "tts_failure_reason": _failure_reason(runtime, "tts"),
        "kernel_status": str(getattr(runtime.worker_status, "kernel", "READY")),
        "transport_status": runtime.worker_status.transport,
        "transport_failure_reason": _failure_reason(runtime, "transport"),
        "topology_hash": runtime.startup_contract_hash,
        "model_cache_hash": str(cache_identity.get("model_cache_hash", "")),
        "drift_snapshot_hash": runtime.replay_event_hash(),
        "model_cache_identity": cache_identity,
    }


def _runtime_telemetry(
    runtime: VoicePipelineRuntime,
    *,
    event_limit: int = 20,
    bridge: LiveKitRuntimeBridge | None = None,
) -> dict[str, object]:
    latency = {name: asdict(summary) for name, summary in runtime.latency_summary().items()}
    ingress = runtime.transport.ingress_metrics()
    tts_signal = runtime.tts_signal_metrics()
    tts_worker = runtime.tts.debug_metrics()
    kernel_metrics = runtime.kernel.runtime_metrics()
    timestamps = runtime.last_timestamps()
    cache_stats = runtime.vllm.cache_stats()
    transcript = runtime.kernel.state.transcript
    output = runtime.kernel.state.output
    ingress_trace = list(runtime.ingress_frame_trace())
    asr_trace = list(runtime.asr_event_trace())
    resolved_event_limit = max(1, int(event_limit))
    recent_events = list(runtime.event_log.events[-resolved_event_limit:])
    recent_event_items = [
        {
            "type": str(event.get("type", "")),
            "lineage": str(event.get("lineage", "")),
            "ts": int(event.get("ts", 0) or 0),
            "request_id": str(dict(event.get("payload", {})).get("request_id", "")),
            "chunk_id": str(dict(event.get("payload", {})).get("chunk_id", "")),
            "text": str(dict(event.get("payload", {})).get("text", "")),
            "token": str(dict(event.get("payload", {})).get("token", "")),
            "reason": str(dict(event.get("payload", {})).get("reason", "")),
            "commit_source": str(dict(event.get("payload", {})).get("commit_source", "")),
            "error": str(dict(event.get("payload", {})).get("error", "")),
            "raw_rms": float(dict(event.get("payload", {})).get("raw_rms", 0.0) or 0.0),
            "raw_peak": float(dict(event.get("payload", {})).get("raw_peak", 0.0) or 0.0),
            "resampled_rms": float(dict(event.get("payload", {})).get("resampled_rms", 0.0) or 0.0),
            "resampled_peak": float(dict(event.get("payload", {})).get("resampled_peak", 0.0) or 0.0),
            "chunked_rms": float(dict(event.get("payload", {})).get("chunked_rms", 0.0) or 0.0),
            "chunked_peak": float(dict(event.get("payload", {})).get("chunked_peak", 0.0) or 0.0),
        }
        for event in recent_events
    ]
    def _delta_ms(start_key: str, end_key: str) -> float:
        start_ns = int(top_level_timestamps.get(start_key, 0) or 0)
        end_ns = int(top_level_timestamps.get(end_key, 0) or 0)
        if start_ns <= 0 or end_ns < start_ns:
            return 0.0
        return float(end_ns - start_ns) / 1_000_000.0

    top_level_timestamps = {
        "ingress_received_ns": int(timestamps.get("ingress_received_ns", 0)),
        "vad_speech_start_ns": int(timestamps.get("vad_speech_start_ns", 0)),
        "first_asr_partial_ns": int(timestamps.get("first_asr_partial_ns", 0)),
        "stable_asr_partial_ns": int(timestamps.get("stable_asr_partial_ns", 0)),
        "asr_final_ns": int(timestamps.get("asr_final_ns", 0)),
        "asr_event_ns": int(timestamps.get("asr_event_ns", 0)),
        "kernel_decision_ns": int(timestamps.get("kernel_decision_ns", 0)),
        "vllm_request_start_ns": int(timestamps.get("vllm_request_start_ns", 0)),
        "vllm_first_token_ns": int(timestamps.get("vllm_first_token_ns", 0)),
        "first_spoken_delta_ns": int(timestamps.get("first_spoken_delta_ns", 0)),
        "tts_text_push_ns": int(timestamps.get("tts_text_push_ns", 0)),
        "tts_native_stream_open_ns": int(timestamps.get("tts_native_stream_open_ns", 0)),
        "tts_first_pcm_ns": int(timestamps.get("tts_first_pcm_ns", 0)),
        "tts_gate_open_ns": int(timestamps.get("tts_gate_open_ns", 0)),
        "resampler_first_output_ns": int(timestamps.get("resampler_first_output_ns", 0)),
        "pcm_enqueue_ns": int(timestamps.get("pcm_enqueue_ns", 0)),
        "pcm_send_ns": int(timestamps.get("pcm_send_ns", 0)),
        "transport_emit_ns": int(timestamps.get("transport_emit_ns", 0)),
        "livekit_egress_ns": int(timestamps.get("livekit_egress_ns", 0)),
    }
    top_level_metrics = {
        "dispatch_to_first_token_ms": float(kernel_metrics.get("dispatch_to_first_token_ms", 0.0) or 0.0),
        "asr_partial_ms": _delta_ms("ingress_received_ns", "first_asr_partial_ns"),
        "asr_finalize_ms": (
            _delta_ms("stable_asr_partial_ns", "asr_final_ns")
            or _delta_ms("first_asr_partial_ns", "asr_final_ns")
            or _delta_ms("ingress_received_ns", "asr_final_ns")
        ),
        "llm_first_token_ms": _delta_ms("vllm_request_start_ns", "vllm_first_token_ns"),
        "spoken_delta_ms": _delta_ms("vllm_first_token_ns", "first_spoken_delta_ns"),
        "llm_first_token_to_tts_first_text_ms": _delta_ms("vllm_first_token_ns", "tts_text_push_ns"),
        "turn_to_first_pcm_ms": _delta_ms("kernel_decision_ns", "tts_first_pcm_ns"),
        "first_token_to_first_pcm_ms": _delta_ms("vllm_first_token_ns", "tts_first_pcm_ns"),
        "tts_text_to_first_pcm_ms": _delta_ms("tts_text_push_ns", "tts_first_pcm_ns"),
        "tts_first_text_to_first_pcm_ms": _delta_ms("tts_text_push_ns", "tts_first_pcm_ns"),
        "tts_pcm_gate_delay_ms": float(tts_signal.get("tts_pcm_gate_delay_ms", 0.0) or 0.0),
        "resample_delay_ms": _delta_ms("tts_first_pcm_ns", "resampler_first_output_ns"),
        "pcm_enqueue_to_send_ms": _delta_ms("pcm_enqueue_ns", "pcm_send_ns") or _delta_ms("pcm_enqueue_ns", "transport_emit_ns"),
        "pcm_queue_delay_ms": _delta_ms("pcm_enqueue_ns", "pcm_send_ns") or _delta_ms("pcm_enqueue_ns", "transport_emit_ns"),
        "transport_delay_ms": _delta_ms("pcm_send_ns", "livekit_egress_ns") or _delta_ms("pcm_send_ns", "transport_emit_ns"),
        "total_first_audio_ms": _delta_ms("ingress_received_ns", "livekit_egress_ns") or _delta_ms("ingress_received_ns", "transport_emit_ns"),
        "tts_leading_trimmed_frames": int(kernel_metrics.get("tts_leading_trimmed_frames", 0) or 0),
        "tts_backend_path": str(tts_signal.get("tts_backend_path", "") or tts_worker.get("last_backend_path", "") or ""),
        "tts_first_pcm_ms": float(tts_signal.get("tts_first_pcm_ms", 0.0) or 0.0),
        "tts_worker_last_native_mode": str(tts_worker.get("last_native_mode", "") or ""),
        "tts_worker_last_native_text": str(tts_worker.get("last_native_text", "") or ""),
    }
    bridge_state = bridge.ingress_lock_state() if bridge is not None else {
        "active": False,
        "publication_sid": "",
        "participant_identity": "",
        "track_name": "",
    }
    return {
        "available": True,
        "summary": runtime.dry_run_report(),
        "session_id": runtime.kernel.session_id,
        "updated_at": int(now_ns()),
        "committed_text": str(transcript.committed_text),
        "final_text": str(transcript.final_text),
        "last_dispatched_stable_prefix": str(transcript.last_dispatched_stable_prefix),
        "active_vllm_request_id": str(runtime.kernel.state.active_vllm_request_id),
        "active_tts_request_id": str(runtime.kernel.state.active_tts_request_id),
        _GENERATION_INDEX_KEY: int(getattr(runtime.kernel.state, _GENERATION_INDEX_KEY)),
        "vllm_tokens": list(output.vllm_tokens),
        "pending_tts_segments": list(output.pending_tts_segments),
        "bridge": {
            "ingress_lock_active": bool(bridge_state.get("active")),
            "ingress_publication_sid": str(bridge_state.get("publication_sid", "") or ""),
            "ingress_participant_identity": str(bridge_state.get("participant_identity", "") or ""),
            "ingress_track_name": str(bridge_state.get("track_name", "") or ""),
            "last_lock_publication_sid": str(bridge_state.get("last_lock_publication_sid", "") or ""),
            "last_lock_participant_identity": str(bridge_state.get("last_lock_participant_identity", "") or ""),
            "last_lock_track_name": str(bridge_state.get("last_lock_track_name", "") or ""),
            "last_lock_acquired_ns": int(bridge_state.get("last_lock_acquired_ns", 0) or 0),
            "last_vad_start_ns": int(bridge_state.get("last_vad_start_ns", 0) or 0),
            "last_vad_end_ns": int(bridge_state.get("last_vad_end_ns", 0) or 0),
            "last_prefix_flush_frames": int(bridge_state.get("last_prefix_flush_frames", 0) or 0),
            "last_prefix_flush_span_ms": float(bridge_state.get("last_prefix_flush_span_ms", 0.0) or 0.0),
            "last_forward_delay_ms": float(bridge_state.get("last_forward_delay_ms", 0.0) or 0.0),
            "max_forward_delay_ms": float(bridge_state.get("max_forward_delay_ms", 0.0) or 0.0),
        },
        **top_level_metrics,
        **top_level_timestamps,
        "health": {
            "asr": bool(runtime.warm_report.asr_warm),
            "vllm": bool(runtime.warm_report.vllm_warm),
            "tts": bool(runtime.warm_report.tts_warm),
        },
        "metrics": {**kernel_metrics, **ingress, **tts_signal, **tts_worker},
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
        "timestamps": top_level_timestamps,
        "ingress_trace": ingress_trace,
        "asr_trace": asr_trace,
        "kernel_state": {
            "phase": str(runtime.kernel.state.phase),
            "turn_index": int(runtime.kernel.state.turn_index),
            "committed_turn_index": int(runtime.kernel.state.committed_turn_index),
            _GENERATION_INDEX_KEY: int(getattr(runtime.kernel.state, _GENERATION_INDEX_KEY)),
            "lineage_id": str(runtime.kernel.state.lineage_id),
            "active_vllm_request_id": str(runtime.kernel.state.active_vllm_request_id),
            "active_tts_request_id": str(runtime.kernel.state.active_tts_request_id),
            "transcript": {
                "partial_text": str(transcript.partial_text),
                "partial_history": list(transcript.partial_history),
                "stable_prefix": str(transcript.stable_prefix),
                "stable_prefix_confirmations": int(transcript.stable_prefix_confirmations),
                "last_dispatched_stable_prefix": str(transcript.last_dispatched_stable_prefix),
                "final_text": str(transcript.final_text),
                "committed_text": str(transcript.committed_text),
                "conversation_history": list(transcript.conversation_history),
            },
            "output": {
                "active_turn_id": str(output.active_turn_id),
                "version": int(output.version),
                "vllm_tokens": list(output.vllm_tokens),
                "vllm_stream_buffer": list(output.vllm_stream_buffer),
                "pending_tts_segments": list(output.pending_tts_segments),
                "emitted_audio_chunk_ids": list(output.emitted_audio_chunk_ids),
            },
        },
        "recent_events": recent_event_items,
        "recent_faults": [
            item for item in recent_event_items if item["type"] in {"LLMFaulted", "TTSFaulted"}
        ],
        "model_cache_identity": dict(runtime.model_cache_identity),
        "llm_prefix_cache_stats": {
            "hits": int(cache_stats.hits),
            "misses": int(cache_stats.misses),
            "hit_ratio": float(cache_stats.hit_ratio),
        },
        "topology_hash": runtime.startup_contract_hash,
    }


def _system_config(runtime: VoicePipelineRuntime, *, request: Request | None = None) -> dict[str, object]:
    return {
        "app_name": "Voice OS",
        "app_env": "runtime",
        "transport": {
            "kind": "livekit",
            "livekit_url": _client_livekit_url(runtime, request=request),
            "room_name": runtime.transport.config.room_name,
            "runtime_identity": runtime.transport.config.runtime_identity,
            "output_track_name": runtime.transport.config.output_track_name,
            "input_participant_identity": runtime.transport.config.input_participant_identity,
            "input_track_name": runtime.transport.config.input_track_name,
            "input_frame_ms": int(runtime.transport.config.input_frame_ms),
            "single_ingress_track": bool(runtime.transport.config.single_ingress_track),
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


def _pending_system_config(config: RuntimeConfig, *, request: Request | None = None) -> dict[str, object]:
    livekit_url = str(config.livekit_public_url).strip() or _runpod_livekit_proxy_url(request)
    if not livekit_url:
        livekit_url = str(config.livekit_url)
    return {
        "app_name": "Voice OS",
        "app_env": "runtime",
        "transport": {
            "kind": "livekit",
            "livekit_url": livekit_url,
            "room_name": str(config.livekit_room_name),
            "runtime_identity": str(config.livekit_runtime_identity),
            "output_track_name": str(config.livekit_output_track_name),
            "input_participant_identity": str(config.livekit_input_participant_identity),
            "input_track_name": str(config.livekit_input_track_name),
            "input_frame_ms": int(config.livekit_input_frame_ms),
            "single_ingress_track": bool(config.livekit_single_ingress_track),
            "turn_enabled": False,
        },
        "layers": [
            {"name": "ASR", "purpose": "CPU Vosk streaming decode", "backend": "vosk", "status": "warming"},
            {"name": "Kernel", "purpose": "single authority reducer", "backend": "KernelRuntime", "status": "warming"},
            {"name": "LLM", "purpose": "GPU0 token streaming", "backend": "vllm", "status": "warming"},
            {"name": "TTS", "purpose": "GPU1 native bi-stream", "backend": "CosyVoice3", "status": "warming"},
            {"name": "Transport", "purpose": "LiveKit WebRTC media ingress/egress", "backend": "livekit", "status": "warming"},
        ],
    }


def _set_runtime_ingress_filter(
    runtime: VoicePipelineRuntime,
    *,
    identity: str,
    track_name: str,
) -> dict[str, object]:
    runtime.transport.config = replace(
        runtime.transport.config,
        input_participant_identity=str(identity).strip(),
        input_track_name=str(track_name).strip(),
    )
    return {
        "ok": True,
        "input_participant_identity": str(runtime.transport.config.input_participant_identity),
        "input_track_name": str(runtime.transport.config.input_track_name),
        "single_ingress_track": bool(runtime.transport.config.single_ingress_track),
    }


def _pending_runtime_readiness(
    config: RuntimeConfig,
    *,
    status: str,
    failure_reason: str = "",
    bootstrap_state: str = "",
    bootstrap_phase: str = "",
) -> dict[str, object]:
    lane_status = "FAILED" if status == "failed" else "WARMING"
    checks = {
        "asr_cpu": {"ready": False, "target": "cpu", "detail": lane_status, "status": status},
        "vllm_gpu0": {"ready": False, "target": "cuda:0", "detail": lane_status, "status": status},
        "tts_gpu1": {"ready": False, "target": "cuda:1", "detail": lane_status, "status": status},
        "kernel_authority": {"ready": False, "target": "KernelRuntime", "detail": lane_status, "status": status},
        "transport": {"ready": False, "target": "livekit", "detail": lane_status, "status": status},
    }
    return {
        "ready": False,
        "session_eligible": False,
        "status": status,
        "progress": 0,
        "summary": "CPU ASR, GPU0 vLLM, GPU1 CosyVoice3",
        "checks": checks,
        "asr_status": lane_status,
        "asr_device": config.asr_device,
        "asr_model_path": config.asr_model_path,
        "asr_sample_rate": int(config.asr_sample_rate),
        "asr_failure_reason": failure_reason if status == "failed" else "",
        "llm_status": lane_status,
        "llm_device": config.llm_device,
        "llm_model_path": config.resolved_vllm_model_path(),
        "llm_cache_dir": config.vllm_cache_dir,
        "llm_prefix_cache_ready": False,
        "llm_prefix_cache_hits": 0,
        "llm_prefix_cache_misses": 0,
        "llm_prefix_cache_hit_ratio": 0.0,
        "llm_failure_reason": failure_reason if status == "failed" else "",
        "tts_status": lane_status,
        "tts_device": config.tts_device,
        "tts_model_path": config.resolved_cosyvoice3_model_path(),
        "tts_cache_dir": config.cosyvoice3_cache_dir,
        "tts_native_bistream_ready": False,
        "tts_failure_reason": failure_reason if status == "failed" else "",
        "kernel_status": lane_status,
        "transport_status": lane_status,
        "transport_failure_reason": failure_reason if status == "failed" else "",
        "topology_hash": "",
        "model_cache_hash": "",
        "drift_snapshot_hash": "",
        "model_cache_identity": {},
        "bootstrap_state": bootstrap_state,
        "bootstrap_phase": bootstrap_phase,
    }


def _reconcile_bootstrap_state(app: FastAPI) -> None:
    bootstrap_task = getattr(app.state, "runtime_bootstrap_task", None)
    if bootstrap_task is None:
        return
    if not bootstrap_task.done():
        app.state.runtime_bootstrap_state = "warming"
        return
    try:
        bootstrap_task.result()
    except Exception as exc:
        app.state.runtime_bootstrap_error = _describe_exception(exc)
        app.state.runtime_bootstrap_state = "failed"
    else:
        app.state.runtime_bootstrap_state = "ready"


def _set_bootstrap_phase(app: FastAPI, phase: str) -> None:
    app.state.runtime_bootstrap_phase = str(phase)
    LOGGER.info("runtime bootstrap phase=%s", phase)


async def _bootstrap_runtime_background(app: FastAPI, *, session_id: str, config: RuntimeConfig) -> None:
    timeout_seconds = float(str(os.getenv("VOICE_PIPELINE_BOOTSTRAP_TIMEOUT_SECONDS", "900")).strip() or "900")
    _set_bootstrap_phase(app, "bootstrap_runtime")
    def _progress(phase: str) -> None:
        _set_bootstrap_phase(app, f"bootstrap_runtime:{phase}")

    try:
        runtime = await asyncio.wait_for(
            asyncio.to_thread(bootstrap_runtime, session_id=session_id, config=config, progress_callback=_progress),
            timeout=max(1.0, timeout_seconds),
        )
    except TimeoutError as exc:
        raise RuntimeError(f"runtime_bootstrap_timeout_after_{int(max(1.0, timeout_seconds))}s") from exc
    _set_bootstrap_phase(app, "bridge_init")
    bridge = LiveKitRuntimeBridge(runtime=runtime, transport=runtime.transport)
    runtime.worker_status.transport = "WARMING"
    try:
        _set_bootstrap_phase(app, "bridge_start")
        await bridge.start()
        _set_bootstrap_phase(app, "runtime_start")
        await runtime.start()
    except Exception:
        await bridge.stop()
        await runtime.stop()
        raise
    app.state.runtime = runtime
    app.state.livekit_bridge = bridge
    app.state.runtime_bootstrap_state = "ready"
    _set_bootstrap_phase(app, "ready")


def create_app() -> FastAPI:
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        session_id = os.getenv("VOICE_PIPELINE_SESSION_ID", "voice-runtime")
        config = RuntimeConfig.from_env()
        app.state.runtime = None
        app.state.runtime_config = config
        app.state.livekit_bridge = None
        app.state.runtime_bootstrap_error = ""
        app.state.runtime_bootstrap_state = "warming"
        app.state.runtime_bootstrap_phase = "starting"
        bootstrap_task = asyncio.create_task(
            _bootstrap_runtime_background(app, session_id=session_id, config=config),
            name="voice-runtime-bootstrap",
        )
        app.state.runtime_bootstrap_task = bootstrap_task
        try:
            yield
        finally:
            bootstrap_task = getattr(app.state, "runtime_bootstrap_task", None)
            runtime_bootstrap_error = ""
            if bootstrap_task is not None and not bootstrap_task.done():
                bootstrap_task.cancel()
                try:
                    await bootstrap_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    runtime_bootstrap_error = _describe_exception(exc)
            elif bootstrap_task is not None:
                try:
                    bootstrap_task.result()
                except Exception as exc:
                    runtime_bootstrap_error = _describe_exception(exc)
            if runtime_bootstrap_error:
                app.state.runtime_bootstrap_error = runtime_bootstrap_error
                app.state.runtime_bootstrap_state = "failed"
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
        runtime: VoicePipelineRuntime | None = getattr(app.state, "runtime", None)
        if runtime is None:
            return {"ok": True, "summary": "CPU ASR, GPU0 vLLM, GPU1 CosyVoice3"}
        return {"ok": True, "summary": runtime.dry_run_report()}

    @app.get("/v1/system/config")
    async def system_config(request: Request) -> dict[str, object]:
        runtime: VoicePipelineRuntime | None = getattr(app.state, "runtime", None)
        if runtime is None:
            return _pending_system_config(app.state.runtime_config, request=request)
        return _system_config(runtime, request=request)

    @app.get("/config")
    async def config_alias(request: Request) -> dict[str, object]:
        runtime: VoicePipelineRuntime | None = getattr(app.state, "runtime", None)
        if runtime is None:
            return _pending_system_config(app.state.runtime_config, request=request)
        return _system_config(runtime, request=request)

    @app.get("/v1/system/readiness")
    async def system_readiness() -> dict[str, object]:
        runtime: VoicePipelineRuntime | None = getattr(app.state, "runtime", None)
        if runtime is None:
            _reconcile_bootstrap_state(app)
            if str(getattr(app.state, "runtime_bootstrap_state", "warming")) == "failed":
                return _pending_runtime_readiness(
                    app.state.runtime_config,
                    status="failed",
                    failure_reason=str(getattr(app.state, "runtime_bootstrap_error", "") or ""),
                    bootstrap_state="failed",
                    bootstrap_phase=str(getattr(app.state, "runtime_bootstrap_phase", "") or ""),
                )
            return _pending_runtime_readiness(
                app.state.runtime_config,
                status="warming",
                bootstrap_state=str(getattr(app.state, "runtime_bootstrap_state", "warming")),
                bootstrap_phase=str(getattr(app.state, "runtime_bootstrap_phase", "") or ""),
            )
        return _runtime_readiness(runtime)

    @app.get("/ready")
    async def ready_alias() -> dict[str, object]:
        runtime: VoicePipelineRuntime | None = getattr(app.state, "runtime", None)
        if runtime is None:
            _reconcile_bootstrap_state(app)
            if str(getattr(app.state, "runtime_bootstrap_state", "warming")) == "failed":
                return _pending_runtime_readiness(
                    app.state.runtime_config,
                    status="failed",
                    failure_reason=str(getattr(app.state, "runtime_bootstrap_error", "") or ""),
                    bootstrap_state="failed",
                    bootstrap_phase=str(getattr(app.state, "runtime_bootstrap_phase", "") or ""),
                )
            return _pending_runtime_readiness(
                app.state.runtime_config,
                status="warming",
                bootstrap_state=str(getattr(app.state, "runtime_bootstrap_state", "warming")),
                bootstrap_phase=str(getattr(app.state, "runtime_bootstrap_phase", "") or ""),
            )
        return _runtime_readiness(runtime)

    @app.get("/v1/system/runtime")
    async def system_runtime(limit: int = 20) -> dict[str, object]:
        runtime: VoicePipelineRuntime | None = getattr(app.state, "runtime", None)
        if runtime is None:
            _reconcile_bootstrap_state(app)
            return {
                "available": False,
                "summary": "CPU ASR, GPU0 vLLM, GPU1 CosyVoice3",
                "updated_at": int(now_ns()),
                "bootstrap_error": str(getattr(app.state, "runtime_bootstrap_error", "") or ""),
                "bootstrap_state": str(getattr(app.state, "runtime_bootstrap_state", "warming") or "warming"),
                "bootstrap_phase": str(getattr(app.state, "runtime_bootstrap_phase", "") or ""),
            }
        bridge: LiveKitRuntimeBridge | None = getattr(app.state, "livekit_bridge", None)
        return _runtime_telemetry(runtime, event_limit=int(limit), bridge=bridge)

    @app.post("/v1/system/warmup")
    async def system_warmup() -> dict[str, object]:
        runtime: VoicePipelineRuntime | None = getattr(app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="runtime_not_ready_for_live_audio")
        token = await runtime.warm_vllm_runtime_probe()
        tts_token = await runtime.warm_tts_runtime_probe()
        return {"ok": True, "vllm_probe_token": token, "tts_probe_token": tts_token}

    @app.post("/v1/system/reset")
    async def system_reset() -> dict[str, object]:
        runtime: VoicePipelineRuntime | None = getattr(app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="runtime_not_ready_for_live_audio")
        await runtime.reset_session_state()
        return {"ok": True, "session_id": runtime.kernel.session_id}

    @app.post("/v1/system/ingress-debug")
    async def system_ingress_debug(path: str = "", seconds: float = 0.0) -> dict[str, object]:
        bridge: LiveKitRuntimeBridge | None = getattr(app.state, "livekit_bridge", None)
        if bridge is None:
            raise HTTPException(status_code=503, detail="runtime_not_ready_for_live_audio")
        bridge.configure_debug_ingress_capture(wav_path=str(path), max_seconds=float(seconds))
        return {
            "ok": True,
            "path": str(path),
            "seconds": float(seconds),
            "enabled": bool(str(path).strip()) and float(seconds) > 0.0,
        }

    @app.post("/v1/system/ingress-filter")
    async def system_ingress_filter(request: Request) -> dict[str, object]:
        runtime: VoicePipelineRuntime | None = getattr(app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="runtime_not_ready_for_live_audio")
        payload = await request.json()
        return _set_runtime_ingress_filter(
            runtime,
            identity=str(payload.get("identity", "") or ""),
            track_name=str(payload.get("track_name", "") or ""),
        )

    @app.get("/telemetry")
    async def telemetry_alias(limit: int = 20) -> dict[str, object]:
        runtime: VoicePipelineRuntime | None = getattr(app.state, "runtime", None)
        if runtime is None:
            _reconcile_bootstrap_state(app)
            return {
                "available": False,
                "summary": "CPU ASR, GPU0 vLLM, GPU1 CosyVoice3",
                "updated_at": int(now_ns()),
                "bootstrap_error": str(getattr(app.state, "runtime_bootstrap_error", "") or ""),
                "bootstrap_state": str(getattr(app.state, "runtime_bootstrap_state", "warming") or "warming"),
                "bootstrap_phase": str(getattr(app.state, "runtime_bootstrap_phase", "") or ""),
            }
        bridge: LiveKitRuntimeBridge | None = getattr(app.state, "livekit_bridge", None)
        return _runtime_telemetry(runtime, event_limit=int(limit), bridge=bridge)

    @app.get("/v1/livekit/token")
    async def livekit_token(request: Request, identity: str = "voice-web-client") -> dict[str, object]:
        runtime: VoicePipelineRuntime | None = getattr(app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="runtime_not_ready_for_live_audio")
        if not runtime.global_ready():
            raise HTTPException(status_code=503, detail="runtime_not_ready_for_live_audio")
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
            "url": _client_livekit_url(runtime, request=request),
            "room_name": runtime.transport.config.room_name,
            "identity": str(identity),
            "token": token,
            "turn_enabled": bool(runtime.transport.config.turn_enabled),
        }

    @app.get("/livekit/token")
    async def livekit_token_alias(request: Request, identity: str = "voice-web-client") -> dict[str, object]:
        return await livekit_token(request=request, identity=identity)

    return app


__all__ = ["create_app"]
