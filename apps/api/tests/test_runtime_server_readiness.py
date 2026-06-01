from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from voice_pipeline.runtime import server


class _FakeRuntime:
    def __init__(self) -> None:
        self.transport = SimpleNamespace(
            config=SimpleNamespace(
                url="ws://127.0.0.1:7880",
                room_name="voice-runtime",
                runtime_identity="voice-runtime-backend",
                output_track_name="voice-runtime-out",
                input_participant_identity="voice-test-client",
                input_track_name="voice-test-input",
                input_frame_ms=20,
                single_ingress_track=True,
                turn_enabled=True,
                api_key="devkey",
                api_secret="devsecret",
                token_ttl_seconds=3600,
            ),
            ingress_metrics=lambda: {},
        )
        self.kernel = SimpleNamespace(
            session_id="ready-runtime",
            queued_event_count=0,
            state=SimpleNamespace(output=SimpleNamespace(version=1)),
            current_lease=lambda: SimpleNamespace(epoch_id="ready-runtime:epoch:1"),
            runtime_metrics=lambda: {"ingress_queue_oldest_age_ms": 0.0, "ingress_drop_count": 0, "stale_pcm_drop_count": 0, "backpressure_action": "none"},
        )
        self.vllm = SimpleNamespace(
            prefix_cache_ready=True,
            cache_stats=lambda: SimpleNamespace(hits=1, misses=0, hit_ratio=1.0),
        )
        self.config = SimpleNamespace(
            asr_device="cpu",
            asr_model_path="D:/models/vosk",
            asr_sample_rate=16_000,
            llm_device="cuda:0",
            vllm_cache_dir="D:/models/cache/vllm",
            tts_device="cuda:1",
            cosyvoice3_cache_dir="D:/models/cache/cosy",
            frame_ms=20,
            ingress_max_items=2048,
            resolved_vllm_model_path=lambda: "D:/models/vllm/Qwen3-8B",
            resolved_cosyvoice3_model_path=lambda: "D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512",
        )
        self.worker_status = SimpleNamespace(asr="READY", vllm="READY", tts="READY", kernel="READY", transport="READY")
        self.warm_report = SimpleNamespace(asr_warm=True, vllm_warm=True, tts_warm=True)
        self.model_cache_identity = {"model_cache_hash": "model-cache"}
        self.startup_contract_hash = "topology-hash"
        self.event_log = SimpleNamespace(events=[])
        self.pcm_clock = SimpleNamespace(depth=0, max_buffer_frames=3, dropped_overflow_frames=0, dropped_stale_frames=0, oldest_age_ms=0.0)

    async def start(self) -> None:
        self.worker_status.transport = "READY"

    async def stop(self) -> None:
        return None

    def global_ready(self) -> bool:
        return True

    def dry_run_report(self) -> str:
        return "CPU ASR, GPU0 vLLM, GPU1 CosyVoice3"

    def replay_event_hash(self) -> str:
        return "replay-hash"

    def latency_summary(self) -> dict[str, object]:
        return {}

    def last_timestamps(self) -> dict[str, int]:
        return {
            "ingress_received_ns": 0,
            "asr_event_ns": 0,
            "kernel_decision_ns": 0,
            "vllm_first_token_ns": 0,
            "tts_text_push_ns": 0,
            "tts_native_stream_open_ns": 0,
            "tts_first_pcm_ns": 0,
            "tts_gate_open_ns": 0,
            "pcm_enqueue_ns": 0,
            "transport_emit_ns": 0,
        }


class _FakeBridge:
    def __init__(self, runtime, transport) -> None:
        self.runtime = runtime
        self.transport = transport

    async def start(self) -> None:
        self.runtime.worker_status.transport = "READY"

    async def stop(self) -> None:
        return None


def test_runtime_readiness_endpoint_has_required_contract_keys(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(server, "bootstrap_runtime", lambda session_id, config, progress_callback=None: runtime)
    monkeypatch.setattr(server, "LiveKitRuntimeBridge", _FakeBridge)
    app = server.create_app()

    with TestClient(app) as client:
        payload = client.get("/ready").json()

    assert payload["ready"] is True
    assert payload["asr_device"] == "cpu"
    assert payload["llm_device"] == "cuda:0"
    assert payload["tts_device"] == "cuda:1"
    assert "topology_hash" in payload
    assert "model_cache_hash" in payload
    assert "drift_snapshot_hash" in payload
