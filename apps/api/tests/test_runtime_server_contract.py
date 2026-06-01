from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from voice_pipeline.runtime import server


class _FakeLease:
    def __init__(self, epoch_id: str) -> None:
        self.epoch_id = epoch_id


class _FakeKernel:
    def __init__(self) -> None:
        self.session_id = "runtime-contract"
        self.queued_event_count = 0
        self.state = SimpleNamespace(
            phase="idle",
            turn_index=0,
            committed_turn_index=0,
            generation_index=0,
            lineage_id="runtime-contract:epoch:1",
            active_vllm_request_id="",
            active_tts_request_id="",
            transcript=SimpleNamespace(
                partial_text="",
                partial_history=(),
                stable_prefix="",
                stable_prefix_confirmations=0,
                last_dispatched_stable_prefix="",
                final_text="",
                committed_text="",
                conversation_history=(),
            ),
            output=SimpleNamespace(
                active_turn_id="",
                version=1,
                vllm_tokens=(),
                vllm_stream_buffer=(),
                pending_tts_segments=(),
                emitted_audio_chunk_ids=(),
            ),
        )

    def current_lease(self) -> _FakeLease:
        return _FakeLease("runtime-contract:epoch:1")

    def runtime_metrics(self) -> dict[str, object]:
        return {
            "ingress_queue_depth": 0,
            "ingress_queue_oldest_age_ms": 0.0,
            "ingress_drop_count": 0,
            "stale_pcm_drop_count": 0,
            "backpressure_action": "none",
        }


class _FakeVLLM:
    @property
    def prefix_cache_ready(self) -> bool:
        return True

    @staticmethod
    def cache_stats():
        return SimpleNamespace(hits=3, misses=1, hit_ratio=0.75)


class _FakePCMClock:
    depth = 0
    max_buffer_frames = 3
    dropped_overflow_frames = 0
    dropped_stale_frames = 0
    oldest_age_ms = 0.0


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
        self.kernel = _FakeKernel()
        self.vllm = _FakeVLLM()
        self.tts = SimpleNamespace(debug_metrics=lambda: {"last_backend_path": "", "last_native_mode": "", "last_native_text": ""})
        class _Config:
            asr_device = "cpu"
            asr_model_path = "D:/models/vosk"
            asr_sample_rate = 16_000
            llm_device = "cuda:0"
            tts_device = "cuda:1"
            vllm_cache_dir = "D:/models/cache/vllm"
            cosyvoice3_cache_dir = "D:/models/cache/cosyvoice3"
            frame_ms = 20
            ingress_max_items = 2048

            @staticmethod
            def resolved_vllm_model_path() -> str:
                return "D:/models/vllm/Qwen3-8B"

            @staticmethod
            def resolved_cosyvoice3_model_path() -> str:
                return "D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512"

        self.config = _Config()
        self.worker_status = SimpleNamespace(asr="READY", vllm="READY", tts="READY", kernel="READY", transport="READY")
        self.warm_report = SimpleNamespace(asr_warm=True, vllm_warm=True, tts_warm=True)
        self.model_cache_identity = {"model_cache_hash": "model-cache-hash"}
        self.startup_contract_hash = "topology-hash"
        self.event_log = SimpleNamespace(events=[])
        self.pcm_clock = _FakePCMClock()

    async def start(self) -> None:
        self.worker_status.transport = "READY"

    async def stop(self) -> None:
        return None

    def global_ready(self) -> bool:
        return True

    def dry_run_report(self) -> str:
        return "CPU ASR, GPU0 vLLM, GPU1 CosyVoice3"

    def replay_event_hash(self) -> str:
        return "drift-snapshot"

    def latency_summary(self) -> dict[str, object]:
        return {}

    def tts_signal_metrics(self) -> dict[str, object]:
        return {}

    def ingress_frame_trace(self) -> tuple[dict[str, object], ...]:
        return ()

    def asr_event_trace(self) -> tuple[dict[str, object], ...]:
        return ()

    def last_timestamps(self) -> dict[str, int]:
        return {
            "ingress_received_ns": 1,
            "vad_speech_start_ns": 2,
            "first_asr_partial_ns": 3,
            "stable_asr_partial_ns": 4,
            "asr_final_ns": 5,
            "asr_event_ns": 2,
            "kernel_decision_ns": 3,
            "vllm_request_start_ns": 6,
            "vllm_first_token_ns": 7,
            "first_spoken_delta_ns": 8,
            "tts_text_push_ns": 9,
            "tts_native_stream_open_ns": 10,
            "tts_first_pcm_ns": 11,
            "tts_gate_open_ns": 12,
            "resampler_first_output_ns": 13,
            "pcm_enqueue_ns": 14,
            "pcm_send_ns": 15,
            "transport_emit_ns": 16,
            "livekit_egress_ns": 17,
        }


class _FakeBridge:
    def __init__(self, runtime, transport) -> None:
        self.runtime = runtime
        self.transport = transport

    async def start(self) -> None:
        self.runtime.worker_status.transport = "READY"

    async def stop(self) -> None:
        return None

    def ingress_lock_state(self) -> dict[str, object]:
        return {
            "active": False,
            "publication_sid": "",
            "participant_identity": "",
            "track_name": "",
        }


def test_runtime_readiness_contains_required_fields(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(server, "bootstrap_runtime", lambda session_id, config, progress_callback=None: runtime)
    monkeypatch.setattr(server, "LiveKitRuntimeBridge", _FakeBridge)

    app = server.create_app()
    with TestClient(app) as client:
        payload = client.get("/v1/system/readiness").json()

    required_fields = {
        "asr_status",
        "asr_device",
        "asr_model_path",
        "asr_sample_rate",
        "llm_status",
        "llm_device",
        "llm_model_path",
        "llm_cache_dir",
        "llm_prefix_cache_ready",
        "llm_prefix_cache_hits",
        "llm_prefix_cache_misses",
        "llm_prefix_cache_hit_ratio",
        "tts_status",
        "tts_device",
        "tts_model_path",
        "tts_cache_dir",
        "tts_native_bistream_ready",
        "kernel_status",
        "transport_status",
        "topology_hash",
        "model_cache_hash",
        "drift_snapshot_hash",
    }
    assert required_fields.issubset(payload.keys())
    assert payload["ready"] is True
    assert payload["llm_prefix_cache_ready"] is True
    assert payload["llm_prefix_cache_hits"] == 3
    assert payload["llm_prefix_cache_misses"] == 1
    assert payload["llm_prefix_cache_hit_ratio"] == 0.75
    assert payload["kernel_status"] == "READY"


def test_runtime_telemetry_contains_contract_timestamps_and_backpressure(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(server, "bootstrap_runtime", lambda session_id, config, progress_callback=None: runtime)
    monkeypatch.setattr(server, "LiveKitRuntimeBridge", _FakeBridge)

    app = server.create_app()
    with TestClient(app) as client:
        payload = client.get("/v1/system/runtime").json()

    assert set(payload["timestamps"].keys()) == {
        "ingress_received_ns",
        "vad_speech_start_ns",
        "first_asr_partial_ns",
        "stable_asr_partial_ns",
        "asr_final_ns",
        "asr_event_ns",
        "kernel_decision_ns",
        "vllm_request_start_ns",
        "vllm_first_token_ns",
        "first_spoken_delta_ns",
        "tts_text_push_ns",
        "tts_native_stream_open_ns",
        "tts_first_pcm_ns",
        "tts_gate_open_ns",
        "resampler_first_output_ns",
        "pcm_enqueue_ns",
        "pcm_send_ns",
        "transport_emit_ns",
        "livekit_egress_ns",
    }
    assert payload["backpressure"]["text_oldest_age_ms"] >= 0
    assert payload["backpressure"]["text_drops"] >= 0
    assert "backpressure_action" in payload["backpressure"]
    assert payload["backpressure"]["audio_stale_drops"] >= 0
    assert payload["llm_prefix_cache_stats"]["hits"] == 3
    assert payload["llm_prefix_cache_stats"]["misses"] == 1
    assert payload["tts_backend_path"] == ""
    assert payload["tts_first_pcm_ms"] >= 0
    assert payload["llm_first_token_ms"] >= 0
    assert payload["tts_first_text_to_first_pcm_ms"] >= 0
    assert payload["transport_delay_ms"] >= 0


def test_runtime_alias_endpoints_match_v1_system_routes(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(server, "bootstrap_runtime", lambda session_id, config, progress_callback=None: runtime)
    monkeypatch.setattr(server, "LiveKitRuntimeBridge", _FakeBridge)

    app = server.create_app()
    with TestClient(app) as client:
        assert client.get("/ready").json() == client.get("/v1/system/readiness").json()

        telemetry_alias = client.get("/telemetry").json()
        telemetry_v1 = client.get("/v1/system/runtime").json()
        assert telemetry_alias["summary"] == telemetry_v1["summary"]
        assert telemetry_alias["backpressure"] == telemetry_v1["backpressure"]
        assert telemetry_alias["timestamps"] == telemetry_v1["timestamps"]

        assert client.get("/config").json() == client.get("/v1/system/config").json()


def test_runtime_readiness_is_false_when_kernel_not_ready(monkeypatch) -> None:
    runtime = _FakeRuntime()
    runtime.worker_status.kernel = "WARMING"
    monkeypatch.setattr(server, "bootstrap_runtime", lambda session_id, config, progress_callback=None: runtime)
    monkeypatch.setattr(server, "LiveKitRuntimeBridge", _FakeBridge)

    app = server.create_app()
    with TestClient(app) as client:
        payload = client.get("/v1/system/readiness").json()

    assert payload["kernel_status"] == "WARMING"
    assert payload["ready"] is False
    assert payload["session_eligible"] is False
