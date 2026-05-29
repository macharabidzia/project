from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from voice_pipeline.runtime import server
from voice_pipeline.runtime.livekit_bridge import LiveKitRuntimeBridge
from voice_pipeline.transport.livekit_transport import LiveKitTransport, LiveKitTransportConfig


class _FakeRuntime:
    def __init__(self) -> None:
        self.transport = SimpleNamespace(
            config=SimpleNamespace(
                url="ws://127.0.0.1:7880",
                room_name="voice-runtime",
                runtime_identity="voice-runtime-backend",
                output_track_name="voice-runtime-out",
                turn_enabled=True,
                api_key="devkey",
                api_secret="devsecret",
                token_ttl_seconds=3600,
            ),
            ingress_metrics=lambda: {},
        )
        self.kernel = SimpleNamespace(
            session_id="livekit-e2e",
            queued_event_count=0,
            state=SimpleNamespace(output=SimpleNamespace(version=1)),
            current_lease=lambda: SimpleNamespace(epoch_id="livekit-e2e:epoch:1"),
            runtime_metrics=lambda: {"stale_pcm_drop_count": 0},
        )
        self.config = SimpleNamespace(
            asr_device="cpu",
            asr_model_path="D:/models/vosk",
            asr_sample_rate=16_000,
            input_sample_rate=48_000,
            llm_device="cuda:0",
            tts_device="cuda:1",
            vllm_cache_dir="D:/models/cache/vllm",
            cosyvoice3_cache_dir="D:/models/cache/cosyvoice3",
            output_sample_rate=48_000,
            frame_ms=20,
            resolved_vllm_model_path=lambda: "D:/models/vllm/Qwen3-8B",
            resolved_cosyvoice3_model_path=lambda: "D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512",
        )
        self.worker_status = SimpleNamespace(asr="READY", vllm="READY", tts="READY", kernel="READY", transport="READY")
        self.warm_report = SimpleNamespace(asr_warm=True, vllm_warm=True, tts_warm=True)
        self.model_cache_identity = {"model_cache_hash": "hash"}
        self.startup_contract_hash = "contract-hash"
        self.event_log = SimpleNamespace(events=[])
        self.pcm_clock = SimpleNamespace(depth=0, max_buffer_frames=3, dropped_overflow_frames=0, dropped_stale_frames=0, oldest_age_ms=0.0)
        self.vllm = SimpleNamespace(prefix_cache_ready=True, cache_stats=lambda: SimpleNamespace(hits=1, misses=0, hit_ratio=1.0))

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def global_ready(self) -> bool:
        return True

    def dry_run_report(self) -> str:
        return "CPU ASR, GPU0 vLLM, GPU1 CosyVoice3"

    def latency_summary(self) -> dict[str, object]:
        return {}

    def replay_event_hash(self) -> str:
        return "replay-hash"

    def last_timestamps(self) -> dict[str, int]:
        return {
            "ingress_received_ns": 0,
            "asr_event_ns": 0,
            "kernel_decision_ns": 0,
            "vllm_first_token_ns": 0,
            "tts_first_pcm_ns": 0,
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


def test_livekit_token_endpoint_returns_local_room_credentials(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(server, "bootstrap_runtime", lambda session_id, config: runtime)
    monkeypatch.setattr(server, "LiveKitRuntimeBridge", _FakeBridge)

    app = server.create_app()
    with TestClient(app) as client:
        response = client.get("/v1/livekit/token?identity=voice-web-test")
        payload = response.json()

    assert response.status_code == 200
    assert payload["url"] == "ws://127.0.0.1:7880"
    assert payload["room_name"] == "voice-runtime"
    assert payload["identity"] == "voice-web-test"
    assert isinstance(payload["token"], str)
    assert len(payload["token"].split(".")) == 3


def test_livekit_token_alias_matches_primary_endpoint(monkeypatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(server, "bootstrap_runtime", lambda session_id, config: runtime)
    monkeypatch.setattr(server, "LiveKitRuntimeBridge", _FakeBridge)

    app = server.create_app()
    with TestClient(app) as client:
        primary = client.get("/v1/livekit/token?identity=voice-web-test").json()
        alias = client.get("/livekit/token?identity=voice-web-test").json()

    assert alias["url"] == primary["url"]
    assert alias["room_name"] == primary["room_name"]
    assert alias["identity"] == primary["identity"]
    assert alias["turn_enabled"] == primary["turn_enabled"]
    assert len(str(alias["token"]).split(".")) == 3
    assert len(str(primary["token"]).split(".")) == 3


def test_livekit_token_uses_public_url_when_configured(monkeypatch) -> None:
    runtime = _FakeRuntime()
    runtime.config.livekit_public_url = "wss://example-7880.proxy.runpod.net"
    monkeypatch.setattr(server, "bootstrap_runtime", lambda session_id, config: runtime)
    monkeypatch.setattr(server, "LiveKitRuntimeBridge", _FakeBridge)

    app = server.create_app()
    with TestClient(app) as client:
        payload = client.get("/v1/livekit/token?identity=voice-web-test").json()

    assert payload["url"] == "wss://example-7880.proxy.runpod.net"


def test_server_lifespan_activates_transport_before_tick_loop(monkeypatch) -> None:
    call_order: list[str] = []

    class _OrderedRuntime(_FakeRuntime):
        async def start(self) -> None:
            call_order.append("runtime.start")

        async def stop(self) -> None:
            call_order.append("runtime.stop")

    class _OrderedBridge(_FakeBridge):
        async def start(self) -> None:
            call_order.append("bridge.start")
            self.runtime.worker_status.transport = "READY"

        async def stop(self) -> None:
            call_order.append("bridge.stop")

    runtime = _OrderedRuntime()
    monkeypatch.setattr(server, "bootstrap_runtime", lambda session_id, config: runtime)
    monkeypatch.setattr(server, "LiveKitRuntimeBridge", _OrderedBridge)

    app = server.create_app()
    with TestClient(app) as _client:
        pass

    assert call_order.index("bridge.start") < call_order.index("runtime.start")
    assert call_order.index("bridge.stop") < call_order.index("runtime.stop")


def test_livekit_bridge_normalizes_egress_frame_to_exact_20ms() -> None:
    bridge = LiveKitRuntimeBridge(
        runtime=SimpleNamespace(),
        transport=LiveKitTransport(
            config=LiveKitTransportConfig(
                frame_ms=20,
                output_sample_rate=48_000,
                num_channels=1,
            )
        ),
    )
    expected = int(48_000 * 20 / 1000) * 2
    short = b"\x01\x02" * 100
    normalized_short = bridge._normalize_egress_pcm_frame(short)
    assert len(normalized_short) == expected
    assert normalized_short.startswith(short)
    long = b"\x01\x02" * 1_500
    normalized_long = bridge._normalize_egress_pcm_frame(long)
    assert len(normalized_long) == expected
    assert normalized_long == long[:expected]
