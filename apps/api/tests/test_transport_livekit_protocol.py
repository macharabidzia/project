from __future__ import annotations

from voice_pipeline.transport.livekit_transport import LiveKitTransport, LiveKitTransportConfig


def test_livekit_transport_metrics_track_ingress_egress_and_drops() -> None:
    transport = LiveKitTransport(config=LiveKitTransportConfig())
    transport.record_ingress_frame(1920)
    transport.record_ingress_drop()
    transport.record_egress_frame(960)
    transport.mark_bridge_connected()
    metrics = transport.ingress_metrics()

    assert metrics["transport_ingress_frames"] == 1
    assert metrics["transport_ingress_bytes"] == 1920
    assert metrics["transport_ingress_dropped"] == 1
    assert metrics["transport_egress_frames"] == 1
    assert metrics["transport_egress_bytes"] == 960
    assert metrics["transport_bridge_connected"] == 1
