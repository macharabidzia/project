from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LiveKitTransportConfig:
    url: str = "ws://127.0.0.1:7880"
    api_key: str = ""
    api_secret: str = ""
    room_name: str = "voice-runtime"
    runtime_identity: str = "voice-runtime-backend"
    output_track_name: str = "voice-runtime-out"
    input_track_source: str = "microphone"
    frame_ms: int = 20
    input_sample_rate: int = 48_000
    output_sample_rate: int = 48_000
    num_channels: int = 1
    token_ttl_seconds: int = 3600
    turn_enabled: bool = True


class LiveKitTransport:
    def __init__(self, *, config: LiveKitTransportConfig) -> None:
        self.config = config
        self._ingress_frames = 0
        self._ingress_bytes = 0
        self._ingress_dropped = 0
        self._egress_frames = 0
        self._egress_bytes = 0
        self._bridge_connected = False

    def mark_bridge_connected(self) -> None:
        self._bridge_connected = True

    def mark_bridge_disconnected(self) -> None:
        self._bridge_connected = False

    def record_ingress_frame(self, payload_size: int) -> None:
        self._ingress_frames += 1
        self._ingress_bytes += max(0, int(payload_size))

    def record_ingress_drop(self) -> None:
        self._ingress_dropped += 1

    def record_egress_frame(self, payload_size: int) -> None:
        self._egress_frames += 1
        self._egress_bytes += max(0, int(payload_size))

    def ingress_metrics(self) -> dict[str, int]:
        return {
            "transport_ingress_frames": int(self._ingress_frames),
            "transport_ingress_bytes": int(self._ingress_bytes),
            "transport_ingress_dropped": int(self._ingress_dropped),
            "transport_egress_frames": int(self._egress_frames),
            "transport_egress_bytes": int(self._egress_bytes),
            "transport_bridge_connected": 1 if self._bridge_connected else 0,
        }


__all__ = ["LiveKitTransport", "LiveKitTransportConfig"]
