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
    input_participant_identity: str = ""
    input_track_name: str = ""
    input_track_source: str = "microphone"
    frame_ms: int = 20
    input_frame_ms: int = 20
    input_sample_rate: int = 48_000
    output_sample_rate: int = 48_000
    num_channels: int = 1
    input_queue_size_ms: int = 40
    output_queue_size_ms: int = 40
    output_preconnect_buffer: bool = False
    single_peer_connection: bool = False
    single_ingress_track: bool = True
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
        self._egress_last_rms = 0.0
        self._egress_last_peak = 0.0
        self._egress_max_rms = 0.0
        self._egress_max_peak = 0.0
        self._egress_active_request_id = ""
        self._egress_request_frames = 0
        self._egress_request_bytes = 0
        self._egress_request_last_rms = 0.0
        self._egress_request_last_peak = 0.0
        self._egress_request_max_rms = 0.0
        self._egress_request_max_peak = 0.0
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

    def start_egress_request(self, request_id: str) -> None:
        self._egress_active_request_id = str(request_id or "").strip()
        self._egress_request_frames = 0
        self._egress_request_bytes = 0
        self._egress_request_last_rms = 0.0
        self._egress_request_last_peak = 0.0
        self._egress_request_max_rms = 0.0
        self._egress_request_max_peak = 0.0

    def record_egress_frame(
        self,
        payload_size: int,
        *,
        frame_rms: float = 0.0,
        frame_peak: float = 0.0,
        request_id: str = "",
    ) -> None:
        self._egress_frames += 1
        self._egress_bytes += max(0, int(payload_size))
        self._egress_last_rms = float(max(0.0, frame_rms))
        self._egress_last_peak = float(max(0.0, frame_peak))
        self._egress_max_rms = max(float(self._egress_max_rms), float(max(0.0, frame_rms)))
        self._egress_max_peak = max(float(self._egress_max_peak), float(max(0.0, frame_peak)))
        if str(request_id or "").strip() == str(self._egress_active_request_id):
            self._egress_request_frames += 1
            self._egress_request_bytes += max(0, int(payload_size))
            self._egress_request_last_rms = float(max(0.0, frame_rms))
            self._egress_request_last_peak = float(max(0.0, frame_peak))
            self._egress_request_max_rms = max(float(self._egress_request_max_rms), float(max(0.0, frame_rms)))
            self._egress_request_max_peak = max(float(self._egress_request_max_peak), float(max(0.0, frame_peak)))

    def ingress_metrics(self) -> dict[str, int]:
        return {
            "transport_ingress_frames": int(self._ingress_frames),
            "transport_ingress_bytes": int(self._ingress_bytes),
            "transport_ingress_dropped": int(self._ingress_dropped),
            "transport_egress_frames": int(self._egress_frames),
            "transport_egress_bytes": int(self._egress_bytes),
            "transport_egress_last_rms": float(self._egress_last_rms),
            "transport_egress_last_peak": float(self._egress_last_peak),
            "transport_egress_max_rms": float(self._egress_max_rms),
            "transport_egress_max_peak": float(self._egress_max_peak),
            "transport_egress_request_id": str(self._egress_active_request_id),
            "transport_egress_request_frames": int(self._egress_request_frames),
            "transport_egress_request_bytes": int(self._egress_request_bytes),
            "transport_egress_request_last_rms": float(self._egress_request_last_rms),
            "transport_egress_request_last_peak": float(self._egress_request_last_peak),
            "transport_egress_request_max_rms": float(self._egress_request_max_rms),
            "transport_egress_request_max_peak": float(self._egress_request_max_peak),
            "transport_bridge_connected": 1 if self._bridge_connected else 0,
        }


__all__ = ["LiveKitTransport", "LiveKitTransportConfig"]
