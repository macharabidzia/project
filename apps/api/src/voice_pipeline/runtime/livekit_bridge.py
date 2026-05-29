from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from time import time
from typing import Any

from voice_pipeline.runtime.bootstrap import VoicePipelineRuntime
from voice_pipeline.transport.livekit_transport import LiveKitTransport


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def create_livekit_access_token(
    *,
    api_key: str,
    api_secret: str,
    identity: str,
    room_name: str,
    can_publish: bool = True,
    can_subscribe: bool = True,
    ttl_seconds: int = 3600,
    name: str | None = None,
) -> str:
    if not str(api_key).strip():
        raise RuntimeError("livekit_api_key_missing")
    if not str(api_secret).strip():
        raise RuntimeError("livekit_api_secret_missing")
    now = int(time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload: dict[str, Any] = {
        "iss": str(api_key),
        "sub": str(identity),
        "nbf": now - 5,
        "exp": now + max(60, int(ttl_seconds)),
        "video": {
            "roomJoin": True,
            "room": str(room_name),
            "canPublish": bool(can_publish),
            "canSubscribe": bool(can_subscribe),
        },
    }
    if name:
        payload["name"] = str(name)
    header_part = _b64url(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload_part = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{header_part}.{payload_part}".encode("ascii")
    signature = hmac.new(str(api_secret).encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_part}.{payload_part}.{_b64url(signature)}"


@dataclass(slots=True)
class LiveKitRuntimeBridge:
    runtime: VoicePipelineRuntime
    transport: LiveKitTransport
    _room: Any = None
    _audio_source: Any = None
    _audio_track: Any = None
    _tasks: set[asyncio.Task[Any]] = None  # type: ignore[assignment]
    _running: bool = False

    def __post_init__(self) -> None:
        self._tasks = set()

    def _add_task(self, coro: Any, *, name: str) -> None:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(lambda done: self._tasks.discard(done))

    def _normalize_egress_pcm_frame(self, pcm_bytes: bytes) -> bytes:
        frame_ms = max(1, int(self.transport.config.frame_ms))
        sample_rate = max(1, int(self.transport.config.output_sample_rate))
        channels = max(1, int(self.transport.config.num_channels))
        expected_bytes = max(2, int(sample_rate * frame_ms / 1000) * channels * 2)
        payload = bytes(pcm_bytes or b"")
        if len(payload) >= expected_bytes:
            return payload[:expected_bytes]
        return payload + (b"\x00" * (expected_bytes - len(payload)))

    async def start(self) -> None:
        if self._running:
            return
        try:
            from livekit import rtc  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            self.runtime.worker_status.transport = "FAILED"
            self.runtime.worker_failure_reason.transport = "livekit_python_sdk_missing: install `livekit`"
            raise RuntimeError("livekit_python_sdk_missing: install `livekit`") from exc

        try:
            token = create_livekit_access_token(
                api_key=self.transport.config.api_key,
                api_secret=self.transport.config.api_secret,
                identity=self.transport.config.runtime_identity,
                room_name=self.transport.config.room_name,
                can_publish=True,
                can_subscribe=True,
                ttl_seconds=int(self.transport.config.token_ttl_seconds),
                name="Voice Runtime Backend",
            )

            room = rtc.Room()

            @room.on("track_subscribed")
            def _on_track_subscribed(track: Any, publication: Any, participant: Any) -> None:
                if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
                    return
                if str(getattr(participant, "identity", "")) == str(self.transport.config.runtime_identity):
                    return
                self._add_task(
                    self._consume_remote_audio(track),
                    name=f"livekit-ingress-{getattr(publication, 'sid', 'audio')}",
                )

            await room.connect(str(self.transport.config.url), token, options=rtc.RoomOptions(auto_subscribe=True))

            source = rtc.AudioSource(
                int(self.transport.config.output_sample_rate),
                int(self.transport.config.num_channels),
            )
            track = rtc.LocalAudioTrack.create_audio_track(str(self.transport.config.output_track_name), source)
            publish_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            await room.local_participant.publish_track(track, publish_options)

            self._room = room
            self._audio_source = source
            self._audio_track = track
            self._running = True
            self.transport.mark_bridge_connected()
            self.runtime.worker_failure_reason.transport = ""
            self.runtime.worker_status.transport = "READY"
            self._add_task(self._emit_runtime_pcm(), name="livekit-egress")
        except Exception as exc:
            self.transport.mark_bridge_disconnected()
            self.runtime.worker_status.transport = "FAILED"
            self.runtime.worker_failure_reason.transport = str(exc)
            raise

    async def stop(self) -> None:
        self._running = False
        self.transport.mark_bridge_disconnected()
        for task in tuple(self._tasks):
            task.cancel()
        for task in tuple(self._tasks):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._tasks.clear()
        if self._room is not None:
            await self._room.disconnect()
            self._room = None
        audio_source = self._audio_source
        self._audio_source = None
        if audio_source is not None:
            close_fn = getattr(audio_source, "aclose", None)
            if callable(close_fn):
                await close_fn()

    async def _consume_remote_audio(self, track: Any) -> None:
        from livekit import rtc  # type: ignore

        stream = rtc.AudioStream(
            track=track,
            sample_rate=int(self.transport.config.input_sample_rate),
            num_channels=int(self.transport.config.num_channels),
            frame_size_ms=int(self.transport.config.frame_ms),
        )
        frame_bytes_target = max(
            2,
            int(int(self.transport.config.input_sample_rate) * int(self.transport.config.frame_ms) / 1000)
            * int(self.transport.config.num_channels)
            * 2,
        )
        async for frame_event in stream:
            if not self._running:
                break
            frame = getattr(frame_event, "frame", frame_event)
            data_obj = getattr(frame, "data", b"")
            pcm_bytes = bytes(memoryview(data_obj).cast("B")) if data_obj is not None else b""
            if len(pcm_bytes) < frame_bytes_target:
                self.transport.record_ingress_drop()
                continue
            trimmed = pcm_bytes[:frame_bytes_target]
            self.transport.record_ingress_frame(len(trimmed))
            await self.runtime.process_pcm_frame(trimmed)

    async def _emit_runtime_pcm(self) -> None:
        from livekit import rtc  # type: ignore

        async def _send_frame(pcm_bytes: bytes, sample_rate: int) -> None:
            if not self._running or not pcm_bytes:
                return
            if self._audio_source is None:
                return
            channels = int(self.transport.config.num_channels)
            normalized_pcm = self._normalize_egress_pcm_frame(pcm_bytes)
            samples_per_channel = max(1, int(len(normalized_pcm) / (2 * channels)))
            frame = rtc.AudioFrame(
                data=normalized_pcm[: samples_per_channel * channels * 2],
                sample_rate=int(sample_rate),
                num_channels=channels,
                samples_per_channel=samples_per_channel,
            )
            await self._audio_source.capture_frame(frame)
            self.transport.record_egress_frame(len(normalized_pcm))

        while self._running:
            await self.runtime.send_pcm_once(_send_frame)


__all__ = ["LiveKitRuntimeBridge", "create_livekit_access_token"]
