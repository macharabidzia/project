from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from time import time
from voice_pipeline.shared.time import now_ns
from typing import Any
import wave

import numpy as np

from voice_pipeline.runtime.bootstrap import VoicePipelineRuntime
from voice_pipeline.transport.livekit_transport import LiveKitTransport

LOGGER = logging.getLogger(__name__)


class _LocalInferenceExecutor:
    def __init__(self) -> None:
        self._runners: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def do_inference(self, method: str, data: bytes) -> bytes | None:
        runner = self._runners.get(str(method))
        if runner is None:
            async with self._lock:
                runner = self._runners.get(str(method))
                if runner is None:
                    from livekit.agents.inference_runner import _InferenceRunner  # type: ignore

                    runner_cls = _InferenceRunner.registered_runners.get(str(method))
                    if runner_cls is None:
                        raise RuntimeError(f"unknown_turn_detector_inference_method: {method}")
                    runner = runner_cls()
                    await asyncio.to_thread(runner.initialize)
                    self._runners[str(method)] = runner
        return await asyncio.to_thread(runner.run, data)


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
    _silero_vad: Any = None
    _turn_detector: Any = None
    _debug_ingress_wav_path: str = ""
    _debug_ingress_max_bytes: int = 0
    _debug_ingress_pcm: bytearray = None  # type: ignore[assignment]
    _debug_ingress_last_flush_bytes: int = 0
    _active_ingress_publication_sid: str = ""
    _active_ingress_participant_identity: str = ""
    _active_ingress_track_name: str = ""
    _last_ingress_lock_publication_sid: str = ""
    _last_ingress_lock_participant_identity: str = ""
    _last_ingress_lock_track_name: str = ""
    _last_ingress_lock_acquired_ns: int = 0
    _last_vad_start_ns: int = 0
    _last_vad_end_ns: int = 0
    _active_ingress_saw_speech: bool = False
    _ingress_tasks_by_publication_sid: dict[str, asyncio.Task[Any]] = None  # type: ignore[assignment]
    _last_prefix_flush_frames: int = 0
    _last_prefix_flush_span_ms: float = 0.0
    _last_forward_delay_ms: float = 0.0
    _max_forward_delay_ms: float = 0.0

    def __post_init__(self) -> None:
        self._tasks = set()
        self._debug_ingress_wav_path = str(os.getenv("VOICE_PIPELINE_DEBUG_SAVE_INGRESS_WAV", "")).strip()
        debug_seconds = max(0.0, float(os.getenv("VOICE_PIPELINE_DEBUG_SAVE_INGRESS_SECONDS", "0") or 0.0))
        bytes_per_second = (
            int(self.transport.config.input_sample_rate)
            * int(self.transport.config.num_channels)
            * 2
        )
        self._debug_ingress_max_bytes = int(debug_seconds * float(bytes_per_second))
        self._debug_ingress_pcm = bytearray()
        self._debug_ingress_last_flush_bytes = 0
        self._active_ingress_publication_sid = ""
        self._active_ingress_participant_identity = ""
        self._active_ingress_track_name = ""
        self._last_ingress_lock_publication_sid = ""
        self._last_ingress_lock_participant_identity = ""
        self._last_ingress_lock_track_name = ""
        self._last_ingress_lock_acquired_ns = 0
        self._last_vad_start_ns = 0
        self._last_vad_end_ns = 0
        self._active_ingress_saw_speech = False
        self._ingress_tasks_by_publication_sid = {}
        self._last_prefix_flush_frames = 0
        self._last_prefix_flush_span_ms = 0.0
        self._last_forward_delay_ms = 0.0
        self._max_forward_delay_ms = 0.0

    def configure_debug_ingress_capture(self, *, wav_path: str, max_seconds: float) -> None:
        self._debug_ingress_wav_path = str(wav_path or "").strip()
        if not self._debug_ingress_wav_path:
            self._debug_ingress_max_bytes = 0
            self._debug_ingress_pcm.clear()
            self._debug_ingress_last_flush_bytes = 0
            return
        debug_seconds = max(0.0, float(max_seconds))
        bytes_per_second = (
            int(self.transport.config.input_sample_rate)
            * int(self.transport.config.num_channels)
            * 2
        )
        self._debug_ingress_max_bytes = int(debug_seconds * float(bytes_per_second))
        self._debug_ingress_pcm.clear()
        self._debug_ingress_last_flush_bytes = 0

    def ingress_lock_state(self) -> dict[str, object]:
        return {
            "active": bool(str(self._active_ingress_publication_sid or "").strip()),
            "publication_sid": str(self._active_ingress_publication_sid or "").strip(),
            "participant_identity": str(self._active_ingress_participant_identity or "").strip(),
            "track_name": str(self._active_ingress_track_name or "").strip(),
            "last_lock_publication_sid": str(self._last_ingress_lock_publication_sid or "").strip(),
            "last_lock_participant_identity": str(self._last_ingress_lock_participant_identity or "").strip(),
            "last_lock_track_name": str(self._last_ingress_lock_track_name or "").strip(),
            "last_lock_acquired_ns": int(self._last_ingress_lock_acquired_ns),
            "last_vad_start_ns": int(self._last_vad_start_ns),
            "last_vad_end_ns": int(self._last_vad_end_ns),
            "active_ingress_saw_speech": bool(self._active_ingress_saw_speech),
            "last_prefix_flush_frames": int(self._last_prefix_flush_frames),
            "last_prefix_flush_span_ms": float(self._last_prefix_flush_span_ms),
            "last_forward_delay_ms": float(self._last_forward_delay_ms),
            "max_forward_delay_ms": float(self._max_forward_delay_ms),
        }

    def _add_task(self, coro: Any, *, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(lambda done: self._tasks.discard(done))
        return task

    async def _finalize_after_ingress_unsubscribe(
        self,
        *,
        publication_sid: str,
        consume_task: asyncio.Task[Any] | None,
        saw_speech: bool,
    ) -> None:
        if not saw_speech:
            return
        input_frame_ms = max(1, int(self.transport.config.input_frame_ms))
        max_wait_seconds = max(0.20, min(0.75, float(input_frame_ms * 25) / 1000.0))
        poll_seconds = max(0.02, min(0.05, float(input_frame_ms) / 1000.0))
        if consume_task is not None and not consume_task.done():
            deadline = asyncio.get_running_loop().time() + max_wait_seconds
            while not consume_task.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0.0:
                    LOGGER.info(
                        "livekit bridge ingress finalize fallback after unsubscribe timeout",
                        extra={
                            "publication_sid": publication_sid,
                            "max_wait_seconds": max_wait_seconds,
                        },
                    )
                    break
                try:
                    await asyncio.wait_for(
                        asyncio.shield(consume_task),
                        timeout=min(poll_seconds, remaining),
                    )
                    break
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    return
                except Exception:
                    LOGGER.exception("livekit ingress consume task failed before finalize fallback")
                    break
        transcript = self.runtime.kernel.state.transcript
        committed_text = " ".join(str(transcript.committed_text or "").split())
        candidate_text = " ".join(
            str(
                transcript.final_text
                or transcript.partial_text
                or transcript.stable_prefix
                or transcript.last_dispatched_stable_prefix
                or ""
            ).split()
        )
        if not candidate_text or candidate_text == committed_text:
            return
        await self.runtime.finalize_asr_turn()
        self._active_ingress_saw_speech = False

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
            LOGGER.info("livekit bridge start: loading silero vad")
            self._silero_vad = self._build_silero_vad()
            LOGGER.info("livekit bridge start: silero vad ready")
            LOGGER.info("livekit bridge start: loading turn detector")
            self._turn_detector = self._build_turn_detector()
            LOGGER.info("livekit bridge start: turn detector ready")
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
                participant_identity = str(getattr(participant, "identity", "") or "").strip()
                if participant_identity == str(self.transport.config.runtime_identity):
                    return
                publication_name = str(getattr(publication, "name", "") or getattr(track, "name", "") or "").strip()
                publication_sid = str(getattr(publication, "sid", "") or "").strip()
                expected_identity = str(self.transport.config.input_participant_identity or "").strip()
                expected_track_name = str(self.transport.config.input_track_name or "").strip()
                if expected_identity and participant_identity != expected_identity:
                    LOGGER.info(
                        "livekit bridge ingress skip: unexpected participant",
                        extra={
                            "participant_identity": participant_identity,
                            "publication_name": publication_name,
                            "publication_sid": publication_sid,
                        },
                    )
                    return
                if expected_track_name and publication_name != expected_track_name:
                    LOGGER.info(
                        "livekit bridge ingress skip: unexpected track",
                        extra={
                            "participant_identity": participant_identity,
                            "publication_name": publication_name,
                            "publication_sid": publication_sid,
                        },
                    )
                    return
                if bool(self.transport.config.single_ingress_track):
                    active_sid = str(self._active_ingress_publication_sid or "").strip()
                    if active_sid and publication_sid and publication_sid != active_sid:
                        LOGGER.info(
                            "livekit bridge ingress skip: second remote track",
                            extra={
                                "participant_identity": participant_identity,
                                "publication_name": publication_name,
                                "publication_sid": publication_sid,
                                "active_publication_sid": active_sid,
                            },
                        )
                        return
                    if publication_sid and not active_sid:
                        self._active_ingress_publication_sid = publication_sid
                        self._active_ingress_participant_identity = participant_identity
                        self._active_ingress_track_name = publication_name
                        self._active_ingress_saw_speech = False
                        self._last_ingress_lock_publication_sid = publication_sid
                        self._last_ingress_lock_participant_identity = participant_identity
                        self._last_ingress_lock_track_name = publication_name
                        self._last_ingress_lock_acquired_ns = int(now_ns())
                        LOGGER.info(
                            "livekit bridge ingress lock acquired",
                            extra={
                                "participant_identity": participant_identity,
                                "publication_name": publication_name,
                                "publication_sid": publication_sid,
                            },
                        )
                consume_task = self._add_task(
                    self._consume_remote_audio(track),
                    name=f"livekit-ingress-{getattr(publication, 'sid', 'audio')}",
                )
                if publication_sid:
                    self._ingress_tasks_by_publication_sid[publication_sid] = consume_task
                    consume_task.add_done_callback(
                        lambda done, sid=publication_sid: self._ingress_tasks_by_publication_sid.pop(sid, None)
                    )

            @room.on("track_unsubscribed")
            def _on_track_unsubscribed(track: Any, publication: Any, participant: Any) -> None:
                publication_sid = str(getattr(publication, "sid", "") or "").strip()
                if publication_sid and publication_sid == str(self._active_ingress_publication_sid or "").strip():
                    saw_speech = bool(self._active_ingress_saw_speech)
                    LOGGER.info(
                        "livekit bridge ingress lock released",
                        extra={
                            "participant_identity": str(getattr(participant, "identity", "") or "").strip(),
                            "publication_name": str(getattr(publication, "name", "") or getattr(track, "name", "") or "").strip(),
                            "publication_sid": publication_sid,
                        },
                    )
                    self._active_ingress_publication_sid = ""
                    self._active_ingress_participant_identity = ""
                    self._active_ingress_track_name = ""
                    self._add_task(
                        self._finalize_after_ingress_unsubscribe(
                            publication_sid=publication_sid,
                            consume_task=self._ingress_tasks_by_publication_sid.get(publication_sid),
                            saw_speech=saw_speech,
                        ),
                        name=f"livekit-unsubscribe-finalize-{publication_sid or 'audio'}",
                    )
            LOGGER.info("livekit bridge start: connecting room")
            await asyncio.wait_for(
                room.connect(
                    str(self.transport.config.url),
                    token,
                    options=rtc.RoomOptions(
                        auto_subscribe=True,
                        single_peer_connection=bool(self.transport.config.single_peer_connection),
                    ),
                ),
                timeout=30.0,
            )
            LOGGER.info("livekit bridge start: room connected")

            source = rtc.AudioSource(
                int(self.transport.config.output_sample_rate),
                int(self.transport.config.num_channels),
                queue_size_ms=int(self.transport.config.output_queue_size_ms),
            )
            track = rtc.LocalAudioTrack.create_audio_track(str(self.transport.config.output_track_name), source)
            publish_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            publish_options.dtx = False
            publish_options.red = False
            publish_options.preconnect_buffer = bool(self.transport.config.output_preconnect_buffer)
            LOGGER.info("livekit bridge start: publishing output track")
            await asyncio.wait_for(
                room.local_participant.publish_track(track, publish_options),
                timeout=30.0,
            )
            LOGGER.info("livekit bridge start: output track published")

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
        self._flush_debug_ingress_capture()

    def _record_debug_ingress_pcm(self, pcm_bytes: bytes) -> None:
        if not self._debug_ingress_wav_path or self._debug_ingress_max_bytes <= 0:
            return
        remaining = int(self._debug_ingress_max_bytes) - len(self._debug_ingress_pcm)
        if remaining <= 0:
            return
        self._debug_ingress_pcm.extend(bytes(pcm_bytes[:remaining]))
        current_size = len(self._debug_ingress_pcm)
        if current_size >= int(self._debug_ingress_max_bytes) or current_size >= int(self._debug_ingress_last_flush_bytes) + 4096:
            self._flush_debug_ingress_capture()

    def _flush_debug_ingress_capture(self) -> None:
        if not self._debug_ingress_wav_path:
            return
        try:
            if not self._debug_ingress_pcm:
                return
            with wave.open(str(self._debug_ingress_wav_path), "wb") as wav_file:
                wav_file.setnchannels(int(self.transport.config.num_channels))
                wav_file.setsampwidth(2)
                wav_file.setframerate(int(self.transport.config.input_sample_rate))
                wav_file.writeframes(bytes(self._debug_ingress_pcm))
            self._debug_ingress_last_flush_bytes = len(self._debug_ingress_pcm)
        except Exception:
            return

    async def _consume_remote_audio(self, track: Any) -> None:
        from livekit import rtc  # type: ignore

        stream = rtc.AudioStream(
            track=track,
            sample_rate=int(self.transport.config.input_sample_rate),
            num_channels=int(self.transport.config.num_channels),
            frame_size_ms=int(self.transport.config.input_frame_ms),
        )
        frame_bytes_target = max(
            2,
            int(int(self.transport.config.input_sample_rate) * int(self.transport.config.input_frame_ms) / 1000)
            * int(self.transport.config.num_channels)
            * 2,
        )
        vad_window_ms = 32
        vad_frame_bytes = max(
            2,
            int(int(self.transport.config.input_sample_rate) * int(vad_window_ms) / 1000)
            * int(self.transport.config.num_channels)
            * 2,
        )
        if self._silero_vad is None:
            raise RuntimeError("livekit_silero_vad_required")
        saw_active_speech = False
        speech_active = False
        forward_all_audio = bool(getattr(self.runtime.config, "livekit_forward_all_audio", True))
        vad_stream = self._silero_vad.stream()
        vad_generation = 0
        pending_finalize_task: asyncio.Task[Any] | None = None
        pending_tail_finalize_generation = 0
        post_vad_tail_frames_remaining = 0
        prefix_padding_frames = max(
            0,
            int(round(float(self.runtime.config.livekit_silero_vad_prefix_padding_ms) / max(1.0, float(self.transport.config.input_frame_ms)))),
        )
        prefix_buffer: deque[tuple[int, bytes]] = deque(
            maxlen=max(1, prefix_padding_frames) if prefix_padding_frames > 0 else 1
        )
        forward_queue: asyncio.Queue[tuple[int, bytes, bool] | None] = asyncio.Queue()
        frame_interval_seconds = max(0.0, float(self.transport.config.input_frame_ms) / 1000.0)
        prefix_replay_spacing_seconds = min(frame_interval_seconds, 0.010) if frame_interval_seconds > 0.0 else 0.010
        vad_pcm_buffer = bytearray()
        vad_buffer_oldest_ns = 0

        async def _forward_pcm_frames() -> None:
            last_observed_ns = 0
            last_item_replayed_prefix = False
            while True:
                queued_item = await forward_queue.get()
                if queued_item is None:
                    break
                observed_ns, pcm_bytes, replayed_prefix = queued_item
                if last_observed_ns > 0 and observed_ns > last_observed_ns:
                    delay_seconds = 0.0
                    observed_gap_seconds = min(
                        0.25,
                        max(0.0, float(observed_ns - last_observed_ns) / 1_000_000_000.0),
                    )
                    if replayed_prefix and frame_interval_seconds > 0.0:
                        # Preserve spacing for ASR stability, but compress the
                        # replayed prefix so VAD recovery does not add the full
                        # original prefix span back into startup latency.
                        delay_seconds = prefix_replay_spacing_seconds
                    elif last_item_replayed_prefix and frame_interval_seconds > 0.0:
                        # After replaying the preserved prefix, switch to live
                        # speech immediately instead of paying the original gap
                        # between the buffered prefix tail and current audio.
                        delay_seconds = min(observed_gap_seconds, frame_interval_seconds)
                    if delay_seconds > 0.0:
                        self._last_forward_delay_ms = float(delay_seconds * 1000.0)
                        self._max_forward_delay_ms = max(
                            float(self._max_forward_delay_ms),
                            float(self._last_forward_delay_ms),
                        )
                        await asyncio.sleep(delay_seconds)
                last_observed_ns = max(last_observed_ns, int(observed_ns))
                last_item_replayed_prefix = bool(replayed_prefix)
                try:
                    await self.runtime.process_pcm_frame(pcm_bytes)
                except RuntimeError as exc:
                    if str(exc) == "runtime_not_ready_for_live_audio":
                        self.transport.record_ingress_drop()
                        continue
                    raise

        forward_task = asyncio.create_task(_forward_pcm_frames(), name="livekit-ingress-forward")

        async def _schedule_turn_finalize(delay_seconds: float, generation: int) -> None:
            nonlocal saw_active_speech, speech_active
            await asyncio.sleep(max(0.0, float(delay_seconds)))
            if generation != vad_generation:
                return
            await self.runtime.finalize_asr_turn()
            saw_active_speech = False
            speech_active = False

        async def _consume_vad_events() -> None:
            nonlocal saw_active_speech
            nonlocal speech_active
            nonlocal vad_generation
            nonlocal pending_finalize_task
            nonlocal pending_tail_finalize_generation
            nonlocal post_vad_tail_frames_remaining
            try:
                async for vad_event in vad_stream:
                    event_type = str(getattr(vad_event, "type", ""))
                    if event_type == "VADEventType.START_OF_SPEECH" or event_type.endswith("start_of_speech"):
                        self._last_vad_start_ns = int(now_ns())
                        self.runtime.note_vad_speech_start(self._last_vad_start_ns)
                        self._active_ingress_saw_speech = True
                        vad_generation += 1
                        pending_tail_finalize_generation = 0
                        post_vad_tail_frames_remaining = 0
                        saw_active_speech = True
                        speech_active = True
                        if pending_finalize_task is not None:
                            pending_finalize_task.cancel()
                            pending_finalize_task = None
                        if forward_all_audio:
                            prefix_buffer.clear()
                            self._last_prefix_flush_frames = 0
                            self._last_prefix_flush_span_ms = 0.0
                        else:
                            prefix_items = list(prefix_buffer)
                            self._last_prefix_flush_frames = len(prefix_items)
                            if len(prefix_items) >= 2:
                                self._last_prefix_flush_span_ms = float(
                                    max(0, int(prefix_items[-1][0]) - int(prefix_items[0][0])) / 1_000_000.0
                                )
                            else:
                                self._last_prefix_flush_span_ms = 0.0
                            while prefix_buffer:
                                observed_ns, pcm_bytes = prefix_buffer.popleft()
                                await forward_queue.put((observed_ns, pcm_bytes, True))
                    elif event_type == "VADEventType.END_OF_SPEECH" or event_type.endswith("end_of_speech"):
                        self._last_vad_end_ns = int(now_ns())
                        if saw_active_speech:
                            generation = vad_generation
                            self._active_ingress_saw_speech = False
                            if pending_finalize_task is not None:
                                pending_finalize_task.cancel()
                                pending_finalize_task = None
                            if forward_all_audio:
                                tail_ms = max(
                                    0,
                                    int(getattr(self.runtime.config, "livekit_post_vad_tail_ms", 120) or 0),
                                )
                                post_vad_tail_frames_remaining = max(
                                    1,
                                    int(round(float(tail_ms) / max(1.0, float(self.transport.config.input_frame_ms)))),
                                )
                                pending_tail_finalize_generation = generation
                            else:
                                delay_seconds = await self._turn_finalize_delay_seconds()
                                pending_finalize_task = asyncio.create_task(
                                    _schedule_turn_finalize(delay_seconds, generation),
                                    name="livekit-turn-finalize",
                                )
            except RuntimeError as exc:
                if str(exc) == "runtime_not_ready_for_live_audio":
                    return
                raise
            except asyncio.CancelledError:
                return
            except Exception:
                LOGGER.exception("silero vad event consumer failed")

        vad_task = asyncio.create_task(_consume_vad_events(), name="livekit-silero-vad")
        async for frame_event in stream:
            if not self._running:
                break
            if not self.runtime.global_ready():
                self.transport.record_ingress_drop()
                continue
            frame = getattr(frame_event, "frame", frame_event)
            data_obj = getattr(frame, "data", b"")
            pcm_bytes = bytes(memoryview(data_obj).cast("B")) if data_obj is not None else b""
            if len(pcm_bytes) < frame_bytes_target:
                self.transport.record_ingress_drop()
                continue
            trimmed = pcm_bytes[:frame_bytes_target]
            observed_ns = int(now_ns())
            self.transport.record_ingress_frame(len(trimmed))
            self._record_debug_ingress_pcm(trimmed)
            if not vad_pcm_buffer:
                vad_buffer_oldest_ns = observed_ns
            vad_pcm_buffer.extend(trimmed)
            while len(vad_pcm_buffer) >= vad_frame_bytes:
                vad_chunk = bytes(vad_pcm_buffer[:vad_frame_bytes])
                del vad_pcm_buffer[:vad_frame_bytes]
                samples_per_channel = max(
                    1,
                    int(len(vad_chunk) // (2 * int(self.transport.config.num_channels))),
                )
                vad_stream.push_frame(
                    rtc.AudioFrame(
                        data=vad_chunk,
                        sample_rate=int(self.transport.config.input_sample_rate),
                        num_channels=int(self.transport.config.num_channels),
                        samples_per_channel=samples_per_channel,
                    )
                )
                if vad_pcm_buffer:
                    vad_buffer_oldest_ns = min(
                        observed_ns,
                        int(vad_buffer_oldest_ns) + int(vad_window_ms * 1_000_000),
                    )
                else:
                    vad_buffer_oldest_ns = 0
            if forward_all_audio or speech_active:
                await forward_queue.put((observed_ns, trimmed, False))
            elif prefix_padding_frames > 0:
                prefix_buffer.append((observed_ns, trimmed))
            if post_vad_tail_frames_remaining > 0:
                post_vad_tail_frames_remaining = max(0, int(post_vad_tail_frames_remaining) - 1)
                if post_vad_tail_frames_remaining == 0 and pending_tail_finalize_generation == vad_generation:
                    delay_seconds = await self._turn_finalize_delay_seconds()
                    pending_finalize_task = asyncio.create_task(
                        _schedule_turn_finalize(delay_seconds, int(pending_tail_finalize_generation)),
                        name="livekit-turn-finalize",
                    )
                    pending_tail_finalize_generation = 0
        if vad_pcm_buffer:
            padded = bytes(vad_pcm_buffer) + (b"\x00" * max(0, vad_frame_bytes - len(vad_pcm_buffer)))
            samples_per_channel = max(
                1,
                int(len(padded) // (2 * int(self.transport.config.num_channels))),
            )
            vad_stream.push_frame(
                rtc.AudioFrame(
                    data=padded,
                    sample_rate=int(self.transport.config.input_sample_rate),
                    num_channels=int(self.transport.config.num_channels),
                    samples_per_channel=samples_per_channel,
                )
            )
            vad_pcm_buffer.clear()
        vad_stream.end_input()
        await vad_task
        await forward_queue.put(None)
        await forward_task
        if post_vad_tail_frames_remaining > 0 and saw_active_speech and pending_finalize_task is None:
            await self.runtime.finalize_asr_turn()
            self._active_ingress_saw_speech = False
            saw_active_speech = False
            speech_active = False
        if pending_finalize_task is not None:
            try:
                await pending_finalize_task
            except asyncio.CancelledError:
                pass
        elif saw_active_speech:
            await self.runtime.finalize_asr_turn()
            self._active_ingress_saw_speech = False

    def _build_silero_vad(self) -> Any:
        if not bool(getattr(self.runtime.config, "livekit_use_silero_vad", True)):
            raise RuntimeError("livekit_silero_vad_required")
        try:
            from livekit.plugins import silero  # type: ignore
        except Exception as exc:
            raise RuntimeError("livekit_silero_vad_unavailable") from exc
        return silero.VAD.load(
            min_speech_duration=max(0.0, float(self.runtime.config.livekit_silero_vad_min_speech_ms) / 1000.0),
            min_silence_duration=max(0.0, float(self.runtime.config.livekit_silero_vad_min_silence_ms) / 1000.0),
            prefix_padding_duration=max(0.0, float(self.runtime.config.livekit_silero_vad_prefix_padding_ms) / 1000.0),
            activation_threshold=float(self.runtime.config.livekit_silero_vad_activation_threshold),
            sample_rate=16000,
            force_cpu=True,
        )

    def _build_turn_detector(self) -> Any:
        if not bool(getattr(self.runtime.config, "livekit_use_turn_detector", True)):
            raise RuntimeError("livekit_turn_detector_required")
        try:
            from livekit.plugins.turn_detector.base import EOUModelBase  # type: ignore
            from livekit.plugins.turn_detector.english import _EUORunnerEn  # type: ignore
        except Exception as exc:
            raise RuntimeError("livekit_turn_detector_unavailable") from exc
        try:
            class _BridgeEnglishTurnModel(EOUModelBase):
                def __init__(self, *, inference_executor: Any, unlikely_threshold: float | None = None):
                    super().__init__(
                        model_type="en",
                        inference_executor=inference_executor,
                        unlikely_threshold=unlikely_threshold,
                    )

                def _inference_method(self) -> str:
                    return _EUORunnerEn.INFERENCE_METHOD

            executor = _LocalInferenceExecutor()
            return _BridgeEnglishTurnModel(
                inference_executor=executor,
                unlikely_threshold=float(self.runtime.config.livekit_turn_detector_unlikely_threshold),
            )
        except Exception as exc:
            raise RuntimeError("livekit_turn_detector_init_failed") from exc

    async def _turn_finalize_delay_seconds(self) -> float:
        min_delay = max(0.0, float(self.runtime.config.livekit_turn_detector_min_endpoint_ms) / 1000.0)
        max_delay = max(
            min_delay,
            float(self.runtime.config.livekit_turn_detector_max_endpoint_ms) / 1000.0,
        )
        detector = self._turn_detector
        if detector is None:
            return max_delay
        try:
            from livekit.agents import llm  # type: ignore

            transcript = self.runtime.kernel.state.transcript
            current_text = " ".join(
                str(
                    transcript.final_text
                    or transcript.partial_text
                    or transcript.stable_prefix
                    or transcript.last_dispatched_stable_prefix
                    or ""
                ).split()
            )
            if not current_text:
                return max_delay
            chat_ctx = llm.ChatContext()
            for item in tuple(transcript.conversation_history)[-4:]:
                text = " ".join(str(item or "").split())
                if text:
                    chat_ctx.add_message(role="user", content=text)
            chat_ctx.add_message(role="user", content=current_text)
            probability = float(await detector.predict_end_of_turn(chat_ctx, timeout=1.0))
            threshold = float(self.runtime.config.livekit_turn_detector_unlikely_threshold)
            return min_delay if probability >= threshold else max_delay
        except Exception:
            LOGGER.exception("livekit turn detector inference failed; using max delay")
            return max_delay

    @staticmethod
    def _pcm_frame_rms(pcm_bytes: bytes) -> float:
        if not pcm_bytes:
            return 0.0
        samples = np.frombuffer(bytes(pcm_bytes), dtype="<i2").astype(np.float32)
        if samples.size == 0:
            return 0.0
        normalized = samples / 32768.0
        return float(np.sqrt(np.mean(np.square(normalized))))

    @staticmethod
    def _pcm_frame_peak(pcm_bytes: bytes) -> float:
        if not pcm_bytes:
            return 0.0
        samples = np.frombuffer(bytes(pcm_bytes), dtype="<i2").astype(np.float32)
        if samples.size == 0:
            return 0.0
        normalized = np.abs(samples / 32768.0)
        return float(np.max(normalized))

    async def _emit_runtime_pcm(self) -> None:
        from livekit import rtc  # type: ignore

        async def _send_frame(pcm_bytes: bytes, sample_rate: int, request_id: str) -> None:
            if not self._running or not pcm_bytes:
                return
            if self._audio_source is None:
                return
            channels = int(self.transport.config.num_channels)
            normalized_pcm = self._normalize_egress_pcm_frame(pcm_bytes)
            sample_rate = int(sample_rate)
            bytes_per_sample = 2
            direct_capture = int(self.transport.config.output_queue_size_ms) <= 0
            if direct_capture:
                target_samples_per_channel = max(1, int(round(float(sample_rate) * 0.01)))
            else:
                target_samples_per_channel = max(1, int(len(normalized_pcm) / (bytes_per_sample * channels)))
            frame_bytes = max(
                bytes_per_sample * channels,
                int(target_samples_per_channel) * bytes_per_sample * channels,
            )
            for offset in range(0, len(normalized_pcm), frame_bytes):
                chunk = bytes(normalized_pcm[offset : offset + frame_bytes])
                if not chunk:
                    continue
                if len(chunk) < frame_bytes:
                    chunk = chunk + (b"\x00" * (frame_bytes - len(chunk)))
                frame = rtc.AudioFrame(
                    data=bytearray(chunk),
                    sample_rate=sample_rate,
                    num_channels=channels,
                    samples_per_channel=target_samples_per_channel,
                )
                await self._audio_source.capture_frame(frame)
            if str(request_id or "").strip():
                self.runtime.mark_livekit_egress(now_ns())
            self.transport.record_egress_frame(
                len(normalized_pcm),
                frame_rms=self._pcm_frame_rms(normalized_pcm),
                frame_peak=self._pcm_frame_peak(normalized_pcm),
                request_id=str(request_id or ""),
            )

        while self._running:
            await self.runtime.send_pcm_once(_send_frame)


__all__ = ["LiveKitRuntimeBridge", "create_livekit_access_token"]
