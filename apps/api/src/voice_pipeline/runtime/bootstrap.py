from __future__ import annotations

import asyncio
import hashlib
import json
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from threading import Thread
from typing import Any

import numpy as np

from voice_pipeline.bus.ring_topology import RingTopology
from voice_pipeline.bus.ring_types import EventType
from voice_pipeline.gpu.tts_worker.engine import TTSEngine
from voice_pipeline.gpu.tts_worker.stream import TTSAudioStreamer
from voice_pipeline.gpu.vllm_worker.engine import VLLMEngine, VLLMEngineConfig
from voice_pipeline.gpu.vllm_worker.stream import VLLMTokenStreamer
from voice_pipeline.kernel.dispatch import (
    DispatchCommand,
    build_tts_cancel_command,
    build_vllm_cancel_command,
)
from voice_pipeline.kernel.kernel_runtime import KernelConfig, KernelRuntime
from voice_pipeline.kernel.recovery import build_recovery_snapshot
from voice_pipeline.observability.metrics import LatencySummary, summarize_latency
from voice_pipeline.replay.determinism import canonical_event_stream_hash, canonical_state_hash
from voice_pipeline.replay.event_log import EventLog
from voice_pipeline.runtime.admission_gate import hardware_admission_check
from voice_pipeline.runtime.config import RuntimeConfig
from voice_pipeline.runtime.topology import LaneConfig, RuntimeTopology
from voice_pipeline.shared.audio_resample import StreamingAudioResampler
from voice_pipeline.shared.time import now_ns
from voice_pipeline.shared.types import AuthorityEvent, new_authority_event
from voice_pipeline.stt.asr_engine import ASREngine, ASREvent, ASRRuntimeConfig
from voice_pipeline.transport.pcm_clock import PCMClockSender, PCMFrame
from voice_pipeline.transport.livekit_transport import LiveKitTransport, LiveKitTransportConfig


@dataclass(slots=True)
class WarmReport:
    asr_warm: bool = False
    vllm_warm: bool = False
    tts_warm: bool = False


@dataclass(slots=True)
class WorkerStatus:
    asr: str = "WARMING"
    vllm: str = "WARMING"
    tts: str = "WARMING"
    kernel: str = "WARMING"
    transport: str = "WARMING"


@dataclass(slots=True)
class WorkerFailureReason:
    asr: str = ""
    vllm: str = ""
    tts: str = ""
    kernel: str = ""
    transport: str = ""


@dataclass(slots=True)
class TopologyReport:
    asr: str
    vllm: str
    tts: str
    pcm: str


@dataclass(slots=True)
class VoicePipelineRuntime:
    kernel: KernelRuntime
    asr: ASREngine
    vllm: VLLMEngine
    tts: TTSEngine
    warm_report: WarmReport
    config: RuntimeConfig
    topology: RuntimeTopology
    rings: RingTopology
    transport: LiveKitTransport
    pcm_clock: PCMClockSender
    event_log: EventLog = field(default_factory=EventLog)
    worker_status: WorkerStatus = field(default_factory=WorkerStatus)
    worker_failure_reason: WorkerFailureReason = field(default_factory=WorkerFailureReason)
    model_cache_identity: dict[str, str] = field(default_factory=dict)
    startup_contract_hash: str = ""
    _vllm_streamer: VLLMTokenStreamer | None = None
    _tts_streamer: TTSAudioStreamer | None = None
    _output_resampler: StreamingAudioResampler | None = None
    _tick_task: asyncio.Task[None] | None = None
    _run_event: asyncio.Event = field(default_factory=asyncio.Event)
    _dispatch_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _latency_samples: dict[str, list[float]] = field(
        default_factory=lambda: {"asr": [], "kernel": [], "vllm": [], "tts": [], "transport": []}
    )
    _last_timestamps_ns: dict[str, int] = field(
        default_factory=lambda: {
            "ingress_received_ns": 0,
            "asr_event_ns": 0,
            "kernel_decision_ns": 0,
            "vllm_first_token_ns": 0,
            "tts_first_pcm_ns": 0,
            "transport_emit_ns": 0,
        }
    )
    _tts_frame_carry: bytes = b""
    _tts_frame_carry_epoch_id: str = ""
    _tts_frame_carry_output_version: int = -1

    def __post_init__(self) -> None:
        self._vllm_streamer = VLLMTokenStreamer(self.vllm)
        self._tts_streamer = TTSAudioStreamer(self.tts)
        self._output_resampler = StreamingAudioResampler(target_rate=int(self.config.output_sample_rate))

    def global_ready(self) -> bool:
        return (
            bool(self.warm_report.asr_warm)
            and bool(self.warm_report.vllm_warm)
            and bool(self.warm_report.tts_warm)
            and str(self.worker_status.kernel) == "READY"
            and str(self.worker_status.transport) == "READY"
        )

    def assert_ready_for_live_audio(self) -> None:
        if not self.global_ready():
            raise RuntimeError("runtime_not_ready_for_live_audio")

    def tick(self):
        return self.kernel.tick()

    def topology_report(self) -> TopologyReport:
        return TopologyReport(asr="CPU ASR", vllm="GPU0 vLLM", tts="GPU1 CosyVoice3", pcm="CPU PCM")

    def dry_run_report(self) -> str:
        report = self.topology_report()
        return f"{report.asr}, {report.vllm}, {report.tts}"

    def next_sequence_no(self) -> int:
        return self.kernel.next_sequence_no()

    async def start(self) -> None:
        if self._tick_task is not None and not self._tick_task.done():
            return
        self._run_event.set()
        self._tick_task = asyncio.create_task(self._tick_loop(), name="voice-pipeline-kernel-tick")

    async def stop(self) -> None:
        self._run_event.clear()
        if self._tick_task is None:
            return
        self._tick_task.cancel()
        try:
            await self._tick_task
        except asyncio.CancelledError:
            pass
        finally:
            self._tick_task = None

    async def run_forever(self) -> None:
        await self.start()
        while self._run_event.is_set():
            await asyncio.sleep(max(0.001, float(self.config.tick_interval_ms) / 1000.0))

    async def _tick_loop(self) -> None:
        try:
            while self._run_event.is_set():
                if self.kernel.queued_event_count:
                    await self.run_tick_and_dispatch()
                else:
                    await asyncio.sleep(max(0.001, float(self.config.tick_interval_ms) / 1000.0))
        except asyncio.CancelledError:
            raise

    def _record_latency(self, key: str, started_ns: int, ended_ns: int) -> None:
        self._latency_samples.setdefault(key, []).append(float(max(0, ended_ns - started_ns)) / 1_000_000.0)

    def _authority_event(
        self,
        *,
        event_type: str,
        lineage_id: str,
        payload: dict[str, object],
        causation_id: str = "",
    ) -> AuthorityEvent:
        event_payload = dict(payload)
        event_payload.setdefault("timestamp_ns", int(now_ns()))
        event_payload.setdefault("lane", event_type)
        return new_authority_event(
            event_type=event_type,
            session_id=self.kernel.session_id,
            sequence_no=self.next_sequence_no(),
            lineage_id=lineage_id,
            payload=event_payload,
            causation_id=causation_id,
            observations={
                "ingress_received_ns": int(event_payload.get("ingress_received_ns", event_payload["timestamp_ns"])),
                "trace_id": str(event_payload.get("trace_id", "")),
            },
        )

    def _append_event(self, event: AuthorityEvent) -> None:
        self.event_log.append(
            {
                "event_id": event.event_id,
                "type": event.event_type,
                "lineage": event.lineage_id,
                "payload": dict(event.payload),
                "ts": int(event.payload.get("timestamp_ns", 0) or 0),
            }
        )
        self._mirror_authority_event_to_ring(event)
        self.kernel.enqueue_event(event)

    def _asr_events_to_authority(
        self,
        events: tuple[ASREvent, ...],
        *,
        ingress_received_ns: int,
    ) -> tuple[AuthorityEvent, ...]:
        authority_events: list[AuthorityEvent] = []
        for event in events:
            authority_events.append(
                self._authority_event(
                    event_type=event.event_type,
                    lineage_id=str(event.lineage_id),
                    payload={
                        "text": str(event.text),
                        "timestamp_ns": int(event.emitted_at_ns),
                        "lane": "asr",
                        "asr_event_ns": int(event.emitted_at_ns),
                        "ingress_received_ns": int(ingress_received_ns),
                    },
                )
            )
        return tuple(authority_events)

    @staticmethod
    def _encode_slot_payload(payload: dict[str, object]) -> bytes:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")

    def _push_lane_slot(
        self,
        *,
        lane: str,
        event_type: EventType,
        payload: bytes,
        lineage_id: str,
        sequence_no: int,
        epoch_id: str,
        metadata: tuple[tuple[str, object], ...] = (),
    ) -> None:
        if self.rings is None:
            return
        lane_name = str(lane).strip().lower()
        if lane_name == "asr":
            lane_ring = self.rings.asr_ring
        elif lane_name == "vllm":
            lane_ring = self.rings.vllm_ring
        elif lane_name == "tts":
            lane_ring = self.rings.tts_ring
        elif lane_name == "pcm":
            lane_ring = self.rings.pcm_ring
        else:
            return
        self.rings.kernel_stream_ring.push_bytes(
            event_type=event_type,
            payload=bytes(payload),
            lineage_id=str(lineage_id),
            sequence_no=int(sequence_no),
            epoch_id=str(epoch_id),
            metadata=metadata,
        )
        _ = lane_ring.pop()

    def _mirror_authority_event_to_ring(self, event: AuthorityEvent) -> None:
        event_type = str(event.event_type)
        if event_type.startswith("ASR"):
            lane = "asr"
            slot_type = EventType.ASR_SLOT
        elif event_type.startswith("VLLM"):
            lane = "vllm"
            slot_type = EventType.VLLM_TOKEN_SLOT
        elif event_type.startswith("TTS"):
            lane = "tts"
            slot_type = EventType.TTS_REQUEST_SLOT
        else:
            return
        payload = self._encode_slot_payload(
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "causation_id": event.causation_id,
                "payload": dict(event.payload),
            }
        )
        self._push_lane_slot(
            lane=lane,
            event_type=slot_type,
            payload=payload,
            lineage_id=event.lineage_id,
            sequence_no=int(event.sequence_no),
            epoch_id=str(event.observations.get("epoch_id", event.lineage_id)),
            metadata=(("authority_event_type", event.event_type),),
        )

    def _mirror_dispatch_command_to_ring(self, command: DispatchCommand) -> None:
        payload = dict(command.payload)
        lineage_id = str(payload.get("lineage_id", self.kernel.current_lease().epoch_id))
        epoch_id = str(payload.get("epoch_id", self.kernel.current_lease().epoch_id))
        sequence_no = max(1, int(self.kernel.next_sequence_no()))
        encoded = self._encode_slot_payload(
            {
                "kind": command.kind,
                "request_id": command.request_id,
                "payload": payload,
            }
        )
        if command.kind == "VLLM":
            self._push_lane_slot(
                lane="vllm",
                event_type=EventType.VLLM_REQUEST_SLOT,
                payload=encoded,
                lineage_id=lineage_id,
                sequence_no=sequence_no,
                epoch_id=epoch_id,
                metadata=(("dispatch_kind", command.kind),),
            )
        elif command.kind == "TTS":
            self._push_lane_slot(
                lane="tts",
                event_type=EventType.TTS_REQUEST_SLOT,
                payload=encoded,
                lineage_id=lineage_id,
                sequence_no=sequence_no,
                epoch_id=epoch_id,
                metadata=(("dispatch_kind", command.kind),),
            )

    def _mirror_pcm_frame_to_ring(self, frame: PCMFrame) -> None:
        self._push_lane_slot(
            lane="pcm",
            event_type=EventType.PCM_SLOT,
            payload=bytes(frame.pcm),
            lineage_id=str(frame.epoch_id),
            sequence_no=max(1, int(self.kernel.next_sequence_no())),
            epoch_id=str(frame.epoch_id),
            metadata=(
                ("sample_rate", int(frame.sample_rate)),
                ("output_version", int(frame.output_version)),
            ),
        )

    async def process_pcm_frame(self, pcm: bytes, *, session_id: str | None = None) -> tuple[PCMFrame, ...]:
        _ = session_id
        self.assert_ready_for_live_audio()
        ingress_received_ns = now_ns()
        self._last_timestamps_ns["ingress_received_ns"] = int(ingress_received_ns)
        asr_started_ns = now_ns()
        asr_events = self.asr.ingest_audio(bytes(pcm), lineage_id=self.kernel.current_lease().epoch_id)
        self._record_latency("asr", asr_started_ns, now_ns())
        authority_events = self._asr_events_to_authority(asr_events, ingress_received_ns=int(ingress_received_ns))
        for event in authority_events:
            payload = dict(event.payload)
            self._last_timestamps_ns["asr_event_ns"] = int(payload.get("asr_event_ns", 0) or 0)
            self._append_event(event)
        return await self.run_tick_and_dispatch()

    async def _tick_and_stamp_commands(self) -> tuple[DispatchCommand, ...]:
        pre_vllm_request_id = ""
        pre_tts_request_id = ""
        pre_epoch_id = ""
        pre_vllm_output_version = -1
        pre_tts_output_version = -1
        async with self._dispatch_lock:
            pre_vllm_request_id = str(self.kernel.state.active_vllm_request_id or "").strip()
            pre_tts_request_id = str(self.kernel.state.active_tts_request_id or "").strip()
            pre_epoch_id = str(self.kernel.current_lease().epoch_id or "").strip()
            if pre_vllm_request_id:
                pre_vllm_output_version = int(self.kernel.state.request_output_version(pre_vllm_request_id))
            if pre_tts_request_id:
                pre_tts_output_version = int(self.kernel.state.request_output_version(pre_tts_request_id))
            kernel_started_ns = now_ns()
            commands = self.kernel.tick()
            self._record_latency("kernel", kernel_started_ns, now_ns())
            post_vllm_request_id = str(self.kernel.state.active_vllm_request_id or "").strip()
            post_tts_request_id = str(self.kernel.state.active_tts_request_id or "").strip()
            post_epoch_id = str(self.kernel.current_lease().epoch_id or "").strip()

            if pre_vllm_request_id and pre_vllm_request_id != post_vllm_request_id:
                commands = (
                    *commands,
                    build_vllm_cancel_command(
                        session_id=self.kernel.session_id,
                        request_id=pre_vllm_request_id,
                        output_version=pre_vllm_output_version,
                        lineage_id=pre_epoch_id or post_epoch_id,
                        epoch_id=pre_epoch_id or post_epoch_id,
                    ),
                )
            if pre_tts_request_id and pre_tts_request_id != post_tts_request_id:
                commands = (
                    *commands,
                    build_tts_cancel_command(
                        session_id=self.kernel.session_id,
                        request_id=pre_tts_request_id,
                        output_version=pre_tts_output_version,
                        lineage_id=pre_epoch_id or post_epoch_id,
                        epoch_id=pre_epoch_id or post_epoch_id,
                    ),
                )
        stamped_commands: list[DispatchCommand] = []
        for command in commands:
            stamped_payload = dict(command.payload)
            stamped_payload.setdefault("kernel_decision_ns", int(now_ns()))
            self._last_timestamps_ns["kernel_decision_ns"] = int(stamped_payload["kernel_decision_ns"])
            stamped_commands.append(
                DispatchCommand(
                    kind=command.kind,
                    request_id=command.request_id,
                    payload=stamped_payload,
                )
            )
        return tuple(stamped_commands)

    async def _dispatch_commands(
        self,
        commands: tuple[DispatchCommand, ...],
        *,
        blocked_kinds: frozenset[str] | None = None,
    ) -> tuple[tuple[PCMFrame, ...], tuple[DispatchCommand, ...]]:
        queue: deque[DispatchCommand] = deque(commands)
        emitted_frames: list[PCMFrame] = []
        deferred_commands: list[DispatchCommand] = []

        while queue:
            command = queue.popleft()
            if blocked_kinds and command.kind in blocked_kinds:
                deferred_commands.append(command)
                continue
            if command.kind not in {"VLLM", "TTS", "VLLM_CANCEL", "TTS_CANCEL"}:
                raise RuntimeError(f"unsupported_dispatch_command_kind: {command.kind}")
            if command.kind in {"VLLM", "TTS"}:
                self._mirror_dispatch_command_to_ring(command)
            if command.kind == "VLLM":
                command_frames, spawned_commands = await self._execute_vllm_command(command)
            elif command.kind == "TTS":
                command_frames, spawned_commands = await self._execute_tts_command(command)
            elif command.kind == "VLLM_CANCEL":
                command_frames, spawned_commands = await self._execute_vllm_cancel_command(command)
            else:
                command_frames, spawned_commands = await self._execute_tts_cancel_command(command)
            emitted_frames.extend(command_frames)
            queue.extend(spawned_commands)

        return tuple(emitted_frames), tuple(deferred_commands)

    async def run_tick_and_dispatch(self) -> tuple[PCMFrame, ...]:
        commands = await self._tick_and_stamp_commands()
        output_frames, deferred_commands = await self._dispatch_commands(commands)
        if deferred_commands:
            deferred_frames, _ = await self._dispatch_commands(deferred_commands)
            output_frames = tuple((*output_frames, *deferred_frames))
        return output_frames

    async def _execute_vllm_cancel_command(
        self,
        command: DispatchCommand,
    ) -> tuple[tuple[PCMFrame, ...], tuple[DispatchCommand, ...]]:
        payload = dict(command.payload)
        request_id = str(command.request_id or payload.get("request_id", "")).strip()
        if not request_id:
            return (), ()
        try:
            await self.vllm.cancel_request(request_id)
            return (), ()
        except Exception as exc:
            self._append_event(
                self._authority_event(
                    event_type="LLMFaulted",
                    lineage_id=str(payload.get("lineage_id", self.kernel.current_lease().epoch_id)),
                    payload={"request_id": request_id, "error": str(exc), "lane": "vllm"},
                )
            )
            return (), ()

    async def _execute_tts_cancel_command(
        self,
        command: DispatchCommand,
    ) -> tuple[tuple[PCMFrame, ...], tuple[DispatchCommand, ...]]:
        payload = dict(command.payload)
        request_id = str(command.request_id or payload.get("request_id", "")).strip()
        epoch_id = str(payload.get("epoch_id", "")).strip()
        if not request_id and not epoch_id:
            return (), ()
        try:
            self.tts.cancel(request_id=request_id, epoch_id=epoch_id)
            return (), ()
        except Exception as exc:
            self._append_event(
                self._authority_event(
                    event_type="TTSFaulted",
                    lineage_id=str(payload.get("lineage_id", self.kernel.current_lease().epoch_id)),
                    payload={"request_id": request_id, "error": str(exc), "lane": "tts"},
                )
            )
            return (), ()

    async def _execute_vllm_command(
        self,
        command: DispatchCommand,
    ) -> tuple[tuple[PCMFrame, ...], tuple[DispatchCommand, ...]]:
        assert self._vllm_streamer is not None
        payload = dict(command.payload)
        request_id = str(command.request_id)
        request_event_id = self.kernel.state.request_event_id(request_id)
        lineage_id = str(payload.get("lineage_id", self.kernel.current_lease().epoch_id))
        token_frames: list[PCMFrame] = []
        deferred_commands: list[DispatchCommand] = []
        first_token_seen = False
        try:
            async for token in self._vllm_streamer.stream(
                str(payload.get("prompt", "")),
                cache_key=str(payload.get("prompt_cache_key", "")),
                request_id=request_id,
            ):
                if self.kernel.state.active_vllm_request_id != request_id:
                    break
                if int(self.kernel.state.request_output_version(request_id)) != int(payload.get("output_version", -1)):
                    break
                token_ns = now_ns()
                if not first_token_seen:
                    self._record_latency("vllm", int(payload.get("kernel_decision_ns", token_ns) or token_ns), token_ns)
                    first_token_seen = True
                    self._last_timestamps_ns["vllm_first_token_ns"] = int(token_ns)
                self._append_event(
                    self._authority_event(
                        event_type="VLLMChunkReceived",
                        lineage_id=lineage_id,
                        payload={
                            "request_id": request_id,
                            "token": str(token),
                            "output_version": int(payload.get("output_version", 0)),
                            "vllm_first_token_ns": int(token_ns),
                            "lane": "vllm",
                        },
                        causation_id=request_event_id,
                    )
                )
                nested_commands = await self._tick_and_stamp_commands()
                nested_frames, nested_deferred = await self._dispatch_commands(
                    nested_commands,
                    blocked_kinds=frozenset({"VLLM"}),
                )
                token_frames.extend(nested_frames)
                deferred_commands.extend(nested_deferred)
            if self.kernel.state.request_output_version(request_id) == int(payload.get("output_version", -1)):
                self._append_event(
                    self._authority_event(
                        event_type="VLLMCompleted",
                        lineage_id=lineage_id,
                        payload={
                            "request_id": request_id,
                            "text": "".join(self.kernel.state.output.vllm_tokens),
                            "output_version": int(payload.get("output_version", 0)),
                            "lane": "vllm",
                        },
                        causation_id=request_event_id,
                    )
                )
                nested_commands = await self._tick_and_stamp_commands()
                nested_frames, nested_deferred = await self._dispatch_commands(
                    nested_commands,
                    blocked_kinds=frozenset({"VLLM"}),
                )
                token_frames.extend(nested_frames)
                deferred_commands.extend(nested_deferred)
            return tuple(token_frames), tuple(deferred_commands)
        except Exception as exc:
            self._append_event(
                self._authority_event(
                    event_type="LLMFaulted",
                    lineage_id=lineage_id,
                    payload={"request_id": request_id, "error": str(exc), "lane": "vllm"},
                    causation_id=request_event_id,
                )
            )
            nested_commands = await self._tick_and_stamp_commands()
            nested_frames, nested_deferred = await self._dispatch_commands(
                nested_commands,
                blocked_kinds=frozenset({"VLLM"}),
            )
            token_frames.extend(nested_frames)
            deferred_commands.extend(nested_deferred)
            return tuple(token_frames), tuple(deferred_commands)

    async def _execute_tts_command(
        self,
        command: DispatchCommand,
    ) -> tuple[tuple[PCMFrame, ...], tuple[DispatchCommand, ...]]:
        assert self._tts_streamer is not None
        assert self._output_resampler is not None
        payload = dict(command.payload)
        request_id = str(command.request_id)
        request_event_id = self.kernel.state.request_event_id(request_id)
        lineage_id = str(payload.get("lineage_id", self.kernel.current_lease().epoch_id))
        epoch_id = str(payload.get("epoch_id", ""))
        output_version = int(payload.get("output_version", 0))
        stream_fragment = bool(payload.get("stream_fragment", False))
        pcm_frames: list[PCMFrame] = []
        deferred_commands: list[DispatchCommand] = []
        first_pcm_ns = 0
        try:
            async for pcm_chunk, sample_rate, is_final in self._tts_streamer.stream(
                str(payload.get("text", "")),
                epoch_id=epoch_id,
            ):
                if self.kernel.state.active_tts_request_id != request_id:
                    break
                if int(self.kernel.state.request_output_version(request_id)) != int(payload.get("output_version", -1)):
                    break
                observed_ns = now_ns()
                resampled = self._resample_output(bytes(pcm_chunk), int(sample_rate))
                flush_final = bool(is_final) and not stream_fragment
                output_frames = self._chunk_output_pcm(
                    resampled,
                    epoch_id=epoch_id,
                    output_version=output_version,
                    flush=flush_final,
                )
                for frame_pcm in output_frames:
                    if not first_pcm_ns:
                        first_pcm_ns = observed_ns
                        self._record_latency("tts", int(payload.get("kernel_decision_ns", observed_ns) or observed_ns), observed_ns)
                        self._last_timestamps_ns["tts_first_pcm_ns"] = int(first_pcm_ns)
                    frame = PCMFrame(
                        pcm=frame_pcm,
                        sample_rate=int(self.config.output_sample_rate),
                        epoch_id=epoch_id,
                        output_version=output_version,
                    )
                    pcm_frames.append(frame)
                    self._mirror_pcm_frame_to_ring(frame)
                    self.pcm_clock.enqueue(frame)
                    self._append_event(
                        self._authority_event(
                            event_type="TTSChunkReceived",
                            lineage_id=lineage_id,
                            payload={
                                "request_id": request_id,
                                "chunk_id": f"{request_id}:{len(pcm_frames)}",
                                "output_version": output_version,
                                "tts_first_pcm_ns": int(first_pcm_ns),
                                "lane": "tts",
                            },
                            causation_id=request_event_id,
                        )
                    )
                nested_commands = await self._tick_and_stamp_commands()
                nested_frames, nested_deferred = await self._dispatch_commands(
                    nested_commands,
                    blocked_kinds=frozenset({"TTS"}),
                )
                pcm_frames.extend(nested_frames)
                deferred_commands.extend(nested_deferred)
                if is_final:
                    break
            if self.kernel.state.request_output_version(request_id) == int(payload.get("output_version", -1)):
                self._append_event(
                    self._authority_event(
                        event_type="TTSCompleted",
                        lineage_id=lineage_id,
                        payload={
                            "request_id": request_id,
                            "output_version": int(payload.get("output_version", 0)),
                            "lane": "tts",
                        },
                        causation_id=request_event_id,
                    )
                )
                nested_commands = await self._tick_and_stamp_commands()
                nested_frames, nested_deferred = await self._dispatch_commands(
                    nested_commands,
                    blocked_kinds=frozenset({"TTS"}),
                )
                pcm_frames.extend(nested_frames)
                deferred_commands.extend(nested_deferred)
            return tuple(pcm_frames), tuple(deferred_commands)
        except Exception as exc:
            self._append_event(
                self._authority_event(
                    event_type="TTSFaulted",
                    lineage_id=lineage_id,
                    payload={"request_id": request_id, "error": str(exc), "lane": "tts"},
                    causation_id=request_event_id,
                )
            )
            nested_commands = await self._tick_and_stamp_commands()
            nested_frames, nested_deferred = await self._dispatch_commands(
                nested_commands,
                blocked_kinds=frozenset({"TTS"}),
            )
            pcm_frames.extend(nested_frames)
            deferred_commands.extend(nested_deferred)
            return tuple(pcm_frames), tuple(deferred_commands)

    def _resample_output(self, pcm_bytes: bytes, sample_rate: int) -> bytes:
        assert self._output_resampler is not None
        audio = np.frombuffer(bytes(pcm_bytes), dtype=np.int16).astype(np.float32) / 32768.0
        resampled = self._output_resampler.resample(audio, int(sample_rate))
        if resampled.size == 0:
            return b""
        return np.clip(resampled * 32767.0, -32768.0, 32767.0).astype("<i2").tobytes()

    def _output_frame_bytes(self) -> int:
        samples_per_frame = max(1, int(int(self.config.output_sample_rate) * int(self.config.frame_ms) / 1000))
        return samples_per_frame * 2

    def _chunk_output_pcm(
        self,
        pcm_bytes: bytes,
        *,
        epoch_id: str,
        output_version: int,
        flush: bool,
    ) -> tuple[bytes, ...]:
        if str(self._tts_frame_carry_epoch_id) != str(epoch_id) or int(self._tts_frame_carry_output_version) != int(output_version):
            self._tts_frame_carry = b""
            self._tts_frame_carry_epoch_id = str(epoch_id)
            self._tts_frame_carry_output_version = int(output_version)
        frame_bytes = self._output_frame_bytes()
        carry = bytes(self._tts_frame_carry) + bytes(pcm_bytes)
        frames: list[bytes] = []
        while len(carry) >= frame_bytes:
            frames.append(carry[:frame_bytes])
            carry = carry[frame_bytes:]
        if flush and carry:
            padded = carry + (b"\x00" * (frame_bytes - len(carry)))
            frames.append(padded)
            carry = b""
        self._tts_frame_carry = carry
        return tuple(frames)

    async def send_pcm_once(self, send_fn: Callable[[bytes, int], Awaitable[None]]) -> None:
        emit_started_ns = now_ns()
        silence_frame = b"\x00" * self._output_frame_bytes()
        await self.pcm_clock.run_once(
            send_fn,
            current_epoch_id=self.kernel.current_lease().epoch_id,
            current_output_version=int(self.kernel.state.output.version),
            silence_frame=silence_frame,
            silence_sample_rate=int(self.config.output_sample_rate),
        )
        emitted_ns = now_ns()
        self._record_latency("transport", emit_started_ns, emitted_ns)
        self._last_timestamps_ns["transport_emit_ns"] = int(emitted_ns)

    def latency_summary(self) -> dict[str, LatencySummary]:
        return {name: summarize_latency(samples) for name, samples in self._latency_samples.items()}

    def replay_state_hash(self) -> str:
        return canonical_state_hash(self.kernel.state.__dict__ if hasattr(self.kernel.state, "__dict__") else repr(self.kernel.state))

    def replay_event_hash(self) -> str:
        return canonical_event_stream_hash(tuple(self.event_log.as_records()))

    def last_timestamps(self) -> dict[str, int]:
        return dict(self._last_timestamps_ns)

    def recovery_snapshot(self):
        return build_recovery_snapshot(self.kernel.state)

def _bind_cuda_device(device_name: str) -> None:
    resolved = str(device_name or "").strip().lower()
    if not resolved:
        raise RuntimeError("cuda device is not configured")
    if not resolved.startswith("cuda:"):
        raise RuntimeError(f"expected cuda device binding, got {device_name}")
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(f"torch unavailable for CUDA binding {device_name}") from exc
    device_index = int(resolved.split(":", 1)[1])
    torch.cuda.set_device(device_index)


def _assert_contract(runtime_config: RuntimeConfig) -> None:
    if str(runtime_config.asr_device).strip().lower() != "cpu":
        raise RuntimeError("contract_violation: asr device must be cpu")
    if str(runtime_config.llm_device).strip().lower() != "cuda:0":
        raise RuntimeError("contract_violation: vllm device must be cuda:0")
    if str(runtime_config.tts_device).strip().lower() != "cuda:1":
        raise RuntimeError("contract_violation: tts device must be cuda:1")
    if int(runtime_config.frame_ms) != 20:
        raise RuntimeError("contract_violation: frame_ms must be 20")


def _identity_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_model_cache_identity(config: RuntimeConfig) -> dict[str, str]:
    payload = {
        "vosk_model_path": str(config.asr_model_path),
        "vllm_model_path": str(config.resolved_vllm_model_path()),
        "vllm_cache_dir": str(config.vllm_cache_dir),
        "cosyvoice3_model_path": str(config.resolved_cosyvoice3_model_path()),
        "cosyvoice3_cache_dir": str(config.cosyvoice3_cache_dir),
        "cosyvoice3_speaker_path": str(config.cosyvoice3_speaker_path),
    }
    return {
        "vosk": _identity_hash({"path": payload["vosk_model_path"]}),
        "vllm": _identity_hash({"path": payload["vllm_model_path"], "cache_dir": payload["vllm_cache_dir"]}),
        "cosyvoice3": _identity_hash(
            {
                "path": payload["cosyvoice3_model_path"],
                "cache_dir": payload["cosyvoice3_cache_dir"],
                "speaker_path": payload["cosyvoice3_speaker_path"],
            }
        ),
        "model_cache_hash": _identity_hash(payload),
    }


def _build_topology(config: RuntimeConfig) -> RuntimeTopology:
    return RuntimeTopology(
        asr=LaneConfig(name="asr", ring_size=config.asr_ring_size, device=config.asr_device),
        vllm=LaneConfig(name="vllm", ring_size=config.vllm_ring_size, device=config.llm_device),
        tts=LaneConfig(name="tts", ring_size=config.tts_ring_size, device=config.tts_device),
        pcm=LaneConfig(name="pcm", ring_size=config.pcm_ring_size, device="cpu"),
    )


def _build_kernel(config: RuntimeConfig, topology: RuntimeTopology, rings: RingTopology, session_id: str) -> KernelRuntime:
    return KernelRuntime(
        session_id=session_id,
        config=KernelConfig(
            ingress_max_items=int(config.ingress_max_items),
            max_events_per_tick=int(config.max_events_per_tick),
            partial_history_size=int(config.partial_history_size),
            stable_prefix_min_repeats=int(config.stable_prefix_min_repeats),
            stable_prefix_min_tokens=int(config.stable_prefix_min_tokens),
            stable_prefix_max_window=int(config.stable_prefix_max_window),
            tick_interval_ms=int(config.tick_interval_ms),
            tts_fragment_min_tokens=int(config.tts_fragment_min_tokens),
            tts_fragment_max_tokens=int(config.tts_fragment_max_tokens),
            tts_context_window_tokens=int(config.tts_context_window_tokens),
            latency_budget_ms=float(config.latency_budget_ms),
        ),
        topology=topology,
        rings=rings,
    )


def _warm_asr_engine(asr: ASREngine, config: RuntimeConfig, session_id: str) -> bool:
    asr.warm(strict=True)
    asr.start_session(lineage_id=session_id)
    probe_frame = b"\x00\x00" * max(1, int(config.input_sample_rate * config.frame_ms / 1000))
    expected_bytes = max(2, int(config.input_sample_rate * config.frame_ms / 1000) * 2)
    if len(probe_frame) != expected_bytes:
        raise RuntimeError("asr_warmup_probe_invalid_20ms_frame")
    _ = asr.ingest_audio(probe_frame, lineage_id=session_id)
    return bool(asr.is_warm)


def _run_async_probe(coro_factory: Callable[[], Awaitable[bool]]) -> bool:
    result: dict[str, bool] = {"ok": False}
    error: dict[str, Exception] = {}

    def _runner() -> None:
        try:
            result["ok"] = bool(asyncio.run(coro_factory()))
        except Exception as exc:  # pragma: no cover - optional dependency path
            error["exc"] = exc

    thread = Thread(target=_runner, name="voice-pipeline-warm-probe", daemon=True)
    thread.start()
    thread.join()
    if "exc" in error:
        raise RuntimeError("warm_probe_failed") from error["exc"]
    return bool(result["ok"])


def _warm_vllm_engine(vllm: VLLMEngine, config: RuntimeConfig) -> bool:
    _bind_cuda_device(config.llm_device)
    vllm.warm(strict=True)
    vllm.prewarm_prefix_cache(vllm.model_name, "voice_pipeline_system_prefix")
    if not vllm.prefix_cache_ready:
        raise RuntimeError("vllm_prefix_cache_not_ready")

    async def _probe() -> bool:
        async for token in vllm.stream_tokens(
            "ok",
            cache_key="voice_pipeline_system_prefix",
            request_id="warmup-probe-vllm",
            max_tokens=1,
            temperature=0.0,
        ):
            if str(token).strip():
                return True
        return False

    return bool(vllm.is_warm and vllm.prefix_cache_ready and _run_async_probe(_probe))


def _warm_tts_engine(tts: TTSEngine, config: RuntimeConfig, session_id: str) -> bool:
    _bind_cuda_device(config.tts_device)
    tts.warm(strict=True)
    tts.start_persistent_session(
        epoch_id=session_id,
        prompt_speech_path=config.cosyvoice3_speaker_path,
    )

    async def _probe() -> bool:
        async for pcm_chunk, sample_rate, _is_final in tts.stream_pcm(
            "warmup",
            epoch_id=session_id,
        ):
            if pcm_chunk and int(sample_rate) > 0:
                return True
        return False

    return bool(tts.is_warm and _run_async_probe(_probe))


def bootstrap_runtime(*, session_id: str, config: RuntimeConfig | None = None) -> VoicePipelineRuntime:
    runtime_config = config or RuntimeConfig.from_env()
    _assert_contract(runtime_config)
    hardware_admission_check(runtime_config)

    topology = _build_topology(runtime_config)
    rings = RingTopology.with_capacity(
        asr=topology.asr.ring_size,
        vllm=topology.vllm.ring_size,
        tts=topology.tts.ring_size,
        pcm=topology.pcm.ring_size,
        slot_bytes=runtime_config.slot_bytes,
    )

    asr = ASREngine(
        config=ASRRuntimeConfig(
            model_path=runtime_config.asr_model_path,
            sample_rate=int(runtime_config.asr_sample_rate),
            input_sample_rate=int(runtime_config.input_sample_rate),
        )
    )
    vllm = VLLMEngine(
        runtime_config.resolved_vllm_model_path(),
        config=VLLMEngineConfig(
            model_name=runtime_config.resolved_vllm_model_path(),
            model_path=runtime_config.resolved_vllm_model_path(),
            cache_dir=runtime_config.vllm_cache_dir,
            temperature=float(runtime_config.vllm_temperature),
            top_p=float(runtime_config.vllm_top_p),
            max_tokens=int(runtime_config.vllm_max_tokens),
        ),
    )
    tts = TTSEngine(
        runtime_config.resolved_cosyvoice3_model_path(),
        sample_rate=24_000,
        prompt_speech_path=runtime_config.cosyvoice3_speaker_path,
    )

    warm_report = WarmReport()
    worker_status = WorkerStatus()
    worker_failure_reason = WorkerFailureReason()

    worker_status.asr = "WARMING"
    try:
        warm_report.asr_warm = _warm_asr_engine(asr, runtime_config, session_id)
    except Exception as exc:
        warm_report.asr_warm = False
        worker_failure_reason.asr = str(exc)
    worker_status.asr = "READY" if warm_report.asr_warm else "FAILED"

    worker_status.vllm = "WARMING"
    try:
        warm_report.vllm_warm = _warm_vllm_engine(vllm, runtime_config)
        vllm.prewarm_prefix_cache(f"{session_id}:stable_session_scaffold")
    except Exception as exc:
        warm_report.vllm_warm = False
        worker_failure_reason.vllm = str(exc)
    worker_status.vllm = "READY" if warm_report.vllm_warm else "FAILED"

    worker_status.tts = "WARMING"
    try:
        warm_report.tts_warm = _warm_tts_engine(tts, runtime_config, session_id)
    except Exception as exc:
        warm_report.tts_warm = False
        worker_failure_reason.tts = str(exc)
    worker_status.tts = "READY" if warm_report.tts_warm else "FAILED"

    if not (warm_report.asr_warm and warm_report.vllm_warm and warm_report.tts_warm):
        raise RuntimeError(
            "startup_failed: required lane warmup did not reach READY"
            f" asr={worker_failure_reason.asr or worker_status.asr}"
            f" vllm={worker_failure_reason.vllm or worker_status.vllm}"
            f" tts={worker_failure_reason.tts or worker_status.tts}"
        )

    kernel = _build_kernel(runtime_config, topology, rings, session_id)
    worker_status.kernel = "READY"
    transport = LiveKitTransport(
        config=LiveKitTransportConfig(
            url=str(runtime_config.livekit_url),
            api_key=str(runtime_config.livekit_api_key),
            api_secret=str(runtime_config.livekit_api_secret),
            room_name=str(runtime_config.livekit_room_name),
            runtime_identity=str(runtime_config.livekit_runtime_identity),
            output_track_name=str(runtime_config.livekit_output_track_name),
            frame_ms=int(runtime_config.frame_ms),
            input_sample_rate=int(runtime_config.input_sample_rate),
            output_sample_rate=int(runtime_config.output_sample_rate),
            token_ttl_seconds=int(runtime_config.livekit_token_ttl_seconds),
        )
    )
    pcm_clock = PCMClockSender(
        tick_ms=int(runtime_config.frame_ms),
        target_buffer_frames=1,
        max_buffer_frames=2,
    )

    runtime = VoicePipelineRuntime(
        kernel=kernel,
        asr=asr,
        vllm=vllm,
        tts=tts,
        warm_report=warm_report,
        config=runtime_config,
        topology=topology,
        rings=rings,
        transport=transport,
        pcm_clock=pcm_clock,
        worker_status=worker_status,
        worker_failure_reason=worker_failure_reason,
        model_cache_identity=_build_model_cache_identity(runtime_config),
        startup_contract_hash=_identity_hash(
            {
                "asr_device": runtime_config.asr_device,
                "llm_device": runtime_config.llm_device,
                "tts_device": runtime_config.tts_device,
                "frame_ms": runtime_config.frame_ms,
            }
        ),
    )
    return runtime


__all__ = [
    "TopologyReport",
    "VoicePipelineRuntime",
    "WarmReport",
    "WorkerFailureReason",
    "WorkerStatus",
    "bootstrap_runtime",
]
