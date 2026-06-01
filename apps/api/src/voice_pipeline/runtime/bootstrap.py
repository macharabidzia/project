from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
from collections import deque
from contextlib import suppress
import queue
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from threading import Thread
from typing import Any

import numpy as np

from voice_pipeline.bus.ring_topology import RingTopology
from voice_pipeline.bus.ring_types import EventType
from voice_pipeline.gpu.tts_worker.engine import TTSEngine
from voice_pipeline.gpu.tts_worker.stream import TTSAudioStreamer
from voice_pipeline.gpu.vllm_worker.engine import (
    VLLMEngine,
    VLLMEngineConfig,
    build_prompt_cache_key,
)
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

LOGGER = logging.getLogger(__name__)


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


class _BlockingTextStream:
    _SENTINEL = object()

    def __init__(self) -> None:
        self._queue: "queue.Queue[object]" = queue.Queue()

    def push(self, text: str, *, final: bool) -> None:
        normalized = str(text or "").strip()
        if normalized:
            self._queue.put_nowait(normalized)
        if final:
            self._queue.put_nowait(self._SENTINEL)

    def close(self) -> None:
        self._queue.put_nowait(self._SENTINEL)

    def generator(self):
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            yield str(item)


def _single_fragment_generator(text: str):
    normalized = str(text or "").strip()
    if normalized:
        yield normalized


def _default_runtime_timestamps() -> dict[str, int]:
    return {
        "ingress_received_ns": 0,
        "vad_speech_start_ns": 0,
        "first_asr_partial_ns": 0,
        "stable_asr_partial_ns": 0,
        "asr_final_ns": 0,
        "asr_event_ns": 0,
        "kernel_decision_ns": 0,
        "vllm_request_start_ns": 0,
        "vllm_first_token_ns": 0,
        "first_spoken_delta_ns": 0,
        "tts_text_push_ns": 0,
        "tts_native_stream_open_ns": 0,
        "tts_first_pcm_ns": 0,
        "tts_gate_open_ns": 0,
        "resampler_first_output_ns": 0,
        "pcm_enqueue_ns": 0,
        "pcm_send_ns": 0,
        "transport_emit_ns": 0,
        "livekit_egress_ns": 0,
    }


def _default_tts_signal_metrics() -> dict[str, object]:
    return {
        "tts_request_id": "",
        "tts_request_started_ns": 0,
        "tts_backend_path": "",
        "tts_first_pcm_ms": 0.0,
        "tts_pcm_gate_delay_ms": 0.0,
        "tts_leading_trimmed_frames": 0,
        "tts_leading_trimmed_ms": 0.0,
        "tts_leading_gate_open": 0,
        "tts_raw_last_rms": 0.0,
        "tts_raw_last_peak": 0.0,
        "tts_raw_max_rms": 0.0,
        "tts_raw_max_peak": 0.0,
        "tts_raw_last_bytes": 0,
        "tts_resampled_last_rms": 0.0,
        "tts_resampled_last_peak": 0.0,
        "tts_resampled_max_rms": 0.0,
        "tts_resampled_max_peak": 0.0,
        "tts_resampled_last_bytes": 0,
        "tts_chunked_last_rms": 0.0,
        "tts_chunked_last_peak": 0.0,
        "tts_chunked_max_rms": 0.0,
        "tts_chunked_max_peak": 0.0,
        "tts_chunked_last_bytes": 0,
        "tts_chunked_frames": 0,
        "tts_chunk_trace": [],
        "spec_tts_started_before_final_asr": False,
        "spec_tts_cancel_count": 0,
        "spec_tts_promote_count": 0,
    }


def _token_fragment_generator(text: str):
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return
    for token in normalized.split(" "):
        if token:
            yield token


def _push_tokenized_text(stream: _BlockingTextStream, text: str, *, final: bool) -> None:
    tokens = list(_token_fragment_generator(text))
    if not tokens:
        if final:
            stream.close()
        return
    last_index = len(tokens) - 1
    for index, token in enumerate(tokens):
        stream.push(str(token), final=bool(final and index == last_index))


def _join_spoken_tokens(tokens: tuple[str, ...] | list[str]) -> str:
    joined = ""
    for token in tokens:
        resolved = str(token or "")
        if not resolved:
            continue
        if joined and _should_insert_spoken_boundary_space(joined[-1], resolved[0]):
            joined = f"{joined} {resolved}"
        else:
            joined = f"{joined}{resolved}"
    return joined


def _should_insert_spoken_boundary_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left.isspace() or right.isspace():
        return False
    if right in "'’":
        return False
    if left in "-/":
        return False
    if not left.isalnum() or not right.isalnum():
        return False
    return True


@dataclass(slots=True)
class _ActiveTTSStreamSession:
    request_id: str
    request_event_id: str
    lineage_id: str
    epoch_id: str
    output_version: int
    text_stream: _BlockingTextStream | None
    text_generator: object
    task: asyncio.Task[None]
    chunk_counter: int = 0


@dataclass(slots=True)
class _SpeculativeVLLMRequest:
    request_id: str
    source_text: str
    cache_key: str
    rendered_prompt: str
    task: asyncio.Task[None] | None = None
    tokens: list[str] = field(default_factory=list)
    completed: bool = False
    completed_text: str = ""
    error: str = ""


@dataclass(slots=True)
class _PreparedTTSFrame:
    pcm: bytes
    raw_rms: float
    raw_peak: float
    resampled_rms: float
    resampled_peak: float
    chunked_rms: float
    chunked_peak: float
    chunk_index: int


@dataclass(slots=True)
class _SpeculativeTTSRequest:
    request_id: str
    source_text: str
    task: asyncio.Task[None] | None = None
    drain_task: asyncio.Task[None] | None = None
    pending_frames: deque[_PreparedTTSFrame] = field(default_factory=deque)
    completed: bool = False
    error: str = ""
    promoted: bool = False
    live_request_id: str = ""
    live_request_event_id: str = ""
    live_lineage_id: str = ""
    live_epoch_id: str = ""
    live_output_version: int = 0
    live_kernel_decision_ns: int = 0
    first_pcm_ns: int = 0
    completed_event_emitted: bool = False


@dataclass
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
    _last_timestamps_ns: dict[str, int] = field(default_factory=_default_runtime_timestamps)
    _tts_frame_carry: bytes = b""
    _tts_frame_carry_epoch_id: str = ""
    _tts_frame_carry_output_version: int = -1
    _output_resampler_epoch_id: str = ""
    _output_resampler_output_version: int = -1
    _vllm_runtime_probe_complete: bool = False
    _tts_runtime_probe_complete: bool = False
    _active_tts_streams: dict[str, _ActiveTTSStreamSession] = field(default_factory=dict)
    _tts_signal_metrics: dict[str, object] = field(default_factory=_default_tts_signal_metrics)
    _ingress_frame_trace: deque[dict[str, object]] = field(default_factory=lambda: deque(maxlen=32))
    _asr_event_trace: deque[dict[str, object]] = field(default_factory=lambda: deque(maxlen=32))
    _trace_origin_ns: int = 0
    _last_vad_speech_start_ns: int = 0
    _last_suppressed_greeting_extension_text: str = ""
    _last_suppressed_greeting_extension_ns: int = 0
    _last_speculative_vllm_cache_key: str = ""
    _speculative_vllm: _SpeculativeVLLMRequest | None = None
    _speculative_tts: _SpeculativeTTSRequest | None = None

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

    async def reset_session_state(self) -> None:
        await self._cancel_speculative_vllm()
        await self._cancel_speculative_tts()
        active_sessions = tuple(self._active_tts_streams.values())
        self._active_tts_streams.clear()
        for session in active_sessions:
            if session.text_stream is not None:
                with suppress(Exception):
                    session.text_stream.close()
            with suppress(Exception):
                self.tts.cancel(request_id=session.request_id, epoch_id=session.epoch_id)
            if not session.task.done():
                with suppress(Exception):
                    await asyncio.wait_for(session.task, timeout=5.0)
            if not session.task.done():
                session.task.cancel()
                with suppress(Exception):
                    await session.task

        self.pcm_clock.clear()
        self.transport.start_egress_request("")
        self._tts_frame_carry = b""
        self._tts_frame_carry_epoch_id = ""
        self._tts_frame_carry_output_version = -1
        self._output_resampler = StreamingAudioResampler(target_rate=int(self.config.output_sample_rate))
        self._output_resampler_epoch_id = ""
        self._output_resampler_output_version = -1
        self._vllm_runtime_probe_complete = False
        self._tts_runtime_probe_complete = False
        self._last_timestamps_ns = _default_runtime_timestamps()
        self._tts_signal_metrics = _default_tts_signal_metrics()
        self._ingress_frame_trace.clear()
        self._asr_event_trace.clear()
        self._trace_origin_ns = 0
        self._last_vad_speech_start_ns = 0
        self._last_suppressed_greeting_extension_text = ""
        self._last_suppressed_greeting_extension_ns = 0
        self._last_speculative_vllm_cache_key = ""
        self.event_log = EventLog()
        self.kernel.reset()
        self.asr.start_session(lineage_id=self.kernel.session_id)
        self.tts.start_persistent_session(
            epoch_id=self.kernel.session_id,
            prompt_text=self.config.resolved_cosyvoice3_prompt_text(),
            prompt_speech_path=self.config.cosyvoice3_speaker_path,
        )
        await self.warm_vllm_runtime_probe()
        await self.warm_tts_runtime_probe()

    def _stable_session_summary(self) -> str:
        if int(self.config.vllm_session_summary_turns) <= 0:
            return ""
        conversation_history = tuple(self.kernel.state.transcript.conversation_history)
        if not conversation_history:
            return ""
        max_turns = max(1, int(self.config.vllm_session_summary_turns))
        window = conversation_history[-max_turns:]
        return " | ".join(" ".join(str(item).strip().split()) for item in window if str(item).strip())

    async def _cancel_speculative_vllm(self) -> None:
        speculative = self._speculative_vllm
        self._speculative_vllm = None
        if speculative is None:
            return
        with suppress(Exception):
            await self.vllm.cancel_request(speculative.request_id)
        if speculative.task is not None and not speculative.task.done():
            speculative.task.cancel()
        if speculative.task is not None:
            with suppress(asyncio.CancelledError, Exception):
                await speculative.task

    async def _cancel_speculative_tts(self) -> None:
        speculative = self._speculative_tts
        self._speculative_tts = None
        if speculative is None:
            return
        self._tts_signal_metrics["spec_tts_cancel_count"] = int(
            self._tts_signal_metrics.get("spec_tts_cancel_count", 0) or 0
        ) + 1
        with suppress(Exception):
            self.tts.cancel(request_id=speculative.request_id, epoch_id=str(self.kernel.current_lease().epoch_id))
        if speculative.task is not None and not speculative.task.done():
            speculative.task.cancel()
        if speculative.drain_task is not None and not speculative.drain_task.done():
            speculative.drain_task.cancel()
        if speculative.task is not None:
            with suppress(asyncio.CancelledError, Exception):
                await speculative.task
        if speculative.drain_task is not None:
            with suppress(asyncio.CancelledError, Exception):
                await speculative.drain_task

    async def _run_speculative_vllm_request(self, speculative: _SpeculativeVLLMRequest) -> None:
        speculative_tts_started = False
        try:
            async for token in self.vllm.stream_tokens(
                speculative.rendered_prompt,
                cache_key=speculative.cache_key,
                request_id=speculative.request_id,
            ):
                resolved = str(token or "").strip()
                if resolved:
                    speculative.tokens.append(resolved)
                    if not speculative_tts_started and self._speculative_tts is None:
                        token_threshold = max(1, int(self.config.speculative_tts_start_tokens))
                        if len(speculative.tokens) >= token_threshold:
                            speculative_tts_started = True
                            self._start_speculative_tts_for_text(_join_spoken_tokens(tuple(speculative.tokens)))
            speculative.completed_text = _join_spoken_tokens(speculative.tokens)
            speculative.completed = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            speculative.error = str(exc)

    def _start_speculative_tts_for_text(self, text: str) -> None:
        normalized = " ".join(str(text or "").strip().split())
        if not normalized:
            return
        speculative = self._speculative_tts
        if speculative is not None and speculative.source_text == normalized:
            return
        if speculative is not None:
            asyncio.create_task(self._cancel_speculative_tts(), name="cancel-speculative-tts")
        request_id_seed = f"{self.kernel.session_id}|spec-tts|{normalized}"
        request_id = f"{self.kernel.session_id}:spec-tts:{hashlib.sha1(request_id_seed.encode('utf-8')).hexdigest()[:12]}"
        next_speculative = _SpeculativeTTSRequest(
            request_id=request_id,
            source_text=normalized,
        )
        self._tts_signal_metrics["spec_tts_started_before_final_asr"] = True
        next_speculative.task = asyncio.create_task(
            self._run_speculative_tts_request(next_speculative),
            name=f"spec-tts:{normalized[:24]}",
        )
        self._speculative_tts = next_speculative

    async def _run_speculative_tts_request(self, speculative: _SpeculativeTTSRequest) -> None:
        assert self._tts_streamer is not None
        sample_resampler = StreamingAudioResampler(target_rate=int(self.config.output_sample_rate))
        carry = b""
        frame_bytes = self._output_frame_bytes()
        leading_gate_open = False
        chunk_index = 0
        try:
            epoch_id = str(self.kernel.current_lease().epoch_id)
            self.tts.reset(epoch_id=epoch_id)
            async for pcm_chunk, sample_rate, is_final in self._tts_streamer.stream(
                speculative.source_text,
                request_id=speculative.request_id,
                epoch_id=epoch_id,
            ):
                raw_rms, raw_peak = self._pcm_bytes_rms_peak(bytes(pcm_chunk))
                audio = np.frombuffer(bytes(pcm_chunk), dtype=np.int16).astype(np.float32) / 32768.0
                resampled = sample_resampler.resample(audio, int(sample_rate))
                if bool(is_final):
                    tail = sample_resampler.flush()
                    if tail.size:
                        resampled = np.concatenate((resampled, tail))
                resampled_pcm = (
                    np.clip(resampled * 32767.0, -32768.0, 32767.0).astype("<i2").tobytes()
                    if resampled.size
                    else b""
                )
                resampled_rms, resampled_peak = self._pcm_bytes_rms_peak(resampled_pcm)
                carry = bytes(carry) + bytes(resampled_pcm)
                output_frames: list[bytes] = []
                while len(carry) >= frame_bytes:
                    output_frames.append(carry[:frame_bytes])
                    carry = carry[frame_bytes:]
                if bool(is_final) and carry:
                    output_frames.append(carry + (b"\x00" * (frame_bytes - len(carry))))
                    carry = b""
                frame_batch: list[tuple[bytes, float, float]] = []
                for frame_pcm in output_frames:
                    chunked_rms, chunked_peak = self._pcm_bytes_rms_peak(frame_pcm)
                    frame_batch.append((frame_pcm, chunked_rms, chunked_peak))
                batch_start_index = self._tts_leading_batch_start_index(
                    gate_open=leading_gate_open,
                    frame_metrics=[(chunked_rms, chunked_peak) for _, chunked_rms, chunked_peak in frame_batch],
                    is_final=bool(is_final),
                )
                batch_gate_open = batch_start_index is not None
                for frame_index, (frame_pcm, chunked_rms, chunked_peak) in enumerate(frame_batch):
                    if batch_gate_open and not leading_gate_open:
                        should_emit = frame_index >= int(batch_start_index or 0)
                    else:
                        should_emit = self._should_emit_tts_frame(
                            gate_open=batch_gate_open,
                            chunked_rms=chunked_rms,
                            chunked_peak=chunked_peak,
                            is_final=bool(is_final),
                        )
                    if not should_emit:
                        continue
                    leading_gate_open = True
                    batch_gate_open = True
                    chunk_index += 1
                    speculative.pending_frames.append(
                        _PreparedTTSFrame(
                            pcm=bytes(frame_pcm),
                            raw_rms=float(raw_rms),
                            raw_peak=float(raw_peak),
                            resampled_rms=float(resampled_rms),
                            resampled_peak=float(resampled_peak),
                            chunked_rms=float(chunked_rms),
                            chunked_peak=float(chunked_peak),
                            chunk_index=int(chunk_index),
                        )
                    )
                    if speculative.promoted:
                        self._ensure_speculative_tts_drain_task(speculative)
            speculative.completed = True
            if speculative.promoted:
                self._ensure_speculative_tts_drain_task(speculative)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            speculative.error = str(exc)
            if speculative.promoted:
                self._append_event(
                    self._authority_event(
                        event_type="TTSFaulted",
                        lineage_id=str(speculative.live_lineage_id or self.kernel.current_lease().epoch_id),
                        payload={"request_id": speculative.live_request_id, "error": str(exc), "lane": "tts"},
                        causation_id=str(speculative.live_request_event_id or ""),
                    )
                )

    def _ensure_speculative_tts_drain_task(self, speculative: _SpeculativeTTSRequest) -> None:
        if speculative.drain_task is not None and not speculative.drain_task.done():
            return
        speculative.drain_task = asyncio.create_task(
            self._drain_promoted_speculative_tts(speculative),
            name=f"spec-tts-drain:{speculative.live_request_id or speculative.request_id}",
        )

    async def _drain_promoted_speculative_tts(self, speculative: _SpeculativeTTSRequest) -> None:
        while speculative.promoted:
            while speculative.pending_frames and self.pcm_clock.depth < int(self.pcm_clock.max_buffer_frames):
                prepared = speculative.pending_frames.popleft()
                observed_ns = now_ns()
                self._mark_tts_gate_open(observed_ns)
                self._mark_resampler_first_output(observed_ns)
                if not speculative.first_pcm_ns:
                    speculative.first_pcm_ns = int(observed_ns)
                    self._mark_tts_first_pcm(
                        observed_ns=int(observed_ns),
                        decision_ns=int(speculative.live_kernel_decision_ns or observed_ns),
                    )
                frame = PCMFrame(
                    pcm=bytes(prepared.pcm),
                    sample_rate=int(self.config.output_sample_rate),
                    epoch_id=str(speculative.live_epoch_id),
                    output_version=int(speculative.live_output_version),
                    request_id=str(speculative.live_request_id),
                )
                self._mirror_pcm_frame_to_ring(frame)
                self.pcm_clock.enqueue(frame)
                self._mark_pcm_enqueued(observed_ns)
                self._append_event(
                    self._authority_event(
                        event_type="TTSChunkReceived",
                        lineage_id=str(speculative.live_lineage_id),
                        payload={
                            "request_id": str(speculative.live_request_id),
                            "chunk_id": f"{speculative.live_request_id}:{prepared.chunk_index}",
                            "output_version": int(speculative.live_output_version),
                            "tts_first_pcm_ns": int(speculative.first_pcm_ns),
                            "raw_rms": float(prepared.raw_rms),
                            "raw_peak": float(prepared.raw_peak),
                            "resampled_rms": float(prepared.resampled_rms),
                            "resampled_peak": float(prepared.resampled_peak),
                            "chunked_rms": float(prepared.chunked_rms),
                            "chunked_peak": float(prepared.chunked_peak),
                            "lane": "tts",
                        },
                        causation_id=str(speculative.live_request_event_id),
                    )
                )
            if speculative.completed and not speculative.pending_frames:
                if not speculative.completed_event_emitted:
                    speculative.completed_event_emitted = True
                    self._append_event(
                        self._authority_event(
                            event_type="TTSCompleted",
                            lineage_id=str(speculative.live_lineage_id),
                            payload={
                                "request_id": str(speculative.live_request_id),
                                "output_version": int(speculative.live_output_version),
                                "lane": "tts",
                            },
                            causation_id=str(speculative.live_request_event_id),
                        )
                    )
                    nested_commands = await self._tick_and_stamp_commands()
                    await self._dispatch_commands(
                        nested_commands,
                        blocked_kinds=frozenset({"TTS", "TTS_APPEND"}),
                    )
                break
            await asyncio.sleep(max(0.001, float(self.config.tick_interval_ms) / 1000.0))
        if self._speculative_tts is speculative:
            self._speculative_tts = None

    async def _promote_speculative_tts_request(
        self,
        speculative: _SpeculativeTTSRequest,
        *,
        request_id: str,
        request_event_id: str,
        lineage_id: str,
        epoch_id: str,
        output_version: int,
        kernel_decision_ns: int,
    ) -> None:
        speculative.promoted = True
        speculative.live_request_id = str(request_id)
        speculative.live_request_event_id = str(request_event_id)
        speculative.live_lineage_id = str(lineage_id)
        speculative.live_epoch_id = str(epoch_id)
        speculative.live_output_version = int(output_version)
        speculative.live_kernel_decision_ns = int(kernel_decision_ns)
        self._start_tts_request_metrics(
            request_id=str(request_id),
            started_ns=int(kernel_decision_ns or now_ns()),
        )
        self._tts_signal_metrics["spec_tts_started_before_final_asr"] = True
        self._tts_signal_metrics["spec_tts_promote_count"] = int(
            self._tts_signal_metrics.get("spec_tts_promote_count", 0) or 0
        ) + 1
        self._mark_tts_text_push(int(kernel_decision_ns or now_ns()))
        self._mark_tts_native_stream_open(int(kernel_decision_ns or now_ns()))
        self.transport.start_egress_request(str(request_id))
        self._ensure_speculative_tts_drain_task(speculative)

    def _maybe_prewarm_vllm_stable_prefix(self) -> None:
        if not bool(self.config.livekit_use_turn_detector):
            return
        if str(self.kernel.state.active_vllm_request_id or "").strip():
            return
        transcript = self.kernel.state.transcript
        stable_prefix = self._speculative_vllm_source_text()
        if not stable_prefix:
            if self._speculative_vllm is not None:
                asyncio.create_task(self._cancel_speculative_vllm(), name="cancel-speculative-vllm")
            if self._speculative_tts is not None:
                asyncio.create_task(self._cancel_speculative_tts(), name="cancel-speculative-tts")
            return
        self._mark_stable_asr_partial(int(self._last_timestamps_ns.get("asr_event_ns", 0) or now_ns()))
        committed_text = " ".join(str(transcript.committed_text or "").strip().split())
        if committed_text and stable_prefix == committed_text:
            if self._speculative_vllm is not None:
                asyncio.create_task(self._cancel_speculative_vllm(), name="cancel-speculative-vllm")
            if self._speculative_tts is not None:
                asyncio.create_task(self._cancel_speculative_tts(), name="cancel-speculative-tts")
            return
        stable_session_summary = self._stable_session_summary()
        cache_key = build_prompt_cache_key(
            system_prompt=str(self.config.vllm_system_prompt),
            context_prefix=stable_session_summary,
            stable_prefix=stable_prefix,
        )
        speculative = self._speculative_vllm
        if speculative is not None and speculative.source_text == stable_prefix:
            return
        # Speculative preparation still stays off-authority. We render the
        # likely prompt and prewarm the prefix cache while LiveKit still owns
        # the real turn close, then start a hidden speculative vLLM request.
        # Any speculative TTS work remains promote-only and never commits
        # semantic state outside the kernel path.
        rendered_prompt = self.vllm.render_prompt(
            user_text=stable_prefix,
            stable_session_summary=stable_session_summary,
        )
        self.vllm.prewarm_prefix_cache(cache_key)
        self._last_speculative_vllm_cache_key = str(cache_key)
        if speculative is not None:
            asyncio.create_task(self._cancel_speculative_vllm(), name="cancel-speculative-vllm")
        speculative_tts = self._speculative_tts
        if speculative_tts is not None and speculative_tts.source_text != stable_prefix:
            asyncio.create_task(self._cancel_speculative_tts(), name="cancel-speculative-tts")
        request_id_seed = f"{self.kernel.session_id}|{stable_prefix}|{stable_session_summary}"
        request_id = f"{self.kernel.session_id}:spec-vllm:{hashlib.sha1(request_id_seed.encode('utf-8')).hexdigest()[:12]}"
        self._speculative_vllm = _SpeculativeVLLMRequest(
            request_id=request_id,
            source_text=stable_prefix,
            cache_key=cache_key,
            rendered_prompt=rendered_prompt,
        )
        speculative_request = self._speculative_vllm
        assert speculative_request is not None
        task = asyncio.create_task(
            self._run_speculative_vllm_request(speculative_request),
            name=f"spec-vllm:{stable_prefix[:24]}",
        )
        speculative_request.task = task

    def _speculative_vllm_source_text(self) -> str:
        transcript = self.kernel.state.transcript
        stable_prefix = " ".join(str(transcript.stable_prefix or "").strip().split())
        if stable_prefix:
            return stable_prefix
        partial_text = " ".join(str(transcript.partial_text or "").strip().split())
        if not partial_text:
            return ""
        committed_text = " ".join(str(transcript.committed_text or "").strip().split())
        if committed_text and partial_text == committed_text:
            return ""
        if committed_text and not partial_text.startswith(f"{committed_text} "):
            return ""
        min_partial_tokens = max(
            1,
            int(self.config.speculative_partial_min_tokens),
        )
        if _token_count(partial_text) < min_partial_tokens:
            return ""
        return partial_text

    async def warm_vllm_runtime_probe(self) -> str:
        if self._vllm_runtime_probe_complete:
            return "already_warm"
        async def _probe() -> str:
            warm_user_text = "hello"
            cache_key = build_prompt_cache_key(
                system_prompt=str(self.config.vllm_system_prompt),
                context_prefix="",
                stable_prefix=warm_user_text,
            )
            self.vllm.prewarm_prefix_cache(cache_key)
            prompt = self.vllm.render_prompt(user_text=warm_user_text, stable_session_summary="")
            prompt_token_count = self.vllm.prompt_token_count(prompt)
            if prompt_token_count is not None and prompt_token_count >= int(self.config.vllm_max_model_len):
                # Keep low-memory live profiles warmable without weakening the
                # real runtime prompt scaffold used for authoritative turns.
                prompt = self.vllm.render_minimal_probe_prompt(user_text=warm_user_text)
            cache_key = build_prompt_cache_key(
                system_prompt=str(self.config.vllm_system_prompt),
                context_prefix="",
                stable_prefix=warm_user_text,
            )
            first_token = ""
            async for token in self.vllm.stream_tokens(
                prompt,
                cache_key=cache_key,
                request_id=f"{self.kernel.session_id}:runtime-warmup",
                max_tokens=1,
                temperature=0.0,
            ):
                first_token = str(token or "")
                if first_token.strip():
                    break
            return first_token or "ok"

        first_token = await asyncio.wait_for(_probe(), timeout=30.0)
        # Some ultra-short greedy probes on this constrained profile complete
        # without yielding a visible non-whitespace token. That still exercises
        # the live generation path, so treat it as a successful warm pass.
        self._vllm_runtime_probe_complete = True
        return first_token

    async def _warm_tts_generator_runtime_probe(
        self,
        *,
        warm_request_id: str,
        probe_text: str = "hi there",
    ) -> str:
        if self._tts_runtime_probe_complete:
            return "already_warm"
        async def _probe() -> None:
            # Fully drain one generator-shaped warmup request after the runtime is
            # already READY. This primes the real CosyVoice native bi-stream path
            # without relying on the safer reset-time string probe.
            self.tts.start_persistent_session(
                epoch_id=self.kernel.session_id,
                prompt_text=self.config.resolved_cosyvoice3_prompt_text(),
                prompt_speech_path=self.config.cosyvoice3_speaker_path,
            )
            async for _pcm_chunk, _sample_rate, _is_final in self.tts.stream_pcm(
                _single_fragment_generator(str(probe_text)),
                request_id=warm_request_id,
                epoch_id=self.kernel.session_id,
            ):
                pass

        try:
            await asyncio.wait_for(_probe(), timeout=10.0)
        except asyncio.TimeoutError:
            LOGGER.warning("tts runtime warm probe timed out; recovering with fresh persistent session")
            self.tts.cancel(request_id=warm_request_id, epoch_id=self.kernel.session_id)
            await asyncio.sleep(0)
            self.tts.start_persistent_session(
                epoch_id=self.kernel.session_id,
                prompt_text=self.config.resolved_cosyvoice3_prompt_text(),
                prompt_speech_path=self.config.cosyvoice3_speaker_path,
            )
            return "timeout_recovered"
        # Rebind a fresh persistent session after the warm probe so the next
        # live request starts from a clean prompt-conditioned state instead of
        # inheriting any decoder tail from the probe utterance.
        self.tts.start_persistent_session(
            epoch_id=self.kernel.session_id,
            prompt_text=self.config.resolved_cosyvoice3_prompt_text(),
            prompt_speech_path=self.config.cosyvoice3_speaker_path,
        )
        self._tts_runtime_probe_complete = True
        return "ok"

    async def warm_tts_runtime_probe(self) -> str:
        if self._tts_runtime_probe_complete:
            return "already_warm"
        warm_request_id = f"{self.kernel.session_id}:tts-runtime-warmup"

        async def _safe_probe() -> bool:
            # Use bounded plain-string probes instead of the generator-shaped
            # path that previously crashed during reset. We only need to step
            # beyond the very first native onset blip; deeper drains keep
            # timing out and turn reset warmup itself into a latency problem.
            self.tts.start_persistent_session(
                epoch_id=self.kernel.session_id,
                prompt_text=self.config.resolved_cosyvoice3_prompt_text(),
                prompt_speech_path=self.config.cosyvoice3_speaker_path,
            )

            async def _probe_text(text: str, *, target_non_empty_chunks: int) -> int:
                non_empty_chunks = 0
                request_id = f"{warm_request_id}:{text.replace(' ', '_')}"
                async for pcm_chunk, sample_rate, _is_final in self.tts.stream_pcm(
                    text,
                    request_id=request_id,
                    epoch_id=self.kernel.session_id,
                ):
                    if pcm_chunk and int(sample_rate) > 0:
                        non_empty_chunks += 1
                        if non_empty_chunks >= int(target_non_empty_chunks):
                            self.tts.cancel(request_id=request_id, epoch_id=self.kernel.session_id)
                            await asyncio.sleep(0)
                            break
                return non_empty_chunks

            first_chunks = await _probe_text("hi there", target_non_empty_chunks=2)
            second_chunks = await _probe_text("hello", target_non_empty_chunks=2)
            return bool(first_chunks or second_chunks)

        if bool(self.config.tts_skip_stream_probe):
            try:
                # Two short real-reply probes need a larger bound than the old
                # single-probe reset path; otherwise the intended native warmup
                # is cut off before the second stronger burst completes.
                await asyncio.wait_for(_safe_probe(), timeout=12.0)
            except asyncio.TimeoutError:
                LOGGER.warning("tts runtime safe probe timed out; recovering with fresh persistent session")
                self.tts.cancel(request_id=warm_request_id, epoch_id=self.kernel.session_id)
                await asyncio.sleep(0)
            finally:
                self.tts.start_persistent_session(
                    epoch_id=self.kernel.session_id,
                    prompt_text=self.config.resolved_cosyvoice3_prompt_text(),
                    prompt_speech_path=self.config.cosyvoice3_speaker_path,
                )
            self._tts_runtime_probe_complete = True
            return "safe_probe"
        return await self._warm_tts_generator_runtime_probe(
            warm_request_id=warm_request_id,
            probe_text="hi there",
        )

    async def start(self) -> None:
        if self._tick_task is not None and not self._tick_task.done():
            return
        self._run_event.set()
        self._tick_task = asyncio.create_task(self._tick_loop(), name="voice-pipeline-kernel-tick")

    async def stop(self) -> None:
        self._run_event.clear()
        if self._tick_task is None:
            pass
        else:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            finally:
                self._tick_task = None
        _shutdown_lane_backend("tts", self.tts)
        _shutdown_lane_backend("vllm", self.vllm)
        _shutdown_lane_backend("asr", self.asr)

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

    @staticmethod
    def _pcm_bytes_rms_peak(pcm_bytes: bytes) -> tuple[float, float]:
        if not pcm_bytes:
            return 0.0, 0.0
        samples = np.frombuffer(bytes(pcm_bytes), dtype="<i2").astype(np.float32)
        if samples.size == 0:
            return 0.0, 0.0
        normalized = samples / 32768.0
        rms = float(np.sqrt(np.mean(np.square(normalized))))
        peak = float(np.max(np.abs(normalized)))
        return rms, peak

    def _update_tts_signal_metrics(self, *, stage: str, pcm_bytes: bytes) -> tuple[float, float]:
        rms, peak = self._pcm_bytes_rms_peak(pcm_bytes)
        metrics = self._tts_signal_metrics
        last_rms_key = f"tts_{stage}_last_rms"
        last_peak_key = f"tts_{stage}_last_peak"
        max_rms_key = f"tts_{stage}_max_rms"
        max_peak_key = f"tts_{stage}_max_peak"
        last_bytes_key = f"tts_{stage}_last_bytes"
        metrics[last_rms_key] = float(rms)
        metrics[last_peak_key] = float(peak)
        metrics[last_bytes_key] = int(len(pcm_bytes))
        metrics[max_rms_key] = max(float(metrics.get(max_rms_key, 0.0) or 0.0), float(rms))
        metrics[max_peak_key] = max(float(metrics.get(max_peak_key, 0.0) or 0.0), float(peak))
        if stage == "chunked":
            metrics["tts_chunked_frames"] = int(metrics.get("tts_chunked_frames", 0) or 0) + 1
        return rms, peak

    def _start_tts_request_metrics(self, *, request_id: str, started_ns: int) -> None:
        metrics = self._tts_signal_metrics
        metrics["tts_request_id"] = str(request_id or "")
        metrics["tts_request_started_ns"] = int(started_ns)
        metrics["tts_backend_path"] = ""
        metrics["tts_first_pcm_ms"] = 0.0
        metrics["tts_pcm_gate_delay_ms"] = 0.0
        metrics["spec_tts_started_before_final_asr"] = False
        self._last_timestamps_ns["tts_text_push_ns"] = 0
        self._last_timestamps_ns["tts_native_stream_open_ns"] = 0
        self._last_timestamps_ns["tts_first_pcm_ns"] = 0
        self._last_timestamps_ns["tts_gate_open_ns"] = 0
        self._last_timestamps_ns["resampler_first_output_ns"] = 0
        self._last_timestamps_ns["pcm_enqueue_ns"] = 0
        self._last_timestamps_ns["pcm_send_ns"] = 0
        self._last_timestamps_ns["transport_emit_ns"] = 0
        self._last_timestamps_ns["livekit_egress_ns"] = 0
        metrics["tts_leading_trimmed_frames"] = 0
        metrics["tts_leading_trimmed_ms"] = 0.0
        metrics["tts_leading_gate_open"] = 0
        metrics["tts_chunk_trace"] = []
        for key in (
            "tts_raw_last_rms",
            "tts_raw_last_peak",
            "tts_raw_max_rms",
            "tts_raw_max_peak",
            "tts_raw_last_bytes",
            "tts_resampled_last_rms",
            "tts_resampled_last_peak",
            "tts_resampled_max_rms",
            "tts_resampled_max_peak",
            "tts_resampled_last_bytes",
            "tts_chunked_last_rms",
            "tts_chunked_last_peak",
            "tts_chunked_max_rms",
            "tts_chunked_max_peak",
            "tts_chunked_last_bytes",
            "tts_chunked_frames",
        ):
            metrics[key] = 0.0 if "rms" in key or "peak" in key else 0

    def _mark_timestamp_once(self, key: str, observed_ns: int) -> None:
        if not hasattr(self, "_last_timestamps_ns"):
            self._last_timestamps_ns = _default_runtime_timestamps()
        if int(self._last_timestamps_ns.get(key, 0) or 0) <= 0:
            self._last_timestamps_ns[str(key)] = int(observed_ns)

    def _mark_vad_speech_start(self, observed_ns: int) -> None:
        self._mark_timestamp_once("vad_speech_start_ns", observed_ns)

    def _mark_first_asr_partial(self, observed_ns: int) -> None:
        self._mark_timestamp_once("first_asr_partial_ns", observed_ns)

    def _mark_stable_asr_partial(self, observed_ns: int) -> None:
        self._mark_timestamp_once("stable_asr_partial_ns", observed_ns)

    def _mark_asr_final(self, observed_ns: int) -> None:
        self._mark_timestamp_once("asr_final_ns", observed_ns)

    def _mark_vllm_request_start(self, observed_ns: int) -> None:
        self._mark_timestamp_once("vllm_request_start_ns", observed_ns)

    def _mark_first_spoken_delta(self, observed_ns: int) -> None:
        self._mark_timestamp_once("first_spoken_delta_ns", observed_ns)

    def _mark_tts_text_push(self, observed_ns: int) -> None:
        self._mark_timestamp_once("tts_text_push_ns", observed_ns)

    def _mark_tts_native_stream_open(self, observed_ns: int) -> None:
        self._mark_timestamp_once("tts_native_stream_open_ns", observed_ns)
        self._tts_signal_metrics["tts_backend_path"] = str(self.tts.debug_metrics().get("last_backend_path", "") or "")

    def _mark_tts_gate_open(self, observed_ns: int) -> None:
        self._mark_timestamp_once("tts_gate_open_ns", observed_ns)

    def _mark_tts_first_pcm(self, *, observed_ns: int, decision_ns: int) -> None:
        if int(self._last_timestamps_ns.get("tts_first_pcm_ns", 0) or 0) > 0:
            return
        self._record_latency("tts", int(decision_ns or observed_ns), int(observed_ns))
        self._last_timestamps_ns["tts_first_pcm_ns"] = int(observed_ns)
        if not str(self._tts_signal_metrics.get("tts_backend_path", "") or "").strip():
            self._tts_signal_metrics["tts_backend_path"] = str(self.tts.debug_metrics().get("last_backend_path", "") or "")
        request_started_ns = int(self._tts_signal_metrics.get("tts_request_started_ns", 0) or 0)
        gate_open_ns = int(self._last_timestamps_ns.get("tts_gate_open_ns", 0) or 0)
        if request_started_ns > 0 and int(observed_ns) >= request_started_ns:
            self._tts_signal_metrics["tts_first_pcm_ms"] = float(int(observed_ns) - request_started_ns) / 1_000_000.0
        if gate_open_ns > 0 and int(observed_ns) >= gate_open_ns:
            self._tts_signal_metrics["tts_pcm_gate_delay_ms"] = float(int(observed_ns) - gate_open_ns) / 1_000_000.0

    def _mark_resampler_first_output(self, observed_ns: int) -> None:
        self._mark_timestamp_once("resampler_first_output_ns", observed_ns)

    def _mark_pcm_enqueued(self, observed_ns: int) -> None:
        self._mark_timestamp_once("pcm_enqueue_ns", observed_ns)

    def _mark_pcm_send(self, observed_ns: int) -> None:
        self._mark_timestamp_once("pcm_send_ns", observed_ns)

    def mark_livekit_egress(self, observed_ns: int) -> None:
        self._mark_timestamp_once("livekit_egress_ns", observed_ns)

    def _record_tts_chunk_trace(
        self,
        *,
        observed_ns: int,
        raw_rms: float,
        raw_peak: float,
        resampled_rms: float,
        resampled_peak: float,
        chunked_rms: float,
        chunked_peak: float,
        emitted: bool,
        is_final: bool,
    ) -> None:
        metrics = self._tts_signal_metrics
        chunk_trace = metrics.setdefault("tts_chunk_trace", [])
        if not isinstance(chunk_trace, list):
            chunk_trace = []
            metrics["tts_chunk_trace"] = chunk_trace
        if len(chunk_trace) >= 24:
            return
        started_ns = int(metrics.get("tts_request_started_ns", 0) or 0)
        relative_ms = 0.0
        if started_ns > 0:
            relative_ms = float(max(0, int(observed_ns) - started_ns)) / 1_000_000.0
        chunk_trace.append(
            {
                "index": len(chunk_trace) + 1,
                "relative_ms": relative_ms,
                "raw_rms": float(raw_rms),
                "raw_peak": float(raw_peak),
                "resampled_rms": float(resampled_rms),
                "resampled_peak": float(resampled_peak),
                "chunked_rms": float(chunked_rms),
                "chunked_peak": float(chunked_peak),
                "emitted": bool(emitted),
                "is_final": bool(is_final),
            }
        )

    def _trace_relative_ms(self, timestamp_ns: int) -> float:
        if int(self._trace_origin_ns) <= 0:
            self._trace_origin_ns = int(timestamp_ns)
        return float(max(0, int(timestamp_ns) - int(self._trace_origin_ns))) / 1_000_000.0

    def _record_ingress_frame_trace(self, *, observed_ns: int, pcm_bytes: bytes) -> None:
        rms, peak = self._pcm_bytes_rms_peak(pcm_bytes)
        self._ingress_frame_trace.append(
            {
                "index": len(self._ingress_frame_trace) + 1,
                "relative_ms": self._trace_relative_ms(int(observed_ns)),
                "bytes": int(len(pcm_bytes)),
                "rms": float(rms),
                "peak": float(peak),
            }
        )

    def _record_asr_event_trace(self, event: ASREvent) -> None:
        self._asr_event_trace.append(
            {
                "index": len(self._asr_event_trace) + 1,
                "relative_ms": self._trace_relative_ms(int(event.emitted_at_ns)),
                "type": str(event.event_type),
                "text": str(event.text),
                "lineage_id": str(event.lineage_id),
                "timestamp_ns": int(event.emitted_at_ns),
            }
        )

    def note_vad_speech_start(self, observed_ns: int) -> None:
        self._last_vad_speech_start_ns = max(0, int(observed_ns))
        self._mark_vad_speech_start(int(observed_ns))
        self._last_suppressed_greeting_extension_text = ""
        self._last_suppressed_greeting_extension_ns = 0

    @staticmethod
    def _completed_greeting_tail_suffix(
        *,
        completed_text: str,
        extension_text: str,
        candidate_text: str,
    ) -> bool:
        completed_tokens = tuple(str(completed_text or "").split())
        extension_tokens = tuple(str(extension_text or "").split())
        candidate_tokens = tuple(str(candidate_text or "").split())
        if not completed_tokens or not extension_tokens or not candidate_tokens:
            return False
        if len(extension_tokens) <= len(completed_tokens):
            return False
        if tuple(extension_tokens[: len(completed_tokens)]) != completed_tokens:
            return False
        tail_tokens = extension_tokens[len(completed_tokens) :]
        if len(candidate_tokens) > len(tail_tokens):
            return False
        if tuple(tail_tokens[-len(candidate_tokens) :]) == candidate_tokens:
            return True
        if len(candidate_tokens) != 1:
            return False
        candidate = candidate_tokens[0]
        if len(candidate) < 2:
            return False
        return any(str(tail_token).startswith(candidate) for tail_token in tail_tokens)

    def _should_suppress_stale_greeting_asr_extension(self, event: ASREvent) -> bool:
        event_type = str(event.event_type or "").strip()
        if event_type not in {"ASRPartialReceived", "ASRFinalReceived"}:
            return False
        event_text = " ".join(str(event.text or "").strip().split())
        if not event_text:
            return False
        completed = " ".join(
            str(getattr(self.kernel, "_diagnostics").last_completed_committed_text or "").strip().split()
        )
        if not completed:
            return False
        completed_tokens = tuple(completed.split())
        if len(completed_tokens) != 1 or completed_tokens[0] not in {"hello", "hi", "hey", "howdy"}:
            return False
        if str(self.kernel.state.active_vllm_request_id or "").strip():
            return False
        if str(self.kernel.state.active_tts_request_id or "").strip():
            return False
        diagnostics = getattr(self.kernel, "_diagnostics")
        last_turn_completed_ns = int(getattr(diagnostics, "last_turn_completed_ns", 0) or 0)
        if last_turn_completed_ns <= 0:
            return False
        if int(event.emitted_at_ns) - int(last_turn_completed_ns) > 6_000_000_000:
            return False
        if int(self._last_vad_speech_start_ns) > int(last_turn_completed_ns):
            return False
        suppressed_extension = str(self._last_suppressed_greeting_extension_text or "").strip()
        suppressed_extension_ns = int(self._last_suppressed_greeting_extension_ns or 0)
        if suppressed_extension and suppressed_extension_ns > 0:
            if int(event.emitted_at_ns) >= suppressed_extension_ns and self._completed_greeting_tail_suffix(
                completed_text=completed,
                extension_text=suppressed_extension,
                candidate_text=event_text,
            ):
                return True
        if event_text == completed or not event_text.startswith(f"{completed} "):
            return False
        self._last_suppressed_greeting_extension_text = event_text
        self._last_suppressed_greeting_extension_ns = int(event.emitted_at_ns)
        return True

    def _should_emit_tts_frame(
        self,
        *,
        gate_open: bool,
        chunked_rms: float,
        chunked_peak: float,
        is_final: bool,
    ) -> bool:
        if gate_open:
            return True
        if bool(is_final):
            return True
        return bool(
            float(chunked_rms) >= float(self.config.tts_leading_silence_rms_threshold)
            or float(chunked_peak) >= float(self.config.tts_leading_silence_peak_threshold)
        )

    def _record_trimmed_tts_frame(self) -> None:
        metrics = self._tts_signal_metrics
        trimmed_frames = int(metrics.get("tts_leading_trimmed_frames", 0) or 0) + 1
        metrics["tts_leading_trimmed_frames"] = trimmed_frames
        metrics["tts_leading_trimmed_ms"] = float(trimmed_frames * int(self.config.frame_ms))

    def _batch_opens_tts_leading_gate(
        self,
        *,
        gate_open: bool,
        frame_metrics: Sequence[tuple[float, float]],
        is_final: bool,
    ) -> bool:
        if gate_open or bool(is_final):
            return True
        rms_threshold = float(self.config.tts_leading_silence_rms_threshold)
        peak_threshold = float(self.config.tts_leading_silence_peak_threshold)
        if any(
            float(chunked_rms) >= rms_threshold or float(chunked_peak) >= peak_threshold
            for chunked_rms, chunked_peak in frame_metrics
        ):
            return True

        # Preserve a short rising onset burst when the whole first batch is
        # clearly building toward speech, even if no single frame crosses the
        # normal lead-in threshold on its own.
        if len(frame_metrics) < 4:
            return False
        rms_values = [float(chunked_rms) for chunked_rms, _ in frame_metrics]
        peak_values = [float(chunked_peak) for _, chunked_peak in frame_metrics]
        cumulative_rms = float(sum(rms_values))
        final_rms = float(rms_values[-1]) if rms_values else 0.0
        max_peak = float(max(peak_values)) if peak_values else 0.0
        rising_steps = sum(1 for left, right in zip(rms_values, rms_values[1:]) if right >= left)
        return bool(
            cumulative_rms >= (rms_threshold * 1.75)
            and final_rms >= (rms_threshold * 0.65)
            and max_peak >= (peak_threshold * 0.45)
            and rising_steps >= max(3, len(rms_values) - 2)
        )

    def _tts_leading_batch_start_index(
        self,
        *,
        gate_open: bool,
        frame_metrics: Sequence[tuple[float, float]],
        is_final: bool,
    ) -> int | None:
        if not frame_metrics:
            return None
        if gate_open or bool(is_final):
            return 0
        rms_threshold = float(self.config.tts_leading_silence_rms_threshold)
        peak_threshold = float(self.config.tts_leading_silence_peak_threshold)
        threshold_crossings = [
            idx
            for idx, (chunked_rms, chunked_peak) in enumerate(frame_metrics)
            if float(chunked_rms) >= rms_threshold or float(chunked_peak) >= peak_threshold
        ]
        if threshold_crossings:
            # If the strongest qualifying frame lands at the tail of the first
            # batch, keep a short rising onset attached to it. Returning only
            # the final subframe turns a measured 60 ms native onset into a
            # 20 ms blip, which then leaves the room capture waiting for the
            # next native burst to hear real speech.
            best_index, _ = max(
                enumerate(frame_metrics),
                key=lambda item: (
                    float(item[1][1]),  # peak first
                    float(item[1][0]),  # then rms
                    int(item[0]),       # prefer the later frame on ties
                ),
            )
            tail_distance = int(len(frame_metrics) - 1 - int(best_index))
            if tail_distance <= 4 and len(frame_metrics) >= 10:
                supportive_subframes = 4
                return max(0, int(best_index) - supportive_subframes)
            if tail_distance <= 2:
                if len(frame_metrics) <= 5:
                    if threshold_crossings and int(threshold_crossings[0]) == 0:
                        return max(0, int(best_index) - 2)
                    return int(best_index)
                if len(frame_metrics) <= 4 and int(best_index) > 0:
                    first_rms, first_peak = frame_metrics[0]
                    best_rms, best_peak = frame_metrics[int(best_index)]
                    first_is_weak = bool(
                        float(first_rms) < (rms_threshold * 0.85)
                        and float(first_peak) < (peak_threshold * 0.7)
                    )
                    best_is_strong = bool(
                        float(best_rms) >= (rms_threshold * 1.5)
                        or float(best_peak) >= (peak_threshold * 1.5)
                    )
                    if first_is_weak and best_is_strong:
                        # Very short first batches can already contain the
                        # real speech crest in frame 2. In that shape, keeping
                        # the weak pre-onset head frame only delays audibility
                        # without preserving useful onset context.
                        if int(best_index) == 1:
                            return 1
                        return max(0, int(best_index) - 1)
                # The real room trace is more important than a narrow local
                # heuristic here: when native CosyVoice stacks the strongest
                # onset into the tail of the first batch, keeping only the
                # very end produces a weak blip and delays intelligible speech
                # until the later native burst. Keep a short rising onset
                # attached to that crest. Very long first batches need one
                # more supportive subframe because the strongest crest may land
                # slightly before the final frame, followed by a weak tail.
                if len(frame_metrics) >= 10:
                    supportive_subframes = 4
                elif len(frame_metrics) >= 6:
                    supportive_subframes = 3
                else:
                    supportive_subframes = 2
                return max(0, int(best_index) - supportive_subframes)
            # For ordinary batches, prefer the strongest tail among already
            # qualified subframes instead of leaking an earlier threshold-edge
            # frame into playout.
            return int(threshold_crossings[-1])
        if not self._batch_opens_tts_leading_gate(
            gate_open=gate_open,
            frame_metrics=frame_metrics,
            is_final=is_final,
        ):
            return None
        # If the batch only qualifies through the rising-onset fallback, start
        # at the strongest frame in that batch instead of the earliest frame
        # near the floor. This avoids leaking a weak pre-onset frame when the
        # actual speech crest is already present later in the same batch.
        best_index, _ = max(
            enumerate(frame_metrics),
            key=lambda item: (
                float(item[1][1]),  # peak first
                float(item[1][0]),  # then rms
                int(item[0]),       # prefer the later frame on ties
            ),
        )
        return int(best_index)

    def _should_drop_final_tts_resampler_tail(
        self,
        *,
        last_emitted_ns: int,
        observed_ns: int,
        raw_pcm: bytes,
        raw_rms: float,
        raw_peak: float,
        resampled_pcm: bytes,
        resampled_rms: float,
        resampled_peak: float,
        is_final: bool,
    ) -> bool:
        if not bool(is_final):
            return False
        if not resampled_pcm:
            return False
        if raw_pcm and (float(raw_rms) > 0.0 or float(raw_peak) > 0.0):
            return False
        if int(last_emitted_ns) <= 0:
            return False
        gap_ms = float(max(0, int(observed_ns) - int(last_emitted_ns))) / 1_000_000.0
        if gap_ms < max(250.0, float(int(self.config.frame_ms) * 8)):
            return False
        return bool(float(resampled_rms) <= 0.04 and float(resampled_peak) <= 0.12)

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
            if self._should_suppress_stale_greeting_asr_extension(event):
                continue
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

    def _filter_stale_asr_events(self, events: tuple[ASREvent, ...]) -> tuple[ASREvent, ...]:
        kept_events: list[ASREvent] = []
        for event in events:
            if self._should_suppress_stale_greeting_asr_extension(event):
                continue
            kept_events.append(event)
        return tuple(kept_events)

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
        self._record_ingress_frame_trace(observed_ns=int(ingress_received_ns), pcm_bytes=bytes(pcm))
        asr_started_ns = now_ns()
        asr_events = self.asr.ingest_audio(bytes(pcm), lineage_id=self.kernel.current_lease().epoch_id)
        asr_events = self._filter_stale_asr_events(asr_events)
        for asr_event in asr_events:
            self._record_asr_event_trace(asr_event)
        self._record_latency("asr", asr_started_ns, now_ns())
        authority_events = self._asr_events_to_authority(asr_events, ingress_received_ns=int(ingress_received_ns))
        for event in authority_events:
            payload = dict(event.payload)
            asr_event_ns = int(payload.get("asr_event_ns", 0) or 0)
            self._last_timestamps_ns["asr_event_ns"] = asr_event_ns
            if str(event.event_type) == "ASRPartialReceived":
                self._mark_first_asr_partial(asr_event_ns)
            elif str(event.event_type) == "ASRFinalReceived":
                self._mark_asr_final(asr_event_ns)
            self._append_event(event)
        frames = await self.run_tick_and_dispatch()
        self._maybe_prewarm_vllm_stable_prefix()
        return frames

    async def finalize_asr_turn(self, *, lineage_id: str | None = None) -> tuple[PCMFrame, ...]:
        self.assert_ready_for_live_audio()
        resolved_lineage_id = str(lineage_id or self.kernel.current_lease().epoch_id).strip()
        final_event = self.asr.finalize(lineage_id=resolved_lineage_id)
        if final_event is None:
            return ()
        final_events = self._filter_stale_asr_events((final_event,))
        for filtered_event in final_events:
            self._record_asr_event_trace(filtered_event)
        for authority_event in self._asr_events_to_authority(final_events, ingress_received_ns=int(now_ns())):
            payload = dict(authority_event.payload)
            asr_event_ns = int(payload.get("asr_event_ns", 0) or 0)
            self._last_timestamps_ns["asr_event_ns"] = asr_event_ns
            if str(authority_event.event_type) == "ASRPartialReceived":
                self._mark_first_asr_partial(asr_event_ns)
            elif str(authority_event.event_type) == "ASRFinalReceived":
                self._mark_asr_final(asr_event_ns)
            self._append_event(authority_event)
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
            should_cancel_tts = False
            if pre_tts_request_id:
                current_output_version = int(self.kernel.state.output.version)
                output_superseded = (
                    pre_tts_output_version >= 0 and int(pre_tts_output_version) != int(current_output_version)
                )
                epoch_changed = bool(pre_epoch_id) and bool(post_epoch_id) and str(pre_epoch_id) != str(post_epoch_id)
                request_replaced = bool(post_tts_request_id) and str(pre_tts_request_id) != str(post_tts_request_id)
                should_cancel_tts = bool(output_superseded or epoch_changed or request_replaced)
            if should_cancel_tts:
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
            if command.kind not in {"VLLM", "TTS", "TTS_APPEND", "VLLM_CANCEL", "TTS_CANCEL"}:
                raise RuntimeError(f"unsupported_dispatch_command_kind: {command.kind}")
            if command.kind in {"VLLM", "TTS", "TTS_APPEND"}:
                self._mirror_dispatch_command_to_ring(command)
            if command.kind == "VLLM":
                command_frames, spawned_commands = await self._execute_vllm_command(command)
            elif command.kind == "TTS":
                command_frames, spawned_commands = await self._execute_tts_command(command)
            elif command.kind == "TTS_APPEND":
                command_frames, spawned_commands = await self._execute_tts_append_command(command)
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
        output_version = int(payload.get("output_version", -1))
        if not request_id and not epoch_id:
            return (), ()
        try:
            session = self._active_tts_streams.pop(request_id, None) if request_id else None
            if session is not None:
                session.text_stream.close()
            speculative = self._speculative_tts
            if speculative is not None and request_id and request_id in {
                str(speculative.request_id),
                str(speculative.live_request_id),
            }:
                await self._cancel_speculative_tts()
            self.tts.cancel(request_id=request_id, epoch_id=epoch_id)
            current_lease = self.kernel.current_lease()
            current_output_version = int(self.kernel.state.output.version)
            superseded_output = output_version >= 0 and int(output_version) != int(current_output_version)
            epoch_changed = bool(epoch_id) and str(epoch_id) != str(current_lease.epoch_id)
            if superseded_output or epoch_changed:
                self.pcm_clock.clear()
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
        user_text = " ".join(str(payload.get("prompt", "")).strip().split())
        token_frames: list[PCMFrame] = []
        deferred_commands: list[DispatchCommand] = []
        first_token_seen = False
        self._mark_vllm_request_start(int(payload.get("kernel_decision_ns", now_ns()) or now_ns()))
        speculative = self._speculative_vllm
        if speculative is not None and speculative.source_text == user_text and not speculative.error:
            token_index = 0
            while True:
                while token_index < len(speculative.tokens):
                    if self.kernel.state.active_vllm_request_id != request_id:
                        break
                    if int(self.kernel.state.request_output_version(request_id)) != int(payload.get("output_version", -1)):
                        break
                    token = str(speculative.tokens[token_index] or "").strip()
                    token_index += 1
                    if not token:
                        continue
                    token_ns = now_ns()
                    if not first_token_seen:
                        self._record_latency("vllm", int(payload.get("kernel_decision_ns", token_ns) or token_ns), token_ns)
                        first_token_seen = True
                        self._last_timestamps_ns["vllm_first_token_ns"] = int(token_ns)
                        self._mark_first_spoken_delta(int(token_ns))
                    self._append_event(
                        self._authority_event(
                            event_type="VLLMChunkReceived",
                            lineage_id=lineage_id,
                            payload={
                                "request_id": request_id,
                                "token": token,
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
                if speculative.completed or speculative.error:
                    break
                if self.kernel.state.active_vllm_request_id != request_id:
                    break
                await asyncio.sleep(0.005)
            self._speculative_vllm = None
            if speculative.completed and self.kernel.state.request_output_version(request_id) == int(payload.get("output_version", -1)):
                self._append_event(
                    self._authority_event(
                        event_type="VLLMCompleted",
                        lineage_id=lineage_id,
                        payload={
                            "request_id": request_id,
                            "text": speculative.completed_text or _join_spoken_tokens(speculative.tokens),
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
        elif speculative is not None:
            await self._cancel_speculative_vllm()
        stable_session_summary = self._stable_session_summary()
        rendered_prompt = self.vllm.render_prompt(
            user_text=user_text,
            stable_session_summary=stable_session_summary,
        )
        prompt_cache_key = build_prompt_cache_key(
            system_prompt=str(self.config.vllm_system_prompt),
            context_prefix=stable_session_summary,
            stable_prefix=user_text,
        )
        try:
            async for token in self._vllm_streamer.stream(
                rendered_prompt,
                cache_key=prompt_cache_key,
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
                    self._mark_first_spoken_delta(int(token_ns))
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
                            "text": _join_spoken_tokens(self.kernel.state.output.vllm_tokens),
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
        close_stream_immediately = bool(payload.get("close_stream_immediately", False))
        pcm_frames: list[PCMFrame] = []
        deferred_commands: list[DispatchCommand] = []
        first_pcm_ns = 0
        leading_gate_open = False
        text_input: object = str(payload.get("text", ""))
        try:
            if stream_fragment:
                session = self._active_tts_streams.get(request_id)
                if session is None:
                    text_stream = _BlockingTextStream()
                    text_generator = text_stream.generator()
                    session = _ActiveTTSStreamSession(
                        request_id=request_id,
                        request_event_id=str(request_event_id),
                        lineage_id=lineage_id,
                        epoch_id=epoch_id,
                        output_version=output_version,
                        text_stream=text_stream,
                        text_generator=text_generator,
                        task=asyncio.create_task(
                            self._run_tts_stream_session(
                                request_id=request_id,
                                request_event_id=str(request_event_id),
                                lineage_id=lineage_id,
                                epoch_id=epoch_id,
                                output_version=output_version,
                                text_stream=text_generator,
                            ),
                            name=f"tts-stream:{request_id}",
                        ),
                    )
                    self._active_tts_streams[request_id] = session
                if session.text_stream is not None:
                    self._mark_tts_text_push(int(payload.get("kernel_decision_ns", now_ns()) or now_ns()))
                    if close_stream_immediately:
                        session.text_stream.push(str(payload.get("text", "")), final=True)
                    else:
                        session.text_stream.push(str(payload.get("text", "")), final=False)
                return (), ()
            else:
                text_input = " ".join(str(payload.get("text", "")).strip().split())
                speculative_tts = self._speculative_tts
                if (
                    speculative_tts is not None
                    and not speculative_tts.promoted
                    and not speculative_tts.error
                    and speculative_tts.source_text == str(text_input)
                ):
                    await self._promote_speculative_tts_request(
                        speculative_tts,
                        request_id=request_id,
                        request_event_id=str(request_event_id),
                        lineage_id=lineage_id,
                        epoch_id=epoch_id,
                        output_version=output_version,
                        kernel_decision_ns=int(payload.get("kernel_decision_ns", now_ns()) or now_ns()),
                    )
                    return (), ()
                if speculative_tts is not None and not speculative_tts.promoted:
                    await self._cancel_speculative_tts()
                self._start_tts_request_metrics(
                    request_id=request_id,
                    started_ns=int(payload.get("kernel_decision_ns", now_ns()) or now_ns()),
                )
                self.transport.start_egress_request(request_id)
                self._mark_tts_text_push(int(payload.get("kernel_decision_ns", now_ns()) or now_ns()))
            last_emitted_ns = 0
            if epoch_id:
                self.tts.reset(epoch_id=epoch_id)
            self._mark_tts_native_stream_open(now_ns())
            async for pcm_chunk, sample_rate, is_final in self._tts_streamer.stream(
                text_input,
                request_id=request_id,
                epoch_id=epoch_id,
            ):
                raw_rms, raw_peak = self._update_tts_signal_metrics(stage="raw", pcm_bytes=bytes(pcm_chunk))
                if self.kernel.state.active_tts_request_id != request_id:
                    break
                if int(self.kernel.state.request_output_version(request_id)) != int(payload.get("output_version", -1)):
                    break
                observed_ns = now_ns()
                resampled = self._resample_output(
                    bytes(pcm_chunk),
                    int(sample_rate),
                    epoch_id=epoch_id,
                    output_version=output_version,
                    flush=bool(is_final),
                )
                if resampled:
                    self._mark_resampler_first_output(observed_ns)
                resampled_rms, resampled_peak = self._update_tts_signal_metrics(stage="resampled", pcm_bytes=resampled)
                drop_final_tail = self._should_drop_final_tts_resampler_tail(
                    last_emitted_ns=last_emitted_ns,
                    observed_ns=observed_ns,
                    raw_pcm=bytes(pcm_chunk),
                    raw_rms=raw_rms,
                    raw_peak=raw_peak,
                    resampled_pcm=resampled,
                    resampled_rms=resampled_rms,
                    resampled_peak=resampled_peak,
                    is_final=bool(is_final),
                )
                if drop_final_tail:
                    resampled = b""
                flush_final = bool(is_final)
                output_frames = self._chunk_output_pcm(
                    resampled,
                    epoch_id=epoch_id,
                    output_version=output_version,
                    flush=flush_final,
                    drop_carry_on_flush=bool(drop_final_tail),
                )
                frame_batch: list[tuple[bytes, float, float]] = []
                for frame_pcm in output_frames:
                    chunked_rms, chunked_peak = self._update_tts_signal_metrics(stage="chunked", pcm_bytes=frame_pcm)
                    frame_batch.append((frame_pcm, chunked_rms, chunked_peak))
                batch_start_index = self._tts_leading_batch_start_index(
                    gate_open=leading_gate_open,
                    frame_metrics=[(chunked_rms, chunked_peak) for _, chunked_rms, chunked_peak in frame_batch],
                    is_final=bool(is_final),
                )
                batch_gate_open = batch_start_index is not None
                for frame_index, (frame_pcm, chunked_rms, chunked_peak) in enumerate(frame_batch):
                    if batch_gate_open and not leading_gate_open:
                        should_emit = frame_index >= int(batch_start_index or 0)
                    else:
                        should_emit = self._should_emit_tts_frame(
                            gate_open=batch_gate_open,
                            chunked_rms=chunked_rms,
                            chunked_peak=chunked_peak,
                            is_final=bool(is_final),
                        )
                    self._record_tts_chunk_trace(
                        observed_ns=observed_ns,
                        raw_rms=raw_rms,
                        raw_peak=raw_peak,
                        resampled_rms=resampled_rms,
                        resampled_peak=resampled_peak,
                        chunked_rms=chunked_rms,
                        chunked_peak=chunked_peak,
                        emitted=bool(should_emit),
                        is_final=bool(is_final),
                    )
                    if not should_emit:
                        self._record_trimmed_tts_frame()
                        continue
                    leading_gate_open = True
                    batch_gate_open = True
                    self._tts_signal_metrics["tts_leading_gate_open"] = 1
                    self._mark_tts_gate_open(observed_ns)
                    if not first_pcm_ns:
                        first_pcm_ns = observed_ns
                        self._mark_tts_first_pcm(
                            observed_ns=observed_ns,
                            decision_ns=int(payload.get("kernel_decision_ns", observed_ns) or observed_ns),
                        )
                    frame = PCMFrame(
                        pcm=frame_pcm,
                        sample_rate=int(self.config.output_sample_rate),
                        epoch_id=epoch_id,
                        output_version=output_version,
                        request_id=request_id,
                    )
                    pcm_frames.append(frame)
                    self._mirror_pcm_frame_to_ring(frame)
                    self.pcm_clock.enqueue(frame)
                    self._mark_pcm_enqueued(observed_ns)
                    last_emitted_ns = int(observed_ns)
                    self._append_event(
                        self._authority_event(
                            event_type="TTSChunkReceived",
                            lineage_id=lineage_id,
                            payload={
                                "request_id": request_id,
                                "chunk_id": f"{request_id}:{len(pcm_frames)}",
                                "output_version": output_version,
                                "tts_first_pcm_ns": int(first_pcm_ns),
                                "raw_rms": float(raw_rms),
                                "raw_peak": float(raw_peak),
                                "resampled_rms": float(resampled_rms),
                                "resampled_peak": float(resampled_peak),
                                "chunked_rms": float(chunked_rms),
                                "chunked_peak": float(chunked_peak),
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

    async def _execute_tts_append_command(
        self,
        command: DispatchCommand,
    ) -> tuple[tuple[PCMFrame, ...], tuple[DispatchCommand, ...]]:
        payload = dict(command.payload)
        request_id = str(command.request_id or "").strip()
        if not request_id:
            return (), ()
        session = self._active_tts_streams.get(request_id)
        if session is None:
            return (), ()
        if int(session.output_version) != int(payload.get("output_version", -1)):
            return (), ()
        session.text_stream.push(
            str(payload.get("text", "")),
            final=bool(payload.get("final_fragment", False)),
        )
        if bool(payload.get("final_fragment", False)):
            await session.task
        return (), ()

    async def _run_tts_stream_session(
        self,
        *,
        request_id: str,
        request_event_id: str,
        lineage_id: str,
        epoch_id: str,
        output_version: int,
        text_stream: object,
    ) -> None:
        assert self._tts_streamer is not None
        assert self._output_resampler is not None
        first_pcm_ns = 0
        leading_gate_open = False
        last_emitted_ns = 0
        try:
            self._start_tts_request_metrics(request_id=request_id, started_ns=now_ns())
            self._mark_tts_text_push(now_ns())
            self.transport.start_egress_request(request_id)
            self._mark_tts_native_stream_open(now_ns())
            if epoch_id:
                self.tts.reset(epoch_id=epoch_id)
            async for pcm_chunk, sample_rate, is_final in self._tts_streamer.stream(
                text_stream,
                request_id=request_id,
                epoch_id=epoch_id,
            ):
                raw_rms, raw_peak = self._update_tts_signal_metrics(stage="raw", pcm_bytes=bytes(pcm_chunk))
                if self.kernel.state.active_tts_request_id != request_id:
                    break
                if int(self.kernel.state.request_output_version(request_id)) != int(output_version):
                    break
                observed_ns = now_ns()
                resampled = self._resample_output(
                    bytes(pcm_chunk),
                    int(sample_rate),
                    epoch_id=epoch_id,
                    output_version=output_version,
                    flush=bool(is_final),
                )
                if resampled:
                    self._mark_resampler_first_output(observed_ns)
                resampled_rms, resampled_peak = self._update_tts_signal_metrics(stage="resampled", pcm_bytes=resampled)
                drop_final_tail = self._should_drop_final_tts_resampler_tail(
                    last_emitted_ns=last_emitted_ns,
                    observed_ns=observed_ns,
                    raw_pcm=bytes(pcm_chunk),
                    raw_rms=raw_rms,
                    raw_peak=raw_peak,
                    resampled_pcm=resampled,
                    resampled_rms=resampled_rms,
                    resampled_peak=resampled_peak,
                    is_final=bool(is_final),
                )
                if drop_final_tail:
                    resampled = b""
                output_frames = self._chunk_output_pcm(
                    resampled,
                    epoch_id=epoch_id,
                    output_version=output_version,
                    flush=bool(is_final),
                    drop_carry_on_flush=bool(drop_final_tail),
                )
                frame_batch: list[tuple[bytes, float, float]] = []
                for frame_pcm in output_frames:
                    session = self._active_tts_streams.get(request_id)
                    if session is not None:
                        session.chunk_counter = int(session.chunk_counter) + 1
                        chunk_index = int(session.chunk_counter)
                    else:
                        chunk_index = 0
                    chunked_rms, chunked_peak = self._update_tts_signal_metrics(stage="chunked", pcm_bytes=frame_pcm)
                    frame_batch.append((frame_pcm, chunked_rms, chunked_peak, chunk_index))
                batch_start_index = self._tts_leading_batch_start_index(
                    gate_open=leading_gate_open,
                    frame_metrics=[(chunked_rms, chunked_peak) for _, chunked_rms, chunked_peak, _ in frame_batch],
                    is_final=bool(is_final),
                )
                batch_gate_open = batch_start_index is not None
                for frame_index, (frame_pcm, chunked_rms, chunked_peak, chunk_index) in enumerate(frame_batch):
                    if batch_gate_open and not leading_gate_open:
                        should_emit = frame_index >= int(batch_start_index or 0)
                    else:
                        should_emit = self._should_emit_tts_frame(
                            gate_open=batch_gate_open,
                            chunked_rms=chunked_rms,
                            chunked_peak=chunked_peak,
                            is_final=bool(is_final),
                        )
                    self._record_tts_chunk_trace(
                        observed_ns=observed_ns,
                        raw_rms=raw_rms,
                        raw_peak=raw_peak,
                        resampled_rms=resampled_rms,
                        resampled_peak=resampled_peak,
                        chunked_rms=chunked_rms,
                        chunked_peak=chunked_peak,
                        emitted=bool(should_emit),
                        is_final=bool(is_final),
                    )
                    if not should_emit:
                        self._record_trimmed_tts_frame()
                        continue
                    leading_gate_open = True
                    batch_gate_open = True
                    self._tts_signal_metrics["tts_leading_gate_open"] = 1
                    self._mark_tts_gate_open(observed_ns)
                    if not first_pcm_ns:
                        first_pcm_ns = observed_ns
                        self._mark_tts_first_pcm(observed_ns=observed_ns, decision_ns=observed_ns)
                    frame = PCMFrame(
                        pcm=frame_pcm,
                        sample_rate=int(self.config.output_sample_rate),
                        epoch_id=epoch_id,
                        output_version=output_version,
                        request_id=request_id,
                    )
                    self._mirror_pcm_frame_to_ring(frame)
                    self.pcm_clock.enqueue(frame)
                    self._mark_pcm_enqueued(observed_ns)
                    last_emitted_ns = int(observed_ns)
                    self._append_event(
                        self._authority_event(
                            event_type="TTSChunkReceived",
                            lineage_id=lineage_id,
                            payload={
                                "request_id": request_id,
                                "chunk_id": f"{request_id}:{chunk_index}",
                                "output_version": output_version,
                                "tts_first_pcm_ns": int(first_pcm_ns),
                                "raw_rms": float(raw_rms),
                                "raw_peak": float(raw_peak),
                                "resampled_rms": float(resampled_rms),
                                "resampled_peak": float(resampled_peak),
                                "chunked_rms": float(chunked_rms),
                                "chunked_peak": float(chunked_peak),
                                "lane": "tts",
                            },
                            causation_id=request_event_id,
                        )
                    )
                nested_commands = await self._tick_and_stamp_commands()
                await self._dispatch_commands(nested_commands, blocked_kinds=frozenset({"TTS", "TTS_APPEND"}))
                if is_final:
                    break
            if self.kernel.state.request_output_version(request_id) == int(output_version):
                self._append_event(
                    self._authority_event(
                        event_type="TTSCompleted",
                        lineage_id=lineage_id,
                        payload={
                            "request_id": request_id,
                            "output_version": int(output_version),
                            "lane": "tts",
                        },
                        causation_id=request_event_id,
                    )
                )
                nested_commands = await self._tick_and_stamp_commands()
                await self._dispatch_commands(nested_commands, blocked_kinds=frozenset({"TTS", "TTS_APPEND"}))
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
            await self._dispatch_commands(nested_commands, blocked_kinds=frozenset({"TTS", "TTS_APPEND"}))
        finally:
            self._active_tts_streams.pop(request_id, None)

    def _reset_output_resampler(self, *, epoch_id: str, output_version: int) -> None:
        self._output_resampler = StreamingAudioResampler(target_rate=int(self.config.output_sample_rate))
        self._output_resampler_epoch_id = str(epoch_id)
        self._output_resampler_output_version = int(output_version)

    def _resample_output(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        *,
        epoch_id: str,
        output_version: int,
        flush: bool,
    ) -> bytes:
        assert self._output_resampler is not None
        if (
            str(self._output_resampler_epoch_id) != str(epoch_id)
            or int(self._output_resampler_output_version) != int(output_version)
        ):
            self._reset_output_resampler(epoch_id=epoch_id, output_version=output_version)
        audio = np.frombuffer(bytes(pcm_bytes), dtype=np.int16).astype(np.float32) / 32768.0
        resampled = self._output_resampler.resample(audio, int(sample_rate))
        if flush:
            tail = self._output_resampler.flush()
            if tail.size:
                resampled = np.concatenate((resampled, tail))
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
        drop_carry_on_flush: bool = False,
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
            if not bool(drop_carry_on_flush):
                padded = carry + (b"\x00" * (frame_bytes - len(carry)))
                frames.append(padded)
            carry = b""
        self._tts_frame_carry = carry
        return tuple(frames)

    async def send_pcm_once(self, send_fn: Callable[..., Awaitable[None]]) -> None:
        emit_started_ns = now_ns()
        silence_frame = b"\x00" * self._output_frame_bytes()
        playout_lease = self.pcm_clock.head_lease()
        if playout_lease is None:
            current_epoch_id = self.kernel.current_lease().epoch_id
            current_output_version = int(self.kernel.state.output.version)
        else:
            current_epoch_id, current_output_version = playout_lease
        async def _instrumented_send_fn(pcm_bytes: bytes, sample_rate: int, request_id: str) -> None:
            if pcm_bytes and str(request_id or "").strip():
                self._mark_pcm_send(now_ns())
            try:
                parameter_count = len(inspect.signature(send_fn).parameters)
            except (TypeError, ValueError):
                parameter_count = 3
            if parameter_count <= 2:
                await send_fn(pcm_bytes, sample_rate)
                return
            await send_fn(pcm_bytes, sample_rate, request_id)
        sent_request_id = await self.pcm_clock.run_once(
            _instrumented_send_fn,
            current_epoch_id=str(current_epoch_id),
            current_output_version=int(current_output_version),
            silence_frame=silence_frame,
            silence_sample_rate=int(self.config.output_sample_rate),
        )
        emitted_ns = now_ns()
        self._record_latency("transport", emit_started_ns, emitted_ns)
        if sent_request_id and int(self._last_timestamps_ns.get("transport_emit_ns", 0) or 0) == 0:
            self._last_timestamps_ns["transport_emit_ns"] = int(emitted_ns)

    def latency_summary(self) -> dict[str, LatencySummary]:
        return {name: summarize_latency(samples) for name, samples in self._latency_samples.items()}

    def replay_state_hash(self) -> str:
        return canonical_state_hash(self.kernel.state.__dict__ if hasattr(self.kernel.state, "__dict__") else repr(self.kernel.state))

    def replay_event_hash(self) -> str:
        return canonical_event_stream_hash(tuple(self.event_log.as_records()))

    def last_timestamps(self) -> dict[str, int]:
        return dict(self._last_timestamps_ns)

    def tts_signal_metrics(self) -> dict[str, object]:
        return dict(self._tts_signal_metrics)

    def ingress_frame_trace(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(item) for item in self._ingress_frame_trace)

    def asr_event_trace(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(item) for item in self._asr_event_trace)

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
            allow_partial_turn_commit=bool(
                bool(config.allow_partial_turn_commit) and not bool(config.livekit_use_turn_detector)
            ),
            tick_interval_ms=int(config.tick_interval_ms),
            tts_fragment_min_tokens=int(config.tts_fragment_min_tokens),
            tts_first_fragment_min_tokens=int(config.tts_first_fragment_min_tokens),
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


def _run_async_probe(
    coro_factory: Callable[[], Awaitable[bool]],
    *,
    timeout_s: float | None = None,
) -> bool:
    result: dict[str, bool] = {"ok": False}
    error: dict[str, Exception] = {}

    def _runner() -> None:
        try:
            result["ok"] = bool(asyncio.run(coro_factory()))
        except Exception as exc:  # pragma: no cover - optional dependency path
            error["exc"] = exc

    thread = Thread(target=_runner, name="voice-pipeline-warm-probe", daemon=True)
    thread.start()
    join_timeout = None if timeout_s is None else max(0.0, float(timeout_s))
    thread.join(timeout=join_timeout)
    if thread.is_alive():
        raise TimeoutError(f"warm_probe_timeout_after_{join_timeout:.1f}s")
    if "exc" in error:
        raise RuntimeError(f"warm_probe_failed: {error['exc']}") from error["exc"]
    return bool(result["ok"])


def _token_count(text: str) -> int:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return 0
    return len(normalized.split(" "))


def _shutdown_lane_backend(label: str, backend: object) -> str:
    shutdown = getattr(backend, "shutdown", None)
    if not callable(shutdown):
        return ""
    try:
        shutdown()
    except Exception as exc:
        LOGGER.warning("lane shutdown failed: %s (%s)", label, exc)
        return str(exc)
    return ""


def _warm_vllm_engine(vllm: VLLMEngine, config: RuntimeConfig) -> bool:
    vllm.warm(strict=True)
    warm_summary = "voice runtime stable session scaffold"
    warm_user_text = "hello"
    vllm.prewarm_prefix_cache(
        build_prompt_cache_key(
            system_prompt=str(config.vllm_system_prompt),
            context_prefix=warm_summary,
            stable_prefix="warmup",
        ),
        build_prompt_cache_key(
            system_prompt=str(config.vllm_system_prompt),
            context_prefix="",
            stable_prefix=warm_user_text,
        ),
    )
    if not vllm.prefix_cache_ready:
        raise RuntimeError("vllm_prefix_cache_not_ready")
    return bool(vllm.is_warm and vllm.prefix_cache_ready)


def _warm_tts_engine(tts: TTSEngine, config: RuntimeConfig, session_id: str) -> bool:
    _bind_cuda_device(config.tts_device)
    tts.warm(strict=True)
    tts.start_persistent_session(
        epoch_id=session_id,
        prompt_text=config.resolved_cosyvoice3_prompt_text(),
        prompt_speech_path=config.cosyvoice3_speaker_path,
    )
    if bool(config.tts_skip_stream_probe):
        return bool(tts.is_warm)
    warm_request_id = f"{session_id}:bootstrap-tts-warmup"

    async def _probe() -> bool:
        # Keep bootstrap bounded: touch the generator decoder path,
        # but return on the first emitted PCM frame instead of draining
        # the whole warmup utterance.
        async for pcm_chunk, sample_rate, _is_final in tts.stream_pcm(
            _single_fragment_generator("warmup"),
            request_id=warm_request_id,
            epoch_id=session_id,
        ):
            if pcm_chunk and int(sample_rate) > 0:
                return True
        return False

    try:
        return bool(tts.is_warm and _run_async_probe(_probe, timeout_s=10.0))
    except TimeoutError:
        LOGGER.warning("tts bootstrap warm probe timed out; recovering with fresh persistent session")
        tts.cancel(request_id=warm_request_id, epoch_id=session_id)
        tts.start_persistent_session(
            epoch_id=session_id,
            prompt_text=config.resolved_cosyvoice3_prompt_text(),
            prompt_speech_path=config.cosyvoice3_speaker_path,
        )
        return bool(tts.is_warm)


def bootstrap_runtime(
    *,
    session_id: str,
    config: RuntimeConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> VoicePipelineRuntime:
    runtime_config = config or RuntimeConfig.from_env()
    if progress_callback is not None:
        progress_callback("contract_check")
    _assert_contract(runtime_config)
    hardware_admission_check(runtime_config)

    if progress_callback is not None:
        progress_callback("topology_init")
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
            required_device=runtime_config.llm_device,
            max_num_seqs=int(runtime_config.vllm_max_num_seqs),
            gpu_memory_utilization=float(runtime_config.vllm_gpu_memory_utilization),
            max_model_len=int(runtime_config.vllm_max_model_len),
            max_num_batched_tokens=int(runtime_config.vllm_max_num_batched_tokens),
            offload_backend=str(runtime_config.vllm_offload_backend),
            cpu_offload_gb=float(runtime_config.vllm_cpu_offload_gb),
            kv_offloading_size=float(runtime_config.vllm_kv_offloading_size),
            kv_offloading_backend=str(runtime_config.vllm_kv_offloading_backend),
            kv_cache_dtype=str(runtime_config.vllm_kv_cache_dtype),
            kv_cache_memory_bytes=int(runtime_config.vllm_kv_cache_memory_bytes),
            num_gpu_blocks_override=int(runtime_config.vllm_num_gpu_blocks_override),
            attention_backend=str(runtime_config.vllm_attention_backend),
            safetensors_load_strategy=str(runtime_config.vllm_safetensors_load_strategy),
            temperature=float(runtime_config.vllm_temperature),
            top_p=float(runtime_config.vllm_top_p),
            max_tokens=int(runtime_config.vllm_max_tokens),
            system_prompt=str(runtime_config.vllm_system_prompt),
        ),
    )
    tts = TTSEngine(
        runtime_config.resolved_cosyvoice3_model_path(),
        sample_rate=24_000,
        max_queue_depth=int(runtime_config.tts_max_queue_depth),
        prompt_text=runtime_config.resolved_cosyvoice3_prompt_text(),
        prompt_speech_path=runtime_config.cosyvoice3_speaker_path,
        required_device=runtime_config.tts_device,
        cache_dir=runtime_config.cosyvoice3_cache_dir,
    )

    warm_report = WarmReport()
    worker_status = WorkerStatus()
    worker_failure_reason = WorkerFailureReason()

    worker_status.asr = "WARMING"
    if progress_callback is not None:
        progress_callback("asr_warm")
    try:
        warm_report.asr_warm = _warm_asr_engine(asr, runtime_config, session_id)
    except Exception as exc:
        warm_report.asr_warm = False
        worker_failure_reason.asr = str(exc)
        shutdown_error = _shutdown_lane_backend("asr", asr)
        if shutdown_error:
            worker_failure_reason.asr = f"{worker_failure_reason.asr}; shutdown_failed: {shutdown_error}"
    worker_status.asr = "READY" if warm_report.asr_warm else "FAILED"

    worker_status.vllm = "WARMING"
    if progress_callback is not None:
        progress_callback("vllm_warm")
    try:
        warm_report.vllm_warm = _warm_vllm_engine(vllm, runtime_config)
        vllm.prewarm_prefix_cache(f"{session_id}:stable_session_scaffold")
    except Exception as exc:
        warm_report.vllm_warm = False
        worker_failure_reason.vllm = str(exc)
        shutdown_error = _shutdown_lane_backend("vllm", vllm)
        if shutdown_error:
            worker_failure_reason.vllm = f"{worker_failure_reason.vllm}; shutdown_failed: {shutdown_error}"
    worker_status.vllm = "READY" if warm_report.vllm_warm else "FAILED"

    worker_status.tts = "WARMING"
    if progress_callback is not None:
        progress_callback("tts_warm")
    try:
        warm_report.tts_warm = _warm_tts_engine(tts, runtime_config, session_id)
    except Exception as exc:
        warm_report.tts_warm = False
        worker_failure_reason.tts = str(exc)
        shutdown_error = _shutdown_lane_backend("tts", tts)
        if shutdown_error:
            worker_failure_reason.tts = f"{worker_failure_reason.tts}; shutdown_failed: {shutdown_error}"
    worker_status.tts = "READY" if warm_report.tts_warm else "FAILED"

    if progress_callback is not None:
        progress_callback("kernel_init")
    kernel = _build_kernel(runtime_config, topology, rings, session_id)
    worker_status.kernel = "READY"
    if progress_callback is not None:
        progress_callback("transport_init")
    transport = LiveKitTransport(
        config=LiveKitTransportConfig(
            url=str(runtime_config.livekit_url),
            api_key=str(runtime_config.livekit_api_key),
            api_secret=str(runtime_config.livekit_api_secret),
            room_name=str(runtime_config.livekit_room_name),
            runtime_identity=str(runtime_config.livekit_runtime_identity),
            output_track_name=str(runtime_config.livekit_output_track_name),
            input_participant_identity=str(runtime_config.livekit_input_participant_identity),
            input_track_name=str(runtime_config.livekit_input_track_name),
            frame_ms=int(runtime_config.frame_ms),
            input_frame_ms=int(runtime_config.livekit_input_frame_ms),
            input_sample_rate=int(runtime_config.input_sample_rate),
            output_sample_rate=int(runtime_config.output_sample_rate),
            input_queue_size_ms=int(runtime_config.livekit_input_queue_size_ms),
            output_queue_size_ms=int(runtime_config.livekit_output_queue_size_ms),
            output_preconnect_buffer=bool(runtime_config.livekit_output_preconnect_buffer),
            single_peer_connection=bool(runtime_config.livekit_single_peer_connection),
            single_ingress_track=bool(runtime_config.livekit_single_ingress_track),
            token_ttl_seconds=int(runtime_config.livekit_token_ttl_seconds),
        )
    )
    pcm_clock = PCMClockSender(
        tick_ms=int(runtime_config.frame_ms),
        target_buffer_frames=int(runtime_config.pcm_target_buffer_frames),
        max_buffer_frames=int(runtime_config.pcm_max_buffer_frames),
    )
    if progress_callback is not None:
        progress_callback("runtime_object_init")

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
    if not (warm_report.asr_warm and warm_report.vllm_warm and warm_report.tts_warm):
        runtime.worker_status.kernel = "FAILED"
        runtime.worker_failure_reason.kernel = (
            "startup_failed: required lane warmup did not reach READY"
            f" asr={worker_failure_reason.asr or worker_status.asr}"
            f" vllm={worker_failure_reason.vllm or worker_status.vllm}"
            f" tts={worker_failure_reason.tts or worker_status.tts}"
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
