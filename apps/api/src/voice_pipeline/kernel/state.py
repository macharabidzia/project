from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal


ReducerPhase = Literal["idle", "listening", "generating", "playing", "cancelled"]


@dataclass(frozen=True, slots=True)
class TranscriptState:
    partial_text: str = ""
    partial_history: tuple[str, ...] = ()
    stable_prefix: str = ""
    stable_prefix_confirmations: int = 0
    last_dispatched_stable_prefix: str = ""
    final_text: str = ""
    committed_text: str = ""
    conversation_history: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OutputState:
    active_turn_id: str = ""
    version: int = 0
    vllm_tokens: tuple[str, ...] = ()
    vllm_stream_buffer: tuple[str, ...] = ()
    pending_tts_segments: tuple[str, ...] = ()
    emitted_audio_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecoveryStatus:
    state: Literal["healthy", "recovering", "recovered"] = "healthy"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class KernelState:
    session_id: str
    phase: ReducerPhase = "idle"
    transcript: TranscriptState = field(default_factory=TranscriptState)
    output: OutputState = field(default_factory=OutputState)
    recovery: RecoveryStatus = field(default_factory=RecoveryStatus)
    turn_index: int = 0
    committed_turn_index: int = 0
    generation_index: int = 0
    lineage_id: str = ""
    active_vllm_request_id: str = ""
    active_tts_request_id: str = ""
    last_event_id: str = ""
    last_sequence_no: int = 0
    recent_event_ids: tuple[str, ...] = ()
    request_event_ids: tuple[tuple[str, str, int], ...] = ()

    def remember_event(self, event_id: str, *, limit: int = 128) -> "KernelState":
        resolved_event_id = str(event_id or "").strip()
        if not resolved_event_id:
            return self
        recent = tuple(item for item in self.recent_event_ids if item != resolved_event_id)
        recent = (*recent, resolved_event_id)
        bounded = max(1, int(limit))
        if len(recent) > bounded:
            recent = recent[-bounded:]
        return replace(self, recent_event_ids=recent)

    def bind_request_event(self, request_id: str, event_id: str, *, output_version: int) -> "KernelState":
        resolved_request_id = str(request_id or "").strip()
        resolved_event_id = str(event_id or "").strip()
        if not resolved_request_id or not resolved_event_id:
            return self
        pairs = tuple(item for item in self.request_event_ids if item[0] != resolved_request_id)
        return replace(self, request_event_ids=(*pairs, (resolved_request_id, resolved_event_id, int(output_version))))

    def request_event_id(self, request_id: str) -> str:
        resolved_request_id = str(request_id or "").strip()
        for candidate_request_id, event_id, _output_version in reversed(self.request_event_ids):
            if candidate_request_id == resolved_request_id:
                return str(event_id)
        return ""

    def request_output_version(self, request_id: str) -> int:
        resolved_request_id = str(request_id or "").strip()
        for candidate_request_id, _event_id, output_version in reversed(self.request_event_ids):
            if candidate_request_id == resolved_request_id:
                return int(output_version)
        return -1


__all__ = ["KernelState", "OutputState", "RecoveryStatus", "ReducerPhase", "TranscriptState"]


