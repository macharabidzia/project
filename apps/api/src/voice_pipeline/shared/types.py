from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, TypedDict

from voice_pipeline.shared.lineage import (
    build_identity_closure,
    canonical_session_id,
    canonical_turn_id,
    validate_identity_closure,
)


EventType = str
WorkerKind = Literal["asr", "vllm", "tts"]


class DispatchPayload(TypedDict, total=False):
    prompt: str
    text: str
    output_version: int
    lineage_id: str
    turn_id: str


EVENT_SCHEMA_VERSION = "v1"
AUTHORITY_EVENT_TYPES = frozenset(
    {
        "ASRPartialReceived",
        "ASRFinalReceived",
        "TurnCommitted",
        "TTSRequested",
        "TTSChunkReceived",
        "TTSCompleted",
        "VLLMRequested",
        "VLLMChunkReceived",
        "VLLMCompleted",
        "InterruptRequested",
        "CancelRequested",
        "RecoveryRequested",
        "RecoveryCompleted",
        "WorkerDrainRequested",
        "WorkerWarming",
        "WorkerRebound",
        "WorkerDetached",
        "LLMFaulted",
        "TTSFaulted",
    }
)
ENGINE_OUTPUT_EVENT_TYPES = frozenset(
    {
        "TTSChunkReceived",
        "TTSCompleted",
        "VLLMChunkReceived",
        "VLLMCompleted",
    }
)


class EventValidationError(ValueError):
    pass


def _freeze_mapping(payload: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(payload or {}))


@dataclass(frozen=True, slots=True)
class AuthorityEvent:
    schema_version: str
    event_type: str
    event_id: str
    session_id: str
    sequence_no: int
    lineage_id: str
    causation_id: str = ""
    payload: Mapping[str, object] = field(default_factory=dict)
    observations: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", str(self.schema_version or "").strip())
        object.__setattr__(self, "event_type", str(self.event_type or "").strip())
        object.__setattr__(self, "event_id", str(self.event_id or "").strip())
        object.__setattr__(self, "session_id", canonical_session_id(self.session_id))
        object.__setattr__(self, "sequence_no", int(self.sequence_no))
        object.__setattr__(self, "lineage_id", str(self.lineage_id or "").strip())
        object.__setattr__(self, "causation_id", str(self.causation_id or "").strip())
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))
        object.__setattr__(self, "observations", _freeze_mapping(self.observations))
        validate_authority_event(self)


def new_authority_event(
    *,
    event_type: str,
    session_id: str,
    sequence_no: int,
    lineage_id: str,
    payload: Mapping[str, object] | None = None,
    causation_id: str = "",
    event_id: str | None = None,
    observations: Mapping[str, object] | None = None,
) -> AuthorityEvent:
    resolved_session_id = canonical_session_id(session_id)
    resolved_event_type = str(event_type or "").strip()
    resolved_event_id = str(event_id or "").strip() or f"{resolved_session_id}:{int(sequence_no)}:{resolved_event_type}:{uuid.uuid4().hex[:8]}"
    turn_epoch = int((payload or {}).get("turn_epoch", 0) or 0)
    generation_epoch = int((payload or {}).get("generation_epoch", 0) or 0)
    closure = build_identity_closure(
        session_id=resolved_session_id,
        event_id=resolved_event_id,
        turn_id=(payload or {}).get("turn_id", canonical_turn_id(session_id=resolved_session_id, turn_epoch=turn_epoch)),
        epoch_id=lineage_id,
        turn_epoch=turn_epoch,
        generation_epoch=generation_epoch,
    )
    merged_observations = dict(observations or {})
    merged_observations.setdefault("session_id", closure.session_id)
    merged_observations.setdefault("turn_id", closure.turn_id)
    merged_observations.setdefault("epoch_id", closure.epoch_id)
    merged_observations.setdefault("event_id", closure.event_id)
    merged_observations.setdefault("trace_id", closure.trace_id)

    return AuthorityEvent(
        schema_version=EVENT_SCHEMA_VERSION,
        event_type=resolved_event_type,
        event_id=resolved_event_id,
        session_id=resolved_session_id,
        sequence_no=int(sequence_no),
        lineage_id=str(lineage_id or "").strip(),
        causation_id=str(causation_id or "").strip(),
        payload=payload,
        observations=merged_observations,
    )


def validate_authority_event(
    event: AuthorityEvent,
    *,
    expected_session_id: str | None = None,
    expected_sequence_no: int | None = None,
) -> None:
    if str(event.schema_version or "").strip() != EVENT_SCHEMA_VERSION:
        raise EventValidationError(f"unsupported schema version: {event.schema_version!r}")
    if str(event.event_type or "").strip() not in AUTHORITY_EVENT_TYPES:
        raise EventValidationError(f"unsupported event type: {event.event_type!r}")
    if not str(event.event_id or "").strip():
        raise EventValidationError("event_id is required")
    if int(event.sequence_no) <= 0:
        raise EventValidationError("sequence_no must be positive")
    if not str(event.lineage_id or "").strip():
        raise EventValidationError("lineage_id is required")
    if expected_session_id is not None and canonical_session_id(expected_session_id) != canonical_session_id(event.session_id):
        raise EventValidationError("session_id mismatch")
    if expected_sequence_no is not None and int(event.sequence_no) != int(expected_sequence_no):
        raise EventValidationError(f"expected sequence {int(expected_sequence_no)}, got {int(event.sequence_no)}")
    closure = build_identity_closure(
        session_id=event.session_id,
        event_id=event.event_id,
        turn_id=event.observations.get("turn_id", ""),
        epoch_id=event.observations.get("epoch_id", event.lineage_id),
        turn_epoch=int(event.payload.get("turn_epoch", 0) or 0),
        generation_epoch=int(event.payload.get("generation_epoch", 0) or 0),
    )
    if not validate_identity_closure(closure):
        raise EventValidationError("event identity closure is incomplete")


__all__ = [
    "AUTHORITY_EVENT_TYPES",
    "ENGINE_OUTPUT_EVENT_TYPES",
    "EVENT_SCHEMA_VERSION",
    "AuthorityEvent",
    "DispatchPayload",
    "EventType",
    "EventValidationError",
    "WorkerKind",
    "new_authority_event",
    "validate_authority_event",
]
