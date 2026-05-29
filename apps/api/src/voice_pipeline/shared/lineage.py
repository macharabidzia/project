from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DEFAULT_EVENT_SCHEMA_VERSION = "1.0"
DEFAULT_TELEMETRY_SCHEMA_VERSION = "1.0"
DEFAULT_SESSION_ID = "session-unbound"


def canonical_session_id(session_id: object, *, default: str = DEFAULT_SESSION_ID) -> str:
    normalized = str(session_id or "").strip()
    return normalized or str(default).strip() or DEFAULT_SESSION_ID


def canonical_turn_id(*, session_id: object, turn_epoch: int) -> str:
    return f"{canonical_session_id(session_id)}:turn:{int(turn_epoch)}"


def canonical_epoch_id(*, session_id: object, generation_epoch: object) -> str:
    return f"{canonical_session_id(session_id)}:epoch:{int(generation_epoch)}"


def canonical_response_request_id(
    *,
    session_id: object,
    epoch_id: object,
    suffix: object,
) -> str:
    normalized_epoch = str(epoch_id or "").strip()
    normalized_suffix = str(suffix or "").strip()
    prefix = normalized_epoch or canonical_session_id(session_id)
    return f"{prefix}:req:{normalized_suffix}" if normalized_suffix else prefix


@dataclass(frozen=True, slots=True)
class EpochLineage:
    turn_epoch: int
    committed_epoch: int
    generation_epoch: int

    def as_id(self) -> str:
        return f"t{int(self.turn_epoch)}:c{int(self.committed_epoch)}:g{int(self.generation_epoch)}"


def build_lineage_id(*, turn_epoch: int, committed_epoch: int, generation_epoch: int) -> str:
    return EpochLineage(
        turn_epoch=int(turn_epoch),
        committed_epoch=int(committed_epoch),
        generation_epoch=int(generation_epoch),
    ).as_id()


def build_trace_id(*, session_id: str, turn_epoch: int, generation_epoch: int) -> str:
    normalized_session = canonical_session_id(session_id, default="session")
    return f"{normalized_session}-t{int(turn_epoch)}-g{int(generation_epoch)}"


@dataclass(frozen=True, slots=True)
class IdentityClosure:
    session_id: str
    turn_id: str
    epoch_id: str
    event_id: str
    trace_id: str


def build_identity_closure(
    *,
    session_id: object,
    event_id: object,
    turn_id: object = "",
    epoch_id: object = "",
    turn_epoch: int = 0,
    generation_epoch: int = 0,
) -> IdentityClosure:
    canonical_session = canonical_session_id(session_id)
    resolved_turn_id = str(turn_id or "").strip() or canonical_turn_id(session_id=canonical_session, turn_epoch=int(turn_epoch))
    resolved_epoch_id = str(epoch_id or "").strip() or canonical_epoch_id(session_id=canonical_session, generation_epoch=int(generation_epoch))
    resolved_event_id = str(event_id or "").strip()
    resolved_trace_id = build_trace_id(session_id=canonical_session, turn_epoch=int(turn_epoch), generation_epoch=int(generation_epoch))
    return IdentityClosure(
        session_id=canonical_session,
        turn_id=resolved_turn_id,
        epoch_id=resolved_epoch_id,
        event_id=resolved_event_id,
        trace_id=resolved_trace_id,
    )


def validate_identity_closure(closure: IdentityClosure) -> bool:
    return bool(
        str(closure.session_id).strip()
        and str(closure.turn_id).strip()
        and str(closure.epoch_id).strip()
        and str(closure.event_id).strip()
        and str(closure.trace_id).strip()
    )


def validate_lineage_match(
    payload: Mapping[str, object],
    *,
    turn_epoch: int,
    generation_epoch: int,
) -> bool:
    payload_turn = int(payload.get("turn_epoch", -1) or -1)
    payload_generation = int(payload.get("generation_epoch", -1) or -1)
    return payload_turn == int(turn_epoch) and payload_generation == int(generation_epoch)


__all__ = [
    "DEFAULT_SESSION_ID",
    "DEFAULT_EVENT_SCHEMA_VERSION",
    "DEFAULT_TELEMETRY_SCHEMA_VERSION",
    "EpochLineage",
    "IdentityClosure",
    "build_lineage_id",
    "build_identity_closure",
    "build_trace_id",
    "canonical_epoch_id",
    "canonical_response_request_id",
    "canonical_session_id",
    "canonical_turn_id",
    "validate_identity_closure",
    "validate_lineage_match",
]