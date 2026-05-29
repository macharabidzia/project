from __future__ import annotations

from voice_pipeline.shared.types import AuthorityEvent
from voice_pipeline.shared.lineage import build_identity_closure, validate_identity_closure


def assert_state_equal(left: object, right: object) -> None:
    if left != right:
        raise AssertionError("state mismatch")


def assert_event_identity_closure(event: AuthorityEvent) -> None:
    closure = build_identity_closure(
        session_id=event.session_id,
        event_id=event.event_id,
        turn_id=event.observations.get("turn_id", ""),
        epoch_id=event.observations.get("epoch_id", event.lineage_id),
        turn_epoch=int(event.payload.get("turn_epoch", 0) or 0),
        generation_epoch=int(event.payload.get("generation_epoch", 0) or 0),
    )
    if not validate_identity_closure(closure):
        raise AssertionError("event identity closure is incomplete")


__all__ = ["assert_event_identity_closure", "assert_state_equal"]

