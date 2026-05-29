from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from voice_pipeline.kernel.state import KernelState
from voice_pipeline.shared.types import AuthorityEvent, new_authority_event


@dataclass(frozen=True, slots=True)
class OrderedEventKey:
    global_seq: int
    session_seq: int
    lane_priority: int
    timestamp_ns: int


def ordering_key(
    *,
    global_seq: int,
    session_seq: int,
    lane_priority: int,
    timestamp_ns: int,
) -> OrderedEventKey:
    return OrderedEventKey(
        global_seq=int(global_seq),
        session_seq=int(session_seq),
        lane_priority=int(lane_priority),
        timestamp_ns=int(timestamp_ns),
    )


def expected_next_sequence(state: KernelState) -> int:
    return int(state.last_sequence_no) + 1


def make_derived_event(
    *,
    session_id: str,
    state: KernelState,
    parent_event: AuthorityEvent,
    derived_event,
    sequence_no: int | None = None,
) -> AuthorityEvent:
    event_type = str(derived_event.event_type)
    resolved_sequence_no = expected_next_sequence(state) if sequence_no is None else int(sequence_no)
    return new_authority_event(
        event_type=event_type,
        session_id=str(session_id),
        sequence_no=resolved_sequence_no,
        lineage_id=str(derived_event.lineage_id or state.lineage_id or parent_event.lineage_id).strip(),
        payload=dict(derived_event.payload),
        causation_id=parent_event.event_id,
        event_id=f"{session_id}:{resolved_sequence_no}:{event_type}:derived:{parent_event.event_id}",
    )


def push_front(queue: deque[AuthorityEvent], events: Iterable[AuthorityEvent]) -> None:
    for event in reversed(tuple(events)):
        queue.appendleft(event)


__all__ = [
    "OrderedEventKey",
    "expected_next_sequence",
    "make_derived_event",
    "ordering_key",
    "push_front",
]
