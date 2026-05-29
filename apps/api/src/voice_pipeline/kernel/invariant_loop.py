from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from voice_pipeline.bus.ring_types import RingSlot


@dataclass(frozen=True, slots=True)
class InvariantSnapshot:
    epoch: str
    tick_id: int
    ring_depth: int


def _no_partial_slot(slots: Iterable[RingSlot]) -> bool:
    # In this Python runtime, slots are immutable dataclasses and considered committed on push.
    for slot in slots:
        if int(slot.size) < 0 or int(slot.sequence_no) < 0:
            return False
    return True


def _no_stale_epoch(slots: Iterable[RingSlot], epoch: str) -> bool:
    current = str(epoch).strip()
    for slot in slots:
        if slot.epoch_id and str(slot.epoch_id).strip() != current:
            return False
    return True


def pre_tick_validate(*, epoch: str, ring_depth: int, slots: Iterable[RingSlot]) -> None:
    if int(ring_depth) < 0:
        raise RuntimeError("KERNEL_INVARIANT_FAILURE:ring_depth_negative")
    if int(ring_depth) > 3:
        raise RuntimeError("KERNEL_INVARIANT_FAILURE:ring_depth_overflow")
    if not _no_partial_slot(slots):
        raise RuntimeError("KERNEL_INVARIANT_FAILURE:partial_slot")
    if not _no_stale_epoch(slots, epoch):
        raise RuntimeError("KERNEL_INVARIANT_FAILURE:stale_epoch_slot")


def post_tick_validate(*, epoch: str, result_epoch: str, last_seq: int, prev_seq: int) -> None:
    if str(result_epoch).strip() != str(epoch).strip():
        raise RuntimeError("KERNEL_INVARIANT_FAILURE:epoch_drift")
    if int(last_seq) < int(prev_seq):
        raise RuntimeError("KERNEL_INVARIANT_FAILURE:non_monotonic_sequence")


__all__ = ["InvariantSnapshot", "post_tick_validate", "pre_tick_validate"]

