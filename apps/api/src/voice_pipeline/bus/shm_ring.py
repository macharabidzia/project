from __future__ import annotations

from collections.abc import Callable
from time import perf_counter_ns
from typing import Generic, TypeVar

from voice_pipeline.bus.ring_types import EventType, RingSlot


T = TypeVar("T")


class SharedMemoryRing(Generic[T]):
    """In-process bounded ring used by the single-runtime voice pipeline."""

    def __init__(self, size: int, *, slot_bytes: int = 4096) -> None:
        self._capacity = max(1, int(size))
        self._items: list[T | None] = [None] * self._capacity
        self._head = 0
        self._tail = 0
        self._count = 0
        self._overwrite_count = 0
        self._slot_bytes = max(64, int(slot_bytes))
        self._payload_index = 0
        self._payloads: list[bytes] = [b""] * self._capacity
        self._enqueued_ns: list[int] = [0] * self._capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def depth(self) -> int:
        return self._count

    @property
    def overwrite_count(self) -> int:
        return self._overwrite_count

    @property
    def shared_memory_name(self) -> str:
        return ""

    @property
    def oldest_age_ms(self) -> float:
        if self._count <= 0:
            return 0.0
        oldest_ns = int(self._enqueued_ns[self._head])
        if oldest_ns <= 0:
            return 0.0
        return float(max(0, perf_counter_ns() - oldest_ns)) / 1_000_000.0

    def push(self, item: T) -> bool:
        if self._count >= self._capacity:
            self._items[self._head] = None
            self._enqueued_ns[self._head] = 0
            self._head = (self._head + 1) % self._capacity
            self._count -= 1
            self._overwrite_count += 1
        self._items[self._tail] = item
        self._enqueued_ns[self._tail] = perf_counter_ns()
        self._tail = (self._tail + 1) % self._capacity
        self._count += 1
        return True

    def pop(self) -> T | None:
        if self._count <= 0:
            return None
        item = self._items[self._head]
        self._items[self._head] = None
        self._enqueued_ns[self._head] = 0
        self._head = (self._head + 1) % self._capacity
        self._count -= 1
        return item

    def drain(self, consumer: Callable[[T], None]) -> int:
        drained = 0
        while True:
            item = self.pop()
            if item is None:
                return drained
            consumer(item)
            drained += 1

    def push_bytes(
        self,
        *,
        event_type: EventType,
        payload: bytes,
        lineage_id: str,
        sequence_no: int,
        epoch_id: str = "",
        flags: int = 0,
        metadata: tuple[tuple[str, object], ...] = (),
    ) -> RingSlot:
        body = bytes(payload)
        if len(body) > self._slot_bytes:
            raise ValueError("payload larger than slot bytes")
        slot_index = self._payload_index % self._capacity
        self._payloads[slot_index] = body
        slot = RingSlot(
            event_type=event_type,
            ptr=int(slot_index),
            size=len(body),
            lineage_id=str(lineage_id),
            sequence_no=int(sequence_no),
            epoch_id=str(epoch_id),
            flags=int(flags),
            metadata=tuple(metadata),
        )
        self._payload_index += 1
        self.push(slot)  # type: ignore[arg-type]
        return slot

    def read_slot_bytes(self, slot: RingSlot) -> bytes:
        slot_index = int(slot.ptr) % self._capacity
        return bytes(self._payloads[slot_index][: int(slot.size)])

    def close(self) -> None:
        self._payloads = [b""] * self._capacity
        self._enqueued_ns = [0] * self._capacity


__all__ = ["SharedMemoryRing"]
