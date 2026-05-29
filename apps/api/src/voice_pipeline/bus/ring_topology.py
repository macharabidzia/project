from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from voice_pipeline.bus.ring_types import EventType, RingSlot
from voice_pipeline.bus.shm_ring import SharedMemoryRing


class _LaneRingView:
    def __init__(self, shared_ring: SharedMemoryRing[RingSlot], *, lane_event_types: set[EventType]) -> None:
        self._shared_ring = shared_ring
        self._accepted = set(lane_event_types)
        self._local = deque()

    def push(self, slot: RingSlot) -> bool:
        if slot.event_type not in self._accepted:
            raise ValueError("slot event type does not belong to lane")
        return self._shared_ring.push(slot)

    def pop(self) -> RingSlot | None:
        if self._local:
            return self._local.popleft()

        scanned = 0
        max_scan = max(1, int(self._shared_ring.capacity))
        while scanned < max_scan:
            scanned += 1
            slot = self._shared_ring.pop()
            if slot is None:
                return None
            if slot.event_type in self._accepted:
                return slot
            self._local.append(slot)

        return None


@dataclass(slots=True)
class RingTopology:
    kernel_stream_ring: SharedMemoryRing[RingSlot]
    asr_ring: _LaneRingView
    vllm_ring: _LaneRingView
    tts_ring: _LaneRingView
    pcm_ring: _LaneRingView

    @classmethod
    def with_capacity(
        cls,
        *,
        asr: int = 1024,
        vllm: int = 1024,
        tts: int = 1024,
        pcm: int = 1024,
        slot_bytes: int = 4096,
    ) -> "RingTopology":
        capacity = max(int(asr), int(vllm), int(tts), int(pcm))
        kernel_stream_ring: SharedMemoryRing[RingSlot] = SharedMemoryRing(size=capacity, slot_bytes=int(slot_bytes))
        return cls(
            kernel_stream_ring=kernel_stream_ring,
            asr_ring=_LaneRingView(kernel_stream_ring, lane_event_types={EventType.ASR_SLOT}),
            vllm_ring=_LaneRingView(
                kernel_stream_ring,
                lane_event_types={EventType.VLLM_REQUEST_SLOT, EventType.VLLM_TOKEN_SLOT},
            ),
            tts_ring=_LaneRingView(
                kernel_stream_ring,
                lane_event_types={EventType.TTS_REQUEST_SLOT},
            ),
            pcm_ring=_LaneRingView(
                kernel_stream_ring,
                lane_event_types={EventType.PCM_SLOT},
            ),
        )


__all__ = ["RingTopology"]



