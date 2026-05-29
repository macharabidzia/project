from __future__ import annotations

import asyncio

from voice_pipeline.bus.shm_ring import SharedMemoryRing
from voice_pipeline.transport.pcm_clock import PCMClockSender, PCMFrame


def test_shared_ring_exposes_bounded_mechanical_metrics() -> None:
    ring: SharedMemoryRing[int] = SharedMemoryRing(size=2)
    assert ring.capacity == 2
    assert ring.depth == 0
    assert ring.overwrite_count == 0
    assert ring.oldest_age_ms == 0.0

    ring.push(1)
    ring.push(2)
    ring.push(3)

    assert ring.depth == 2
    assert ring.overwrite_count == 1
    assert ring.oldest_age_ms >= 0.0


def test_pcm_clock_drops_stale_lineage_before_emit() -> None:
    sender = PCMClockSender(tick_ms=1, target_buffer_frames=1, max_buffer_frames=3)
    sender.enqueue(PCMFrame(pcm=b"stale", sample_rate=24_000, epoch_id="s:epoch:1", output_version=1))
    sender.enqueue(PCMFrame(pcm=b"fresh", sample_rate=24_000, epoch_id="s:epoch:2", output_version=2))

    sent: list[tuple[bytes, int]] = []

    async def _send(pcm: bytes, rate: int) -> None:
        sent.append((bytes(pcm), int(rate)))

    asyncio.run(
        sender.run_once(
            _send,
            current_epoch_id="s:epoch:2",
            current_output_version=2,
            silence_frame=b"",
            silence_sample_rate=24_000,
        )
    )

    assert sent == [(b"fresh", 24_000)]
    assert sender.dropped_stale_frames == 1
