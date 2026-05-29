from __future__ import annotations

import asyncio

from voice_pipeline.transport.pcm_clock import PCMClockSender, PCMFrame


def test_pcm_clock_drops_stale_epoch_and_output_version_before_emit() -> None:
    sender = PCMClockSender(tick_ms=1, target_buffer_frames=1, max_buffer_frames=3)
    sender.enqueue(PCMFrame(pcm=b"old-epoch", sample_rate=24_000, epoch_id="session:epoch:1", output_version=1))
    sender.enqueue(PCMFrame(pcm=b"old-version", sample_rate=24_000, epoch_id="session:epoch:2", output_version=1))
    sender.enqueue(PCMFrame(pcm=b"fresh", sample_rate=24_000, epoch_id="session:epoch:2", output_version=2))

    sent: list[tuple[bytes, int]] = []

    async def _send(pcm: bytes, sample_rate: int) -> None:
        sent.append((bytes(pcm), int(sample_rate)))

    asyncio.run(
        sender.run_once(
            _send,
            current_epoch_id="session:epoch:2",
            current_output_version=2,
            silence_frame=b"",
            silence_sample_rate=24_000,
        )
    )

    assert sent == [(b"fresh", 24_000)]
    assert sender.dropped_stale_frames == 2


def test_pcm_clock_is_bounded_and_counts_overflow() -> None:
    sender = PCMClockSender(tick_ms=1, target_buffer_frames=1, max_buffer_frames=2)
    sender.enqueue(PCMFrame(pcm=b"frame-1", sample_rate=24_000, epoch_id="session:epoch:1", output_version=1))
    sender.enqueue(PCMFrame(pcm=b"frame-2", sample_rate=24_000, epoch_id="session:epoch:1", output_version=1))
    sender.enqueue(PCMFrame(pcm=b"frame-3", sample_rate=24_000, epoch_id="session:epoch:1", output_version=1))

    assert sender.depth == 2
    assert sender.dropped_overflow_frames == 1

