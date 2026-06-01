from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import asyncio
import inspect

from voice_pipeline.shared.time import now_ns, ns_to_ms


@dataclass(frozen=True, slots=True)
class PCMFrame:
    pcm: bytes
    sample_rate: int
    epoch_id: str
    output_version: int
    request_id: str = ""


class PCMClockSender:
    def __init__(
        self,
        *,
        tick_ms: int = 20,
        target_buffer_frames: int = 2,
        max_buffer_frames: int = 3,
    ) -> None:
        self.tick_ns = max(1, int(tick_ms)) * 1_000_000
        self.target_buffer_frames = max(1, int(target_buffer_frames))
        self.max_buffer_frames = max(self.target_buffer_frames, int(max_buffer_frames))
        self._frames: deque[PCMFrame] = deque()
        self._enqueued_ns: deque[int] = deque()
        self.dropped_overflow_frames = 0
        self.dropped_stale_frames = 0
        self.last_tick_elapsed_ms = 0.0
        self.last_tick_jitter_ms = 0.0

    @property
    def depth(self) -> int:
        return len(self._frames)

    def head_lease(self) -> tuple[str, int] | None:
        if not self._frames:
            return None
        frame = self._frames[0]
        return str(frame.epoch_id), int(frame.output_version)

    def enqueue(self, frame: PCMFrame) -> bool:
        if len(self._frames) >= self.max_buffer_frames:
            self._frames.popleft()
            if self._enqueued_ns:
                self._enqueued_ns.popleft()
            self.dropped_overflow_frames += 1
        self._frames.append(frame)
        self._enqueued_ns.append(now_ns())
        return True

    def clear(self) -> None:
        self._frames.clear()
        self._enqueued_ns.clear()

    def _pop_fresh(self, *, current_epoch_id: str, current_output_version: int) -> PCMFrame | None:
        while self._frames:
            candidate = self._frames.popleft()
            if self._enqueued_ns:
                self._enqueued_ns.popleft()
            if str(candidate.epoch_id) != str(current_epoch_id) or int(candidate.output_version) != int(current_output_version):
                self.dropped_stale_frames += 1
                continue
            return candidate
        return None

    @property
    def oldest_age_ms(self) -> float:
        if not self._enqueued_ns:
            return 0.0
        return ns_to_ms(max(0, now_ns() - int(self._enqueued_ns[0])))

    async def run_once(
        self,
        send_fn: Callable[..., Awaitable[None]],
        *,
        current_epoch_id: str,
        current_output_version: int,
        silence_frame: bytes = b"",
        silence_sample_rate: int = 24_000,
    ) -> str:
        started_ns = now_ns()
        frame = self._pop_fresh(current_epoch_id=str(current_epoch_id), current_output_version=int(current_output_version))
        if frame is None:
            await _call_send_fn(send_fn, bytes(silence_frame), int(silence_sample_rate), "")
            sent_request_id = ""
        else:
            await _call_send_fn(send_fn, bytes(frame.pcm), int(frame.sample_rate), str(frame.request_id))
            sent_request_id = str(frame.request_id)

        elapsed_ns = now_ns() - started_ns
        remaining_ns = self.tick_ns - elapsed_ns
        if remaining_ns > 0:
            await asyncio.sleep(remaining_ns / 1_000_000_000.0)
        total_ns = now_ns() - started_ns
        self.last_tick_elapsed_ms = ns_to_ms(total_ns)
        self.last_tick_jitter_ms = abs(self.last_tick_elapsed_ms - (float(self.tick_ns) / 1_000_000.0))
        return sent_request_id


async def _call_send_fn(
    send_fn: Callable[..., Awaitable[None]],
    pcm: bytes,
    sample_rate: int,
    request_id: str,
) -> None:
    try:
        parameter_count = len(inspect.signature(send_fn).parameters)
    except (TypeError, ValueError):
        parameter_count = 3
    if parameter_count <= 2:
        await send_fn(pcm, sample_rate)
        return
    await send_fn(pcm, sample_rate, request_id)


__all__ = ["PCMClockSender", "PCMFrame"]
