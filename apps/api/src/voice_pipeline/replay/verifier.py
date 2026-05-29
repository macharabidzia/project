from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

class DeterminismError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReplayRun:
    event_log: tuple[dict[str, Any], ...]
    kernel_state_timeline: tuple[dict[str, Any], ...]
    token_stream: tuple[bytes, ...]
    pcm_stream: tuple[bytes, ...]


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def verify_deterministic_replay(left: ReplayRun, right: ReplayRun) -> None:
    if left.event_log != right.event_log:
        raise DeterminismError("event log divergence")

    if len(left.kernel_state_timeline) != len(right.kernel_state_timeline):
        raise DeterminismError("kernel timeline length mismatch")

    for idx, (lhs, rhs) in enumerate(zip(left.kernel_state_timeline, right.kernel_state_timeline)):
        if _canonical_hash(lhs) != _canonical_hash(rhs):
            raise DeterminismError(f"kernel state divergence at tick {idx}")

    if tuple(left.token_stream) != tuple(right.token_stream):
        raise DeterminismError("token stream divergence")
    if tuple(left.pcm_stream) != tuple(right.pcm_stream):
        raise DeterminismError("pcm stream divergence")


__all__ = ["DeterminismError", "ReplayRun", "verify_deterministic_replay"]
