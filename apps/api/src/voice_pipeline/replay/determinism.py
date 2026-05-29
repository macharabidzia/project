from __future__ import annotations

import hashlib
import json
from typing import Any

from voice_pipeline.replay.validator import assert_state_equal
from voice_pipeline.replay.verifier import ReplayRun, verify_deterministic_replay


def verify_replay(left: object, right: object) -> None:
    assert_state_equal(left, right)


def verify_replay_runs(left: ReplayRun, right: ReplayRun) -> None:
    verify_deterministic_replay(left, right)


def canonical_state_hash(state: Any) -> str:
    payload = json.dumps(
        state if isinstance(state, dict) else {"state": repr(state)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def canonical_event_stream_hash(events: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    payload = json.dumps(list(events), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["canonical_event_stream_hash", "canonical_state_hash", "verify_replay", "verify_replay_runs"]
