from __future__ import annotations

from voice_pipeline.replay.determinism import canonical_event_stream_hash, canonical_state_hash
from voice_pipeline.replay.verifier import ReplayRun, verify_deterministic_replay


def test_canonical_hashes_are_stable_for_equal_payloads() -> None:
    left = {"phase": "idle", "version": 1}
    right = {"version": 1, "phase": "idle"}
    assert canonical_state_hash(left) == canonical_state_hash(right)
    assert canonical_event_stream_hash(({"type": "a"}, {"type": "b"})) == canonical_event_stream_hash(
        ({"type": "a"}, {"type": "b"})
    )


def test_replay_runs_match_when_event_state_and_streams_match() -> None:
    run = ReplayRun(
        event_log=({"type": "ASRFinalReceived"},),
        kernel_state_timeline=({"phase": "idle", "version": 1},),
        token_stream=(b"ok",),
        pcm_stream=(b"\x00\x01",),
    )
    verify_deterministic_replay(run, run)
