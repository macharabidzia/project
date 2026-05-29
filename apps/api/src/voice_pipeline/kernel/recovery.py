from __future__ import annotations

from dataclasses import dataclass

from voice_pipeline.kernel.leases import EpochLease, lease_snapshot
from voice_pipeline.kernel.state import KernelState, RecoveryStatus


@dataclass(frozen=True, slots=True)
class RecoverySnapshot:
    kernel_state: KernelState
    lease: EpochLease
    lease_table: tuple[EpochLease, ...]
    ordering_state: dict[str, int]
    cursor_positions: dict[str, int]
    sequence_number: int


def build_recovery_snapshot(state: KernelState) -> RecoverySnapshot:
    return RecoverySnapshot(
        kernel_state=state,
        lease=lease_snapshot(state),
        lease_table=(lease_snapshot(state),),
        ordering_state={"last_sequence_no": int(state.last_sequence_no)},
        cursor_positions={
            "audio_cursor": int(state.turn_index),
            "transcript_cursor": int(state.committed_turn_index),
            "token_cursor": int(len(state.output.vllm_tokens)),
            "pcm_cursor": int(len(state.output.emitted_audio_chunk_ids)),
        },
        sequence_number=int(state.last_sequence_no),
    )


def recovering_status(reason: str) -> RecoveryStatus:
    return RecoveryStatus(state="recovering", reason=str(reason or ""))


def recovered_status(reason: str) -> RecoveryStatus:
    return RecoveryStatus(state="recovered", reason=str(reason or ""))


__all__ = [
    "RecoverySnapshot",
    "build_recovery_snapshot",
    "recovered_status",
    "recovering_status",
]
