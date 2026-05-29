from __future__ import annotations

from dataclasses import dataclass

from voice_pipeline.kernel.state import KernelState
from voice_pipeline.shared.lineage import canonical_epoch_id


@dataclass(frozen=True, slots=True)
class EpochLease:
    epoch_id: str
    turn_id: str
    version: int


def epoch_id_for_state(state: KernelState) -> str:
    return canonical_epoch_id(
        session_id=state.session_id,
        generation_epoch=int(state.generation_index),
    )


def lease_snapshot(state: KernelState) -> EpochLease:
    return EpochLease(
        epoch_id=epoch_id_for_state(state),
        turn_id=str(state.output.active_turn_id or "").strip(),
        version=int(state.output.version),
    )


__all__ = ["EpochLease", "epoch_id_for_state", "lease_snapshot"]
