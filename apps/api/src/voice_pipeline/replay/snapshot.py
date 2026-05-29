from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Snapshot:
    sequence_no: int
    state: object


__all__ = ["Snapshot"]
