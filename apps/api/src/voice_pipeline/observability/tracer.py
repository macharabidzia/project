from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Trace:
    records: list[object] = field(default_factory=list)

    def add(self, record: object) -> None:
        self.records.append(record)


__all__ = ["Trace"]
