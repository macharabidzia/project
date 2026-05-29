from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voice_pipeline.shared.types import AuthorityEvent, new_authority_event


@dataclass(slots=True)
class EventLog:
    events: list[object] = field(default_factory=list)

    def append(self, event: object) -> None:
        self.events.append(event)

    def append_runtime_event(self, event: Any) -> None:
        self.events.append(
            {
                "ts": int(event.ts),
                "type": str(event.type),
                "node": str(event.node),
                "lineage": str(event.lineage),
                "payload": dict(event.payload),
            }
        )

    def as_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for event in self.events:
            if isinstance(event, dict):
                records.append(dict(event))
            else:
                records.append({"event": repr(event)})
        return records

    def replay_into_kernel(self, *, session_id: str, apply_event: Any) -> None:
        for record in self.as_records():
            if record.get("event"):
                continue
            event_type = str(record.get("type", "")).strip()
            if not event_type:
                continue
            payload = dict(record.get("payload", {}))
            sequence_no = int(payload.get("sequence_no", 1) or 1)
            lineage_id = str(record.get("lineage", "") or payload.get("epoch_id", "")).strip()
            event: AuthorityEvent = new_authority_event(
                event_type=event_type,
                session_id=str(session_id),
                sequence_no=sequence_no,
                lineage_id=lineage_id,
                payload=payload,
                event_id=str(payload.get("event_id", "")).strip() or None,
                observations={
                    "session_id": str(session_id),
                    "turn_id": str(payload.get("turn_id", "") or ""),
                    "epoch_id": str(payload.get("epoch_id", lineage_id) or lineage_id),
                    "event_id": str(payload.get("event_id", "") or ""),
                    "trace_id": str(payload.get("trace_id", "") or ""),
                },
            )
            apply_event(event)


__all__ = ["EventLog"]

