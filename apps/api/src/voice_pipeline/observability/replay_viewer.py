from __future__ import annotations

from voice_pipeline.replay.event_log import EventLog


def view_replay(log: EventLog) -> tuple[object, ...]:
    return tuple(log.events)


__all__ = ["view_replay"]
