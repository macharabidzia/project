from __future__ import annotations


def timeline(events: list[object] | tuple[object, ...]) -> tuple[object, ...]:
    return tuple(events)


__all__ = ["timeline"]
