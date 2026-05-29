from __future__ import annotations

import time


def now_ns() -> int:
    return int(time.perf_counter_ns())


def ns_to_ms(value_ns: int) -> float:
    return float(value_ns) / 1_000_000.0


__all__ = ["now_ns", "ns_to_ms"]
