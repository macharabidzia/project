from __future__ import annotations

from dataclasses import dataclass
from statistics import quantiles


@dataclass(frozen=True, slots=True)
class LatencySummary:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    worst_ms: float


def summarize_latency(samples_ms: list[float] | tuple[float, ...]) -> LatencySummary:
    samples = [float(item) for item in samples_ms]
    if not samples:
        return LatencySummary(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, worst_ms=0.0)
    ordered = sorted(samples)
    p50 = ordered[len(ordered) // 2]
    if len(ordered) < 20:
        p95 = ordered[-1]
        p99 = ordered[-1]
    else:
        p95 = quantiles(ordered, n=100)[94]
        p99 = quantiles(ordered, n=100)[98]
    return LatencySummary(p50_ms=float(p50), p95_ms=float(p95), p99_ms=float(p99), worst_ms=float(ordered[-1]))


__all__ = ["LatencySummary", "summarize_latency"]
