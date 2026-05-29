from __future__ import annotations

import time

from voice_pipeline.shared.time import now_ns


class TickEngine:
    def __init__(self, *, interval_ms: int) -> None:
        clamped_interval_ms = max(1, min(5, int(interval_ms)))
        self.interval_ns = clamped_interval_ms * 1_000_000
        self.target_interval_ns = self.interval_ns
        self.last_elapsed_ns = 0
        self.last_drift_us = 0.0
        self.max_drift_us = 0.0
        self.worst_interrupt_latency_us = 0.0
        self.catch_up_ticks = 0
        self.drift_correction_ns = 0
        self._last_loop_end_ns = 0

    def run_once(self, apply_fn) -> int:
        started_ns = now_ns()
        if self._last_loop_end_ns > 0:
            observed_interrupt_ns = max(0, started_ns - self._last_loop_end_ns)
            interrupt_budget_ns = self.target_interval_ns
            interrupt_overrun_ns = max(0, observed_interrupt_ns - interrupt_budget_ns)
            interrupt_overrun_us = float(interrupt_overrun_ns) / 1_000.0
            self.worst_interrupt_latency_us = max(self.worst_interrupt_latency_us, interrupt_overrun_us)

        apply_fn()
        elapsed = now_ns() - started_ns
        # Drift correction formula:
        # corrected_remaining = target_interval - elapsed - correction_accumulator
        remaining = self.target_interval_ns - elapsed - self.drift_correction_ns

        if remaining > 0:
            time.sleep(remaining / 1_000_000_000.0)
            self.drift_correction_ns = 0
        else:
            # Catch-up rule: when behind budget, do not sleep this cycle and
            # carry the overrun into the next cycle correction accumulator.
            self.catch_up_ticks += 1
            self.drift_correction_ns = min(self.target_interval_ns, abs(int(remaining)))

        total_elapsed = now_ns() - started_ns
        self._last_loop_end_ns = started_ns + total_elapsed
        self.last_elapsed_ns = int(total_elapsed)
        self.last_drift_us = abs(float(total_elapsed - self.target_interval_ns)) / 1_000.0
        self.max_drift_us = max(self.max_drift_us, self.last_drift_us)
        return total_elapsed

    def drift_alarm_triggered(self, *, threshold_us: float = 200.0) -> bool:
        return self.last_drift_us > float(threshold_us)

    def drift_snapshot(self) -> dict[str, float]:
        return {
            "target_tick_ms": float(self.target_interval_ns) / 1_000_000.0,
            "last_elapsed_ms": float(self.last_elapsed_ns) / 1_000_000.0,
            "last_drift_us": float(self.last_drift_us),
            "max_drift_us": float(self.max_drift_us),
            "worst_interrupt_latency_us": float(self.worst_interrupt_latency_us),
            "catch_up_ticks": float(self.catch_up_ticks),
            "drift_correction_ns": float(self.drift_correction_ns),
        }


__all__ = ["TickEngine"]
