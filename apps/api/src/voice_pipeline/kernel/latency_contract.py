from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LaneAction = Literal["ok", "throttle", "drop", "degrade_tts", "pause_llm"]


@dataclass(frozen=True, slots=True)
class LaneLatencyBudget:
    asr_ms: float = 40.0
    kernel_ms: float = 10.0
    vllm_ms: float = 60.0
    tts_ms: float = 80.0
    transport_ms: float = 10.0
    total_ms: float = 150.0
    max_queue_depth: int = 3


@dataclass(frozen=True, slots=True)
class PipelineLatencySample:
    asr_ms: float = 0.0
    kernel_ms: float = 0.0
    vllm_ms: float = 0.0
    tts_ms: float = 0.0
    transport_ms: float = 0.0
    queue_depth: int = 0

    @property
    def total_ms(self) -> float:
        return float(self.asr_ms) + float(self.kernel_ms) + float(self.vllm_ms) + float(self.tts_ms) + float(self.transport_ms)


@dataclass(frozen=True, slots=True)
class LatencyDecision:
    action: LaneAction = "ok"
    reason: str = ""
    over_budget_ms: float = 0.0


def evaluate_latency(
    sample: PipelineLatencySample,
    *,
    budget: LaneLatencyBudget | None = None,
) -> LatencyDecision:
    resolved_budget = budget or LaneLatencyBudget()
    total_ms = float(sample.total_ms)
    queue_depth = int(sample.queue_depth)
    queue_overflow = queue_depth > int(resolved_budget.max_queue_depth)

    stage_overages = {
        "asr": max(0.0, float(sample.asr_ms) - float(resolved_budget.asr_ms)),
        "kernel": max(0.0, float(sample.kernel_ms) - float(resolved_budget.kernel_ms)),
        "vllm": max(0.0, float(sample.vllm_ms) - float(resolved_budget.vllm_ms)),
        "tts": max(0.0, float(sample.tts_ms) - float(resolved_budget.tts_ms)),
        "transport": max(0.0, float(sample.transport_ms) - float(resolved_budget.transport_ms)),
    }
    total_overage = max(0.0, total_ms - float(resolved_budget.total_ms))

    if queue_overflow:
        if total_overage > 0.0:
            return LatencyDecision(
                action="pause_llm",
                reason=f"queue depth {queue_depth} exceeds {int(resolved_budget.max_queue_depth)} and total budget is over by {total_overage:.2f}ms",
                over_budget_ms=total_overage,
            )
        return LatencyDecision(
            action="throttle",
            reason=f"queue depth {queue_depth} exceeds {int(resolved_budget.max_queue_depth)}",
            over_budget_ms=float(queue_depth - int(resolved_budget.max_queue_depth)),
        )

    if stage_overages["tts"] > 0.0:
        return LatencyDecision(
            action="degrade_tts",
            reason=f"tts exceeded budget by {stage_overages['tts']:.2f}ms",
            over_budget_ms=stage_overages["tts"],
        )

    if stage_overages["vllm"] > 0.0:
        return LatencyDecision(
            action="pause_llm",
            reason=f"vllm exceeded budget by {stage_overages['vllm']:.2f}ms",
            over_budget_ms=stage_overages["vllm"],
        )

    if stage_overages["asr"] > 0.0 or stage_overages["kernel"] > 0.0 or stage_overages["transport"] > 0.0:
        dominant = max(stage_overages.items(), key=lambda item: item[1])
        return LatencyDecision(
            action="throttle",
            reason=f"{dominant[0]} exceeded budget by {dominant[1]:.2f}ms",
            over_budget_ms=dominant[1],
        )

    if total_overage > 0.0:
        return LatencyDecision(
            action="throttle",
            reason=f"pipeline exceeded total budget by {total_overage:.2f}ms",
            over_budget_ms=total_overage,
        )

    return LatencyDecision(action="ok", reason="within budget", over_budget_ms=0.0)


__all__ = [
    "LaneAction",
    "LaneLatencyBudget",
    "LatencyDecision",
    "PipelineLatencySample",
    "evaluate_latency",
]
