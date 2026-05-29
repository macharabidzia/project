from __future__ import annotations

from voice_pipeline.kernel.latency_contract import LaneLatencyBudget, PipelineLatencySample, evaluate_latency


def test_latency_contract_returns_ok_when_within_budget() -> None:
    decision = evaluate_latency(
        PipelineLatencySample(
            asr_ms=20.0,
            kernel_ms=5.0,
            vllm_ms=30.0,
            tts_ms=40.0,
            transport_ms=5.0,
            queue_depth=1,
        ),
        budget=LaneLatencyBudget(total_ms=150.0, max_queue_depth=3),
    )

    assert decision.action == "ok"
    assert decision.over_budget_ms == 0.0


def test_latency_contract_throttles_or_pauses_on_pressure() -> None:
    decision = evaluate_latency(
        PipelineLatencySample(
            asr_ms=20.0,
            kernel_ms=12.0,
            vllm_ms=80.0,
            tts_ms=92.0,
            transport_ms=15.0,
            queue_depth=5,
        ),
        budget=LaneLatencyBudget(total_ms=150.0, max_queue_depth=3),
    )

    assert decision.action in {"pause_llm", "throttle", "degrade_tts"}
    assert decision.over_budget_ms > 0.0
