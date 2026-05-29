from __future__ import annotations

from voice_pipeline.kernel.tts_fragment_planner import TTSFragmentPlannerConfig, plan_tts_fragment


def test_tts_fragment_planner_starts_early_on_stable_prefix() -> None:
    plan = plan_tts_fragment(
        ["hello"],
        stable_prefix="hello",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=6, context_window_tokens=24),
    )

    assert plan.start_now is True
    assert plan.flush_text == "hello"
    assert plan.remaining_tokens == ()


def test_tts_fragment_planner_keeps_overlap_window_when_fragment_grows() -> None:
    plan = plan_tts_fragment(
        ["hello", "world", "again"],
        stable_prefix="hello",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=2, context_window_tokens=2),
    )

    assert plan.start_now is True
    assert plan.flush_text == "hello world"
    assert plan.remaining_tokens == ("again",)
