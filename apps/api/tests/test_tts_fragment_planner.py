from __future__ import annotations

from voice_pipeline.kernel.tts_fragment_planner import TTSFragmentPlannerConfig, plan_tts_fragment


def test_tts_fragment_planner_waits_for_min_tokens_even_with_stable_prefix() -> None:
    plan = plan_tts_fragment(
        ["hello"],
        stable_prefix="hello",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=6, context_window_tokens=24),
    )

    assert plan.start_now is False
    assert plan.flush_text == ""
    assert plan.remaining_tokens == ()


def test_tts_fragment_planner_flushes_single_token_first_fragment_in_low_latency_mode() -> None:
    plan = plan_tts_fragment(
        ["hello"],
        stable_prefix="hello",
        config=TTSFragmentPlannerConfig(min_tokens=1, max_tokens=4, context_window_tokens=24),
    )

    assert plan.start_now is True
    assert plan.flush_text == "hello"
    assert plan.remaining_tokens == ()
    assert plan.stream_fragment is True
    assert plan.reason == "low_latency"


def test_tts_fragment_planner_flushes_short_reply_early_for_low_latency() -> None:
    plan = plan_tts_fragment(
        ["hello", "world"],
        stable_prefix="hello",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=6, context_window_tokens=24),
    )

    assert plan.start_now is True
    assert plan.flush_text == "hello world"
    assert plan.remaining_tokens == ()
    assert plan.stream_fragment is False
    assert plan.reason == "low_latency"


def test_tts_fragment_planner_flushes_boundary_even_if_only_tiny_tail_remains() -> None:
    plan = plan_tts_fragment(
        ["He", "keeps", "pushing", "for", "it", ".", "Let me know if"],
        stable_prefix="he insists about it",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=24, context_window_tokens=24),
    )

    assert plan.start_now is True
    assert plan.flush_text == "He keeps pushing for it ."
    assert plan.remaining_tokens == ("Let me know if",)
    assert plan.stream_fragment is True
    assert plan.reason == "window_overlap"


def test_tts_fragment_planner_flushes_first_window_immediately() -> None:
    plan = plan_tts_fragment(
        ["He", "keeps", "pushing", "for", "it", "."],
        stable_prefix="he insists about it",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=6, context_window_tokens=24),
    )

    assert plan.start_now is True
    assert plan.flush_text == "He keeps pushing for it ."
    assert plan.remaining_tokens == ()
    assert plan.stream_fragment is False
    assert plan.reason == "low_latency"


def test_tts_fragment_planner_waits_for_short_sentence_final_reply_inside_first_window() -> None:
    plan = plan_tts_fragment(
        ["Hey", "!", "What", "'s", "up", "?"],
        stable_prefix="hi",
        config=TTSFragmentPlannerConfig(min_tokens=6, max_tokens=8, context_window_tokens=24),
    )

    assert plan.start_now is False
    assert plan.flush_text == ""
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


def test_tts_fragment_planner_ignores_tiny_early_boundary_when_window_is_full() -> None:
    plan = plan_tts_fragment(
        ["Hey", "!", "What", "'s", "up", "?"],
        stable_prefix="hi",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=4, context_window_tokens=24),
    )

    assert plan.start_now is True
    assert plan.flush_text == "Hey ! What 's up ?"
    assert plan.remaining_tokens == ()
    assert plan.stream_fragment is False
    assert plan.reason == "boundary"


def test_tts_fragment_planner_waits_for_internal_boundary_before_window_fills() -> None:
    plan = plan_tts_fragment(
        ["Hey", "!", "What"],
        stable_prefix="hi",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=4, context_window_tokens=24),
    )

    assert plan.start_now is False
    assert plan.flush_text == ""
    assert plan.remaining_tokens == ()


def test_tts_fragment_planner_waits_for_boundaryless_lexical_opener_without_stable_prefix() -> None:
    early = plan_tts_fragment(
        ["You're", "saying"],
        stable_prefix="",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=4, context_window_tokens=24),
    )

    assert early.start_now is False
    assert early.flush_text == ""
    assert early.remaining_tokens == ()


def test_tts_fragment_planner_flushes_boundaryless_lexical_opener_when_stable_prefix_exists() -> None:
    early = plan_tts_fragment(
        ["You're", "saying"],
        stable_prefix="hi",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=4, context_window_tokens=24),
    )

    assert early.start_now is True
    assert early.flush_text == "You're saying"
    assert early.remaining_tokens == ()
    assert early.stream_fragment is True


def test_tts_fragment_planner_flushes_fuller_boundaryless_phrase() -> None:
    fuller = plan_tts_fragment(
        ["You're", "saying", '"he', 'only" —'],
        stable_prefix="hi",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=4, context_window_tokens=24),
    )

    assert fuller.start_now is True
    assert fuller.flush_text == 'You\'re saying "he only" —'
    assert fuller.remaining_tokens == ()
    assert fuller.stream_fragment is True


def test_tts_fragment_planner_keeps_trailing_punctuation_with_first_clause() -> None:
    plan = plan_tts_fragment(
        ["hey!", "what", "'s", "up", "?"],
        stable_prefix="hi",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=4, context_window_tokens=24),
    )

    assert plan.start_now is True
    assert plan.flush_text == "hey! what 's up ?"
    assert plan.remaining_tokens == ()


def test_tts_fragment_planner_waits_at_full_window_when_internal_boundary_prefix_is_incomplete() -> None:
    plan = plan_tts_fragment(
        ["Hey", "!", "What", "'s"],
        stable_prefix="hi",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=4, context_window_tokens=24),
    )

    assert plan.start_now is False
    assert plan.flush_text == ""
    assert plan.remaining_tokens == ()


def test_tts_fragment_planner_waits_at_full_window_when_tail_is_dangling_function_word() -> None:
    early = plan_tts_fragment(
        ["hello!", "how", "can", "i"],
        stable_prefix="hello there",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=4, context_window_tokens=24),
    )

    assert early.start_now is False
    assert early.flush_text == ""
    assert early.remaining_tokens == ()

    fuller = plan_tts_fragment(
        ["hello!", "how", "can", "i", "help"],
        stable_prefix="hello there",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=4, context_window_tokens=24),
    )

    assert fuller.start_now is True
    assert fuller.flush_text == "hello! how can i help"
    assert fuller.remaining_tokens == ()
    assert fuller.stream_fragment is True


def test_tts_fragment_planner_includes_one_extra_token_for_contraction_split_first_clause() -> None:
    plan = plan_tts_fragment(
        ["Hey", "!", "What", "'s", "up"],
        stable_prefix="hi",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=4, context_window_tokens=24),
    )

    assert plan.start_now is True
    assert plan.flush_text == "Hey ! What 's up"
    assert plan.remaining_tokens == ()
    assert plan.stream_fragment is False
    assert plan.reason == "boundary"


def test_tts_fragment_planner_keeps_sentence_final_punctuation_after_contraction_extension() -> None:
    plan = plan_tts_fragment(
        ["Hey", "!", "What", "'s", "up", "?"],
        stable_prefix="hi",
        config=TTSFragmentPlannerConfig(min_tokens=2, max_tokens=4, context_window_tokens=24),
    )

    assert plan.start_now is True
    assert plan.flush_text == "Hey ! What 's up ?"
    assert plan.remaining_tokens == ()


def test_tts_fragment_planner_waits_for_larger_append_chunk_when_low_latency_is_disabled() -> None:
    plan = plan_tts_fragment(
        ["pushing", "for"],
        stable_prefix="he insists about it",
        config=TTSFragmentPlannerConfig(
            min_tokens=2,
            max_tokens=6,
            context_window_tokens=24,
            allow_early_low_latency=False,
        ),
    )

    assert plan.start_now is False
    assert plan.flush_text == ""
    assert plan.remaining_tokens == ()


def test_tts_fragment_planner_flushes_append_chunk_at_boundary_when_low_latency_is_disabled() -> None:
    plan = plan_tts_fragment(
        ["pushing", "for", "it."],
        stable_prefix="he insists about it",
        config=TTSFragmentPlannerConfig(
            min_tokens=2,
            max_tokens=6,
            context_window_tokens=24,
            allow_early_low_latency=False,
        ),
    )

    assert plan.start_now is True
    assert plan.flush_text == "pushing for it."
    assert plan.remaining_tokens == ()
    assert plan.stream_fragment is True


def test_tts_fragment_planner_does_not_flush_standalone_punctuation_append_when_low_latency_is_disabled() -> None:
    plan = plan_tts_fragment(
        ["?", "**"],
        stable_prefix="hi yo year",
        config=TTSFragmentPlannerConfig(
            min_tokens=2,
            max_tokens=4,
            context_window_tokens=24,
            allow_early_low_latency=False,
        ),
    )

    assert plan.start_now is False
    assert plan.flush_text == ""
    assert plan.remaining_tokens == ()
