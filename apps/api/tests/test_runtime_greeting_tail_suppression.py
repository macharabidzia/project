from __future__ import annotations

from types import SimpleNamespace

from voice_pipeline.runtime.bootstrap import VoicePipelineRuntime
from voice_pipeline.stt.asr_engine import ASREvent


def _runtime_for_suppression(*, completed_text: str = "hello", completed_ns: int = 1_000) -> VoicePipelineRuntime:
    runtime = object.__new__(VoicePipelineRuntime)
    runtime.kernel = SimpleNamespace(
        _diagnostics=SimpleNamespace(
            last_completed_committed_text=completed_text,
            last_turn_completed_ns=completed_ns,
        ),
        state=SimpleNamespace(
            active_vllm_request_id="",
            active_tts_request_id="",
        ),
    )
    runtime._last_vad_speech_start_ns = 0
    runtime._last_suppressed_greeting_extension_text = ""
    runtime._last_suppressed_greeting_extension_ns = 0
    return runtime


def test_runtime_suppresses_suffix_tail_after_suppressed_greeting_extension() -> None:
    runtime = _runtime_for_suppression()

    extension = ASREvent(
        event_type="ASRFinalReceived",
        text="hello there",
        lineage_id="runtime:epoch:1",
        emitted_at_ns=2_000,
    )
    assert runtime._should_suppress_stale_greeting_asr_extension(extension) is True

    suffix_tail = ASREvent(
        event_type="ASRFinalReceived",
        text="there",
        lineage_id="runtime:epoch:1",
        emitted_at_ns=3_000,
    )
    assert runtime._should_suppress_stale_greeting_asr_extension(suffix_tail) is True

    suffix_fragment = ASREvent(
        event_type="ASRFinalReceived",
        text="the",
        lineage_id="runtime:epoch:1",
        emitted_at_ns=4_000,
    )
    assert runtime._should_suppress_stale_greeting_asr_extension(suffix_fragment) is True


def test_runtime_allows_new_vad_speech_after_suppressed_greeting_extension() -> None:
    runtime = _runtime_for_suppression()

    extension = ASREvent(
        event_type="ASRPartialReceived",
        text="hello there",
        lineage_id="runtime:epoch:1",
        emitted_at_ns=2_000,
    )
    assert runtime._should_suppress_stale_greeting_asr_extension(extension) is True

    runtime.note_vad_speech_start(5_000)

    new_speech = ASREvent(
        event_type="ASRPartialReceived",
        text="there",
        lineage_id="runtime:epoch:2",
        emitted_at_ns=6_000,
    )
    assert runtime._should_suppress_stale_greeting_asr_extension(new_speech) is False


def test_runtime_filter_stale_asr_events_drops_extension_and_suffix_tail() -> None:
    runtime = _runtime_for_suppression()

    kept = ASREvent(
        event_type="ASRPartialReceived",
        text="hello",
        lineage_id="runtime:epoch:1",
        emitted_at_ns=1_500,
    )
    extension = ASREvent(
        event_type="ASRPartialReceived",
        text="hello there",
        lineage_id="runtime:epoch:1",
        emitted_at_ns=2_000,
    )
    suffix_tail = ASREvent(
        event_type="ASRFinalReceived",
        text="there",
        lineage_id="runtime:epoch:1",
        emitted_at_ns=3_000,
    )
    suffix_fragment = ASREvent(
        event_type="ASRFinalReceived",
        text="the",
        lineage_id="runtime:epoch:1",
        emitted_at_ns=4_000,
    )

    assert runtime._filter_stale_asr_events((kept, extension, suffix_tail, suffix_fragment)) == (kept,)
