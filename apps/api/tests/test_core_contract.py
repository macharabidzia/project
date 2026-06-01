from __future__ import annotations

from dataclasses import replace

import pytest

from voice_pipeline.kernel.reducer import ReducerConfig, ReducerDiagnostics, reduce_event
from voice_pipeline.kernel.state import KernelState, OutputState, TranscriptState
from voice_pipeline.kernel.kernel_runtime import KernelConfig
from voice_pipeline.kernel.kernel_runtime import KernelRuntime
from voice_pipeline.shared.time import now_ns
from voice_pipeline.shared.types import new_authority_event


def test_stable_prefix_commit_starts_llm_without_final_asr() -> None:
    session_id = "core-contract"
    kernel = KernelRuntime(session_id=session_id)

    partial_1 = new_authority_event(
        event_type="ASRPartialReceived",
        session_id=session_id,
        sequence_no=1,
        lineage_id=f"{session_id}:epoch:0",
        payload={"text": "tell me"},
    )
    partial_2 = new_authority_event(
        event_type="ASRPartialReceived",
        session_id=session_id,
        sequence_no=2,
        lineage_id=f"{session_id}:epoch:0",
        payload={"text": "tell me the"},
    )

    kernel.apply_event(partial_1)
    kernel.apply_event(partial_2)

    assert kernel.state.active_vllm_request_id
    assert kernel.state.transcript.committed_text == "tell me"


def test_stable_prefix_does_not_commit_when_partial_turn_commit_disabled() -> None:
    session_id = "core-contract-no-partial-commit"
    kernel = KernelRuntime(
        session_id=session_id,
        config=KernelConfig(
            stable_prefix_min_repeats=1,
            stable_prefix_min_tokens=2,
            allow_partial_turn_commit=False,
        ),
    )

    kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "tell me"},
        )
    )
    kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=2,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "tell me the"},
        )
    )

    assert not kernel.state.active_vllm_request_id
    assert kernel.state.transcript.committed_text == ""
    assert kernel.state.transcript.partial_text == "tell me the"
    assert kernel.state.transcript.stable_prefix == "tell me"


def test_greeting_partial_commits_early_and_extension_stays_same_turn() -> None:
    session_id = "core-contract-greeting"
    kernel = KernelRuntime(
        session_id=session_id,
        config=KernelConfig(stable_prefix_min_repeats=1, stable_prefix_min_tokens=2),
    )

    kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "hello"},
        )
    )

    assert kernel.state.active_vllm_request_id
    assert kernel.state.transcript.committed_text == "hello"
    assert kernel.state.transcript.partial_text == ""

    active_vllm_request_id = kernel.state.active_vllm_request_id
    generation_index = kernel.state.generation_index
    output_version = kernel.state.output.version

    kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=2,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "hello there"},
        )
    )

    assert kernel.state.active_vllm_request_id == active_vllm_request_id
    assert kernel.state.generation_index == generation_index
    assert kernel.state.output.version == output_version
    assert kernel.state.transcript.committed_text == "hello"
    assert kernel.state.transcript.partial_text == ""

    kernel.apply_event(
        new_authority_event(
            event_type="ASRFinalReceived",
            session_id=session_id,
            sequence_no=3,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "hello there"},
        )
    )

    assert kernel.state.active_vllm_request_id == active_vllm_request_id
    assert kernel.state.generation_index == generation_index
    assert kernel.state.output.version == output_version
    assert kernel.state.transcript.committed_text == "hello"
    assert kernel.state.transcript.final_text == "hello"


def test_greeting_partial_does_not_commit_when_partial_turn_commit_disabled() -> None:
    session_id = "core-contract-no-greeting-partial-commit"
    kernel = KernelRuntime(
        session_id=session_id,
        config=KernelConfig(
            stable_prefix_min_repeats=1,
            stable_prefix_min_tokens=2,
            allow_partial_turn_commit=False,
        ),
    )

    kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "hello"},
        )
    )

    assert not kernel.state.active_vllm_request_id
    assert kernel.state.transcript.committed_text == ""
    assert kernel.state.transcript.partial_text == "hello"
    assert kernel.state.transcript.stable_prefix == ""


def test_greeting_extension_partial_does_not_interrupt_playing_turn() -> None:
    session_id = "core-contract-greeting-playing"
    kernel = KernelRuntime(
        session_id=session_id,
        config=KernelConfig(stable_prefix_min_repeats=1, stable_prefix_min_tokens=2),
    )

    kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "hello"},
        )
    )
    active_vllm_request_id = kernel.state.active_vllm_request_id

    tts_result = kernel.apply_event(
        new_authority_event(
            event_type="TTSRequested",
            session_id=session_id,
            sequence_no=2,
            lineage_id=f"{session_id}:epoch:0",
            payload={"request_id": "tts-1", "text": "hello", "output_version": kernel.state.output.version},
        )
    )
    assert tts_result.applied_events[-1].event_type == "TTSRequested"

    active_tts_request_id = kernel.state.active_tts_request_id
    generation_index = kernel.state.generation_index
    output_version = kernel.state.output.version

    result = kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=3,
            lineage_id=f"{session_id}:epoch:1",
            payload={"text": "hello there"},
        )
    )

    event_types = [event.event_type for event in result.applied_events]
    assert "InterruptRequested" not in event_types
    assert kernel.state.active_vllm_request_id == active_vllm_request_id
    assert kernel.state.active_tts_request_id == active_tts_request_id
    assert kernel.state.generation_index == generation_index
    assert kernel.state.output.version == output_version
    assert kernel.state.transcript.committed_text == "hello"
    assert kernel.state.transcript.partial_text == ""


def test_greeting_extension_after_tts_completed_does_not_start_second_turn() -> None:
    session_id = "core-contract-greeting-idle-extension"
    kernel = KernelRuntime(
        session_id=session_id,
        config=KernelConfig(stable_prefix_min_repeats=1, stable_prefix_min_tokens=2),
    )

    kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "hello"},
        )
    )
    kernel.apply_event(
        new_authority_event(
            event_type="TTSRequested",
            session_id=session_id,
            sequence_no=2,
            lineage_id=f"{session_id}:epoch:0",
            payload={"request_id": "tts-1", "text": "hello", "output_version": kernel.state.output.version},
        )
    )
    kernel.apply_event(
        new_authority_event(
            event_type="TTSCompleted",
            session_id=session_id,
            sequence_no=3,
            lineage_id=f"{session_id}:epoch:0",
            payload={"request_id": "tts-1", "output_version": kernel.state.output.version},
            causation_id=kernel.state.request_event_id("tts-1"),
        )
    )

    generation_index = kernel.state.generation_index
    output_version = kernel.state.output.version

    result = kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=4,
            lineage_id=f"{session_id}:epoch:1",
            payload={"text": "hello there"},
        )
    )

    event_types = [event.event_type for event in result.applied_events]
    assert "TurnCommitted" not in event_types
    assert kernel.state.generation_index == generation_index
    assert kernel.state.output.version == output_version
    assert kernel.state.transcript.committed_text == "hello"
    assert kernel.state.transcript.final_text == "hello"


def test_greeting_extension_after_tts_completed_is_ignored_even_if_phase_is_stale_generating() -> None:
    session_id = "core-contract-greeting-stale-generating"
    kernel = KernelRuntime(
        session_id=session_id,
        config=KernelConfig(stable_prefix_min_repeats=1, stable_prefix_min_tokens=2),
    )

    kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "hello"},
        )
    )
    kernel.apply_event(
        new_authority_event(
            event_type="TTSRequested",
            session_id=session_id,
            sequence_no=2,
            lineage_id=f"{session_id}:epoch:0",
            payload={"request_id": "tts-1", "text": "hello", "output_version": kernel.state.output.version},
        )
    )
    kernel.apply_event(
        new_authority_event(
            event_type="TTSCompleted",
            session_id=session_id,
            sequence_no=3,
            lineage_id=f"{session_id}:epoch:0",
            payload={"request_id": "tts-1", "output_version": kernel.state.output.version},
            causation_id=kernel.state.request_event_id("tts-1"),
        )
    )

    kernel.state = replace(kernel.state, phase="generating")
    kernel.diagnostics = replace(
        kernel.diagnostics,
        last_completed_committed_text="hello",
    )
    generation_index = kernel.state.generation_index
    output_version = kernel.state.output.version

    result = kernel.apply_event(
        new_authority_event(
            event_type="ASRFinalReceived",
            session_id=session_id,
            sequence_no=4,
            lineage_id=f"{session_id}:epoch:1",
            payload={"text": "hello there"},
        )
    )

    event_types = [event.event_type for event in result.applied_events]
    assert "TurnCommitted" not in event_types
    assert "CancelRequested" not in event_types
    assert kernel.state.generation_index == generation_index
    assert kernel.state.output.version == output_version
    assert kernel.state.transcript.committed_text == "hello"
    assert kernel.state.transcript.final_text == "hello"


def test_completed_greeting_extension_is_ignored_within_realistic_post_tts_window() -> None:
    session_id = "core-contract-greeting-extension-window"
    kernel = KernelRuntime(
        session_id=session_id,
        config=KernelConfig(stable_prefix_min_repeats=1, stable_prefix_min_tokens=2),
    )
    state = KernelState(
        session_id=session_id,
        phase="idle",
        transcript=TranscriptState(
            committed_text="hello",
            final_text="hello",
            last_dispatched_stable_prefix="hello",
            conversation_history=("hello",),
        ),
        output=OutputState(version=1),
        turn_index=1,
        committed_turn_index=1,
        generation_index=1,
        lineage_id=f"{session_id}:epoch:0",
    )
    diagnostics = ReducerDiagnostics(
        last_completed_committed_text="hello",
        last_turn_completed_ns=1_000_000_000,
    )
    event = new_authority_event(
        event_type="ASRFinalReceived",
        session_id=session_id,
        sequence_no=1,
        lineage_id=f"{session_id}:epoch:1",
        payload={"text": "hello there"},
    )

    transition = reduce_event(
        state,
        event,
        config=ReducerConfig(stable_prefix_min_repeats=1, stable_prefix_min_tokens=2),
        diagnostics=diagnostics,
        event_time_ns=6_159_000_000,
    )

    assert transition.next_state.transcript.committed_text == "hello"
    assert transition.next_state.transcript.final_text == "hello"
    assert transition.next_state.generation_index == 1
    assert not transition.derived_events


def test_completed_turn_short_tail_final_is_ignored_within_grace_window() -> None:
    session_id = "core-contract-short-tail-window"
    kernel = KernelRuntime(session_id=session_id)
    state = KernelState(
        session_id=session_id,
        phase="idle",
        transcript=TranscriptState(
            committed_text="hello there",
            final_text="hello there",
            last_dispatched_stable_prefix="hello there",
            conversation_history=("hello there",),
        ),
        output=OutputState(version=1),
        turn_index=1,
        committed_turn_index=1,
        generation_index=1,
        lineage_id=f"{session_id}:epoch:0",
    )
    diagnostics = ReducerDiagnostics(
        last_completed_committed_text="hello there",
        last_turn_completed_ns=1_000_000_000,
    )
    event = new_authority_event(
        event_type="ASRFinalReceived",
        session_id=session_id,
        sequence_no=1,
        lineage_id=f"{session_id}:epoch:1",
        payload={"text": "the"},
    )

    transition = reduce_event(
        state,
        event,
        diagnostics=diagnostics,
        event_time_ns=5_500_000_000,
    )

    assert transition.next_state.transcript.committed_text == "hello there"
    assert transition.next_state.transcript.final_text == "hello there"
    assert transition.next_state.generation_index == 1
    assert not transition.derived_events

    kernel.state = replace(
        kernel.state,
        phase="idle",
        transcript=TranscriptState(
            committed_text="hello",
            final_text="hello",
            last_dispatched_stable_prefix="hello",
            conversation_history=("hello",),
        ),
        output=OutputState(version=1),
        turn_index=1,
        committed_turn_index=1,
        generation_index=1,
        lineage_id=f"{session_id}:epoch:0",
    )
    kernel.diagnostics = replace(
        kernel.diagnostics,
        last_completed_committed_text="hello",
        last_turn_completed_ns=now_ns() - 1_000_000_000,
    )
    generation_index = kernel.state.generation_index
    output_version = kernel.state.output.version

    result = kernel.apply_event(
        new_authority_event(
            event_type="ASRFinalReceived",
            session_id=session_id,
            sequence_no=5,
            lineage_id=f"{session_id}:epoch:1",
            payload={"text": "hello there"},
        )
    )

    event_types = [event.event_type for event in result.applied_events]
    assert "TurnCommitted" not in event_types
    assert kernel.state.generation_index == generation_index
    assert kernel.state.output.version == output_version
    assert kernel.state.transcript.committed_text == "hello"
    assert kernel.state.transcript.final_text == "hello"


def test_matching_final_does_not_interrupt_generating_turn_on_new_epoch() -> None:
    session_id = "core-contract-final-confirm-generating"
    kernel = KernelRuntime(
        session_id=session_id,
        config=KernelConfig(stable_prefix_min_repeats=1, stable_prefix_min_tokens=2),
    )

    kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "hello"},
        )
    )
    kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=2,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "hello there"},
        )
    )

    active_vllm_request_id = kernel.state.active_vllm_request_id
    generation_index = kernel.state.generation_index
    output_version = kernel.state.output.version

    result = kernel.apply_event(
        new_authority_event(
            event_type="ASRFinalReceived",
            session_id=session_id,
            sequence_no=3,
            lineage_id=f"{session_id}:epoch:1",
            payload={"text": "hello there"},
        )
    )

    event_types = [event.event_type for event in result.applied_events]
    assert "CancelRequested" not in event_types
    assert kernel.state.active_vllm_request_id == active_vllm_request_id
    assert kernel.state.generation_index == generation_index
    assert kernel.state.output.version == output_version
    assert kernel.state.lineage_id == f"{session_id}:epoch:0"
    assert kernel.state.transcript.final_text == "hello there"


def test_matching_final_does_not_interrupt_playing_turn_on_new_epoch() -> None:
    session_id = "core-contract-final-confirm-playing"
    kernel = KernelRuntime(
        session_id=session_id,
        config=KernelConfig(stable_prefix_min_repeats=1, stable_prefix_min_tokens=2),
    )

    kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "hello"},
        )
    )
    kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id=session_id,
            sequence_no=2,
            lineage_id=f"{session_id}:epoch:0",
            payload={"text": "hello there"},
        )
    )

    tts_result = kernel.apply_event(
        new_authority_event(
            event_type="TTSRequested",
            session_id=session_id,
            sequence_no=3,
            lineage_id=f"{session_id}:epoch:0",
            payload={"request_id": "tts-1", "text": "hello there", "output_version": kernel.state.output.version},
        )
    )
    assert tts_result.applied_events[-1].event_type == "TTSRequested"

    active_tts_request_id = kernel.state.active_tts_request_id
    generation_index = kernel.state.generation_index
    output_version = kernel.state.output.version

    result = kernel.apply_event(
        new_authority_event(
            event_type="ASRFinalReceived",
            session_id=session_id,
            sequence_no=4,
            lineage_id=f"{session_id}:epoch:1",
            payload={"text": "hello there"},
        )
    )

    event_types = [event.event_type for event in result.applied_events]
    assert "CancelRequested" not in event_types
    assert kernel.state.active_tts_request_id == active_tts_request_id
    assert kernel.state.generation_index == generation_index
    assert kernel.state.output.version == output_version
    assert kernel.state.lineage_id == f"{session_id}:epoch:0"
    assert kernel.state.transcript.final_text == "hello there"


def test_interrupt_bumps_generation_epoch_and_output_version() -> None:
    session_id = "core-contract-interrupt"
    kernel = KernelRuntime(session_id=session_id)

    initial_generation = kernel.state.generation_index
    initial_output_version = kernel.state.output.version

    kernel.apply_event(
        new_authority_event(
            event_type="InterruptRequested",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:0",
            payload={"reason": "barge_in"},
        )
    )

    assert kernel.state.generation_index == initial_generation + 1
    assert kernel.state.output.version == initial_output_version + 1


def test_engine_output_rejects_stale_output_version() -> None:
    session_id = "core-contract-stale"
    kernel = KernelRuntime(session_id=session_id)

    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id

    stale_token = new_authority_event(
        event_type="VLLMChunkReceived",
        session_id=session_id,
        sequence_no=2,
        lineage_id=f"{session_id}:epoch:1",
        payload={"request_id": "req-1", "token": "late", "output_version": 999},
        causation_id=request_event_id,
    )

    with pytest.raises(ValueError, match="output version mismatch"):
        kernel.apply_event(stale_token)


def test_tts_output_rejects_stale_output_version() -> None:
    session_id = "core-contract-tts-stale"
    kernel = KernelRuntime(session_id=session_id)
    first = kernel.apply_event(
        new_authority_event(
            event_type="TTSRequested",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:1",
            payload={"request_id": "tts-1", "text": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id

    with pytest.raises(ValueError, match="output version mismatch"):
        kernel.apply_event(
            new_authority_event(
                event_type="TTSCompleted",
                session_id=session_id,
                sequence_no=2,
                lineage_id=f"{session_id}:epoch:1",
                payload={"request_id": "tts-1", "output_version": 999},
                causation_id=request_event_id,
            )
        )
