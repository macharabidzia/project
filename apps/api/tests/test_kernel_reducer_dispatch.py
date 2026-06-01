from __future__ import annotations

from dataclasses import replace

from voice_pipeline.kernel.kernel_runtime import KernelConfig, KernelRuntime
from voice_pipeline.kernel.state import OutputState
from voice_pipeline.shared.types import new_authority_event


def test_asr_final_turn_commit_emits_vllm_dispatch_with_contract_shape() -> None:
    kernel = KernelRuntime(session_id="dispatch-shape")
    result = kernel.apply_event(
        new_authority_event(
            event_type="ASRFinalReceived",
            session_id="dispatch-shape",
            sequence_no=1,
            lineage_id="dispatch-shape:epoch:1",
            payload={"text": "hello world"},
        )
    )

    vllm_commands = [command for command in result.dispatch_commands if command.kind == "VLLM"]
    assert len(vllm_commands) == 1
    payload = dict(vllm_commands[0].payload)
    assert payload["session_id"] == "dispatch-shape"
    assert payload["lineage_id"] == "dispatch-shape:epoch:1"
    assert "epoch_id" in payload
    assert "turn_id" in payload
    assert "output_version" in payload


def test_vllm_completion_emits_tts_dispatch_with_contract_shape() -> None:
    kernel = KernelRuntime(session_id="tts-dispatch")
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-dispatch",
            sequence_no=1,
            lineage_id="tts-dispatch:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    kernel.apply_event(
        new_authority_event(
            event_type="VLLMChunkReceived",
            session_id="tts-dispatch",
            sequence_no=2,
            lineage_id="tts-dispatch:epoch:1",
            payload={"request_id": "req-1", "token": "hello", "output_version": output_version},
            causation_id=request_event_id,
        )
    )
    result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMCompleted",
            session_id="tts-dispatch",
            sequence_no=3,
            lineage_id="tts-dispatch:epoch:1",
            payload={"request_id": "req-1", "text": "hello", "output_version": output_version},
            causation_id=request_event_id,
        )
    )

    tts_commands = [command for command in result.dispatch_commands if command.kind == "TTS"]
    assert len(tts_commands) == 1
    payload = dict(tts_commands[0].payload)
    assert payload["session_id"] == "tts-dispatch"
    assert payload["lineage_id"] == "tts-dispatch:epoch:1"
    assert "epoch_id" in payload
    assert "turn_id" in payload
    assert "output_version" in payload


def test_vllm_chunk_tts_dispatch_keeps_stream_open_without_local_remainder() -> None:
    kernel = KernelRuntime(session_id="tts-chunk-stream-open")
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-chunk-stream-open",
            sequence_no=1,
            lineage_id="tts-chunk-stream-open:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    for index, token in enumerate(("He", "keeps"), start=2):
        result = kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-chunk-stream-open",
                sequence_no=index,
                lineage_id="tts-chunk-stream-open:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )

    tts_commands = [command for command in result.dispatch_commands if command.kind == "TTS"]
    assert len(tts_commands) == 1
    payload = dict(tts_commands[0].payload)
    assert payload["text"] == "He keeps"
    assert payload["stream_fragment"] is False


def test_vllm_completion_closes_active_tts_append_stream_when_buffer_is_drained() -> None:
    kernel = KernelRuntime(
        session_id="tts-append-close",
        config=KernelConfig(tts_first_fragment_min_tokens=1),
    )
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-append-close",
            sequence_no=1,
            lineage_id="tts-append-close:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    chunk_result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMChunkReceived",
            session_id="tts-append-close",
            sequence_no=2,
            lineage_id="tts-append-close:epoch:1",
            payload={"request_id": "req-1", "token": "hello", "output_version": output_version},
            causation_id=request_event_id,
        )
    )
    tts_event = next(event for event in chunk_result.applied_events if event.event_type == "TTSRequested")
    tts_request_id = str(tts_event.payload.get("request_id", "")).strip() or f"{tts_event.lineage_id}:tts:{tts_event.event_id}"

    # Mirror the reducer state after the first TTS request has been accepted.
    kernel.apply_event(tts_event)
    assert kernel.state.active_tts_request_id == tts_request_id

    result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMCompleted",
            session_id="tts-append-close",
            sequence_no=3,
            lineage_id="tts-append-close:epoch:1",
            payload={"request_id": "req-1", "text": "hello", "output_version": output_version},
            causation_id=request_event_id,
        )
    )

    append_commands = [command for command in result.dispatch_commands if command.kind == "TTS_APPEND"]
    assert len(append_commands) == 1
    payload = dict(append_commands[0].payload)
    assert payload["text"] == ""
    assert payload["final_fragment"] is True
    assert append_commands[0].request_id == tts_request_id


def test_vllm_chunk_holds_short_boundaryless_reply_until_completion() -> None:
    kernel = KernelRuntime(
        session_id="tts-short-boundaryless-hold",
        config=KernelConfig(tts_first_fragment_min_tokens=3),
    )
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-short-boundaryless-hold",
            sequence_no=1,
            lineage_id="tts-short-boundaryless-hold:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello there"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    for index, token in enumerate(("hello", "there"), start=2):
        result = kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-short-boundaryless-hold",
                sequence_no=index,
                lineage_id="tts-short-boundaryless-hold:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )

    assert [event for event in result.applied_events if event.event_type == "TTSRequested"] == []
    assert kernel.state.output.vllm_stream_buffer == ("hello", "there")


def test_vllm_completion_starts_short_boundaryless_reply_as_immediate_close_stream() -> None:
    kernel = KernelRuntime(
        session_id="tts-short-boundaryless-complete",
        config=KernelConfig(tts_first_fragment_min_tokens=3),
    )
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-short-boundaryless-complete",
            sequence_no=1,
            lineage_id="tts-short-boundaryless-complete:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello there"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    for index, token in enumerate(("hello", "there"), start=2):
        kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-short-boundaryless-complete",
                sequence_no=index,
                lineage_id="tts-short-boundaryless-complete:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )

    result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMCompleted",
            session_id="tts-short-boundaryless-complete",
            sequence_no=4,
            lineage_id="tts-short-boundaryless-complete:epoch:1",
            payload={"request_id": "req-1", "text": "hello there", "output_version": output_version},
            causation_id=request_event_id,
        )
    )

    tts_commands = [command for command in result.dispatch_commands if command.kind == "TTS"]
    assert len(tts_commands) == 1
    payload = dict(tts_commands[0].payload)
    assert payload["text"] == "hello there"
    assert payload["stream_fragment"] is True
    assert payload["close_stream_immediately"] is True


def test_vllm_chunk_append_waits_for_boundary_after_stream_has_started() -> None:
    kernel = KernelRuntime(session_id="tts-append-cadence")
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-append-cadence",
            sequence_no=1,
            lineage_id="tts-append-cadence:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    first_chunk_result = None
    for index, token in enumerate(("He", "keeps"), start=2):
        first_chunk_result = kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-append-cadence",
                sequence_no=index,
                lineage_id="tts-append-cadence:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )
    assert first_chunk_result is not None
    tts_event = next(event for event in first_chunk_result.applied_events if event.event_type == "TTSRequested")
    kernel.apply_event(tts_event)

    pushing_result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMChunkReceived",
            session_id="tts-append-cadence",
            sequence_no=4,
            lineage_id="tts-append-cadence:epoch:1",
            payload={"request_id": "req-1", "token": "pushing", "output_version": output_version},
            causation_id=request_event_id,
        )
    )
    assert [command for command in pushing_result.dispatch_commands if command.kind == "TTS_APPEND"] == []

    for_result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMChunkReceived",
            session_id="tts-append-cadence",
            sequence_no=5,
            lineage_id="tts-append-cadence:epoch:1",
            payload={"request_id": "req-1", "token": "for", "output_version": output_version},
            causation_id=request_event_id,
        )
    )
    assert [command for command in for_result.dispatch_commands if command.kind == "TTS_APPEND"] == []

    boundary_result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMChunkReceived",
            session_id="tts-append-cadence",
            sequence_no=6,
            lineage_id="tts-append-cadence:epoch:1",
            payload={"request_id": "req-1", "token": "it.", "output_version": output_version},
            causation_id=request_event_id,
        )
    )


def test_tts_requested_starts_new_stream_when_active_request_is_from_older_output_version() -> None:
    kernel = KernelRuntime(session_id="tts-stale-active-request")
    kernel._state = replace(
        kernel.state,
        phase="playing",
        active_tts_request_id="old-tts",
        output=OutputState(version=2, active_turn_id="turn-2"),
        request_event_ids=(("old-tts", "event-1", 1),),
    )

    result = kernel.apply_event(
        new_authority_event(
            event_type="TTSRequested",
            session_id="tts-stale-active-request",
            sequence_no=1,
            lineage_id="tts-stale-active-request:epoch:2",
            payload={"text": "hello there", "stream_fragment": True},
        )
    )

    tts_commands = [command for command in result.dispatch_commands if command.kind == "TTS"]
    append_commands = [command for command in result.dispatch_commands if command.kind == "TTS_APPEND"]
    assert len(tts_commands) == 1
    assert append_commands == []


def test_vllm_chunk_does_not_open_on_boundaryless_two_word_opener() -> None:
    kernel = KernelRuntime(
        session_id="tts-boundaryless-opener",
        config=KernelConfig(tts_fragment_min_tokens=2, tts_fragment_max_tokens=4, tts_context_window_tokens=24),
    )
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-boundaryless-opener",
            sequence_no=1,
            lineage_id="tts-boundaryless-opener:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    early_result = None
    for index, token in enumerate(("You're", "saying"), start=2):
        early_result = kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-boundaryless-opener",
                sequence_no=index,
                lineage_id="tts-boundaryless-opener:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )
    assert early_result is not None
    assert [command for command in early_result.dispatch_commands if command.kind == "TTS"] == []

    fuller_result = None
    for index, token in enumerate(('"he', 'only" —'), start=4):
        fuller_result = kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-boundaryless-opener",
                sequence_no=index,
                lineage_id="tts-boundaryless-opener:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )
    assert fuller_result is not None
    tts_commands = [command for command in fuller_result.dispatch_commands if command.kind == "TTS"]
    assert len(tts_commands) == 1
    payload = dict(tts_commands[0].payload)
    assert payload["text"] == 'You\'re saying "he only" —'
    assert payload["stream_fragment"] is True


def test_vllm_chunk_does_not_open_on_dangling_function_word_tail() -> None:
    kernel = KernelRuntime(
        session_id="tts-dangling-function-tail",
        config=KernelConfig(tts_fragment_min_tokens=2, tts_fragment_max_tokens=4, tts_context_window_tokens=24),
    )
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-dangling-function-tail",
            sequence_no=1,
            lineage_id="tts-dangling-function-tail:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    early_result = None
    for index, token in enumerate(("hello!", "how", "can", "i"), start=2):
        early_result = kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-dangling-function-tail",
                sequence_no=index,
                lineage_id="tts-dangling-function-tail:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )
    assert early_result is not None
    assert [command for command in early_result.dispatch_commands if command.kind == "TTS"] == []

    fuller_result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMChunkReceived",
            session_id="tts-dangling-function-tail",
            sequence_no=6,
            lineage_id="tts-dangling-function-tail:epoch:1",
            payload={"request_id": "req-1", "token": "help", "output_version": output_version},
            causation_id=request_event_id,
        )
    )
    tts_commands = [command for command in fuller_result.dispatch_commands if command.kind == "TTS"]
    assert len(tts_commands) == 1
    payload = dict(tts_commands[0].payload)
    assert payload["text"] == "hello! how can i help"
    assert payload["stream_fragment"] is True




def test_vllm_chunk_append_does_not_flush_standalone_punctuation_tail() -> None:
    kernel = KernelRuntime(
        session_id="tts-punctuation-append-hold",
        config=KernelConfig(tts_fragment_min_tokens=2, tts_fragment_max_tokens=4, tts_context_window_tokens=24),
    )
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-punctuation-append-hold",
            sequence_no=1,
            lineage_id="tts-punctuation-append-hold:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    first_chunk_result = None
    for index, token in enumerate(("Hey", "!", "What", "'s", "up"), start=2):
        first_chunk_result = kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-punctuation-append-hold",
                sequence_no=index,
                lineage_id="tts-punctuation-append-hold:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )
    assert first_chunk_result is not None
    tts_event = next(event for event in first_chunk_result.applied_events if event.event_type == "TTSRequested")
    kernel.apply_event(tts_event)

    punctuation_result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMChunkReceived",
            session_id="tts-punctuation-append-hold",
            sequence_no=7,
            lineage_id="tts-punctuation-append-hold:epoch:1",
            payload={"request_id": "req-1", "token": "?", "output_version": output_version},
            causation_id=request_event_id,
        )
    )
    assert [command for command in punctuation_result.dispatch_commands if command.kind == "TTS_APPEND"] == []

    note_result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMChunkReceived",
            session_id="tts-punctuation-append-hold",
            sequence_no=8,
            lineage_id="tts-punctuation-append-hold:epoch:1",
            payload={"request_id": "req-1", "token": "**", "output_version": output_version},
            causation_id=request_event_id,
        )
    )
    assert [command for command in note_result.dispatch_commands if command.kind == "TTS_APPEND"] == []

    colon_result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMChunkReceived",
            session_id="tts-punctuation-append-hold",
            sequence_no=9,
            lineage_id="tts-punctuation-append-hold:epoch:1",
            payload={"request_id": "req-1", "token": ":", "output_version": output_version},
            causation_id=request_event_id,
        )
    )
    assert [command for command in colon_result.dispatch_commands if command.kind == "TTS_APPEND"] == []

    tail_result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMChunkReceived",
            session_id="tts-punctuation-append-hold",
            sequence_no=10,
            lineage_id="tts-punctuation-append-hold:epoch:1",
            payload={"request_id": "req-1", "token": "The", "output_version": output_version},
            causation_id=request_event_id,
        )
    )
    append_commands = [command for command in tail_result.dispatch_commands if command.kind == "TTS_APPEND"]
    assert len(append_commands) == 1


def test_vllm_first_chunk_does_not_flush_tiny_boundary_prefix_when_window_is_full() -> None:
    kernel = KernelRuntime(
        session_id="tts-first-window-shape",
        config=KernelConfig(tts_fragment_min_tokens=2, tts_fragment_max_tokens=4, tts_context_window_tokens=24),
    )
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-first-window-shape",
            sequence_no=1,
            lineage_id="tts-first-window-shape:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    first_tts_result = None
    for index, token in enumerate(("Hey", "!", "What", "'s", "up", "?"), start=2):
        current_result = kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-first-window-shape",
                sequence_no=index,
                lineage_id="tts-first-window-shape:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )
        if first_tts_result is None and [command for command in current_result.dispatch_commands if command.kind == "TTS"]:
            first_tts_result = current_result

    assert first_tts_result is not None
    tts_commands = [command for command in first_tts_result.dispatch_commands if command.kind == "TTS"]
    assert len(tts_commands) == 1
    payload = dict(tts_commands[0].payload)
    assert payload["text"] == "Hey ! What 's up"
    assert payload["stream_fragment"] is False


def test_vllm_does_not_start_tts_on_internal_boundary_prefix_before_window_fills() -> None:
    kernel = KernelRuntime(
        session_id="tts-internal-boundary-hold",
        config=KernelConfig(tts_fragment_min_tokens=2, tts_fragment_max_tokens=4, tts_context_window_tokens=24),
    )
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-internal-boundary-hold",
            sequence_no=1,
            lineage_id="tts-internal-boundary-hold:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    last_result = None
    for index, token in enumerate(("Hey", "!", "What"), start=2):
        last_result = kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-internal-boundary-hold",
                sequence_no=index,
                lineage_id="tts-internal-boundary-hold:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )

    assert last_result is not None
    assert [command for command in last_result.dispatch_commands if command.kind == "TTS"] == []


def test_vllm_does_not_start_tts_when_full_window_stops_before_rest_of_short_clause() -> None:
    kernel = KernelRuntime(
        session_id="tts-full-window-hold",
        config=KernelConfig(tts_fragment_min_tokens=2, tts_fragment_max_tokens=4, tts_context_window_tokens=24),
    )
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-full-window-hold",
            sequence_no=1,
            lineage_id="tts-full-window-hold:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    last_result = None
    for index, token in enumerate(("Hey", "!", "What", "'s"), start=2):
        last_result = kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-full-window-hold",
                sequence_no=index,
                lineage_id="tts-full-window-hold:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )

    assert last_result is not None
    assert [command for command in last_result.dispatch_commands if command.kind == "TTS"] == []


def test_vllm_starts_tts_after_one_extra_token_for_contraction_split_first_clause() -> None:
    kernel = KernelRuntime(
        session_id="tts-short-clause-extra-word-hold",
        config=KernelConfig(tts_fragment_min_tokens=2, tts_fragment_max_tokens=4, tts_context_window_tokens=24),
    )
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-short-clause-extra-word-hold",
            sequence_no=1,
            lineage_id="tts-short-clause-extra-word-hold:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    last_result = None
    for index, token in enumerate(("Hey", "!", "What", "'s", "up"), start=2):
        last_result = kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-short-clause-extra-word-hold",
                sequence_no=index,
                lineage_id="tts-short-clause-extra-word-hold:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )

    assert last_result is not None
    tts_commands = [command for command in last_result.dispatch_commands if command.kind == "TTS"]
    assert len(tts_commands) == 1
    payload = dict(tts_commands[0].payload)
    assert payload["text"] == "Hey ! What 's up"
    assert payload["stream_fragment"] is False


def test_vllm_short_sentence_final_reply_starts_as_single_initial_request_with_8_token_window() -> None:
    kernel = KernelRuntime(
        session_id="tts-short-sentence-hold",
        config=KernelConfig(tts_fragment_min_tokens=8, tts_fragment_max_tokens=8, tts_context_window_tokens=24),
    )
    first = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="tts-short-sentence-hold",
            sequence_no=1,
            lineage_id="tts-short-sentence-hold:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = first.applied_events[-1].event_id
    output_version = kernel.state.request_output_version("req-1")

    tokens = ("Hey", "!", "What", "'s", "up", "?", "Hey", "!")
    first_tts_result = None
    for index, token in enumerate(tokens, start=2):
        current_result = kernel.apply_event(
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id="tts-short-sentence-hold",
                sequence_no=index,
                lineage_id="tts-short-sentence-hold:epoch:1",
                payload={"request_id": "req-1", "token": token, "output_version": output_version},
                causation_id=request_event_id,
            )
        )
        if first_tts_result is None and [event for event in current_result.applied_events if event.event_type == "TTSRequested"]:
            first_tts_result = current_result

    assert first_tts_result is not None
    tts_events = [event for event in first_tts_result.applied_events if event.event_type == "TTSRequested"]
    assert len(tts_events) == 1
    tts_commands = [command for command in first_tts_result.dispatch_commands if command.kind == "TTS"]
    assert len(tts_commands) == 1
    payload = dict(tts_commands[0].payload)
    assert payload["text"] == "Hey ! What 's up ?"
    assert payload["stream_fragment"] is True

    complete_result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMCompleted",
            session_id="tts-short-sentence-hold",
            sequence_no=20,
            lineage_id="tts-short-sentence-hold:epoch:1",
            payload={"request_id": "req-1", "text": "Hey ! What 's up ? Hey !", "output_version": output_version},
            causation_id=request_event_id,
        )
    )

    append_commands = [command for command in complete_result.dispatch_commands if command.kind == "TTS_APPEND"]
    assert len(append_commands) == 1
    payload = dict(append_commands[0].payload)
    assert payload["text"] == "Hey !"
    assert payload["final_fragment"] is True
