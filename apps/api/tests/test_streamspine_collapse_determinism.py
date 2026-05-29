from __future__ import annotations

from voice_pipeline.kernel.kernel_runtime import KernelRuntime
from voice_pipeline.shared.types import new_authority_event


def _base_events(session_id: str) -> list:
    return [
        new_authority_event(
            event_type="ASRFinalReceived",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:1",
            payload={"text": "hello world", "end_of_speech_observed": True},
        ),
    ]


def test_identical_event_streams_reconstruct_identical_state() -> None:
    session_id = "determinism"
    events = _base_events(session_id)

    left = KernelRuntime(session_id=session_id).replay(events)
    right = KernelRuntime(session_id=session_id).replay(events)

    assert left.state == right.state
    assert left.applied_events == right.applied_events


def test_reordered_transport_events_keep_semantic_phase_stable() -> None:
    session_id = "reorder"
    loop = KernelRuntime(session_id=session_id)

    vllm_requested = new_authority_event(
        event_type="VLLMRequested",
        session_id=session_id,
        sequence_no=1,
        lineage_id=f"{session_id}:epoch:1",
        payload={"request_id": "req-1", "prompt": "hello"},
    )
    first = loop.apply_event(vllm_requested)
    vllm_request_event = first.applied_events[-1]

    token_a = new_authority_event(
        event_type="VLLMChunkReceived",
        session_id=session_id,
        sequence_no=2,
        lineage_id=f"{session_id}:epoch:1",
        payload={"request_id": "req-1", "token": "A", "output_version": first.state.output.version},
        causation_id=vllm_request_event.event_id,
    )
    token_b = new_authority_event(
        event_type="VLLMChunkReceived",
        session_id=session_id,
        sequence_no=3,
        lineage_id=f"{session_id}:epoch:1",
        payload={"request_id": "req-1", "token": "B", "output_version": first.state.output.version},
        causation_id=vllm_request_event.event_id,
    )
    token_b_first = new_authority_event(
        event_type="VLLMChunkReceived",
        session_id=session_id,
        sequence_no=2,
        lineage_id=f"{session_id}:epoch:1",
        payload={"request_id": "req-1", "token": "B", "output_version": first.state.output.version},
        causation_id=vllm_request_event.event_id,
    )
    token_a_second = new_authority_event(
        event_type="VLLMChunkReceived",
        session_id=session_id,
        sequence_no=3,
        lineage_id=f"{session_id}:epoch:1",
        payload={"request_id": "req-1", "token": "A", "output_version": first.state.output.version},
        causation_id=vllm_request_event.event_id,
    )

    ordered = KernelRuntime(session_id=session_id).replay([vllm_requested, token_a, token_b])
    reordered = KernelRuntime(session_id=session_id).replay([vllm_requested, token_b_first, token_a_second])

    assert ordered.state.phase == reordered.state.phase
    assert ordered.state.output.version == reordered.state.output.version


def test_delayed_adapter_emissions_do_not_diverge_authority_state() -> None:
    session_id = "delay"
    loop_a = KernelRuntime(session_id=session_id)
    loop_b = KernelRuntime(session_id=session_id)

    events = [
        new_authority_event(
            event_type="VLLMRequested",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:1",
            payload={"request_id": "req-vllm", "text": "hello"},
        ),
    ]

    first_a = loop_a.replay(events)
    first_b = loop_b.replay(events)

    request_event_id = first_a.applied_events[-1].event_id
    delayed_chunk = new_authority_event(
        event_type="VLLMChunkReceived",
        session_id=session_id,
        sequence_no=2,
        lineage_id=f"{session_id}:epoch:1",
        payload={"request_id": "req-vllm", "chunk_id": "late-1", "output_version": first_a.state.output.version},
        causation_id=request_event_id,
    )

    final_a = loop_a.replay([events[0], delayed_chunk])
    final_b = loop_b.replay([events[0], delayed_chunk])

    assert final_a.state == final_b.state


def test_cancellation_precedence_clears_active_requests() -> None:
    session_id = "cancel"
    loop = KernelRuntime(session_id=session_id)

    vllm_requested = new_authority_event(
        event_type="VLLMRequested",
        session_id=session_id,
        sequence_no=1,
        lineage_id=f"{session_id}:epoch:1",
        payload={"request_id": "req-1", "prompt": "hello"},
    )
    cancel = new_authority_event(
        event_type="CancelRequested",
        session_id=session_id,
        sequence_no=2,
        lineage_id=f"{session_id}:epoch:1",
        payload={"reason": "user_interrupt"},
    )

    result = loop.replay([vllm_requested, cancel])

    assert result.state.phase in {"cancelled", "idle"}
    assert result.state.active_vllm_request_id == ""
    assert result.state.active_tts_request_id == ""



