from __future__ import annotations

from voice_pipeline.kernel.kernel_runtime import KernelRuntime
from voice_pipeline.shared.types import new_authority_event


def test_kernel_runtime_emits_tts_request_from_first_llm_fragment() -> None:
    kernel = KernelRuntime(session_id="session-1")

    first_partial = new_authority_event(
        event_type="ASRPartialReceived",
        session_id="session-1",
        sequence_no=1,
        lineage_id="session-1:epoch:1",
        payload={"text": "hello there"},
    )
    kernel.apply_event(first_partial)

    second_partial = new_authority_event(
        event_type="ASRPartialReceived",
        session_id="session-1",
        sequence_no=2,
        lineage_id="session-1:epoch:1",
        payload={"text": "hello there again"},
    )
    kernel.apply_event(second_partial)

    request_id = kernel.state.active_vllm_request_id
    assert request_id

    chunk_event = new_authority_event(
        event_type="VLLMChunkReceived",
        session_id="session-1",
        sequence_no=kernel.state.last_sequence_no + 1,
        lineage_id="session-1:epoch:1",
        payload={
            "request_id": request_id,
            "token": "stream",
            "output_version": kernel.state.output.version,
        },
        causation_id=kernel.state.request_event_id(request_id),
    )
    result = kernel.apply_event(chunk_event)

    assert any(event.event_type == "TTSRequested" for event in result.applied_events)
    tts_event = next(event for event in result.applied_events if event.event_type == "TTSRequested")
    assert str(tts_event.payload.get("text", "")) == "stream"
