from __future__ import annotations

from voice_pipeline.kernel.kernel_runtime import KernelRuntime
from voice_pipeline.shared.types import new_authority_event


def test_stale_llm_and_tts_outputs_are_suppressed_after_cancel() -> None:
    kernel = KernelRuntime(session_id="stale-suppression")
    request = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="stale-suppression",
            sequence_no=1,
            lineage_id="stale-suppression:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )
    request_event_id = request.applied_events[-1].event_id
    stale_output_version = kernel.state.request_output_version("req-1")

    kernel.enqueue_event(
        new_authority_event(
            event_type="CancelRequested",
            session_id="stale-suppression",
            sequence_no=2,
            lineage_id="stale-suppression:epoch:1",
            payload={"reason": "HARD_INTERRUPT"},
        )
    )
    kernel.enqueue_event(
        new_authority_event(
            event_type="VLLMChunkReceived",
            session_id="stale-suppression",
            sequence_no=3,
            lineage_id="stale-suppression:epoch:1",
            payload={"request_id": "req-1", "token": "late", "output_version": stale_output_version},
            causation_id=request_event_id,
        )
    )
    kernel.enqueue_event(
        new_authority_event(
            event_type="TTSChunkReceived",
            session_id="stale-suppression",
            sequence_no=4,
            lineage_id="stale-suppression:epoch:1",
            payload={"request_id": "req-1", "chunk_id": "late-1", "output_version": stale_output_version},
            causation_id=request_event_id,
        )
    )

    kernel.tick()
    metrics = kernel.runtime_metrics()
    assert metrics["stale_token_drop_count"] >= 1
    assert metrics["stale_pcm_drop_count"] >= 1
