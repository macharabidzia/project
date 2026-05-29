from __future__ import annotations

import pytest

from voice_pipeline.kernel.kernel_runtime import KernelRuntime
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

