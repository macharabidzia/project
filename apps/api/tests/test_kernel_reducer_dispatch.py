from __future__ import annotations

from voice_pipeline.kernel.kernel_runtime import KernelRuntime
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
