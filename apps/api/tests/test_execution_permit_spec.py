from __future__ import annotations

from voice_pipeline.kernel.kernel_runtime import KernelRuntime
from voice_pipeline.shared.types import new_authority_event


def test_kernel_runtime_tick_emits_vllm_command_for_committed_asr_final() -> None:
    kernel = KernelRuntime(session_id="session-1")
    kernel.enqueue_event(
        new_authority_event(
            event_type="ASRFinalReceived",
            session_id="session-1",
            sequence_no=1,
            lineage_id="session-1:epoch:1",
            payload={"text": "hello"},
        )
    )

    commands = kernel.tick()

    assert [command.kind for command in commands] == ["VLLM"]
    assert commands[0].payload["prompt"] == "hello"
    assert kernel.state.phase == "generating"


def test_kernel_runtime_commit_result_exposes_dispatch_commands() -> None:
    kernel = KernelRuntime(session_id="session-1")
    result = kernel.apply_event(
        new_authority_event(
            event_type="VLLMRequested",
            session_id="session-1",
            sequence_no=1,
            lineage_id="session-1:epoch:1",
            payload={"request_id": "req-1", "prompt": "hello"},
        )
    )

    assert [command.kind for command in result.dispatch_commands] == ["VLLM"]
