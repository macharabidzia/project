from __future__ import annotations

from dataclasses import replace

import pytest

from voice_pipeline.kernel.kernel_runtime import KernelConfig, KernelRuntime
from voice_pipeline.shared.types import new_authority_event
from voice_pipeline.transport.livekit_transport import LiveKitTransport, LiveKitTransportConfig


def test_kernel_runtime_cancel_advances_authority_without_async_execution() -> None:
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

    kernel.apply_event(
        new_authority_event(
            event_type="CancelRequested",
            session_id="session-1",
            sequence_no=2,
            lineage_id="session-1:epoch:1",
            payload={"reason": "interrupt"},
        )
    )

    assert kernel.state.active_vllm_request_id == ""
    assert kernel.state.active_vllm_request_id == ""
    assert kernel.state.phase in {"cancelled", "idle"}


def test_transport_emitter_is_execution_only() -> None:
    transport = LiveKitTransport(config=LiveKitTransportConfig())
    transport.record_ingress_frame(960 * 2)
    transport.record_egress_frame(960 * 2)
    metrics = transport.ingress_metrics()

    assert metrics["transport_ingress_frames"] == 1
    assert metrics["transport_egress_frames"] == 1
    assert metrics["transport_ingress_dropped"] == 0


def test_engine_output_requires_request_causation_and_current_version() -> None:
    kernel = KernelRuntime(session_id="session-1")
    request = new_authority_event(
        event_type="VLLMRequested",
        session_id="session-1",
        sequence_no=1,
        lineage_id="session-1:epoch:1",
        payload={"request_id": "req-1", "prompt": "hello"},
    )
    first = kernel.apply_event(request)
    request_event_id = first.applied_events[-1].event_id

    stale = new_authority_event(
        event_type="VLLMChunkReceived",
        session_id="session-1",
        sequence_no=2,
        lineage_id="session-1:epoch:1",
        payload={"request_id": "req-1", "token": "A", "output_version": 999},
        causation_id=request_event_id,
    )

    with pytest.raises(ValueError, match="output version mismatch"):
        kernel.apply_event(stale)


def test_kernel_runtime_metrics_projection_exists() -> None:
    kernel = KernelRuntime(session_id="session-1")

    metrics = kernel.runtime_metrics()

    assert metrics["ingress_queue_depth"] == 0
    assert metrics["backpressure_retune_count"] == 0
    assert metrics["vllm_drop_latency_ms"] == 0.0


def test_kernel_tick_respects_max_events_per_tick_bound() -> None:
    kernel = KernelRuntime(session_id="session-1", config=KernelConfig(max_events_per_tick=1))
    kernel.enqueue_event(
        new_authority_event(
            event_type="WorkerDetached",
            session_id="session-1",
            sequence_no=1,
            lineage_id="session-1:epoch:1",
            payload={"worker": "tts"},
        )
    )
    kernel.enqueue_event(
        new_authority_event(
            event_type="WorkerDetached",
            session_id="session-1",
            sequence_no=2,
            lineage_id="session-1:epoch:1",
            payload={"worker": "vllm"},
        )
    )

    kernel.tick()
    assert kernel.queued_event_count == 1
    assert kernel.state.last_sequence_no == 1

    kernel.tick()
    assert kernel.queued_event_count == 0
    assert kernel.state.last_sequence_no == 2


def test_kernel_queue_protects_interrupt_and_final_events() -> None:
    kernel = KernelRuntime(session_id="session-1", config=KernelConfig(ingress_max_items=2, max_events_per_tick=8))
    kernel.enqueue_event(
        new_authority_event(
            event_type="WorkerDetached",
            session_id="session-1",
            sequence_no=1,
            lineage_id="session-1:epoch:1",
            payload={"worker": "tts"},
        )
    )
    kernel.enqueue_event(
        new_authority_event(
            event_type="ASRFinalReceived",
            session_id="session-1",
            sequence_no=2,
            lineage_id="session-1:epoch:1",
            payload={"text": "hello world"},
        )
    )
    kernel.enqueue_event(
        new_authority_event(
            event_type="InterruptRequested",
            session_id="session-1",
            sequence_no=3,
            lineage_id="session-1:epoch:1",
            payload={"reason": "SOFT_PRE_INTERRUPT"},
        )
    )

    kernel.tick()
    event_types = [event.event_type for event in kernel.event_log]
    assert "InterruptRequested" in event_types
    assert "ASRFinalReceived" in event_types
    assert "WorkerDetached" not in event_types
    assert kernel.runtime_metrics()["ingress_drop_count"] >= 1


def test_tick_suppresses_stale_engine_outputs_after_cancel() -> None:
    kernel = KernelRuntime(session_id="session-1")
    request = new_authority_event(
        event_type="VLLMRequested",
        session_id="session-1",
        sequence_no=1,
        lineage_id="session-1:epoch:1",
        payload={"request_id": "req-1", "prompt": "hello"},
    )
    first = kernel.apply_event(request)
    request_event_id = first.applied_events[-1].event_id
    stale_output_version = kernel.state.output.version

    kernel.enqueue_event(
        new_authority_event(
            event_type="CancelRequested",
            session_id="session-1",
            sequence_no=2,
            lineage_id="session-1:epoch:1",
            payload={"reason": "interrupt"},
        )
    )
    kernel.enqueue_event(
        new_authority_event(
            event_type="VLLMChunkReceived",
            session_id="session-1",
            sequence_no=3,
            lineage_id="session-1:epoch:1",
            payload={"request_id": "req-1", "token": "late", "output_version": stale_output_version},
            causation_id=request_event_id,
        )
    )

    kernel.tick()
    metrics = kernel.runtime_metrics()
    assert kernel.state.active_vllm_request_id == ""
    assert kernel.queued_event_count == 0
    assert metrics["stale_token_drop_count"] >= 1


def test_kernel_reducer_handles_soft_interrupt_from_asr_partial_during_playing() -> None:
    kernel = KernelRuntime(session_id="session-1")
    kernel._state = replace(kernel.state, phase="playing", active_tts_request_id="tts-1")

    result = kernel.apply_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id="session-1",
            sequence_no=1,
            lineage_id="session-1:epoch:1",
            payload={"text": "hello there"},
        )
    )

    reasons = [str(event.payload.get("reason", "")) for event in kernel.event_log if event.event_type == "InterruptRequested"]
    assert "SOFT_PRE_INTERRUPT" in reasons
    sequence_numbers = [int(event.sequence_no) for event in kernel.event_log]
    assert sequence_numbers == sorted(sequence_numbers)
    assert kernel.state.transcript.partial_text == "hello there"
    assert kernel.state.phase in {"listening", "generating", "playing", "idle"}
    assert result.dispatch_commands == ()


def test_kernel_reducer_handles_hard_interrupt_from_asr_final_during_generating() -> None:
    kernel = KernelRuntime(session_id="session-1")
    kernel._state = replace(kernel.state, phase="generating", active_vllm_request_id="old-vllm")

    result = kernel.apply_event(
        new_authority_event(
            event_type="ASRFinalReceived",
            session_id="session-1",
            sequence_no=1,
            lineage_id="session-1:epoch:1",
            payload={"text": "new request"},
        )
    )

    reasons = [str(event.payload.get("reason", "")) for event in kernel.event_log if event.event_type == "CancelRequested"]
    assert "HARD_INTERRUPT" in reasons
    sequence_numbers = [int(event.sequence_no) for event in kernel.event_log]
    assert sequence_numbers == sorted(sequence_numbers)
    assert any(command.kind == "VLLM" for command in result.dispatch_commands)


