from __future__ import annotations

from voice_pipeline.kernel.kernel_runtime import KernelRuntime
from voice_pipeline.shared.types import new_authority_event


def test_kernel_tick_prioritizes_interrupt_then_final_then_partial() -> None:
    kernel = KernelRuntime(session_id="priority-order")
    kernel.enqueue_event(
        new_authority_event(
            event_type="ASRPartialReceived",
            session_id="priority-order",
            sequence_no=1,
            lineage_id="priority-order:epoch:1",
            payload={"text": "partial"},
        )
    )
    kernel.enqueue_event(
        new_authority_event(
            event_type="ASRFinalReceived",
            session_id="priority-order",
            sequence_no=2,
            lineage_id="priority-order:epoch:1",
            payload={"text": "final"},
        )
    )
    kernel.enqueue_event(
        new_authority_event(
            event_type="InterruptRequested",
            session_id="priority-order",
            sequence_no=3,
            lineage_id="priority-order:epoch:1",
            payload={"reason": "SOFT_PRE_INTERRUPT"},
        )
    )

    kernel.tick()
    event_types = [event.event_type for event in kernel.event_log]
    interrupt_index = event_types.index("InterruptRequested")
    final_index = event_types.index("ASRFinalReceived")
    partial_index = event_types.index("ASRPartialReceived")

    assert interrupt_index < final_index < partial_index
