from __future__ import annotations

from dataclasses import dataclass

from voice_pipeline.kernel.kernel_runtime import KernelRuntime
from voice_pipeline.shared.types import AuthorityEvent, new_authority_event
from voice_pipeline.kernel.state import KernelState


@dataclass(frozen=True, slots=True)
class ReplayBaseline:
    name: str
    events: tuple[AuthorityEvent, ...]
    expected_state: KernelState
    duplicate_event_ids: tuple[str, ...] = ()


def _baseline(name: str, session_id: str, events: list[AuthorityEvent], duplicate_event_ids: tuple[str, ...] = ()) -> ReplayBaseline:
    kernel = KernelRuntime(session_id=session_id)
    result = kernel.replay(events)
    return ReplayBaseline(
        name=name,
        events=tuple(events),
        expected_state=result.state,
        duplicate_event_ids=duplicate_event_ids,
    )


def build_replay_baselines(*, session_id: str) -> tuple[ReplayBaseline, ...]:
    baselines: list[ReplayBaseline] = []

    happy_loop = KernelRuntime(session_id=session_id)
    happy_events = [
        new_authority_event(
            event_type="ASRFinalReceived",
            session_id=session_id,
            sequence_no=1,
            lineage_id=f"{session_id}:epoch:1",
            payload={"text": "hello world", "end_of_speech_observed": True},
        )
    ]
    happy_result = happy_loop.replay(happy_events)
    vllm_request_event = next(event for event in happy_result.applied_events if event.event_type == "VLLMRequested")
    happy_events.extend(
        [
            new_authority_event(
                event_type="VLLMChunkReceived",
                session_id=session_id,
                sequence_no=3,
                lineage_id=f"{session_id}:epoch:1",
                payload={"request_id": f"{session_id}:epoch:1:vllm", "token": "hi ", "output_version": 1},
                causation_id=vllm_request_event.event_id,
            ),
            new_authority_event(
                event_type="VLLMCompleted",
                session_id=session_id,
                sequence_no=4,
                lineage_id=f"{session_id}:epoch:1",
                payload={"request_id": f"{session_id}:epoch:1:vllm", "text": "hi there", "output_version": 1},
                causation_id=vllm_request_event.event_id,
            ),
        ]
    )
    baselines.append(_baseline("happy_path", session_id, happy_events))

    return tuple(baselines)


__all__ = ["ReplayBaseline", "build_replay_baselines"]


