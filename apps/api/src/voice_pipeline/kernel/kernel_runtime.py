from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from threading import Lock

from voice_pipeline.bus.ring_topology import RingTopology
from voice_pipeline.kernel.invariant_loop import post_tick_validate, pre_tick_validate
from voice_pipeline.kernel.latency_contract import LaneLatencyBudget, PipelineLatencySample, evaluate_latency
from voice_pipeline.kernel.leases import lease_snapshot
from voice_pipeline.kernel.ordering import expected_next_sequence, make_derived_event, push_front
from voice_pipeline.kernel.reducer import ReducerConfig, ReducerDiagnostics, ReducerTransition, reduce_event
from voice_pipeline.kernel.tick_engine import TickEngine
from voice_pipeline.replay.validator import assert_event_identity_closure
from voice_pipeline.runtime.topology import RuntimeTopology
from voice_pipeline.shared.lineage import canonical_session_id
from voice_pipeline.shared.time import now_ns, ns_to_ms
from voice_pipeline.shared.types import ENGINE_OUTPUT_EVENT_TYPES, AuthorityEvent, EventValidationError, validate_authority_event


@dataclass(frozen=True, slots=True)
class KernelConfig:
    ingress_max_items: int = 2048
    partial_history_size: int = 6
    stable_prefix_min_repeats: int = 2
    stable_prefix_min_tokens: int = 2
    stable_prefix_max_window: int = 3
    tick_interval_ms: int = 2
    tts_fragment_min_tokens: int = 2
    tts_fragment_max_tokens: int = 6
    tts_context_window_tokens: int = 24
    latency_budget_ms: float = 150.0
    max_events_per_tick: int = 64

    def reducer_config(self) -> ReducerConfig:
        return ReducerConfig(
            partial_history_size=int(self.partial_history_size),
            stable_prefix_min_repeats=int(self.stable_prefix_min_repeats),
            stable_prefix_min_tokens=int(self.stable_prefix_min_tokens),
            stable_prefix_max_window=int(self.stable_prefix_max_window),
            tts_fragment_min_tokens=int(self.tts_fragment_min_tokens),
            tts_fragment_max_tokens=int(self.tts_fragment_max_tokens),
            tts_context_window_tokens=int(self.tts_context_window_tokens),
        )


@dataclass(frozen=True, slots=True)
class KernelCommitResult:
    state: object
    applied_events: tuple[AuthorityEvent, ...] = ()
    dispatch_commands: tuple[object, ...] = ()
    duplicate_event_ids: tuple[str, ...] = ()


class KernelRuntime:
    def __init__(
        self,
        *,
        session_id: str,
        config: KernelConfig | None = None,
        topology: RuntimeTopology | None = None,
        rings: RingTopology | None = None,
    ) -> None:
        from voice_pipeline.kernel.state import KernelState

        self.session_id = canonical_session_id(session_id)
        self.config = config or KernelConfig()
        self._topology = topology
        self._rings = rings
        self._state = KernelState(session_id=self.session_id)
        self._diagnostics = ReducerDiagnostics()
        self._event_log: list[AuthorityEvent] = []
        self._apply_lock = Lock()
        self._in_apply = False
        self._queue_capacity = max(1, int(self.config.ingress_max_items))
        self._queued_events: deque[AuthorityEvent] = deque()
        self._queued_event_enqueued_ns: dict[str, int] = {}
        self._ingress_drop_count = 0
        self._tick_engine = TickEngine(interval_ms=int(self.config.tick_interval_ms))
        self._last_tick_ms = 0.0
        self._latency_budget = LaneLatencyBudget(total_ms=float(self.config.latency_budget_ms))

    @property
    def topology(self) -> RuntimeTopology | None:
        return self._topology

    @property
    def rings(self) -> RingTopology | None:
        return self._rings

    @property
    def state(self):
        return self._state

    @property
    def event_log(self) -> tuple[AuthorityEvent, ...]:
        return tuple(self._event_log)

    @property
    def queued_event_count(self) -> int:
        return int(len(self._queued_events))

    def next_sequence_no(self) -> int:
        return int(self._state.last_sequence_no) + int(len(self._queued_events)) + 1

    def current_lease(self):
        return lease_snapshot(self._state)

    def _commit_transition(self, transition: ReducerTransition) -> None:
        self._state = transition.next_state
        self._diagnostics = transition.diagnostics

    def apply_event(self, event: AuthorityEvent) -> KernelCommitResult:
        with self._apply_lock:
            if self._in_apply:
                raise RuntimeError("reducer loop is not re-entrant")
            self._in_apply = True
            try:
                queue: deque[AuthorityEvent] = deque([event])
                applied_events: list[AuthorityEvent] = []
                dispatch_commands: list[object] = []
                duplicate_event_ids: list[str] = []
                while queue:
                    current = queue.popleft()
                    if current.event_id in self._state.recent_event_ids:
                        duplicate_event_ids.append(current.event_id)
                        continue
                    expected_sequence = expected_next_sequence(self._state)
                    if int(current.sequence_no) != int(expected_sequence):
                        current = self._with_sequence_no(current, expected_sequence)
                    validate_authority_event(
                        current,
                        expected_session_id=self.session_id,
                        expected_sequence_no=expected_sequence,
                    )
                    assert_event_identity_closure(current)
                    transition = reduce_event(
                        self._state,
                        current,
                        config=self.config.reducer_config(),
                        diagnostics=self._diagnostics,
                        event_time_ns=now_ns(),
                    )
                    self._commit_transition(transition)
                    self._event_log.append(current)
                    applied_events.append(current)
                    dispatch_commands.extend(transition.dispatch_commands)
                    next_sequence = expected_next_sequence(self._state)
                    derived_events = tuple(
                        make_derived_event(
                            session_id=self.session_id,
                            state=self._state,
                            parent_event=current,
                            derived_event=derived_event,
                            sequence_no=next_sequence + index,
                        )
                        for index, derived_event in enumerate(transition.derived_events)
                    )
                    push_front(queue, derived_events)
                return KernelCommitResult(
                    state=self._state,
                    applied_events=tuple(applied_events),
                    dispatch_commands=tuple(dispatch_commands),
                    duplicate_event_ids=tuple(duplicate_event_ids),
                )
            finally:
                self._in_apply = False

    def replay(self, events: list[AuthorityEvent] | tuple[AuthorityEvent, ...]) -> KernelCommitResult:
        result = KernelCommitResult(state=self._state)
        for event in events:
            result = self.apply_event(event)
        return result

    def enqueue_event(self, event: AuthorityEvent) -> None:
        self._drop_superseded_partial_events(event)
        if len(self._queued_events) >= self._queue_capacity:
            if self._is_protected_event_type(event.event_type):
                if not self._evict_low_priority_for_protected_event():
                    raise RuntimeError("ingress_queue_overflow_protected_event")
            else:
                self._ingress_drop_count += 1
                return
        self._queued_events.append(event)
        self._queued_event_enqueued_ns[str(event.event_id)] = now_ns()

    def tick(self):
        previous_sequence = int(self._state.last_sequence_no)
        starting_epoch = self.current_lease().epoch_id
        if self._rings is not None:
            pre_tick_validate(
                epoch=starting_epoch,
                ring_depth=int(self._rings.kernel_stream_ring.depth),
                slots=[slot for slot in self._rings.kernel_stream_ring._items if slot is not None],
            )

        dispatch_commands: list[object] = []
        processed = 0
        max_events = max(1, int(self.config.max_events_per_tick))
        while self._queued_events and processed < max_events:
            queued_event = self._pop_next_queued_event()
            if queued_event is None:
                break
            expected_sequence = expected_next_sequence(self._state)
            if int(queued_event.sequence_no) != int(expected_sequence):
                queued_event = self._with_sequence_no(queued_event, expected_sequence)
            try:
                result = self.apply_event(queued_event)
            except EventValidationError as exc:
                if self._is_suppressible_stale_engine_output(queued_event, exc):
                    if queued_event.event_type == "VLLMChunkReceived":
                        self._diagnostics = replace(
                            self._diagnostics,
                            stale_token_drop_count=int(self._diagnostics.stale_token_drop_count) + 1,
                        )
                    elif queued_event.event_type in {"TTSChunkReceived", "TTSCompleted"}:
                        self._diagnostics = replace(
                            self._diagnostics,
                            stale_pcm_drop_count=int(self._diagnostics.stale_pcm_drop_count) + 1,
                        )
                    continue
                raise
            dispatch_commands.extend(result.dispatch_commands)
            processed += 1

        ending_epoch = self.current_lease().epoch_id
        post_tick_validate(
            epoch=ending_epoch,
            result_epoch=ending_epoch,
            last_seq=int(self._state.last_sequence_no),
            prev_seq=previous_sequence,
        )
        return tuple(dispatch_commands)

    @staticmethod
    def _is_protected_event_type(event_type: str) -> bool:
        return str(event_type or "").strip() in {"InterruptRequested", "CancelRequested", "ASRFinalReceived"}

    @staticmethod
    def _event_priority(event_type: str) -> int:
        resolved = str(event_type or "").strip()
        if resolved in {"InterruptRequested", "CancelRequested"}:
            return 0
        if resolved == "ASRFinalReceived":
            return 1
        if resolved == "ASRPartialReceived":
            return 2
        if resolved.startswith("VLLM") or resolved == "TurnCommitted":
            return 3
        if resolved.startswith("TTS"):
            return 4
        return 5

    def _drop_superseded_partial_events(self, incoming: AuthorityEvent) -> None:
        if str(incoming.event_type) != "ASRPartialReceived":
            return
        lineage_id = str(incoming.lineage_id or "").strip()
        if not lineage_id:
            return
        retained: list[AuthorityEvent] = []
        for queued in self._queued_events:
            if str(queued.event_type) == "ASRPartialReceived" and str(queued.lineage_id or "").strip() == lineage_id:
                self._queued_event_enqueued_ns.pop(str(queued.event_id), None)
                self._ingress_drop_count += 1
                continue
            retained.append(queued)
        self._queued_events = deque(retained)

    def _evict_low_priority_for_protected_event(self) -> bool:
        if not self._queued_events:
            return False
        candidates = [index for index, event in enumerate(self._queued_events) if not self._is_protected_event_type(event.event_type)]
        if not candidates:
            return False
        index = max(candidates, key=lambda idx: self._event_priority(self._queued_events[idx].event_type))
        events = list(self._queued_events)
        dropped = events.pop(index)
        self._queued_events = deque(events)
        self._queued_event_enqueued_ns.pop(str(dropped.event_id), None)
        self._ingress_drop_count += 1
        return True

    def _pop_next_queued_event(self) -> AuthorityEvent | None:
        if not self._queued_events:
            return None
        best_index = 0
        best_priority = self._event_priority(self._queued_events[0].event_type)
        for index, event in enumerate(self._queued_events):
            priority = self._event_priority(event.event_type)
            if priority < best_priority:
                best_index = index
                best_priority = priority
        events = list(self._queued_events)
        chosen = events.pop(best_index)
        self._queued_events = deque(events)
        self._queued_event_enqueued_ns.pop(str(chosen.event_id), None)
        return chosen

    @staticmethod
    def _with_sequence_no(event: AuthorityEvent, sequence_no: int) -> AuthorityEvent:
        return AuthorityEvent(
            schema_version=event.schema_version,
            event_type=event.event_type,
            event_id=event.event_id,
            session_id=event.session_id,
            sequence_no=int(sequence_no),
            lineage_id=event.lineage_id,
            causation_id=event.causation_id,
            payload=event.payload,
            observations=event.observations,
        )

    @staticmethod
    def _is_suppressible_stale_engine_output(event: AuthorityEvent, error: EventValidationError) -> bool:
        if event.event_type not in ENGINE_OUTPUT_EVENT_TYPES:
            return False
        message = str(error or "")
        return any(
            token in message
            for token in (
                "output version mismatch",
                "causation mismatch",
                "no authoritative request event",
            )
        )

    def commit_tick(self):
        elapsed_ns = self._tick_engine.run_once(self.tick)
        self._last_tick_ms = ns_to_ms(elapsed_ns)
        return elapsed_ns

    def runtime_metrics(self) -> dict[str, object]:
        latency_sample = PipelineLatencySample(
            asr_ms=float(self._diagnostics.asr_to_dispatch_latency_ms),
            kernel_ms=float(self._last_tick_ms),
            vllm_ms=float(self._diagnostics.first_token_latency_ms),
            tts_ms=0.0,
            transport_ms=0.0,
            queue_depth=int(len(self._queued_events)),
        )
        latency_decision = evaluate_latency(latency_sample, budget=self._latency_budget)
        drift = self._tick_engine.drift_snapshot()
        if self._queued_events:
            oldest_event_id = str(self._queued_events[0].event_id)
            queued_at_ns = int(self._queued_event_enqueued_ns.get(oldest_event_id, now_ns()))
            ingress_oldest_age_ms = ns_to_ms(max(0, now_ns() - queued_at_ns))
        else:
            ingress_oldest_age_ms = 0.0
        return {
            "ingress_queue_depth": int(len(self._queued_events)),
            "ingress_queue_oldest_age_ms": float(ingress_oldest_age_ms),
            "ingress_drop_count": int(self._ingress_drop_count),
            "tick_interval_ms": float(drift["target_tick_ms"]),
            "last_tick_ms": float(self._last_tick_ms),
            "tick_drift_ms": float(drift["last_drift_us"]) / 1000.0,
            "tick_correction_ms": float(drift["drift_correction_ns"]) / 1_000_000.0,
            "tick_catch_up_count": int(drift["catch_up_ticks"]),
            "worst_interrupt_latency_ms": float(drift["worst_interrupt_latency_us"]) / 1000.0,
            "asr_queue_age_ms": 0.0,
            "vllm_queue_age_ms": 0.0,
            "tts_queue_age_ms": 0.0,
            "backpressure_retune_count": 0,
            "tts_drop_latency_ms": 0.0,
            "vllm_drop_latency_ms": 0.0,
            "asr_to_dispatch_ms": float(self._diagnostics.asr_to_dispatch_latency_ms),
            "dispatch_to_first_token_ms": float(self._diagnostics.first_token_latency_ms),
            "first_token_to_first_pcm_ms": 0.0,
            "first_pcm_latency_ms": 0.0,
            "interrupt_to_new_speech_ms": float(self._diagnostics.interrupt_to_new_speech_latency_ms),
            "stale_token_drop_count": int(self._diagnostics.stale_token_drop_count),
            "stale_pcm_drop_count": int(self._diagnostics.stale_pcm_drop_count),
            "latency_action": latency_decision.action,
            "latency_reason": latency_decision.reason,
            "latency_over_budget_ms": float(latency_decision.over_budget_ms),
            "backpressure_action": latency_decision.action,
        }


__all__ = ["KernelCommitResult", "KernelConfig", "KernelRuntime"]
