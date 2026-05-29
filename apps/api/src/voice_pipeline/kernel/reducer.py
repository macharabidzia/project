from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from voice_pipeline.kernel.dispatch import build_tts_command, build_vllm_command
from voice_pipeline.kernel.leases import epoch_id_for_state
from voice_pipeline.kernel.recovery import recovered_status, recovering_status
from voice_pipeline.kernel.stable_prefix import detect_stable_prefix
from voice_pipeline.kernel.state import KernelState, OutputState
from voice_pipeline.kernel.tts_fragment_planner import TTSFragmentPlannerConfig, plan_tts_fragment
from voice_pipeline.shared.lineage import canonical_turn_id
from voice_pipeline.shared.time import ns_to_ms
from voice_pipeline.shared.types import ENGINE_OUTPUT_EVENT_TYPES, AuthorityEvent, EventValidationError


@dataclass(frozen=True, slots=True)
class DerivedEvent:
    event_type: str
    payload: Mapping[str, object]
    lineage_id: str = ""


@dataclass(frozen=True, slots=True)
class ReducerDiagnostics:
    stale_token_drop_count: int = 0
    stale_pcm_drop_count: int = 0
    last_asr_event_ns: int = 0
    last_interrupt_ns: int = 0
    asr_to_dispatch_latency_ms: float = 0.0
    first_token_latency_ms: float = 0.0
    interrupt_to_new_speech_latency_ms: float = 0.0
    dispatch_started_by_request_id: tuple[tuple[str, int], ...] = ()
    first_token_seen_by_request_id: tuple[str, ...] = ()

    def request_started_ns(self, request_id: str) -> int:
        resolved = str(request_id or "").strip()
        for candidate_request_id, started_ns in reversed(self.dispatch_started_by_request_id):
            if candidate_request_id == resolved:
                return int(started_ns)
        return 0

    def mark_request_started(self, request_id: str, started_ns: int) -> "ReducerDiagnostics":
        resolved = str(request_id or "").strip()
        if not resolved:
            return self
        retained = tuple(item for item in self.dispatch_started_by_request_id if item[0] != resolved)
        return replace(self, dispatch_started_by_request_id=(*retained, (resolved, int(started_ns))))

    def has_seen_first_token(self, request_id: str) -> bool:
        return str(request_id or "").strip() in self.first_token_seen_by_request_id

    def mark_first_token_seen(self, request_id: str) -> "ReducerDiagnostics":
        resolved = str(request_id or "").strip()
        if not resolved or resolved in self.first_token_seen_by_request_id:
            return self
        return replace(self, first_token_seen_by_request_id=(*self.first_token_seen_by_request_id, resolved))


@dataclass(frozen=True, slots=True)
class ReducerConfig:
    partial_history_size: int = 6
    stable_prefix_min_repeats: int = 2
    stable_prefix_min_tokens: int = 2
    stable_prefix_max_window: int = 3
    tts_fragment_min_tokens: int = 2
    tts_fragment_max_tokens: int = 6
    tts_context_window_tokens: int = 24


@dataclass(frozen=True, slots=True)
class ReducerTransition:
    next_state: KernelState
    derived_events: tuple[DerivedEvent, ...] = ()
    dispatch_commands: tuple = ()
    diagnostics: ReducerDiagnostics = ReducerDiagnostics()


def _normalized_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _is_interrupt_replay(payload: Mapping[str, object]) -> bool:
    return bool(payload.get("_replayed_after_interrupt", False))


def _apply_event_meta(state: KernelState, event: AuthorityEvent) -> KernelState:
    return replace(
        state.remember_event(event.event_id),
        last_event_id=event.event_id,
        last_sequence_no=int(event.sequence_no),
    )


def validate_engine_output(state: KernelState, event: AuthorityEvent, diagnostics: ReducerDiagnostics) -> ReducerDiagnostics:
    if event.event_type not in ENGINE_OUTPUT_EVENT_TYPES:
        return diagnostics
    request_id = str(event.payload.get("request_id", "")).strip()
    if not request_id:
        raise EventValidationError(f"{event.event_type} requires payload.request_id")
    expected_causation_id = state.request_event_id(request_id)
    if not expected_causation_id:
        raise EventValidationError(f"no authoritative request event recorded for {request_id!r}")
    if str(event.causation_id or "").strip() != expected_causation_id:
        raise EventValidationError("causation mismatch")
    expected_output_version = state.request_output_version(request_id)
    observed_output_version = int(event.payload.get("output_version", -1))
    if observed_output_version != expected_output_version:
        if event.event_type == "VLLMChunkReceived":
            diagnostics = replace(diagnostics, stale_token_drop_count=int(diagnostics.stale_token_drop_count) + 1)
        elif event.event_type in {"TTSChunkReceived", "TTSCompleted"}:
            diagnostics = replace(diagnostics, stale_pcm_drop_count=int(diagnostics.stale_pcm_drop_count) + 1)
        raise EventValidationError("output version mismatch")
    return diagnostics


def reduce_event(
    state: KernelState,
    event: AuthorityEvent,
    *,
    config: ReducerConfig | None = None,
    diagnostics: ReducerDiagnostics | None = None,
    event_time_ns: int = 0,
) -> ReducerTransition:
    resolved_config = config or ReducerConfig()
    resolved_diagnostics = diagnostics or ReducerDiagnostics()
    resolved_diagnostics = validate_engine_output(state, event, resolved_diagnostics)
    state = _apply_event_meta(state, event)
    payload = dict(event.payload)
    lineage_id = str(event.lineage_id or state.lineage_id or "").strip()

    if event.event_type == "ASRPartialReceived":
        resolved_diagnostics = replace(resolved_diagnostics, last_asr_event_ns=int(event_time_ns))
        partial_text = _normalized_text(payload.get("text", ""))
        if state.phase == "playing" and not _is_interrupt_replay(payload):
            derived: list[DerivedEvent] = [
                DerivedEvent(
                    event_type="InterruptRequested",
                    payload={"reason": "SOFT_PRE_INTERRUPT", "source": "asr_partial", "lane": "kernel"},
                    lineage_id=lineage_id,
                )
            ]
            if partial_text:
                derived.append(
                    DerivedEvent(
                        event_type="ASRPartialReceived",
                        payload={"text": partial_text, "_replayed_after_interrupt": True},
                        lineage_id=lineage_id,
                    )
                )
            return ReducerTransition(
                next_state=replace(state, lineage_id=lineage_id or state.lineage_id),
                derived_events=tuple(derived),
                diagnostics=resolved_diagnostics,
            )
        if not partial_text:
            return ReducerTransition(next_state=replace(state, phase="listening", lineage_id=lineage_id), diagnostics=resolved_diagnostics)

        history = (*state.transcript.partial_history, partial_text)
        history_size = max(2, int(resolved_config.partial_history_size))
        if len(history) > history_size:
            history = history[-history_size:]

        stable_decision = detect_stable_prefix(
            history,
            min_repeats=int(resolved_config.stable_prefix_min_repeats),
            min_tokens=int(resolved_config.stable_prefix_min_tokens),
            max_window=int(resolved_config.stable_prefix_max_window),
        )
        transcript = replace(
            state.transcript,
            partial_text=partial_text,
            partial_history=history,
            stable_prefix=stable_decision.prefix,
            stable_prefix_confirmations=int(stable_decision.confirmations),
        )
        next_state = replace(state, phase="listening", transcript=transcript, lineage_id=lineage_id)

        if (
            stable_decision.prefix
            and not state.active_vllm_request_id
            and state.phase in {"idle", "listening"}
            and stable_decision.prefix != state.transcript.last_dispatched_stable_prefix
            and stable_decision.prefix != state.transcript.committed_text
        ):
            next_state = replace(
                next_state,
                transcript=replace(next_state.transcript, last_dispatched_stable_prefix=stable_decision.prefix),
            )
            return ReducerTransition(
                next_state=next_state,
                derived_events=(
                    DerivedEvent(
                        event_type="TurnCommitted",
                        payload={"text": stable_decision.prefix, "commit_source": "stable_prefix"},
                        lineage_id=lineage_id,
                    ),
                ),
                diagnostics=resolved_diagnostics,
            )

        return ReducerTransition(next_state=next_state, diagnostics=resolved_diagnostics)

    if event.event_type == "ASRFinalReceived":
        resolved_diagnostics = replace(resolved_diagnostics, last_asr_event_ns=int(event_time_ns))
        final_text = _normalized_text(payload.get("text", ""))
        if state.phase in {"playing", "generating"} and not _is_interrupt_replay(payload):
            derived: list[DerivedEvent] = [
                DerivedEvent(
                    event_type="CancelRequested",
                    payload={"reason": "HARD_INTERRUPT", "source": "asr_final", "lane": "kernel"},
                    lineage_id=lineage_id,
                )
            ]
            if final_text:
                derived.append(
                    DerivedEvent(
                        event_type="ASRFinalReceived",
                        payload={"text": final_text, "_replayed_after_interrupt": True},
                        lineage_id=lineage_id,
                    )
                )
            return ReducerTransition(
                next_state=replace(state, lineage_id=lineage_id or state.lineage_id),
                derived_events=tuple(derived),
                diagnostics=resolved_diagnostics,
            )
        transcript = replace(
            state.transcript,
            partial_text="",
            partial_history=(),
            stable_prefix="",
            stable_prefix_confirmations=0,
            final_text=final_text,
        )
        listening_state = replace(state, phase="listening", transcript=transcript, lineage_id=lineage_id)
        should_start = bool(final_text) and not state.active_vllm_request_id and final_text != state.transcript.committed_text
        if not should_start:
            return ReducerTransition(next_state=listening_state, diagnostics=resolved_diagnostics)
        return ReducerTransition(
            next_state=listening_state,
            derived_events=(
                DerivedEvent(
                    event_type="TurnCommitted",
                    payload={"text": final_text, "commit_source": "final"},
                    lineage_id=lineage_id,
                ),
            ),
            diagnostics=resolved_diagnostics,
        )

    if event.event_type == "TurnCommitted":
        text = _normalized_text(payload.get("text", ""))
        if not text:
            return ReducerTransition(next_state=replace(state, lineage_id=lineage_id or state.lineage_id), diagnostics=resolved_diagnostics)

        if resolved_diagnostics.last_interrupt_ns:
            resolved_diagnostics = replace(
                resolved_diagnostics,
                interrupt_to_new_speech_latency_ms=ns_to_ms(int(event_time_ns) - int(resolved_diagnostics.last_interrupt_ns)),
                last_interrupt_ns=0,
            )

        committed_turn_index = state.committed_turn_index + 1
        generation_index = state.generation_index + 1
        active_turn_id = canonical_turn_id(session_id=state.session_id, turn_epoch=committed_turn_index)
        next_state = replace(
            state,
            phase="generating",
            transcript=replace(
                state.transcript,
                partial_text="",
                partial_history=(),
                stable_prefix="",
                stable_prefix_confirmations=0,
                final_text=text,
                committed_text=text,
                last_dispatched_stable_prefix=text,
                conversation_history=(*state.transcript.conversation_history, text),
            ),
            output=OutputState(active_turn_id=active_turn_id, version=int(state.output.version) + 1),
            turn_index=max(int(state.turn_index), committed_turn_index),
            committed_turn_index=committed_turn_index,
            generation_index=generation_index,
            lineage_id=lineage_id,
        )
        return ReducerTransition(
            next_state=next_state,
            derived_events=(
                DerivedEvent(
                    event_type="VLLMRequested",
                    payload={
                        "prompt": text,
                        "prompt_cache_key": str(payload.get("prompt_cache_key", text)),
                        "commit_source": str(payload.get("commit_source", "")),
                    },
                    lineage_id=lineage_id,
                ),
            ),
            diagnostics=resolved_diagnostics,
        )

    if event.event_type == "VLLMRequested":
        request_id = str(payload.get("request_id", "")).strip() or f"{lineage_id}:vllm"
        next_state = replace(state, phase="generating", active_vllm_request_id=request_id, lineage_id=lineage_id)
        next_state = next_state.bind_request_event(request_id, event.event_id, output_version=next_state.output.version)
        resolved_diagnostics = resolved_diagnostics.mark_request_started(request_id, int(event_time_ns))
        if resolved_diagnostics.last_asr_event_ns:
            resolved_diagnostics = replace(
                resolved_diagnostics,
                asr_to_dispatch_latency_ms=ns_to_ms(int(event_time_ns) - int(resolved_diagnostics.last_asr_event_ns)),
            )
        return ReducerTransition(
            next_state=next_state,
            dispatch_commands=(
                build_vllm_command(
                    session_id=state.session_id,
                    request_id=request_id,
                    prompt=str(payload.get("prompt", "")),
                    prompt_cache_key=str(payload.get("prompt_cache_key", payload.get("prompt", ""))),
                    output_version=int(next_state.output.version),
                    lineage_id=lineage_id,
                    turn_id=str(next_state.output.active_turn_id),
                    epoch_id=epoch_id_for_state(next_state),
                ),
            ),
            diagnostics=resolved_diagnostics,
        )

    if event.event_type == "VLLMChunkReceived":
        request_id = str(payload.get("request_id", "")).strip()
        token = str(payload.get("token", "")).strip()
        if not token:
            return ReducerTransition(next_state=replace(state, phase="generating", lineage_id=lineage_id), diagnostics=resolved_diagnostics)

        if request_id and not resolved_diagnostics.has_seen_first_token(request_id):
            started_ns = resolved_diagnostics.request_started_ns(request_id)
            if started_ns:
                resolved_diagnostics = replace(
                    resolved_diagnostics,
                    first_token_latency_ms=ns_to_ms(int(event_time_ns) - int(started_ns)),
                )
            resolved_diagnostics = resolved_diagnostics.mark_first_token_seen(request_id)

        stream_buffer = (*state.output.vllm_stream_buffer, token)
        output = replace(state.output, vllm_tokens=(*state.output.vllm_tokens, token), vllm_stream_buffer=stream_buffer)
        next_state = replace(state, phase="generating", output=output)
        stable_prefix_context = state.transcript.committed_text or state.transcript.stable_prefix

        tts_plan = plan_tts_fragment(
            stream_buffer,
            stable_prefix=stable_prefix_context,
            config=TTSFragmentPlannerConfig(
                min_tokens=int(resolved_config.tts_fragment_min_tokens),
                max_tokens=int(resolved_config.tts_fragment_max_tokens),
                context_window_tokens=int(resolved_config.tts_context_window_tokens),
                start_on_stable_prefix=True,
            ),
        )
        if tts_plan.flush_text:
            next_state = replace(
                next_state,
                output=replace(next_state.output, vllm_stream_buffer=tts_plan.remaining_tokens),
            )
            return ReducerTransition(
                next_state=next_state,
                derived_events=(
                    DerivedEvent(
                        event_type="TTSRequested",
                        payload={
                            "text": tts_plan.flush_text,
                            "stream_fragment": bool(tts_plan.remaining_tokens),
                            "reason": tts_plan.reason,
                        },
                        lineage_id=lineage_id,
                    ),
                ),
                diagnostics=resolved_diagnostics,
            )

        return ReducerTransition(next_state=next_state, diagnostics=resolved_diagnostics)

    if event.event_type == "VLLMCompleted":
        text = _normalized_text(payload.get("text", "")) or "".join(state.output.vllm_tokens)
        stable_prefix_context = state.transcript.committed_text or state.transcript.stable_prefix
        tts_plan = plan_tts_fragment(
            state.output.vllm_stream_buffer,
            stable_prefix=stable_prefix_context,
            drain=True,
            config=TTSFragmentPlannerConfig(
                min_tokens=int(resolved_config.tts_fragment_min_tokens),
                max_tokens=int(resolved_config.tts_fragment_max_tokens),
                context_window_tokens=int(resolved_config.tts_context_window_tokens),
                start_on_stable_prefix=True,
            ),
        )
        next_state = replace(state, active_vllm_request_id="", phase="idle")
        if not tts_plan.flush_text and not text:
            return ReducerTransition(next_state=next_state, diagnostics=resolved_diagnostics)

        if tts_plan.flush_text:
            next_state = replace(next_state, output=replace(next_state.output, vllm_stream_buffer=()))
            return ReducerTransition(
                next_state=next_state,
                derived_events=(
                    DerivedEvent(
                        event_type="TTSRequested",
                        payload={"text": tts_plan.flush_text, "stream_fragment": tts_plan.stream_fragment, "reason": tts_plan.reason},
                        lineage_id=lineage_id,
                    ),
                ),
                diagnostics=resolved_diagnostics,
            )

        return ReducerTransition(
            next_state=next_state,
            derived_events=(DerivedEvent(event_type="TTSRequested", payload={"text": text, "stream_fragment": False}, lineage_id=lineage_id),),
            diagnostics=resolved_diagnostics,
        )

    if event.event_type == "TTSRequested":
        request_id = str(payload.get("request_id", "")).strip() or f"{lineage_id}:tts"
        text = str(payload.get("text", "")).strip()
        if not text:
            return ReducerTransition(next_state=replace(state, lineage_id=lineage_id or state.lineage_id), diagnostics=resolved_diagnostics)
        output = replace(state.output, pending_tts_segments=(*state.output.pending_tts_segments, text))
        next_state = replace(state, phase="playing", active_tts_request_id=request_id, output=output)
        next_state = next_state.bind_request_event(request_id, event.event_id, output_version=next_state.output.version)
        return ReducerTransition(
            next_state=next_state,
            dispatch_commands=(
                build_tts_command(
                    session_id=state.session_id,
                    request_id=request_id,
                    text=text,
                    output_version=int(next_state.output.version),
                    lineage_id=lineage_id,
                    turn_id=str(next_state.output.active_turn_id),
                    epoch_id=epoch_id_for_state(next_state),
                    stream_fragment=bool(payload.get("stream_fragment", False)),
                ),
            ),
            diagnostics=resolved_diagnostics,
        )

    if event.event_type == "TTSChunkReceived":
        chunk_id = str(payload.get("chunk_id", event.event_id))
        output = replace(state.output, emitted_audio_chunk_ids=(*state.output.emitted_audio_chunk_ids, chunk_id))
        return ReducerTransition(next_state=replace(state, phase="playing", output=output), diagnostics=resolved_diagnostics)

    if event.event_type == "TTSCompleted":
        next_phase = "generating" if state.active_vllm_request_id else "idle"
        return ReducerTransition(
            next_state=replace(state, phase=next_phase, active_tts_request_id="", output=replace(state.output, pending_tts_segments=())),
            diagnostics=resolved_diagnostics,
        )

    if event.event_type in {"InterruptRequested", "CancelRequested"}:
        resolved_diagnostics = replace(resolved_diagnostics, last_interrupt_ns=int(event_time_ns))
        return ReducerTransition(
            next_state=replace(
                state,
                phase="cancelled",
                active_tts_request_id="",
                active_vllm_request_id="",
                generation_index=int(state.generation_index) + 1,
                request_event_ids=(),
                transcript=replace(
                    state.transcript,
                    partial_text="",
                    partial_history=(),
                    stable_prefix="",
                    stable_prefix_confirmations=0,
                ),
                output=OutputState(version=int(state.output.version) + 1),
            ),
            derived_events=(DerivedEvent(event_type="RecoveryCompleted", payload={"reason": "cancel_reset"}, lineage_id=lineage_id),),
            diagnostics=resolved_diagnostics,
        )

    if event.event_type == "RecoveryRequested":
        return ReducerTransition(
            next_state=replace(
                state,
                phase="cancelled",
                active_tts_request_id="",
                active_vllm_request_id="",
                generation_index=int(state.generation_index) + 1,
                request_event_ids=(),
                output=OutputState(version=int(state.output.version) + 1),
                recovery=recovering_status(str(payload.get("reason", ""))),
            ),
            diagnostics=resolved_diagnostics,
        )

    if event.event_type == "RecoveryCompleted":
        return ReducerTransition(
            next_state=replace(state, phase="idle", recovery=recovered_status(str(payload.get("reason", "")))),
            diagnostics=resolved_diagnostics,
        )

    return ReducerTransition(next_state=replace(state, lineage_id=lineage_id or state.lineage_id), diagnostics=resolved_diagnostics)


__all__ = [
    "DerivedEvent",
    "ReducerConfig",
    "ReducerDiagnostics",
    "ReducerTransition",
    "reduce_event",
    "validate_engine_output",
]
