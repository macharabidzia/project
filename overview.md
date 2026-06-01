# Voice Pipeline Python Overview

Generated from `apps/api/src/voice_pipeline` on 2026-06-01.

Includes every Python module currently present in the package, with top-level classes, methods, and functions plus short descriptions.

Total Python files: 57

## Bus

### bus/__init__.py
Module with runtime support code.

- No top-level classes discovered in this module.

- No top-level functions discovered in this module.

### bus/ring_topology.py
Module with runtime support code.

### Classes
- _LaneRingView: Container or runtime type for Lane Ring View.
  - _LaneRingView.__init__: Initializes the object.
  - _LaneRingView.push: Handles push.
  - _LaneRingView.pop: Handles pop.
- RingTopology: Container or runtime type for Ring Topology.
  - RingTopology.with_capacity: Builds capacity.

- No top-level functions discovered in this module.

### bus/ring_types.py
Module with runtime support code.

### Classes
- EventType: Container or runtime type for Event Type.
  - No methods defined in this class.
- LaneId: Container or runtime type for Lane Id.
  - No methods defined in this class.
- SlotType: Container or runtime type for Slot Type.
  - No methods defined in this class.
- KernelSlotABI: Container or runtime type for Kernel Slot A B I.
  - No methods defined in this class.
- RingSlot: Container or runtime type for Ring Slot.
  - RingSlot.__post_init__: Normalizes and validates fields after initialization.

### Functions
- event_type_lane: Handles event type lane.
- event_type_slot_type: Handles event type slot type.
- assert_kernel_slot_abi: Asserts kernel slot abi.

### bus/shm_ring.py
Module with runtime support code.

### Classes
- SharedMemoryRing: In-process bounded ring used by the single-runtime voice pipeline.
  - SharedMemoryRing.__init__: Initializes the object.
  - SharedMemoryRing.capacity: Handles capacity.
  - SharedMemoryRing.depth: Handles depth.
  - SharedMemoryRing.overwrite_count: Handles overwrite count.
  - SharedMemoryRing.shared_memory_name: Handles shared memory name.
  - SharedMemoryRing.oldest_age_ms: Handles oldest age ms.
  - SharedMemoryRing.push: Handles push.
  - SharedMemoryRing.pop: Handles pop.
  - SharedMemoryRing.drain: Handles drain.
  - SharedMemoryRing.push_bytes: Pushes bytes.
  - SharedMemoryRing.read_slot_bytes: Reads slot bytes.
  - SharedMemoryRing.close: Handles close.

- No top-level functions discovered in this module.

## Governance

### governance/__init__.py
Module with runtime support code.

- No top-level classes discovered in this module.

- No top-level functions discovered in this module.

### governance/single_writer_audit.py
Module with runtime support code.

### Classes
- AuditResult: Container or runtime type for Audit Result.
  - No methods defined in this class.

### Functions
- _collect_violations: Handles collect violations.
- audit_single_writers: Handles audit single writers.
- write_audit_artifact: Writes audit artifact.

## Gpu

### gpu/__init__.py
Compute workers for ASR, vLLM, and CosyVoice lanes.

- No top-level classes discovered in this module.

- No top-level functions discovered in this module.

### gpu/tts_worker/__init__.py
Module with runtime support code.

- No top-level classes discovered in this module.

- No top-level functions discovered in this module.

### gpu/tts_worker/engine.py
Module with runtime support code.

### Classes
- CosyVoiceBiStreamSession: Container or runtime type for Cosy Voice Bi Stream Session.
  - No methods defined in this class.
- TTSEngine: Container or runtime type for T T S Engine.
  - TTSEngine.__init__: Initializes the object.
  - TTSEngine.is_warm: Checks whether warm.
  - TTSEngine.warm: Handles warm.
  - TTSEngine.start_persistent_session: Starts persistent session.
  - TTSEngine.reset: Handles reset.
  - TTSEngine.cancel: Handles cancel.
  - TTSEngine.debug_metrics: Handles debug metrics.
  - TTSEngine._record_generator_fragment: Handles record generator fragment.
  - TTSEngine.stream_pcm: Streams pcm.
  - TTSEngine._stream_generator_pcm_threaded: Asynchronously handles stream generator pcm threaded.
  - TTSEngine._should_drop_resumed_tail: Handles should drop resumed tail.

### Functions
- _pcm_bytes_rms_peak: Handles pcm bytes rms peak.
- _normalize_fragment: Handles normalize fragment.
- _iter_fragments_with_last: Asynchronously handles iter fragments with last.
- _iter_text_tokens: Handles iter text tokens.
- _resolve_native_stream_emit_thresholds: Handles resolve native stream emit thresholds.
- _resolve_native_max_prompt_speech_tokens: Handles resolve native max prompt speech tokens.
- _resolve_native_oneshot_max_tokens: Handles resolve native oneshot max tokens.
- _force_zero_shot_native_stream: Handles force zero shot native stream.
- _ensure_endofprompt: Handles ensure endofprompt.
- _resolve_native_stream_inference: Handles resolve native stream inference.
- _call_cosyvoice_native_stream: Handles call cosyvoice native stream.
- _call_cosyvoice_native_once: Handles call cosyvoice native once.
- _prime_native_prompt_cache: Handles prime native prompt cache.
- _trim_native_prompt_kwargs: Handles trim native prompt kwargs.
- _ends_sentence: Handles ends sentence.
- _token_count: Handles token count.
- _strip_followup_metadata_tail: Handles strip followup metadata tail.
- _streaming_text_fragments: Handles streaming text fragments.
- _contains_cjk: Handles contains cjk.
- _latin_ratio: Handles latin ratio.
- _should_use_cross_lingual_mode: Handles should use cross lingual mode.
- _ensure_english_lang_tag: Handles ensure english lang tag.
- _tts_speech_to_pcm_bytes: Handles tts speech to pcm bytes.
- _prepend_sys_path: Handles prepend sys path.
- _candidate_cosyvoice_roots: Handles candidate cosyvoice roots.
- _ensure_cosyvoice_runtime_paths: Handles ensure cosyvoice runtime paths.
- _assert_cuda_device_binding: Handles assert cuda device binding.
- _cuda_device_context: Handles cuda device context.

### gpu/tts_worker/stream.py
Module with runtime support code.

### Classes
- TTSAudioStreamer: Container or runtime type for T T S Audio Streamer.
  - TTSAudioStreamer.__init__: Initializes the object.
  - TTSAudioStreamer.stream: Asynchronously handles stream.

- No top-level functions discovered in this module.

### gpu/vllm_worker/__init__.py
Module with runtime support code.

- No top-level classes discovered in this module.

- No top-level functions discovered in this module.

### gpu/vllm_worker/engine.py
Module with runtime support code.

### Classes
- VLLMEngineConfig: Container or runtime type for V L L M Engine Config.
  - No methods defined in this class.
- PrefixCacheStats: Container or runtime type for Prefix Cache Stats.
  - PrefixCacheStats.hit_ratio: Handles hit ratio.
- VLLMEngine: GPU0 streaming token runtime backed by vLLM when available.
  - VLLMEngine.__init__: Initializes the object.
  - VLLMEngine.is_warm: Checks whether warm.
  - VLLMEngine.prefix_cache_ready: Handles prefix cache ready.
  - VLLMEngine.prewarm_prefix_cache: Prewarms prefix cache.
  - VLLMEngine.warm: Handles warm.
  - VLLMEngine._maybe_load_tokenizer: Handles maybe load tokenizer.
  - VLLMEngine.cache_stats: Handles cache stats.
  - VLLMEngine.cancel_request: Asynchronously handles cancel request.
  - VLLMEngine.render_prompt: Handles render prompt.
  - VLLMEngine.stream_tokens: Streams tokens.

### Functions
- build_prompt_cache_key: Builds prompt cache key.
- _assert_cuda_device_binding: Handles assert cuda device binding.
- _strip_reasoning_sections: Handles strip reasoning sections.
- _normalize_spoken_output: Handles normalize spoken output.
- _append_spoken_delta: Handles append spoken delta.
- _split_flushable_spoken_prefix: Handles split flushable spoken prefix.
- _cuda_device_context: Handles cuda device context.

### gpu/vllm_worker/stream.py
Module with runtime support code.

### Classes
- VLLMTokenStreamer: Container or runtime type for V L L M Token Streamer.
  - VLLMTokenStreamer.__init__: Initializes the object.
  - VLLMTokenStreamer.stream: Asynchronously handles stream.

- No top-level functions discovered in this module.

## Kernel

### kernel/__init__.py
Module with runtime support code.

- No top-level classes discovered in this module.

- No top-level functions discovered in this module.

### kernel/dispatch.py
Module with runtime support code.

### Classes
- DispatchCommand: Container or runtime type for Dispatch Command.
  - No methods defined in this class.

### Functions
- build_vllm_command: Builds vllm command.
- build_tts_command: Builds tts command.
- build_tts_append_command: Builds tts append command.
- build_vllm_cancel_command: Builds vllm cancel command.
- build_tts_cancel_command: Builds tts cancel command.

### kernel/invariant_loop.py
Module with runtime support code.

### Classes
- InvariantSnapshot: Container or runtime type for Invariant Snapshot.
  - No methods defined in this class.

### Functions
- _no_partial_slot: Handles no partial slot.
- _no_stale_epoch: Handles no stale epoch.
- pre_tick_validate: Handles pre tick validate.
- post_tick_validate: Handles post tick validate.

### kernel/kernel_runtime.py
Module with runtime support code.

### Classes
- KernelConfig: Container or runtime type for Kernel Config.
  - KernelConfig.reducer_config: Handles reducer config.
- KernelCommitResult: Container or runtime type for Kernel Commit Result.
  - No methods defined in this class.
- KernelRuntime: Container or runtime type for Kernel Runtime.
  - KernelRuntime.__init__: Initializes the object.
  - KernelRuntime.topology: Handles topology.
  - KernelRuntime.rings: Handles rings.
  - KernelRuntime.state: Handles state.
  - KernelRuntime.event_log: Handles event log.
  - KernelRuntime.queued_event_count: Handles queued event count.
  - KernelRuntime.next_sequence_no: Handles next sequence no.
  - KernelRuntime.current_lease: Handles current lease.
  - KernelRuntime.reset: Handles reset.
  - KernelRuntime._commit_transition: Handles commit transition.
  - KernelRuntime.apply_event: Handles apply event.
  - KernelRuntime.replay: Handles replay.
  - KernelRuntime.enqueue_event: Enqueues event.
  - KernelRuntime.tick: Handles tick.
  - KernelRuntime._is_protected_event_type: Handles is protected event type.
  - KernelRuntime._event_priority: Handles event priority.
  - KernelRuntime._drop_superseded_partial_events: Handles drop superseded partial events.
  - KernelRuntime._evict_low_priority_for_protected_event: Handles evict low priority for protected event.
  - KernelRuntime._pop_next_queued_event: Handles pop next queued event.
  - KernelRuntime._with_sequence_no: Handles with sequence no.
  - KernelRuntime._is_suppressible_stale_engine_output: Handles is suppressible stale engine output.
  - KernelRuntime.commit_tick: Handles commit tick.
  - KernelRuntime.runtime_metrics: Handles runtime metrics.

- No top-level functions discovered in this module.

### kernel/latency_contract.py
Module with runtime support code.

### Classes
- LaneLatencyBudget: Container or runtime type for Lane Latency Budget.
  - No methods defined in this class.
- PipelineLatencySample: Container or runtime type for Pipeline Latency Sample.
  - PipelineLatencySample.total_ms: Handles total ms.
- LatencyDecision: Container or runtime type for Latency Decision.
  - No methods defined in this class.

### Functions
- evaluate_latency: Evaluates latency.

### kernel/leases.py
Module with runtime support code.

### Classes
- EpochLease: Container or runtime type for Epoch Lease.
  - No methods defined in this class.

### Functions
- epoch_id_for_state: Handles epoch id for state.
- lease_snapshot: Handles lease snapshot.

### kernel/ordering.py
Module with runtime support code.

### Classes
- OrderedEventKey: Container or runtime type for Ordered Event Key.
  - No methods defined in this class.

### Functions
- ordering_key: Handles ordering key.
- expected_next_sequence: Handles expected next sequence.
- make_derived_event: Creates derived event.
- push_front: Pushes front.

### kernel/recovery.py
Module with runtime support code.

### Classes
- RecoverySnapshot: Container or runtime type for Recovery Snapshot.
  - No methods defined in this class.

### Functions
- build_recovery_snapshot: Builds recovery snapshot.
- recovering_status: Handles recovering status.
- recovered_status: Handles recovered status.

### kernel/reducer.py
Module with runtime support code.

### Classes
- DerivedEvent: Container or runtime type for Derived Event.
  - No methods defined in this class.
- ReducerDiagnostics: Container or runtime type for Reducer Diagnostics.
  - ReducerDiagnostics.request_started_ns: Handles request started ns.
  - ReducerDiagnostics.mark_request_started: Handles mark request started.
  - ReducerDiagnostics.has_seen_first_token: Checks whether seen first token.
  - ReducerDiagnostics.mark_first_token_seen: Handles mark first token seen.
- ReducerConfig: Container or runtime type for Reducer Config.
  - No methods defined in this class.
- ReducerTransition: Container or runtime type for Reducer Transition.
  - No methods defined in this class.

### Functions
- _normalized_text: Handles normalized text.
- _join_spoken_tokens: Handles join spoken tokens.
- _should_insert_spoken_boundary_space: Handles should insert spoken boundary space.
- _should_close_first_stream_immediately: Handles should close first stream immediately.
- _is_interrupt_replay: Handles is interrupt replay.
- _apply_event_meta: Handles apply event meta.
- _active_tts_request_for_current_output: Handles active tts request for current output.
- _final_confirms_current_turn: Handles final confirms current turn.
- _text_extends_current_turn: Handles text extends current turn.
- _text_extends_completed_turn: Handles text extends completed turn.
- _completed_turn_is_single_greeting: Handles completed turn is single greeting.
- _should_ignore_completed_turn_short_tail: Handles should ignore completed turn short tail.
- _should_ignore_completed_greeting_extension: Handles should ignore completed greeting extension.
- _should_commit_greeting_partial: Handles should commit greeting partial.
- _current_turn_is_single_greeting: Handles current turn is single greeting.
- validate_engine_output: Validates engine output.
- reduce_event: Handles reduce event.

### kernel/stable_prefix.py
Module with runtime support code.

### Classes
- StablePrefixDecision: Container or runtime type for Stable Prefix Decision.
  - No methods defined in this class.

### Functions
- _normalize_text: Handles normalize text.
- _token_prefix: Handles token prefix.
- detect_stable_prefix: Detects stable prefix.

### kernel/state.py
Module with runtime support code.

### Classes
- TranscriptState: Container or runtime type for Transcript State.
  - No methods defined in this class.
- OutputState: Container or runtime type for Output State.
  - No methods defined in this class.
- RecoveryStatus: Container or runtime type for Recovery Status.
  - No methods defined in this class.
- KernelState: Container or runtime type for Kernel State.
  - KernelState.remember_event: Stores event.
  - KernelState.bind_request_event: Binds request event.
  - KernelState.request_event_id: Handles request event id.
  - KernelState.request_output_version: Handles request output version.

- No top-level functions discovered in this module.

### kernel/tick_engine.py
Module with runtime support code.

### Classes
- TickEngine: Container or runtime type for Tick Engine.
  - TickEngine.__init__: Initializes the object.
  - TickEngine.run_once: Handles run once.
  - TickEngine.drift_alarm_triggered: Handles drift alarm triggered.
  - TickEngine.drift_snapshot: Handles drift snapshot.

- No top-level functions discovered in this module.

### kernel/tts_fragment_planner.py
Module with runtime support code.

### Classes
- TTSFragmentPlannerConfig: Container or runtime type for T T S Fragment Planner Config.
  - No methods defined in this class.
- TTSFragmentPlan: Container or runtime type for T T S Fragment Plan.
  - TTSFragmentPlan.flush_text: Handles flush text.

### Functions
- _normalize_text: Handles normalize text.
- _normalize_tokens: Handles normalize tokens.
- _token_has_boundary: Handles token has boundary.
- _token_is_punctuation_only: Handles token is punctuation only.
- _token_is_contraction_suffix: Handles token is contraction suffix.
- _token_is_dangling_function_word: Handles token is dangling function word.
- _has_internal_boundary: Handles has internal boundary.
- _lexical_token_count: Handles lexical token count.
- _is_complete_short_clause: Handles is complete short clause.
- _is_boundaryless_lexical_prefix: Handles is boundaryless lexical prefix.
- _should_keep_stream_open: Handles should keep stream open.
- plan_tts_fragment: Plans tts fragment.

## Observability

### observability/__init__.py
Module with runtime support code.

- No top-level classes discovered in this module.

- No top-level functions discovered in this module.

### observability/metrics.py
Module with runtime support code.

### Classes
- LatencySummary: Container or runtime type for Latency Summary.
  - No methods defined in this class.

### Functions
- summarize_latency: Handles summarize latency.

### observability/replay_viewer.py
Module with runtime support code.

- No top-level classes discovered in this module.

### Functions
- view_replay: Handles view replay.

### observability/timeline.py
Module with runtime support code.

- No top-level classes discovered in this module.

### Functions
- timeline: Handles timeline.

### observability/tracer.py
Module with runtime support code.

### Classes
- Trace: Container or runtime type for Trace.
  - Trace.add: Handles add.

- No top-level functions discovered in this module.

## Replay

### replay/__init__.py
Module with runtime support code.

- No top-level classes discovered in this module.

- No top-level functions discovered in this module.

### replay/determinism.py
Module with runtime support code.

- No top-level classes discovered in this module.

### Functions
- verify_replay: Handles verify replay.
- verify_replay_runs: Handles verify replay runs.
- canonical_state_hash: Canonicalizes state hash.
- canonical_event_stream_hash: Canonicalizes event stream hash.

### replay/event_log.py
Module with runtime support code.

### Classes
- EventLog: Container or runtime type for Event Log.
  - EventLog.append: Handles append.
  - EventLog.append_runtime_event: Handles append runtime event.
  - EventLog.as_records: Handles as records.
  - EventLog.replay_into_kernel: Handles replay into kernel.

- No top-level functions discovered in this module.

### replay/snapshot.py
Module with runtime support code.

### Classes
- Snapshot: Container or runtime type for Snapshot.
  - No methods defined in this class.

- No top-level functions discovered in this module.

### replay/validator.py
Module with runtime support code.

- No top-level classes discovered in this module.

### Functions
- assert_state_equal: Asserts state equal.
- assert_event_identity_closure: Asserts event identity closure.

### replay/verifier.py
Module with runtime support code.

### Classes
- DeterminismError: Container or runtime type for Determinism Error.
  - No methods defined in this class.
- ReplayRun: Container or runtime type for Replay Run.
  - No methods defined in this class.

### Functions
- _canonical_hash: Handles canonical hash.
- verify_deterministic_replay: Handles verify deterministic replay.

## Root

### __init__.py
Module with runtime support code.

- No top-level classes discovered in this module.

- No top-level functions discovered in this module.

### runtime_registry.py
Module with runtime support code.

- No top-level classes discovered in this module.

### Functions
- module_runtime_owner: Handles module runtime owner.
- assert_valid_runtime_import: Asserts valid runtime import.

## Runtime

### runtime/admission_gate.py
Module with runtime support code.

### Classes
- AdmissionError: Container or runtime type for Admission Error.
  - No methods defined in this class.
- AdmissionConfig: Container or runtime type for Admission Config.
  - No methods defined in this class.

### Functions
- _cpu_flags: Handles cpu flags.
- _check_avx2: Handles check avx2.
- _check_clock: Handles check clock.
- _check_socket_limits: Handles check socket limits.
- _check_forbidden_distributed_env: Handles check forbidden distributed env.
- _require_path: Handles require path.
- _require_cache_dir: Handles require cache dir.
- _require_value: Handles require value.
- _require_file: Handles require file.
- _require_any_glob: Handles require any glob.
- _check_vosk_artifacts: Handles check vosk artifacts.
- _check_vllm_artifacts: Handles check vllm artifacts.
- _check_cosyvoice3_artifacts: Handles check cosyvoice3 artifacts.
- _check_optional_speaker_asset: Handles check optional speaker asset.
- _check_cuda_device: Handles check cuda device.
- _visible_cuda_device_count: Handles visible cuda device count.
- _check_livekit_config: Handles check livekit config.
- hardware_admission_check: Handles hardware admission check.

### runtime/bootstrap.py
Module with runtime support code.

### Classes
- WarmReport: Container or runtime type for Warm Report.
  - No methods defined in this class.
- WorkerStatus: Container or runtime type for Worker Status.
  - No methods defined in this class.
- WorkerFailureReason: Container or runtime type for Worker Failure Reason.
  - No methods defined in this class.
- TopologyReport: Container or runtime type for Topology Report.
  - No methods defined in this class.
- _BlockingTextStream: Container or runtime type for Blocking Text Stream.
  - _BlockingTextStream.__init__: Initializes the object.
  - _BlockingTextStream.push: Handles push.
  - _BlockingTextStream.close: Handles close.
  - _BlockingTextStream.generator: Handles generator.
- _ActiveTTSStreamSession: Container or runtime type for Active T T S Stream Session.
  - No methods defined in this class.
- _SpeculativeVLLMRequest: Container or runtime type for Speculative V L L M Request.
  - No methods defined in this class.
- _PreparedTTSFrame: Container or runtime type for Prepared T T S Frame.
  - No methods defined in this class.
- _SpeculativeTTSRequest: Container or runtime type for Speculative T T S Request.
  - No methods defined in this class.
- VoicePipelineRuntime: Container or runtime type for Voice Pipeline Runtime.
  - VoicePipelineRuntime.__post_init__: Normalizes and validates fields after initialization.
  - VoicePipelineRuntime.global_ready: Handles global ready.
  - VoicePipelineRuntime.assert_ready_for_live_audio: Asserts ready for live audio.
  - VoicePipelineRuntime.tick: Handles tick.
  - VoicePipelineRuntime.topology_report: Handles topology report.
  - VoicePipelineRuntime.dry_run_report: Handles dry run report.
  - VoicePipelineRuntime.next_sequence_no: Handles next sequence no.
  - VoicePipelineRuntime.reset_session_state: Resets session state.
  - VoicePipelineRuntime._stable_session_summary: Handles stable session summary.
  - VoicePipelineRuntime._cancel_speculative_vllm: Asynchronously handles cancel speculative vllm.
  - VoicePipelineRuntime._cancel_speculative_tts: Asynchronously handles cancel speculative tts.
  - VoicePipelineRuntime._run_speculative_vllm_request: Asynchronously handles run speculative vllm request.
  - VoicePipelineRuntime._start_speculative_tts_for_text: Handles start speculative tts for text.
  - VoicePipelineRuntime._run_speculative_tts_request: Asynchronously handles run speculative tts request.
  - VoicePipelineRuntime._ensure_speculative_tts_drain_task: Handles ensure speculative tts drain task.
  - VoicePipelineRuntime._drain_promoted_speculative_tts: Asynchronously handles drain promoted speculative tts.
  - VoicePipelineRuntime._promote_speculative_tts_request: Asynchronously handles promote speculative tts request.
  - VoicePipelineRuntime._maybe_prewarm_vllm_stable_prefix: Handles maybe prewarm vllm stable prefix.
  - VoicePipelineRuntime.warm_vllm_runtime_probe: Warms vllm runtime probe.
  - VoicePipelineRuntime._warm_tts_generator_runtime_probe: Asynchronously handles warm tts generator runtime probe.
  - VoicePipelineRuntime.warm_tts_runtime_probe: Warms tts runtime probe.
  - VoicePipelineRuntime.start: Asynchronously handles start.
  - VoicePipelineRuntime.stop: Asynchronously handles stop.
  - VoicePipelineRuntime.run_forever: Asynchronously handles run forever.
  - VoicePipelineRuntime._tick_loop: Asynchronously handles tick loop.
  - VoicePipelineRuntime._record_latency: Handles record latency.
  - VoicePipelineRuntime._pcm_bytes_rms_peak: Handles pcm bytes rms peak.
  - VoicePipelineRuntime._update_tts_signal_metrics: Handles update tts signal metrics.
  - VoicePipelineRuntime._start_tts_request_metrics: Handles start tts request metrics.
  - VoicePipelineRuntime._record_tts_chunk_trace: Handles record tts chunk trace.
  - VoicePipelineRuntime._trace_relative_ms: Handles trace relative ms.
  - VoicePipelineRuntime._record_ingress_frame_trace: Handles record ingress frame trace.
  - VoicePipelineRuntime._record_asr_event_trace: Handles record asr event trace.
  - VoicePipelineRuntime.note_vad_speech_start: Handles note vad speech start.
  - VoicePipelineRuntime._completed_greeting_tail_suffix: Handles completed greeting tail suffix.
  - VoicePipelineRuntime._should_suppress_stale_greeting_asr_extension: Handles should suppress stale greeting asr extension.
  - VoicePipelineRuntime._should_emit_tts_frame: Handles should emit tts frame.
  - VoicePipelineRuntime._record_trimmed_tts_frame: Handles record trimmed tts frame.
  - VoicePipelineRuntime._batch_opens_tts_leading_gate: Handles batch opens tts leading gate.
  - VoicePipelineRuntime._tts_leading_batch_start_index: Handles tts leading batch start index.
  - VoicePipelineRuntime._should_drop_final_tts_resampler_tail: Handles should drop final tts resampler tail.
  - VoicePipelineRuntime._authority_event: Handles authority event.
  - VoicePipelineRuntime._append_event: Handles append event.
  - VoicePipelineRuntime._asr_events_to_authority: Handles asr events to authority.
  - VoicePipelineRuntime._filter_stale_asr_events: Handles filter stale asr events.
  - VoicePipelineRuntime._encode_slot_payload: Handles encode slot payload.
  - VoicePipelineRuntime._push_lane_slot: Handles push lane slot.
  - VoicePipelineRuntime._mirror_authority_event_to_ring: Handles mirror authority event to ring.
  - VoicePipelineRuntime._mirror_dispatch_command_to_ring: Handles mirror dispatch command to ring.
  - VoicePipelineRuntime._mirror_pcm_frame_to_ring: Handles mirror pcm frame to ring.
  - VoicePipelineRuntime.process_pcm_frame: Asynchronously handles process pcm frame.
  - VoicePipelineRuntime.finalize_asr_turn: Asynchronously handles finalize asr turn.
  - VoicePipelineRuntime._tick_and_stamp_commands: Asynchronously handles tick and stamp commands.
  - VoicePipelineRuntime._dispatch_commands: Asynchronously handles dispatch commands.
  - VoicePipelineRuntime.run_tick_and_dispatch: Asynchronously handles run tick and dispatch.
  - VoicePipelineRuntime._execute_vllm_cancel_command: Asynchronously handles execute vllm cancel command.
  - VoicePipelineRuntime._execute_tts_cancel_command: Asynchronously handles execute tts cancel command.
  - VoicePipelineRuntime._execute_vllm_command: Asynchronously handles execute vllm command.
  - VoicePipelineRuntime._execute_tts_command: Asynchronously handles execute tts command.
  - VoicePipelineRuntime._execute_tts_append_command: Asynchronously handles execute tts append command.
  - VoicePipelineRuntime._run_tts_stream_session: Asynchronously handles run tts stream session.
  - VoicePipelineRuntime._reset_output_resampler: Handles reset output resampler.
  - VoicePipelineRuntime._resample_output: Handles resample output.
  - VoicePipelineRuntime._output_frame_bytes: Handles output frame bytes.
  - VoicePipelineRuntime._chunk_output_pcm: Handles chunk output pcm.
  - VoicePipelineRuntime.send_pcm_once: Asynchronously handles send pcm once.
  - VoicePipelineRuntime.latency_summary: Handles latency summary.
  - VoicePipelineRuntime.replay_state_hash: Handles replay state hash.
  - VoicePipelineRuntime.replay_event_hash: Handles replay event hash.
  - VoicePipelineRuntime.last_timestamps: Handles last timestamps.
  - VoicePipelineRuntime.tts_signal_metrics: Handles tts signal metrics.
  - VoicePipelineRuntime.ingress_frame_trace: Handles ingress frame trace.
  - VoicePipelineRuntime.asr_event_trace: Handles asr event trace.
  - VoicePipelineRuntime.recovery_snapshot: Handles recovery snapshot.

### Functions
- _single_fragment_generator: Handles single fragment generator.
- _token_fragment_generator: Handles token fragment generator.
- _push_tokenized_text: Handles push tokenized text.
- _join_spoken_tokens: Handles join spoken tokens.
- _should_insert_spoken_boundary_space: Handles should insert spoken boundary space.
- _bind_cuda_device: Handles bind cuda device.
- _assert_contract: Handles assert contract.
- _identity_hash: Handles identity hash.
- _build_model_cache_identity: Handles build model cache identity.
- _build_topology: Handles build topology.
- _build_kernel: Handles build kernel.
- _warm_asr_engine: Handles warm asr engine.
- _run_async_probe: Handles run async probe.
- _warm_vllm_engine: Handles warm vllm engine.
- _warm_tts_engine: Handles warm tts engine.
- bootstrap_runtime: Handles bootstrap runtime.

### runtime/cli.py
Module with runtime support code.

- No top-level classes discovered in this module.

### Functions
- _repo_root: Handles repo root.
- _default_env_file: Handles default env file.
- _configure_runtime_env: Handles configure runtime env.
- _configure_cuda_library_path: Handles configure cuda library path.
- _preload_cuda_runtime_libraries: Handles preload cuda runtime libraries.
- _assert_runtime_contract: Handles assert runtime contract.
- _parse_args: Handles parse args.
- main: Handles main.

### runtime/config.py
Module with runtime support code.

### Classes
- RuntimeConfig: Container or runtime type for Runtime Config.
  - RuntimeConfig.resolved_vllm_model_path: Handles resolved vllm model path.
  - RuntimeConfig.resolved_cosyvoice3_model_path: Handles resolved cosyvoice3 model path.
  - RuntimeConfig.resolved_cosyvoice3_prompt_text: Handles resolved cosyvoice3 prompt text.
  - RuntimeConfig.from_env: Converts from env.

### Functions
- _apply_runtime_defaults: Handles apply runtime defaults.
- _load_env_file_once: Handles load env file once.

### runtime/livekit_bridge.py
Module with runtime support code.

### Classes
- _LocalInferenceExecutor: Container or runtime type for Local Inference Executor.
  - _LocalInferenceExecutor.__init__: Initializes the object.
  - _LocalInferenceExecutor.do_inference: Asynchronously handles do inference.
- LiveKitRuntimeBridge: Container or runtime type for Live Kit Runtime Bridge.
  - LiveKitRuntimeBridge.__post_init__: Normalizes and validates fields after initialization.
  - LiveKitRuntimeBridge.configure_debug_ingress_capture: Handles configure debug ingress capture.
  - LiveKitRuntimeBridge.ingress_lock_state: Handles ingress lock state.
  - LiveKitRuntimeBridge._add_task: Handles add task.
  - LiveKitRuntimeBridge._normalize_egress_pcm_frame: Handles normalize egress pcm frame.
  - LiveKitRuntimeBridge.start: Asynchronously handles start.
  - LiveKitRuntimeBridge.stop: Asynchronously handles stop.
  - LiveKitRuntimeBridge._record_debug_ingress_pcm: Handles record debug ingress pcm.
  - LiveKitRuntimeBridge._flush_debug_ingress_capture: Handles flush debug ingress capture.
  - LiveKitRuntimeBridge._consume_remote_audio: Asynchronously handles consume remote audio.
  - LiveKitRuntimeBridge._build_silero_vad: Handles build silero vad.
  - LiveKitRuntimeBridge._build_turn_detector: Handles build turn detector.
  - LiveKitRuntimeBridge._turn_finalize_delay_seconds: Asynchronously handles turn finalize delay seconds.
  - LiveKitRuntimeBridge._pcm_frame_rms: Handles pcm frame rms.
  - LiveKitRuntimeBridge._pcm_frame_peak: Handles pcm frame peak.
  - LiveKitRuntimeBridge._emit_runtime_pcm: Asynchronously handles emit runtime pcm.

### Functions
- _b64url: Handles b64url.
- create_livekit_access_token: Creates livekit access token.

### runtime/server.py
Module with runtime support code.

- No top-level classes discovered in this module.

### Functions
- _describe_exception: Handles describe exception.
- _runpod_livekit_proxy_url: Handles runpod livekit proxy url.
- _client_livekit_url: Handles client livekit url.
- _allowed_cors_origins: Handles allowed cors origins.
- _failure_reason: Handles failure reason.
- _runtime_readiness: Handles runtime readiness.
- _runtime_telemetry: Handles runtime telemetry.
- _system_config: Handles system config.
- _pending_system_config: Handles pending system config.
- _set_runtime_ingress_filter: Handles set runtime ingress filter.
- _pending_runtime_readiness: Handles pending runtime readiness.
- _reconcile_bootstrap_state: Handles reconcile bootstrap state.
- _set_bootstrap_phase: Handles set bootstrap phase.
- _bootstrap_runtime_background: Asynchronously handles bootstrap runtime background.
- create_app: Creates app.

### runtime/topology.py
Module with runtime support code.

### Classes
- LaneConfig: Container or runtime type for Lane Config.
  - No methods defined in this class.
- RuntimeTopology: Container or runtime type for Runtime Topology.
  - No methods defined in this class.

- No top-level functions discovered in this module.

## Shared

### shared/__init__.py
Module with runtime support code.

- No top-level classes discovered in this module.

- No top-level functions discovered in this module.

### shared/audio_resample.py
Module with runtime support code.

### Classes
- StreamingAudioResampler: Container or runtime type for Streaming Audio Resampler.
  - StreamingAudioResampler.__init__: Initializes the object.
  - StreamingAudioResampler._reset_stream: Handles reset stream.
  - StreamingAudioResampler.resample: Handles resample.
  - StreamingAudioResampler.flush: Handles flush.

- No top-level functions discovered in this module.

### shared/lineage.py
Module with runtime support code.

### Classes
- EpochLineage: Container or runtime type for Epoch Lineage.
  - EpochLineage.as_id: Handles as id.
- IdentityClosure: Container or runtime type for Identity Closure.
  - No methods defined in this class.

### Functions
- canonical_session_id: Canonicalizes session id.
- canonical_turn_id: Canonicalizes turn id.
- canonical_epoch_id: Canonicalizes epoch id.
- canonical_response_request_id: Canonicalizes response request id.
- build_lineage_id: Builds lineage id.
- build_trace_id: Builds trace id.
- build_identity_closure: Builds identity closure.
- validate_identity_closure: Validates identity closure.
- validate_lineage_match: Validates lineage match.

### shared/text.py
Module with runtime support code.

- No top-level classes discovered in this module.

### Functions
- normalize_text: Handles normalize text.
- normalize_transcript: Handles normalize transcript.
- preview_text: Handles preview text.

### shared/time.py
Module with runtime support code.

- No top-level classes discovered in this module.

### Functions
- now_ns: Handles now ns.
- ns_to_ms: Handles ns to ms.

### shared/types.py
Module with runtime support code.

### Classes
- DispatchPayload: Container or runtime type for Dispatch Payload.
  - No methods defined in this class.
- EventValidationError: Container or runtime type for Event Validation Error.
  - No methods defined in this class.
- AuthorityEvent: Container or runtime type for Authority Event.
  - AuthorityEvent.__post_init__: Normalizes and validates fields after initialization.

### Functions
- _freeze_mapping: Handles freeze mapping.
- new_authority_event: Creates authority event.
- validate_authority_event: Validates authority event.

## Stt

### stt/__init__.py
Module with runtime support code.

- No top-level classes discovered in this module.

- No top-level functions discovered in this module.

### stt/asr_engine.py
Module with runtime support code.

### Classes
- ASREvent: Container or runtime type for A S R Event.
  - No methods defined in this class.
- ASRRuntimeConfig: Container or runtime type for A S R Runtime Config.
  - No methods defined in this class.
- ASREngine: CPU streaming ASR runtime backed by Vosk when available.
  - ASREngine.__init__: Initializes the object.
  - ASREngine.is_warm: Checks whether warm.
  - ASREngine.sample_rate: Handles sample rate.
  - ASREngine.warm: Handles warm.
  - ASREngine.start_session: Starts session.
  - ASREngine._resample_input_audio: Handles resample input audio.
  - ASREngine.ingest_partial: Handles ingest partial.
  - ASREngine.ingest_final: Handles ingest final.
  - ASREngine.ingest_audio: Process PCM16 mono audio and emit partial/final transcript events.
  - ASREngine.finalize: Handles finalize.

### Functions
- _safe_json_loads: Handles safe json loads.

## Transport

### transport/__init__.py
Module with runtime support code.

- No top-level classes discovered in this module.

- No top-level functions discovered in this module.

### transport/livekit_transport.py
Module with runtime support code.

### Classes
- LiveKitTransportConfig: Container or runtime type for Live Kit Transport Config.
  - No methods defined in this class.
- LiveKitTransport: Container or runtime type for Live Kit Transport.
  - LiveKitTransport.__init__: Initializes the object.
  - LiveKitTransport.mark_bridge_connected: Handles mark bridge connected.
  - LiveKitTransport.mark_bridge_disconnected: Handles mark bridge disconnected.
  - LiveKitTransport.record_ingress_frame: Handles record ingress frame.
  - LiveKitTransport.record_ingress_drop: Handles record ingress drop.
  - LiveKitTransport.start_egress_request: Starts egress request.
  - LiveKitTransport.record_egress_frame: Handles record egress frame.
  - LiveKitTransport.ingress_metrics: Handles ingress metrics.

- No top-level functions discovered in this module.

### transport/pcm_clock.py
Module with runtime support code.

### Classes
- PCMFrame: Container or runtime type for P C M Frame.
  - No methods defined in this class.
- PCMClockSender: Container or runtime type for P C M Clock Sender.
  - PCMClockSender.__init__: Initializes the object.
  - PCMClockSender.depth: Handles depth.
  - PCMClockSender.head_lease: Handles head lease.
  - PCMClockSender.enqueue: Handles enqueue.
  - PCMClockSender.clear: Handles clear.
  - PCMClockSender._pop_fresh: Handles pop fresh.
  - PCMClockSender.oldest_age_ms: Handles oldest age ms.
  - PCMClockSender.run_once: Asynchronously handles run once.

- No top-level functions discovered in this module.
