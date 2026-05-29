# Voice Pipeline Finalization Tasks

Source baseline:
- `overview.md` generated from current `apps/api/src/voice_pipeline` Python files (56 files).
- `ARCHITECTURE.md`, `AGENTS.md`, `plan.md`.
- User completion checklist (A through O).

Goal:
- Finish hardening and validation of the current single-design runtime without re-architecture.

## Current module map lock

- [ ] Confirm module map still matches `overview.md` before each hardening pass.
- [ ] `bus`: `ring_topology.py`, `ring_types.py`, `shm_ring.py`.
- [ ] `governance`: `single_writer_audit.py`.
- [ ] `gpu/tts_worker`: `engine.py`, `stream.py`.
- [ ] `gpu/vllm_worker`: `engine.py`, `stream.py`.
- [ ] `kernel`: `dispatch.py`, `invariant_loop.py`, `kernel_runtime.py`, `latency_contract.py`, `leases.py`, `ordering.py`, `recovery.py`, `reducer.py`, `stable_prefix.py`, `state.py`, `tick_engine.py`, `tts_fragment_planner.py`.
- [ ] `runtime`: `admission_gate.py`, `bootstrap.py`, `cli.py`, `config.py`, `server.py`, `topology.py`.
- [ ] `stt`: `asr_engine.py`.
- [ ] `transport`: `pcm_clock.py`, `livekit_transport.py`.

## A) Delete/remove legacy overlap

### A1. Delete old duplicate authority leftovers

- [ ] Run: `rg "StreamSpine|stream_spine|token_router|stream_control|EffectLog|command_log|CreditController|TTSCreditController|VLLMCreditController|orchestrator|control_fsm|tts_scheduler|CreditExhaustedError" apps/api/src/voice_pipeline`
- [ ] Remove or refactor any live-code matches into current `KernelRuntime` design.
- [ ] Final condition: zero live-code matches.

### A2. Delete distributed/runtime drift

- [ ] Run: `rg "docker|kubernetes|pod|grpc|rpc|remote worker|worker process|multiprocessing|subprocess" apps/api/src/voice_pipeline`
- [ ] Remove/refactor distributed or multi-process implications from live runtime code.
- [ ] Keep mentions only where explicitly documenting forbidden patterns.

### A3. Delete fake TTS streaming

- [ ] Run: `rg "_fallback_stream_pcm|zero_shot|full_text|non_streaming|batch_tts|_iter_chunks|_chunks" apps/api/src/voice_pipeline/gpu/tts_worker`
- [ ] Remove any fake-streaming fallback paths.
- [ ] Keep `_tts_speech_to_pcm_bytes` only if it is pure PCM conversion.
- [ ] Ensure `_iter_fragments` maps to Kernel-committed fragments only.
- [ ] Add test that TTS fails fast if native bi-stream API is unavailable.
- [ ] Final rule: no simulated streaming via repeated full-text calls.

### A4. Delete batch ASR live path

- [ ] Run: `rg "transcribe_file|wavfile|full_audio|batch_asr|recognize_file" apps/api/src/voice_pipeline/stt`
- [ ] Remove any live full-file/batch ASR path.
- [ ] Allow offline-only helpers in test/tooling paths if clearly isolated.

## B) Bus tasks

- [ ] Keep `apps/api/src/voice_pipeline/bus/shm_ring.py`.
- [ ] Verify/extend ring metrics: `capacity`, `depth`, `overwrite_count`, `oldest_age_ms` if feasible.
- [ ] Add tests: bounded capacity, overflow behavior, no semantic routing, no policy decisions.
- [ ] Ensure bus layer only performs byte/event movement.
- [ ] Ensure bus layer does not own routing/backpressure/epoch-policy decisions.

## C) Governance tasks

- [ ] Wire `audit_single_writers()` into CI/local audit command path.
- [ ] Enforce: `KernelRuntime` is the only authority writer.
- [ ] Enforce: runtime does not mutate kernel truth directly.
- [ ] Enforce: workers do not mutate session/epoch/output state.
- [ ] Ensure artifact output includes file path, pass/fail, violating modules.
- [ ] Add regression test with a synthetic violation.

## D) GPU / vLLM tasks

- [ ] Ensure `VLLMEngine.warm()` binds only `cuda:0`.
- [ ] Ensure `prewarm_prefix_cache()` uses stable system/session scaffold only.
- [ ] Ensure `build_prompt_cache_key()` excludes ASR partials, timestamps, random IDs, queue metrics, stale epoch text.
- [ ] Ensure `stream_tokens()` is the only live generation path.
- [ ] Ensure one-token warm probe exists.
- [ ] Expose cache stats in readiness/telemetry.
- [ ] Add tests: cuda:0 binding, prefix cache ready, stable cache key for same scaffold, ASR partial excluded, first token before completion, no TTS coupling.
- [ ] Run coupling audit: `rg "TTSEngine|tts_worker|CosyVoice" apps/api/src/voice_pipeline/gpu/vllm_worker`
- [ ] Final condition: zero vLLM->TTS coupling matches.

## E) GPU / CosyVoice3 TTS tasks

- [ ] Ensure `TTSEngine.warm()` binds only `cuda:1`.
- [ ] Ensure `start_persistent_session()` initializes native bi-stream session.
- [ ] Ensure `stream_pcm()` accepts only Kernel-committed fragments.
- [ ] Ensure `_resolve_native_stream_inference()` fails if native API unavailable.
- [ ] Ensure `_call_cosyvoice_native_stream()` is incremental PCM path.
- [ ] Ensure `reset()` is only driven by Kernel lifecycle.
- [ ] Ensure first-PCM warm probe exists.
- [ ] Add tests: cuda:1 binding, native bi-stream required, first PCM before LLM completion, no fake fallback, interrupt stale suppression, no vLLM coupling.
- [ ] Run coupling audit: `rg "VLLMEngine|vllm_worker|stream_tokens" apps/api/src/voice_pipeline/gpu/tts_worker`
- [ ] Final condition: zero TTS->vLLM coupling matches.

## F) Kernel tasks

### F1. KernelRuntime queue/tick finalization

- [ ] Verify `enqueue_event()` is the only live ingress.
- [ ] Verify bounded `max_events_per_tick`.
- [ ] Verify priority order: interrupts > ASR finals > ASR stability > LLM tokens > TTS PCM > telemetry.
- [ ] Verify protected event types include interrupts and ASR finals.
- [ ] Verify superseded partial drops are safe-only.
- [ ] Verify protected events are never evicted.
- [ ] Verify stale output suppression by epoch/output version.
- [ ] Verify `commit_tick()` emits dispatch commands only.
- [ ] Add tests: full-queue protected interrupt/final survival, partial drop behavior, stale TTS/LLM drops, bounded per-tick processing.

### F2. Reducer finalization

- [ ] Reducer must own generation/epoch/output version transitions.
- [ ] Reducer must own interrupt/cancel/reset transitions.
- [ ] Reducer must own dispatch intent creation.
- [ ] Reducer must not execute workers.
- [ ] Reducer output shape must stay: `next_state`, `derived_events`, `dispatch_commands`, `diagnostics`.
- [ ] Add tests: ASR final -> stable prefix -> vLLM dispatch.
- [ ] Add tests: LLM tokens -> fragment commit -> TTS dispatch.
- [ ] Add tests: HARD_INTERRUPT -> cancel old lanes + create new epoch.

### F3. Dispatch finalization

- [ ] Keep `DispatchCommand` as plain data.
- [ ] Ensure `dispatch.py` does not call engines directly.
- [ ] Ensure command shape includes session/epoch/lineage/output-version/payload fields.
- [ ] Add tests: vLLM command shape, TTS command shape, stale command ignored post-epoch change.

### F4. Stable prefix and fragment planner

- [ ] Keep policy helpers kernel-only.
- [ ] Ensure runtime does not make policy decisions using these helpers.
- [ ] Ensure `detect_stable_prefix()` handles repeated partials and ASR final paths.
- [ ] Ensure `plan_tts_fragment()` remains pure and does not call TTS.
- [ ] Add tests: stability on repeated partials, immediate stability on final, token boundary fragmenting, no full-response wait.
- [ ] Audit runtime policy leakage: `rg "detect_stable_prefix|plan_tts_fragment" apps/api/src/voice_pipeline/runtime`
- [ ] Final condition: zero runtime decision-making leakage matches.

## G) Runtime tasks

### G1. Lock safety

- [ ] Verify `_tick_and_stamp_commands()` only ticks kernel and collects dispatch commands.
- [ ] Verify `_dispatch_commands()` executes outside kernel lock.
- [ ] Verify no recursive deadlock path in dispatch handlers.
- [ ] Verify worker outputs always re-enter via authority event enqueue.
- [ ] Add test: dispatch-time enqueue path does not deadlock.

### G2. Runtime must not make policy

- [ ] Audit: `rg "detect_stable_prefix|plan_tts_fragment|pressure_score|build_vllm_command|build_tts_command|lease_snapshot|epoch_id_for_state" apps/api/src/voice_pipeline/runtime`
- [ ] Move any decision-making logic into kernel reducer if found.
- [ ] Keep runtime role: host engines, execute commands, convert lane outputs to authority events.

### G3. Startup/admission/config

- [ ] Ensure admission fails if `cuda:0`/`cuda:1` missing.
- [ ] Ensure admission fails on missing required model paths/cache dirs.
- [ ] Ensure `RuntimeConfig.from_env()` requires core env keys:
- [ ] `VOSK_MODEL_PATH`
- [ ] `VLLM_MODEL_PATH`
- [ ] `VLLM_CACHE_DIR`
- [ ] `COSYVOICE3_MODEL_PATH`
- [ ] `COSYVOICE3_CACHE_DIR`
- [ ] Ensure no startup model download/network path.
- [ ] Ensure topology report includes required device map and identity hashes.

### G4. Server/frontend session tasks

- [ ] Expose readiness endpoint (or equivalent) and enforce readiness gating.
- [ ] Expose telemetry endpoint with latency and queue metrics.
- [ ] Expose safe runtime config endpoint.
- [ ] Verify WS framed path: frontend PCM -> transport -> `process_pcm_frame()`.
- [ ] Ensure runtime refuses PCM before READY.
- [ ] Ensure session/epoch/lineage identity assignment is consistent.

## H) Transport tasks

- [ ] Ensure `decode_ingress()` only decodes frame bytes.
- [ ] Ensure ingress guard enforces mechanical rules only.
- [ ] Ensure PCM queue carries enough epoch/output-version metadata for stale suppression.
- [ ] Ensure `run_once()` never emits stale PCM.
- [ ] Ensure `oldest_age_ms` is exported through telemetry.
- [ ] Add tests: out-of-order ingress handling, stale PCM suppression, bounded queue behavior, age metric stability.
- [ ] Audit policy leakage: `rg "reduce_event|detect_stable_prefix|plan_tts_fragment|build_vllm_command|build_tts_command|pressure_score" apps/api/src/voice_pipeline/transport`
- [ ] Final condition: zero policy/helper matches in transport.

## I) STT / ASR tasks

- [ ] Ensure `ASREngine.warm()` loads Vosk model once per engine lifecycle.
- [ ] Ensure `start_session()` initializes recognizer/session state.
- [ ] Ensure `ingest_audio()` consumes 20ms PCM16 mono frames (with resample as needed).
- [ ] Ensure emitted ASR events carry required identity/timestamp metadata in authority payload path.
- [ ] Ensure `finalize()` never dispatches LLM directly.
- [ ] Add tests: warm probe, partial shape, final shape, final retention/no-drop, no vLLM imports.

## J) Replay / recovery tasks

- [ ] Ensure every authority event is appended to event log.
- [ ] Ensure replay is passive (no live-control authority).
- [ ] Ensure `replay_into_kernel` is isolated from live runtime flow.
- [ ] Add replay determinism tests: normal turn, barge-in turn, stale suppression, interrupt/reset path.
- [ ] Ensure recovery snapshot includes kernel state, lease/order/output cursors, sequence context.

## K) Observability tasks

- [ ] Ensure latency summary includes p50/p95/p99 for ASR, kernel, vLLM first-token, TTS first-PCM, transport, and E2E.
- [ ] Ensure queue telemetry includes depth, oldest_age_ms, drops, stale_drops.
- [ ] Ensure telemetry includes topology/model/drift/replay hash keys.
- [ ] Add tests for stable telemetry key presence and type shape.

## L) Shared / lineage tasks

- [ ] Ensure all live events use `AuthorityEvent`.
- [ ] Ensure event identity closure includes session/epoch/lane/lineage/timestamp/sequence.
- [ ] Ensure stale suppression checks epoch/output-version/lineage coherently.
- [ ] Add tests: lineage match, stale lineage rejection, identity closure validation.

## M) Production command/runbook tasks

- [ ] Document command for dry-run topology.
- [ ] Document command for warm all backends.
- [ ] Document command for runtime start.
- [ ] Document command for mocked E2E continuous audio.
- [ ] Document command for real Vosk smoke.
- [ ] Document command for real vLLM cuda:0 smoke.
- [ ] Document command for real CosyVoice3 cuda:1 native bi-stream smoke.
- [ ] Document command for full real backend smoke/benchmark path.
- [ ] Document command for replay determinism check.
- [ ] Document command for CI drift audit.
- [ ] Document command for single-writer governance audit.
- [ ] For each command include env vars, expected success output, common failure reasons.

## N) Final test matrix

- [ ] `test_architecture_contract.py`
- [ ] `test_single_writer_audit.py`
- [ ] `test_runtime_bootstrap_readiness.py`
- [ ] `test_runtime_no_deadlock_dispatch.py`
- [ ] `test_kernel_event_priority.py`
- [ ] `test_kernel_stale_output_suppression.py`
- [ ] `test_kernel_reducer_dispatch.py`
- [ ] `test_asr_streaming_contract.py`
- [ ] `test_vllm_prefix_cache_contract.py`
- [ ] `test_tts_native_bistream_contract.py`
- [ ] `test_transport_pcm_lineage.py`
- [ ] `test_replay_determinism.py`
- [ ] `test_latency_gates_mocked.py`
- [ ] `test_runtime_server_readiness.py`

## O) Final done checklist

- [ ] No legacy authority names found.
- [ ] No Docker/distributed/RPC hot-path assumptions found.
- [ ] No fake TTS streaming path in live runtime.
- [ ] No batch ASR path in live runtime.
- [ ] Runtime does not import kernel policy helpers for decisions.
- [ ] `KernelRuntime.enqueue_event()` is the only live ingress.
- [ ] Dispatch executes outside locks.
- [ ] No recursive tick/dispatch deadlock path.
- [ ] ASR is CPU-only and emits partial/final authority events.
- [ ] vLLM binds `cuda:0` and cache key excludes ASR partials.
- [ ] CosyVoice3 binds `cuda:1` and native bi-stream is required.
- [ ] Runtime rejects live audio before READY.
- [ ] PCM egress drops stale epoch/output-version frames.
- [ ] Barge-in suppression p95 <= 40ms.
- [ ] Mocked continuous-session E2E p95 <= 150ms.
- [ ] Real E2E p95 <= 150ms, or explicit dependency/hardware reason is emitted.
- [ ] Replay state/event hash parity passes.
- [ ] Readiness endpoint exposes stable required keys.
- [ ] Telemetry endpoint exposes stable latency/queue/hash keys.
- [ ] CLI/runbook includes start/smoke/benchmark/audit commands.

## Tracking log

- [ ] Add dated entries here for each completed subsection with command evidence and result snapshots.
- [x] 2026-05-29: Implemented hardening wave without duplicate runtime implementations:
- [x] A scans pass with zero matches for legacy authority names, distributed/runtime drift, fake TTS streaming fallback tokens, batch ASR tokens, runtime policy helper leakage, and transport policy leakage.
- [x] B implemented `SharedMemoryRing.oldest_age_ms` mechanical metric and added transport/lineage contract tests.
- [x] C strengthened `single_writer_audit` checks (kernel mutation/import boundaries) and added regression tests + artifact metadata fields.
- [x] D/E added/confirmed vLLM/TTS contract tests (`test_vllm_prefix_cache_contract.py`, `test_tts_native_bistream_contract.py`) and telemetry/readiness now expose vLLM prefix cache stats.
- [x] F added kernel dispatch shape tests and included `session_id` in dispatch payload builders; reducer call sites updated accordingly.
- [x] G added endpoint aliases (`/ready`, `/telemetry`, `/config`) mapped to existing system endpoints; added no-deadlock dispatch contract test.
- [x] H/I/J/K/L validated through full suite + added matrix tests (`test_transport_pcm_lineage.py`, `test_replay_determinism.py`, `test_latency_gates_mocked.py`, `test_runtime_bootstrap_readiness.py`, `test_runtime_server_readiness.py`, `test_architecture_contract.py`).
- [x] M runbook update: added explicit single-writer governance audit command in `README.md`.
- [x] Validation evidence:
- [x] `python -m pytest apps/api/tests -q` -> `91 passed` (1 existing pytest warning about `asyncio_mode` option).
- [x] `python apps/api/scripts/check_drift_guards.py` -> `drift guard: ok`.
- [x] `python scripts/run_drift_audit.py` -> `drift-audit: ok`.
- [x] `python scripts/run_mocked_e2e.py` -> `3 passed`.
- [x] `python scripts/run_replay_determinism.py` -> `4 passed`.
- [x] `python scripts/run_latency_benchmark.py` -> mocked contract `2 passed`.
- [x] `python scripts/run_voice_runtime.py --dry-run` -> `CPU ASR, GPU0 vLLM, GPU1 CosyVoice3`.
- [ ] External blocker remains for real backend completion: missing local model artifacts for Vosk/vLLM/CosyVoice3 paths reported by `python scripts/check_runtime_env.py`.
- [ ] For every checked item, record exact command output, test name, artifact path, or PR/commit reference.
