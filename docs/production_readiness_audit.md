# Production Readiness Audit (Single-Design Voice Runtime)

Date: 2026-05-29

Scope audited against:
- `AGENTS.md`
- `ARCHITECTURE.md`

## Summary

- Runtime architecture is locked to one authority path (`KernelRuntime`) and one live topology (CPU Vosk, GPU0 vLLM, GPU1 CosyVoice3 native bi-stream).
- Transport live path is constrained to framed PCM byte movement only (no control/token semantic helper path).
- Drift guards, startup guards, import guards, and single-writer audit are passing.
- Dead compatibility shim surfaces and unused TTS config export surfaces have been removed from shared/runtime package exports where not used by live lanes.
- API/runtime and determinism test suites are passing.
- Real-backend signoff remains blocked by missing local model artifacts (explicit fail-fast).

## Requirement Matrix

1. Single-process runtime, no Docker/K8s/distributed runtime assumptions: PASS  
Evidence: `apps/api/scripts/check_drift_guards.py` + `scripts/run_drift_audit.py` pass; runtime boots via `scripts/run_voice_runtime.py` with no container orchestration layer.

2. One authority ingress (`KernelRuntime.enqueue_event`) and no worker-to-worker authority bypass: PASS  
Evidence: runtime path enqueues authority events through `VoicePipelineRuntime._append_event()` into `kernel.enqueue_event`; drift guard forbids `apply_event` outside `kernel_runtime.py`.

3. ASR CPU streaming partial/final events only in live path: PASS  
Evidence: `stt/asr_engine.py` requires warmed recognizer and continuously ingests PCM frames without batch-ASR shortcuts; batch-ASR drift patterns blocked by guard; tests pass.

4. vLLM on `cuda:0`, streaming tokens, strict warmup: PASS  
Evidence: topology and startup contract guards enforce `cuda:0`; strict warmup probe in bootstrap; tests pass.

5. CosyVoice3 on `cuda:1`, native bi-stream only, no fake streaming: PASS  
Evidence: runtime enforces native streaming by API shape (no `stream` mode toggle surface); live inference resolver now permits only `inference_bistream`; tests cover absent non-streaming toggle and warm-state strictness.

6. Runtime lock safety and no lock-held dispatch execution: PASS  
Evidence: `run_tick_and_dispatch()` now uses non-recursive iterative command draining (`_tick_and_stamp_commands` + `_dispatch_commands`) with lock held only around `kernel.tick()`, and command execution strictly outside lock.

7. Kernel-only interrupt policy and barge-in authority: PASS  
Evidence: reducer emits `SOFT_PRE_INTERRUPT`/`HARD_INTERRUPT` derived events; runtime-side semantic interrupt injection removed.

8. Queue boundedness, stale suppression, interrupt/final prioritization: PASS  
Evidence: kernel queue policy and priority ordering tests pass; PCM clock contract tests verify stale epoch/output-version frame drops and bounded overflow behavior; browser PCM capture path now enforces bounded buffering without dynamic growth and is guarded by `scripts/check_frontend_contract.py`; transport protocol path is now PCM-only.

9. Deterministic startup with strict fail-fast model/cache/device validation: PASS  
Evidence: startup contract guard passes; runtime config env surface is restricted to required architecture keys (legacy alias fallbacks and strict-loading toggles removed); env/warmup checks fail fast with explicit reasons when artifacts are missing.

10. Runtime rejects live audio before global readiness: PASS  
Evidence: `runtime_not_ready_for_live_audio` guard covered in tests; global readiness now includes explicit kernel and transport READY state.

11. Required readiness payload fields and telemetry timestamps: PASS  
Evidence: runtime server contract tests validate required payload fields and readiness false-path when kernel lane is not READY.

12. Replay determinism checks: PASS  
Evidence: `scripts/run_replay_determinism.py` passing.

13. Real-only smoke/benchmark entrypoints: PASS  
Evidence: mocked helper execution has been removed from tracked runtime scripts; `scripts/run_real_backend_smoke.py` remains the supported full-chain smoke path and `scripts/run_latency_benchmark.py` now fails fast unless real benchmark inputs are present.

14. Real backend warmup/smoke and real E2E latency proof: BLOCKED (external artifacts missing)  
Evidence: `scripts/check_runtime_env.py` and `scripts/run_warmup_check.py` fail-fast on missing:
  - `D:/models/vosk/vosk-model-en-us-0.22`
  - `D:/models/vllm/Qwen3-8B`
  - `D:/models/cosyvoice3/Fun-CosyVoice3-0.5B-2512`

## Command Evidence (latest)

- `python apps/api/scripts/check_drift_guards.py` -> `ok`
- `python apps/api/scripts/check_startup_contract.py` -> `ok`
- `python scripts/run_drift_audit.py` -> `ok`
- `python -m pytest apps/api/tests -q` -> `70 passed`
- Mocked E2E helper removed; use `python scripts/run_real_backend_smoke.py`
- `python scripts/run_replay_determinism.py` -> `4 passed`
- `python scripts/run_voice_runtime.py --dry-run` -> `CPU ASR, GPU0 vLLM, GPU1 CosyVoice3`
- `python scripts/check_runtime_env.py` -> fail-fast on missing model artifacts (expected in this environment)

## Remaining External Dependencies

To clear blocked real-backend signoff items, place local model artifacts at configured paths and rerun warmup/smoke/real benchmark scripts.
