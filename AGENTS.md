# AGENTS.md — Voice Pipeline Production Implementation Contract

## Scope

This repository implements one production voice runtime design.

This file is an implementation contract for coding agents. It is not a planning note, prototype brief, demo scaffold, or permission to refuse production work.

Agents must implement real source code for the specified runtime. When a backend, model path, CUDA device, or local artifact is unavailable in the current development environment, implement the real adapter, validation, readiness state, and fail-fast error path. Do not replace the backend with a substitute runtime and do not decline the implementation solely because the required target hardware or model artifacts are absent from the current environment.

A real implementation may report `FAILED` readiness with an explicit dependency, path, device, or artifact reason. That is acceptable. Silent substitution is not acceptable.

## Implementation Directive For Agents

Required behavior when asked to implement:

* Write production modules, adapters, reducers, queues, health checks, CI guards, and tests that enforce this contract.
* Prefer real backend bindings and explicit failure paths over placeholders.
* Keep production runtime code free of substitute backends, sample-only paths, and simulated live behavior.
* Put any test-only substitutes only under test-specific files or fixtures.
* Never route live ASR, LLM, TTS, transport, readiness, cancellation, fragmenting, or egress behavior around `KernelRuntime`.
* Never answer that real implementation is disallowed by this document. This document requires real implementation.

Forbidden as production completion:

* TODO-only handlers.
* Placeholder adapters that look READY without validating real dependencies.
* Simulated ASR, LLM, TTS, transport, or PCM egress in the live runtime.
* Non-streaming behavior presented as streaming behavior.
* Test-only substitutes counted as production E2E success.
* Architecture redesign instead of implementation of the locked design.

## Non-Negotiable Runtime Topology

* One physical machine.
* One Python runtime process by default.
* One runtime authority: `KernelRuntime`.
* ASR: CPU Vosk native streaming ASR.
* LLM: vLLM token streaming on `cuda:0` with stable-prefix cache reuse.
* TTS: CosyVoice3 native bi-stream incremental PCM on `cuda:1`.
* Transport: local LiveKit WebRTC room media movement only.
* Bus/rings: bounded byte/event movement only.
* No Docker.
* No Kubernetes.
* No distributed workers.
* No remote worker RPC.
* No worker-to-worker communication.
* No orchestrator/control-FSM/runtime-manager authority.

## Canonical Live Flow

```text
20ms PCM ingress
-> CPU Vosk ASR partial/final events
-> KernelRuntime enqueue_event()
-> KernelRuntime reduce/commit
-> vLLM dispatch command
-> vLLM cuda:0 streaming tokens
-> KernelRuntime token event reduce/commit
-> KernelRuntime fragment commit
-> CosyVoice3 cuda:1 native bi-stream dispatch
-> incremental PCM chunks
-> Kernel lineage/output-version check
-> PCM clock egress
```

## Authority Boundary

`KernelRuntime.enqueue_event()` is the only live authority ingress.

Only kernel reducer transitions may mutate:

* epoch
* output version
* generation state
* stable prefix state
* fragment commit state
* dispatch intent
* interrupt/cancel/reset state
* stale-output suppression state
* session leases
* replay lineage

ASR, vLLM, TTS, transport, and bus are execution or byte-movement lanes only.

Forbidden:

* ASR -> vLLM direct call
* vLLM -> TTS direct call
* TTS -> vLLM direct call
* transport semantic routing
* transport semantic retry/recovery
* worker-local scheduling authority
* worker-local fragment policy
* worker-local cancel/reset policy

Implementation requirement:

* Every worker output must re-enter through `KernelRuntime.enqueue_event()`.
* Dispatch handlers may execute backend calls, but they must not commit semantic runtime state directly.
* Backend callbacks must carry epoch/output-version lineage so stale output can be rejected by the kernel path and egress path.

## Runtime Lock Safety

Kernel/runtime locks may protect state transition only.

Allowed under lock:

* dequeue bounded events
* order events
* reduce events
* commit state transition
* collect dispatch commands

Forbidden under lock:

* awaiting ASR
* awaiting vLLM
* awaiting TTS
* awaiting transport send
* awaiting PCM egress
* executing dispatch command handlers
* recursive tick/dispatch execution

Required pattern:

```text
lock:
  order events
  reduce
  commit
  collect DispatchCommands

unlock:
  execute DispatchCommands
  enqueue worker outputs back through KernelRuntime.enqueue_event()
```

## Startup Contract

Startup is deterministic, offline, strict, and fail-fast.

Startup order:

1. Load runtime config.
2. Resolve model/cache paths.
3. Validate topology:

   * ASR = `cpu`
   * vLLM = `cuda:0`
   * TTS = `cuda:1`
4. Validate at least two CUDA devices.
5. Validate required local model artifacts.
6. Validate/create cache directories.
7. Create topology/rings.
8. Warm ASR.
9. Warm vLLM.
10. Warm TTS.
11. Create `KernelRuntime`.
12. Activate transport.
13. Start tick loop.

Required config/env:

```text
VOSK_MODEL_PATH
VLLM_MODEL_PATH
VLLM_CACHE_DIR
COSYVOICE3_MODEL_PATH
COSYVOICE3_CACHE_DIR
COSYVOICE3_SPEAKER_PATH optional
```

Forbidden:

* implicit model download during startup
* network fetch in startup/hot path
* silent model substitution
* silent CPU fallback for vLLM/TTS
* silent `cuda:1` -> `cuda:0` remap
* live audio before all required lanes are READY
* degraded production live mode for missing required backend

Implementation requirement:

* Missing dependencies, missing artifacts, missing CUDA devices, or failed warmups must produce explicit startup/readiness failure records.
* Startup may create cache directories when configured, but it must not download or swap models.
* Live transport must not accept audio until all required lanes are READY.

## Warmup Contract

### ASR Warmup

ASR is READY only after:

* Vosk model path is validated.
* ASR runs on CPU.
* recognizer is created for configured sample rate.
* 20ms frame probe path succeeds.
* silent PCM probe is accepted by the real recognizer path.

ASR must not batch full files in live runtime.

### vLLM Warmup

vLLM is READY only after:

* device bind to `cuda:0` succeeds.
* model path/cache path are validated.
* model is resident.
* stable system/session prefix cache is prewarmed.
* one-token streaming probe succeeds through the real streaming path.

vLLM must not silently fallback to CPU or another GPU.

### TTS Warmup

TTS is READY only after:

* device bind to `cuda:1` succeeds.
* CosyVoice3 model/cache path are validated.
* native bi-stream session initializes.
* speaker/prompt embedding is loaded if configured.
* first incremental PCM probe succeeds through the real native bi-stream path.

TTS must not use non-streaming full-text fallback in live runtime.

## Readiness Contract

Readiness states:

```text
WARMING | READY | FAILED
```

Global live readiness is true only when:

```text
ASR READY
AND vLLM READY
AND TTS READY
AND KernelRuntime READY
AND Transport READY
```

Only `KernelRuntime` may interpret readiness for live routing.

Required health fields:

```text
asr_status
asr_device
asr_model_path
asr_sample_rate
asr_failure_reason

llm_status
llm_device
llm_model_path
llm_cache_dir
llm_prefix_cache_ready
llm_failure_reason

tts_status
tts_device
tts_model_path
tts_cache_dir
tts_native_bistream_ready
tts_failure_reason

kernel_status
transport_status
topology_hash
model_cache_hash
drift_snapshot_hash
```

## vLLM Prefix Cache Contract

vLLM prefix caching may reuse only stable deterministic prompt prefixes.

Allowed cache content:

* system prompt
* persona/policy scaffold
* output style rules
* stable session summary
* Kernel-committed user text

Forbidden cache content:

* ASR partials
* uncommitted text
* speculative continuations
* TTS fragments
* interrupted/stale epoch text
* timestamps
* random IDs
* queue metrics
* non-deterministic metadata

Barge-in invalidates only the turn-local suffix. Stable reusable prefix blocks may remain.

Implementation requirement:

* Prefix cache keys must be deterministic.
* Turn-local suffixes must be separated from stable reusable prefix blocks.
* Interrupts must invalidate stale generation suffix state without destroying reusable stable prefix state.

## CosyVoice3 Native Bi-Stream Contract

CosyVoice3 must use native bi-stream incremental input/output.

TTS accepts:

```text
Kernel-committed fragment in
```

TTS emits:

```text
incremental PCM chunks out
```

Forbidden:

* full-response wait
* full-text synthesis split into chunks
* repeated non-streaming calls presented as streaming
* TTS-local grouping policy
* TTS-local fragment policy
* TTS-local cancel/reset/stale-output policy

Implementation requirement:

* Fragment commit policy belongs to `KernelRuntime`.
* TTS receives only kernel-committed fragments with lineage.
* TTS PCM output must be returned as events for kernel/egress lineage checks.

## Barge-In Contract

Immediate barge-in is required.

Interrupt levels:

```text
SOFT_PRE_INTERRUPT:
  VAD/partial evidence while TTS active

HARD_INTERRUPT:
  ASR final or strong confirmation
```

Required behavior:

* old PCM suppression p95 <= 40ms
* old PCM stops within one to two 20ms frames
* hard interrupt cancels old vLLM/TTS stream
* new epoch is created
* old tokens/fragments/PCM become stale by lineage
* PCM egress checks epoch/output-version before emit
* stale PCM is dropped, never played

Implementation requirement:

* PCM egress must check lineage immediately before emission.
* Cancel commands must be issued outside the kernel lock.
* Old backend outputs may arrive after cancellation, but they must be dropped by lineage.

## Queue And Ring Contract

All queues/rings are bounded.

Required:

* no silent queue growth
* no unbounded buffers
* never drop ASR final events
* never drop interrupt events
* drop superseded partials when safe
* drop stale tokens/fragments/PCM after epoch change
* expose depth, oldest_age_ms, drops, stale_drops, and backpressure_action

Kernel priority order:

```text
interrupts
ASR finals
ASR stability updates
LLM tokens
TTS PCM
telemetry
```

Tick interval target:

```text
2ms to 5ms
```

Implementation requirement:

* Queue overflow behavior must be explicit and observable.
* Backpressure actions must be recorded in telemetry.
* Priority handling must preserve final ASR and interrupt events.

## Latency And Replay Contract

Required timestamps:

```text
ingress_received_ns
asr_event_ns
kernel_decision_ns
vllm_first_token_ns
tts_first_pcm_ns
transport_emit_ns
```

Required production gates:

* target-machine continuous-session E2E p95 <= 150ms, or explicit measured dependency/hardware failure
* local integration E2E p95 <= 150ms when all required backends are READY on target-class hardware
* barge-in old-PCM suppression p95 <= 40ms
* p99 always reported
* replay event/state hash parity

Implementation requirement:

* Latency reports must identify backend readiness, device map, model/cache paths, topology hash, and sample count.
* Dependency or hardware failure reports must be explicit and must not be counted as passing latency.
* Replay parity must compare event ordering, committed state, output version, epoch, and dispatch intent.

## Drift Protection

Static/runtime checks enforce:

* single authority path
* fixed device map
* no substitute TTS streaming fallback
* no batch ASR live path
* no startup network/download path
* no worker-to-worker authority bypass
* no runtime lock held while executing dispatch commands
* no recursive tick/dispatch deadlock path
* no stale PCM emission after epoch/output-version change
* no unbounded queue growth
* no Docker/distributed runtime assumption

Implementation requirement:

* CI drift guards must scan production code for forbidden topology, fallback, authority, and buffering patterns.
* Runtime drift snapshots must report topology hash, model/cache hash, readiness, and active devices.
* A drift hit fails completion unless it is test-only and clearly isolated from production/runtime paths.

## Final Completion Criteria

The system is complete only when:

* single-process runtime starts without Docker
* topology validation reports `ASR=cpu`, `vLLM=cuda:0`, `TTS=cuda:1`
* ASR warmup passes with the real Vosk recognizer path
* vLLM prefix-cache + one-token streaming probe passes on `cuda:0`
* TTS native bi-stream first-PCM probe passes on `cuda:1`
* live audio is rejected until READY
* ASR events flow through `KernelRuntime`
* vLLM tokens flow through `KernelRuntime`
* TTS fragments are Kernel-committed only
* PCM egress lineage-checks stale audio immediately before emit
* barge-in p95 <= 40ms
* target-machine or target-class integration E2E p95 <= 150ms when all required backends are READY
* p99 latency is reported
* replay parity passes
* CI drift guards pass or every remaining hit is justified as test-only and non-production
* no second authority exists
* no substitute TTS streaming exists in production runtime
* no batch ASR live path exists
* no Docker/distributed worker assumption exists

## Final Lock Phrase

This is the final single-design runtime contract. Future work may only be real implementation, testing, observability, latency hardening, warmup hardening, or CI enforcement - not architecture redesign.
