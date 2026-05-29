# Voice Pipeline Architecture Contract

## Scope

This repository implements a production single-machine, single-runtime voice pipeline.

This is the final architecture contract. Future work may only be implementation, testing, observability, latency hardening, warmup hardening, or CI enforcement - not architecture redesign.

## Non-negotiable topology

* One physical machine.
* One Python runtime process by default.
* One runtime authority: `KernelRuntime`.
* ASR: CPU Vosk native streaming (`cpu`).
* LLM: vLLM token streaming (`cuda:0`) with stable-prefix cache reuse.
* TTS: CosyVoice3 native bi-stream incremental PCM (`cuda:1`).
* Transport: local LiveKit WebRTC room (self-hosted) for media movement only.
* Bus/rings: bounded byte/event movement only.
* No Docker/Kubernetes/distributed workers/remote worker RPC.
* No worker-to-worker coordination.
* No extra orchestrator/control-FSM/runtime-manager authority.

## Canonical live flow

```text
20ms PCM ingress
-> ASR partial/final events
-> KernelRuntime.enqueue_event()
-> KernelRuntime reduce/commit
-> vLLM dispatch command
-> vLLM cuda:0 token stream
-> KernelRuntime token event reduce/commit
-> KernelRuntime fragment commit
-> TTS dispatch command
-> CosyVoice3 cuda:1 native bi-stream incremental PCM
-> Kernel output-version / lineage check
-> PCM egress via LiveKit outbound audio track
```

## Authority boundaries

* `KernelRuntime.enqueue_event()` is the only live authority ingress.
* Only kernel reducer transitions may mutate generation state, epoch, output version, leases, stable prefix, fragment commit, stale suppression, and dispatch intent.
* ASR/vLLM/TTS/transport are execution lanes; they do not own routing, policy, readiness interpretation, or conversation truth.
* No ASR -> vLLM direct call.
* No vLLM -> TTS direct call.
* No TTS -> vLLM direct call.
* No transport semantic recovery logic.
* Runtime/bootstrap may host engines and execute kernel dispatch commands, but it may not create semantic decisions.

## Runtime lock safety

* Kernel/runtime locks must protect state transition only.
* Dispatch command execution must happen outside the kernel/runtime lock.
* Worker output must re-enter through `KernelRuntime.enqueue_event()`.
* No lock may be held while awaiting ASR, vLLM, TTS, transport send, or PCM egress.
* No recursive tick/dispatch path may deadlock.

Required pattern:

```text
lock:
  order bounded events
  reduce
  commit state
  collect dispatch commands

unlock:
  execute dispatch commands
  enqueue worker output back through KernelRuntime.enqueue_event()
```

## Startup contract

Startup is deterministic, offline, strict, and fail-fast.

Startup order:

1. Load runtime config.
2. Resolve model/cache paths.
3. Validate topology (`cpu`, `cuda:0`, `cuda:1`).
4. Validate at least two CUDA devices.
5. Validate required local model artifacts.
6. Validate/create cache directories.
7. Create topology/rings.
8. Warm ASR.
9. Warm vLLM.
10. Warm TTS.
11. Create `KernelRuntime`.
12. Activate LiveKit transport bridge/tick loop.

Required config/env:

```text
VOSK_MODEL_PATH
VLLM_MODEL_PATH
VLLM_CACHE_DIR
COSYVOICE3_MODEL_PATH
COSYVOICE3_CACHE_DIR
COSYVOICE3_SPEAKER_PATH optional
```

Forbidden during startup/live runtime:

* implicit model download
* startup network fetch
* silent model substitution
* silent CPU fallback for vLLM/TTS
* silent `cuda:1` -> `cuda:0` remap
* accepting live audio before all lanes are READY
* degraded production live mode for missing required backend

Runtime does not accept live audio unless ASR/vLLM/TTS/kernel/transport are READY.

## Warmup/readiness requirements

Readiness states:

```text
WARMING | READY | FAILED
```

ASR READY only after:

* Vosk model path is validated.
* ASR runs on CPU.
* recognizer is created for configured sample rate.
* 20ms frame probe path succeeds.
* silent PCM probe is accepted.

vLLM READY only after:

* `cuda:0` bind succeeds.
* model/cache paths are validated.
* model is resident.
* stable prefix cache is prewarmed.
* stream-token probe succeeds.

TTS READY only after:

* `cuda:1` bind succeeds.
* model/cache paths are validated.
* native bi-stream session initializes.
* speaker/prompt embedding is loaded if configured.
* first incremental PCM probe succeeds.

Global readiness is true only when:

```text
ASR READY
AND vLLM READY
AND TTS READY
AND KernelRuntime READY
AND Transport READY
```

Only `KernelRuntime` may interpret global readiness for live routing.

## vLLM prefix cache contract

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
* speculative LLM continuations
* TTS fragments
* interrupted/stale epoch text
* timestamps
* random IDs
* queue metrics
* non-deterministic metadata

Barge-in invalidates only the turn-local suffix. Stable reusable prefix blocks may remain.

## CosyVoice3 native bi-stream contract

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
* repeated non-streaming calls to fake streaming
* TTS-local grouping policy
* TTS-local fragment policy
* TTS-local cancel/reset/stale-output policy

## Barge-in contract

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

## Queue/ring contract

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

## Latency and replay contract

Required timestamps:

```text
ingress_received_ns
asr_event_ns
kernel_decision_ns
vllm_first_token_ns
tts_first_pcm_ns
transport_emit_ns
```

Required gates:

* mocked continuous-session E2E p95 <= 150ms
* real target-machine E2E p95 <= 150ms, or explicit dependency/hardware failure
* barge-in old-PCM suppression p95 <= 40ms
* p99 always reported
* replay event/state hash parity

## Health/readiness payload

Required fields:

```text
asr_status
asr_device
asr_model_path
asr_sample_rate
llm_status
llm_device
llm_model_path
llm_cache_dir
llm_prefix_cache_ready
tts_status
tts_device
tts_model_path
tts_cache_dir
tts_native_bistream_ready
kernel_status
transport_status
topology_hash
model_cache_hash
drift_snapshot_hash
```

## Drift protection

Static/runtime checks enforce:

* single authority path
* fixed device map
* no fake TTS streaming fallback in live path
* no batch ASR in live path
* no startup network/download path
* no worker-to-worker authority bypass
* no runtime lock held while executing dispatch commands
* no recursive tick/dispatch deadlock path
* no stale PCM emission after epoch/output-version change
* no unbounded queue growth
* no Docker/distributed runtime assumption

## Final completion criteria

The system is complete only when:

* single-process runtime starts without Docker
* dry-run topology reports `ASR=cpu`, `vLLM=cuda:0`, `TTS=cuda:1`
* ASR warmup passes
* vLLM prefix-cache + stream-token probe passes
* TTS native bi-stream first-PCM probe passes
* live audio is rejected until READY
* ASR events flow through `KernelRuntime`
* vLLM tokens flow through `KernelRuntime`
* TTS fragments are Kernel-committed only
* PCM egress lineage-checks stale audio
* barge-in p95 <= 40ms
* mocked E2E p95 <= 150ms
* replay parity passes
* CI drift guards pass or every remaining hit is justified
* no second authority exists
* no fake TTS streaming exists
* no batch ASR live path exists
* no Docker/distributed worker assumption exists

## Final lock phrase

This is the final single-design runtime contract. Future work may only be implementation, testing, observability, latency hardening, warmup hardening, or CI enforcement - not architecture redesign.


