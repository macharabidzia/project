# plan.md

## Objective
Finalize production-hardening for the single-process voice pipeline runtime with strict authority under `KernelRuntime` and canonical live flow:
20ms PCM -> CPU Vosk streaming -> KernelRuntime stable-prefix authority -> vLLM cuda:0 token stream -> KernelRuntime fragment commit -> CosyVoice3 cuda:1 native bi-stream PCM -> PCM clock egress.

## Scope
- Preserve one-process architecture and remove/avoid drift paths.
- Enforce startup/warmup/device/cache/readiness contracts.
- Ensure ASR streaming, vLLM streaming with prefix caching, and CosyVoice3 native bi-stream incremental output are wired through KernelRuntime dispatch only.
- Validate with drift audits, startup checks, warmup checks, replay determinism, and mocked latency benchmark.

## Acceptance
Project is only complete when all AGENTS.md completion criteria are proven by code and command/test evidence.
