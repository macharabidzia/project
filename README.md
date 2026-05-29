# Voice OS

Single-machine continuous voice runtime:

- CPU: Vosk streaming ASR
- GPU0: vLLM token streaming
- GPU1: CosyVoice3 native bi-stream PCM
- One Python runtime process
- One `KernelRuntime` authority path

## Runtime Start

1) Verify `.env.voice_pipeline` points at your real local model/cache directories.

Required keys:

- `VOSK_MODEL_PATH`
- `VLLM_MODEL_PATH`
- `VLLM_CACHE_DIR`
- `COSYVOICE3_MODEL_PATH`
- `COSYVOICE3_CACHE_DIR`
- `COSYVOICE3_SPEAKER_PATH` (optional)
- `LIVEKIT_URL` (backend internal, usually `ws://127.0.0.1:7880`)
- `LIVEKIT_PUBLIC_URL` (browser-facing URL, optional but recommended on RunPod)
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `LIVEKIT_ROOM_NAME`

2) Start the runtime with backend hot reload:

```bash
bash scripts/start_voice_runtime_dev.sh
```

## Local LiveKit + TURN ports (self-hosted)

Expected local LiveKit signaling URL in this project:

- `LIVEKIT_URL=ws://127.0.0.1:7880`

Recommended exposed ports on the LiveKit host:

- `7880/tcp` signaling (WebSocket)
- `7881/tcp` ICE over TCP fallback (if enabled in your LiveKit config)
- `3478/udp` TURN/STUN
- `3478/tcp` TURN/STUN TCP fallback
- `50000-60000/udp` RTP/RTCP media range

When `livekit-turn-plugin` is enabled, keep the TURN listener ports and media UDP range open or WebRTC will degrade/fail behind restrictive NATs/firewalls.

## RunPod port wiring

Frontend/API defaults in this repo:

- Frontend dev server: `5173/tcp`
- Runtime API server: `8000/tcp`
- LiveKit signaling: `7880/tcp`

For browser WebRTC reliability on RunPod, also expose these additional ports:

- `7881/tcp` ICE over TCP fallback (recommended)
- `3478/udp` TURN/STUN
- `3478/tcp` TURN/STUN TCP fallback
- `50000-50100/udp` media range (minimum practical range; wider is safer)

If you can open one more port beyond your current set, open `3478/udp` first.

RunPod-friendly env example:

```bash
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_PUBLIC_URL=wss://<pod-id>-7880.proxy.runpod.net
VITE_BACKEND_URL=https://<pod-id>-8000.proxy.runpod.net
VOICE_PIPELINE_ALLOWED_ORIGINS=https://<pod-id>-5173.proxy.runpod.net
```

Dry-run topology check:

```bash
python scripts/run_voice_runtime.py --dry-run
```

## Runtime flow

```text
Browser mic PCM
-> LiveKit WebRTC room ingress
-> CPU Vosk ASR
-> KernelRuntime
-> GPU0 vLLM
-> KernelRuntime
-> GPU1 CosyVoice3
-> PCM clock
-> LiveKit WebRTC room egress
-> Browser audio output
```

## Runbook

Dry-run topology:

```bash
python scripts/run_voice_runtime.py --dry-run
```

Validate required env/model paths:

```bash
python scripts/check_runtime_env.py
```

Validate installed real backend runtimes:

```bash
python scripts/check_installed_runtime_backends.py
```

Warm all backends:

```bash
python scripts/run_warmup_check.py
```

Start runtime:

```bash
bash scripts/start_voice_runtime_dev.sh
```

Real backend smoke (ASR cpu + vLLM cuda:0 + CosyVoice3 cuda:1):

```bash
python scripts/run_real_backend_smoke.py
```

Real lane smokes:

```bash
python scripts/run_real_vosk_smoke.py
python scripts/run_real_vllm_smoke.py
python scripts/run_real_cosyvoice3_smoke.py
```

Full latency benchmark:

```bash
python scripts/run_latency_benchmark.py
```

Replay determinism:

```bash
python scripts/run_replay_determinism.py
```

CI drift audit:

```bash
python scripts/run_drift_audit.py
```

Single-writer governance audit:

```bash
python scripts/audit_voice_pipeline_writers.py
```

## Source of truth

- `ARCHITECTURE.md`
- `overview.md`
- `AGENTS.md`
