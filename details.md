Your 150 ms target is valid only if you treat it as “first audible assistant audio after usable input / first stable LLM text,” not full answer completion. CosyVoice3 itself claims bi-streaming support: text can stream in while audio streams out, with latency “as low as 150 ms.” But your uploaded overview shows a much bigger cascade around it: ASR, VAD/turn detection, vLLM, TTS fragment planning, kernel ticks, speculative TTS, resampling, PCM clocking, LiveKit bridge, rings, and observability.

The core issue: CosyVoice3 native bi-stream can hit 150 ms, but your pipeline can easily destroy that by waiting for stable text, batching fragments, finalizing ASR, opening a TTS leading gate late, resampling/chunking too coarsely, or routing through a non-native path.

Biggest bugs drifting you away from 150 ms
1. You may be calling “streaming,” but not true native bi-stream

Your TTS engine has both native streaming and one-shot paths: _call_cosyvoice_native_stream, _call_cosyvoice_native_once, _resolve_native_stream_inference, _force_zero_shot_native_stream, and CosyVoiceBiStreamSession.

That means the first bug to check is: any fallback to _call_cosyvoice_native_once is a 150 ms killer. For the 150 ms path, every real-time request must use persistent native bi-stream:

LLM token stream
  -> token/text fragment stream
  -> CosyVoice3 native text-in stream
  -> CosyVoice3 audio-out stream
  -> small PCM chunks
  -> LiveKit/PCM clock immediately

Tasks:

Log which TTS path is used for every request: native_bistream, native_stream, native_once, or fallback.
Make fallback to _call_cosyvoice_native_once loud, not silent.
Reject live mode if native bi-stream is unavailable.
Keep CosyVoiceBiStreamSession persistent across turns.
Do not create a fresh CosyVoice session per fragment.
Do not re-prime prompt/audio cache inside the hot path.
Make _resolve_native_stream_inference return native bi-stream for live sessions only.
Make _force_zero_shot_native_stream impossible to bypass accidentally.
Add metric: tts_backend_path.
Add metric: tts_first_pcm_ms.

Also be careful if you are assuming vLLM-Omni already gives you CosyVoice3 text-in bi-stream. A vLLM-Omni issue opened on April 29, 2026 asked to “Add bi-streaming support for CosyVoice3,” so that integration may lag behind native CosyVoice. Use native CosyVoice3 directly unless you have confirmed your exact vLLM-Omni version supports text-in/audio-out bi-stream.

2. Your fragment planner may be waiting for “nice text” instead of “speakable text”

You have kernel/tts_fragment_planner.py with plan_tts_fragment, flush_text, _is_complete_short_clause, _is_boundaryless_lexical_prefix, _should_keep_stream_open, token boundary checks, punctuation checks, and dangling-function-word checks.

That is probably a major latency source. For quality, planners often wait for punctuation or a complete clause. For 150 ms, that is too late.

Bug pattern:

Wait for sentence end
Wait for comma
Wait for stable prefix
Wait for 3–8 tokens
Then start TTS

That gives smooth speech, but it drifts away from 150 ms.

Tasks:

Add a 150 ms mode to TTSFragmentPlannerConfig.
Allow first fragment after 1–2 lexical tokens.
Allow first fragment without punctuation.
Add max_wait_ms_before_first_tts_fragment, probably 20–40 ms.
Add max_tokens_before_first_tts_fragment, probably 1–3 tokens.
Keep later fragments smarter, but make the first fragment aggressive.
Never let _should_keep_stream_open mean “do not emit audio.”
Track llm_first_token_to_tts_first_text_ms.
Track tts_first_text_to_first_pcm_ms.
Track fragment_wait_reason: punctuation, boundaryless prefix, dangling word, stable prefix, max wait, etc.

For 150 ms, the first fragment can be tiny. The native CosyVoice3 bi-stream feature exists exactly so you do not need full sentence buffering.

3. Your “TTS leading gate” can silently add 50–300 ms

Your runtime has _should_emit_tts_frame, _batch_opens_tts_leading_gate, _tts_leading_batch_start_index, _record_trimmed_tts_frame, and _should_drop_final_tts_resampler_tail.

That sounds like you are gating or trimming early audio. This is dangerous for 150 ms.

Bug pattern:

CosyVoice emits first PCM
runtime stores it
gate waits for enough frames / enough RMS / batch open
then sends

That defeats native streaming.

Tasks:

Log when first PCM arrives from TTS.
Log when first PCM is actually sent to transport.
Compute tts_pcm_gate_delay_ms.
For 150 ms mode, open the leading gate on the first non-silent valid frame.
Do not wait for a batch to open the gate.
Do not trim first audible frames unless there is proven noise.
Make _batch_opens_tts_leading_gate optional or disabled for live mode.
Keep a tiny pre-roll only if needed, like 10–20 ms, not 100+ ms.
Make _should_emit_tts_frame return false only for stale/cancelled/empty frames, not “too early.”
Add test: first generated PCM must reach send_pcm_once within 15 ms.
4. Speculative TTS may exist but not be promoted early enough

Your runtime has _SpeculativeVLLMRequest, _SpeculativeTTSRequest, _start_speculative_tts_for_text, _run_speculative_tts_request, _promote_speculative_tts_request, and _drain_promoted_speculative_tts.

That is good, but speculation only helps if it starts before final ASR and promotes immediately.

Bug pattern:

ASR final arrives
then vLLM starts
then first text arrives
then TTS starts

That is not 150 ms. You need:

ASR partial/stable prefix arrives
speculative vLLM starts
first spoken tokens arrive
speculative TTS starts
user finishes
promote already-running TTS
send first audio

Tasks:

Start speculative vLLM on stable ASR partials, not only final ASR.
Start speculative TTS on first flushable LLM text.
Promote speculative TTS immediately on final ASR confirmation.
Do not cancel/restart TTS if the final transcript differs only slightly.
Add fuzzy stable-prefix matching between partial and final ASR.
Track spec_tts_started_before_final_asr.
Track spec_tts_promoted_ms_after_final_asr.
Track spec_tts_cancel_count.
Track spec_tts_restart_count.
Target: promoted TTS already has PCM buffered or nearly ready when user finishes.

Without speculation, 150 ms after user stop is very hard because ASR finalization + LLM first token + TTS first PCM can easily exceed it.

5. ASR finalization / VAD delay can consume the whole budget

Your LiveKit bridge has _build_silero_vad, _build_turn_detector, _turn_finalize_delay_seconds, _consume_remote_audio, and _emit_runtime_pcm. Your ASR engine has streaming partial/final paths: ingest_partial, ingest_final, ingest_audio, and finalize.

If _turn_finalize_delay_seconds is even 200 ms, the 150 ms target is already lost before LLM/TTS start.

Tasks:

Measure user_last_audio_to_asr_final_ms.
Measure vad_speech_end_to_finalize_ms.
In live mode, do not wait for final ASR before speculative response.
Reduce turn-finalize delay for low-latency mode.
Use ASR partials for stable prefix.
Make finalize_asr_turn confirm, not initiate, the response path.
Use short input frames, ideally 10–20 ms.
Avoid input resampling in the hot path when possible.
Log ASR partial cadence.
Add fail condition: if ASR finalization alone exceeds 80 ms, 150 ms end-to-end is impossible for that turn.
6. vLLM first-token latency may be larger than your entire remaining budget

Your vLLM engine supports streaming tokens, prefix cache readiness, prewarm_prefix_cache, stream_tokens, _split_flushable_spoken_prefix, _append_spoken_delta, and spoken-output normalization.

Bug pattern:

LLM waits for full prompt render
prefix cache miss
first token slow
spoken text normalization waits
flushable prefix waits
TTS starts late

Tasks:

Measure asr_final_or_partial_to_vllm_request_ms.
Measure vllm_request_to_first_token_ms.
Measure vllm_first_token_to_first_spoken_delta_ms.
Prewarm prefix cache for the stable system prompt.
Make prefix_cache_ready required for live mode.
Keep system prompt short.
Strip reasoning before speech, as your code already has _strip_reasoning_sections.
Do not wait for a full sentence in _split_flushable_spoken_prefix.
Emit spoken deltas as soon as they are safe to say.
Add test: first spoken delta under 40 ms in warm path.

For 150 ms, vLLM cannot behave like a normal chat completion. It must be a warmed, prefix-cached, streaming, spoken-delta engine.

7. Kernel tick scheduling can add hidden frame delay

Your kernel has TickEngine.run_once, drift_alarm_triggered, and drift_snapshot; KernelRuntime.tick, commit_tick, event queues, priority, stale suppression, and runtime metrics.

Bug pattern:

event arrives
waits until next tick
command produced
waits until dispatch tick
worker emits
waits until next tick
transport sends

If one tick is 20–50 ms and you need several ticks, you lose 60–150 ms just in scheduling.

Tasks:

Record event_enqueued_to_tick_ms.
Record tick_to_dispatch_ms.
Record dispatch_to_worker_start_ms.
Record worker_output_to_kernel_commit_ms.
Reduce live tick interval.
Allow immediate tick on high-priority events: ASR partial, first LLM token, first TTS PCM, cancel.
Bypass normal queue delay for first-audio path.
Ensure _event_priority treats first audio / cancel / ASR final as protected.
Watch queued_event_count.
Fail test if kernel scheduling adds more than 10–20 ms to first audio.

The tick engine should protect determinism, not become a metronome that delays first audio.

8. Ring buffers can hide stale latency

Your bus has SharedMemoryRing.depth, overwrite_count, oldest_age_ms, push, pop, drain, and byte-slot methods.

Bug pattern:

ring depth grows
consumer pops old frame
oldest_age_ms climbs
audio is technically streaming, but stale

Tasks:

Track oldest_age_ms per ring.
Alert when any live ring has oldest age over 30 ms.
Drain stale non-protected events aggressively.
Keep TTS PCM ring small for live mode.
Drop old PCM after cancellation or new epoch.
Never play stale audio from previous epoch.
Make overwrite count visible in telemetry.
Add ring depth to latency summary.
Give first PCM priority over bulk audio.
Add test: ring age must stay under 20–30 ms during first-audio path.

For real-time voice, a full buffer is not “safe”; it is latency.

9. Output chunking / PCM clock may be too coarse

Your runtime has _resample_output, _output_frame_bytes, _chunk_output_pcm, and send_pcm_once; transport has PCMClockSender.enqueue, _pop_fresh, oldest_age_ms, and run_once.

Bug pattern:

TTS emits audio
runtime waits to make a big chunk
resampler buffers
PCM clock queues
sender releases on next interval

Tasks:

Use 10–20 ms output chunks.
Do not wait for 100 ms chunks.
Keep PCMClockSender.depth near zero.
Make _pop_fresh drop stale frames, not preserve them.
Track pcm_enqueue_to_send_ms.
Track output_resampler_buffer_ms.
Do not call resampler flush on every fragment.
Reset resampler only on turn/session boundary, not each tiny fragment.
Send first chunk immediately when clock-safe.
Add test: first PCM chunk leaves transport within one frame period.
10. Resampling can add delay if sample rates do not match

Your ASR engine has _resample_input_audio; shared code has StreamingAudioResampler; runtime has _resample_output.

Bug pattern:

LiveKit 48k -> ASR 16k
TTS 24k/22.05k -> LiveKit 48k
resampler buffers internally
flush/tail handling waits

Tasks:

Standardize sample rates where possible.
Use streaming resampler, not whole-buffer resampling.
Measure resampler input-to-output delay.
Avoid final-tail flush before first audio.
Do not drop first valid frames while trying to remove tail.
Keep output chunks frame-aligned.
Pre-create resamplers at session start.
Avoid resampler reset per fragment.
Track resample_ms_per_chunk.
Add test: resampling adds under 5–10 ms to first audio.
11. Warmup exists, but live mode must require it

Your runtime has _warm_asr_engine, _warm_vllm_engine, _warm_tts_engine, warm_vllm_runtime_probe, warm_tts_runtime_probe, and _warm_tts_generator_runtime_probe.

Bug pattern:

first real user triggers model load / CUDA init / prompt cache / tokenizer load

Tasks:

Live mode must fail closed if ASR, vLLM, and TTS are not warm.
Warm CosyVoice native bi-stream, not only one-shot TTS.
Warm the exact speaker/prompt path.
Warm vLLM prefix cache.
Warm tokenizer.
Warm CUDA context on the correct device.
Warm resamplers and PCM sender.
Run a synthetic first-audio probe at boot.
Store warm report with measured first-token and first-PCM times.
Refuse “ready” if warm probe exceeds target.
12. CUDA/device binding mistakes can cause random latency spikes

Your code has _assert_cuda_device_binding in TTS and vLLM areas, _cuda_device_context, _bind_cuda_device, and admission checks for CUDA devices.

Bug pattern:

vLLM and TTS fight same GPU
wrong CUDA_VISIBLE_DEVICES
context switch
first call initializes CUDA lazily

Tasks:

Pin ASR CPU, vLLM GPU, TTS GPU explicitly.
Confirm device binding at worker start and every hot request.
Avoid TTS and vLLM starving each other.
Keep CUDA context warm.
Log GPU id per worker.
Log CUDA sync stalls.
Avoid per-request model/device moves.
Separate GPUs if possible.
Use admission check to reject bad deployment.
Add p95/p99 latency alerts, not just average.
13. Text normalization can block first audio

CosyVoice3 advertises text normalization support, including numbers and special symbols. But your pipeline also has text normalization functions in shared text, vLLM spoken normalization, TTS fragment normalization, and language-mode detection.

Bug pattern:

LLM emits first words
normalizer waits for more context
language detector waits
metadata stripper waits
TTS receives text late

Tasks:

Normalize incrementally.
Do not run heavy text normalization before first fragment.
Let CosyVoice handle supported normalization when safe.
Cache language decision per session.
Avoid cross-lingual mode switching mid-first-fragment.
Strip metadata before live response path.
Make _strip_followup_metadata_tail non-blocking.
Track text_normalization_ms.
Track first_spoken_delta_block_reason.
Add test with numbers, abbreviations, and punctuation.
14. Cancellation / epoch logic can kill good speculative audio

Your kernel has leases, lineage, stale epoch checks, stale output suppression, request binding, and output versioning.

Bug pattern:

speculative TTS starts
final ASR changes epoch
kernel marks TTS stale
TTS cancels/restarts
first audio delayed

Tasks:

Make speculative output promotable across compatible epoch changes.
Only cancel if text meaning changed, not if transcript spacing changed.
Make lineage matching tolerant for stable-prefix continuation.
Log every stale suppression reason.
Track good_speculative_audio_discarded_ms.
Track tts_cancel_to_restart_ms.
Avoid cancelling TTS on harmless ASR final updates.
Keep request-output version stable when final confirms partial.
Add deterministic replay tests for partial→final promotion.
Add test: “hello there” partial to “hello there.” final should not restart TTS.
15. Observability is present, but the exact 150 ms path needs its own timeline

You have latency_summary, last_timestamps, tts_signal_metrics, ingress_frame_trace, asr_event_trace, Trace.add, timeline, and summarize_latency.

Bug pattern:

you measure total latency
but not the exact delay segment that broke 150 ms

Tasks:

Add these exact timestamps:

t_user_audio_frame_in
t_vad_speech_start
t_first_asr_partial
t_stable_asr_partial
t_asr_final
t_vllm_request_start
t_vllm_first_token
t_first_spoken_delta
t_tts_text_push
t_tts_native_stream_open
t_tts_first_pcm
t_tts_gate_open
t_resampler_first_output
t_pcm_enqueue
t_pcm_send
t_livekit_egress

Then compute:

asr_partial_ms
asr_finalize_ms
llm_first_token_ms
spoken_delta_ms
tts_first_pcm_ms
tts_gate_delay_ms
resample_delay_ms
pcm_queue_delay_ms
transport_delay_ms
total_first_audio_ms

Do not chase “150 ms” until this timeline exists. Otherwise you are guessing.

The minimum 150 ms budget

Use this as your target budget:

ASR partial / stable prefix:        10–30 ms
Speculative vLLM first spoken text: 20–50 ms
Text -> CosyVoice push:              1–5 ms
CosyVoice first PCM:                60–90 ms
Gate/resample/chunk/send:           10–20 ms
--------------------------------------------
Target first audible audio:        100–150 ms

If you wait for ASR final before vLLM, your budget becomes more like:

VAD final delay:       100–500 ms
ASR final:              20–100 ms
vLLM first token:       30–200 ms
TTS first PCM:          60–150 ms
transport:              10–50 ms

That will not reliably hit 150 ms.

Priority order

Fix in this order:

Force true native CosyVoice3 bi-stream. No one-shot fallback, no fake streaming.
Start speculative vLLM/TTS from ASR partials.
Make first TTS fragment emit after 1–2 speakable tokens.
Remove TTS leading-gate batching for first audio.
Shrink output chunks to 10–20 ms.
Keep PCM/ring buffers fresh, not deep.
Prewarm ASR, vLLM prefix cache, CosyVoice session, CUDA, tokenizer, and resamplers.
Make kernel tick react immediately to first-token and first-PCM events.
Instrument the full first-audio timeline.
Add a hard CI test: warm path first audio must be under 150 ms.
Your likely “real bug” in one sentence

You probably already have the pieces for 150 ms, but the pipeline is still architected like this:

ASR final -> vLLM -> planned text fragment -> TTS stream -> gate -> batch -> resample -> send

For 150 ms it must become:

ASR partial -> speculative vLLM -> tiny spoken delta -> native CosyVoice3 bi-stream -> first PCM immediately -> send

That is the difference between “CosyVoice3 supports 150 ms” and your product actually reaching 150 ms.