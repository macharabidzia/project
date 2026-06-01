from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "voice_pipeline"


def _scan(pattern: str, path: Path) -> list[str]:
    regex = re.compile(pattern)
    hits: list[str] = []
    for file in sorted(path.rglob("*.py")):
        text = file.read_text(encoding="utf-8")
        if regex.search(text):
            hits.append(str(file))
    return hits


def main() -> int:
    failures: list[str] = []

    cross_runtime_hits = _scan(r"ASREngine.*VLLMEngine|VLLMEngine.*TTSEngine|TTSEngine.*VLLMEngine", SRC_ROOT)
    if cross_runtime_hits:
        failures.append(f"forbidden cross-worker coupling: {cross_runtime_hits}")

    remote_hits = _scan(r"docker|grpc|\brpc\b|remote worker|distributed worker|runtime manager|orchestrator", SRC_ROOT)
    if remote_hits:
        failures.append(f"forbidden remote/orchestrator patterns: {remote_hits}")

    tts_fake_hits = _scan(
        r"_fallback_stream_pcm|full_text|non_streaming|batch_tts|\binference_stream\b|stream=False",
        SRC_ROOT / "gpu" / "tts_worker",
    )
    if tts_fake_hits:
        failures.append(f"forbidden fake TTS streaming patterns: {tts_fake_hits}")

    asr_batch_hits = _scan(r"transcribe_file|wavfile|full_audio|batch_asr", SRC_ROOT / "stt")
    if asr_batch_hits:
        failures.append(f"forbidden ASR batch hot-path patterns: {asr_batch_hits}")

    runtime_policy_hits = _scan(r"plan_tts_fragment|detect_stable_prefix|pressure_score", SRC_ROOT / "runtime")
    if runtime_policy_hits:
        failures.append(f"runtime must not host kernel policy helpers: {runtime_policy_hits}")

    runtime_interrupt_policy_hits = _scan(r"SOFT_PRE_INTERRUPT|HARD_INTERRUPT", SRC_ROOT / "runtime")
    if runtime_interrupt_policy_hits:
        failures.append(f"runtime must not host interrupt policy literals: {runtime_interrupt_policy_hits}")

    network_download_hits = _scan(
        r"snapshot_download|hf_hub_download|download_model|requests|httpx|urllib",
        SRC_ROOT,
    )
    if network_download_hits:
        failures.append(f"forbidden startup/live network or implicit download patterns: {network_download_hits}")

    stt_to_gpu_hits = _scan(r"voice_pipeline\.gpu\.(vllm_worker|tts_worker)", SRC_ROOT / "stt")
    if stt_to_gpu_hits:
        failures.append(f"ASR lane must not import GPU worker lanes: {stt_to_gpu_hits}")

    vllm_to_other_lane_hits = _scan(r"voice_pipeline\.gpu\.tts_worker|voice_pipeline\.stt", SRC_ROOT / "gpu" / "vllm_worker")
    if vllm_to_other_lane_hits:
        failures.append(f"vLLM lane must not import TTS/ASR lanes: {vllm_to_other_lane_hits}")

    tts_to_other_lane_hits = _scan(r"voice_pipeline\.gpu\.vllm_worker|voice_pipeline\.stt", SRC_ROOT / "gpu" / "tts_worker")
    if tts_to_other_lane_hits:
        failures.append(f"TTS lane must not import vLLM/ASR lanes: {tts_to_other_lane_hits}")

    stt_kernel_import_hits = _scan(r"voice_pipeline\.kernel\.", SRC_ROOT / "stt")
    if stt_kernel_import_hits:
        failures.append(f"ASR lane must not import kernel authority internals: {stt_kernel_import_hits}")

    gpu_kernel_import_hits = _scan(r"voice_pipeline\.kernel\.", SRC_ROOT / "gpu")
    if gpu_kernel_import_hits:
        failures.append(f"GPU worker lanes must not import kernel authority internals: {gpu_kernel_import_hits}")

    transport_kernel_import_hits = _scan(r"voice_pipeline\.kernel\.", SRC_ROOT / "transport")
    if transport_kernel_import_hits:
        failures.append(f"transport lane must not import kernel authority internals: {transport_kernel_import_hits}")

    transport_semantic_message_hits = _scan(
        r"MSG_TOKEN_EVENT|MSG_CONTROL|MSG_ASR_EVENT|MSG_HEARTBEAT|encode_control_payload|decode_control_payload|encode_token_payload",
        SRC_ROOT / "transport",
    )
    if transport_semantic_message_hits:
        failures.append(
            f"transport must stay framed-byte movement only (no semantic/control message helpers): {transport_semantic_message_hits}"
        )

    hot_reload_hits = _scan(r"hot_reload|hot-reload", SRC_ROOT)
    if hot_reload_hits:
        failures.append(f"hot reload drift is forbidden in locked runtime: {hot_reload_hits}")

    production_mock_hits = _scan(r"\bmock\b|\bfake\b|\bsimulat(e|ed|ion)\b|\bstub\b|\bdummy\b|\bplaceholder\b", SRC_ROOT)
    if production_mock_hits:
        failures.append(f"production runtime must not include mock/fake/simulated fallback surfaces: {production_mock_hits}")

    non_stream_toggle_hits = _scan(r"VOICE_PIPELINE_COSYVOICE_STREAM|cosyvoice_stream", SRC_ROOT)
    if non_stream_toggle_hits:
        failures.append(f"forbidden non-streaming TTS toggle surface detected: {non_stream_toggle_hits}")

    tts_stream_signature_toggle_hits = _scan(r"\bstream\s*:\s*bool", SRC_ROOT / "gpu" / "tts_worker")
    if tts_stream_signature_toggle_hits:
        failures.append(f"TTS native stream mode must be structural, not exposed as bool signature toggle: {tts_stream_signature_toggle_hits}")

    apply_event_bypass_hits: list[str] = []
    for file in sorted(SRC_ROOT.rglob("*.py")):
        if str(file).replace("\\", "/").endswith("/kernel/kernel_runtime.py"):
            continue
        text = file.read_text(encoding="utf-8")
        if ".apply_event(" in text:
            apply_event_bypass_hits.append(str(file))
    if apply_event_bypass_hits:
        failures.append(f"kernel authority bypass via apply_event outside kernel runtime: {apply_event_bypass_hits}")

    asr_auto_warm_hits = _scan(r"self\.warm\(", SRC_ROOT / "stt")
    if asr_auto_warm_hits:
        failures.append(f"ASR lane must not auto-warm during live ingest path: {asr_auto_warm_hits}")

    tts_generic_inference_hits = _scan(r'"inference",', SRC_ROOT / "gpu" / "tts_worker")
    if tts_generic_inference_hits:
        failures.append(f"TTS lane must not fall back to non-native generic inference entrypoint: {tts_generic_inference_hits}")

    tts_oneshot_hits = _scan(r"VOICE_PIPELINE_TTS_NATIVE_ONESHOT_MAX_TOKENS|native_once|oneshot", SRC_ROOT / "gpu" / "tts_worker")
    if tts_oneshot_hits:
        failures.append(f"TTS lane must not contain one-shot fallback paths in locked live runtime: {tts_oneshot_hits}")

    vllm_random_request_hits = _scan(r"os\.urandom", SRC_ROOT / "gpu" / "vllm_worker")
    if vllm_random_request_hits:
        failures.append(f"vLLM request IDs must be authority-provided, not random fallback: {vllm_random_request_hits}")

    legacy_env_alias_hits = _scan(
        r"VOICE_PIPELINE_VLLM_MODEL|VOICE_PIPELINE_COSYVOICE_MODEL_DIR|VOICE_PIPELINE_ASR_INPUT_SAMPLE_RATE|COSYVOICE_PROMPT_TEXT|COSYVOICE_PROMPT_AUDIO_PATH",
        SRC_ROOT,
    )
    if legacy_env_alias_hits:
        failures.append(f"forbidden legacy env alias/prompt fallback surface detected: {legacy_env_alias_hits}")

    strict_toggle_hits = _scan(r"strict_model_loading", SRC_ROOT)
    if strict_toggle_hits:
        failures.append(f"strict model-loading toggle surface is forbidden in locked runtime: {strict_toggle_hits}")

    nvlink_toggle_hits = _scan(r"VOICE_PIPELINE_NVLINK_ENABLED|nvlink_enabled", SRC_ROOT)
    if nvlink_toggle_hits:
        failures.append(f"forbidden NVLINK topology toggle surface detected: {nvlink_toggle_hits}")

    forbidden_paths = [
        SRC_ROOT / "gpu" / "tts_worker" / "executor.py",
        SRC_ROOT / "gpu" / "vllm_worker" / "executor.py",
        SRC_ROOT / "kernel" / "backpressure.py",
        SRC_ROOT / "kernel" / "invariants.py",
        SRC_ROOT / "kernel" / "replay.py",
        SRC_ROOT / "stt" / "stream.py",
        SRC_ROOT / "stt" / "frame_buffer.py",
        SRC_ROOT / "stt" / "vad.py",
        SRC_ROOT / "transport" / "frame_adapter.py",
        SRC_ROOT / "transport" / "ws_framed.py",
        SRC_ROOT / "shared" / "audio.py",
        SRC_ROOT / "shared" / "errors.py",
    ]
    restored_forbidden_paths = [str(path) for path in forbidden_paths if path.exists()]
    if restored_forbidden_paths:
        failures.append(f"forbidden drift modules reintroduced: {restored_forbidden_paths}")

    bootstrap_path = SRC_ROOT / "runtime" / "bootstrap.py"
    bootstrap_text = bootstrap_path.read_text(encoding="utf-8")
    vllm_start = bootstrap_text.find("async def _execute_vllm_command")
    tts_start = bootstrap_text.find("async def _execute_tts_command")
    resample_start = bootstrap_text.find("def _resample_output")
    if -1 in {vllm_start, tts_start, resample_start}:
        failures.append(f"runtime bootstrap missing dispatch execution sections: {bootstrap_path}")
    else:
        vllm_block = bootstrap_text[vllm_start:tts_start]
        tts_block = bootstrap_text[tts_start:resample_start]
        if "run_tick_and_dispatch(" in vllm_block:
            failures.append("vLLM dispatch path must not recursively re-enter run_tick_and_dispatch")
        if "run_tick_and_dispatch(" in tts_block:
            failures.append("TTS dispatch path must not recursively re-enter run_tick_and_dispatch")
    if "def execute_dispatch_command" in bootstrap_text:
        failures.append("legacy execute_dispatch_command path is forbidden in non-recursive runtime dispatch design")

    runtime_server = SRC_ROOT / "runtime" / "server.py"
    server_text = runtime_server.read_text(encoding="utf-8")
    if "/v1/voice/ws" in server_text:
        failures.append("manual voice websocket endpoint is forbidden; use livekit transport only")
    if "WebSocket" in server_text:
        failures.append("manual websocket transport dependency is forbidden in runtime server")

    if failures:
        for item in failures:
            print(item)
        raise SystemExit(1)

    print("drift guard: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
