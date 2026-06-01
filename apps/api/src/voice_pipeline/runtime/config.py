from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


_ENV_LOADED = False
_COSYVOICE_ZERO_SHOT_PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"


def _apply_runtime_defaults() -> None:
    os.environ.setdefault("VOICE_PIPELINE_OFFLINE", "1")
    os.environ.setdefault("COSYVOICE_TEXT_FRONTEND", "off")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    os.environ.setdefault("VLLM_USE_STANDALONE_COMPILE", "0")
    os.environ.setdefault("VLLM_ENABLE_PREGRAD_PASSES", "0")
    os.environ.setdefault("VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", "0")
    os.environ.setdefault("VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING", "0")
    os.environ.setdefault("VLLM_SKIP_OPTIONAL_ENV_OVERRIDE_PATCHES", "1")
    os.environ.setdefault("VLLM_SKIP_ENV_OVERRIDE_IMPORT", "1")
    os.environ.setdefault("VLLM_TEXT_ONLY_RUNTIME", "1")
    os.environ.setdefault("TRANSFORMERS_TEXT_ONLY_RUNTIME", "1")
    os.environ.setdefault("TRANSFORMERS_DISABLE_FLASH_ATTN_2", "1")
    os.environ.setdefault("TRANSFORMERS_SKIP_MODEL_DEBUGGING_UTILS", "1")
    os.environ.setdefault("TRANSFORMERS_SKIP_SKLEARN_CANDIDATE_GENERATOR", "1")
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ.setdefault("TORCH_LIBRARY_SKIP_TRACEBACK", "1")
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    os.environ.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")
    os.environ.setdefault("VLLM_DISABLE_IR_FAKE_REGISTRATION", "1")
    ptxas_path = Path("/usr/local/cuda/bin/ptxas")
    if ptxas_path.is_file():
        os.environ.setdefault("TRITON_PTXAS_PATH", str(ptxas_path))
    os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/project/.cache/triton")
    os.environ.setdefault("COSYVOICE_STREAM_MIN_HOP_TOKENS", "2")
    os.environ.setdefault("COSYVOICE_STREAM_MAX_HOP_TOKENS", "4")
    os.environ.setdefault("COSYVOICE_STREAM_OVERLAP_TOKENS", "2")
    os.environ.setdefault("COSYVOICE_STREAM_SCALE_FACTOR", "1")
    os.environ.setdefault("COSYVOICE_BISTREAM_TEXT_MIX_TOKENS", "2")
    os.environ.setdefault("COSYVOICE_BISTREAM_SPEECH_MIX_TOKENS", "6")
    os.environ.setdefault("COSYVOICE_BISTREAM_MIN_STREAM_SPEECH_TOKENS", "4")


def _load_env_file_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_file = str(os.getenv("VOICE_PIPELINE_ENV_FILE", ".env.voice_pipeline")).strip()
    if env_file:
        path = Path(env_file)
        if path.exists() and path.is_file():
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                value = value.strip().strip("\"'")
                os.environ.setdefault(key, value)
    _apply_runtime_defaults()
    _ENV_LOADED = True


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    asr_device: str = "cpu"
    llm_device: str = "cuda:0"
    tts_device: str = "cuda:1"
    input_sample_rate: int = 48_000
    asr_sample_rate: int = 16_000
    output_sample_rate: int = 48_000
    frame_ms: int = 20
    pcm_target_buffer_frames: int = 6
    pcm_max_buffer_frames: int = 192
    tick_interval_ms: int = 2
    max_events_per_tick: int = 64
    ingress_max_items: int = 2048
    partial_history_size: int = 6
    stable_prefix_min_repeats: int = 2
    stable_prefix_min_tokens: int = 2
    stable_prefix_max_window: int = 3
    allow_partial_turn_commit: bool = True
    tts_fragment_min_tokens: int = 2
    tts_first_fragment_min_tokens: int = 1
    tts_fragment_max_tokens: int = 6
    tts_context_window_tokens: int = 24
    tts_max_queue_depth: int = 64
    speculative_partial_min_tokens: int = 1
    speculative_tts_start_tokens: int = 1
    tts_skip_stream_probe: bool = True
    tts_leading_silence_rms_threshold: float = 0.003
    tts_leading_silence_peak_threshold: float = 0.01
    latency_budget_ms: float = 150.0
    asr_ring_size: int = 1024
    vllm_ring_size: int = 1024
    tts_ring_size: int = 1024
    pcm_ring_size: int = 1024
    slot_bytes: int = 4096
    asr_model_path: str = ""
    vllm_model_path: str = ""
    vllm_cache_dir: str = ""
    vllm_temperature: float = 0.2
    vllm_top_p: float = 0.95
    vllm_max_tokens: int = 128
    vllm_gpu_memory_utilization: float = 0.98
    vllm_max_num_seqs: int = 1
    vllm_max_model_len: int = 1024
    vllm_max_num_batched_tokens: int = 64
    vllm_offload_backend: str = "auto"
    vllm_cpu_offload_gb: float = 0.0
    vllm_kv_offloading_size: float = 0.0
    vllm_kv_offloading_backend: str = "native"
    vllm_kv_cache_dtype: str = "auto"
    vllm_kv_cache_memory_bytes: int = 0
    vllm_num_gpu_blocks_override: int = 0
    vllm_attention_backend: str = "auto"
    vllm_safetensors_load_strategy: str = "prefetch"
    vllm_system_prompt: str = (
        "Reply briefly in plain spoken text only."
    )
    vllm_session_summary_turns: int = 0
    cosyvoice3_model_path: str = ""
    cosyvoice3_cache_dir: str = ""
    cosyvoice3_speaker_path: str = ""
    cosyvoice3_prompt_text: str = ""
    livekit_url: str = "ws://127.0.0.1:7880"
    livekit_public_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_room_name: str = "voice-runtime"
    livekit_runtime_identity: str = "voice-runtime-backend"
    livekit_output_track_name: str = "voice-runtime-out"
    livekit_input_participant_identity: str = ""
    livekit_input_track_name: str = ""
    livekit_input_frame_ms: int = 20
    livekit_input_queue_size_ms: int = 40
    livekit_output_queue_size_ms: int = 40
    livekit_output_preconnect_buffer: bool = False
    livekit_single_peer_connection: bool = False
    livekit_single_ingress_track: bool = True
    livekit_use_silero_vad: bool = True
    livekit_forward_all_audio: bool = True
    livekit_silero_vad_min_speech_ms: int = 40
    livekit_silero_vad_min_silence_ms: int = 160
    livekit_silero_vad_prefix_padding_ms: int = 80
    livekit_post_vad_tail_ms: int = 120
    livekit_silero_vad_activation_threshold: float = 0.35
    livekit_use_turn_detector: bool = True
    livekit_turn_detector_min_endpoint_ms: int = 60
    livekit_turn_detector_max_endpoint_ms: int = 400
    livekit_turn_detector_unlikely_threshold: float = 0.5
    livekit_token_ttl_seconds: int = 3600

    def resolved_vllm_model_path(self) -> str:
        return str(self.vllm_model_path).strip()

    def resolved_cosyvoice3_model_path(self) -> str:
        return str(self.cosyvoice3_model_path).strip()

    def resolved_cosyvoice3_prompt_text(self) -> str:
        explicit = " ".join(str(self.cosyvoice3_prompt_text or "").strip().split())
        if explicit:
            return explicit
        speaker_path = Path(str(self.cosyvoice3_speaker_path or "").strip())
        if speaker_path.name == "zero_shot_prompt.wav":
            return _COSYVOICE_ZERO_SHOT_PROMPT_TEXT
        return ""

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        _load_env_file_once()
        cosyvoice_model = os.getenv("COSYVOICE3_MODEL_PATH", "")
        return cls(
            asr_device=os.getenv("VOICE_PIPELINE_ASR_DEVICE", "cpu"),
            llm_device=os.getenv("VOICE_PIPELINE_LLM_DEVICE", "cuda:0"),
            tts_device=os.getenv("VOICE_PIPELINE_TTS_DEVICE", "cuda:1"),
            input_sample_rate=int(os.getenv("VOICE_PIPELINE_INPUT_SAMPLE_RATE", "48000")),
            asr_sample_rate=int(os.getenv("VOICE_PIPELINE_ASR_SAMPLE_RATE", "16000")),
            output_sample_rate=int(os.getenv("VOICE_PIPELINE_OUTPUT_SAMPLE_RATE", "48000")),
            frame_ms=int(os.getenv("VOICE_PIPELINE_FRAME_MS", "20")),
            pcm_target_buffer_frames=int(os.getenv("VOICE_PIPELINE_PCM_TARGET_BUFFER_FRAMES", "6")),
            pcm_max_buffer_frames=int(os.getenv("VOICE_PIPELINE_PCM_MAX_BUFFER_FRAMES", "192")),
            tick_interval_ms=int(os.getenv("VOICE_PIPELINE_TICK_INTERVAL_MS", "2")),
            max_events_per_tick=int(os.getenv("VOICE_PIPELINE_MAX_EVENTS_PER_TICK", "64")),
            ingress_max_items=int(os.getenv("VOICE_PIPELINE_INGRESS_MAX_ITEMS", "2048")),
            partial_history_size=int(os.getenv("VOICE_PIPELINE_PARTIAL_HISTORY_SIZE", "6")),
            stable_prefix_min_repeats=int(os.getenv("VOICE_PIPELINE_STABLE_PREFIX_MIN_REPEATS", "2")),
            stable_prefix_min_tokens=int(os.getenv("VOICE_PIPELINE_STABLE_PREFIX_MIN_TOKENS", "2")),
            stable_prefix_max_window=int(os.getenv("VOICE_PIPELINE_STABLE_PREFIX_MAX_WINDOW", "3")),
            allow_partial_turn_commit=str(
                os.getenv("VOICE_PIPELINE_ALLOW_PARTIAL_TURN_COMMIT", "1")
            ).strip().lower() in {"1", "true", "yes", "on"},
            tts_fragment_min_tokens=int(os.getenv("VOICE_PIPELINE_TTS_FRAGMENT_MIN_TOKENS", "2")),
            tts_first_fragment_min_tokens=int(os.getenv("VOICE_PIPELINE_TTS_FIRST_FRAGMENT_MIN_TOKENS", "1")),
            tts_fragment_max_tokens=int(os.getenv("VOICE_PIPELINE_TTS_FRAGMENT_MAX_TOKENS", "6")),
            tts_context_window_tokens=int(os.getenv("VOICE_PIPELINE_TTS_CONTEXT_WINDOW_TOKENS", "24")),
            tts_max_queue_depth=int(os.getenv("VOICE_PIPELINE_TTS_MAX_QUEUE_DEPTH", "64")),
            speculative_partial_min_tokens=int(os.getenv("VOICE_PIPELINE_SPECULATIVE_PARTIAL_MIN_TOKENS", "1")),
            speculative_tts_start_tokens=int(os.getenv("VOICE_PIPELINE_SPECULATIVE_TTS_START_TOKENS", "1")),
            tts_skip_stream_probe=str(os.getenv("VOICE_PIPELINE_TTS_SKIP_STREAM_PROBE", "1")).strip().lower()
            in {"1", "true", "yes", "on"},
            tts_leading_silence_rms_threshold=float(
                os.getenv("VOICE_PIPELINE_TTS_LEADING_SILENCE_RMS_THRESHOLD", "0.003")
            ),
            tts_leading_silence_peak_threshold=float(
                os.getenv("VOICE_PIPELINE_TTS_LEADING_SILENCE_PEAK_THRESHOLD", "0.01")
            ),
            latency_budget_ms=float(os.getenv("VOICE_PIPELINE_LATENCY_BUDGET_MS", "150")),
            asr_ring_size=int(os.getenv("VOICE_PIPELINE_ASR_RING_SIZE", "1024")),
            vllm_ring_size=int(os.getenv("VOICE_PIPELINE_VLLM_RING_SIZE", "1024")),
            tts_ring_size=int(os.getenv("VOICE_PIPELINE_TTS_RING_SIZE", "1024")),
            pcm_ring_size=int(os.getenv("VOICE_PIPELINE_PCM_RING_SIZE", "1024")),
            slot_bytes=int(os.getenv("VOICE_PIPELINE_SLOT_BYTES", "4096")),
            asr_model_path=os.getenv("VOSK_MODEL_PATH", ""),
            vllm_model_path=os.getenv("VLLM_MODEL_PATH", ""),
            vllm_cache_dir=os.getenv("VLLM_CACHE_DIR", ""),
            vllm_temperature=float(os.getenv("VOICE_PIPELINE_VLLM_TEMPERATURE", "0.2")),
            vllm_top_p=float(os.getenv("VOICE_PIPELINE_VLLM_TOP_P", "0.95")),
            vllm_max_tokens=int(os.getenv("VOICE_PIPELINE_VLLM_MAX_TOKENS", "128")),
            vllm_gpu_memory_utilization=float(os.getenv("VOICE_PIPELINE_VLLM_GPU_MEMORY_UTILIZATION", "0.98")),
            vllm_max_num_seqs=int(os.getenv("VOICE_PIPELINE_VLLM_MAX_NUM_SEQS", "1")),
            vllm_max_model_len=int(os.getenv("VOICE_PIPELINE_VLLM_MAX_MODEL_LEN", "1024")),
            vllm_max_num_batched_tokens=int(os.getenv("VOICE_PIPELINE_VLLM_MAX_NUM_BATCHED_TOKENS", "64")),
            vllm_offload_backend=os.getenv("VOICE_PIPELINE_VLLM_OFFLOAD_BACKEND", "auto"),
            vllm_cpu_offload_gb=float(os.getenv("VOICE_PIPELINE_VLLM_CPU_OFFLOAD_GB", "0")),
            vllm_kv_offloading_size=float(os.getenv("VOICE_PIPELINE_VLLM_KV_OFFLOADING_SIZE", "0")),
            vllm_kv_offloading_backend=os.getenv("VOICE_PIPELINE_VLLM_KV_OFFLOADING_BACKEND", "native"),
            vllm_kv_cache_dtype=os.getenv("VOICE_PIPELINE_VLLM_KV_CACHE_DTYPE", "auto"),
            vllm_kv_cache_memory_bytes=int(os.getenv("VOICE_PIPELINE_VLLM_KV_CACHE_MEMORY_BYTES", "0")),
            vllm_num_gpu_blocks_override=int(os.getenv("VOICE_PIPELINE_VLLM_NUM_GPU_BLOCKS_OVERRIDE", "0")),
            vllm_attention_backend=os.getenv("VOICE_PIPELINE_VLLM_ATTENTION_BACKEND", "auto"),
            vllm_safetensors_load_strategy=os.getenv(
                "VOICE_PIPELINE_VLLM_SAFETENSORS_LOAD_STRATEGY",
                "prefetch",
            ),
            vllm_system_prompt=os.getenv(
                "VOICE_PIPELINE_VLLM_SYSTEM_PROMPT",
                (
                    "Reply briefly in plain spoken text only."
                ),
            ),
            vllm_session_summary_turns=int(os.getenv("VOICE_PIPELINE_VLLM_SESSION_SUMMARY_TURNS", "0")),
            cosyvoice3_model_path=str(cosyvoice_model),
            cosyvoice3_cache_dir=os.getenv("COSYVOICE3_CACHE_DIR", ""),
            cosyvoice3_speaker_path=os.getenv("COSYVOICE3_SPEAKER_PATH", ""),
            cosyvoice3_prompt_text=os.getenv("COSYVOICE3_PROMPT_TEXT", ""),
            livekit_url=os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880"),
            livekit_public_url=os.getenv("LIVEKIT_PUBLIC_URL", ""),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", ""),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
            livekit_room_name=os.getenv("LIVEKIT_ROOM_NAME", "voice-runtime"),
            livekit_runtime_identity=os.getenv("LIVEKIT_RUNTIME_IDENTITY", "voice-runtime-backend"),
            livekit_output_track_name=os.getenv("LIVEKIT_OUTPUT_TRACK_NAME", "voice-runtime-out"),
            livekit_input_participant_identity=os.getenv("LIVEKIT_INPUT_PARTICIPANT_IDENTITY", ""),
            livekit_input_track_name=os.getenv("LIVEKIT_INPUT_TRACK_NAME", ""),
            livekit_input_frame_ms=int(os.getenv("LIVEKIT_INPUT_FRAME_MS", os.getenv("VOICE_PIPELINE_FRAME_MS", "20"))),
            livekit_input_queue_size_ms=int(os.getenv("LIVEKIT_INPUT_QUEUE_SIZE_MS", "40")),
            livekit_output_queue_size_ms=int(os.getenv("LIVEKIT_OUTPUT_QUEUE_SIZE_MS", "40")),
            livekit_output_preconnect_buffer=str(
                os.getenv("LIVEKIT_OUTPUT_PRECONNECT_BUFFER", "0")
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            livekit_single_peer_connection=str(
                os.getenv("LIVEKIT_SINGLE_PEER_CONNECTION", "0")
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            livekit_single_ingress_track=str(
                os.getenv("LIVEKIT_SINGLE_INGRESS_TRACK", "1")
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            livekit_use_silero_vad=str(
                os.getenv("LIVEKIT_USE_SILERO_VAD", "1")
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            livekit_forward_all_audio=str(
                os.getenv("LIVEKIT_FORWARD_ALL_AUDIO", "1")
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            livekit_silero_vad_min_speech_ms=int(
                os.getenv("LIVEKIT_SILERO_VAD_MIN_SPEECH_MS", "40")
            ),
            livekit_silero_vad_min_silence_ms=int(
                os.getenv("LIVEKIT_SILERO_VAD_MIN_SILENCE_MS", "160")
            ),
            livekit_silero_vad_prefix_padding_ms=int(
                os.getenv("LIVEKIT_SILERO_VAD_PREFIX_PADDING_MS", "80")
            ),
            livekit_post_vad_tail_ms=int(
                os.getenv("LIVEKIT_POST_VAD_TAIL_MS", "120")
            ),
            livekit_silero_vad_activation_threshold=float(
                os.getenv("LIVEKIT_SILERO_VAD_ACTIVATION_THRESHOLD", "0.35")
            ),
            livekit_use_turn_detector=str(
                os.getenv("LIVEKIT_USE_TURN_DETECTOR", "1")
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            livekit_turn_detector_min_endpoint_ms=int(
                os.getenv("LIVEKIT_TURN_DETECTOR_MIN_ENDPOINT_MS", "60")
            ),
            livekit_turn_detector_max_endpoint_ms=int(
                os.getenv("LIVEKIT_TURN_DETECTOR_MAX_ENDPOINT_MS", "400")
            ),
            livekit_turn_detector_unlikely_threshold=float(
                os.getenv("LIVEKIT_TURN_DETECTOR_UNLIKELY_THRESHOLD", "0.5")
            ),
            livekit_token_ttl_seconds=int(os.getenv("LIVEKIT_TOKEN_TTL_SECONDS", "3600")),
        )


__all__ = ["RuntimeConfig"]
