from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


_ENV_LOADED = False


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
    tick_interval_ms: int = 2
    max_events_per_tick: int = 64
    ingress_max_items: int = 2048
    partial_history_size: int = 6
    stable_prefix_min_repeats: int = 2
    stable_prefix_min_tokens: int = 2
    stable_prefix_max_window: int = 3
    tts_fragment_min_tokens: int = 2
    tts_fragment_max_tokens: int = 6
    tts_context_window_tokens: int = 24
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
    cosyvoice3_model_path: str = ""
    cosyvoice3_cache_dir: str = ""
    cosyvoice3_speaker_path: str = ""
    livekit_url: str = "ws://127.0.0.1:7880"
    livekit_public_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_room_name: str = "voice-runtime"
    livekit_runtime_identity: str = "voice-runtime-backend"
    livekit_output_track_name: str = "voice-runtime-out"
    livekit_token_ttl_seconds: int = 3600

    def resolved_vllm_model_path(self) -> str:
        return str(self.vllm_model_path).strip()

    def resolved_cosyvoice3_model_path(self) -> str:
        return str(self.cosyvoice3_model_path).strip()

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
            tick_interval_ms=int(os.getenv("VOICE_PIPELINE_TICK_INTERVAL_MS", "2")),
            max_events_per_tick=int(os.getenv("VOICE_PIPELINE_MAX_EVENTS_PER_TICK", "64")),
            ingress_max_items=int(os.getenv("VOICE_PIPELINE_INGRESS_MAX_ITEMS", "2048")),
            partial_history_size=int(os.getenv("VOICE_PIPELINE_PARTIAL_HISTORY_SIZE", "6")),
            stable_prefix_min_repeats=int(os.getenv("VOICE_PIPELINE_STABLE_PREFIX_MIN_REPEATS", "2")),
            stable_prefix_min_tokens=int(os.getenv("VOICE_PIPELINE_STABLE_PREFIX_MIN_TOKENS", "2")),
            stable_prefix_max_window=int(os.getenv("VOICE_PIPELINE_STABLE_PREFIX_MAX_WINDOW", "3")),
            tts_fragment_min_tokens=int(os.getenv("VOICE_PIPELINE_TTS_FRAGMENT_MIN_TOKENS", "2")),
            tts_fragment_max_tokens=int(os.getenv("VOICE_PIPELINE_TTS_FRAGMENT_MAX_TOKENS", "6")),
            tts_context_window_tokens=int(os.getenv("VOICE_PIPELINE_TTS_CONTEXT_WINDOW_TOKENS", "24")),
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
            cosyvoice3_model_path=str(cosyvoice_model),
            cosyvoice3_cache_dir=os.getenv("COSYVOICE3_CACHE_DIR", ""),
            cosyvoice3_speaker_path=os.getenv("COSYVOICE3_SPEAKER_PATH", ""),
            livekit_url=os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880"),
            livekit_public_url=os.getenv("LIVEKIT_PUBLIC_URL", ""),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", ""),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
            livekit_room_name=os.getenv("LIVEKIT_ROOM_NAME", "voice-runtime"),
            livekit_runtime_identity=os.getenv("LIVEKIT_RUNTIME_IDENTITY", "voice-runtime-backend"),
            livekit_output_track_name=os.getenv("LIVEKIT_OUTPUT_TRACK_NAME", "voice-runtime-out"),
            livekit_token_ttl_seconds=int(os.getenv("LIVEKIT_TOKEN_TTL_SECONDS", "3600")),
        )


__all__ = ["RuntimeConfig"]
