from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from runtime_entrypoint import ensure_runtime_library_path, ensure_runtime_python

REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "apps" / "api" / "src"

ensure_runtime_python()
ensure_runtime_library_path()
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from voice_pipeline.gpu.tts_worker.engine import TTSEngine
from voice_pipeline.runtime.admission_gate import AdmissionError
from voice_pipeline.runtime.bootstrap import _bind_cuda_device
from voice_pipeline.runtime.config import RuntimeConfig


def _validate_tts_inputs(config: RuntimeConfig) -> None:
    model_path = Path(str(config.resolved_cosyvoice3_model_path()).strip())
    if not str(model_path):
        raise AdmissionError("CosyVoice3 model path is not configured")
    if not model_path.exists():
        raise AdmissionError(f"CosyVoice3 model path does not exist: {model_path}")
    cache_dir = Path(str(config.cosyvoice3_cache_dir or "").strip())
    if not str(cache_dir):
        raise AdmissionError("CosyVoice3 cache directory is not configured")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not cache_dir.is_dir():
        raise AdmissionError(f"CosyVoice3 cache directory is not a directory: {cache_dir}")


async def _run() -> int:
    config = RuntimeConfig.from_env()
    _validate_tts_inputs(config)
    _bind_cuda_device(config.tts_device)

    engine = TTSEngine(
        config.resolved_cosyvoice3_model_path(),
        sample_rate=24_000,
        prompt_text=config.resolved_cosyvoice3_prompt_text(),
        prompt_speech_path=config.cosyvoice3_speaker_path,
        required_device=config.tts_device,
        cache_dir=config.cosyvoice3_cache_dir,
    )
    engine.warm(strict=True)
    engine.start_persistent_session(
        epoch_id="real-cosyvoice3-smoke:epoch:1",
        prompt_text=config.resolved_cosyvoice3_prompt_text(),
        prompt_speech_path=config.cosyvoice3_speaker_path,
    )

    saw_pcm = False
    async for pcm, sample_rate, _is_final in engine.stream_pcm(
        "hello",
        epoch_id="real-cosyvoice3-smoke:epoch:1",
    ):
        if pcm and int(sample_rate) > 0:
            saw_pcm = True
            break

    if not saw_pcm:
        raise RuntimeError("no PCM emitted from CosyVoice3 stream probe")

    print("real-cosyvoice3-smoke: READY")
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except Exception as exc:
        print(f"real-cosyvoice3-smoke: FAILED ({exc})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
