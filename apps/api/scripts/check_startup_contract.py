from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO_ROOT / "src" / "voice_pipeline" / "runtime"
HOT_STARTUP_FILES = (
    RUNTIME_ROOT / "bootstrap.py",
    RUNTIME_ROOT / "admission_gate.py",
    RUNTIME_ROOT / "config.py",
)

FORBIDDEN_NETWORK = re.compile(
    r"from_pretrained\(|snapshot_download|hf_hub_download|download_model|requests\.get\(|\bhttpx\b|\burllib\b"
)

REQUIRED_ENV_KEYS = (
    "VOSK_MODEL_PATH",
    "VLLM_MODEL_PATH",
    "VLLM_CACHE_DIR",
    "COSYVOICE3_MODEL_PATH",
    "COSYVOICE3_CACHE_DIR",
    "COSYVOICE3_SPEAKER_PATH",
)

FORBIDDEN_RUNTIME_TOGGLES = (
    "warm_strict",
    "warmup_required",
    "hot_reload_enabled",
    "cosyvoice_stream",
    "VOICE_PIPELINE_COSYVOICE_STREAM",
    "VOICE_PIPELINE_VLLM_MODEL_PATH",
    "VOICE_PIPELINE_VLLM_CACHE_DIR",
    "VOICE_PIPELINE_COSYVOICE_CACHE_DIR",
    "VOICE_PIPELINE_VLLM_MODEL",
    "VOICE_PIPELINE_COSYVOICE_MODEL_DIR",
    "VOICE_PIPELINE_ASR_INPUT_SAMPLE_RATE",
    "COSYVOICE_PROMPT_TEXT",
    "COSYVOICE_PROMPT_AUDIO_PATH",
    "cosyvoice_prompt_text",
    "cosyvoice_prompt_audio_path",
    "strict_model_loading",
    "asr_input_sample_rate",
    "VOICE_PIPELINE_NVLINK_ENABLED",
    "nvlink_enabled",
)


def _expect_contains(violations: list[str], *, text: str, token: str, label: str) -> None:
    if token not in text:
        violations.append(f"missing required startup contract token {token!r} in {label}")


def _expect_not_contains(violations: list[str], *, text: str, token: str, label: str) -> None:
    if token in text:
        violations.append(f"forbidden startup contract token {token!r} present in {label}")


def main() -> int:
    violations: list[str] = []
    bootstrap_path = RUNTIME_ROOT / "bootstrap.py"
    admission_path = RUNTIME_ROOT / "admission_gate.py"
    config_path = RUNTIME_ROOT / "config.py"
    cli_path = RUNTIME_ROOT / "cli.py"

    bootstrap_text = bootstrap_path.read_text(encoding="utf-8")
    admission_text = admission_path.read_text(encoding="utf-8")
    config_text = config_path.read_text(encoding="utf-8")
    cli_text = cli_path.read_text(encoding="utf-8")

    for path in HOT_STARTUP_FILES:
        text = path.read_text(encoding="utf-8")
        if FORBIDDEN_NETWORK.search(text):
            violations.append(f"forbidden network/download reference in startup path: {path}")

    for key in REQUIRED_ENV_KEYS:
        _expect_contains(violations, text=config_text, token=key, label=str(config_path))

    for token in FORBIDDEN_RUNTIME_TOGGLES:
        _expect_not_contains(violations, text=config_text, token=token, label=str(config_path))

    _expect_contains(violations, text=bootstrap_text, token='contract_violation: asr device must be cpu', label=str(bootstrap_path))
    _expect_contains(violations, text=bootstrap_text, token='contract_violation: vllm device must be cuda:0', label=str(bootstrap_path))
    _expect_contains(violations, text=bootstrap_text, token='contract_violation: tts device must be cuda:1', label=str(bootstrap_path))
    _expect_contains(violations, text=bootstrap_text, token='contract_violation: frame_ms must be 20', label=str(bootstrap_path))

    _expect_contains(violations, text=bootstrap_text, token="hardware_admission_check(runtime_config)", label=str(bootstrap_path))
    _expect_contains(violations, text=admission_text, token="fewer than 2 CUDA devices are visible", label=str(admission_path))
    _expect_contains(violations, text=admission_text, token="vLLM device must be cuda:0", label=str(admission_path))
    _expect_contains(violations, text=admission_text, token="CosyVoice3 device must be cuda:1", label=str(admission_path))
    _expect_contains(violations, text=admission_text, token="ASR device must be cpu", label=str(admission_path))

    _expect_contains(violations, text=bootstrap_text, token="asr.warm(strict=True)", label=str(bootstrap_path))
    _expect_contains(violations, text=bootstrap_text, token="vllm.warm(strict=True)", label=str(bootstrap_path))
    _expect_contains(violations, text=bootstrap_text, token="tts.warm(strict=True)", label=str(bootstrap_path))
    _expect_contains(
        violations,
        text=bootstrap_text,
        token='startup_failed: required lane warmup did not reach READY',
        label=str(bootstrap_path),
    )
    _expect_contains(violations, text=bootstrap_text, token='worker_status.kernel = "READY"', label=str(bootstrap_path))
    _expect_contains(violations, text=bootstrap_text, token='and str(self.worker_status.kernel) == "READY"', label=str(bootstrap_path))
    _expect_contains(violations, text=bootstrap_text, token="async def _tick_and_stamp_commands", label=str(bootstrap_path))
    _expect_contains(violations, text=bootstrap_text, token="async def _dispatch_commands", label=str(bootstrap_path))
    _expect_contains(violations, text=bootstrap_text, token='blocked_kinds=frozenset({"VLLM"})', label=str(bootstrap_path))
    _expect_contains(violations, text=bootstrap_text, token='blocked_kinds=frozenset({"TTS"})', label=str(bootstrap_path))
    _expect_not_contains(
        violations,
        text=bootstrap_text,
        token="token_frames.extend(await self.run_tick_and_dispatch())",
        label=str(bootstrap_path),
    )

    warm_asr_index = bootstrap_text.find("warm_report.asr_warm = _warm_asr_engine")
    warm_vllm_index = bootstrap_text.find("warm_report.vllm_warm = _warm_vllm_engine")
    warm_tts_index = bootstrap_text.find("warm_report.tts_warm = _warm_tts_engine")
    if -1 in {warm_asr_index, warm_vllm_index, warm_tts_index}:
        violations.append(f"missing required warmup call sequence in {bootstrap_path}")
    elif not (warm_asr_index < warm_vllm_index < warm_tts_index):
        violations.append(f"warmup call order must be ASR -> vLLM -> TTS in {bootstrap_path}")

    _expect_contains(violations, text=cli_text, token="_assert_runtime_contract(config)", label=str(cli_path))
    _expect_contains(violations, text=cli_text, token='print("CPU ASR, GPU0 vLLM, GPU1 CosyVoice3")', label=str(cli_path))
    _expect_contains(violations, text=cli_text, token="reload=True", label=str(cli_path))
    _expect_not_contains(violations, text=cli_text, token="--profile", label=str(cli_path))
    _expect_not_contains(violations, text=bootstrap_text, token="DEGRADED", label=str(bootstrap_path))

    if violations:
        for item in violations:
            print(item)
        raise SystemExit(1)

    print("startup contract guard: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
