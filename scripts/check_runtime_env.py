from __future__ import annotations

import os
from pathlib import Path

REQUIRED_PATH_KEYS = (
    "VOSK_MODEL_PATH",
    "VLLM_MODEL_PATH",
    "VLLM_CACHE_DIR",
    "COSYVOICE3_MODEL_PATH",
    "COSYVOICE3_CACHE_DIR",
)
OPTIONAL_PATH_KEYS = ("COSYVOICE3_SPEAKER_PATH",)
REQUIRED_FIXED_VALUES = {
    "VOICE_PIPELINE_ASR_DEVICE": "cpu",
    "VOICE_PIPELINE_LLM_DEVICE": "cuda:0",
    "VOICE_PIPELINE_TTS_DEVICE": "cuda:1",
    "VOICE_PIPELINE_FRAME_MS": "20",
}


def _check_required_artifacts() -> list[str]:
    violations: list[str] = []
    vosk_root = Path(str(os.getenv("VOSK_MODEL_PATH", "")).strip())
    if str(vosk_root):
        for artifact in (vosk_root / "am" / "final.mdl", vosk_root / "conf" / "model.conf"):
            if not artifact.exists():
                violations.append(f"missing Vosk artifact: {artifact}")

    vllm_root = Path(str(os.getenv("VLLM_MODEL_PATH", "")).strip())
    if str(vllm_root):
        if not (vllm_root / "config.json").exists():
            violations.append(f"missing vLLM artifact: {vllm_root / 'config.json'}")
        if not any(vllm_root.glob("model*.safetensors")) and not any(vllm_root.glob("pytorch_model*.bin")):
            violations.append(f"missing vLLM weight shards under: {vllm_root}")

    cosy_root = Path(str(os.getenv("COSYVOICE3_MODEL_PATH", "")).strip())
    if str(cosy_root):
        for artifact_name in ("cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt"):
            artifact = cosy_root / artifact_name
            if not artifact.exists():
                violations.append(f"missing CosyVoice3 artifact: {artifact}")
    return violations


def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
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
        os.environ.setdefault(key, value.strip().strip("\"'"))


def _check_path(key: str, required: bool) -> tuple[bool, str]:
    value = str(os.getenv(key, "")).strip()
    if not value:
        if required:
            return False, "missing"
        return True, "not-set(optional)"
    if value.startswith("http://") or value.startswith("https://"):
        return False, f"remote-path-forbidden: {value}"
    path = Path(value)
    if key.endswith("_DIR"):
        path.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.is_dir():
            return True, "ok"
        return False, f"not-directory: {value}"
    if path.exists():
        return True, "ok"
    return False, f"missing-path: {value}"


def main() -> int:
    env_file = Path(os.getenv("VOICE_PIPELINE_ENV_FILE", ".env.voice_pipeline"))
    _load_env_file(env_file)

    all_ok = True
    print(f"env-file: {env_file} ({'found' if env_file.exists() else 'missing'})")

    for key in REQUIRED_PATH_KEYS:
        ok, detail = _check_path(key, required=True)
        all_ok = all_ok and ok
        print(f"{key}: {detail}")

    for key in OPTIONAL_PATH_KEYS:
        ok, detail = _check_path(key, required=False)
        all_ok = all_ok and ok
        print(f"{key}: {detail}")

    for key, expected in REQUIRED_FIXED_VALUES.items():
        observed = str(os.getenv(key, expected)).strip()
        print(f"{key}: {observed}")
        if observed.lower() != str(expected).lower():
            all_ok = False
            print(f"{key}: expected {expected}, got {observed}")

    for violation in _check_required_artifacts():
        all_ok = False
        print(violation)

    if not all_ok:
        print("runtime-env-check: FAILED")
        return 1
    print("runtime-env-check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
