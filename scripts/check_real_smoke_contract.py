from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "scripts" / "run_real_backend_smoke.py"


def _expect_contains(violations: list[str], *, text: str, token: str) -> None:
    if token not in text:
        violations.append(f"missing required full-chain smoke token {token!r}")


def main() -> int:
    text = SMOKE_PATH.read_text(encoding="utf-8")
    violations: list[str] = []
    required_tokens = (
        "LiveKitRuntimeBridge",
        "bootstrap_runtime(",
        "runtime.process_pcm_frame(",
        "runtime.asr.finalize(",
        "runtime._asr_events_to_authority(",
        "runtime._append_event(",
        "runtime.run_tick_and_dispatch(",
        "\"ASRFinalReceived\"",
        "\"VLLMChunkReceived\"",
        "\"VLLMCompleted\"",
        "\"TTSChunkReceived\"",
        "\"TTSCompleted\"",
        "transport_egress_frames",
    )
    for token in required_tokens:
        _expect_contains(violations, text=text, token=token)

    if violations:
        for item in violations:
            print(item)
        raise SystemExit(1)

    print("real-smoke contract guard: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
