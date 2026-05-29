from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "voice_pipeline"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_transport_layers_do_not_own_turn_or_cancel_semantics() -> None:
    livekit_transport = _read(SRC_ROOT / "transport" / "livekit_transport.py")

    forbidden_tokens = (
        "TurnCommitted",
        "CancelRequested",
        "InterruptRequested",
        "conversation_history",
    )
    for token in forbidden_tokens:
        assert token not in livekit_transport


def test_vllm_worker_does_not_store_conversation_truth() -> None:
    engine_source = _read(SRC_ROOT / "gpu" / "vllm_worker" / "engine.py")
    forbidden_tokens = (
        "conversation_history",
        "active_turn_id",
        "cancelled",
    )
    for token in forbidden_tokens:
        assert token not in engine_source
