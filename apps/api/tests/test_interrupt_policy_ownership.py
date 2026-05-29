from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REDUCER_PATH = REPO_ROOT / "apps" / "api" / "src" / "voice_pipeline" / "kernel" / "reducer.py"
BOOTSTRAP_PATH = REPO_ROOT / "apps" / "api" / "src" / "voice_pipeline" / "runtime" / "bootstrap.py"


def test_interrupt_policy_literals_live_in_kernel_reducer_only() -> None:
    reducer_text = REDUCER_PATH.read_text(encoding="utf-8")
    bootstrap_text = BOOTSTRAP_PATH.read_text(encoding="utf-8")

    assert "SOFT_PRE_INTERRUPT" in reducer_text
    assert "HARD_INTERRUPT" in reducer_text
    assert "SOFT_PRE_INTERRUPT" not in bootstrap_text
    assert "HARD_INTERRUPT" not in bootstrap_text

