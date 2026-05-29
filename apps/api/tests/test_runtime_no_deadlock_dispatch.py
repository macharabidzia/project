from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP_PATH = REPO_ROOT / "apps" / "api" / "src" / "voice_pipeline" / "runtime" / "bootstrap.py"


def test_runtime_dispatch_path_uses_non_recursive_primitives() -> None:
    text = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    assert "async def _tick_and_stamp_commands" in text
    assert "async def _dispatch_commands" in text
    assert "blocked_kinds=frozenset({\"VLLM\"})" in text
    assert "blocked_kinds=frozenset({\"TTS\"})" in text
    assert "token_frames.extend(await self.run_tick_and_dispatch())" not in text
    assert "def execute_dispatch_command" not in text
