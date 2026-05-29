from __future__ import annotations

from voice_pipeline.kernel.kernel_runtime import KernelRuntime


def test_kernel_runtime_session_initial_state_is_idle() -> None:
    kernel = KernelRuntime(session_id="integration-speech-session")
    assert kernel.state.phase == "idle"
