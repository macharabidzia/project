from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VP_ROOT = REPO_ROOT / "apps" / "api" / "src" / "voice_pipeline"


def test_locked_voice_pipeline_module_map_exists() -> None:
    required_paths = (
        VP_ROOT / "bus" / "ring_topology.py",
        VP_ROOT / "bus" / "ring_types.py",
        VP_ROOT / "bus" / "shm_ring.py",
        VP_ROOT / "governance" / "single_writer_audit.py",
        VP_ROOT / "gpu" / "tts_worker" / "engine.py",
        VP_ROOT / "gpu" / "vllm_worker" / "engine.py",
        VP_ROOT / "kernel" / "kernel_runtime.py",
        VP_ROOT / "runtime" / "bootstrap.py",
        VP_ROOT / "runtime" / "livekit_bridge.py",
        VP_ROOT / "stt" / "asr_engine.py",
        VP_ROOT / "transport" / "pcm_clock.py",
        VP_ROOT / "transport" / "livekit_transport.py",
    )
    for path in required_paths:
        assert path.exists(), f"missing locked runtime module: {path}"
