from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "voice_pipeline"
KERNEL_WRITER_FILES = {
    SRC_ROOT / "kernel" / "kernel_runtime.py",
    SRC_ROOT / "kernel" / "leases.py",
    SRC_ROOT / "kernel" / "reducer.py",
    SRC_ROOT / "kernel" / "state.py",
}
READ_ONLY_RUNTIME_FILES = {
    SRC_ROOT / "runtime" / "bootstrap.py",
    SRC_ROOT / "runtime" / "server.py",
}


def test_generation_and_output_version_writes_are_kernel_scoped() -> None:
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path in KERNEL_WRITER_FILES:
            continue
        content = path.read_text(encoding="utf-8")
        assert "generation_index" not in content
        if path in READ_ONLY_RUNTIME_FILES:
            continue
        assert "output.version" not in content
