from __future__ import annotations

from pathlib import Path

from voice_pipeline.governance.single_writer_audit import audit_single_writers, write_audit_artifact


def _write_module(repo_root: Path, relative_path: str, source: str) -> None:
    target = repo_root / "apps" / "api" / "src" / "voice_pipeline" / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def test_single_writer_audit_passes_for_clean_minimal_layout(tmp_path: Path) -> None:
    _write_module(tmp_path, "kernel/kernel_runtime.py", "class KernelRuntime:\n    pass\n")
    _write_module(tmp_path, "runtime/bootstrap.py", "def bootstrap_runtime():\n    return None\n")
    _write_module(tmp_path, "gpu/vllm_worker/engine.py", "class VLLMEngine:\n    pass\n")

    result = audit_single_writers(tmp_path)

    assert result.ok is True
    assert result.violations == ()


def test_single_writer_audit_flags_runtime_and_worker_violations(tmp_path: Path) -> None:
    _write_module(tmp_path, "kernel/kernel_runtime.py", "class KernelRuntime:\n    pass\n")
    _write_module(
        tmp_path,
        "runtime/bootstrap.py",
        "def bad(runtime):\n    runtime.kernel._state = object()\n    return reduce_event(None, None)\n",
    )
    _write_module(
        tmp_path,
        "gpu/tts_worker/engine.py",
        "from voice_pipeline.kernel.reducer import reduce_event\nvalue = 'generation_index'\n",
    )

    result = audit_single_writers(tmp_path)

    assert result.ok is False
    assert any("runtime/bootstrap.py mutates kernel private state directly" in item for item in result.violations)
    assert any("runtime/bootstrap.py references reduce_event outside kernel" in item for item in result.violations)
    assert any("gpu/tts_worker/engine.py imports kernel authority internals from worker lane" in item for item in result.violations)

    artifact = tmp_path / "docs" / "authority-writer-audit.json"
    write_audit_artifact(result, artifact)
    saved = artifact.read_text(encoding="utf-8")
    assert "\"ok\": false" in saved
    assert "\"output_path\":" in saved
