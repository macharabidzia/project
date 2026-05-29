from __future__ import annotations

from datetime import datetime, timezone
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AuditResult:
    ok: bool
    violations: tuple[str, ...]

_WORKER_PREFIXES = ("gpu/", "stt/", "transport/")
_KERNEL_STATE_WRITE_TOKENS = (
    "generation" + "_index",
    "committed_turn" + "_index",
    "turn" + "_index",
    "output" + ".version",
    "Output" + "State(",
    "request_event" + "_ids",
)
_KERNEL_APPLY_EVENT_TOKEN = ".apply_" + "event("


def _collect_violations(*, rel: str, text: str) -> list[str]:
    hits: list[str] = []
    if rel.startswith("governance/"):
        return hits

    if not rel.startswith("kernel/"):
        if "reduce_event(" in text:
            hits.append(f"{rel} references reduce_event outside kernel")
        if "enqueue_authority_event(" in text:
            hits.append(f"{rel} references enqueue_authority_event outside kernel")
        if _KERNEL_APPLY_EVENT_TOKEN in text:
            hits.append(f"{rel} references apply_event outside kernel runtime")
        if "kernel._state" in text:
            hits.append(f"{rel} mutates kernel private state directly")
        if "kernel.state =" in text:
            hits.append(f"{rel} attempts direct kernel.state replacement")

    if rel.startswith(_WORKER_PREFIXES):
        if "voice_pipeline.kernel." in text:
            hits.append(f"{rel} imports kernel authority internals from worker lane")
        for token in _KERNEL_STATE_WRITE_TOKENS:
            if token in text:
                hits.append(f"{rel} references kernel authority state token {token!r}")

    return hits


def audit_single_writers(repo_root: Path) -> AuditResult:
    src_root = Path(repo_root) / "apps" / "api" / "src" / "voice_pipeline"
    violations: list[str] = []

    for path in sorted(src_root.rglob("*.py")):
        rel = path.relative_to(src_root).as_posix()
        text = path.read_text(encoding="utf-8")
        violations.extend(_collect_violations(rel=rel, text=text))

    return AuditResult(ok=not violations, violations=tuple(violations))


def write_audit_artifact(result: AuditResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": bool(result.ok),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "output_path": str(output),
        "violations": list(result.violations),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
