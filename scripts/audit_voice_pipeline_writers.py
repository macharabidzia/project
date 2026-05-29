#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "apps" / "api" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice_pipeline.governance.single_writer_audit import audit_single_writers, write_audit_artifact


def main() -> int:
    result = audit_single_writers(REPO_ROOT)
    output = REPO_ROOT / "docs" / "authority-writer-audit.json"
    write_audit_artifact(result, output)
    if result.ok:
        print(f"single-writer audit: OK ({output})")
        return 0
    print(f"single-writer audit: FAILED ({output})")
    for item in result.violations:
        print(f"- {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
