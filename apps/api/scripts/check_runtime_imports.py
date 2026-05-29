from __future__ import annotations

from pathlib import Path

from voice_pipeline.runtime_registry import assert_valid_runtime_import, module_runtime_owner


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "voice_pipeline"


def _imports(path: Path) -> list[str]:
    targets: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("from "):
            targets.append(line.split(" import ", 1)[0].replace("from ", "", 1).strip())
        elif line.startswith("import "):
            payload = line.replace("import ", "", 1)
            for item in payload.split(","):
                targets.append(item.strip().split(" as ", 1)[0].strip())
    return targets


def main() -> int:
    for path in sorted(SRC_ROOT.rglob("*.py")):
        module = "voice_pipeline." + path.relative_to(SRC_ROOT).with_suffix("").as_posix().replace("/", ".")
        if module_runtime_owner(module) == "shared":
            continue
        assert_valid_runtime_import(module, _imports(path))
    print("runtime import guard: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
