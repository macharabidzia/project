from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = REPO_ROOT / "scripts" / "bootstrap_runpod_backend.sh"


def _expect_contains(violations: list[str], *, text: str, token: str) -> None:
    if token not in text:
        violations.append(f"missing required backend-install token {token!r}")


def main() -> int:
    text = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    violations: list[str] = []
    required_tokens = (
        'API_EXTRAS="${API_EXTRAS:+${API_EXTRAS},}asr"',
        'WORKER_EXTRAS="asr,gpu,response-model"',
        'WORKER_EXTRAS="dev,asr,gpu,response-model"',
        'verify_asr_runtime "${ACTIVE_WORKER_PYTHON}"',
        'install_cosyvoice_runtime "${ACTIVE_WORKER_PYTHON}"',
    )
    for token in required_tokens:
        _expect_contains(violations, text=text, token=token)

    if violations:
        for item in violations:
            print(item)
        raise SystemExit(1)

    print("backend-install contract guard: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
