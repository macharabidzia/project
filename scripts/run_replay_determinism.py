from __future__ import annotations

import pytest


def main() -> int:
    return int(
        pytest.main(
            [
                "-q",
                "apps/api/tests/test_streamspine_collapse_determinism.py",
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
