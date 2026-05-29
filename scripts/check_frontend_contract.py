from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "apps" / "web" / "src" / "App.tsx"
API_PATH = REPO_ROOT / "apps" / "web" / "src" / "lib" / "api.ts"


def _expect_contains(violations: list[str], *, text: str, token: str, label: str) -> None:
    if token not in text:
        violations.append(f"missing required frontend contract token {token!r} in {label}")


def _expect_not_contains(violations: list[str], *, text: str, token: str, label: str) -> None:
    if token in text:
        violations.append(f"forbidden frontend drift token {token!r} present in {label}")


def main() -> int:
    violations: list[str] = []

    app_text = APP_PATH.read_text(encoding="utf-8")
    api_text = API_PATH.read_text(encoding="utf-8")

    _expect_contains(violations, text=app_text, token="livekit-client", label=str(APP_PATH))
    _expect_contains(violations, text=app_text, token="RoomEvent.TrackSubscribed", label=str(APP_PATH))
    _expect_contains(violations, text=app_text, token="setMicrophoneEnabled(true)", label=str(APP_PATH))
    _expect_not_contains(violations, text=app_text, token="AudioWorkletNode", label=str(APP_PATH))
    _expect_not_contains(violations, text=app_text, token="WebSocket(", label=str(APP_PATH))
    _expect_not_contains(violations, text=app_text, token="encodePcmFrame", label=str(APP_PATH))
    _expect_not_contains(violations, text=app_text, token="decodePcmFrame", label=str(APP_PATH))

    _expect_contains(violations, text=api_text, token="/v1/livekit/token", label=str(API_PATH))
    _expect_not_contains(violations, text=api_text, token="/v1/voice/ws", label=str(API_PATH))

    if violations:
        for violation in violations:
            print(violation)
        raise SystemExit(1)

    print("frontend contract guard: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
