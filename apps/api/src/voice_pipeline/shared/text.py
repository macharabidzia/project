from __future__ import annotations


def normalize_text(text: object) -> str:
    return " ".join(str(text).split()).strip()


def normalize_transcript(text: object) -> str:
    return normalize_text(text)


def preview_text(text: object, limit: int = 80) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."