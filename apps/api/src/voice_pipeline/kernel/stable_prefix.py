from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _token_prefix(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    limit = min(len(left), len(right))
    prefix: list[str] = []
    for index in range(limit):
        if left[index] != right[index]:
            break
        prefix.append(left[index])
    return tuple(prefix)


@dataclass(frozen=True, slots=True)
class StablePrefixDecision:
    prefix: str = ""
    confirmations: int = 0


def detect_stable_prefix(
    partials: Sequence[str],
    *,
    min_repeats: int = 2,
    min_tokens: int = 2,
    max_window: int = 3,
    max_relative_drift: float = 0.34,
) -> StablePrefixDecision:
    normalized = [_normalize_text(item) for item in partials if _normalize_text(item)]
    if len(normalized) < max(1, int(min_repeats)):
        return StablePrefixDecision()

    window = normalized[-max(1, int(max_window)) :]
    common_tokens = tuple(window[0].split())
    for candidate in window[1:]:
        common_tokens = _token_prefix(common_tokens, tuple(candidate.split()))
        if not common_tokens:
            return StablePrefixDecision()

    if len(common_tokens) < max(1, int(min_tokens)):
        return StablePrefixDecision()

    stable_prefix = " ".join(common_tokens)
    confirmations = sum(1 for item in window if item.startswith(stable_prefix))
    if confirmations < max(1, int(min_repeats)):
        return StablePrefixDecision()

    avg_token_len = sum(len(item.split()) for item in window) / float(len(window))
    drift = (avg_token_len - float(len(common_tokens))) / max(1.0, avg_token_len)
    if drift > float(max_relative_drift) and confirmations < len(window):
        return StablePrefixDecision()

    return StablePrefixDecision(prefix=stable_prefix, confirmations=int(confirmations))


__all__ = ["StablePrefixDecision", "detect_stable_prefix"]
