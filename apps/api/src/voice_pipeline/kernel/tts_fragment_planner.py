from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


_PUNCTUATION_BOUNDARY = (".", "!", "?", ",", ";", ":")


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _normalize_tokens(buffered_tokens: Sequence[str]) -> tuple[str, ...]:
    return tuple(token for token in (_normalize_text(item) for item in buffered_tokens) if token)


def _token_has_boundary(token: str) -> bool:
    return token.endswith(_PUNCTUATION_BOUNDARY)


@dataclass(frozen=True, slots=True)
class TTSFragmentPlannerConfig:
    min_tokens: int = 2
    max_tokens: int = 6
    context_window_tokens: int = 24
    start_on_stable_prefix: bool = True


@dataclass(frozen=True, slots=True)
class TTSFragmentPlan:
    flush_tokens: tuple[str, ...] = ()
    remaining_tokens: tuple[str, ...] = ()
    start_now: bool = False
    stream_fragment: bool = True
    reason: str = ""

    @property
    def flush_text(self) -> str:
        return " ".join(self.flush_tokens)


def plan_tts_fragment(
    buffered_tokens: Sequence[str],
    *,
    stable_prefix: str = "",
    drain: bool = False,
    config: TTSFragmentPlannerConfig | None = None,
) -> TTSFragmentPlan:
    tokens = _normalize_tokens(buffered_tokens)
    if not tokens:
        return TTSFragmentPlan()

    resolved_config = config or TTSFragmentPlannerConfig()
    stable_prefix_present = bool(_normalize_text(stable_prefix))
    effective_min_tokens = 1 if stable_prefix_present and resolved_config.start_on_stable_prefix else max(1, int(resolved_config.min_tokens))
    max_tokens = max(effective_min_tokens, int(resolved_config.max_tokens))
    context_window = max(1, int(resolved_config.context_window_tokens))
    flush_limit = min(max_tokens, context_window)

    if drain:
        return TTSFragmentPlan(
            flush_tokens=tokens,
            remaining_tokens=(),
            start_now=True,
            stream_fragment=False,
            reason="drain",
        )

    if len(tokens) < effective_min_tokens:
        return TTSFragmentPlan()

    flush_count = min(len(tokens), flush_limit)
    boundary_index = next((index for index, token in enumerate(tokens[:flush_count]) if _token_has_boundary(token)), -1)
    if boundary_index >= 0:
        flush_count = boundary_index + 1

    if stable_prefix_present and len(tokens) == 1:
        flush_count = 1

    flush_count = max(1, min(flush_count, len(tokens)))
    flush_tokens = tokens[:flush_count]
    remaining_tokens = tokens[flush_count:]
    if not flush_tokens:
        return TTSFragmentPlan()

    if remaining_tokens:
        reason = "window_overlap"
    elif boundary_index >= 0:
        reason = "boundary"
    elif stable_prefix_present:
        reason = "stable_prefix"
    else:
        reason = "min_tokens"

    return TTSFragmentPlan(
        flush_tokens=flush_tokens,
        remaining_tokens=remaining_tokens,
        start_now=True,
        stream_fragment=bool(remaining_tokens),
        reason=reason,
    )


__all__ = ["TTSFragmentPlan", "TTSFragmentPlannerConfig", "plan_tts_fragment"]
