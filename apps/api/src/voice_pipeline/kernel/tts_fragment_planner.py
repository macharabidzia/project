from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


_PUNCTUATION_BOUNDARY = (".", "!", "?", ",", ";", ":")
_PUNCTUATION_ONLY = frozenset(_PUNCTUATION_BOUNDARY)
_CONTRACTION_SUFFIXES = frozenset({"'s", "'re", "'m", "'ve", "'ll", "'d", "n't"})
_DANGLING_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "but",
        "can",
        "could",
        "for",
        "he",
        "her",
        "him",
        "i",
        "if",
        "is",
        "it",
        "me",
        "my",
        "of",
        "or",
        "our",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "those",
        "to",
        "us",
        "we",
        "will",
        "would",
        "you",
        "your",
    }
)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _normalize_tokens(buffered_tokens: Sequence[str]) -> tuple[str, ...]:
    return tuple(token for token in (_normalize_text(item) for item in buffered_tokens) if token)


def _token_has_boundary(token: str) -> bool:
    return token.endswith(_PUNCTUATION_BOUNDARY)


def _token_is_punctuation_only(token: str) -> bool:
    return token.strip() in _PUNCTUATION_ONLY


def _token_is_contraction_suffix(token: str) -> bool:
    return token.strip().lower() in _CONTRACTION_SUFFIXES


def _token_is_dangling_function_word(token: str) -> bool:
    return token.strip().lower() in _DANGLING_FUNCTION_WORDS


def _has_internal_boundary(tokens: Sequence[str]) -> bool:
    if len(tokens) < 2:
        return False
    return any(_token_has_boundary(token) for token in tokens[:-1])


def _lexical_token_count(tokens: Sequence[str]) -> int:
    return sum(1 for token in tokens if any(ch.isalnum() for ch in str(token or "")))


def _is_boundaryless_lexical_prefix(tokens: Sequence[str]) -> bool:
    return bool(tokens) and not any(_token_has_boundary(token) for token in tokens)


def _boundary_count(tokens: Sequence[str]) -> int:
    return sum(1 for token in tokens if _token_has_boundary(token))


def _likely_boundaryless_continuation(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    if len(tokens) == 1:
        return True
    normalized = [str(token or "").strip().lower() for token in tokens]
    tail = normalized[-1]
    head = normalized[0]
    if len(tokens) >= 3:
        return True
    if tail.endswith("ing"):
        return True
    if "'" in head or "’" in head:
        return True
    if _token_is_dangling_function_word(tail):
        return True
    return False


@dataclass(frozen=True, slots=True)
class TTSFragmentPlannerConfig:
    min_tokens: int = 2
    max_tokens: int = 6
    context_window_tokens: int = 24
    start_on_stable_prefix: bool = True
    allow_early_low_latency: bool = True


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
    effective_min_tokens = max(1, int(resolved_config.min_tokens))
    max_tokens = max(effective_min_tokens, int(resolved_config.max_tokens))
    context_window = max(1, int(resolved_config.context_window_tokens))
    flush_limit = min(max_tokens, context_window)
    last_boundary_index = max(
        (index for index, token in enumerate(tokens) if _token_has_boundary(token)),
        default=-1,
    )
    boundary_count = _boundary_count(tokens)

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

    if len(tokens) == 1 and effective_min_tokens == 1:
        return TTSFragmentPlan(
            flush_tokens=tokens,
            remaining_tokens=(),
            start_now=True,
            stream_fragment=True,
            reason="low_latency",
        )

    if not bool(resolved_config.allow_early_low_latency):
        if _lexical_token_count(tokens) <= 0:
            return TTSFragmentPlan()
        if len(tokens) <= flush_limit and last_boundary_index < 0:
            if len(tokens) >= flush_limit:
                return TTSFragmentPlan(
                    flush_tokens=tokens,
                    remaining_tokens=(),
                    start_now=True,
                    stream_fragment=True,
                    reason="window_overlap",
                )
            return TTSFragmentPlan()
        if len(tokens) <= flush_limit and last_boundary_index >= 0:
            flush_tokens = tokens[: last_boundary_index + 1]
            if _lexical_token_count(flush_tokens) <= 0:
                if len(tokens) >= flush_limit and _lexical_token_count(tokens) > 0:
                    return TTSFragmentPlan(
                        flush_tokens=tokens,
                        remaining_tokens=(),
                        start_now=True,
                        stream_fragment=True,
                        reason="window_overlap",
                    )
                return TTSFragmentPlan()
            return TTSFragmentPlan(
                flush_tokens=flush_tokens,
                remaining_tokens=tokens[last_boundary_index + 1 :],
                start_now=True,
                stream_fragment=True,
                reason="boundary" if last_boundary_index == len(tokens) - 1 else "window_overlap",
            )
        return TTSFragmentPlan(
            flush_tokens=tokens,
            remaining_tokens=(),
            start_now=True,
            stream_fragment=True,
            reason="window_overlap",
        )

    if _has_internal_boundary(tokens) and len(tokens) < flush_limit:
        if last_boundary_index == len(tokens) - 1:
            return TTSFragmentPlan()
        candidate = tokens[: last_boundary_index + 1] if last_boundary_index >= 0 else ()
        if _lexical_token_count(candidate) < max(2, effective_min_tokens):
            return TTSFragmentPlan()

    if (
        not stable_prefix_present
        and _is_boundaryless_lexical_prefix(tokens)
        and len(tokens) < flush_limit
        and _likely_boundaryless_continuation(tokens)
    ):
        return TTSFragmentPlan()

    if len(tokens) == flush_limit and _has_internal_boundary(tokens):
        if _token_is_contraction_suffix(tokens[-1]) or _token_is_dangling_function_word(tokens[-1]):
            return TTSFragmentPlan()

    if last_boundary_index >= 0 and last_boundary_index < len(tokens) - 1:
        flush_tokens = tokens[: last_boundary_index + 1]
        if _lexical_token_count(flush_tokens) >= max(2, effective_min_tokens):
            return TTSFragmentPlan(
                flush_tokens=flush_tokens,
                remaining_tokens=tokens[last_boundary_index + 1 :],
                start_now=True,
                stream_fragment=True,
                reason="window_overlap",
            )

    flush_count = min(len(tokens), flush_limit)
    stream_fragment = False
    reason = "low_latency"

    if len(tokens) > flush_limit:
        if (
            _has_internal_boundary(tokens)
            and _token_is_contraction_suffix(tokens[flush_limit - 1])
            and not _token_has_boundary(tokens[flush_limit])
        ):
            flush_count = min(len(tokens), flush_limit + 1)
            stream_fragment = False
            reason = "boundary"
        elif (
            _has_internal_boundary(tokens)
            and _token_is_dangling_function_word(tokens[flush_limit - 1])
            and not _token_has_boundary(tokens[flush_limit])
        ):
            flush_count = min(len(tokens), flush_limit + 1)
            stream_fragment = True
            reason = "boundary"
        else:
            if last_boundary_index < 0:
                flush_count = flush_limit
                stream_fragment = True
                reason = "window_overlap"
            else:
                flush_count = len(tokens)
                stream_fragment = False
                reason = "boundary"
    elif last_boundary_index == len(tokens) - 1:
        if _lexical_token_count(tokens) < max(2, effective_min_tokens):
            return TTSFragmentPlan()
        if _has_internal_boundary(tokens) and len(tokens) == flush_limit and boundary_count >= 3:
            stream_fragment = True
            reason = "boundary"
        else:
            stream_fragment = False
            reason = "low_latency"
    else:
        stream_fragment = _likely_boundaryless_continuation(tokens)

    if len(tokens) > flush_count and _token_is_punctuation_only(tokens[flush_count]):
        flush_count += 1
    flush_count = max(1, min(flush_count, len(tokens)))

    flush_tokens = tokens[:flush_count]
    remaining_tokens = tokens[flush_count:]
    if remaining_tokens:
        stream_fragment = True
        if reason != "window_overlap":
            reason = "window_overlap"

    return TTSFragmentPlan(
        flush_tokens=flush_tokens,
        remaining_tokens=remaining_tokens,
        start_now=True,
        stream_fragment=stream_fragment,
        reason=reason,
    )


__all__ = ["TTSFragmentPlan", "TTSFragmentPlannerConfig", "plan_tts_fragment"]
