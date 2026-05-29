from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import date
import json
from pathlib import Path
from statistics import quantiles
from typing import Any


HARD_LIMITS_MS: dict[str, dict[str, float]] = {
    "asr_to_dispatch_ms": {"p50": 40.0, "p95": 70.0},
    "dispatch_to_first_token_ms": {"p50": 40.0, "p95": 70.0},
    "first_token_to_first_pcm_ms": {"p50": 30.0, "p95": 60.0},
    "first_audible_frame_ms": {"p50": 150.0, "p95": 220.0},
    "pcm_jitter_ms": {"p50": 2.0, "p95": 5.0},
}

WARNING_LIMITS_MS: dict[str, dict[str, float]] = {
    "interrupt_to_new_speech_ms": {"p95": 140.0},
}

COUNTER_LIMITS: dict[str, float] = {
    "stale_token_drop_count": 0.0,
    "stale_pcm_drop_count": 0.0,
}


def _as_samples(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        rows = payload["samples"]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        rows = []

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        runtime_metrics = row.get("runtime_metrics")
        if isinstance(runtime_metrics, dict):
            normalized.append(runtime_metrics)
            continue
        normalized.append(row)
    return normalized


def _load_samples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = _as_samples(payload)
    if not samples:
        raise ValueError(f"no metric samples found in {path}")
    return samples


def _collect_metric(samples: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        raw = sample.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        values.append(value)
    return values


def _percentiles(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    p50 = ordered[len(ordered) // 2]
    if len(ordered) < 20:
        p95 = ordered[-1]
    else:
        p95 = quantiles(ordered, n=100)[94]
    return float(p50), float(p95)


def _summarize(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    tracked_metrics = list(HARD_LIMITS_MS) + list(WARNING_LIMITS_MS) + list(COUNTER_LIMITS)
    for metric_name in tracked_metrics:
        values = _collect_metric(samples, metric_name)
        p50, p95 = _percentiles(values)
        worst = max(values) if values else 0.0
        summary[metric_name] = {
            "count": float(len(values)),
            "p50": p50,
            "p95": p95,
            "worst": float(worst),
        }
    return summary


def _evaluate(summary: dict[str, dict[str, float]]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    for metric, limits in HARD_LIMITS_MS.items():
        for percentile, limit in limits.items():
            observed = summary[metric][percentile]
            if observed > float(limit):
                failures.append(
                    f"{metric} {percentile} {observed:.2f} ms exceeds hard limit {float(limit):.2f} ms"
                )

    for metric, limits in WARNING_LIMITS_MS.items():
        for percentile, limit in limits.items():
            observed = summary[metric][percentile]
            if observed > float(limit):
                warnings.append(
                    f"{metric} {percentile} {observed:.2f} ms exceeds warning limit {float(limit):.2f} ms"
                )

    for metric, limit in COUNTER_LIMITS.items():
        observed = summary[metric]["worst"]
        if observed > float(limit):
            failures.append(f"{metric} observed {observed:.2f} exceeds hard counter limit {float(limit):.2f}")

    return failures, warnings


def _fmt(label: str, metric: str, summary: dict[str, dict[str, float]]) -> str:
    if metric not in summary:
        return f"- {label}: n/a"
    row = summary[metric]
    return f"- {label}: p50={row['p50']:.2f} ms, p95={row['p95']:.2f} ms, worst={row['worst']:.2f} ms"


def _write_report(
    output_path: Path,
    *,
    baseline: dict[str, dict[str, float]] | None,
    candidate: dict[str, dict[str, float]],
    failures: list[str],
    warnings: list[str],
) -> None:
    lines: list[str] = [
        "# Ultra-Low-Latency Regression Report",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## Baseline",
    ]
    if baseline is None:
        lines.append("- no baseline provided")
    else:
        lines.extend(
            [
                _fmt("ASR -> LLM", "asr_to_llm_ms", baseline),
                _fmt("LLM -> TTS", "llm_to_tts_ms", baseline),
                _fmt("End-to-end", "end_to_end_ms", baseline),
                _fmt("Cancel -> transport emit", "cancel_to_transport_emit_ms", baseline),
                _fmt("Stale-output drop", "stale_drop_latency_ms", baseline),
            ]
        )

    lines.extend(
        [
            _fmt("ASR -> dispatch", "asr_to_dispatch_ms", baseline or {}),
            _fmt("Dispatch -> first token", "dispatch_to_first_token_ms", baseline or {}),
            _fmt("First token -> first PCM", "first_token_to_first_pcm_ms", baseline or {}),
            _fmt("First audible frame", "first_audible_frame_ms", baseline or {}),
            _fmt("PCM jitter", "pcm_jitter_ms", baseline or {}),
            _fmt("Interrupt -> new speech", "interrupt_to_new_speech_ms", baseline or {}),
            _fmt("Stale token drops", "stale_token_drop_count", baseline or {}),
            _fmt("Stale PCM drops", "stale_pcm_drop_count", baseline or {}),
            "",
            "## Candidate",
            _fmt("ASR -> dispatch", "asr_to_dispatch_ms", candidate),
            _fmt("Dispatch -> first token", "dispatch_to_first_token_ms", candidate),
            _fmt("First token -> first PCM", "first_token_to_first_pcm_ms", candidate),
            _fmt("First audible frame", "first_audible_frame_ms", candidate),
            _fmt("PCM jitter", "pcm_jitter_ms", candidate),
            _fmt("Interrupt -> new speech", "interrupt_to_new_speech_ms", candidate),
            _fmt("Stale token drops", "stale_token_drop_count", candidate),
            _fmt("Stale PCM drops", "stale_pcm_drop_count", candidate),
            "",
            "## Threshold Evaluation",
        ]
    )

    if not failures and not warnings:
        lines.append("- all thresholds satisfied")
    else:
        for message in failures:
            lines.append(f"- FAIL: {message}")
        for message in warnings:
            lines.append(f"- WARN: {message}")

    lines.extend(
        [
            "",
            "## Limits",
            "- Hard: ASR->dispatch p50<=40/p95<=70, dispatch->first-token p50<=40/p95<=70",
            "- Hard: first-token->first-PCM p50<=30/p95<=60, first-audible-frame p50<=150/p95<=220",
            "- Hard: PCM jitter p50<=2/p95<=5, stale drop counters must remain at 0",
            "- Warning: interrupt->new-speech p95 <= 140 ms",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KernelRuntime latency metrics and emit a report artifact.")
    parser.add_argument("--candidate", required=True, help="Path to candidate run metrics JSON.")
    parser.add_argument("--baseline", help="Path to baseline metrics JSON.")
    parser.add_argument(
        "--report",
        default="docs/v2/phase2-latency-report.md",
        help="Output report path (default: docs/v2/phase2-latency-report.md).",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return non-zero when warning thresholds are exceeded.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidate_path = Path(args.candidate)
    baseline_path = Path(args.baseline) if args.baseline else None
    report_path = Path(args.report)

    candidate_samples = _load_samples(candidate_path)
    candidate_summary = _summarize(candidate_samples)

    baseline_summary: dict[str, dict[str, float]] | None = None
    if baseline_path is not None:
        baseline_summary = _summarize(_load_samples(baseline_path))

    failures, warnings = _evaluate(candidate_summary)
    _write_report(
        report_path,
        baseline=baseline_summary,
        candidate=candidate_summary,
        failures=failures,
        warnings=warnings,
    )

    for message in failures:
        print(f"FAIL: {message}")
    for message in warnings:
        print(f"WARN: {message}")

    if failures:
        return 1
    if warnings and bool(args.fail_on_warning):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
