from __future__ import annotations

from voice_pipeline.observability.metrics import summarize_latency


def test_latency_summary_reports_p50_p95_p99_and_worst() -> None:
    summary = summarize_latency([10.0, 20.0, 30.0, 40.0, 50.0])

    assert summary.p50_ms >= 0.0
    assert summary.p95_ms >= summary.p50_ms
    assert summary.p99_ms >= summary.p95_ms
    assert summary.worst_ms >= summary.p99_ms
