# Latency Metric Artifacts

The latency regression CI workflow reads these files:

- `metrics/latency/before.json`
- `metrics/latency/after.json`

Accepted JSON formats:

1. A single metrics object.
2. A list of metrics objects.
3. An object with a `samples` array of metrics objects.

Each metrics object may either contain metric keys directly or under `runtime_metrics`.

Required keys for threshold checks:

- `asr_to_llm_ms`
- `llm_to_tts_ms`
- `end_to_end_ms`
- `cancel_to_transport_emit_ms`
- `stale_drop_latency_ms`
