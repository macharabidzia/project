from __future__ import annotations

from voice_pipeline.kernel.kernel_runtime import KernelRuntime
from voice_pipeline.observability.replay_viewer import view_replay
from voice_pipeline.observability.timeline import timeline
from voice_pipeline.observability.tracer import Trace


def test_runtime_observers_do_not_mutate_kernel_truth(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VOICE_PIPELINE_ROOT_DIR", str(tmp_path))
    kernel = KernelRuntime(session_id="observe")
    before = {
        "phase": kernel.state.phase,
        "turn_epoch": kernel.state.turn_index,
        "committed_epoch": kernel.state.committed_turn_index,
        "generation_epoch": kernel.state.generation_index,
    }

    trace = Trace()
    trace.add({"event": "snapshot", "phase": kernel.state.phase})
    stream = timeline([{"event_id": "e1"}])
    replay_rows = view_replay(type("_Log", (), {"events": [{"event_id": "e1"}]})())

    after = {
        "phase": kernel.state.phase,
        "turn_epoch": kernel.state.turn_index,
        "committed_epoch": kernel.state.committed_turn_index,
        "generation_epoch": kernel.state.generation_index,
    }

    assert before == after
    assert len(trace.records) == 1
    assert stream == ({"event_id": "e1"},)
    assert replay_rows == ({"event_id": "e1"},)
