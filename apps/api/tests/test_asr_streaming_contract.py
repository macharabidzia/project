from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from voice_pipeline.stt.asr_engine import ASREngine, ASRRuntimeConfig


def _install_fake_vosk(
    monkeypatch,
    *,
    script: tuple[tuple[bool, str], ...],
    final_result: str = '{"text": ""}',
) -> None:
    class _FakeModel:
        def __init__(self, path: str) -> None:
            self.path = str(path)

    class _FakeRecognizer:
        def __init__(self, _model, _sample_rate: int) -> None:
            self._script = list(script)
            self._current: tuple[bool, str] = (False, '{"partial": ""}')
            self._final_result = str(final_result)

        def SetWords(self, _enabled: bool) -> None:
            return None

        def AcceptWaveform(self, _pcm: bytes) -> bool:
            if self._script:
                self._current = self._script.pop(0)
            else:
                self._current = (False, '{"partial": ""}')
            return bool(self._current[0])

        def Result(self) -> str:
            return str(self._current[1]) if bool(self._current[0]) else '{"text": ""}'

        def PartialResult(self) -> str:
            return str(self._current[1]) if not bool(self._current[0]) else '{"partial": ""}'

        def FinalResult(self) -> str:
            return self._final_result

    fake_module = ModuleType("vosk")
    fake_module.Model = _FakeModel
    fake_module.KaldiRecognizer = _FakeRecognizer
    monkeypatch.setitem(sys.modules, "vosk", fake_module)


def test_asr_streaming_emits_partial_then_final_events(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "vosk-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    _install_fake_vosk(
        monkeypatch,
        script=(
            (False, '{"partial": "hello"}'),
            (True, '{"text": "hello world"}'),
        ),
    )

    engine = ASREngine(
        config=ASRRuntimeConfig(
            model_path=str(model_dir),
            sample_rate=16_000,
            input_sample_rate=16_000,
        )
    )
    engine.warm(strict=True)
    engine.start_session(lineage_id="asr-session:epoch:1")

    frame = b"\x01\x00" * 320
    partial_events = engine.ingest_audio(frame, lineage_id="asr-session:epoch:1")
    final_events = engine.ingest_audio(frame, lineage_id="asr-session:epoch:1")

    assert len(partial_events) == 1
    assert partial_events[0].event_type == "ASRPartialReceived"
    assert partial_events[0].text == "hello"
    assert partial_events[0].lineage_id == "asr-session:epoch:1"
    assert int(partial_events[0].emitted_at_ns) > 0

    assert len(final_events) == 1
    assert final_events[0].event_type == "ASRFinalReceived"
    assert final_events[0].text == "hello world"
    assert final_events[0].lineage_id == "asr-session:epoch:1"
    assert int(final_events[0].emitted_at_ns) > 0


def test_asr_finalize_reuses_partial_when_final_result_is_empty(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "vosk-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    _install_fake_vosk(
        monkeypatch,
        script=((False, '{"partial": "unfinished phrase"}'),),
        final_result='{"text": ""}',
    )

    engine = ASREngine(
        config=ASRRuntimeConfig(
            model_path=str(model_dir),
            sample_rate=16_000,
            input_sample_rate=16_000,
        )
    )
    engine.warm(strict=True)
    engine.start_session(lineage_id="asr-session:epoch:1")

    frame = b"\x01\x00" * 320
    _ = engine.ingest_audio(frame, lineage_id="asr-session:epoch:1")
    final_event = engine.finalize(lineage_id="asr-session:epoch:1")

    assert final_event is not None
    assert final_event.event_type == "ASRFinalReceived"
    assert final_event.text == "unfinished phrase"
    assert final_event.lineage_id == "asr-session:epoch:1"
