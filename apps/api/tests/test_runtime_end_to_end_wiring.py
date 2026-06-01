from __future__ import annotations

import asyncio
from dataclasses import replace
import inspect
from pathlib import Path

from voice_pipeline.kernel.dispatch import DispatchCommand
from voice_pipeline.runtime.bootstrap import bootstrap_runtime
from voice_pipeline.runtime.config import RuntimeConfig
from voice_pipeline.stt.asr_engine import ASREvent
from voice_pipeline.transport.pcm_clock import PCMFrame


class _FakeVLLMStreamer:
    async def stream(self, prompt: str, *, cache_key: str = "", request_id: str = ""):
        _ = (prompt, cache_key, request_id)
        yield "hello"
        yield "world"


class _UnexpectedVLLMStreamer:
    async def stream(self, prompt: str, *, cache_key: str = "", request_id: str = ""):
        _ = (prompt, cache_key, request_id)
        raise AssertionError("authoritative vllm stream should not run when speculative result is promoted")


class _UnexpectedTTSStreamer:
    async def stream(
        self,
        text: str,
        *,
        request_id: str = "",
        epoch_id: str = "",
        prompt_text: str = "",
        prompt_speech_path: str = "",
    ):
        _ = (text, request_id, epoch_id, prompt_text, prompt_speech_path)
        raise AssertionError("authoritative tts stream should not run when speculative pcm is promoted")


class _FakeTTSStreamer:
    def __init__(self, *, sample_rate: int = 24_000) -> None:
        self.sample_rate = int(sample_rate)
        self.chunk = b"\xff\x7f" * int(self.sample_rate * 20 / 1000)

    async def stream(
        self,
        text: str,
        *,
        request_id: str = "",
        epoch_id: str = "",
        prompt_text: str = "",
        prompt_speech_path: str = "",
    ):
        _ = (text, request_id, epoch_id, prompt_text, prompt_speech_path)
        yield self.chunk, int(self.sample_rate), True


class _FakeBurstTTSStreamer:
    def __init__(self, *, sample_rate: int = 48_000) -> None:
        self.sample_rate = int(sample_rate)
        quiet_frame = (50).to_bytes(2, byteorder="little", signed=True) * int(self.sample_rate * 20 / 1000)
        loud_frame = (400).to_bytes(2, byteorder="little", signed=True) * int(self.sample_rate * 20 / 1000)
        self.chunk = quiet_frame * 5 + loud_frame

    async def stream(
        self,
        text: str,
        *,
        request_id: str = "",
        epoch_id: str = "",
        prompt_text: str = "",
        prompt_speech_path: str = "",
    ):
        _ = (text, request_id, epoch_id, prompt_text, prompt_speech_path)
        yield self.chunk, int(self.sample_rate), True


class _FakeWeakThenStrongTTSStreamer:
    def __init__(self, *, sample_rate: int = 48_000) -> None:
        self.sample_rate = int(sample_rate)
        weak_frame = (50).to_bytes(2, byteorder="little", signed=True) * int(self.sample_rate * 20 / 1000)
        strong_frame = (900).to_bytes(2, byteorder="little", signed=True) * int(self.sample_rate * 20 / 1000)
        self.weak_chunk = weak_frame * 5
        self.strong_chunk = strong_frame * 5

    async def stream(
        self,
        text: str,
        *,
        request_id: str = "",
        epoch_id: str = "",
        prompt_text: str = "",
        prompt_speech_path: str = "",
    ):
        _ = (text, request_id, epoch_id, prompt_text, prompt_speech_path)
        yield self.weak_chunk, int(self.sample_rate), False
        yield self.strong_chunk, int(self.sample_rate), True


class _CapturingTTSStreamer:
    def __init__(self, *, sample_rate: int = 24_000) -> None:
        self.sample_rate = int(sample_rate)
        self.chunk = b"\xff\x7f" * int(self.sample_rate * 20 / 1000)
        self.received_fragments: list[str] = []
        self.received_is_generator = False

    async def stream(
        self,
        text,
        *,
        request_id: str = "",
        epoch_id: str = "",
        prompt_text: str = "",
        prompt_speech_path: str = "",
    ):
        _ = (request_id, epoch_id, prompt_text, prompt_speech_path)
        self.received_is_generator = inspect.isgenerator(text)
        if self.received_is_generator:
            self.received_fragments = list(text)
        else:
            self.received_fragments = [str(text)]
        yield self.chunk, int(self.sample_rate), True


def _build_runtime(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap.hardware_admission_check", lambda config: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._bind_cuda_device", lambda device_name: None)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_asr_engine", lambda asr, config, session_id: True)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_vllm_engine", lambda vllm, config: True)
    monkeypatch.setattr("voice_pipeline.runtime.bootstrap._warm_tts_engine", lambda tts, config, session_id: True)
    runtime = bootstrap_runtime(
        session_id="runtime-e2e-wiring",
        config=RuntimeConfig(
            asr_model_path=str(tmp_path),
            vllm_model_path=str(tmp_path),
            vllm_cache_dir=str(tmp_path / "vllm-cache"),
            cosyvoice3_model_path=str(tmp_path),
            cosyvoice3_cache_dir=str(tmp_path / "cosy-cache"),
            tts_first_fragment_min_tokens=2,
        ),
    )
    runtime.worker_status.transport = "READY"
    return runtime


async def _noop_tick_and_stamp_commands():
    return ()


async def _noop_dispatch_commands(*_args, **_kwargs):
    return (), ()


def test_runtime_process_pcm_frame_runs_authority_chain_to_pcm(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.asr.ingest_audio = lambda _pcm, *, lineage_id="": (
        ASREvent(
            event_type="ASRFinalReceived",
            text="hello world",
            lineage_id=str(lineage_id or runtime.kernel.current_lease().epoch_id),
            emitted_at_ns=123,
        ),
    )
    runtime._vllm_streamer = _FakeVLLMStreamer()
    runtime._tts_streamer = _FakeTTSStreamer()

    frames = asyncio.run(runtime.process_pcm_frame(b"\x00\x01" * 960))
    frame_bytes = runtime._output_frame_bytes()

    assert frames
    assert any(len(frame.pcm) == frame_bytes for frame in frames)
    event_types = [record["type"] for record in runtime.event_log.as_records() if "type" in record]
    assert "ASRFinalReceived" in event_types
    assert "VLLMChunkReceived" in event_types
    assert "VLLMCompleted" in event_types
    assert "TTSChunkReceived" in event_types
    assert "TTSCompleted" in event_types
    assert runtime.rings.kernel_stream_ring.depth == 0

    sent: list[tuple[bytes, int]] = []

    async def _send(pcm: bytes, sample_rate: int) -> None:
        sent.append((bytes(pcm), int(sample_rate)))

    asyncio.run(runtime.send_pcm_once(_send))
    assert sent
    assert len(sent[0][0]) == frame_bytes
    assert sent[0][1] == int(runtime.config.output_sample_rate)
    assert any(byte != 0 for byte in sent[0][0])


def test_runtime_trims_weak_leading_frames_when_same_tts_batch_opens_gate(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.asr.ingest_audio = lambda _pcm, *, lineage_id="": (
        ASREvent(
            event_type="ASRFinalReceived",
            text="hello world",
            lineage_id=str(lineage_id or runtime.kernel.current_lease().epoch_id),
            emitted_at_ns=123,
        ),
    )
    runtime._vllm_streamer = _FakeVLLMStreamer()
    runtime._tts_streamer = _FakeBurstTTSStreamer()

    frames = asyncio.run(runtime.process_pcm_frame(b"\x00\x01" * 960))

    assert len(frames) >= 2
    assert runtime._tts_signal_metrics["tts_leading_trimmed_frames"] >= 0
    first_frame = frames[0].pcm
    assert any(frame.pcm != first_frame for frame in frames[1:])


def test_runtime_opens_gate_for_rising_onset_burst(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)

    frame_metrics = [
        (0.0024920394644141197, 0.0079345703125),
        (0.004714091774076223, 0.0167236328125),
        (0.007328914478421211, 0.026275634765625),
        (0.010924122296273708, 0.03375244140625),
        (0.013989079743623734, 0.04638671875),
    ]

    assert runtime._batch_opens_tts_leading_gate(
        gate_open=False,
        frame_metrics=frame_metrics,
        is_final=False,
    )

    # When the batch opens only through the rising-onset fallback, emission
    # should start at the strongest tail frame, not the earliest near-floor
    # frame.
    assert (
        runtime._tts_leading_batch_start_index(
            gate_open=False,
            frame_metrics=frame_metrics,
            is_final=False,
        )
        == len(frame_metrics) - 1
    )


def test_runtime_keeps_short_rising_tail_for_threshold_qualified_first_batch(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)

    frame_metrics = [
        (0.002764859702438116, 0.0108642578125),
        (0.006702034268528223, 0.023956298828125),
        (0.011182175017893314, 0.034698486328125),
        (0.01269116997718811, 0.0438232421875),
        (0.07227134704589844, 0.17657470703125),
    ]

    # The first live native batch already contains a meaningful rising onset.
    # Keeping only the last 20 ms tail turns it into a weak blip and leaves
    # room playout waiting on the later native burst to hear real speech.
    assert runtime._tts_leading_batch_start_index(
        gate_open=False,
        frame_metrics=frame_metrics,
        is_final=False,
    ) == 2


def test_runtime_skips_weak_pre_onset_head_when_short_first_batch_already_has_real_crest(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.config = replace(
        runtime.config,
        tts_leading_silence_rms_threshold=0.02,
        tts_leading_silence_peak_threshold=0.10,
    )

    frame_metrics = [
        (0.015204736962914467, 0.055389404296875),
        (0.12291330844163895, 0.3585205078125),
        (0.03281070664525032, 0.1512451171875),
    ]

    assert runtime._tts_leading_batch_start_index(
        gate_open=False,
        frame_metrics=frame_metrics,
        is_final=False,
    ) == 1


def test_runtime_keeps_one_more_supportive_frame_for_longer_tail_heavy_first_batch(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)

    frame_metrics = [
        (0.00255430000834167, 0.00836181640625),
        (0.004792120307683945, 0.0159912109375),
        (0.007586419582366943, 0.024169921875),
        (0.012040261179208755, 0.03704833984375),
        (0.03629864379763603, 0.115386962890625),
        (0.04293864592909813, 0.146484375),
        (0.06225450336933136, 0.220672607421875),
    ]

    assert runtime._tts_leading_batch_start_index(
        gate_open=False,
        frame_metrics=frame_metrics,
        is_final=False,
    ) == 3


def test_runtime_keeps_supportive_onset_when_strongest_crest_is_near_tail_not_last(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)

    frame_metrics = [
        (0.0026114650536328554, 0.008392333984375),
        (0.004999966360628605, 0.01715087890625),
        (0.008011913858354092, 0.027374267578125),
        (0.012578285299241543, 0.03564453125),
        (0.012127222493290901, 0.042449951171875),
        (0.03484238311648369, 0.07440185546875),
        (0.03426771238446236, 0.08245849609375),
        (0.042350079864263535, 0.1285400390625),
        (0.06392613053321838, 0.17449951171875),
        (0.08252914249897003, 0.2352294921875),
        (0.13110016286373138, 0.34197998046875),
        (0.1373199075460434, 0.35797119140625),
        (0.1290043443441391, 0.332427978515625),
        (0.11840041726827621, 0.28973388671875),
        (0.05244864895939827, 0.130279541015625),
    ]

    assert runtime._tts_leading_batch_start_index(
        gate_open=False,
        frame_metrics=frame_metrics,
        is_final=False,
    ) == 7


def test_runtime_reset_rewarms_tts_runtime_probe_when_stream_probe_disabled(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.config = replace(runtime.config, tts_skip_stream_probe=True)

    tts_rebinds: list[tuple[str, str, str]] = []
    warm_calls: list[str] = []

    runtime.asr.start_session = lambda *, lineage_id="": None
    runtime.tts.start_persistent_session = lambda *, epoch_id="", prompt_text="", prompt_speech_path="": tts_rebinds.append(
        (str(epoch_id), str(prompt_text), str(prompt_speech_path))
    )

    async def _warm_vllm() -> str:
        warm_calls.append("vllm")
        return "ok"

    async def _warm_tts() -> str:
        warm_calls.append("tts")
        return "ok"

    runtime.warm_vllm_runtime_probe = _warm_vllm
    runtime.warm_tts_runtime_probe = _warm_tts

    asyncio.run(runtime.reset_session_state())

    assert "vllm" in warm_calls
    assert "tts" in warm_calls
    assert tts_rebinds


def test_runtime_safe_tts_probe_runs_two_bounded_string_probes(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.config = replace(runtime.config, tts_skip_stream_probe=True)

    rebinds: list[tuple[str, str, str]] = []
    calls: list[tuple[str, str, str]] = []
    yielded_chunks: list[int] = []
    cancelled: list[tuple[str, str]] = []

    runtime.tts.start_persistent_session = lambda *, epoch_id="", prompt_text="", prompt_speech_path="": rebinds.append(
        (str(epoch_id), str(prompt_text), str(prompt_speech_path))
    )
    runtime.tts.cancel = lambda *, request_id="", epoch_id="": cancelled.append((str(request_id), str(epoch_id)))

    async def _stream_pcm(text, *, request_id: str = "", epoch_id: str = ""):
        probe_text = str(text)
        calls.append((probe_text, str(request_id), str(epoch_id)))
        for index in range(1, 9):
            yielded_chunks.append(index)
            yield (bytes([index & 0xFF, (index + 1) & 0xFF]) * 960, 24_000, index >= 8)

    runtime.tts.stream_pcm = _stream_pcm

    result = asyncio.run(runtime.warm_tts_runtime_probe())

    assert result == "safe_probe"
    assert calls == [
        ("hi there", f"{runtime.kernel.session_id}:tts-runtime-warmup:hi_there", runtime.kernel.session_id),
        ("hello", f"{runtime.kernel.session_id}:tts-runtime-warmup:hello", runtime.kernel.session_id),
    ]
    assert yielded_chunks == [1, 2, 1, 2]
    assert cancelled == [
        (f"{runtime.kernel.session_id}:tts-runtime-warmup:hi_there", runtime.kernel.session_id),
        (f"{runtime.kernel.session_id}:tts-runtime-warmup:hello", runtime.kernel.session_id),
    ]
    assert len(rebinds) >= 2


def test_runtime_tts_warmup_returns_already_warm_after_success(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.config = replace(runtime.config, tts_skip_stream_probe=True)

    rebinds: list[tuple[str, str, str]] = []
    calls: list[str] = []

    runtime.tts.start_persistent_session = lambda *, epoch_id="", prompt_text="", prompt_speech_path="": rebinds.append(
        (str(epoch_id), str(prompt_text), str(prompt_speech_path))
    )

    async def _stream_pcm(text, *, request_id: str = "", epoch_id: str = ""):
        _ = (request_id, epoch_id)
        calls.append(str(text))
        for index in range(6):
            yield (bytes([index & 0xFF, (index + 1) & 0xFF]) * 960, 24_000, index >= 5)

    runtime.tts.stream_pcm = _stream_pcm

    first_result = asyncio.run(runtime.warm_tts_runtime_probe())
    second_result = asyncio.run(runtime.warm_tts_runtime_probe())

    assert first_result == "safe_probe"
    assert second_result == "already_warm"
    assert calls == ["hi there", "hello"]
    assert len(rebinds) >= 2


def test_runtime_immediate_close_tts_uses_stream_generator_by_default(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    streamer = _CapturingTTSStreamer()
    runtime._tts_streamer = streamer

    request_id = "tts-immediate-close"
    event_id = "evt-1"
    output_version = 1
    runtime.kernel._state = replace(
        runtime.kernel.state.bind_request_event(request_id, event_id, output_version=output_version),
        phase="playing",
        active_tts_request_id=request_id,
    )

    command = DispatchCommand(
        kind="TTS",
        request_id=request_id,
        payload={
            "text": "hello there",
            "stream_fragment": True,
            "close_stream_immediately": True,
            "output_version": output_version,
            "lineage_id": runtime.kernel.current_lease().epoch_id,
            "epoch_id": runtime.kernel.current_lease().epoch_id,
        },
    )

    async def _run() -> tuple[tuple[PCMFrame, ...], tuple[object, ...]]:
        frames, deferred = await runtime._execute_tts_command(command)
        session = runtime._active_tts_streams.get(request_id)
        if session is not None:
            await session.task
        return frames, deferred

    frames, deferred = asyncio.run(_run())

    assert frames == ()
    assert deferred == ()
    assert request_id not in runtime._active_tts_streams
    assert streamer.received_is_generator
    assert streamer.received_fragments == ["hello there"]


def test_runtime_immediate_close_tts_keeps_generator_path_even_with_removed_oneshot_env(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VOICE_PIPELINE_TTS_NATIVE_ONESHOT_MAX_TOKENS", "3")
    runtime = _build_runtime(monkeypatch, tmp_path)
    streamer = _CapturingTTSStreamer()
    runtime._tts_streamer = streamer

    request_id = "tts-immediate-close-oneshot"
    event_id = "evt-1"
    output_version = 1
    runtime.kernel._state = replace(
        runtime.kernel.state.bind_request_event(request_id, event_id, output_version=output_version),
        phase="playing",
        active_tts_request_id=request_id,
    )

    command = DispatchCommand(
        kind="TTS",
        request_id=request_id,
        payload={
            "text": "hello there",
            "stream_fragment": True,
            "close_stream_immediately": True,
            "output_version": output_version,
            "lineage_id": runtime.kernel.current_lease().epoch_id,
            "epoch_id": runtime.kernel.current_lease().epoch_id,
        },
    )

    async def _run() -> tuple[tuple[PCMFrame, ...], tuple[object, ...]]:
        frames, deferred = await runtime._execute_tts_command(command)
        session = runtime._active_tts_streams.get(request_id)
        if session is not None:
            await session.task
        return frames, deferred

    frames, deferred = asyncio.run(_run())

    assert frames == ()
    assert deferred == ()
    assert request_id not in runtime._active_tts_streams
    assert streamer.received_is_generator
    assert streamer.received_fragments == ["hello there"]


def test_runtime_prewarms_vllm_prefix_cache_for_stable_prefix_when_livekit_owns_turn_commit(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.config = replace(runtime.config, livekit_use_turn_detector=True)

    warmed: list[str] = []
    rendered: list[tuple[str, str]] = []

    runtime.vllm.prewarm_prefix_cache = lambda *keys: warmed.extend(str(key) for key in keys)
    runtime.vllm.render_prompt = lambda *, user_text, stable_session_summary="": rendered.append(
        (str(user_text), str(stable_session_summary))
    ) or "prompt"

    runtime.kernel._state = replace(
        runtime.kernel.state,
        transcript=replace(
            runtime.kernel.state.transcript,
            stable_prefix="hello there",
            committed_text="",
        ),
    )

    async def _run() -> None:
        runtime._maybe_prewarm_vllm_stable_prefix()
        assert runtime._speculative_vllm is not None
        assert runtime._speculative_vllm.task is not None
        await runtime._speculative_vllm.task
        runtime._maybe_prewarm_vllm_stable_prefix()

    asyncio.run(_run())

    assert rendered == [("hello there", "")]
    assert len(warmed) == 1
    assert "hello there" in warmed[0]


def test_runtime_starts_hidden_speculative_vllm_for_changed_stable_prefix(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.config = replace(runtime.config, livekit_use_turn_detector=True)

    warmed: list[str] = []
    rendered: list[tuple[str, str]] = []
    streamed: list[tuple[str, str, str]] = []

    runtime.vllm.prewarm_prefix_cache = lambda *keys: warmed.extend(str(key) for key in keys)
    runtime.vllm.render_prompt = lambda *, user_text, stable_session_summary="": rendered.append(
        (str(user_text), str(stable_session_summary))
    ) or "prompt"

    async def _stream_tokens(prompt: str, *, cache_key: str = "", request_id: str = "", **_kwargs):
        streamed.append((str(prompt), str(cache_key), str(request_id)))
        yield "hello"
        yield "there"

    runtime.vllm.stream_tokens = _stream_tokens
    runtime.kernel._state = replace(
        runtime.kernel.state,
        transcript=replace(
            runtime.kernel.state.transcript,
            stable_prefix="hello there",
            committed_text="",
        ),
    )

    async def _run() -> None:
        runtime._maybe_prewarm_vllm_stable_prefix()
        assert runtime._speculative_vllm is not None
        assert runtime._speculative_vllm.task is not None
        await runtime._speculative_vllm.task

    asyncio.run(_run())

    assert rendered == [("hello there", "")]
    assert len(warmed) == 1
    assert len(streamed) == 1
    assert runtime._speculative_vllm is not None
    assert runtime._speculative_vllm.completed
    assert runtime._speculative_vllm.tokens == ["hello", "there"]
    assert runtime._speculative_vllm.completed_text == "hello there"


def test_runtime_starts_speculative_tts_from_first_flushable_speculative_vllm_text(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)

    started: list[str] = []

    async def _stream_tokens(prompt: str, *, cache_key: str = "", request_id: str = "", **_kwargs):
        _ = (prompt, cache_key, request_id)
        yield "hello"
        yield "there"
        yield "friend"

    runtime.vllm.stream_tokens = _stream_tokens
    monkeypatch.setattr(type(runtime), "_start_speculative_tts_for_text", lambda self, text: started.append(str(text)))

    from voice_pipeline.runtime.bootstrap import _SpeculativeVLLMRequest

    speculative = _SpeculativeVLLMRequest(
        request_id="spec-vllm-1",
        source_text="hello there",
        cache_key="cache",
        rendered_prompt="prompt",
    )

    asyncio.run(runtime._run_speculative_vllm_request(speculative))

    assert started == ["hello"]
    assert speculative.completed_text == "hello there friend"


def test_runtime_uses_partial_text_for_speculative_vllm_when_stable_prefix_is_empty(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.config = replace(runtime.config, livekit_use_turn_detector=True)

    warmed: list[str] = []
    rendered: list[tuple[str, str]] = []

    runtime.vllm.prewarm_prefix_cache = lambda *keys: warmed.extend(str(key) for key in keys)
    runtime.vllm.render_prompt = lambda *, user_text, stable_session_summary="": rendered.append(
        (str(user_text), str(stable_session_summary))
    ) or "prompt"

    runtime.kernel._state = replace(
        runtime.kernel.state,
        transcript=replace(
            runtime.kernel.state.transcript,
            partial_text="hello there",
            stable_prefix="",
            committed_text="",
        ),
    )

    async def _run() -> None:
        runtime._maybe_prewarm_vllm_stable_prefix()
        assert runtime._speculative_vllm is not None
        assert runtime._speculative_vllm.task is not None
        await runtime._speculative_vllm.task

    asyncio.run(_run())

    assert rendered == [("hello there", "")]
    assert len(warmed) == 1
    assert runtime._speculative_vllm is not None
    assert runtime._speculative_vllm.source_text == "hello there"


def test_runtime_uses_single_partial_token_for_speculative_vllm_in_low_latency_mode(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.config = replace(runtime.config, livekit_use_turn_detector=True, speculative_partial_min_tokens=1)

    rendered: list[tuple[str, str]] = []
    runtime.vllm.prewarm_prefix_cache = lambda *keys: None
    runtime.vllm.render_prompt = lambda *, user_text, stable_session_summary="": rendered.append(
        (str(user_text), str(stable_session_summary))
    ) or "prompt"

    runtime.kernel._state = replace(
        runtime.kernel.state,
        transcript=replace(
            runtime.kernel.state.transcript,
            partial_text="hello",
            stable_prefix="",
            committed_text="",
        ),
    )

    async def _run() -> None:
        runtime._maybe_prewarm_vllm_stable_prefix()
        assert runtime._speculative_vllm is not None
        assert runtime._speculative_vllm.task is not None
        await runtime._speculative_vllm.task

    asyncio.run(_run())

    assert rendered == [("hello", "")]
    assert runtime._speculative_vllm is not None
    assert runtime._speculative_vllm.source_text == "hello"


def test_runtime_promotes_matching_speculative_vllm_result_on_confirmed_turn(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime._vllm_streamer = _UnexpectedVLLMStreamer()
    monkeypatch.setattr(type(runtime), "_tick_and_stamp_commands", lambda self: _noop_tick_and_stamp_commands())
    monkeypatch.setattr(type(runtime), "_dispatch_commands", lambda self, *args, **kwargs: _noop_dispatch_commands())

    request_id = "vllm-spec-promote"
    event_id = "evt-vllm-spec"
    output_version = 1
    runtime.kernel._state = replace(
        runtime.kernel.state.bind_request_event(request_id, event_id, output_version=output_version),
        phase="generating",
        active_vllm_request_id=request_id,
    )

    # Build a completed speculative request matching the authoritative prompt.
    from voice_pipeline.runtime.bootstrap import _SpeculativeVLLMRequest

    runtime._speculative_vllm = _SpeculativeVLLMRequest(
        request_id="spec-1",
        source_text="hello there",
        cache_key="cache",
        rendered_prompt="prompt",
        tokens=["hello", "there"],
        completed=True,
        completed_text="hello there",
    )

    command = DispatchCommand(
        kind="VLLM",
        request_id=request_id,
        payload={
            "prompt": "hello there",
            "output_version": output_version,
            "lineage_id": runtime.kernel.current_lease().epoch_id,
            "epoch_id": runtime.kernel.current_lease().epoch_id,
            "kernel_decision_ns": 100,
        },
    )

    asyncio.run(runtime._execute_vllm_command(command))

    event_types = [record["type"] for record in runtime.event_log.as_records() if "type" in record]
    assert event_types.count("VLLMChunkReceived") == 2
    assert "VLLMCompleted" in event_types
    assert runtime._speculative_vllm is None


def test_runtime_clears_stale_speculative_tts_when_stable_prefix_changes(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.config = replace(runtime.config, livekit_use_turn_detector=True)

    cancelled: list[str] = []
    rendered: list[str] = []
    warmed: list[str] = []

    async def _cancel_speculative_tts(self) -> None:
        cancelled.append("tts")
        self._speculative_tts = None

    async def _cancel_speculative_vllm(self) -> None:
        self._speculative_vllm = None

    async def _stream_tokens(prompt: str, *, cache_key: str = "", request_id: str = "", **_kwargs):
        _ = (prompt, cache_key, request_id)
        yield "hello"

    monkeypatch.setattr(type(runtime), "_cancel_speculative_tts", _cancel_speculative_tts)
    monkeypatch.setattr(type(runtime), "_cancel_speculative_vllm", _cancel_speculative_vllm)
    runtime.vllm.prewarm_prefix_cache = lambda *keys: warmed.extend(str(key) for key in keys)
    runtime.vllm.render_prompt = lambda *, user_text, stable_session_summary="": rendered.append(str(user_text)) or "prompt"
    runtime.vllm.stream_tokens = _stream_tokens

    from voice_pipeline.runtime.bootstrap import _SpeculativeTTSRequest

    runtime._speculative_tts = _SpeculativeTTSRequest(
        request_id="spec-tts-1",
        source_text="hello",
    )
    runtime.kernel._state = replace(
        runtime.kernel.state,
        transcript=replace(
            runtime.kernel.state.transcript,
            stable_prefix="hello there",
            committed_text="",
        ),
    )

    async def _run() -> None:
        runtime._maybe_prewarm_vllm_stable_prefix()
        await asyncio.sleep(0)
        if runtime._speculative_vllm is not None and runtime._speculative_vllm.task is not None:
            await runtime._speculative_vllm.task

    asyncio.run(_run())

    assert cancelled == ["tts"]
    assert rendered == ["hello there"]
    assert warmed


def test_runtime_promotes_matching_speculative_tts_result_on_confirmed_turn(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime._tts_streamer = _UnexpectedTTSStreamer()
    monkeypatch.setattr(type(runtime), "_tick_and_stamp_commands", lambda self: _noop_tick_and_stamp_commands())
    monkeypatch.setattr(type(runtime), "_dispatch_commands", lambda self, *args, **kwargs: _noop_dispatch_commands())

    request_id = "tts-spec-promote"
    event_id = "evt-tts-spec"
    output_version = 1
    runtime.kernel._state = replace(
        runtime.kernel.state.bind_request_event(request_id, event_id, output_version=output_version),
        phase="playing",
        active_tts_request_id=request_id,
    )

    from voice_pipeline.runtime.bootstrap import _PreparedTTSFrame, _SpeculativeTTSRequest
    from collections import deque

    frame_pcm = b"\x01\x00" * (runtime._output_frame_bytes() // 2)
    runtime._speculative_tts = _SpeculativeTTSRequest(
        request_id="spec-tts-1",
        source_text="hello there",
        pending_frames=deque(
            [
                _PreparedTTSFrame(
                    pcm=frame_pcm,
                    raw_rms=0.1,
                    raw_peak=0.2,
                    resampled_rms=0.1,
                    resampled_peak=0.2,
                    chunked_rms=0.1,
                    chunked_peak=0.2,
                    chunk_index=1,
                )
            ]
        ),
        completed=True,
    )

    command = DispatchCommand(
        kind="TTS",
        request_id=request_id,
        payload={
            "text": "hello there",
            "output_version": output_version,
            "lineage_id": runtime.kernel.current_lease().epoch_id,
            "epoch_id": runtime.kernel.current_lease().epoch_id,
            "kernel_decision_ns": 100,
        },
    )

    async def _run() -> None:
        frames, deferred = await runtime._execute_tts_command(command)
        assert frames == ()
        assert deferred == ()
        speculative = runtime._speculative_tts
        assert speculative is not None
        assert speculative.drain_task is not None
        await speculative.drain_task

    asyncio.run(_run())

    event_types = [record["type"] for record in runtime.event_log.as_records() if "type" in record]
    assert "TTSChunkReceived" in event_types
    assert "TTSCompleted" in event_types
    assert runtime._speculative_tts is None
    assert runtime.pcm_clock.depth >= 1


def test_runtime_drops_weak_final_resampler_tail_after_long_gap(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)

    assert runtime._should_drop_final_tts_resampler_tail(
        last_emitted_ns=1_000_000_000,
        observed_ns=1_350_000_000,
        raw_pcm=b"",
        raw_rms=0.0,
        raw_peak=0.0,
        resampled_pcm=b"\x01\x00" * 960,
        resampled_rms=0.021,
        resampled_peak=0.054,
        is_final=True,
    )

    assert not runtime._should_drop_final_tts_resampler_tail(
        last_emitted_ns=1_000_000_000,
        observed_ns=1_050_000_000,
        raw_pcm=b"",
        raw_rms=0.0,
        raw_peak=0.0,
        resampled_pcm=b"\x01\x00" * 960,
        resampled_rms=0.021,
        resampled_peak=0.054,
        is_final=True,
    )

    assert not runtime._should_drop_final_tts_resampler_tail(
        last_emitted_ns=1_000_000_000,
        observed_ns=1_350_000_000,
        raw_pcm=b"\x02\x00" * 960,
        raw_rms=0.05,
        raw_peak=0.12,
        resampled_pcm=b"\x01\x00" * 960,
        resampled_rms=0.021,
        resampled_peak=0.054,
        is_final=True,
    )


def test_chunk_output_pcm_can_drop_carry_on_final_flush(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    epoch_id = "runtime-e2e-wiring:epoch:0"
    output_version = 1
    frame_bytes = runtime._output_frame_bytes()

    frames = runtime._chunk_output_pcm(
        b"\x01\x00" * (frame_bytes // 4),
        epoch_id=epoch_id,
        output_version=output_version,
        flush=False,
    )
    assert frames == ()

    dropped = runtime._chunk_output_pcm(
        b"",
        epoch_id=epoch_id,
        output_version=output_version,
        flush=True,
        drop_carry_on_flush=True,
    )
    assert dropped == ()
    assert runtime._tts_frame_carry == b""


def test_barge_in_rotates_epoch_and_suppresses_stale_pcm(monkeypatch, tmp_path: Path) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    initial_epoch = runtime.kernel.current_lease().epoch_id
    initial_output_version = int(runtime.kernel.state.output.version)
    runtime.pcm_clock.enqueue(
        PCMFrame(
            pcm=b"\x01\x02" * (runtime._output_frame_bytes() // 2),
            sample_rate=int(runtime.config.output_sample_rate),
            epoch_id=initial_epoch,
            output_version=initial_output_version,
        )
    )
    runtime.kernel._state = replace(runtime.kernel.state, phase="playing", active_tts_request_id="tts-active")
    runtime.asr.ingest_audio = lambda _pcm, *, lineage_id="": (
        ASREvent(
            event_type="ASRPartialReceived",
            text="barge now",
            lineage_id=str(lineage_id or runtime.kernel.current_lease().epoch_id),
            emitted_at_ns=456,
        ),
    )

    _ = asyncio.run(runtime.process_pcm_frame(b"\x00\x01" * 960))
    assert runtime.kernel.state.output.version > initial_output_version
    assert runtime.kernel.current_lease().epoch_id != initial_epoch
    assert runtime.rings.kernel_stream_ring.depth == 0

    sent: list[tuple[bytes, int]] = []

    async def _send(pcm: bytes, sample_rate: int) -> None:
        sent.append((bytes(pcm), int(sample_rate)))

    asyncio.run(runtime.send_pcm_once(_send))
    assert sent
    assert sent[0][0] == (b"\x00" * runtime._output_frame_bytes())
    assert runtime.pcm_clock.depth == 0
