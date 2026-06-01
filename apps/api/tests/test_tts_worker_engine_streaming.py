from __future__ import annotations

from types import SimpleNamespace

import torch

from voice_pipeline.gpu.tts_worker.engine import (
    _call_cosyvoice_native_stream,
    _resolve_native_stream_emit_thresholds,
    _streaming_text_fragments,
    _trim_native_prompt_kwargs,
)


def test_resolve_native_stream_emit_thresholds_targets_multiple_mix_windows(monkeypatch) -> None:
    monkeypatch.delenv("VOICE_PIPELINE_TTS_NATIVE_STREAM_MIN_EMIT_TOKENS", raising=False)
    monkeypatch.delenv("VOICE_PIPELINE_TTS_NATIVE_STREAM_MIN_SENTENCE_TOKENS", raising=False)
    monkeypatch.setenv("COSYVOICE_BISTREAM_TEXT_MIX_TOKENS", "8")
    monkeypatch.setenv("VOICE_PIPELINE_TTS_NATIVE_STREAM_EMIT_WINDOWS", "2")

    min_emit_tokens, min_sentence_tokens = _resolve_native_stream_emit_thresholds(
        SimpleNamespace(token_min_hop_len=2)
    )

    assert min_emit_tokens == 16
    assert min_sentence_tokens == 8


def test_streaming_text_fragments_waits_for_sentence_sized_append() -> None:
    fragments = list(
        _streaming_text_fragments(
            iter(["That's", "really", "good.", "Keep", "going"]),
            add_english_lang_tag=False,
            min_emit_tokens=16,
            min_sentence_tokens=8,
        )
    )

    assert fragments == ["That's really good. Keep going"]


def test_streaming_text_fragments_emits_long_buffer_without_boundary() -> None:
    fragments = list(
        _streaming_text_fragments(
            iter(["one two three four", "five six seven eight", "nine ten"]),
            add_english_lang_tag=False,
            min_emit_tokens=8,
            min_sentence_tokens=6,
        )
    )

    assert fragments == ["one two three four five six seven eight", "nine ten"]


def test_streaming_text_fragments_flushes_short_final_reply_as_one_fragment() -> None:
    fragments = list(
        _streaming_text_fragments(
            iter(["hello", "there"]),
            add_english_lang_tag=False,
            min_emit_tokens=4,
            min_sentence_tokens=4,
        )
    )

    assert fragments == ["hello there"]


def test_trim_native_prompt_kwargs_keeps_head_for_generator_stream() -> None:
    prompt_kwargs = {
        "llm_prompt_speech_token": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int32),
        "llm_prompt_speech_token_len": torch.tensor([6], dtype=torch.int32),
        "flow_prompt_speech_token": torch.tensor([[11, 12, 13, 14, 15, 16]], dtype=torch.int32),
        "flow_prompt_speech_token_len": torch.tensor([6], dtype=torch.int32),
        "prompt_speech_feat": torch.arange(24, dtype=torch.float32).reshape(1, 6, 4),
        "prompt_speech_feat_len": torch.tensor([6], dtype=torch.int32),
    }

    trimmed = _trim_native_prompt_kwargs(
        prompt_kwargs,
        max_prompt_tokens=4,
        generator_stream=True,
    )

    assert trimmed["llm_prompt_speech_token"].tolist() == [[1, 2, 3, 4]]
    assert trimmed["flow_prompt_speech_token"].tolist() == [[11, 12, 13, 14]]
    assert trimmed["prompt_speech_feat"].tolist() == prompt_kwargs["prompt_speech_feat"][:, :4, :].tolist()
    assert trimmed["llm_prompt_speech_token_len"].tolist() == [4]
    assert trimmed["flow_prompt_speech_token_len"].tolist() == [4]
    assert trimmed["prompt_speech_feat_len"].tolist() == [4]


def test_call_cosyvoice_native_stream_uses_head_trim_for_generator_prompt(monkeypatch) -> None:
    monkeypatch.setenv("VOICE_PIPELINE_TTS_NATIVE_MAX_GENERATOR_PROMPT_SPEECH_TOKENS", "4")

    captured: dict[str, object] = {}

    class _Frontend:
        def frontend_zero_shot(self, *_args):
            return {
                "text": "unused",
                "text_len": "unused",
                "llm_prompt_speech_token": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int32),
                "llm_prompt_speech_token_len": torch.tensor([6], dtype=torch.int32),
                "flow_prompt_speech_token": torch.tensor([[11, 12, 13, 14, 15, 16]], dtype=torch.int32),
                "flow_prompt_speech_token_len": torch.tensor([6], dtype=torch.int32),
                "prompt_speech_feat": torch.arange(24, dtype=torch.float32).reshape(1, 6, 4),
                "prompt_speech_feat_len": torch.tensor([6], dtype=torch.int32),
            }

        def _extract_text_token(self, _text_stream):
            return (
                torch.tensor([[101, 102]], dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32),
            )

    class _ModelImpl:
        def tts(self, **kwargs):
            captured.update(kwargs)
            return iter(())

    class _Model:
        sample_rate = 24000
        frontend = _Frontend()
        model = _ModelImpl()

    def _token_stream():
        yield "hello"
        yield "there"

    _call_cosyvoice_native_stream(
        _Model(),
        _token_stream(),
        "prompt text",
        "/tmp/prompt.wav",
    )

    assert captured["llm_prompt_speech_token"].tolist() == [[1, 2, 3, 4]]
    assert captured["flow_prompt_speech_token"].tolist() == [[11, 12, 13, 14]]
    assert captured["prompt_speech_feat"].tolist() == torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)[:, :4, :].tolist()


def test_call_cosyvoice_native_stream_generator_prompt_cache_key_isolated_from_string_mode(monkeypatch) -> None:
    monkeypatch.setenv("VOICE_PIPELINE_TTS_NATIVE_MAX_PROMPT_SPEECH_TOKENS", "24")
    monkeypatch.setenv("VOICE_PIPELINE_TTS_NATIVE_MAX_GENERATOR_PROMPT_SPEECH_TOKENS", "24")

    class _Frontend:
        def frontend_zero_shot(self, *_args):
            return {
                "text": "unused",
                "text_len": "unused",
                "llm_prompt_speech_token": torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
                "llm_prompt_speech_token_len": torch.tensor([4], dtype=torch.int32),
                "flow_prompt_speech_token": torch.tensor([[11, 12, 13, 14]], dtype=torch.int32),
                "flow_prompt_speech_token_len": torch.tensor([4], dtype=torch.int32),
                "prompt_speech_feat": torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
                "prompt_speech_feat_len": torch.tensor([4], dtype=torch.int32),
            }

        def _extract_text_token(self, _text_stream):
            return (
                torch.tensor([[101, 102]], dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32),
            )

    class _ModelImpl:
        def tts(self, **_kwargs):
            return iter(())

    class _Model:
        sample_rate = 24000
        frontend = _Frontend()
        model = _ModelImpl()

    class _Session:
        native_prompt_cache_key = ""
        native_prompt_kwargs = {}

    def _token_stream():
        yield "hello"
        yield "there"

    session = _Session()

    _call_cosyvoice_native_stream(
        _Model(),
        "hello there",
        "prompt text",
        "/tmp/prompt.wav",
        session=session,
    )
    assert session.native_prompt_cache_key.startswith("string|")

    _call_cosyvoice_native_stream(
        _Model(),
        _token_stream(),
        "prompt text",
        "/tmp/prompt.wav",
        session=session,
    )
    assert session.native_prompt_cache_key.startswith("generator|")
