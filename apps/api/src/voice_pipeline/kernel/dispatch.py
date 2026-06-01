from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping


CommandKind = Literal["VLLM", "TTS", "TTS_APPEND", "TRANSPORT", "VLLM_CANCEL", "TTS_CANCEL"]


@dataclass(frozen=True, slots=True)
class DispatchCommand:
    kind: CommandKind
    request_id: str
    payload: Mapping[str, object] = field(default_factory=dict)


def build_vllm_command(
    *,
    session_id: str,
    request_id: str,
    prompt: str,
    prompt_cache_key: str,
    output_version: int,
    lineage_id: str,
    turn_id: str,
    epoch_id: str,
) -> DispatchCommand:
    return DispatchCommand(
        kind="VLLM",
        request_id=str(request_id),
        payload={
            "session_id": str(session_id),
            "prompt": str(prompt),
            "prompt_cache_key": str(prompt_cache_key),
            "output_version": int(output_version),
            "lineage_id": str(lineage_id),
            "turn_id": str(turn_id),
            "epoch_id": str(epoch_id),
        },
    )


def build_tts_command(
    *,
    session_id: str,
    request_id: str,
    text: str,
    output_version: int,
    lineage_id: str,
    turn_id: str,
    epoch_id: str,
    stream_fragment: bool,
    close_stream_immediately: bool = False,
) -> DispatchCommand:
    return DispatchCommand(
        kind="TTS",
        request_id=str(request_id),
        payload={
            "session_id": str(session_id),
            "text": str(text),
            "output_version": int(output_version),
            "lineage_id": str(lineage_id),
            "turn_id": str(turn_id),
            "epoch_id": str(epoch_id),
            "stream_fragment": bool(stream_fragment),
            "close_stream_immediately": bool(close_stream_immediately),
        },
    )


def build_tts_append_command(
    *,
    session_id: str,
    request_id: str,
    text: str,
    output_version: int,
    lineage_id: str,
    turn_id: str,
    epoch_id: str,
    final_fragment: bool,
) -> DispatchCommand:
    return DispatchCommand(
        kind="TTS_APPEND",
        request_id=str(request_id),
        payload={
            "session_id": str(session_id),
            "text": str(text),
            "output_version": int(output_version),
            "lineage_id": str(lineage_id),
            "turn_id": str(turn_id),
            "epoch_id": str(epoch_id),
            "final_fragment": bool(final_fragment),
        },
    )


def build_vllm_cancel_command(
    *,
    session_id: str,
    request_id: str,
    output_version: int,
    lineage_id: str,
    epoch_id: str,
) -> DispatchCommand:
    return DispatchCommand(
        kind="VLLM_CANCEL",
        request_id=str(request_id),
        payload={
            "session_id": str(session_id),
            "request_id": str(request_id),
            "output_version": int(output_version),
            "lineage_id": str(lineage_id),
            "epoch_id": str(epoch_id),
        },
    )


def build_tts_cancel_command(
    *,
    session_id: str,
    request_id: str,
    output_version: int,
    lineage_id: str,
    epoch_id: str,
) -> DispatchCommand:
    return DispatchCommand(
        kind="TTS_CANCEL",
        request_id=str(request_id),
        payload={
            "session_id": str(session_id),
            "request_id": str(request_id),
            "output_version": int(output_version),
            "lineage_id": str(lineage_id),
            "epoch_id": str(epoch_id),
        },
    )


__all__ = [
    "CommandKind",
    "DispatchCommand",
    "build_tts_append_command",
    "build_tts_cancel_command",
    "build_tts_command",
    "build_vllm_cancel_command",
    "build_vllm_command",
]
