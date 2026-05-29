from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LaneConfig:
    name: str
    ring_size: int
    device: str


@dataclass(frozen=True, slots=True)
class RuntimeTopology:
    asr: LaneConfig = LaneConfig(name="asr", ring_size=1024, device="cpu")
    vllm: LaneConfig = LaneConfig(name="vllm", ring_size=1024, device="cuda:0")
    tts: LaneConfig = LaneConfig(name="tts", ring_size=1024, device="cuda:1")
    pcm: LaneConfig = LaneConfig(name="pcm", ring_size=1024, device="cpu")


__all__ = ["LaneConfig", "RuntimeTopology"]
