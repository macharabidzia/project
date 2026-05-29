from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from enum import Enum

from voice_pipeline.shared.time import now_ns


class EventType(str, Enum):
    ASR_SLOT = "ASR_SLOT"
    VLLM_REQUEST_SLOT = "VLLM_REQUEST_SLOT"
    VLLM_TOKEN_SLOT = "VLLM_TOKEN_SLOT"
    TTS_REQUEST_SLOT = "TTS_REQUEST_SLOT"
    PCM_SLOT = "PCM_SLOT"
    CONTROL_SLOT = "CONTROL_SLOT"


class LaneId(int, Enum):
    ASR = 1
    LLM = 2
    TTS = 3
    CTRL = 4


class SlotType(int, Enum):
    EVENT = 1
    TOKEN = 2
    PCM = 3
    CMD = 4


def event_type_lane(event_type: EventType) -> LaneId:
    mapping = {
        EventType.ASR_SLOT: LaneId.ASR,
        EventType.VLLM_REQUEST_SLOT: LaneId.LLM,
        EventType.VLLM_TOKEN_SLOT: LaneId.LLM,
        EventType.TTS_REQUEST_SLOT: LaneId.TTS,
        EventType.PCM_SLOT: LaneId.TTS,
        EventType.CONTROL_SLOT: LaneId.CTRL,
    }
    return mapping[event_type]


def event_type_slot_type(event_type: EventType) -> SlotType:
    mapping = {
        EventType.ASR_SLOT: SlotType.EVENT,
        EventType.VLLM_REQUEST_SLOT: SlotType.CMD,
        EventType.VLLM_TOKEN_SLOT: SlotType.TOKEN,
        EventType.TTS_REQUEST_SLOT: SlotType.CMD,
        EventType.PCM_SLOT: SlotType.PCM,
        EventType.CONTROL_SLOT: SlotType.CMD,
    }
    return mapping[event_type]


class KernelSlotABI(ctypes.Structure):
    # Mirror of the frozen cross-language slot ABI for FFI verification.
    _fields_ = [
        ("epoch", ctypes.c_uint64),
        ("seq", ctypes.c_uint64),
        ("timestamp_ns", ctypes.c_uint64),
        ("lane", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("payload_len", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ptr", ctypes.c_uint64),
        ("_pad", ctypes.c_uint8 * 16),
    ]
    _align_ = 64


KERNEL_SLOT_ABI_SIZE_BYTES = 64
KERNEL_SLOT_ABI_ALIGNMENT_BYTES = 64
KERNEL_SLOT_MAX_PAYLOAD_BYTES = 4096


def assert_kernel_slot_abi() -> None:
    size = ctypes.sizeof(KernelSlotABI)
    alignment = ctypes.alignment(KernelSlotABI)
    assert size == KERNEL_SLOT_ABI_SIZE_BYTES, f"KernelSlot ABI size mismatch: {size}"
    # Python ctypes does not always materialize 64-byte type alignment across all platforms,
    # so we verify the struct stride contract and keep the explicit padding field for 64-byte ABI size.
    assert alignment >= 8, f"KernelSlot ABI alignment too small: {alignment}"

    offsets = {
        "epoch": getattr(KernelSlotABI, "epoch").offset,
        "seq": getattr(KernelSlotABI, "seq").offset,
        "timestamp_ns": getattr(KernelSlotABI, "timestamp_ns").offset,
        "lane": getattr(KernelSlotABI, "lane").offset,
        "type": getattr(KernelSlotABI, "type").offset,
        "payload_len": getattr(KernelSlotABI, "payload_len").offset,
        "flags": getattr(KernelSlotABI, "flags").offset,
        "ptr": getattr(KernelSlotABI, "ptr").offset,
    }
    assert offsets == {
        "epoch": 0,
        "seq": 8,
        "timestamp_ns": 16,
        "lane": 24,
        "type": 28,
        "payload_len": 32,
        "flags": 36,
        "ptr": 40,
    }, f"KernelSlot ABI offsets mismatch: {offsets}"


@dataclass(frozen=True, slots=True)
class RingSlot:
    event_type: EventType
    ptr: int
    size: int
    lineage_id: str
    sequence_no: int
    epoch_id: str = ""
    timestamp_ns: int = field(default_factory=now_ns)
    flags: int = 0
    metadata: tuple[tuple[str, object], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ptr", int(self.ptr))
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "sequence_no", int(self.sequence_no))
        object.__setattr__(self, "lineage_id", str(self.lineage_id or "").strip())
        object.__setattr__(self, "epoch_id", str(self.epoch_id or "").strip())
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "flags", int(self.flags))
        if isinstance(self.metadata, dict):
            normalized = tuple((str(key), value) for key, value in self.metadata.items())
        else:
            normalized = tuple((str(key), value) for key, value in self.metadata)
        object.__setattr__(self, "metadata", normalized)
        if int(self.size) > KERNEL_SLOT_MAX_PAYLOAD_BYTES:
            raise ValueError("slot payload exceeds maximum allowed bytes")


__all__ = [
    "EventType",
    "KERNEL_SLOT_ABI_ALIGNMENT_BYTES",
    "KERNEL_SLOT_ABI_SIZE_BYTES",
    "KERNEL_SLOT_MAX_PAYLOAD_BYTES",
    "KernelSlotABI",
    "LaneId",
    "RingSlot",
    "SlotType",
    "assert_kernel_slot_abi",
    "event_type_lane",
    "event_type_slot_type",
]


