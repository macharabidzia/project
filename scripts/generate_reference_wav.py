from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np


def build_reference_wave(duration_seconds: float, sample_rate: int) -> np.ndarray:
    total_samples = int(sample_rate * duration_seconds)
    timeline = np.linspace(0.0, duration_seconds, total_samples, endpoint=False, dtype=np.float64)

    fundamental_hz = 180.0
    signal = 0.58 * np.sin(2.0 * math.pi * fundamental_hz * timeline)
    signal += 0.24 * np.sin(2.0 * math.pi * fundamental_hz * 2.0 * timeline)
    signal += 0.12 * np.sin(2.0 * math.pi * fundamental_hz * 3.0 * timeline)
    signal += 0.05 * np.sin(2.0 * math.pi * fundamental_hz * 4.0 * timeline)

    syllable_envelope = 0.58 + 0.42 * np.sin(2.0 * math.pi * 3.8 * timeline)
    signal *= syllable_envelope

    fade_samples = max(1, int(sample_rate * 0.05))
    signal[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
    signal[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    signal = np.clip(signal * 0.38, -1.0, 1.0)
    return (signal * 32767.0).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic reference WAV for the speech-runtime voice profile.")
    parser.add_argument("output", help="Output WAV path")
    parser.add_argument("--duration", type=float, default=4.0, help="Audio duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Sample rate in Hz")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcm = build_reference_wave(args.duration, args.sample_rate)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(args.sample_rate)
        wav_file.writeframes(pcm.tobytes())

    print(f"generated reference wav: {output_path}")


if __name__ == "__main__":
    main()
