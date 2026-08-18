#!/usr/bin/env python3
"""Record a WAV fixture through the real AudioManager capture path.

Unlike a raw `arecord` capture, this goes through the exact same
PyAudio-open -> 48kHz-capture -> resample-to-16kHz -> 480-sample-frame path
that VoiceManager uses in production (voice/audio/manager.py). That makes
the output usable for controlled wake-word/VAD/STT comparisons where the
whole point is to test against audio that has already been through the
pipeline's own resampling -- not a separately-recorded clip that only
approximates it.

Usage:
    python3 scripts/record_test_audio.py --seconds 5 --out fixture.wav
    python3 scripts/record_test_audio.py --seconds 5 --out fixture.wav --label "hey_jarvis_close"
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.audio.manager import AudioManager
from voice.config import load_config


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--countdown", type=float, default=2.0,
                         help="Seconds to wait before recording starts, so you have time to get ready.")
    args = parser.parse_args()

    config = load_config(args.config)
    audio = AudioManager(config.audio)
    audio.start()

    try:
        if args.countdown > 0:
            print(f"Recording starts in {args.countdown:.0f}s...")
            time.sleep(args.countdown)
        print(f"Recording {args.seconds:.1f}s at pipeline rate "
              f"{AudioManager.PIPELINE_SAMPLE_RATE}Hz (post-resample, matches production)...")

        chunks = []
        n_frames = int(args.seconds * 1000 / config.audio.frame_duration_ms)
        for i, frame in enumerate(audio.frames()):
            chunks.append(frame)
            if i + 1 >= n_frames:
                break
        pcm = b"".join(chunks)
    finally:
        audio.stop()

    out_path = Path(args.out)
    _write_wav(out_path, pcm, AudioManager.PIPELINE_SAMPLE_RATE)
    print(f"Wrote {out_path} ({len(pcm)} bytes, "
          f"{len(pcm) / 2 / AudioManager.PIPELINE_SAMPLE_RATE:.2f}s at 16kHz mono)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
