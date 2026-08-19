#!/usr/bin/env python3
"""Fake Piper stand-in for tests/test_persistent_piper_tts.py -- a real
subprocess (not a mock) that speaks Piper's --json-input +
--output_file - protocol: one JSON line in, one framed WAV blob out.
Exercises the real os.read()/select() code path in PersistentPiperTTS
faithfully, rather than mocking it away.

Special magic `text` values simulate failure modes for testing:
  __CRASH__      -- exit immediately, no response (simulates a died process)
  __HANG__       -- never responds (simulates a stuck request)
  __SLOW:N__     -- sleeps N seconds before responding
"""
import json
import struct
import sys
import time


def _make_wav(num_bytes: int, sample_rate: int = 22050) -> bytes:
    num_bytes -= num_bytes % 2  # keep it a whole number of int16 samples
    data = b"\x00\x00" * (num_bytes // 2)
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    data_chunk = b"data" + struct.pack("<I", len(data)) + data
    riff_body = b"WAVE" + fmt_chunk + data_chunk
    return b"RIFF" + struct.pack("<I", len(riff_body)) + riff_body


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        text = req.get("text", "")

        if text == "__CRASH__":
            return 1
        if text == "__HANG__":
            time.sleep(3600)
            continue
        if text.startswith("__SLOW:"):
            delay = float(text[len("__SLOW:"):].rstrip("_"))
            time.sleep(delay)
            text = "slow response"

        wav = _make_wav(num_bytes=len(text) * 100 + 200)
        sys.stdout.buffer.write(wav)
        sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
