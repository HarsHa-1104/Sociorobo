#!/usr/bin/env python3
"""Fake `pw-record` stand-in for tests/test_pw_capture.py -- a real
subprocess (not a mock) that writes controllable raw PCM bytes to stdout
instead of actually capturing from PipeWire, so PipeWireCapture's real
subprocess/pipe-reading code (os.read()/select(), not a mocked shortcut)
can be exercised without needing real PipeWire/Bluetooth hardware.

Recognises special `--target` values to simulate failure modes:
  __DIE_IMMEDIATELY__  -- exits at once with no output (simulates a
                          missing/refused transport)
  __HANG__              -- never writes anything, just sleeps (simulates a
                          connection that never delivers data)
  anything else         -- writes a repeating byte-value pattern in small
                          chunks forever, so a test can read as many bytes
                          as it wants
"""
import sys
import time


def _target_arg() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--target" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return ""


def main() -> int:
    target = _target_arg()

    if target == "__DIE_IMMEDIATELY__":
        return 1

    if target == "__HANG__":
        time.sleep(3600)
        return 0

    out = sys.stdout.buffer
    counter = 0
    while True:
        out.write(bytes([counter % 256]) * 64)
        out.flush()
        counter += 1
        time.sleep(0.005)


if __name__ == "__main__":
    sys.exit(main())
