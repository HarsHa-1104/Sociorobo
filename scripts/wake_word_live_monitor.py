#!/usr/bin/env python3
"""Live wake-word score monitor -- for the human-in-the-loop validation
step that automated testing cannot substitute for (see
scripts/wake_word_diagnostic.py's caveat: a speaker->mic loopback of
synthesized speech is a harsher, double-transduction stress test, not a
faithful stand-in for a human talking directly into the room).

Run this, then say the wake phrase ("hey jarvis") a few times at normal
speaking distance and volume during the countdown+run window. Every score
above --log-threshold is printed live and written to --log so it can be
reviewed afterward without anyone having to watch the terminal in real
time.

Usage:
    python3 scripts/wake_word_live_monitor.py --seconds 25
    python3 scripts/wake_word_live_monitor.py --seconds 25 --log-threshold 0.05
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.audio.manager import AudioManager
from voice.config import load_config
from voice.wake.wake_word import WakeWordDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--log-threshold", type=float, default=0.05,
                         help="Only print/log scores at or above this value (0 = log everything).")
    parser.add_argument("--log", type=str, default="tests/fixtures/wake_live_scores.log")
    parser.add_argument("--countdown", type=float, default=3.0)
    args = parser.parse_args()

    config = load_config()
    wake_cfg = config.wake
    wake_cfg.enabled = True  # force on for this diagnostic regardless of live config

    detector = WakeWordDetector(wake_cfg, sample_rate=16000)
    real_predict = detector._model.predict
    log_path = Path(args.log)
    log_lines = []

    def _spy_predict(audio):
        result = real_predict(audio)
        score = max(result.values()) if result else 0.0
        t = time.monotonic() - t_start
        if score >= args.log_threshold:
            line = f"t={t:6.2f}s score={score:.4f} threshold={wake_cfg.threshold} " \
                   f"{'>>> ABOVE THRESHOLD' if score >= wake_cfg.threshold else ''}"
            print(line)
            log_lines.append(line)
        return result

    detector._model.predict = _spy_predict

    audio = AudioManager(config.audio)
    audio.start()

    print(f"Model={wake_cfg.model_name} threshold={wake_cfg.threshold} trigger_level={wake_cfg.trigger_level}")
    print(f"Starting in {args.countdown:.0f}s -- say the wake phrase a few times at normal "
          f"distance/volume once recording starts.")
    for i in range(int(args.countdown), 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)
    print(f"LISTENING for {args.seconds:.0f}s now.")

    fired_at = []
    t_start = time.monotonic()
    try:
        for frame in audio.frames():
            if detector.process_frame(frame):
                t = time.monotonic() - t_start
                fired_at.append(t)
                print(f"  *** WAKE FIRED at t={t:.2f}s ***")
            if time.monotonic() - t_start > args.seconds:
                break
    finally:
        audio.stop()

    print(f"\nDone. {len(fired_at)} wake trigger(s): {[round(t, 2) for t in fired_at]}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"model={wake_cfg.model_name} threshold={wake_cfg.threshold} "
                f"trigger_level={wake_cfg.trigger_level}\n")
        f.write(f"fired_at={fired_at}\n")
        f.write("\n".join(log_lines) + "\n")
    print(f"Full log written to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
