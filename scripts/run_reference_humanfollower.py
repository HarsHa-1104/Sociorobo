#!/usr/bin/env python3
"""Run the reference HumanFollower IPC server standalone.

This is NOT a real HumanFollower -- it has no camera, no motors, no
control loop. It exists so main_voice.py can be run and manually tested
(say the wake word, ask a question, hear a response) BEFORE the real
HumanFollower process is wired up to the IPC contract documented in
humanfollower_integration/README.md.

Run this in one terminal, `python3 main_voice.py` in another.
"""

from __future__ import annotations

import logging
import signal
import sys
import time

from voice.config import load_config
from voice.ipc.server_stub import ReferenceHumanFollowerServer


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logger = logging.getLogger("reference_humanfollower")

    config = load_config()

    def on_pause() -> bool:
        logger.info(">>> [SIMULATED] Decelerating and stopping motors...")
        time.sleep(0.3)  # pretend this takes a moment, like a real deceleration ramp would
        logger.info(">>> [SIMULATED] Motors stopped.")
        return True

    def on_resume() -> None:
        logger.info(">>> [SIMULATED] Resuming following.")

    def on_watchdog_forced_resume(reason: str) -> None:
        logger.warning(">>> [SIMULATED] WATCHDOG forced a resume: %s", reason)

    server = ReferenceHumanFollowerServer(
        config.ipc, config.watchdog,
        on_pause=on_pause, on_resume=on_resume,
        on_watchdog_forced_resume=on_watchdog_forced_resume,
    )
    server.start()
    logger.info("Reference HumanFollower server running. Ctrl+C to stop.")

    stop = False

    def _handle_signal(signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop:
            time.sleep(0.2)
    finally:
        server.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
