#!/usr/bin/env python3
"""Run the manual-control Motion Controller: HC-05 Bluetooth serial ->
MovementSafetyGate -> MovementController, coordinated with the voice
pipeline through the SAME IPC contract HumanFollower was always designed
to use (voice/ipc/) -- see motion/__init__.py for the full picture.

This is a SEPARATE OS process from main_voice.py, exactly like the
"two separate processes, never threads within one process" architecture
documented in docs/ARCHITECTURE.md. Run this in one terminal,
`python3 main_voice.py` in another -- same pattern as
scripts/run_reference_humanfollower.py, which this replaces for real
(manual-control) use. Voice Manager needs zero changes and zero awareness
of which one is running on the other end of the socket.

Usage:
    python3 scripts/run_motion_controller.py --port /dev/ttyUSB0
    python3 scripts/run_motion_controller.py --port /dev/ttyUSB0 --baudrate 9600 --dry-run

--dry-run (the default while no real motor driver class exists -- see
motion/movement_controller.py) uses LoggingMovementController: HC-05
commands and voice-triggered stops are logged, not sent to real hardware.
Once a real MovementController implementation exists for this board's
actual motor wiring, wire it in at the bottom of main() in place of
LoggingMovementController -- nothing else in this script needs to change.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from motion.hc05_controller import HC05Controller
from motion.movement_controller import LoggingMovementController
from motion.safety_gate import MovementSafetyGate
from voice.config import load_config
from voice.ipc.server_stub import ReferenceHumanFollowerServer


def main() -> int:
    parser = argparse.ArgumentParser(description="HC-05 manual-control Motion Controller")
    parser.add_argument("--port", required=True, help="Serial device path for the HC-05 module (e.g. /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=9600, help="HC-05 baud rate (factory default: 9600)")
    parser.add_argument("--config", type=str, default=None, help="Path to voice_config.yaml (for ipc/watchdog settings)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logger = logging.getLogger("motion_controller")

    config = load_config(args.config)

    # No real motor driver/GPIO/Bridge implementation exists for this
    # board yet (see motion/movement_controller.py's docstring for why --
    # the wiring was not present anywhere in the repository or on the
    # live system). LoggingMovementController is a safe, honest stand-in:
    # it drives no real hardware. Replace this one line once a real
    # MovementController implementation exists.
    movement = LoggingMovementController()
    gate = MovementSafetyGate(movement)

    hc05 = HC05Controller(gate, port=args.port, baudrate=args.baudrate)
    hc05.start()

    def on_pause() -> bool:
        gate.suppress_and_stop()
        return True  # synchronous: suppress_and_stop() has already returned, motors are confirmed stopped

    def on_resume() -> None:
        gate.release()

    def on_watchdog_forced_resume(reason: str) -> None:
        logger.warning("Motion Controller watchdog forced a resume: %s", reason)

    server = ReferenceHumanFollowerServer(
        config.ipc, config.watchdog,
        on_pause=on_pause, on_resume=on_resume,
        on_watchdog_forced_resume=on_watchdog_forced_resume,
    )
    server.start()
    logger.info(
        "Motion Controller running: HC-05 on %s @ %d baud, IPC socket %s. Ctrl+C to stop.",
        args.port, args.baudrate, config.ipc.socket_path,
    )

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
        hc05.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
