# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Motion Controller App -- the REAL entrypoint for this board.

This is the only file in the whole project that imports arduino.app_utils
(confirmed only importable inside the arduino-app-cli App runtime -- see
motion/bridge_motor_controller.py's docstring). Its whole job is wiring:

    real Bridge  ->  BridgeMovementController  ->  MovementSafetyGate  <-  IPC server
                      (motion/bridge_motor_controller.py)  (motion/safety_gate.py)   (voice/ipc/server_stub.py,
                                                                                       reused UNCHANGED)

HC-05 command parsing and all motor GPIO/PWM driving happen entirely in
../sketch/sketch.ino, not here -- see that file's header comment for why
(HC-05 is wired to this board's own MCU hardware UART, not something this
Python process can read). This file never touches GPIO and never parses a
Bluetooth byte; it only relays voice-pipeline pause/resume signals to the
sketch via Bridge.call, exactly mirroring what scripts/run_motion_controller.py
already does for the pyserial-based deployment, with BridgeMovementController
in place of LoggingMovementController.

The rest of this project (motion/, voice/) lives in the main repository,
not inside this App folder -- imported via sys.path below rather than
duplicated, so there is exactly one copy of motion/safety_gate.py and
voice/ipc/ to keep in sync. Both packages are pure standard library (no
third-party imports -- confirmed by inspection), so nothing else needs to
be installed into this App's Python environment for the import to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The main repository this App's supporting code lives in -- see the
# module docstring above for why this is imported by path instead of
# duplicated into this App folder.
_REPO_ROOT = Path("/home/arduino/Downloads/SocialRobot-UNOQ")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arduino.app_utils import App, Bridge, Logger  # noqa: E402 -- after sys.path setup, by necessity

from motion.bridge_motor_controller import BridgeMovementController  # noqa: E402
from motion.safety_gate import MovementSafetyGate  # noqa: E402
from voice.config import load_config  # noqa: E402
from voice.ipc.server_stub import ReferenceHumanFollowerServer  # noqa: E402

# Logger("name") is the App-runtime logger (confirmed via bundled examples,
# e.g. inspirational/common/theremin) whose output surfaces in
# `arduino-app-cli app logs` -- used instead of stdlib logging.getLogger
# for anything this App itself wants visible there. motion/ and voice/
# still use stdlib `logging` internally (unchanged, App-runtime-agnostic);
# nothing here reconfigures or interferes with that.
logger = Logger("motion_controller")

# BridgeMovementController never imports arduino.app_utils itself (see its
# docstring) -- the real Bridge object is injected here, the one place
# that actually has it.
movement = BridgeMovementController(Bridge)
gate = MovementSafetyGate(movement)

config = load_config()


def on_pause() -> bool:
    gate.suppress_and_stop()
    return True  # synchronous: suppress_and_stop() has already returned, Bridge.call("voice_suppress")/("stop") sent


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
    "Motion Controller App ready: sketch owns HC-05 + motor GPIO, this process "
    "relays voice-pipeline pause/resume over Bridge and IPC socket %s.",
    config.ipc.socket_path,
)

App.run()
