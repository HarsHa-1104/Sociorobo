"""MovementController: the small hardware-agnostic interface everything
in this package drives motors through.

Extended (real-motor-control phase) from the original five-operation
interface to eight directions (four cardinal + four diagonal, for
mecanum-wheel driving) plus an optional per-call speed -- both genuinely
new requirements (diagonal driving, a speed slider) that the original
bare "five directions, no parameters" interface didn't cover. `speed`
defaults to DEFAULT_SPEED on every method so every existing caller that
never passed one (tests, HC05Controller's plain single-character commands)
keeps working unchanged. stop() deliberately still takes no speed --
"stop" has exactly one meaning.

No concrete real-hardware implementation ships in THIS repository. Real
motor driving lives in a genuine Arduino App (sketch + its own
python/main.py using the real Bridge API) -- see arduino_app/motion_controller/
and this package's BridgeMovementController (motion/bridge_motor_controller.py)
for why: `arduino.app_utils`/Bridge is only importable from inside the
arduino-app-cli App runtime, confirmed empirically (every bundled example,
without exception, imports it that way; it does not exist as a
standalone/installable package). LoggingMovementController remains the
safe stand-in for development, tests, and dry-run bring-up.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

DEFAULT_SPEED = 200  # out of a 0-255 PWM range (see motion/bridge_motor_controller.py's
                      # MAX_PWM) -- a reasonably brisk but not maximum default, used
                      # whenever a caller doesn't specify one (HC05Controller's plain
                      # single-character commands, and every existing test).
MAX_SPEED = 255
MIN_SPEED = 0

# The eight logical directions the Bluetooth RC Controller app's buttons
# map to, plus "stop" -- used as the canonical action names throughout
# this package (HC05Controller's command_map values, MovementSafetyGate's
# request_*() method names minus the "request_" prefix).
DIRECTIONS = (
    "forward", "backward", "left", "right",
    "forward_left", "forward_right", "backward_left", "backward_right",
)


class MovementController(ABC):
    """Hardware-agnostic motor interface. Every method must be safe to
    call repeatedly and safe to call from any thread -- MovementSafetyGate
    is the only intended caller, and it may call these from the HC-05
    reader thread (or, for the real hardware, the sketch's own local
    suppression logic -- see arduino_app/motion_controller/sketch/sketch.ino),
    the voice-IPC dispatch thread, or the watchdog thread.
    """

    @abstractmethod
    def forward(self, speed: int = DEFAULT_SPEED) -> None: ...

    @abstractmethod
    def backward(self, speed: int = DEFAULT_SPEED) -> None: ...

    @abstractmethod
    def left(self, speed: int = DEFAULT_SPEED) -> None: ...

    @abstractmethod
    def right(self, speed: int = DEFAULT_SPEED) -> None: ...

    @abstractmethod
    def forward_left(self, speed: int = DEFAULT_SPEED) -> None: ...

    @abstractmethod
    def forward_right(self, speed: int = DEFAULT_SPEED) -> None: ...

    @abstractmethod
    def backward_left(self, speed: int = DEFAULT_SPEED) -> None: ...

    @abstractmethod
    def backward_right(self, speed: int = DEFAULT_SPEED) -> None: ...

    @abstractmethod
    def stop(self) -> None:
        """Must unconditionally bring all four motors to a stopped state
        -- cancels any active forward/backward/turning/diagonal motion.
        Called far more defensively than the other eight methods (every
        voice interaction, every HC-05 disconnect, every malformed
        command recovery), so it must never raise for "nothing was moving
        anyway" -- stopping an already-stopped motor is always a no-op
        success, never an error. This is always the IMMEDIATE/hard stop
        (the wake-word safety path) -- see
        motion/bridge_motor_controller.py's docstring for how the real
        implementation distinguishes this from manual-release's gradual
        deceleration, which is NOT part of this interface (it's a
        sketch-local behavior triggered by the absence of new Bluetooth
        commands, not something Python ever requests directly).
        """
        ...

    @abstractmethod
    def suppress(self) -> None:
        """Called by MovementSafetyGate.suppress_and_stop(), immediately
        before stop(). Exists for backends where manual commands are NOT
        routed through this Python process's MovementSafetyGate at all --
        the real hardware backend (BridgeMovementController) is exactly
        this case: HC-05 is wired to the MCU's own UART, so Bluetooth
        bytes are parsed and acted on entirely by the sketch, never seen
        by Python. Without a way to tell the sketch "manual commands must
        not move the motors right now", stop() alone would be a one-time
        event -- the very next Bluetooth byte the sketch receives would
        move the car again immediately, defeating the whole point of
        suppression. For backends where MovementSafetyGate IS the only
        gatekeeper (LoggingMovementController, or a future backend whose
        manual-command source genuinely does go through
        MovementSafetyGate.request_*()), this can be a no-op -- the
        Python-side suppression flag already does the job.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Called by MovementSafetyGate.release() -- the mirror image of
        suppress(). Must NOT move the motors or resume any prior motion by
        itself (same "no automatic resume" rule as the gate's own
        release()) -- it only tells the backend that manual commands may
        be honored again from here on.
        """
        ...


class LoggingMovementController(MovementController):
    """Safe default/testable implementation: logs every call, drives no
    real hardware. Tracks the last command and speed issued (for
    tests/inspection), but callers must never rely on this for anything
    safety-relevant -- it exists to make the rest of this package runnable
    and testable before/without real hardware, not to simulate real motor
    physics or deceleration.
    """

    def __init__(self) -> None:
        self.last_command: str | None = None
        self.last_speed: int | None = None
        self.call_count = 0
        self.is_suppressed = False  # tracked for tests/inspection only -- MovementSafetyGate
                                     # is the actual authority on suppression for this backend;
                                     # this just mirrors it so tests can observe suppress()/
                                     # release() were actually called

    def forward(self, speed: int = DEFAULT_SPEED) -> None:
        self._record("forward", speed)

    def backward(self, speed: int = DEFAULT_SPEED) -> None:
        self._record("backward", speed)

    def left(self, speed: int = DEFAULT_SPEED) -> None:
        self._record("left", speed)

    def right(self, speed: int = DEFAULT_SPEED) -> None:
        self._record("right", speed)

    def forward_left(self, speed: int = DEFAULT_SPEED) -> None:
        self._record("forward_left", speed)

    def forward_right(self, speed: int = DEFAULT_SPEED) -> None:
        self._record("forward_right", speed)

    def backward_left(self, speed: int = DEFAULT_SPEED) -> None:
        self._record("backward_left", speed)

    def backward_right(self, speed: int = DEFAULT_SPEED) -> None:
        self._record("backward_right", speed)

    def stop(self) -> None:
        self._record("stop", None)

    def suppress(self) -> None:
        self.is_suppressed = True
        logger.info("[LoggingMovementController] suppress")

    def release(self) -> None:
        self.is_suppressed = False
        logger.info("[LoggingMovementController] release")

    def _record(self, command: str, speed: int | None) -> None:
        self.last_command = command
        self.last_speed = speed
        self.call_count += 1
        logger.info("[LoggingMovementController] %s speed=%s", command, speed)


__all__ = ["MovementController", "LoggingMovementController", "DIRECTIONS", "DEFAULT_SPEED", "MAX_SPEED", "MIN_SPEED"]
