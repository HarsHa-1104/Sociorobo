"""BridgeMovementController: the real motor-driving MovementController
implementation for this board.

Drives NO hardware directly -- every call is delegated, via the injected
`bridge` object's call(name, *args) method, to the sketch
(arduino_app/motion_controller/sketch/sketch.ino), which owns the actual
GPIO/PWM motor driving and the HC-05 UART connection. In production
`bridge` IS `arduino.app_utils.Bridge` (see
arduino_app/motion_controller/python/main.py, the only place that actually
imports it) -- this module never imports `arduino.app_utils` itself, so it
is fully unit-testable with a fake bridge object, without the App runtime,
without real hardware, and without needing to be run as an Arduino App at
all.

Why the manual-driving path is entirely on the sketch, not here: HC-05 is
wired to the UNO Q's primary hardware UART (D0/D1) -- the MCU's own
Serial, not something exposed to Linux as a /dev/tty* device. Bluetooth
bytes are parsed and (when not voice-suppressed) acted on entirely inside
the sketch's own loop; this class's forward()/backward()/etc. exist for
symmetry with the MovementController interface and for any FUTURE
Python-driven movement (e.g. autonomous following, explicitly out of scope
for this phase) -- they are not on the real-time-critical path for manual
Bluetooth driving today.

Two distinct stop mechanisms, both implemented on the sketch, not here:
  * stop() / suppress() -- the IMMEDIATE, wake-word-triggered hard stop.
    No ramp, no deceleration; motors go to zero output right away.
  * Manual release / normal end-of-command deceleration -- entirely a
    sketch-local behavior (ramping PWM toward zero when no new Bluetooth
    command has arrived recently), never triggered by this class or by
    MovementSafetyGate. Python never asks for a "gradual stop"; the
    ramp-rate/timeout constants live in the sketch (see its RAMP_STEP_PWM/
    RAMP_INTERVAL_MS), configurable there without touching this file, the
    HC-05 parsing, or the voice integration at all.
"""

from __future__ import annotations

import logging
from typing import Protocol

from motion.movement_controller import DEFAULT_SPEED, MovementController

logger = logging.getLogger(__name__)


class BridgeLike(Protocol):
    """The minimal interface this class needs from a Bridge object --
    matches arduino.app_utils.Bridge.call's actual usage (see the bundled
    examples in /var/lib/arduino-app-cli/examples/core-and-foundational/
    03-bridge-basics/), so a test can inject a fake without importing
    arduino.app_utils or running inside the App runtime at all."""

    def call(self, name: str, *args): ...  # pragma: no cover - protocol only


class BridgeMovementController(MovementController):
    def __init__(self, bridge: BridgeLike) -> None:
        self._bridge = bridge

    def forward(self, speed: int = DEFAULT_SPEED) -> None:
        self._set_motion("forward", speed)

    def backward(self, speed: int = DEFAULT_SPEED) -> None:
        self._set_motion("backward", speed)

    def left(self, speed: int = DEFAULT_SPEED) -> None:
        self._set_motion("left", speed)

    def right(self, speed: int = DEFAULT_SPEED) -> None:
        self._set_motion("right", speed)

    def forward_left(self, speed: int = DEFAULT_SPEED) -> None:
        self._set_motion("forward_left", speed)

    def forward_right(self, speed: int = DEFAULT_SPEED) -> None:
        self._set_motion("forward_right", speed)

    def backward_left(self, speed: int = DEFAULT_SPEED) -> None:
        self._set_motion("backward_left", speed)

    def backward_right(self, speed: int = DEFAULT_SPEED) -> None:
        self._set_motion("backward_right", speed)

    def _set_motion(self, direction: str, speed: int) -> None:
        try:
            self._bridge.call("set_motion", direction, int(speed))
        except Exception:
            # A Bridge call failing must never propagate as an unhandled
            # exception up through MovementSafetyGate -- the caller (HC-05
            # reader thread, voice-IPC dispatch thread, watchdog thread)
            # has no motor-specific recovery of its own to fall back to,
            # and a raised exception here could leave the gate's lock
            # (see MovementSafetyGate._request/suppress_and_stop) in a
            # torn state if it propagated out of a `with self._lock:` body.
            logger.exception("Bridge call 'set_motion' failed (direction=%s speed=%s)", direction, speed)

    def stop(self) -> None:
        try:
            self._bridge.call("stop")
        except Exception:
            logger.exception("Bridge call 'stop' failed -- motors may not be confirmed stopped.")

    def suppress(self) -> None:
        try:
            self._bridge.call("voice_suppress")
        except Exception:
            logger.exception("Bridge call 'voice_suppress' failed -- sketch may still act on Bluetooth commands.")

    def release(self) -> None:
        try:
            self._bridge.call("voice_release")
        except Exception:
            logger.exception("Bridge call 'voice_release' failed -- manual control may not resume.")


__all__ = ["BridgeMovementController", "BridgeLike"]
