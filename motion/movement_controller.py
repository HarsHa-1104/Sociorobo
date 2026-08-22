"""MovementController: the small hardware-agnostic interface everything
in this package drives motors through.

Deliberately minimal -- five operations, no speed/PWM parameters, no
state queries -- because neither the HC-05 command set (single-character
directions) nor the voice pipeline's need (a bare "stop") requires more
than this. A real hardware implementation is free to internally ramp
speed, debounce, etc.; callers of this interface only ever ask for one of
these five things.

No concrete hardware implementation ships here. The actual motor
driver/GPIO wiring for this board was not present anywhere in this
repository or discoverable on the live system when this was built (see
the motion-integration report) -- inventing pin numbers or a specific
driver chip would violate the explicit instruction not to guess hardware
details. LoggingMovementController is the only concrete implementation
provided: a safe, fully-functional stand-in (mirrors
voice/ipc/server_stub.py's ReferenceHumanFollowerServer's own role for
motor authority -- a real, runnable, honest substitute with no physical
hardware underneath it) usable for development, tests, and dry-run
hardware bring-up before a real motor driver class is written against it.

Write the real implementation once the motor driver/pin/interface details
are known: either a GPIO-backed class (if the motor driver is wired
directly to Linux-exposed GPIO) or a Router-Bridge-backed class (if motor
control lives on the MCU/sketch side -- see the motion-integration report
for why that's the more likely fit for a UNO Q board) -- either way, it
only needs to implement this same five-method interface; nothing else in
this package needs to change.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MovementController(ABC):
    """Hardware-agnostic motor interface. Every method must be safe to
    call repeatedly and safe to call from any thread -- MovementSafetyGate
    is the only intended caller, and it may call these from the HC-05
    reader thread, the voice-IPC dispatch thread, or the watchdog thread.
    """

    @abstractmethod
    def forward(self) -> None: ...

    @abstractmethod
    def backward(self) -> None: ...

    @abstractmethod
    def left(self) -> None: ...

    @abstractmethod
    def right(self) -> None: ...

    @abstractmethod
    def stop(self) -> None:
        """Must unconditionally bring both motors to a stopped state --
        cancels any active forward/backward/turning motion. Called far
        more defensively than the other four methods (every voice
        interaction, every HC-05 disconnect, every malformed command
        recovery), so it must never raise for "nothing was moving
        anyway" -- stopping an already-stopped motor is always a no-op
        success, never an error.
        """
        ...


class LoggingMovementController(MovementController):
    """Safe default/testable implementation: logs every call, drives no
    real hardware. Tracks the last command issued (for tests/inspection),
    but callers must never rely on this for anything safety-relevant --
    it exists to make the rest of this package runnable and testable
    before real hardware is wired in, not to simulate real motor physics.
    """

    def __init__(self) -> None:
        self.last_command: str | None = None
        self.call_count = 0

    def forward(self) -> None:
        self._record("forward")

    def backward(self) -> None:
        self._record("backward")

    def left(self) -> None:
        self._record("left")

    def right(self) -> None:
        self._record("right")

    def stop(self) -> None:
        self._record("stop")

    def _record(self, command: str) -> None:
        self.last_command = command
        self.call_count += 1
        logger.info("[LoggingMovementController] %s", command)


__all__ = ["MovementController", "LoggingMovementController"]
