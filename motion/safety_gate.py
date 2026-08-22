"""MovementSafetyGate: the single place that enforces "wake-word STOP has
priority over any manual command, including one that arrives concurrently."

This is the coordination point between the HC-05 manual controller and the
voice pipeline (via voice/ipc/server_stub.py's ReferenceHumanFollowerServer,
reused unmodified -- see motion/__init__.py). Neither side talks to the
other directly: HC05Controller only ever calls this gate's request_*()
methods, and the IPC server's on_pause/on_resume callbacks only ever call
suppress_and_stop()/release(). Each side stays ignorant of the other's
existence, exactly as required.
"""

from __future__ import annotations

import logging
import threading

from motion.movement_controller import MovementController

logger = logging.getLogger(__name__)


class MovementSafetyGate:
    """Wraps a MovementController, adding a suppression flag.

    While suppressed, every forward/backward/left/right request is
    DROPPED (logged, never queued or buffered for later) -- not delayed,
    not remembered. stop() is always allowed through regardless of
    suppression state, and suppressing itself always issues an immediate
    stop() first, before returning to the caller.

    Race safety: suppress_and_stop() and every request_*() method acquire
    the SAME lock before touching either the suppression flag or the
    underlying MovementController. This means:

      * A movement request that is already inside the lock when
        suppress_and_stop() is called completes first (it was already
        committed to happen), and suppression + stop() are applied
        immediately after -- the wake word never has to wait more than
        one already-in-flight command.
      * A movement request that arrives concurrently but hasn't yet
        acquired the lock will always see the updated suppression state
        once it does -- there is no window where it could sneak through
        after suppression has been set. Stop always wins the race against
        any request that hasn't already started.

    This is deliberately NOT an async/queued design -- HC-05 commands are
    single characters with no inherent ordering guarantee worth
    preserving under suppression, so dropping (not buffering) is the
    correct, safer behavior: a command sent while suppressed must not
    "replay" once released (that would be an unrequested automatic
    resume, explicitly forbidden by the product requirement).
    """

    def __init__(self, movement: MovementController) -> None:
        self._movement = movement
        self._lock = threading.Lock()
        self._suppressed = False

    # ------------------------------------------------------------------
    # HC-05 manual-control side: gated requests.
    # ------------------------------------------------------------------
    def request_forward(self) -> None:
        self._request(self._movement.forward, "forward")

    def request_backward(self) -> None:
        self._request(self._movement.backward, "backward")

    def request_left(self) -> None:
        self._request(self._movement.left, "left")

    def request_right(self) -> None:
        self._request(self._movement.right, "right")

    def _request(self, fn, name: str) -> None:
        with self._lock:
            if self._suppressed:
                logger.info(
                    "Movement request %r ignored -- voice interaction "
                    "currently owns movement state.", name,
                )
                return
            fn()

    # ------------------------------------------------------------------
    # stop() is never gated -- always safe, always allowed, from any
    # caller (HC-05 reader detecting a malformed/disconnect condition,
    # or the voice-IPC side). Does not itself change the suppression
    # flag -- see suppress_and_stop() for the voice-interaction-specific
    # variant that does.
    # ------------------------------------------------------------------
    def stop(self) -> None:
        with self._lock:
            self._movement.stop()

    # ------------------------------------------------------------------
    # Voice-pipeline side, via the IPC server's injected callbacks.
    # ------------------------------------------------------------------
    def suppress_and_stop(self) -> None:
        """Called on PAUSE_REQUEST (wake word confirmed). Sets suppression
        FIRST, then stops, both under the same lock acquisition -- see the
        class docstring for why this ordering, combined with every
        request_*() acquiring the same lock, is what makes "stop always
        wins the race" true rather than just intended.

        Idempotent: calling this again while already suppressed (e.g. a
        second wake word fires before the first session completes) simply
        re-confirms suppression and re-issues stop() -- both are no-ops on
        top of an already-safe state, never an error.
        """
        with self._lock:
            self._suppressed = True
            self._movement.stop()
        logger.info("MovementSafetyGate: suppressed and stopped for voice interaction.")

    def release(self) -> None:
        """Called on VOICE_SESSION_COMPLETE, or by the IPC watchdog if
        voice fails to complete a session at all. ONLY clears the
        suppression flag -- never re-issues a movement command. The car
        remains stopped (motors were already stopped by
        suppress_and_stop() and nothing since has moved them) until a new
        Bluetooth command arrives. This is the load-bearing implementation
        of "do not automatically resume the previous movement direction."
        """
        with self._lock:
            self._suppressed = False
        logger.info("MovementSafetyGate: released -- manual control available again "
                     "(car remains stopped until a new command arrives).")

    @property
    def is_suppressed(self) -> bool:
        with self._lock:
            return self._suppressed


__all__ = ["MovementSafetyGate"]
