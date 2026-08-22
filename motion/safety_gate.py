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

from motion.movement_controller import DEFAULT_SPEED, MovementController

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
    def request_forward(self, speed: int = DEFAULT_SPEED) -> None:
        self._request(self._movement.forward, "forward", speed)

    def request_backward(self, speed: int = DEFAULT_SPEED) -> None:
        self._request(self._movement.backward, "backward", speed)

    def request_left(self, speed: int = DEFAULT_SPEED) -> None:
        self._request(self._movement.left, "left", speed)

    def request_right(self, speed: int = DEFAULT_SPEED) -> None:
        self._request(self._movement.right, "right", speed)

    def request_forward_left(self, speed: int = DEFAULT_SPEED) -> None:
        self._request(self._movement.forward_left, "forward_left", speed)

    def request_forward_right(self, speed: int = DEFAULT_SPEED) -> None:
        self._request(self._movement.forward_right, "forward_right", speed)

    def request_backward_left(self, speed: int = DEFAULT_SPEED) -> None:
        self._request(self._movement.backward_left, "backward_left", speed)

    def request_backward_right(self, speed: int = DEFAULT_SPEED) -> None:
        self._request(self._movement.backward_right, "backward_right", speed)

    def _request(self, fn, name: str, speed: int) -> None:
        with self._lock:
            if self._suppressed:
                logger.info(
                    "Movement request %r (speed=%d) ignored -- voice interaction "
                    "currently owns movement state.", name, speed,
                )
                return
            fn(speed)

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
        """Called on PAUSE_REQUEST (wake word confirmed). Sets the LOCAL
        suppression flag, tells the underlying MovementController to
        suppress too (see MovementController.suppress()'s docstring for
        why this matters for backends where manual commands never pass
        through THIS gate at all -- e.g. HC-05 wired directly to the
        sketch), then stops -- all under the same lock acquisition. See
        the class docstring for why this ordering, combined with every
        request_*() acquiring the same lock, is what makes "stop always
        wins the race" true rather than just intended (that guarantee is
        specifically about THIS gate's own request_*() callers; it says
        nothing about a backend's own independent command source, which
        is exactly why suppress() exists as a separate, explicit signal).

        Idempotent: calling this again while already suppressed (e.g. a
        second wake word fires before the first session completes) simply
        re-confirms suppression and re-issues stop() -- both are no-ops on
        top of an already-safe state, never an error.
        """
        with self._lock:
            self._suppressed = True
            self._movement.suppress()
            self._movement.stop()
        logger.info("MovementSafetyGate: suppressed and stopped for voice interaction.")

    def release(self) -> None:
        """Called on VOICE_SESSION_COMPLETE, or by the IPC watchdog if
        voice fails to complete a session at all. Clears the LOCAL
        suppression flag and tells the underlying MovementController to
        release too -- but this is NOT "re-issuing a movement command":
        MovementController.release() is explicitly documented to never
        move the motors or resume prior motion by itself, only to stop
        actively refusing new manual commands. The car remains stopped
        (motors were already stopped by suppress_and_stop() and nothing
        since has moved them) until a new Bluetooth command arrives. This
        is the load-bearing implementation of "do not automatically
        resume the previous movement direction."
        """
        with self._lock:
            self._suppressed = False
            self._movement.release()
        logger.info("MovementSafetyGate: released -- manual control available again "
                     "(car remains stopped until a new command arrives).")

    @property
    def is_suppressed(self) -> bool:
        with self._lock:
            return self._suppressed


__all__ = ["MovementSafetyGate"]
