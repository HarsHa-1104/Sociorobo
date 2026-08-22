"""Unit tests for motion/safety_gate.py's MovementSafetyGate -- the core
"wake word always wins" safety logic."""

from __future__ import annotations

import threading
import time

from motion.movement_controller import LoggingMovementController
from motion.safety_gate import MovementSafetyGate


def _gate():
    movement = LoggingMovementController()
    return MovementSafetyGate(movement), movement


# ---------------------------------------------------------------------------
# 1-5: basic command -> motion request mapping
# ---------------------------------------------------------------------------

def test_forward_request_reaches_the_movement_controller():
    gate, movement = _gate()
    gate.request_forward()
    assert movement.last_command == "forward"


def test_backward_request_reaches_the_movement_controller():
    gate, movement = _gate()
    gate.request_backward()
    assert movement.last_command == "backward"


def test_left_request_reaches_the_movement_controller():
    gate, movement = _gate()
    gate.request_left()
    assert movement.last_command == "left"


def test_right_request_reaches_the_movement_controller():
    gate, movement = _gate()
    gate.request_right()
    assert movement.last_command == "right"


def test_stop_reaches_the_movement_controller_and_is_never_gated():
    gate, movement = _gate()
    gate.suppress_and_stop()  # suppressed...
    movement.last_command = None  # reset to isolate the next call
    gate.stop()  # ...but stop() must still go through
    assert movement.last_command == "stop"


# ---------------------------------------------------------------------------
# 9: voice interaction suppresses movement
# ---------------------------------------------------------------------------

def test_suppressed_gate_drops_movement_requests_not_queues_them():
    gate, movement = _gate()
    gate.suppress_and_stop()
    movement.call_count = 0  # isolate from suppress_and_stop's own stop() call

    gate.request_forward()
    gate.request_backward()
    gate.request_left()
    gate.request_right()

    assert movement.call_count == 0, "dropped, not buffered -- none of these must reach the controller"
    assert movement.last_command == "stop", "last_command must remain 'stop', proving nothing else fired"


# ---------------------------------------------------------------------------
# 11/12: completion does not auto-resume; a new command is required
# ---------------------------------------------------------------------------

def test_release_does_not_reissue_any_movement_command():
    gate, movement = _gate()
    gate.request_forward()
    gate.suppress_and_stop()
    movement.call_count = 0

    gate.release()

    assert movement.call_count == 0, "release() must never call forward/backward/left/right/stop on its own"
    assert gate.is_suppressed is False


def test_new_command_after_release_is_allowed_through():
    gate, movement = _gate()
    gate.suppress_and_stop()
    gate.release()

    gate.request_forward()

    assert movement.last_command == "forward"


# ---------------------------------------------------------------------------
# 13: repeated wake word remains safely stopped
# ---------------------------------------------------------------------------

def test_repeated_suppress_and_stop_remains_safely_stopped():
    gate, movement = _gate()
    gate.suppress_and_stop()
    gate.suppress_and_stop()  # second wake word before the first session completed
    assert gate.is_suppressed is True
    assert movement.last_command == "stop"

    gate.request_forward()  # still suppressed -- must still be dropped
    assert movement.last_command == "stop"


# ---------------------------------------------------------------------------
# 6/7/8: wake word during any kind of motion -> immediate stop
# ---------------------------------------------------------------------------

def test_wake_word_stops_forward_motion():
    gate, movement = _gate()
    gate.request_forward()
    assert movement.last_command == "forward"
    gate.suppress_and_stop()
    assert movement.last_command == "stop"


def test_wake_word_stops_backward_motion():
    gate, movement = _gate()
    gate.request_backward()
    gate.suppress_and_stop()
    assert movement.last_command == "stop"


def test_wake_word_stops_turning_motion():
    gate, movement = _gate()
    gate.request_left()
    gate.suppress_and_stop()
    assert movement.last_command == "stop"

    gate2, movement2 = _gate()
    gate2.request_right()
    gate2.suppress_and_stop()
    assert movement2.last_command == "stop"


# ---------------------------------------------------------------------------
# 14: Bluetooth command / wake-word race -> stop wins
# ---------------------------------------------------------------------------

def test_concurrent_requests_and_suppression_never_leave_movement_after_stop():
    """Fire many concurrent movement requests and one suppress_and_stop()
    from separate threads. Regardless of interleaving, the final state
    must never be "moving" -- either the request lost the race (dropped)
    or won it (applied, then immediately overridden by stop()). This
    can't deterministically force a specific interleaving, but running it
    many times exercises the lock's actual concurrent behavior rather
    than just asserting the happy path in a single thread."""
    for _ in range(200):
        gate, movement = _gate()
        barrier = threading.Barrier(2)

        def _flood_requests():
            barrier.wait()
            for _ in range(20):
                gate.request_forward()

        def _suppress():
            barrier.wait()
            gate.suppress_and_stop()

        t1 = threading.Thread(target=_flood_requests)
        t2 = threading.Thread(target=_suppress)
        t1.start(); t2.start()
        t1.join(timeout=2.0); t2.join(timeout=2.0)

        assert not t1.is_alive() and not t2.is_alive()
        assert gate.is_suppressed is True
        assert movement.last_command == "stop", (
            "gate.suppress_and_stop() executes stop() while HOLDING the lock, "
            "and every request also takes that lock -- so stop() is always the "
            "last thing to run relative to any request that raced it, whichever "
            "won the race to acquire the lock second"
        )


def test_is_suppressed_reflects_current_state_thread_safely():
    gate, _ = _gate()
    assert gate.is_suppressed is False
    gate.suppress_and_stop()
    assert gate.is_suppressed is True
    gate.release()
    assert gate.is_suppressed is False
