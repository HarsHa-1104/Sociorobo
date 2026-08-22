"""Unit tests for motion/movement_controller.py's LoggingMovementController."""

from __future__ import annotations

from motion.movement_controller import LoggingMovementController


def test_each_method_records_the_correct_command():
    m = LoggingMovementController()
    m.forward()
    assert m.last_command == "forward"
    m.backward()
    assert m.last_command == "backward"
    m.left()
    assert m.last_command == "left"
    m.right()
    assert m.last_command == "right"
    m.stop()
    assert m.last_command == "stop"


def test_call_count_increments_for_every_call_including_repeated_stop():
    m = LoggingMovementController()
    m.forward()
    m.stop()
    m.stop()  # stopping an already-stopped motor must be a safe no-op, not an error
    assert m.call_count == 3
