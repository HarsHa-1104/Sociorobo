"""Unit tests for motion/bridge_motor_controller.py's BridgeMovementController
-- using a fake bridge object, no arduino.app_utils import, no App runtime,
no real hardware."""

from __future__ import annotations

import pytest

from motion.bridge_motor_controller import BridgeMovementController


class _FakeBridge:
    def __init__(self, fail_names=()):
        self.calls: list[tuple] = []
        self._fail_names = set(fail_names)

    def call(self, name, *args):
        self.calls.append((name, *args))
        if name in self._fail_names:
            raise RuntimeError(f"simulated Bridge failure for {name!r}")


@pytest.mark.parametrize("method,direction", [
    ("forward", "forward"),
    ("backward", "backward"),
    ("left", "left"),
    ("right", "right"),
    ("forward_left", "forward_left"),
    ("forward_right", "forward_right"),
    ("backward_left", "backward_left"),
    ("backward_right", "backward_right"),
])
def test_each_direction_calls_set_motion_with_the_correct_direction_and_speed(method, direction):
    bridge = _FakeBridge()
    ctrl = BridgeMovementController(bridge)
    getattr(ctrl, method)(speed=173)
    assert bridge.calls == [("set_motion", direction, 173)]


def test_default_speed_is_used_when_not_specified():
    from motion.movement_controller import DEFAULT_SPEED
    bridge = _FakeBridge()
    ctrl = BridgeMovementController(bridge)
    ctrl.forward()
    assert bridge.calls == [("set_motion", "forward", DEFAULT_SPEED)]


def test_speed_is_coerced_to_int_before_the_bridge_call():
    bridge = _FakeBridge()
    ctrl = BridgeMovementController(bridge)
    ctrl.forward(speed=173.9)  # a caller could pass a float; the sketch expects an int
    assert bridge.calls == [("set_motion", "forward", 173)]


def test_stop_calls_the_stop_bridge_function_with_no_arguments():
    bridge = _FakeBridge()
    ctrl = BridgeMovementController(bridge)
    ctrl.stop()
    assert bridge.calls == [("stop",)]


def test_suppress_calls_the_voice_suppress_bridge_function():
    bridge = _FakeBridge()
    ctrl = BridgeMovementController(bridge)
    ctrl.suppress()
    assert bridge.calls == [("voice_suppress",)]


def test_release_calls_the_voice_release_bridge_function():
    bridge = _FakeBridge()
    ctrl = BridgeMovementController(bridge)
    ctrl.release()
    assert bridge.calls == [("voice_release",)]


# ---------------------------------------------------------------------------
# A failing Bridge call must never raise out of this class -- callers
# (MovementSafetyGate) have no motor-specific recovery and a raised
# exception could leave the gate's lock in a bad state.
# ---------------------------------------------------------------------------

def test_bridge_failure_on_set_motion_does_not_raise():
    bridge = _FakeBridge(fail_names={"set_motion"})
    ctrl = BridgeMovementController(bridge)
    ctrl.forward()  # must not raise
    assert bridge.calls  # the call was still attempted


def test_bridge_failure_on_stop_does_not_raise():
    bridge = _FakeBridge(fail_names={"stop"})
    ctrl = BridgeMovementController(bridge)
    ctrl.stop()  # must not raise


def test_bridge_failure_on_suppress_does_not_raise():
    bridge = _FakeBridge(fail_names={"voice_suppress"})
    ctrl = BridgeMovementController(bridge)
    ctrl.suppress()  # must not raise


def test_bridge_failure_on_release_does_not_raise():
    bridge = _FakeBridge(fail_names={"voice_release"})
    ctrl = BridgeMovementController(bridge)
    ctrl.release()  # must not raise
