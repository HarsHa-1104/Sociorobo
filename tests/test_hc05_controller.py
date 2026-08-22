"""Unit tests for motion/hc05_controller.py's HC05Controller -- command
parsing/dispatch and disconnect handling, using a fake SerialPort (no
`serial` package I/O, no real hardware)."""

from __future__ import annotations

import threading
import time

import pytest

from motion.hc05_controller import HC05Controller
from motion.movement_controller import LoggingMovementController
from motion.safety_gate import MovementSafetyGate


class _FakeSerialPort:
    """Feeds bytes from a queue to read(1); a byte can be a normal
    command, or the sentinels below to simulate failure modes."""

    RAISE_ON_READ = object()

    def __init__(self):
        self._queue: list = []
        self._lock = threading.Lock()
        self.closed = False

    def push(self, byte: bytes) -> None:
        with self._lock:
            self._queue.append(byte)

    def push_read_error(self) -> None:
        with self._lock:
            self._queue.append(self.RAISE_ON_READ)

    def read(self, size: int = 1) -> bytes:
        with self._lock:
            if self._queue:
                item = self._queue.pop(0)
                if item is self.RAISE_ON_READ:
                    raise OSError("simulated HC-05 disconnect")
                return item
        time.sleep(0.01)  # simulate a read timeout with no data available
        return b""

    def close(self) -> None:
        self.closed = True


def _controller(fake_port, **kwargs):
    movement = LoggingMovementController()
    gate = MovementSafetyGate(movement)
    ctrl = HC05Controller(
        gate, port="fake", baudrate=9600,
        serial_factory=lambda port, baud, timeout: fake_port,
        **kwargs,
    )
    return ctrl, gate, movement


def _wait_until(predicate, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# 1-5: each command byte -> the correct motion request
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("byte,expected", [
    (b"F", "forward"),
    (b"B", "backward"),
    (b"L", "left"),
    (b"R", "right"),
    (b"S", "stop"),
    (b"G", "stop"),
])
def test_command_byte_dispatches_to_the_correct_action(byte, expected):
    port = _FakeSerialPort()
    ctrl, gate, movement = _controller(port)
    ctrl.start()
    try:
        port.push(byte)
        assert _wait_until(lambda: movement.last_command == expected)
    finally:
        ctrl.stop()


def test_lowercase_command_byte_is_normalised_and_still_dispatches():
    port = _FakeSerialPort()
    ctrl, gate, movement = _controller(port)
    ctrl.start()
    try:
        port.push(b"f")
        assert _wait_until(lambda: movement.last_command == "forward")
    finally:
        ctrl.stop()


# ---------------------------------------------------------------------------
# 16: malformed command -> no unsafe movement
# ---------------------------------------------------------------------------

def test_unrecognised_byte_causes_no_movement_call_at_all():
    port = _FakeSerialPort()
    ctrl, gate, movement = _controller(port)
    ctrl.start()
    try:
        port.push(b"Z")  # not in the command map
        time.sleep(0.2)
        assert movement.call_count == 0, "an unrecognised byte must never fall back to any movement action"
    finally:
        ctrl.stop()


def test_whitespace_and_non_ascii_noise_is_ignored_silently():
    port = _FakeSerialPort()
    ctrl, gate, movement = _controller(port)
    ctrl.start()
    try:
        port.push(b"\n")
        port.push(b"\r")
        port.push(bytes([0xFF]))
        time.sleep(0.2)
        assert movement.call_count == 0
    finally:
        ctrl.stop()


def test_custom_command_map_overrides_the_default():
    port = _FakeSerialPort()
    ctrl, gate, movement = _controller(port, command_map={"1": "forward", "0": "stop"})
    ctrl.start()
    try:
        port.push(b"1")
        assert _wait_until(lambda: movement.last_command == "forward")
        port.push(b"F")  # the DEFAULT forward byte -- must NOT dispatch under a custom map
        time.sleep(0.2)
        assert movement.call_count == 1, "a byte outside the custom map must be ignored, not fall back to the default map"
    finally:
        ctrl.stop()


# ---------------------------------------------------------------------------
# 17: HC-05 disconnect -> defensive stop, no crash, keeps trying
# ---------------------------------------------------------------------------

def test_read_error_triggers_a_defensive_stop(monkeypatch):
    port = _FakeSerialPort()
    ctrl, gate, movement = _controller(port)
    monkeypatch.setattr(HC05Controller, "RECONNECT_BACKOFF_S", 0.01)
    ctrl.start()
    try:
        port.push(b"F")
        assert _wait_until(lambda: movement.last_command == "forward")

        port.push_read_error()
        assert _wait_until(lambda: movement.last_command == "stop")
    finally:
        ctrl.stop()


def test_read_error_triggers_a_genuine_reconnect_not_just_a_retry_on_the_same_object(monkeypatch):
    """Regression guard: retrying read() on the SAME broken port object
    would safely keep detecting the failure forever but never recover --
    the reader must actually close the stale port and open a genuinely
    new one via the factory."""
    port1 = _FakeSerialPort()
    port2 = _FakeSerialPort()
    ports = [port1, port2]
    factory_calls = []

    def factory(p, b, t):
        factory_calls.append(1)
        return ports.pop(0)

    movement = LoggingMovementController()
    gate = MovementSafetyGate(movement)
    ctrl = HC05Controller(gate, port="fake", serial_factory=factory)
    monkeypatch.setattr(HC05Controller, "RECONNECT_BACKOFF_S", 0.01)

    ctrl.start()
    try:
        assert len(factory_calls) == 1  # initial connection via start()
        port1.push_read_error()
        assert _wait_until(lambda: movement.last_command == "stop")
        assert _wait_until(lambda: len(factory_calls) == 2), "must call the factory again to reconnect"
        assert ctrl._ser is port2, "must have swapped to the genuinely new port object"

        # Prove the NEW port is actually the one being read from now.
        port2.push(b"F")
        assert _wait_until(lambda: movement.last_command == "forward")
    finally:
        ctrl.stop()


def test_reader_thread_survives_a_read_error_and_keeps_processing_commands(monkeypatch):
    """A single transient read error must not kill the reader thread --
    it must keep trying (bounded backoff), so a reconnect can recover
    without restarting the whole controller."""
    port = _FakeSerialPort()
    ctrl, gate, movement = _controller(port)
    monkeypatch.setattr(HC05Controller, "RECONNECT_BACKOFF_S", 0.01)
    ctrl.start()
    try:
        port.push_read_error()
        assert _wait_until(lambda: movement.last_command == "stop")
        assert ctrl.is_running, "reader thread must still be alive after a transient read error"

        port.push(b"F")
        assert _wait_until(lambda: movement.last_command == "forward"), (
            "controller must keep processing commands after recovering from a read error"
        )
    finally:
        ctrl.stop()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_stop_closes_the_serial_port_and_joins_the_thread():
    port = _FakeSerialPort()
    ctrl, gate, movement = _controller(port)
    ctrl.start()
    assert ctrl.is_running
    ctrl.stop()
    assert not ctrl.is_running
    assert port.closed is True


def test_start_is_idempotent_does_not_spawn_a_second_thread():
    port = _FakeSerialPort()
    ctrl, gate, movement = _controller(port)
    ctrl.start()
    try:
        first_thread = ctrl._thread
        ctrl.start()
        assert ctrl._thread is first_thread
    finally:
        ctrl.stop()


def test_stop_before_start_is_a_safe_noop():
    port = _FakeSerialPort()
    ctrl, gate, movement = _controller(port)
    ctrl.stop()  # must not raise
    assert not ctrl.is_running


# ---------------------------------------------------------------------------
# 10: HC-05 remains logically active (thread alive, commands processed)
# regardless of voice-suppression state.
# ---------------------------------------------------------------------------

def test_controller_keeps_reading_and_dispatching_while_gate_is_suppressed():
    """The reader thread and serial connection are entirely independent
    of the gate's suppression state -- HC-05 stays "logically active"
    (reading, parsing, forwarding requests) even while every request is
    being dropped by the gate. This is what "remains connected, keeps
    running" actually means in code: nothing about suppression touches
    HC05Controller at all."""
    port = _FakeSerialPort()
    ctrl, gate, movement = _controller(port)
    ctrl.start()
    try:
        gate.suppress_and_stop()
        assert ctrl.is_running, "HC05Controller must still be running while voice interaction is active"

        port.push(b"F")
        time.sleep(0.2)
        assert movement.call_count == 1, "the gate's own stop() from suppress_and_stop() is the only call recorded"
        assert movement.last_command == "stop", "the forward byte was received and dispatched, then correctly dropped by the gate"
    finally:
        ctrl.stop()
