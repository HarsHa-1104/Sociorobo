"""Integration tests for the full manual-control chain:

    HC05Controller -> MovementSafetyGate -> MovementController
                              ^
                              |
                   ReferenceHumanFollowerServer (voice/ipc/server_stub.py,
                   REUSED UNMODIFIED) <--- real Unix socket --- HumanFollowerLink
                                                                (voice/ipc/client.py,
                                                                 the EXACT class VoiceManager
                                                                 uses -- not a fake)

This is deliberately NOT mocked at the IPC boundary -- a real Unix domain
socket, a real ReferenceHumanFollowerServer, and the real HumanFollowerLink
client class are used, so these tests exercise the actual wire protocol
VoiceManager depends on, not an approximation of it. HC-05 input is still
faked (no real serial hardware), and no VoiceManager instance is
constructed (that's covered by tests/test_voice_manager.py already) --
this file is specifically about what happens on the OTHER end of the
socket from VoiceManager's perspective, i.e. what motion/ actually does
with PAUSE_REQUEST/VOICE_SESSION_COMPLETE/the watchdog.
"""

from __future__ import annotations

import threading
import time

import pytest

from motion.hc05_controller import HC05Controller
from motion.movement_controller import LoggingMovementController
from motion.safety_gate import MovementSafetyGate
from voice.config import IPCConfig, WatchdogConfig
from voice.ipc.client import HumanFollowerLink
from voice.ipc.server_stub import ReferenceHumanFollowerServer


class _FakeSerialPort:
    def __init__(self):
        self._queue: list = []
        self._lock = threading.Lock()

    def push(self, byte: bytes) -> None:
        with self._lock:
            self._queue.append(byte)

    def read(self, size: int = 1) -> bytes:
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
        time.sleep(0.01)
        return b""

    def close(self) -> None:
        pass


def _wait_until(predicate, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def rig(tmp_path):
    """A complete, real, running motion stack + a client exactly like the
    one VoiceManager uses. Everything is torn down after the test."""
    socket_path = str(tmp_path / "motion_test.sock")
    ipc_config = IPCConfig(
        socket_path=socket_path,
        connect_timeout_s=1.0,
        message_timeout_s=1.0,
        heartbeat_interval_s=0.1,
        pause_confirm_timeout_s=1.0,
    )
    watchdog_config = WatchdogConfig(
        max_paused_duration_s=100.0,  # overridden per-test where the watchdog itself is under test
        heartbeat_timeout_s=100.0,
    )

    movement = LoggingMovementController()
    gate = MovementSafetyGate(movement)
    port = _FakeSerialPort()
    hc05 = HC05Controller(gate, port="fake", serial_factory=lambda p, b, t: port)
    hc05.start()

    watchdog_forced_resume_reasons = []

    server = ReferenceHumanFollowerServer(
        ipc_config, watchdog_config,
        on_pause=lambda: (gate.suppress_and_stop(), True)[1],
        on_resume=gate.release,
        on_watchdog_forced_resume=lambda reason: watchdog_forced_resume_reasons.append(reason),
    )
    server.start()

    class Rig:
        pass

    r = Rig()
    r.movement = movement
    r.gate = gate
    r.hc05 = hc05
    r.port = port
    r.server = server
    r.ipc_config = ipc_config
    r.watchdog_config = watchdog_config
    r.watchdog_forced_resume_reasons = watchdog_forced_resume_reasons

    def make_client() -> HumanFollowerLink:
        return HumanFollowerLink(ipc_config)

    r.make_client = make_client

    yield r

    hc05.stop()
    server.stop()


# ---------------------------------------------------------------------------
# 6/7/8: wake word during any kind of motion -> immediate stop
# ---------------------------------------------------------------------------

def test_wake_word_during_forward_motion_stops_immediately(rig):
    rig.port.push(b"F")
    assert _wait_until(lambda: rig.movement.last_command == "forward")

    client = rig.make_client()
    confirmed = client.request_pause()
    assert confirmed is True
    assert rig.movement.last_command == "stop"
    client.close()


def test_wake_word_during_backward_motion_stops_immediately(rig):
    rig.port.push(b"B")
    assert _wait_until(lambda: rig.movement.last_command == "backward")

    client = rig.make_client()
    assert client.request_pause() is True
    assert rig.movement.last_command == "stop"
    client.close()


def test_wake_word_during_turning_stops_immediately(rig):
    rig.port.push(b"L")
    assert _wait_until(lambda: rig.movement.last_command == "left")

    client = rig.make_client()
    assert client.request_pause() is True
    assert rig.movement.last_command == "stop"
    client.close()


# ---------------------------------------------------------------------------
# 9/10: voice interaction suppresses movement; HC-05 stays logically active
# ---------------------------------------------------------------------------

def test_voice_interaction_suppresses_bluetooth_movement(rig):
    client = rig.make_client()
    client.request_pause()

    rig.movement.call_count = 0
    rig.port.push(b"F")
    time.sleep(0.2)
    assert rig.movement.call_count == 0, "Bluetooth movement must be suppressed during voice interaction"
    assert rig.hc05.is_running, "HC-05 reader must remain running -- 'logically active' -- during voice interaction"
    client.close()


# ---------------------------------------------------------------------------
# 11/12: completion does not auto-resume; a new command is required
# ---------------------------------------------------------------------------

def test_session_complete_releases_suppression_without_auto_resuming(rig):
    rig.port.push(b"F")
    assert _wait_until(lambda: rig.movement.last_command == "forward")

    client = rig.make_client()
    client.request_pause()
    assert rig.movement.last_command == "stop"

    client.session_complete(outcome="answered")
    assert _wait_until(lambda: rig.gate.is_suppressed is False)
    time.sleep(0.1)
    assert rig.movement.last_command == "stop", "must NOT auto-resume the forward motion that was active before the wake word"
    client.close()


def test_new_bluetooth_command_after_session_complete_moves_the_car(rig):
    client = rig.make_client()
    client.request_pause()
    client.session_complete(outcome="answered")
    assert _wait_until(lambda: rig.gate.is_suppressed is False)

    rig.port.push(b"F")
    assert _wait_until(lambda: rig.movement.last_command == "forward")
    client.close()


# ---------------------------------------------------------------------------
# 13: repeated wake word remains safely stopped
# ---------------------------------------------------------------------------

def test_repeated_wake_word_remains_safely_stopped(rig):
    client1 = rig.make_client()
    assert client1.request_pause() is True
    assert rig.movement.last_command == "stop"

    client2 = rig.make_client()
    assert client2.request_pause() is True  # second wake word before the first session completed
    assert rig.movement.last_command == "stop"
    assert rig.gate.is_suppressed is True

    client1.close()
    client2.close()


# ---------------------------------------------------------------------------
# 14: Bluetooth command / wake-word race -> stop wins
# ---------------------------------------------------------------------------

def test_bluetooth_command_racing_wake_word_never_ends_up_moving(rig):
    for _ in range(30):
        rig.movement.call_count = 0
        rig.movement.last_command = None
        barrier = threading.Barrier(2)

        def _spam_commands():
            barrier.wait()
            for _ in range(10):
                rig.port.push(b"F")

        client = rig.make_client()

        def _wake_word():
            barrier.wait()
            client.request_pause()

        t1 = threading.Thread(target=_spam_commands)
        t2 = threading.Thread(target=_wake_word)
        t1.start(); t2.start()
        t1.join(timeout=2.0); t2.join(timeout=2.0)
        assert not t1.is_alive() and not t2.is_alive()

        time.sleep(0.2)  # let the HC-05 reader thread drain any pushed bytes
        assert rig.gate.is_suppressed is True
        assert rig.movement.last_command == "stop", (
            "regardless of interleaving, the gate's own lock guarantees stop() "
            "is the last word once suppress_and_stop() has run"
        )
        client.session_complete(outcome="answered")
        _wait_until(lambda: rig.gate.is_suppressed is False)
        client.close()


# ---------------------------------------------------------------------------
# 15: voice-pipeline failure after wake word -> motors remain safely stopped
# (watchdog fires, releases suppression -- but does NOT move the car)
# ---------------------------------------------------------------------------

def test_voice_pipeline_failure_after_wake_word_leaves_car_stopped_not_moving(tmp_path):
    """Simulates VoiceManager crashing right after PAUSE_REQUEST (no
    VOICE_SESSION_COMPLETE, no HEARTBEAT ever sent). The watchdog must
    eventually force a release (per the documented policy in
    server_stub.py -- prefer releasing over staying stopped forever) --
    but "release" for manual control means only "suppression lifted", the
    car must still not move on its own."""
    socket_path = str(tmp_path / "motion_watchdog_test.sock")
    ipc_config = IPCConfig(socket_path=socket_path, connect_timeout_s=1.0,
                            message_timeout_s=1.0, heartbeat_interval_s=0.1,
                            pause_confirm_timeout_s=1.0)
    watchdog_config = WatchdogConfig(max_paused_duration_s=100.0, heartbeat_timeout_s=0.3)

    movement = LoggingMovementController()
    gate = MovementSafetyGate(movement)
    port = _FakeSerialPort()
    hc05 = HC05Controller(gate, port="fake", serial_factory=lambda p, b, t: port)
    hc05.start()

    forced_reasons = []
    server = ReferenceHumanFollowerServer(
        ipc_config, watchdog_config,
        on_pause=lambda: (gate.suppress_and_stop(), True)[1],
        on_resume=gate.release,
        on_watchdog_forced_resume=lambda reason: forced_reasons.append(reason),
    )
    server.start()

    try:
        port.push(b"F")
        assert _wait_until(lambda: movement.last_command == "forward")

        client = HumanFollowerLink(ipc_config)
        client.request_pause()
        assert movement.last_command == "stop"
        # Deliberately do NOT send HEARTBEAT or VOICE_SESSION_COMPLETE --
        # simulates VoiceManager dying immediately after the wake word.
        client.close()  # the crash itself would also drop the connection

        assert _wait_until(lambda: not gate.is_suppressed, timeout_s=3.0), "watchdog must eventually release"
        assert forced_reasons, "on_watchdog_forced_resume must have fired"
        assert movement.last_command == "stop", "must remain stopped -- watchdog release must never move the car"
        assert movement.call_count == 2, "exactly: forward (HC-05) + stop (wake word) -- nothing else"
    finally:
        hc05.stop()
        server.stop()


# ---------------------------------------------------------------------------
# 16: malformed Bluetooth command -> no unsafe movement (also covered at
# the unit level in test_hc05_controller.py; here confirmed through the
# full stack including an active voice interaction).
# ---------------------------------------------------------------------------

def test_malformed_command_during_voice_interaction_causes_no_movement(rig):
    client = rig.make_client()
    client.request_pause()
    rig.movement.call_count = 0

    rig.port.push(b"Q")  # not a recognised command even outside suppression
    time.sleep(0.2)
    assert rig.movement.call_count == 0
    client.close()


# ---------------------------------------------------------------------------
# 17: HC-05 disconnect -> safe stop, independent of voice-interaction state
# ---------------------------------------------------------------------------

def test_hc05_disconnect_during_manual_control_stops_safely(rig):
    rig.port.push(b"F")
    assert _wait_until(lambda: rig.movement.last_command == "forward")

    class _DyingPort:
        def read(self, size=1):
            raise OSError("simulated disconnect")

        def close(self):
            pass

    rig.hc05._ser = _DyingPort()  # simulate the link dying mid-operation
    assert _wait_until(lambda: rig.movement.last_command == "stop")


# ---------------------------------------------------------------------------
# Bluetooth/HC-05 stays connected and running throughout an entire
# realistic session sequence -- an end-to-end sanity check combining
# several of the above in the order a real session actually happens.
# ---------------------------------------------------------------------------

def test_full_realistic_session_sequence(rig):
    # 1. Manual driving.
    rig.port.push(b"F")
    assert _wait_until(lambda: rig.movement.last_command == "forward")

    # 2. Wake word.
    client = rig.make_client()
    assert client.request_pause() is True
    assert rig.movement.last_command == "stop"
    assert rig.hc05.is_running, "HC-05 must still be connected/running"

    # 3. A manual command sent DURING the voice interaction must be ignored.
    rig.movement.call_count = 0
    rig.port.push(b"R")
    time.sleep(0.2)
    assert rig.movement.call_count == 0

    # 4. Voice interaction completes.
    client.session_complete(outcome="answered")
    assert _wait_until(lambda: rig.gate.is_suppressed is False)
    client.close()

    # 5. Car remains stopped -- no auto-resume.
    time.sleep(0.1)
    assert rig.movement.last_command == "stop"

    # 6. A new manual command now works.
    rig.port.push(b"F")
    assert _wait_until(lambda: rig.movement.last_command == "forward")
