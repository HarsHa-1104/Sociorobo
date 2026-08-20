"""Unit/integration tests for voice/audio/pw_capture.py (Phase 3 capture
primitive -- not wired into the production pipeline, see that module's
docstring for why). Uses a real fake subprocess
(tests/fake_pw_record_stub.py), the same real-pipe-not-mocked approach
already used for PersistentPiperTTS's tests.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from voice.audio.pw_capture import CaptureStartError, PipeWireCapture

_FAKE_PW_RECORD = str(Path(__file__).resolve().parent / "fake_pw_record_stub.py")


def _capture(target="some-bluez-source", rate=16000, channels=1) -> PipeWireCapture:
    return PipeWireCapture(target, rate, channels, binary_path=_FAKE_PW_RECORD)


def test_read_returns_exact_number_of_bytes_from_real_pipe():
    cap = _capture()
    cap.start()
    try:
        data = cap.read(256)
        assert data is not None
        assert len(data) == 256
    finally:
        cap.stop()


def test_read_can_be_called_repeatedly_and_stays_in_sync():
    """Reading two chunks in sequence must not interleave/lose bytes --
    the underlying stream is continuous, not framed."""
    cap = _capture()
    cap.start()
    try:
        first = cap.read(64)
        second = cap.read(64)
        assert first is not None and second is not None
        assert len(first) == 64
        assert len(second) == 64
    finally:
        cap.stop()


def test_is_alive_true_while_running_false_after_stop():
    cap = _capture()
    cap.start()
    assert cap.is_alive() is True
    cap.stop()
    assert cap.is_alive() is False


def test_stop_actually_terminates_the_process_no_leak():
    cap = _capture()
    cap.start()
    proc = cap._proc
    cap.stop()
    time.sleep(0.2)
    assert proc.poll() is not None, "pw-record process must actually be terminated after stop()"


def test_stop_before_start_is_a_safe_noop():
    cap = _capture()
    cap.stop()  # must not raise


def test_read_before_start_returns_none_not_raise():
    cap = _capture()
    assert cap.read(64) is None


def test_read_returns_none_on_immediate_process_death_not_hang():
    """The transport dying (e.g. Bluetooth connection refused) must
    surface as a clean None, promptly -- not hang forever or raise."""
    cap = _capture(target="__DIE_IMMEDIATELY__")
    cap.start()
    try:
        t0 = time.monotonic()
        data = cap.read(64, timeout_s=1.0)
        elapsed = time.monotonic() - t0
        assert data is None
        assert elapsed < 1.0, "should return promptly on EOF, not wait out the full timeout"
    finally:
        cap.stop()


def test_read_times_out_promptly_when_source_produces_nothing_not_hang_forever():
    """A hung/silent capture source (e.g. a Bluetooth link that connected
    but never delivers audio) must not be able to hang the caller
    indefinitely -- this is the same class of bug Milestone 7 fixed for
    the USB/PyAudio path, applied here to the PipeWire path."""
    cap = _capture(target="__HANG__")
    cap.start()
    try:
        t0 = time.monotonic()
        data = cap.read(64, timeout_s=0.3)
        elapsed = time.monotonic() - t0
        assert data is None
        assert elapsed < 1.0, f"took {elapsed:.2f}s -- read() was not bounded by timeout_s"
    finally:
        cap.stop()


def test_missing_binary_raises_capture_start_error_not_generic_exception():
    cap = PipeWireCapture("some-target", 16000, 1, binary_path="/nonexistent/pw-record-binary")
    with pytest.raises(CaptureStartError):
        cap.start()


def test_start_is_idempotent_does_not_spawn_a_second_process():
    cap = _capture()
    cap.start()
    try:
        first_proc = cap._proc
        cap.start()  # calling start() again must not replace/leak the running process
        assert cap._proc is first_proc
    finally:
        cap.stop()


def test_command_line_uses_expected_pw_record_flags():
    """Regression guard for the exact flag set -- s16/raw/mono/rate must
    match what AudioManager's pipeline expects, matching the symmetric
    pattern already proven in voice/tts/pipewire_playback.py's pw-play
    invocation."""
    import subprocess
    captured = {}
    real_popen = subprocess.Popen

    def _spy_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return real_popen(cmd, **kwargs)

    import unittest.mock as mock
    with mock.patch("subprocess.Popen", side_effect=_spy_popen):
        cap = _capture(target="my-bt-mic-source", rate=16000, channels=1)
        cap.start()
        cap.stop()

    cmd = captured["cmd"]
    assert cmd[0] == _FAKE_PW_RECORD
    assert "--target" in cmd and cmd[cmd.index("--target") + 1] == "my-bt-mic-source"
    assert "--rate" in cmd and cmd[cmd.index("--rate") + 1] == "16000"
    assert "--channels" in cmd and cmd[cmd.index("--channels") + 1] == "1"
    assert "--format" in cmd and cmd[cmd.index("--format") + 1] == "s16"
    assert "--raw" in cmd
