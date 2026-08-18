"""Unit tests for voice/audio/manager.py's mic-disconnect recovery
(Milestone 7), using a fake PyAudio -- no real audio hardware needed.

Two real-hardware findings this guards against, in order of discovery:

1. A physical USB unplug produces OSError("[Errno -9988] Stream closed")
   on every read. frames()'s original retry loop caught that and retried
   forever at ~31ms intervals, never recovering even after the device was
   reconnected, and because that inner wait didn't block on the real stop
   event, it also prevented shutdown from propagating.

2. The first fix (bounded retry -> reopen via a fresh pyaudio.PyAudio() ->
   give up loudly) was itself found to be unsafe: repeatedly tearing down
   and recreating the PyAudio host instance while the device was still
   physically absent killed the whole process with no Python-level
   exception at all (consistent with a native PortAudio/ALSA crash, not a
   hang -- confirmed via journalctl -k timeline correlation). The current
   design never calls pyaudio.PyAudio() again after the initial start():
   recovery only reopens a *stream* on the existing host instance, gated
   behind a PortAudio-free presence check via /proc/asound/cards so
   PyAudio is never touched while the device is confirmed absent.

These tests are not a substitute for the physical unplug/replug
validation documented separately for this milestone.
"""

from __future__ import annotations

import threading
import time
import types

import pytest

from voice.audio.manager import AudioManager, MicrophoneRecoveryFailedError
from voice.config import AudioConfig


class _FakeStream:
    def __init__(self, owner):
        self.owner = owner
        self.closed = False
        self._active = True

    def read(self, n, exception_on_overflow=False):
        self.owner.read_calls += 1
        if self.owner.broken:
            raise OSError(-9988, "Stream closed")
        return b"\x00\x00" * n

    def is_active(self):
        return self._active

    def stop_stream(self):
        self._active = False

    def start_stream(self):
        self._active = True

    def close(self):
        self.closed = True


class _FakePyAudio:
    """Represents ONE PyAudio host instance. Its identity matters: the
    fix under test must never construct a second one during recovery."""

    def __init__(self, owner):
        self.owner = owner
        self.instance_id = owner.pyaudio_init_calls  # snapshot at construction time

    def open(self, **kwargs):
        self.owner.open_calls += 1
        if self.owner.hang_on_open_s:
            time.sleep(self.owner.hang_on_open_s)
        if self.owner.fail_reopen:
            raise OSError("device busy")
        stream = _FakeStream(self.owner)
        self.owner.last_stream = stream
        return stream

    def terminate(self):
        pass


class _Owner:
    """Shared mutable state the fakes reference, so a test can flip
    `broken`/`fail_reopen`/`cards_present` mid-run to simulate
    unplug/replug timing."""

    def __init__(self):
        self.broken = False
        self.fail_reopen = False
        self.open_calls = 0
        self.read_calls = 0
        self.last_stream = None
        self.hang_on_open_s = 0.0  # simulates pa.open() blocking
        self.pyaudio_init_calls = 0  # how many times pyaudio.PyAudio() itself was constructed
        self.cards_present = True  # whether /proc/asound/cards (faked) currently shows the USB card


@pytest.fixture
def fake_pyaudio(monkeypatch, tmp_path):
    owner = _Owner()

    def _construct():
        owner.pyaudio_init_calls += 1
        return _FakePyAudio(owner)

    fake_module = types.SimpleNamespace(paInt16=8, PyAudio=_construct)
    monkeypatch.setattr("voice.audio.manager.pyaudio", fake_module)

    # Fake /proc/asound/cards as a real temp file so _mic_physically_present()
    # exercises its real file-reading code path, not a mocked shortcut.
    cards_path = tmp_path / "fake_asound_cards"

    def _write_cards():
        if owner.cards_present:
            cards_path.write_text(" 1 [Device ]: USB-Audio - Fake USB Mic\n                      Fake USB Mic at usb-1.1, full speed\n")
        else:
            cards_path.write_text(" 0 [Other ]: not-usb - Some Other Card\n")

    _write_cards()
    owner._write_cards = _write_cards  # let tests re-trigger a write after flipping cards_present
    monkeypatch.setattr(AudioManager, "ASOUND_CARDS_PATH", str(cards_path))

    return owner


def _fast_config():
    return AudioConfig(input_device_index=0, sample_rate=16000, channels=1, frame_duration_ms=5)


def _make_manager(monkeypatch, max_errors=3, max_reopen=3, backoff=(0.02, 0.02, 0.02), reopen_timeout=3.0):
    monkeypatch.setattr(AudioManager, "MAX_CONSECUTIVE_READ_ERRORS", max_errors)
    monkeypatch.setattr(AudioManager, "MAX_REOPEN_ATTEMPTS", max_reopen)
    monkeypatch.setattr(AudioManager, "REOPEN_BACKOFF_S", backoff)
    monkeypatch.setattr(AudioManager, "REOPEN_TIMEOUT_S", reopen_timeout)
    return AudioManager(_fast_config())


# ---------------------------------------------------------------------------
def test_brief_errors_below_threshold_recover_without_reopening(fake_pyaudio, monkeypatch):
    """A handful of transient read errors -- fewer than the threshold --
    must NOT trigger a stream reopen (or touch PyAudio) at all."""
    audio = _make_manager(monkeypatch, max_errors=5)
    audio.start()
    assert fake_pyaudio.pyaudio_init_calls == 1
    assert fake_pyaudio.open_calls == 1

    fake_pyaudio.broken = True

    def _unbreak_after_a_few_reads():
        while fake_pyaudio.read_calls < 2:
            time.sleep(0.005)
        fake_pyaudio.broken = False

    threading.Thread(target=_unbreak_after_a_few_reads, daemon=True).start()

    frame = next(audio.frames())

    assert frame is not None
    assert fake_pyaudio.open_calls == 1, "must not have attempted a reopen for a brief, sub-threshold error"
    assert fake_pyaudio.pyaudio_init_calls == 1
    audio.stop()


def test_recovery_waits_for_presence_before_touching_pyaudio(fake_pyaudio, monkeypatch):
    """While /proc/asound/cards shows no USB card, recovery must not call
    pa.open() (or pyaudio.PyAudio()) at all -- only poll presence and back
    off. Once presence is confirmed, it reopens a stream on the SAME
    PyAudio host instance (pyaudio.PyAudio() is never called a 2nd time)."""
    audio = _make_manager(monkeypatch, max_errors=2, max_reopen=5, backoff=(0.02,) * 5)
    audio.start()
    assert fake_pyaudio.pyaudio_init_calls == 1

    fake_pyaudio.broken = True
    fake_pyaudio.cards_present = False
    fake_pyaudio._write_cards()

    def _replug_after_two_presence_checks():
        # Let a couple of recovery attempts see "absent" first, confirming
        # they skip PyAudio entirely, then simulate the device reappearing.
        time.sleep(0.05)
        fake_pyaudio.cards_present = True
        fake_pyaudio._write_cards()
        fake_pyaudio.broken = False  # the reconnected device's stream reads succeed again

    threading.Thread(target=_replug_after_two_presence_checks, daemon=True).start()

    frame = next(audio.frames())

    assert frame is not None
    assert fake_pyaudio.pyaudio_init_calls == 1, "pyaudio.PyAudio() must never be called again during recovery"
    assert fake_pyaudio.open_calls == 2, "expected exactly 1 recovery-time pa.open() (initial start() + 1 recovery)"
    audio.stop()


def test_exhausted_recovery_raises_visibly_not_hang(fake_pyaudio, monkeypatch):
    """If the mic never comes back (cards never show it present), frames()
    must raise a clear exception after bounded attempts -- not hang."""
    audio = _make_manager(monkeypatch, max_errors=2, max_reopen=3, backoff=(0.01,) * 3)
    audio.start()
    fake_pyaudio.broken = True
    fake_pyaudio.cards_present = False
    fake_pyaudio._write_cards()

    t0 = time.monotonic()
    with pytest.raises(MicrophoneRecoveryFailedError):
        next(audio.frames())
    elapsed = time.monotonic() - t0

    assert elapsed < 2.0
    assert fake_pyaudio.pyaudio_init_calls == 1, "must never reconstruct PyAudio while absent"
    assert fake_pyaudio.open_calls == 1, "must never even attempt pa.open() while absent -- only the initial start()"
    audio.stop()


def test_stop_during_initial_retry_phase_returns_promptly(fake_pyaudio, monkeypatch):
    """Calling stop() while frames() is in the fast pre-recovery retry
    phase must make the generator return quickly, not block until the
    error threshold or any timeout."""
    audio = _make_manager(monkeypatch, max_errors=1000)  # never reach recovery in this test
    audio.start()
    fake_pyaudio.broken = True

    result = {}

    def _consume():
        t0 = time.monotonic()
        try:
            for _ in audio.frames():
                pass
        finally:
            result["elapsed"] = time.monotonic() - t0

    t = threading.Thread(target=_consume, daemon=True)
    t.start()
    time.sleep(0.05)
    audio.stop()
    t.join(timeout=2.0)

    assert not t.is_alive(), "frames() did not return after stop() -- this is the real hang bug"
    assert result["elapsed"] < 1.0


def test_stop_during_recovery_backoff_returns_promptly(fake_pyaudio, monkeypatch):
    """Calling stop() while frames() is backing off between reopen
    attempts must interrupt the wait immediately, not finish out the
    full backoff duration."""
    audio = _make_manager(monkeypatch, max_errors=1, max_reopen=5, backoff=(5.0,) * 5)
    audio.start()
    fake_pyaudio.broken = True
    fake_pyaudio.cards_present = False
    fake_pyaudio._write_cards()

    result = {}

    def _consume():
        t0 = time.monotonic()
        try:
            for _ in audio.frames():
                pass
        except MicrophoneRecoveryFailedError:
            pass
        finally:
            result["elapsed"] = time.monotonic() - t0

    t = threading.Thread(target=_consume, daemon=True)
    t.start()
    time.sleep(0.1)  # let it enter the 5s backoff
    audio.stop()
    t.join(timeout=2.0)

    assert not t.is_alive(), "frames() did not return promptly during a long backoff wait"
    assert result["elapsed"] < 1.0, "stop() should interrupt the backoff wait, not wait out the full 5s"


def test_stop_during_presence_polling_returns_promptly(fake_pyaudio, monkeypatch):
    """stop() called while recovery is repeatedly polling presence (device
    still absent) must also return promptly -- not just during backoff."""
    audio = _make_manager(monkeypatch, max_errors=1, max_reopen=10, backoff=(0.05,) * 10)
    audio.start()
    fake_pyaudio.broken = True
    fake_pyaudio.cards_present = False
    fake_pyaudio._write_cards()

    result = {}

    def _consume():
        t0 = time.monotonic()
        try:
            for _ in audio.frames():
                pass
        except MicrophoneRecoveryFailedError:
            pass
        finally:
            result["elapsed"] = time.monotonic() - t0

    t = threading.Thread(target=_consume, daemon=True)
    t.start()
    time.sleep(0.15)  # let a couple of presence-check cycles happen
    audio.stop()
    t.join(timeout=2.0)

    assert not t.is_alive()
    assert result["elapsed"] < 1.0
    assert fake_pyaudio.pyaudio_init_calls == 1
    assert fake_pyaudio.open_calls == 1


def test_reopen_that_fails_to_open_is_retried_up_to_the_bound(fake_pyaudio, monkeypatch):
    """A reopen attempt where pa.open() itself raises (not just a
    subsequent read) must also count against the bounded attempts, not
    loop forever or skip counting -- while never reconstructing PyAudio."""
    audio = _make_manager(monkeypatch, max_errors=1, max_reopen=3, backoff=(0.01,) * 3)
    audio.start()
    fake_pyaudio.broken = True
    fake_pyaudio.fail_reopen = True  # every reopen attempt's open() itself raises
    fake_pyaudio.cards_present = True  # device visible, but open() keeps failing anyway

    with pytest.raises(MicrophoneRecoveryFailedError):
        next(audio.frames())

    assert fake_pyaudio.open_calls == 1 + 3
    assert fake_pyaudio.pyaudio_init_calls == 1
    audio.stop()


def test_reopen_that_hangs_is_bounded_by_timeout_not_left_hanging(fake_pyaudio, monkeypatch):
    """Even once presence is confirmed, pa.open() can still hang in
    principle -- this must still be bounded by REOPEN_TIMEOUT_S, not left
    to hang, and must not reconstruct PyAudio to work around it."""
    audio = _make_manager(
        monkeypatch, max_errors=1, max_reopen=3,
        backoff=(0.01, 0.01, 0.01), reopen_timeout=0.1,
    )
    audio.start()
    fake_pyaudio.broken = True
    fake_pyaudio.cards_present = True  # present the whole time -- the hang is inside open() itself
    fake_pyaudio.hang_on_open_s = 5.0  # far longer than reopen_timeout=0.1s

    def _stop_hanging_after_first_attempt():
        while fake_pyaudio.open_calls < 2:
            time.sleep(0.005)
        fake_pyaudio.hang_on_open_s = 0.0
        fake_pyaudio.broken = False  # the successfully reopened stream's reads must also succeed

    threading.Thread(target=_stop_hanging_after_first_attempt, daemon=True).start()

    t0 = time.monotonic()
    frame = next(audio.frames())
    elapsed = time.monotonic() - t0

    assert frame is not None
    assert elapsed < 2.0, f"took {elapsed:.2f}s -- the hang was not bounded"
    assert fake_pyaudio.pyaudio_init_calls == 1, "must never reconstruct PyAudio, even to work around a hang"
    audio.stop()


def test_presence_check_never_blocks_configs_without_a_captured_signature(fake_pyaudio, monkeypatch):
    """If no USB-Audio signature was ever captured (e.g. start() somehow
    ran before /proc/asound/cards had the expected content), the presence
    check must not permanently block recovery -- it should fall through
    to attempting the reopen directly."""
    audio = _make_manager(monkeypatch, max_errors=1, max_reopen=2, backoff=(0.01,) * 2)
    audio.start()
    audio._usb_audio_signature = None  # simulate "never captured"

    fake_pyaudio.broken = True

    def _unbreak_soon():
        while fake_pyaudio.open_calls < 2:
            time.sleep(0.005)
        fake_pyaudio.broken = False

    threading.Thread(target=_unbreak_soon, daemon=True).start()

    frame = next(audio.frames())
    assert frame is not None
    audio.stop()
