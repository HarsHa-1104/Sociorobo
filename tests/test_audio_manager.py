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


FAKE_USB_MIC = {
    "name": "Fake USB Mic: USB Audio (hw:0,0)",
    "maxInputChannels": 1, "maxOutputChannels": 0, "defaultSampleRate": 16000.0,
}
DIFFERENT_USB_MIC = {
    "name": "Different USB Mic: USB Audio (hw:2,0)",
    "maxInputChannels": 1, "maxOutputChannels": 0, "defaultSampleRate": 16000.0,
}


class _FakePyAudio:
    """Represents ONE PyAudio host instance. Its identity matters: Tier 1
    recovery under test must never construct a second one; Tier 2
    (rediscovery) deliberately does, exactly once per attempt.

    get_device_count()/get_device_info_by_index() read owner.device_list
    live (not a snapshot frozen at construction time) -- unlike real
    PortAudio, which DOES freeze its enumeration at construction (see
    voice/audio/manager.py's _attempt_device_rediscovery docstring for why
    that matters and why this class always constructs a genuinely new host
    for Tier 2 rather than reusing one). Modeling live reads here is fine
    because nothing in the code under test ever reuses a stale fake host
    for discovery -- only ever a freshly-constructed one -- so there is no
    staleness behavior for this fake to get wrong.
    """

    def __init__(self, owner):
        self.owner = owner
        self.instance_id = owner.pyaudio_init_calls  # snapshot at construction time

    def get_device_count(self):
        return len(self.owner.device_list)

    def get_device_info_by_index(self, i):
        return self.owner.device_list[i]

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
        self.owner.terminate_calls += 1


class _Owner:
    """Shared mutable state the fakes reference, so a test can flip
    `broken`/`fail_reopen`/`cards_present` mid-run to simulate
    unplug/replug timing."""

    def __init__(self):
        self.broken = False
        self.fail_reopen = False
        self.open_calls = 0
        self.read_calls = 0
        self.terminate_calls = 0
        self.last_stream = None
        self.hang_on_open_s = 0.0  # simulates pa.open() blocking
        self.pyaudio_init_calls = 0  # how many times pyaudio.PyAudio() itself was constructed
        self.cards_present = True  # whether /proc/asound/cards (faked) currently shows the ORIGINAL USB card
        self.different_usb_present = False  # whether it instead shows a DIFFERENT USB card (Tier 2 territory)
        self.device_list = [dict(FAKE_USB_MIC)]  # what discovery currently finds via PyAudio enumeration


@pytest.fixture
def fake_pyaudio(monkeypatch, tmp_path):
    owner = _Owner()

    def _construct():
        owner.pyaudio_init_calls += 1
        return _FakePyAudio(owner)

    fake_module = types.SimpleNamespace(paInt16=8, PyAudio=_construct)
    monkeypatch.setattr("voice.audio.manager.pyaudio", fake_module)
    monkeypatch.setattr("voice.audio.discovery.pyaudio", fake_module)

    # Fake /proc/asound/cards as a real temp file so _mic_physically_present()
    # /_any_usb_audio_card_present() exercise their real file-reading code
    # path, not a mocked shortcut.
    cards_path = tmp_path / "fake_asound_cards"

    def _write_cards():
        if owner.cards_present:
            cards_path.write_text(" 1 [Device ]: USB-Audio - Fake USB Mic\n                      Fake USB Mic at usb-1.1, full speed\n")
        elif owner.different_usb_present:
            cards_path.write_text(" 2 [Other  ]: USB-Audio - Different USB Mic\n                      Different USB Mic at usb-2.1, full speed\n")
        else:
            cards_path.write_text(" 0 [Other ]: not-usb - Some Other Card\n")

    _write_cards()
    owner._write_cards = _write_cards  # let tests re-trigger a write after flipping cards_present
    monkeypatch.setattr(AudioManager, "ASOUND_CARDS_PATH", str(cards_path))

    # Discovery also queries pw-dump for Bluetooth/PipeWire input candidates
    # -- irrelevant to these PyAudio/ALSA-focused tests, so make it a no-op
    # (empty result) rather than letting it hit a real subprocess.
    monkeypatch.setattr("voice.audio.discovery._run_pw_dump", lambda: [])

    return owner


def _fast_config(**overrides):
    kwargs = dict(sample_rate=16000, channels=1, frame_duration_ms=5)
    kwargs.update(overrides)
    return AudioConfig(**kwargs)


def _make_manager(monkeypatch, max_errors=3, max_reopen=3, backoff=(0.02, 0.02, 0.02), reopen_timeout=3.0, **config_overrides):
    monkeypatch.setattr(AudioManager, "MAX_CONSECUTIVE_READ_ERRORS", max_errors)
    monkeypatch.setattr(AudioManager, "MAX_REOPEN_ATTEMPTS", max_reopen)
    monkeypatch.setattr(AudioManager, "REOPEN_BACKOFF_S", backoff)
    monkeypatch.setattr(AudioManager, "REOPEN_TIMEOUT_S", reopen_timeout)
    return AudioManager(_fast_config(**config_overrides))


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
    loop forever or skip counting -- while never reconstructing PyAudio.

    Uses microphone_mode="pinned" specifically to isolate Tier 1 mechanics
    from Tier 2 (rediscovery, tested separately below) -- pinned mode never
    attempts Tier 2 at all, matching its "never substitute a different
    device" contract."""
    audio = _make_manager(monkeypatch, max_errors=1, max_reopen=3, backoff=(0.01,) * 3,
                           microphone_mode="pinned", microphone_pin="Fake USB Mic")
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


def test_start_resolves_device_via_discovery_by_stable_id_not_raw_index(fake_pyaudio, monkeypatch):
    """Plug-and-play Phase 2: start() must resolve the microphone through
    discovery/selection (stable_id), not a hand-set input_device_index --
    the default config has no index or pin at all."""
    audio = _make_manager(monkeypatch)
    audio.start()
    assert audio._resolved_stable_id == "Fake USB Mic"
    assert audio._resolved_pyaudio_index == 0
    audio.stop()


def test_start_pinned_mode_fails_clearly_when_pin_absent(fake_pyaudio, monkeypatch):
    """A pinned microphone that isn't currently discoverable must raise
    MicrophoneUnavailableError at start() -- never silently fall back to
    whatever else discovery found."""
    from voice.audio.manager import MicrophoneUnavailableError

    audio = _make_manager(monkeypatch, microphone_mode="pinned", microphone_pin="Nonexistent Mic")
    with pytest.raises(MicrophoneUnavailableError):
        audio.start()
    assert fake_pyaudio.open_calls == 0, "must never open a stream for an unresolved pin"


def test_tier2_rediscovery_swaps_to_a_different_microphone_when_original_is_gone(fake_pyaudio, monkeypatch):
    """The core Phase 2 recovery requirement: if the original mic never
    comes back but a genuinely DIFFERENT USB microphone is now present,
    auto-mode recovery must find and switch to it -- reconstructing
    PyAudio exactly once for this (Tier 2), never silently giving up while
    a usable alternative exists."""
    audio = _make_manager(monkeypatch, max_errors=1, max_reopen=2, backoff=(0.01,) * 2,
                           microphone_mode="auto")
    audio.start()
    assert fake_pyaudio.pyaudio_init_calls == 1

    fake_pyaudio.broken = True  # triggers the initial read error that enters recovery

    def _replace_with_different_device_shortly_after():
        # Give frames() a moment to hit the initial read error and enter
        # Tier 1 (which will skip every attempt via the presence check,
        # since cards_present becomes False below) before switching the
        # simulated hardware out from under it.
        time.sleep(0.02)
        fake_pyaudio.cards_present = False         # original device's signature: gone
        fake_pyaudio.different_usb_present = True  # a DIFFERENT USB card is now visible at the OS level
        fake_pyaudio.device_list = [dict(DIFFERENT_USB_MIC)]  # ...and PyAudio now enumerates it
        fake_pyaudio._write_cards()
        # Tier 1 never actually reaches a real read (presence-gated skip) --
        # only Tier 2's freshly opened stream does, and that read must
        # succeed for rediscovery to declare success.
        fake_pyaudio.broken = False

    threading.Thread(target=_replace_with_different_device_shortly_after, daemon=True).start()

    frame = next(audio.frames())

    assert frame is not None
    assert audio._resolved_stable_id == "Different USB Mic"
    assert fake_pyaudio.pyaudio_init_calls == 2, "Tier 2 must reconstruct PyAudio exactly once, not repeatedly"
    audio.stop()


def test_tier2_rediscovery_never_attempted_in_pinned_mode(fake_pyaudio, monkeypatch):
    """Pinned mode must never silently substitute a different physical
    microphone, even when Tier 1 is fully exhausted and a different device
    is genuinely available -- it must fail visibly instead."""
    audio = _make_manager(monkeypatch, max_errors=1, max_reopen=2, backoff=(0.01,) * 2,
                           microphone_mode="pinned", microphone_pin="Fake USB Mic")
    audio.start()
    assert fake_pyaudio.pyaudio_init_calls == 1

    fake_pyaudio.broken = True
    fake_pyaudio.cards_present = False
    fake_pyaudio.different_usb_present = True
    fake_pyaudio.device_list = [dict(DIFFERENT_USB_MIC)]
    fake_pyaudio._write_cards()

    with pytest.raises(MicrophoneRecoveryFailedError):
        next(audio.frames())

    assert fake_pyaudio.pyaudio_init_calls == 1, "pinned mode must never reconstruct PyAudio for rediscovery"
    audio.stop()


def test_tier2_skipped_entirely_when_no_usb_audio_hardware_at_all(fake_pyaudio, monkeypatch):
    """Tier 2 must not even attempt to reconstruct PyAudio when the OS
    shows no USB audio hardware whatsoever -- the exact condition
    Milestone 7 found fatal for repeated PyAudio() construction."""
    audio = _make_manager(monkeypatch, max_errors=1, max_reopen=2, backoff=(0.01,) * 2,
                           microphone_mode="auto")
    audio.start()

    fake_pyaudio.broken = True
    fake_pyaudio.cards_present = False
    fake_pyaudio.different_usb_present = False  # nothing USB at all
    fake_pyaudio._write_cards()

    with pytest.raises(MicrophoneRecoveryFailedError):
        next(audio.frames())

    assert fake_pyaudio.pyaudio_init_calls == 1, "must never reconstruct PyAudio with zero USB hardware present"
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
