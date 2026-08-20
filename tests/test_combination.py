"""Unit tests for voice/audio/combination.py's ComboGuard -- the "at most
one Bluetooth side" cross-role coordination primitive."""

from __future__ import annotations

import threading

from voice.audio.combination import ComboGuard


def test_fresh_guard_allows_bluetooth_on_either_side():
    guard = ComboGuard()
    assert guard.microphone_allowed("bluez5") is True
    assert guard.speaker_allowed("bluez5") is True


def test_non_bluetooth_backends_are_always_allowed_regardless_of_other_side():
    guard = ComboGuard()
    guard.set_speaker_backend("bluez5")
    guard.set_microphone_backend("bluez5")
    assert guard.microphone_allowed("alsa") is True
    assert guard.speaker_allowed("alsa") is True


def test_speaker_bluetooth_blocks_microphone_bluetooth():
    guard = ComboGuard()
    guard.set_speaker_backend("bluez5")
    assert guard.microphone_allowed("bluez5") is False
    assert guard.microphone_allowed("alsa") is True


def test_microphone_bluetooth_blocks_speaker_bluetooth():
    guard = ComboGuard()
    guard.set_microphone_backend("bluez5")
    assert guard.speaker_allowed("bluez5") is False
    assert guard.speaker_allowed("alsa") is True


def test_both_wired_never_conflicts():
    guard = ComboGuard()
    guard.set_microphone_backend("alsa")
    guard.set_speaker_backend("alsa")
    assert guard.microphone_allowed("alsa") is True
    assert guard.speaker_allowed("alsa") is True
    assert guard.is_bluetooth_conflict() is False


def test_one_side_bluetooth_one_side_wired_is_allowed_and_not_a_conflict():
    guard = ComboGuard()
    guard.set_microphone_backend("bluez5")
    guard.set_speaker_backend("alsa")
    assert guard.is_bluetooth_conflict() is False


def test_is_bluetooth_conflict_true_only_when_both_sides_are_bluetooth():
    guard = ComboGuard()
    guard.set_microphone_backend("bluez5")
    guard.set_speaker_backend("bluez5")
    assert guard.is_bluetooth_conflict() is True


def test_unresolved_backend_none_never_blocks_the_other_side():
    """Before either role has resolved anything (or after a failed
    resolution that leaves it unset), None must not be treated as
    Bluetooth -- an unresolved side must never block the other."""
    guard = ComboGuard()
    assert guard.microphone_allowed("bluez5") is True
    guard.set_speaker_backend(None)
    assert guard.microphone_allowed("bluez5") is True


def test_concurrent_updates_do_not_corrupt_state():
    """Both roles can update concurrently in the real pipeline (mic
    resolved on the audio thread, speaker resolved during a TTS call) --
    the guard must not torn-read/torn-write under concurrent access."""
    guard = ComboGuard()
    stop = threading.Event()
    errors = []

    def _flap_mic():
        try:
            while not stop.is_set():
                guard.set_microphone_backend("alsa")
                guard.microphone_allowed("bluez5")
                guard.set_microphone_backend("bluez5")
                guard.microphone_allowed("alsa")
        except Exception as exc:  # pragma: no cover -- only hit on a real bug
            errors.append(exc)

    def _flap_speaker():
        try:
            while not stop.is_set():
                guard.set_speaker_backend("alsa")
                guard.speaker_allowed("bluez5")
                guard.set_speaker_backend("bluez5")
                guard.speaker_allowed("alsa")
        except Exception as exc:  # pragma: no cover -- only hit on a real bug
            errors.append(exc)

    threads = [threading.Thread(target=_flap_mic), threading.Thread(target=_flap_speaker)]
    for t in threads:
        t.start()
    import time
    time.sleep(0.2)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)
        assert not t.is_alive()
    assert not errors
