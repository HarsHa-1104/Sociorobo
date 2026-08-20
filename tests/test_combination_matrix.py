"""Explicit tests for the full microphone x speaker combination-support
matrix (product requirement: at most one side may be Bluetooth):

                         Speaker
                 USB    Wired    Bluetooth
Mic USB           YES     YES       YES
Mic Wired        YES     YES       YES
Mic Bluetooth    YES     YES       NO

USB and "Wired" (e.g. a 3.5mm output through the board's built-in codec)
are both the "alsa" backend -- voice/audio/combination.py's ComboGuard
only ever restricts Bluetooth, so at the combination-RULE level they are
equivalent; the distinction between them only matters for within-role
class-priority ranking (voice/audio/selection.py's _priority_class, see
its own tests in test_audio_selection.py -- alsa_driver="snd_usb_audio"
vs. something else). This file still models all three categories
explicitly and separately per the requested matrix, exercising the real
DeviceDescriptor/ComboGuard/DeviceSelector objects together -- not
re-deriving the rule, just confirming the actual production classes
produce exactly this table.
"""

from __future__ import annotations

import pytest

from voice.audio.combination import ComboGuard
from voice.audio.discovery import DeviceDescriptor
from voice.audio.selection import SelectionError, make_input_selector, make_output_selector

# Category -> (backend, alsa_driver). alsa_driver is irrelevant to the
# combination rule itself but included so these fixtures are faithful
# stand-ins for what discovery.py would actually produce for each case.
MIC_CATEGORIES = {
    "USB": ("alsa", "snd_usb_audio"),
    "Wired": ("alsa", "snd_soc_sm8250"),  # e.g. the board's built-in codec
    "Bluetooth": ("bluez5", None),
}
SPEAKER_CATEGORIES = {
    "USB": ("alsa", "snd_usb_audio"),
    "Wired": ("alsa", "snd_soc_sm8250"),
    "Bluetooth": ("bluez5", None),
}

# The one and only unsupported cell.
UNSUPPORTED = {("Bluetooth", "Bluetooth")}


def _mic(category: str) -> DeviceDescriptor:
    backend, driver = MIC_CATEGORIES[category]
    return DeviceDescriptor(
        role="input", backend=backend, stable_id=f"mic-{category}", display_name=f"mic-{category}",
        pipewire_node_name=f"node-mic-{category}",
        pyaudio_index=(0 if backend == "alsa" else None),
        alsa_driver=driver,
    )


def _speaker(category: str) -> DeviceDescriptor:
    backend, driver = SPEAKER_CATEGORIES[category]
    return DeviceDescriptor(
        role="output", backend=backend, stable_id=f"speaker-{category}", display_name=f"speaker-{category}",
        pipewire_node_name=f"node-speaker-{category}", alsa_driver=driver,
    )


def _select_mic_then_speaker(mic_category: str, speaker_category: str):
    """Mirrors the real sequence: voice_manager.py's build_voice_manager()
    resolves the microphone once (AudioManager.start()) before any
    speaker selection happens (PersistentPiperTTS.play(), called later,
    possibly many times), sharing one ComboGuard -- see that module's
    combo_guard wiring. Returns (chosen_mic, chosen_speaker_or_None,
    raised_selection_error_or_None).
    """
    guard = ComboGuard()
    mic_selector = make_input_selector()
    speaker_selector = make_output_selector()

    chosen_mic = mic_selector.select(
        [_mic(mic_category)], mode="auto",
        is_allowed=lambda d: guard.microphone_allowed(d.backend),
    )
    guard.set_microphone_backend(chosen_mic.backend)

    try:
        chosen_speaker = speaker_selector.select(
            [_speaker(speaker_category)], mode="auto",
            is_allowed=lambda d: guard.speaker_allowed(d.backend),
        )
    except SelectionError as exc:
        return chosen_mic, None, exc, guard

    guard.set_speaker_backend(chosen_speaker.backend)
    return chosen_mic, chosen_speaker, None, guard


@pytest.mark.parametrize("speaker_category", ["USB", "Wired", "Bluetooth"])
@pytest.mark.parametrize("mic_category", ["USB", "Wired", "Bluetooth"])
def test_combination_matrix_cell(mic_category, speaker_category):
    chosen_mic, chosen_speaker, error, guard = _select_mic_then_speaker(mic_category, speaker_category)

    should_be_unsupported = (mic_category, speaker_category) in UNSUPPORTED

    if should_be_unsupported:
        assert error is not None, (
            f"mic={mic_category} + speaker={speaker_category} must be rejected "
            f"(the one explicitly unsupported combination)"
        )
        assert chosen_speaker is None
        # A rejected combination must never be recorded as active.
        assert guard.is_bluetooth_conflict() is False
    else:
        assert error is None, (
            f"mic={mic_category} + speaker={speaker_category} must be supported, "
            f"but selection raised: {error}"
        )
        assert chosen_speaker is not None
        assert chosen_mic.stable_id == f"mic-{mic_category}"
        assert chosen_speaker.stable_id == f"speaker-{speaker_category}"
        # Never both Bluetooth for a combination that was actually adopted.
        assert guard.is_bluetooth_conflict() is False


def test_full_matrix_produces_exactly_the_one_documented_unsupported_cell():
    """Belt-and-suspenders: enumerate the entire 3x3 grid in one test and
    assert the failure set is EXACTLY {(Bluetooth, Bluetooth)} -- catches
    a regression that accidentally blocks (or accidentally permits) any
    other cell, not just the parametrized per-cell checks above."""
    failures = set()
    for mic_category in MIC_CATEGORIES:
        for speaker_category in SPEAKER_CATEGORIES:
            _, chosen_speaker, error, _ = _select_mic_then_speaker(mic_category, speaker_category)
            if error is not None:
                failures.add((mic_category, speaker_category))
                assert chosen_speaker is None
    assert failures == UNSUPPORTED


def test_bluetooth_plus_bluetooth_never_crashes_and_never_leaves_state_inconsistent():
    """The unsupported combination must fail as a clean, catchable
    exception -- not an unhandled crash -- and must leave the guard in a
    coherent state usable for a subsequent, different attempt (the
    "keep the application alive/recoverable" requirement)."""
    guard = ComboGuard()
    mic_selector = make_input_selector()
    speaker_selector = make_output_selector()

    chosen_mic = mic_selector.select(
        [_mic("Bluetooth")], mode="auto",
        is_allowed=lambda d: guard.microphone_allowed(d.backend),
    )
    guard.set_microphone_backend(chosen_mic.backend)

    with pytest.raises(SelectionError):
        speaker_selector.select(
            [_speaker("Bluetooth")], mode="auto",
            is_allowed=lambda d: guard.speaker_allowed(d.backend),
        )

    # The guard must still work correctly for a follow-up attempt -- e.g.
    # a wired speaker becoming available, exactly the recovery path
    # AudioManager/PersistentPiperTTS actually take (never gets stuck
    # believing a conflict is permanently active).
    chosen_speaker = speaker_selector.select(
        [_speaker("Wired")], mode="auto",
        is_allowed=lambda d: guard.speaker_allowed(d.backend),
    )
    assert chosen_speaker.stable_id == "speaker-Wired"
