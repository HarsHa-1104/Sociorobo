"""Unit tests for voice/audio/selection.py (Phase 1+: deterministic device
selection on top of Phase 0's read-only discovery)."""

from __future__ import annotations

import pytest

from voice.audio.discovery import DeviceDescriptor
from voice.audio.selection import DeviceSelector, SelectionError, make_input_selector, make_output_selector


def _dev(role, backend, stable_id, alsa_driver=None, node="node") -> DeviceDescriptor:
    return DeviceDescriptor(
        role=role, backend=backend, stable_id=stable_id, display_name=stable_id,
        pipewire_node_name=node, alsa_driver=alsa_driver,
    )


HBTS001 = _dev("output", "bluez5", "B3:BB:BE:7F:9B:1A", node="bluez_output.hbts001")
USB_SPEAKER = _dev("output", "alsa", "Audio Array AM-C28 Device", alsa_driver="snd_usb_audio", node="alsa_output.usb")
BUILTIN_HDMI = _dev("output", "alsa", "Arduino-Imola-HPH-LOUT", alsa_driver="snd_soc_sm8250", node="alsa_output.hdmi")

USB_MIC = _dev("input", "alsa", "Audio Array AM-C28 Device", alsa_driver="snd_usb_audio", node="alsa_input.usb")
BT_MIC = _dev("input", "bluez5", "AA:BB:CC:DD:EE:FF", node="bluez_input.headset")
BUILTIN_MIC = _dev("input", "alsa", "Some Builtin Mic", alsa_driver="snd_soc_sm8250", node="alsa_input.builtin")


# ---------------------------------------------------------------------------
# Basic errors
# ---------------------------------------------------------------------------

def test_no_candidates_raises_selection_error():
    selector = make_output_selector()
    with pytest.raises(SelectionError):
        selector.select([])


def test_unknown_mode_raises_value_error():
    selector = make_output_selector()
    with pytest.raises(ValueError):
        selector.select([HBTS001], mode="bogus")


# ---------------------------------------------------------------------------
# Class-priority ranking -- the core "which device wins" behavior
# ---------------------------------------------------------------------------

def test_output_prefers_bluetooth_over_usb_over_builtin():
    selector = make_output_selector()
    chosen = selector.select([BUILTIN_HDMI, USB_SPEAKER, HBTS001], mode="auto")
    assert chosen is HBTS001


def test_output_prefers_usb_over_builtin_when_no_bluetooth_present():
    """This is the exact worked example from the approved design: with only
    the Audio Array (USB) and the built-in HDMI sink present -- no Bluetooth
    speaker connected -- auto mode must pick the USB device, not built-in."""
    selector = make_output_selector()
    chosen = selector.select([BUILTIN_HDMI, USB_SPEAKER], mode="auto")
    assert chosen is USB_SPEAKER


def test_input_prefers_usb_over_bluetooth_over_builtin():
    selector = make_input_selector()
    chosen = selector.select([BUILTIN_MIC, BT_MIC, USB_MIC], mode="auto")
    assert chosen is USB_MIC


def test_single_candidate_is_always_selected_regardless_of_class():
    selector = make_output_selector()
    chosen = selector.select([BUILTIN_HDMI], mode="auto")
    assert chosen is BUILTIN_HDMI


def test_unrecognised_alsa_driver_never_outranks_known_usb_or_bluetooth():
    unknown_driver_device = _dev("output", "alsa", "Mystery Device", alsa_driver="snd_some_future_driver")
    selector = make_output_selector()
    chosen = selector.select([unknown_driver_device, USB_SPEAKER], mode="auto")
    assert chosen is USB_SPEAKER  # known-usb beats unrecognised driver

    # Fresh selector (no sticky state) -- unrecognised driver must also
    # never outrank a known Bluetooth candidate.
    selector2 = make_output_selector()
    chosen2 = selector2.select([unknown_driver_device, HBTS001], mode="auto")
    assert chosen2 is HBTS001


# ---------------------------------------------------------------------------
# Deterministic tie-break
# ---------------------------------------------------------------------------

def test_tie_break_is_deterministic_stable_id_sort_not_arbitrary():
    a = _dev("output", "bluez5", "AA:00:00:00:00:01")
    b = _dev("output", "bluez5", "BB:00:00:00:00:02")
    selector1 = make_output_selector()
    selector2 = make_output_selector()
    # Same inputs in different order must produce the same pick every time.
    assert selector1.select([b, a], mode="auto") is a
    assert selector2.select([a, b], mode="auto") is a


# ---------------------------------------------------------------------------
# Sticky-previous-selection
# ---------------------------------------------------------------------------

def test_sticky_keeps_previous_device_even_if_a_higher_priority_one_appears():
    selector = make_output_selector()
    first = selector.select([USB_SPEAKER, BUILTIN_HDMI], mode="auto")
    assert first is USB_SPEAKER

    # Bluetooth speaker shows up later -- normally higher priority, but the
    # sticky rule means we don't gratuitously switch away from what's
    # already working.
    second = selector.select([USB_SPEAKER, BUILTIN_HDMI, HBTS001], mode="auto")
    assert second is USB_SPEAKER


def test_sticky_falls_through_to_priority_once_previous_device_is_gone():
    selector = make_output_selector()
    first = selector.select([USB_SPEAKER, HBTS001], mode="auto")
    assert first is HBTS001  # bluetooth wins first time (no sticky state yet)

    # HBTS001 disappears -- selection must fall through to the next
    # available candidate by priority, not raise or return something stale.
    second = selector.select([USB_SPEAKER, BUILTIN_HDMI], mode="auto")
    assert second is USB_SPEAKER


def test_reset_sticky_clears_previous_selection():
    selector = make_output_selector()
    selector.select([USB_SPEAKER, HBTS001], mode="auto")  # picks HBTS001
    selector.reset_sticky()
    # Without sticky state, re-ranking [USB_SPEAKER] alone must still work
    # cleanly (regression: reset must not leave the selector in a broken
    # state).
    chosen = selector.select([USB_SPEAKER], mode="auto")
    assert chosen is USB_SPEAKER


def test_separate_selector_instances_do_not_share_sticky_state():
    """A speaker selector and a microphone selector must be fully
    independent -- picking a Bluetooth speaker must never affect
    microphone sticky state or vice versa."""
    output_selector = make_output_selector()
    input_selector = make_input_selector()
    output_selector.select([HBTS001], mode="auto")
    # A fresh input selector has no sticky state at all -- must rank by
    # priority as if nothing happened on the output selector.
    chosen = input_selector.select([BT_MIC, USB_MIC], mode="auto")
    assert chosen is USB_MIC


# ---------------------------------------------------------------------------
# Pinned mode
# ---------------------------------------------------------------------------

def test_pinned_mode_selects_only_the_matching_device():
    selector = make_output_selector()
    chosen = selector.select([USB_SPEAKER, HBTS001, BUILTIN_HDMI], mode="pinned", pin="B3:BB:BE:7F:9B:1A")
    assert chosen is HBTS001


def test_pinned_mode_ignores_class_priority_entirely():
    """Even though USB/built-in would lose to Bluetooth under auto-mode
    priority, an explicit pin to a lower-priority device must still win --
    pinning means deliberate control, not "priority with a hint"."""
    selector = make_output_selector()
    chosen = selector.select([USB_SPEAKER, HBTS001], mode="pinned", pin="Audio Array AM-C28 Device")
    assert chosen is USB_SPEAKER


def test_pinned_mode_fails_clearly_when_pinned_device_absent_never_substitutes():
    selector = make_output_selector()
    with pytest.raises(SelectionError):
        selector.select([USB_SPEAKER, BUILTIN_HDMI], mode="pinned", pin="B3:BB:BE:7F:9B:1A")


def test_pinned_mode_with_no_pin_configured_fails_clearly():
    selector = make_output_selector()
    with pytest.raises(SelectionError):
        selector.select([USB_SPEAKER], mode="pinned", pin=None)


# ---------------------------------------------------------------------------
# is_allowed external constraint (used by voice/audio/combination.py)
# ---------------------------------------------------------------------------

def test_is_allowed_filters_out_disallowed_candidates_in_auto_mode():
    selector = make_output_selector()
    chosen = selector.select(
        [HBTS001, USB_SPEAKER], mode="auto",
        is_allowed=lambda d: d.backend != "bluez5",
    )
    assert chosen is USB_SPEAKER


def test_is_allowed_rejecting_everything_raises_selection_error():
    selector = make_output_selector()
    with pytest.raises(SelectionError):
        selector.select([HBTS001], mode="auto", is_allowed=lambda d: d.backend != "bluez5")


def test_is_allowed_applies_even_in_pinned_mode_never_overridden_by_a_pin():
    """The whole point of an external hard constraint is that a manual pin
    cannot bypass it -- pinning a disallowed device must fail exactly like
    pinning an absent one, never silently succeed."""
    selector = make_output_selector()
    with pytest.raises(SelectionError):
        selector.select(
            [HBTS001, USB_SPEAKER], mode="pinned", pin="B3:BB:BE:7F:9B:1A",
            is_allowed=lambda d: d.backend != "bluez5",
        )


def test_is_allowed_none_means_no_extra_filtering_backward_compatible():
    selector = make_output_selector()
    chosen = selector.select([HBTS001, USB_SPEAKER], mode="auto", is_allowed=None)
    assert chosen is HBTS001  # normal class-priority behavior, unaffected


def test_pinned_mode_never_falls_back_to_a_different_device_after_failure():
    """Regression guard for the exact failure mode the design explicitly
    forbids: a missing pinned device must never result in some OTHER device
    silently being used instead."""
    selector = make_output_selector()
    with pytest.raises(SelectionError):
        selector.select([USB_SPEAKER], mode="pinned", pin="not-connected-anything")
    # Selector must not have recorded a bogus sticky selection from the
    # failed attempt.
    assert selector._last_selected_stable_id is None
