"""Deterministic device-selection policy (Phase 1+ of the plug-and-play design).

Built on top of voice/audio/discovery.py's read-only DeviceDescriptor
listings. Discovery never picks a device (Phase 0 was discovery-only, by
design); this module is where exactly one device gets chosen from a
candidate list.

Two modes, matching the approved design:

  * "auto" (the production default): sticky-previous -> class-priority ->
    deterministic stable_id tie-break. Never requires a config edit when
    hardware changes.
  * "pinned" (opt-in override, for debugging/special deployments): select
    only the device whose stable_id matches the configured pin. If it's
    absent, raise SelectionError -- never silently substitute a different
    physical device. Pinned mode exists specifically so a deliberate choice
    is never accidentally overridden by auto-selection.

Usability: a candidate is only eligible if discovery already returned it.
discover_output_devices() only ever reports PipeWire nodes that exist right
now, and discover_input_devices() only reports PyAudio/PipeWire entries
actually enumerated just now -- so "is this candidate real and currently
present" is already guaranteed by construction before it reaches this
module. No additional liveness probe (e.g. actually opening a stream) is
performed here; that would mean opening real audio hardware just to decide
whether to open it, which is unnecessary given discovery's guarantee.

Class-priority is an explicit, stated value judgment from the approved
design (not a technical fact), kept here as a module-level default so it's
one visible place to read and, if needed, challenge:

  * Output (speaker): Bluetooth > USB > built-in. The built-in HPH output
    was confirmed unused (`Headphone Jack: values=off`) during the original
    Bluetooth investigation; Bluetooth is the proven production path.
  * Input (microphone): USB > Bluetooth > built-in. USB is the validated,
    low-latency path; a Bluetooth mic (Phase 3) shouldn't silently preempt a
    working wired mic just by being connected later, given the latency and
    profile fragility Bluetooth capture carries (see voice/audio/pw_capture.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from voice.audio.discovery import DeviceDescriptor

logger = logging.getLogger(__name__)


class SelectionError(RuntimeError):
    """No usable device could be selected -- e.g. a pinned device is
    absent, or the candidate list is empty. Callers must treat this as a
    visible failure (log it, fail the operation) and must never silently
    fall back to picking a different, unrequested device on their own."""


# LOWER NUMBER = HIGHER PRIORITY. See module docstring for the rationale.
_DEFAULT_OUTPUT_PRIORITY = {"bluez5": 0, "alsa-usb": 1, "alsa-other": 2}
_DEFAULT_INPUT_PRIORITY = {"alsa-usb": 0, "bluez5": 1, "alsa-other": 2}

# ALSA driver names observed on real USB audio hardware on this board family
# (confirmed live: "snd_usb_audio" for the Audio Array USB mic/speaker).
# Matched by prefix, not equality, to also cover other USB audio drivers
# (e.g. "snd_usb_caiaq") without needing to enumerate every possible one.
_USB_ALSA_DRIVER_PREFIX = "snd_usb"


def _priority_class(device: DeviceDescriptor) -> str:
    """Which class-priority bucket a device falls into. Never used as
    identity -- only to rank otherwise-tied candidates."""
    if device.backend == "bluez5":
        return "bluez5"
    if device.backend == "alsa":
        if device.alsa_driver and device.alsa_driver.startswith(_USB_ALSA_DRIVER_PREFIX):
            return "alsa-usb"
        # Includes genuinely built-in devices AND any ALSA device whose
        # driver name we don't recognise -- an unrecognised driver is
        # deliberately NOT assumed to be USB, so it can never silently
        # outrank a known Bluetooth or known-USB candidate.
        return "alsa-other"
    # A backend this module doesn't understand (shouldn't happen -- discovery
    # only ever emits "alsa"/"bluez5") -- treat as lowest priority, never
    # crash on an unexpected value here.
    return "alsa-other"


@dataclass
class DeviceSelector:
    """Chooses one device from a role's candidate list.

    Stateful: remembers the stable_id of the last successful auto selection,
    for the "sticky previous" rule. One instance should live for the
    lifetime of whatever owns a single role (e.g. one instance for TTS
    output, a separate one for microphone input, never shared between the
    two -- a sticky speaker choice means nothing for microphone selection).
    """

    priority: dict = field(default_factory=lambda: dict(_DEFAULT_OUTPUT_PRIORITY))
    _last_selected_stable_id: Optional[str] = field(default=None, init=False, repr=False)

    def select(
        self,
        candidates: list[DeviceDescriptor],
        *,
        mode: str = "auto",
        pin: Optional[str] = None,
    ) -> DeviceDescriptor:
        """Raises SelectionError if no candidate can be chosen -- callers
        must not catch this and substitute a device on their own; the
        caller's job is to fail the operation (or, if it has an explicitly
        documented separate fallback path, use that -- never invent one
        here)."""
        if mode not in ("auto", "pinned"):
            raise ValueError(f"Unknown selection mode: {mode!r} (must be 'auto' or 'pinned')")

        if not candidates:
            raise SelectionError("No usable devices were discovered for this role.")

        if mode == "pinned":
            return self._select_pinned(candidates, pin)
        return self._select_auto(candidates)

    def reset_sticky(self) -> None:
        """Forget the last selection -- e.g. after a hard failure, so the
        next auto selection doesn't keep preferring a device that just
        proved unusable."""
        self._last_selected_stable_id = None

    # ------------------------------------------------------------------
    def _select_pinned(self, candidates: list[DeviceDescriptor], pin: Optional[str]) -> DeviceDescriptor:
        if not pin:
            raise SelectionError(
                "mode=pinned but no pin (stable_id) was configured -- refusing to guess a device."
            )
        matches = [d for d in candidates if d.stable_id == pin]
        if not matches:
            available = sorted({d.stable_id for d in candidates})
            raise SelectionError(
                f"Pinned device {pin!r} is not currently available "
                f"(currently available: {available}). Not substituting a "
                f"different device -- fix the pin or connect the pinned device."
            )
        chosen = matches[0]
        logger.info(
            "Selected device stable_id=%r backend=%s (reason=pinned)",
            chosen.stable_id, chosen.backend,
        )
        self._last_selected_stable_id = chosen.stable_id
        return chosen

    def _select_auto(self, candidates: list[DeviceDescriptor]) -> DeviceDescriptor:
        sticky_id = self._last_selected_stable_id
        if sticky_id is not None:
            sticky_matches = [d for d in candidates if d.stable_id == sticky_id]
            if sticky_matches:
                chosen = sticky_matches[0]
                logger.info(
                    "Selected device stable_id=%r backend=%s (reason=sticky-previous)",
                    chosen.stable_id, chosen.backend,
                )
                return chosen

        ranked = sorted(
            candidates,
            key=lambda d: (self.priority.get(_priority_class(d), 99), d.stable_id),
        )
        chosen = ranked[0]
        logger.info(
            "Selected device stable_id=%r backend=%s (reason=class-priority, "
            "%d candidate(s) considered)",
            chosen.stable_id, chosen.backend, len(candidates),
        )
        self._last_selected_stable_id = chosen.stable_id
        return chosen


def make_output_selector() -> DeviceSelector:
    return DeviceSelector(priority=dict(_DEFAULT_OUTPUT_PRIORITY))


def make_input_selector() -> DeviceSelector:
    return DeviceSelector(priority=dict(_DEFAULT_INPUT_PRIORITY))


__all__ = [
    "DeviceSelector",
    "SelectionError",
    "make_output_selector",
    "make_input_selector",
]
