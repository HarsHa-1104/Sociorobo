"""Read-only audio-device discovery (Phase 0 of the plug-and-play design).

This module answers exactly one question: "what audio devices are currently
present, and what stable information identifies each one?" It does not select
a device and does not open any stream or start any background monitoring --
device *selection* is voice/audio/selection.py's job (Phase 1+), built on top
of this module's read-only listings. Safe to import and call repeatedly from
a REPL, a script, or tests.

Two independent, cross-referenced sources, because each existing subsystem
needs something different from a device:

  * PyAudio/PortAudio enumeration -- the only source that yields a
    ``pyaudio_index`` usable by AudioManager's existing capture path
    (voice/audio/manager.py). Cannot see Bluetooth devices at all: this board
    has no PipeWire-ALSA compatibility plugin (confirmed on real hardware
    during Milestone 8 -- the same reason `aplay` cannot reach a Bluetooth
    sink), so a Bluetooth microphone is invisible to PyAudio by construction,
    not by a bug here.
  * `pw-dump` (PipeWire) -- sees everything, including Bluetooth devices, and
    is the only source used for output/speaker discovery, matching how
    playback actually happens in this codebase today (`aplay`/`pw-play`,
    never PyAudio).

Stable identity (see DeviceDescriptor.stable_id) is deliberately NEVER a raw
PyAudio index or a PipeWire numeric object id -- both are proven unstable on
this board (Milestone 7/8: renumbering after WirePlumber restarts, PipeWire
reassigning object ids on every Bluetooth reconnect). Instead:

  * ALSA-backed devices (USB, built-in): the ALSA card name (e.g.
    "Audio Array AM-C28 Device"), read from PipeWire's `api.alsa.card.name`
    property and, on the PyAudio side, parsed out of PyAudio's own device
    name string (which already embeds it, confirmed live on this board: name
    "Audio Array AM-C28 Device: USB Audio (hw:0,0)" -- the "hw:0,0" part is
    the unstable bit, the card name before it is not). This is the same
    anchor already proven for the *speaker* ALSA string fix in Milestone 8,
    just applied consistently here.
  * Bluetooth devices: the MAC address (`api.bluez5.address`), the same
    anchor already used by voice/tts/pipewire_playback.py.

KNOWN LIMITATION -- duplicate hardware: if two USB devices of the identical
model are connected simultaneously, they report the identical ALSA card name,
so `stable_id` cannot distinguish between them. Both are still returned (never
silently merged into one), so a caller can at least see there are two, but
this module makes no claim about which PyAudio index or PipeWire node belongs
to which physical device in that case. Disambiguating this (e.g. by USB port
path) is out of scope for Phase 0.

Still out of scope for this module specifically, by design: device
*selection* of any kind (that's voice/audio/selection.py), Bluetooth
microphone *capture* (`pw-record` and friends -- this module only discovers
that a Bluetooth source node exists, it never opens or streams from one,
that's Phase 3), and any background/event-driven monitoring (Phase 4).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from voice.subprocess_utils import run_with_group_kill

logger = logging.getLogger(__name__)

try:
    import pyaudio
except ImportError:  # pragma: no cover
    pyaudio = None  # type: ignore


PW_DUMP_TIMEOUT_S = 5.0

# Matches PyAudio's own device-name format for a genuine hardware-backed ALSA
# device, e.g. "Audio Array AM-C28 Device: USB Audio (hw:0,0)" or
# "Arduino-Imola-HPH-LOUT: - (hw:1,1)" -- both confirmed live on this board.
# Deliberately does NOT match ALSA's virtual/plugin devices ("default",
# "sysdefault", "dmix", "front", "surround40", "iec958", "spdif", ...), which
# have no "card_name: ..." prefix and are software aliases for the same
# underlying hardware rather than distinct physical devices -- including them
# would just duplicate the real entries under a meaningless identity.
_HW_DEVICE_NAME_RE = re.compile(r"^(?P<card>.+?):\s.*\(hw:\d+,\d+\)$")


@dataclass(frozen=True)
class DeviceDescriptor:
    """Everything this module currently knows about one audio device.

    Not every field is populated by every discovery source -- see the module
    docstring for which source provides what.
    """

    role: str  # "input" | "output"
    backend: str  # "alsa" | "bluez5"
    stable_id: str  # ALSA: card name (e.g. "Audio Array AM-C28 Device"). BlueZ: MAC address.
    display_name: str  # human-readable, for logs only -- never used as identity
    pyaudio_index: Optional[int] = None  # only ever set for backend == "alsa"; PyAudio cannot see Bluetooth at all
    pipewire_node_name: Optional[str] = None  # set whenever PipeWire currently has a node for this device
    channels: Optional[int] = None
    native_sample_rate: Optional[float] = None  # PortAudio reports this; PipeWire negotiates it dynamically and doesn't expose a static value, so this is None for PipeWire-only (Bluetooth) entries
    connected: bool = True  # Phase 0 only ever returns devices actually observed present at call time, so this is always True today -- kept for the selection/usability-check logic later phases will add
    alsa_driver: Optional[str] = None  # backend == "alsa" only, e.g. "snd_usb_audio" vs "snd_soc_sm8250" -- used by voice/audio/selection.py to distinguish a USB device from a built-in one for class-priority ranking (see that module); never used as identity


# ---------------------------------------------------------------------------
# PipeWire (`pw-dump`) discovery
# ---------------------------------------------------------------------------

def _run_pw_dump() -> Optional[list]:
    try:
        result = run_with_group_kill(["pw-dump"], timeout=PW_DUMP_TIMEOUT_S, text=True)
    except subprocess.TimeoutExpired:
        logger.error("pw-dump timed out during device discovery.")
        return None
    except FileNotFoundError:
        logger.error("`pw-dump` not found -- is pipewire-bin installed?")
        return None

    if result.returncode != 0:
        logger.error("pw-dump failed (exit %d): %s", result.returncode, result.stderr.strip())
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logger.error("pw-dump produced invalid JSON: %s", exc)
        return None


def _pipewire_descriptor_from_props(role: str, props: dict) -> Optional[DeviceDescriptor]:
    node_name = props.get("node.name")
    if not node_name:
        # No node.name means nothing (playback or capture) could ever target
        # this object -- not usable, so not worth describing.
        return None

    mac = props.get("api.bluez5.address")
    if mac:
        backend = "bluez5"
        stable_id = mac
    elif props.get("device.api") == "alsa":
        backend = "alsa"
        card_name = props.get("api.alsa.card.name") or props.get("alsa.card_name")
        if not card_name:
            logger.warning(
                "Skipping PipeWire node %r: device.api=alsa but no card-name "
                "property present -- cannot assign a stable identity.",
                node_name,
            )
            return None
        stable_id = card_name
    else:
        # A backend this module doesn't yet understand -- skip rather than
        # invent an identity for it.
        return None

    channels_raw = props.get("audio.channels")
    channels = int(channels_raw) if channels_raw is not None else None

    return DeviceDescriptor(
        role=role,
        backend=backend,
        stable_id=stable_id,
        display_name=props.get("node.description") or node_name,
        pipewire_node_name=node_name,
        channels=channels,
        native_sample_rate=None,
        connected=True,
        alsa_driver=props.get("alsa.driver_name") if backend == "alsa" else None,
    )


def _discover_pipewire_devices(role: str) -> list[DeviceDescriptor]:
    """Enumerate PipeWire Audio/Source (role="input") or Audio/Sink
    (role="output") nodes. Always queries fresh -- never caches -- mirroring
    voice/tts/pipewire_playback.py's existing "never cache, objects are
    recreated on every reconnect" rationale.
    """
    objects = _run_pw_dump()
    if objects is None:
        return []

    target_media_class = "Audio/Source" if role == "input" else "Audio/Sink"
    devices = []
    for obj in objects:
        props = obj.get("info", {}).get("props", {})
        if props.get("media.class") != target_media_class:
            continue
        descriptor = _pipewire_descriptor_from_props(role, props)
        if descriptor is not None:
            devices.append(descriptor)

    return devices


# ---------------------------------------------------------------------------
# PyAudio/PortAudio discovery
# ---------------------------------------------------------------------------

def _discover_pyaudio_devices(role: str, pyaudio_host: Optional["pyaudio.PyAudio"] = None) -> list[DeviceDescriptor]:
    """Enumerate PyAudio devices with at least one channel in the requested
    direction, restricted to genuine hardware-backed ALSA devices (see
    _HW_DEVICE_NAME_RE). Returns [] (with a logged warning) if PyAudio isn't
    installed, rather than raising -- discovery should never crash a caller
    just because one of its two sources is unavailable.

    If `pyaudio_host` is given, it is used as-is (never constructed or
    terminated here) instead of creating a fresh `pyaudio.PyAudio()`
    instance. Two reasons a caller should pass one, both from real-hardware
    findings on this board:

      * Cost: constructing a PyAudio host measured at ~364ms (Phase 0
        benchmark) -- callers that already hold a live host (e.g.
        AudioManager, which keeps one open for its whole lifetime) should
        reuse it rather than pay that again on every discovery call.
      * Safety: voice/audio/manager.py's own recovery logic (Milestone 7)
        found that repeatedly constructing/tearing down
        `pyaudio.PyAudio()` while a USB device was physically absent
        crashed the whole process with no Python-level exception at all.
        Recovery code must never let discovery construct a second,
        independent PyAudio host of its own -- reusing the caller's
        already-initialized one sidesteps that risk entirely, the same way
        AudioManager's own recovery already never re-constructs its host.
    """
    if pyaudio is None:
        logger.warning("pyaudio is not installed -- skipping PyAudio-based discovery.")
        return []

    channel_key = "maxInputChannels" if role == "input" else "maxOutputChannels"

    pa = pyaudio_host if pyaudio_host is not None else pyaudio.PyAudio()
    devices = []
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            channels = int(info[channel_key])
            if channels <= 0:
                continue

            name = info["name"]
            match = _HW_DEVICE_NAME_RE.match(name)
            if not match:
                # A virtual/plugin ALSA device ("default", "dmix", ...) --
                # not a distinct physical device, so it has no stable
                # hardware identity to discover by. See module docstring.
                continue

            devices.append(DeviceDescriptor(
                role=role,
                backend="alsa",
                stable_id=match.group("card").strip(),
                display_name=name,
                pyaudio_index=i,
                pipewire_node_name=None,
                channels=channels,
                native_sample_rate=float(info["defaultSampleRate"]),
                connected=True,
            ))
    finally:
        if pyaudio_host is None:
            pa.terminate()  # only terminate a host we constructed ourselves

    return devices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_input_devices(pyaudio_host: Optional["pyaudio.PyAudio"] = None) -> list[DeviceDescriptor]:
    """All currently-discoverable microphones: PyAudio-visible ALSA devices,
    merged with whatever PipeWire also reports for the same physical device
    (enriching it with a pipewire_node_name), plus any device only PipeWire
    can see at all (Bluetooth -- PyAudio cannot open a Bluetooth source on
    this board, see module docstring).

    Merging only ever combines a PipeWire entry with a PyAudio entry when
    exactly one PyAudio device shares that PipeWire entry's stable_id. If
    stable_id is ambiguous (duplicate hardware, see KNOWN LIMITATION above),
    every matching entry from both sources is returned separately rather than
    guessing a pairing.

    Pass `pyaudio_host` to reuse an existing, already-initialized PyAudio
    host instance instead of constructing a new one -- see
    _discover_pyaudio_devices' docstring for why this matters (cost and,
    during recovery, safety). Callers holding a long-lived host (like
    AudioManager) should always pass it.
    """
    pyaudio_devices = _discover_pyaudio_devices("input", pyaudio_host=pyaudio_host)
    pipewire_devices = _discover_pipewire_devices("input")

    pyaudio_by_id: dict[str, list[DeviceDescriptor]] = defaultdict(list)
    for d in pyaudio_devices:
        pyaudio_by_id[d.stable_id].append(d)

    merged: list[DeviceDescriptor] = []
    consumed = set()

    for pw_dev in pipewire_devices:
        candidates = pyaudio_by_id.get(pw_dev.stable_id, [])
        if len(candidates) == 1:
            pa_dev = candidates[0]
            merged.append(dataclasses.replace(
                pa_dev,
                pipewire_node_name=pw_dev.pipewire_node_name,
                channels=pw_dev.channels or pa_dev.channels,
            ))
            consumed.add(id(pa_dev))
        else:
            # Zero matches (Bluetooth, or any PyAudio-invisible device) or
            # multiple matches (ambiguous duplicate hardware) -- in both
            # cases, don't guess: report the PipeWire entry on its own, and
            # let every candidate PyAudio entry (if any) surface separately
            # below.
            merged.append(pw_dev)

    for pa_dev in pyaudio_devices:
        if id(pa_dev) not in consumed:
            merged.append(pa_dev)

    return merged


def discover_output_devices() -> list[DeviceDescriptor]:
    """All currently-discoverable speakers/outputs, via PipeWire only.

    Unlike the microphone side, nothing in this codebase plays audio through
    PyAudio (playback is `aplay`/`pw-play`, see
    voice/tts/persistent_piper_tts.py) -- so PipeWire is the only discovery
    source that matches how output actually happens here, and
    ``pyaudio_index`` is always None on every returned descriptor.
    """
    return _discover_pipewire_devices("output")


def discover_all_devices() -> list[DeviceDescriptor]:
    """Convenience: every discoverable input and output device in one list."""
    return discover_input_devices() + discover_output_devices()


def pipewire_reachable() -> bool:
    """True if `pw-dump` can be invoked and succeeds -- independent of
    whether it currently reports any audio devices at all.

    `discover_output_devices()`/`discover_input_devices()` return an empty
    list both when PipeWire genuinely has zero matching devices right now
    AND when `pw-dump` itself is missing/erroring/timing out (the specific
    reason is only logged, not part of their return value). Callers that
    need to tell those two cases apart -- e.g. to decide whether a
    non-PipeWire fallback is appropriate at all -- can use this. Costs one
    extra `pw-dump` subprocess call; only call it after an empty discovery
    result, not on every device lookup.
    """
    return _run_pw_dump() is not None


__all__ = [
    "DeviceDescriptor",
    "discover_input_devices",
    "discover_output_devices",
    "discover_all_devices",
    "pipewire_reachable",
]
