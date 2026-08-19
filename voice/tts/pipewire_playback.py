"""PipeWire-routed playback for the production Bluetooth speaker.

Confirmed on real UNO Q hardware (Milestone 8): `aplay` cannot reach a
Bluetooth sink at all -- there is no PipeWire-ALSA compatibility plugin
installed on this board, so an ALSA client like `aplay` simply has no
device string that resolves to a Bluetooth output. Bluetooth audio only
exists as a PipeWire node, reachable via `pw-play --target <node>`.

The node is looked up by the device's Bluetooth MAC address
(api.bluez5.address), never by a numeric PipeWire object id or a
hardcoded node name string. Confirmed on real hardware that the numeric
id (e.g. "72") is reassigned by PipeWire every time the node is
recreated -- which happens on every Bluetooth reconnect -- so a cached id
would silently start failing on the very next reconnect. The MAC address
is the one thing that's actually fixed.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Optional

from voice.subprocess_utils import run_with_group_kill

logger = logging.getLogger(__name__)

PW_DUMP_TIMEOUT_S = 5.0


def find_bluez_sink_target(mac_address: str) -> Optional[str]:
    """Returns the current PipeWire node.name for the Audio/Sink node
    belonging to the Bluetooth device at mac_address, or None if it isn't
    currently present (not connected, or PipeWire hasn't created the node
    yet). Always queries fresh -- never caches -- since the underlying
    object is recreated on every reconnect.
    """
    try:
        result = run_with_group_kill(["pw-dump"], timeout=PW_DUMP_TIMEOUT_S, text=True)
    except subprocess.TimeoutExpired:
        logger.error("pw-dump timed out while looking up the Bluetooth sink for %s.", mac_address)
        return None
    except FileNotFoundError:
        logger.error("`pw-dump` not found -- is pipewire-bin installed?")
        return None

    if result.returncode != 0:
        logger.error("pw-dump failed (exit %d): %s", result.returncode, result.stderr.strip())
        return None

    try:
        objects = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logger.error("pw-dump produced invalid JSON: %s", exc)
        return None

    for obj in objects:
        props = obj.get("info", {}).get("props", {})
        if (props.get("api.bluez5.address") == mac_address
                and props.get("media.class") == "Audio/Sink"):
            node_name = props.get("node.name")
            if node_name:
                return node_name

    logger.warning(
        "No PipeWire Audio/Sink node found for Bluetooth device %s "
        "(not connected, or not yet registered with PipeWire).",
        mac_address,
    )
    return None


def play_via_pipewire(
    pcm_bytes: bytes,
    sample_rate: int,
    target: str,
    timeout_s: float,
) -> bool:
    """Plays raw S16LE mono PCM to a specific PipeWire node target via
    pw-play (reads from stdin, no temp file -- mirrors the aplay path's
    own stdin-piping design). Returns True on success.
    """
    if not pcm_bytes:
        return True

    cmd = [
        "pw-play",
        "--target", target,
        "--rate", str(sample_rate),
        "--channels", "1",
        "--format", "s16",
        "--raw",
        "-",
    ]
    try:
        proc = run_with_group_kill(cmd, input=pcm_bytes, timeout=timeout_s)
    except FileNotFoundError:
        logger.error("`pw-play` not found -- is pipewire-bin installed?")
        return False
    except subprocess.TimeoutExpired:
        logger.error("pw-play playback exceeded timeout -- killed.")
        return False

    if proc.returncode != 0:
        logger.error("pw-play exited %d: %s", proc.returncode, proc.stderr.decode(errors="replace"))
        return False
    return True


__all__ = ["find_bluez_sink_target", "play_via_pipewire"]
