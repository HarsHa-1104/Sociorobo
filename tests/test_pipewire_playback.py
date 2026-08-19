"""Unit tests for voice/tts/pipewire_playback.py.

Milestone 8 context: aplay cannot reach a Bluetooth sink at all on this
board (confirmed on real hardware -- no PipeWire-ALSA compat plugin
installed). Bluetooth audio only exists as a PipeWire node, found by MAC
address (never a cached numeric id or hardcoded name -- both are
reassigned by PipeWire on every reconnect, confirmed on real hardware by
watching the sink's numeric id change across reconnects).
"""

from __future__ import annotations

import json
import subprocess
from unittest import mock

import pytest

from voice.tts.pipewire_playback import find_bluez_sink_target, play_via_pipewire

_MAC = "B3:BB:BE:7F:9B:1A"


def _pw_dump_result(objects, returncode=0, stderr=""):
    return subprocess.CompletedProcess(
        args=["pw-dump"], returncode=returncode,
        stdout=json.dumps(objects), stderr=stderr,
    )


def test_finds_sink_matching_mac_and_audio_sink_class():
    objects = [
        {"info": {"props": {"api.bluez5.address": _MAC, "media.class": "Audio/Sink",
                             "node.name": "bluez_output.B3_BB_BE_7F_9B_1A.1"}}},
    ]
    with mock.patch("voice.tts.pipewire_playback.run_with_group_kill",
                     return_value=_pw_dump_result(objects)):
        target = find_bluez_sink_target(_MAC)
    assert target == "bluez_output.B3_BB_BE_7F_9B_1A.1"


def test_ignores_non_matching_mac():
    objects = [
        {"info": {"props": {"api.bluez5.address": "AA:AA:AA:AA:AA:AA", "media.class": "Audio/Sink",
                             "node.name": "bluez_output.other.1"}}},
    ]
    with mock.patch("voice.tts.pipewire_playback.run_with_group_kill",
                     return_value=_pw_dump_result(objects)):
        target = find_bluez_sink_target(_MAC)
    assert target is None


def test_ignores_matching_mac_with_wrong_media_class():
    """A Bluetooth device can have both an Audio/Sink (speaker output) and
    an Audio/Source (its mic, e.g. a headset) node with the SAME MAC --
    must not accidentally target the source."""
    objects = [
        {"info": {"props": {"api.bluez5.address": _MAC, "media.class": "Audio/Source",
                             "node.name": "bluez_input.B3_BB_BE_7F_9B_1A.0"}}},
    ]
    with mock.patch("voice.tts.pipewire_playback.run_with_group_kill",
                     return_value=_pw_dump_result(objects)):
        target = find_bluez_sink_target(_MAC)
    assert target is None


def test_returns_none_on_pw_dump_failure():
    with mock.patch("voice.tts.pipewire_playback.run_with_group_kill",
                     return_value=_pw_dump_result([], returncode=1, stderr="boom")):
        target = find_bluez_sink_target(_MAC)
    assert target is None


def test_returns_none_on_invalid_json():
    bad = subprocess.CompletedProcess(args=["pw-dump"], returncode=0, stdout="not json", stderr="")
    with mock.patch("voice.tts.pipewire_playback.run_with_group_kill", return_value=bad):
        target = find_bluez_sink_target(_MAC)
    assert target is None


def test_returns_none_on_timeout():
    with mock.patch("voice.tts.pipewire_playback.run_with_group_kill",
                     side_effect=subprocess.TimeoutExpired(cmd="pw-dump", timeout=5.0)):
        target = find_bluez_sink_target(_MAC)
    assert target is None


def test_never_caches_across_calls():
    """Confirms every call re-queries pw-dump -- required since the sink
    is recreated (new numeric id) on every Bluetooth reconnect."""
    objects = [
        {"info": {"props": {"api.bluez5.address": _MAC, "media.class": "Audio/Sink",
                             "node.name": "bluez_output.B3_BB_BE_7F_9B_1A.1"}}},
    ]
    with mock.patch("voice.tts.pipewire_playback.run_with_group_kill",
                     return_value=_pw_dump_result(objects)) as run_mock:
        find_bluez_sink_target(_MAC)
        find_bluez_sink_target(_MAC)
    assert run_mock.call_count == 2


# ---------------------------------------------------------------------------
def test_play_via_pipewire_empty_pcm_returns_true_without_calling_pw_play():
    with mock.patch("voice.tts.pipewire_playback.run_with_group_kill") as run_mock:
        result = play_via_pipewire(b"", 22050, "bluez_output.x.1", 5.0)
    assert result is True
    run_mock.assert_not_called()


def test_play_via_pipewire_success():
    ok_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
    with mock.patch("voice.tts.pipewire_playback.run_with_group_kill", return_value=ok_result) as run_mock:
        result = play_via_pipewire(b"\x00\x00" * 100, 22050, "bluez_output.x.1", 5.0)
    assert result is True
    called_cmd = run_mock.call_args[0][0]
    assert "--target" in called_cmd
    assert "bluez_output.x.1" in called_cmd
    assert "--raw" in called_cmd


def test_play_via_pipewire_nonzero_exit_returns_false():
    fail_result = subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"device busy")
    with mock.patch("voice.tts.pipewire_playback.run_with_group_kill", return_value=fail_result):
        result = play_via_pipewire(b"\x00\x00" * 100, 22050, "bluez_output.x.1", 5.0)
    assert result is False


def test_play_via_pipewire_timeout_returns_false():
    with mock.patch("voice.tts.pipewire_playback.run_with_group_kill",
                     side_effect=subprocess.TimeoutExpired(cmd="pw-play", timeout=5.0)):
        result = play_via_pipewire(b"\x00\x00" * 100, 22050, "bluez_output.x.1", 5.0)
    assert result is False


def test_play_via_pipewire_missing_binary_returns_false():
    with mock.patch("voice.tts.pipewire_playback.run_with_group_kill",
                     side_effect=FileNotFoundError()):
        result = play_via_pipewire(b"\x00\x00" * 100, 22050, "bluez_output.x.1", 5.0)
    assert result is False
