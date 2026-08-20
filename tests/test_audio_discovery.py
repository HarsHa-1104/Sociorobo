"""Unit tests for voice/audio/discovery.py (Phase 0: read-only device discovery).

No real hardware or subprocess calls are needed for these -- pw-dump and
PyAudio are both faked/mocked, following the same patterns already used in
tests/test_pipewire_playback.py (pw-dump) and tests/test_audio_manager.py
(fake PyAudio). One additional test at the bottom runs read-only against this
board's actual current state, skipped automatically where pw-dump isn't
available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import types

import pytest

from voice.audio import discovery


# ---------------------------------------------------------------------------
# Fixtures mirroring real `pw-dump` output captured live from this UNO Q.
# ---------------------------------------------------------------------------

USB_MIC_SOURCE = {
    "id": 54,
    "info": {"props": {
        "media.class": "Audio/Source",
        "device.api": "alsa",
        "api.alsa.card.name": "Audio Array AM-C28 Device",
        "alsa.driver_name": "snd_usb_audio",
        "node.name": "alsa_input.usb-Audio_Array_Audio_Array_AM-C28_Device_2023-08-22-0001-00.analog-stereo",
        "node.description": "Audio Array AM-C28 Device Analog Stereo",
        "audio.channels": 2,
    }},
}

USB_MIC_SINK = {  # same physical USB device, its *output* side
    "id": 53,
    "info": {"props": {
        "media.class": "Audio/Sink",
        "device.api": "alsa",
        "api.alsa.card.name": "Audio Array AM-C28 Device",
        "node.name": "alsa_output.usb-Audio_Array_Audio_Array_AM-C28_Device_2023-08-22-0001-00.analog-stereo",
        "node.description": "Audio Array AM-C28 Device Analog Stereo",
        "audio.channels": 2,
    }},
}

HDMI_SINK = {
    "id": 43,
    "info": {"props": {
        "media.class": "Audio/Sink",
        "device.api": "alsa",
        "api.alsa.card.name": "Arduino-Imola-HPH-LOUT",
        "node.name": "alsa_output.platform-sound.HDMI__HDMI__sink",
        "node.description": "Built-in Audio HDMI Digital Stereo Output",
        "audio.channels": 2,
    }},
}

BLUETOOTH_SPEAKER_SINK = {
    "id": 99,
    "info": {"props": {
        "media.class": "Audio/Sink",
        "device.api": "bluez5",
        "api.bluez5.address": "B3:BB:BE:7F:9B:1A",
        "node.name": "bluez_output.B3_BB_BE_7F_9B_1A.1",
        "node.description": "HBTS001",
        "audio.channels": 2,
    }},
}

BLUETOOTH_MIC_SOURCE = {
    "id": 100,
    "info": {"props": {
        "media.class": "Audio/Source",
        "device.api": "bluez5",
        "api.bluez5.address": "AA:BB:CC:DD:EE:FF",
        "node.name": "bluez_input.AA_BB_CC_DD_EE_FF",
        "node.description": "Some BT Headset",
        "audio.channels": 1,
    }},
}

NON_AUDIO_OBJECT = {
    "id": 1,
    "info": {"props": {"media.class": "Video/Source"}},
}

REAL_UNO_Q_PYAUDIO_DEVICES = [
    {"index": 0, "name": "Audio Array AM-C28 Device: USB Audio (hw:0,0)",
     "maxInputChannels": 2, "maxOutputChannels": 2, "defaultSampleRate": 44100.0},
    {"index": 1, "name": "Arduino-Imola-HPH-LOUT: - (hw:1,1)",
     "maxInputChannels": 0, "maxOutputChannels": 8, "defaultSampleRate": 44100.0},
    {"index": 2, "name": "Arduino-Imola-HPH-LOUT: - (hw:1,3)",
     "maxInputChannels": 0, "maxOutputChannels": 2, "defaultSampleRate": 44100.0},
    {"index": 3, "name": "sysdefault",
     "maxInputChannels": 128, "maxOutputChannels": 128, "defaultSampleRate": 48000.0},
    {"index": 8, "name": "default",
     "maxInputChannels": 128, "maxOutputChannels": 128, "defaultSampleRate": 48000.0},
]


def _pw_dump_result(objects, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        ["pw-dump"], returncode, stdout=json.dumps(objects), stderr=stderr,
    )


class _FakePyAudioHost:
    def __init__(self, devices):
        self._devices = devices
        self.terminate_call_count = 0

    def get_device_count(self):
        return len(self._devices)

    def get_device_info_by_index(self, i):
        return self._devices[i]

    def terminate(self):
        self.terminate_call_count += 1


def _fake_pyaudio_module(devices):
    module = types.SimpleNamespace()
    module.PyAudio = lambda: _FakePyAudioHost(devices)
    return module


def _no_pipewire_devices(monkeypatch):
    monkeypatch.setattr(discovery, "run_with_group_kill", lambda *a, **k: _pw_dump_result([]))


def _no_pyaudio_devices(monkeypatch):
    monkeypatch.setattr(discovery, "pyaudio", None)


# ---------------------------------------------------------------------------
# PipeWire-side discovery
# ---------------------------------------------------------------------------

def test_discovers_output_sinks_including_alsa_and_bluetooth(monkeypatch):
    _no_pyaudio_devices(monkeypatch)
    monkeypatch.setattr(
        discovery, "run_with_group_kill",
        lambda *a, **k: _pw_dump_result([USB_MIC_SINK, HDMI_SINK, BLUETOOTH_SPEAKER_SINK, NON_AUDIO_OBJECT]),
    )
    devices = discovery.discover_output_devices()
    ids = {d.stable_id for d in devices}
    assert ids == {"Audio Array AM-C28 Device", "Arduino-Imola-HPH-LOUT", "B3:BB:BE:7F:9B:1A"}

    bt = next(d for d in devices if d.backend == "bluez5")
    assert bt.role == "output"
    assert bt.pipewire_node_name == "bluez_output.B3_BB_BE_7F_9B_1A.1"
    assert bt.pyaudio_index is None  # never set for output -- see discover_output_devices docstring


def test_discovers_bluetooth_input_device_pyaudio_cannot_see(monkeypatch):
    _no_pyaudio_devices(monkeypatch)
    monkeypatch.setattr(
        discovery, "run_with_group_kill",
        lambda *a, **k: _pw_dump_result([BLUETOOTH_MIC_SOURCE]),
    )
    devices = discovery.discover_input_devices()
    assert len(devices) == 1
    d = devices[0]
    assert d.role == "input"
    assert d.backend == "bluez5"
    assert d.stable_id == "AA:BB:CC:DD:EE:FF"
    assert d.pyaudio_index is None  # PortAudio cannot open a Bluetooth source on this board


def test_ignores_non_audio_pipewire_objects(monkeypatch):
    _no_pyaudio_devices(monkeypatch)
    monkeypatch.setattr(discovery, "run_with_group_kill", lambda *a, **k: _pw_dump_result([NON_AUDIO_OBJECT]))
    assert discovery.discover_output_devices() == []
    assert discovery.discover_input_devices() == []


def test_pw_dump_missing_binary_returns_empty_list(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError()
    monkeypatch.setattr(discovery, "run_with_group_kill", _raise)
    assert discovery.discover_output_devices() == []


def test_pw_dump_timeout_returns_empty_list(monkeypatch):
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd=["pw-dump"], timeout=discovery.PW_DUMP_TIMEOUT_S)
    monkeypatch.setattr(discovery, "run_with_group_kill", _raise)
    assert discovery.discover_output_devices() == []


def test_pw_dump_nonzero_exit_returns_empty_list(monkeypatch):
    monkeypatch.setattr(
        discovery, "run_with_group_kill",
        lambda *a, **k: _pw_dump_result([], returncode=1, stderr="boom"),
    )
    assert discovery.discover_output_devices() == []


def test_pw_dump_malformed_json_returns_empty_list(monkeypatch):
    result = subprocess.CompletedProcess(["pw-dump"], 0, stdout="{not valid json", stderr="")
    monkeypatch.setattr(discovery, "run_with_group_kill", lambda *a, **k: result)
    assert discovery.discover_output_devices() == []


def test_pipewire_object_missing_node_name_is_skipped(monkeypatch):
    broken = {"info": {"props": {
        "media.class": "Audio/Sink", "device.api": "alsa", "api.alsa.card.name": "X",
    }}}  # no node.name at all
    monkeypatch.setattr(discovery, "run_with_group_kill", lambda *a, **k: _pw_dump_result([broken]))
    assert discovery.discover_output_devices() == []


def test_pipewire_alsa_object_missing_card_name_is_skipped(monkeypatch):
    broken = {"info": {"props": {
        "media.class": "Audio/Sink", "device.api": "alsa", "node.name": "some_node",
    }}}  # no card-name property under either key
    monkeypatch.setattr(discovery, "run_with_group_kill", lambda *a, **k: _pw_dump_result([broken]))
    assert discovery.discover_output_devices() == []


def test_no_pipewire_devices_present(monkeypatch):
    _no_pyaudio_devices(monkeypatch)
    _no_pipewire_devices(monkeypatch)
    assert discovery.discover_output_devices() == []
    assert discovery.discover_input_devices() == []


# ---------------------------------------------------------------------------
# PyAudio-side discovery
# ---------------------------------------------------------------------------

def test_pyaudio_finds_usb_mic_by_stable_card_name(monkeypatch):
    monkeypatch.setattr(discovery, "pyaudio", _fake_pyaudio_module(REAL_UNO_Q_PYAUDIO_DEVICES))
    _no_pipewire_devices(monkeypatch)
    devices = discovery.discover_input_devices()
    assert len(devices) == 1
    d = devices[0]
    assert d.stable_id == "Audio Array AM-C28 Device"
    assert d.pyaudio_index == 0
    assert d.backend == "alsa"
    assert d.channels == 2
    assert d.native_sample_rate == 44100.0


def test_pyaudio_excludes_virtual_alsa_plugin_devices(monkeypatch):
    monkeypatch.setattr(discovery, "pyaudio", _fake_pyaudio_module(REAL_UNO_Q_PYAUDIO_DEVICES))
    _no_pipewire_devices(monkeypatch)
    names = {d.display_name for d in discovery.discover_input_devices()}
    assert "sysdefault" not in names
    assert "default" not in names


def test_pyaudio_missing_module_returns_empty_list(monkeypatch):
    _no_pyaudio_devices(monkeypatch)
    _no_pipewire_devices(monkeypatch)
    assert discovery.discover_input_devices() == []


def test_stable_identity_survives_pyaudio_index_renumbering(monkeypatch):
    """Same physical mic, but now enumerated at index 2 instead of 0 -- the
    exact class of renumbering Milestone 8 found by hand after a
    WirePlumber restart. stable_id must not depend on position."""
    at_index_0 = [REAL_UNO_Q_PYAUDIO_DEVICES[0]]
    renumbered = [
        {"index": 0, "name": "sysdefault", "maxInputChannels": 128, "maxOutputChannels": 128, "defaultSampleRate": 48000.0},
        {"index": 1, "name": "default", "maxInputChannels": 128, "maxOutputChannels": 128, "defaultSampleRate": 48000.0},
        {**REAL_UNO_Q_PYAUDIO_DEVICES[0], "index": 2},
    ]

    monkeypatch.setattr(discovery, "pyaudio", _fake_pyaudio_module(at_index_0))
    _no_pipewire_devices(monkeypatch)
    before = discovery.discover_input_devices()[0]

    monkeypatch.setattr(discovery, "pyaudio", _fake_pyaudio_module(renumbered))
    after = discovery.discover_input_devices()[0]

    assert before.stable_id == after.stable_id == "Audio Array AM-C28 Device"
    assert before.pyaudio_index == 0
    assert after.pyaudio_index == 2


def test_duplicate_device_names_are_not_silently_merged(monkeypatch):
    """Two identical-model USB mics: stable_id is genuinely ambiguous between
    them (documented KNOWN LIMITATION), but both must still be reported --
    never collapsed into one entry or dropped."""
    dup = [
        {"index": 0, "name": "Audio Array AM-C28 Device: USB Audio (hw:0,0)",
         "maxInputChannels": 2, "maxOutputChannels": 2, "defaultSampleRate": 44100.0},
        {"index": 1, "name": "Audio Array AM-C28 Device: USB Audio (hw:2,0)",
         "maxInputChannels": 2, "maxOutputChannels": 2, "defaultSampleRate": 44100.0},
    ]
    monkeypatch.setattr(discovery, "pyaudio", _fake_pyaudio_module(dup))
    _no_pipewire_devices(monkeypatch)
    devices = discovery.discover_input_devices()
    assert len(devices) == 2
    assert devices[0].stable_id == devices[1].stable_id == "Audio Array AM-C28 Device"
    assert {d.pyaudio_index for d in devices} == {0, 1}


# ---------------------------------------------------------------------------
# Merging PyAudio + PipeWire for the same physical microphone
# ---------------------------------------------------------------------------

def test_merges_pyaudio_and_pipewire_entries_for_the_same_physical_mic(monkeypatch):
    monkeypatch.setattr(discovery, "pyaudio", _fake_pyaudio_module([REAL_UNO_Q_PYAUDIO_DEVICES[0]]))
    monkeypatch.setattr(discovery, "run_with_group_kill", lambda *a, **k: _pw_dump_result([USB_MIC_SOURCE]))

    devices = discovery.discover_input_devices()
    assert len(devices) == 1
    d = devices[0]
    assert d.pyaudio_index == 0  # from PyAudio
    assert d.pipewire_node_name == USB_MIC_SOURCE["info"]["props"]["node.name"]  # from PipeWire
    assert d.stable_id == "Audio Array AM-C28 Device"


def test_include_alsa_false_never_touches_pyaudio_at_all(monkeypatch):
    """Safety escape hatch used by voice/audio/manager.py's Bluetooth
    recovery when no USB audio hardware is present at all -- must not
    construct a PyAudio host under any circumstance, only report
    PipeWire-sourced (Bluetooth) candidates."""
    def _fail_if_touched(*a, **k):
        raise AssertionError("include_alsa=False must never touch pyaudio at all")

    monkeypatch.setattr(discovery, "pyaudio", types.SimpleNamespace(PyAudio=_fail_if_touched))
    monkeypatch.setattr(discovery, "run_with_group_kill", lambda *a, **k: _pw_dump_result([BLUETOOTH_MIC_SOURCE]))

    devices = discovery.discover_input_devices(include_alsa=False)
    assert len(devices) == 1
    assert devices[0].backend == "bluez5"


def test_reused_pyaudio_host_is_not_constructed_fresh_or_terminated(monkeypatch):
    """Phase 2 safety requirement: when a caller passes its own already-open
    PyAudio host, discovery must use it as-is and must NOT call terminate()
    on it -- that's the caller's (AudioManager's) responsibility, since
    terminating a host out from under its owner would break an active
    capture stream."""
    fake_host = _FakePyAudioHost(REAL_UNO_Q_PYAUDIO_DEVICES)

    def _fail_if_constructed():
        raise AssertionError("discovery must not construct its own PyAudio() when a host is supplied")

    monkeypatch.setattr(discovery, "pyaudio", types.SimpleNamespace(PyAudio=_fail_if_constructed))
    _no_pipewire_devices(monkeypatch)

    devices = discovery.discover_input_devices(pyaudio_host=fake_host)
    assert len(devices) == 1
    assert devices[0].stable_id == "Audio Array AM-C28 Device"
    assert fake_host.terminate_call_count == 0


def test_own_pyaudio_host_is_still_terminated_when_not_reusing(monkeypatch):
    """Regression guard: the reuse feature must not accidentally leak the
    host discovery constructs for itself in the normal (no host passed)
    case -- that would be a real resource leak."""
    fake_host = _FakePyAudioHost(REAL_UNO_Q_PYAUDIO_DEVICES)
    monkeypatch.setattr(discovery, "pyaudio", types.SimpleNamespace(PyAudio=lambda: fake_host))
    _no_pipewire_devices(monkeypatch)

    discovery.discover_input_devices()
    assert fake_host.terminate_call_count == 1


def test_merge_copies_alsa_driver_from_pipewire_side_onto_the_merged_descriptor(monkeypatch):
    """Regression test: the merged descriptor for an ALSA input must carry
    alsa_driver (only ever populated on the PipeWire side) through to the
    final result -- without this, every wired USB microphone would
    misreport no driver and be misranked as the lowest class-priority
    tier by voice/audio/selection.py, incorrectly losing to a Bluetooth
    microphone candidate."""
    monkeypatch.setattr(discovery, "pyaudio", _fake_pyaudio_module([REAL_UNO_Q_PYAUDIO_DEVICES[0]]))
    monkeypatch.setattr(discovery, "run_with_group_kill", lambda *a, **k: _pw_dump_result([USB_MIC_SOURCE]))
    devices = discovery.discover_input_devices()
    assert len(devices) == 1
    assert devices[0].alsa_driver == "snd_usb_audio"


def test_discover_all_devices_combines_input_and_output(monkeypatch):
    _no_pyaudio_devices(monkeypatch)
    monkeypatch.setattr(
        discovery, "run_with_group_kill",
        lambda *a, **k: _pw_dump_result([USB_MIC_SOURCE, USB_MIC_SINK, HDMI_SINK]),
    )
    devices = discovery.discover_all_devices()
    assert {d.role for d in devices} == {"input", "output"}
    assert len(devices) == 3


# ---------------------------------------------------------------------------
# Live regression against this board's actual current hardware (read-only).
# Skipped automatically wherever pw-dump isn't available (e.g. off-board CI).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("pw-dump") is None, reason="pw-dump not available on this system")
def test_live_regression_against_real_uno_q_hardware():
    """Not mocked -- calls the real pw-dump/PyAudio on whatever machine runs
    this test. On the actual production UNO Q, confirms discovery finds the
    real production microphone by its real stable identity. Does not assert
    anything about the Bluetooth speaker specifically, since that depends on
    HBTS001 actually being paired+connected at test time, which this
    read-only test must not require or change."""
    inputs = discovery.discover_input_devices()
    input_ids = {d.stable_id for d in inputs}
    assert "Audio Array AM-C28 Device" in input_ids

    outputs = discovery.discover_output_devices()
    output_ids = {d.stable_id for d in outputs}
    assert output_ids  # at least the built-in/USB ALSA sinks should be visible
