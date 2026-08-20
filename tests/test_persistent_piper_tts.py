"""Unit tests for voice/tts/persistent_piper_tts.py, using a real fake
Piper subprocess (tests/fake_piper_stub.py) rather than mocks -- this
exercises the real os.read()/select()-based IPC framing faithfully.

Milestone 8 context: the original PiperTTS reloads Piper's ONNX model on
every single call, measured directly (Piper's own --debug output) at
~1.9-2.4s per call regardless of text length. PersistentPiperTTS keeps
one Piper process alive across calls via --json-input + --output_file -.
Real-hardware benchmark (SHORT/MEDIUM/LONG fixed texts, 3 runs each):
SHORT ~2.8s -> ~0.5s after the first call, MEDIUM ~4.8s -> ~2.4s, LONG
~7s -> ~5.4s. These tests cover the IPC correctness and fault-handling
that benchmark alone doesn't prove.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest import mock

import pytest

from voice.config import TTSConfig
from voice.tts.persistent_piper_tts import PersistentPiperTTS

_FAKE_PIPER = str(Path(__file__).resolve().parent / "fake_piper_stub.py")


@pytest.fixture
def fake_config(tmp_path):
    model = tmp_path / "fake_model.onnx"
    model.write_bytes(b"fake")
    model_json = tmp_path / "fake_model.onnx.json"
    model_json.write_text("{}")
    return TTSConfig(
        binary_path=_FAKE_PIPER,
        model_path=str(model),
        model_json_path=str(model_json),
        sample_rate=22050,
        timeout_s=2.0,
    )


def test_synthesize_returns_correct_pcm_length(fake_config):
    tts = PersistentPiperTTS(fake_config)
    try:
        pcm = tts.synthesize("hello")
        # fake stub emits len(text)*100 + 200 bytes of PCM data.
        assert len(pcm) == len("hello") * 100 + 200
        assert all(b == 0 for b in pcm[:20])  # stub emits silence
    finally:
        tts.shutdown()


def test_process_stays_alive_across_multiple_calls(fake_config):
    tts = PersistentPiperTTS(fake_config)
    try:
        pid_after_first = tts._proc.pid
        tts.synthesize("first")
        tts.synthesize("second")
        tts.synthesize("third")
        assert tts._proc.pid == pid_after_first, "the same process must serve multiple requests"
    finally:
        tts.shutdown()


def test_empty_text_returns_empty_without_touching_process(fake_config):
    tts = PersistentPiperTTS(fake_config)
    try:
        pid = tts._proc.pid
        assert tts.synthesize("") == b""
        assert tts.synthesize("   ") == b""
        assert tts._proc.pid == pid, "empty input must not restart or touch the process"
    finally:
        tts.shutdown()


def test_crashed_process_is_detected_and_restarted_for_next_call(fake_config):
    tts = PersistentPiperTTS(fake_config)
    try:
        pid_before = tts._proc.pid
        result = tts.synthesize("__CRASH__")
        assert result == b""
        time.sleep(0.2)  # let the OS reap the dead process

        # Next call must transparently restart the process rather than
        # permanently failing.
        pcm = tts.synthesize("recovered")
        assert len(pcm) == len("recovered") * 100 + 200
        assert tts._proc.pid != pid_before, "expected a fresh process after the crash"
    finally:
        tts.shutdown()


def test_hanging_request_times_out_and_restarts_process(fake_config):
    fake_config.timeout_s = 0.3
    tts = PersistentPiperTTS(fake_config)
    try:
        pid_before = tts._proc.pid
        t0 = time.perf_counter()
        result = tts.synthesize("__HANG__")
        elapsed = time.perf_counter() - t0

        assert result == b""
        assert elapsed < 2.0, f"took {elapsed:.2f}s -- timeout was not bounded"

        # The hung process must be killed, not left running forever, and
        # the next request must get a fresh, working process.
        pcm = tts.synthesize("after hang")
        assert len(pcm) == len("after hang") * 100 + 200
        assert tts._proc.pid != pid_before
    finally:
        tts.shutdown()


def test_slow_but_within_timeout_request_still_succeeds(fake_config):
    fake_config.timeout_s = 3.0
    tts = PersistentPiperTTS(fake_config)
    try:
        pcm = tts.synthesize("__SLOW:0.5__")
        assert len(pcm) == len("slow response") * 100 + 200
    finally:
        tts.shutdown()


def test_shutdown_stops_the_process(fake_config):
    tts = PersistentPiperTTS(fake_config)
    proc = tts._proc
    tts.shutdown()
    time.sleep(0.2)
    assert proc.poll() is not None, "process must actually be terminated after shutdown()"


def _output_device(stable_id, backend, node, alsa_driver=None):
    from voice.audio.discovery import DeviceDescriptor
    return DeviceDescriptor(
        role="output", backend=backend, stable_id=stable_id, display_name=stable_id,
        pipewire_node_name=node, alsa_driver=alsa_driver,
    )


_HBTS001 = _output_device("B3:BB:BE:7F:9B:1A", "bluez5", "bluez_output.B3_BB_BE_7F_9B_1A.1")
_USB_SPEAKER = _output_device("Audio Array AM-C28 Device", "alsa", "alsa_output.usb", alsa_driver="snd_usb_audio")


def test_play_auto_mode_discovers_and_routes_via_pipewire_with_no_config(fake_config):
    """Plug-and-play Phase 1: with NO bluetooth_speaker_mac and NO pin
    configured (the shipped default: speaker_mode='auto'), play() must
    still find and use whatever PipeWire output is actually available --
    no config edit required."""
    assert fake_config.speaker_mode == "auto"
    assert fake_config.bluetooth_speaker_mac is None
    assert fake_config.speaker_pin is None
    tts = PersistentPiperTTS(fake_config)
    try:
        with mock.patch("voice.tts.persistent_piper_tts.discover_output_devices",
                         return_value=[_HBTS001]), \
             mock.patch("voice.tts.pipewire_playback.play_via_pipewire", return_value=True) as play_mock:
            ok = tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert ok is True
        assert play_mock.call_args[0][2] == "bluez_output.B3_BB_BE_7F_9B_1A.1"
    finally:
        tts.shutdown()


def test_play_auto_mode_prefers_bluetooth_over_usb_when_both_present(fake_config):
    tts = PersistentPiperTTS(fake_config)
    try:
        with mock.patch("voice.tts.persistent_piper_tts.discover_output_devices",
                         return_value=[_USB_SPEAKER, _HBTS001]), \
             mock.patch("voice.tts.pipewire_playback.play_via_pipewire", return_value=True) as play_mock:
            tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert play_mock.call_args[0][2] == "bluez_output.B3_BB_BE_7F_9B_1A.1"
    finally:
        tts.shutdown()


def test_play_pinned_mode_uses_bluetooth_speaker_mac_as_backward_compatible_pin(fake_config):
    """The legacy bluetooth_speaker_mac field still works exactly as
    before, but only once speaker_mode is explicitly set to 'pinned' --
    it's no longer consulted in the default 'auto' mode."""
    fake_config.speaker_mode = "pinned"
    fake_config.bluetooth_speaker_mac = "B3:BB:BE:7F:9B:1A"
    tts = PersistentPiperTTS(fake_config)
    try:
        with mock.patch("voice.tts.persistent_piper_tts.discover_output_devices",
                         return_value=[_USB_SPEAKER, _HBTS001]), \
             mock.patch("voice.tts.pipewire_playback.play_via_pipewire", return_value=True) as play_mock:
            ok = tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert ok is True
        assert play_mock.call_args[0][2] == "bluez_output.B3_BB_BE_7F_9B_1A.1"
    finally:
        tts.shutdown()


def test_play_pinned_mode_fails_visibly_when_pinned_device_absent_never_substitutes(fake_config):
    """If the pinned Bluetooth speaker isn't currently a PipeWire sink
    (e.g. disconnected) while OTHER outputs exist, play() must fail and
    say so -- not silently substitute a different device, which would
    look like success while producing no audible output on the actual
    intended speaker."""
    fake_config.speaker_mode = "pinned"
    fake_config.bluetooth_speaker_mac = "B3:BB:BE:7F:9B:1A"
    tts = PersistentPiperTTS(fake_config)
    try:
        with mock.patch("voice.tts.persistent_piper_tts.discover_output_devices",
                         return_value=[_USB_SPEAKER]), \
             mock.patch("voice.subprocess_utils.run_with_group_kill") as aplay_mock, \
             mock.patch("voice.tts.pipewire_playback.play_via_pipewire") as pw_play_mock:
            ok = tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert ok is False
        aplay_mock.assert_not_called()
        pw_play_mock.assert_not_called()
    finally:
        tts.shutdown()


def test_play_fails_visibly_when_pipewire_reports_zero_devices_no_aplay_fallback(fake_config):
    """PipeWire itself working but currently reporting zero sinks must
    fail clearly -- it must NOT fall back to the fixed ALSA device, since
    that device (the built-in HPH jack on this board) was already
    confirmed unused/off, so a silent fallback would report success while
    nothing audible happens."""
    tts = PersistentPiperTTS(fake_config)
    try:
        with mock.patch("voice.tts.persistent_piper_tts.discover_output_devices", return_value=[]), \
             mock.patch("voice.tts.persistent_piper_tts.pipewire_reachable", return_value=True), \
             mock.patch("voice.subprocess_utils.run_with_group_kill") as aplay_mock:
            ok = tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert ok is False
        aplay_mock.assert_not_called()
    finally:
        tts.shutdown()


def test_play_respects_combo_guard_excludes_bluetooth_speaker_when_mic_is_bluetooth(fake_config):
    """Combination requirement: if the microphone is already using
    Bluetooth (reported via ComboGuard), speaker auto-selection must skip
    Bluetooth candidates entirely and fall through to a wired one."""
    from voice.audio.combination import ComboGuard

    guard = ComboGuard()
    guard.set_microphone_backend("bluez5")
    tts = PersistentPiperTTS(fake_config, combo_guard=guard)
    try:
        with mock.patch("voice.tts.persistent_piper_tts.discover_output_devices",
                         return_value=[_USB_SPEAKER, _HBTS001]), \
             mock.patch("voice.tts.pipewire_playback.play_via_pipewire", return_value=True) as play_mock:
            ok = tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert ok is True
        assert play_mock.call_args[0][2] == "alsa_output.usb"  # USB, not Bluetooth
    finally:
        tts.shutdown()


def test_play_reports_bluetooth_conflict_clearly_when_only_bluetooth_speaker_available(fake_config):
    """The one explicitly unsupported combination: mic already Bluetooth,
    and the ONLY available speaker is also Bluetooth. Must fail clearly
    (never silently substitute, never crash) and the app must remain
    usable afterward (no exception escapes, no process left running)."""
    from voice.audio.combination import ComboGuard

    guard = ComboGuard()
    guard.set_microphone_backend("bluez5")
    tts = PersistentPiperTTS(fake_config, combo_guard=guard)
    try:
        with mock.patch("voice.tts.persistent_piper_tts.discover_output_devices",
                         return_value=[_HBTS001]), \
             mock.patch("voice.subprocess_utils.run_with_group_kill") as aplay_mock, \
             mock.patch("voice.tts.pipewire_playback.play_via_pipewire") as pw_play_mock:
            target, reason = tts._resolve_output_target()
            ok = tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert target is None
        assert reason == "bluetooth_conflict"
        assert ok is False
        aplay_mock.assert_not_called()
        pw_play_mock.assert_not_called()
        # The guard itself must not record a false Bluetooth+Bluetooth
        # state -- the rejected speaker pick was never actually adopted.
        assert guard.is_bluetooth_conflict() is False
    finally:
        tts.shutdown()


def test_successful_speaker_selection_reports_its_backend_to_the_combo_guard(fake_config):
    from voice.audio.combination import ComboGuard

    guard = ComboGuard()
    tts = PersistentPiperTTS(fake_config, combo_guard=guard)
    try:
        with mock.patch("voice.tts.persistent_piper_tts.discover_output_devices",
                         return_value=[_HBTS001]), \
             mock.patch("voice.tts.pipewire_playback.play_via_pipewire", return_value=True):
            tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert guard.speaker_backend == "bluez5"
    finally:
        tts.shutdown()


def test_no_combo_guard_behaves_exactly_as_before_backward_compatible(fake_config):
    """Constructing PersistentPiperTTS without a combo_guard (every
    existing caller/test) must be completely unaffected by this feature."""
    tts = PersistentPiperTTS(fake_config)
    try:
        with mock.patch("voice.tts.persistent_piper_tts.discover_output_devices",
                         return_value=[_USB_SPEAKER, _HBTS001]), \
             mock.patch("voice.tts.pipewire_playback.play_via_pipewire", return_value=True) as play_mock:
            tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert play_mock.call_args[0][2] == "bluez_output.B3_BB_BE_7F_9B_1A.1"  # unrestricted class-priority pick
    finally:
        tts.shutdown()


def test_play_falls_back_to_aplay_only_when_pipewire_itself_is_unreachable(fake_config):
    """The ONLY scenario where the legacy aplay/ALSA path may still run:
    PipeWire itself is missing/erroring (not just "no sinks right now")."""
    tts = PersistentPiperTTS(fake_config)
    try:
        with mock.patch("voice.tts.persistent_piper_tts.discover_output_devices", return_value=[]), \
             mock.patch("voice.tts.persistent_piper_tts.pipewire_reachable", return_value=False), \
             mock.patch("voice.subprocess_utils.run_with_group_kill",
                         return_value=_ok_subprocess_result()) as run_mock:
            ok = tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert ok is True
        assert run_mock.call_args[0][0][0] == "aplay"
    finally:
        tts.shutdown()


def _ok_subprocess_result():
    import subprocess as _sp
    return _sp.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")


def test_extract_pcm_finds_data_chunk_even_with_extra_leading_chunk():
    """WAV chunk order isn't format-guaranteed -- _extract_pcm must parse
    chunks properly rather than assume a fixed offset."""
    import struct
    fake_chunk = b"JUNK" + struct.pack("<I", 4) + b"\x01\x02\x03\x04"
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 22050, 44100, 2, 16)
    pcm_data = b"\xAA\xBB" * 10
    data_chunk = b"data" + struct.pack("<I", len(pcm_data)) + pcm_data
    riff_body = b"WAVE" + fake_chunk + fmt_chunk + data_chunk
    wav = b"RIFF" + struct.pack("<I", len(riff_body)) + riff_body

    extracted = PersistentPiperTTS._extract_pcm(wav)
    assert extracted == pcm_data
