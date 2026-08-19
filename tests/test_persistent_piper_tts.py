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


def test_play_uses_aplay_when_no_bluetooth_mac_configured(fake_config):
    """Default behaviour (bluetooth_speaker_mac unset) must be unchanged
    from before this milestone -- routes through aplay/ALSA."""
    assert fake_config.bluetooth_speaker_mac is None
    tts = PersistentPiperTTS(fake_config)
    try:
        with mock.patch("voice.subprocess_utils.run_with_group_kill",
                         return_value=_ok_subprocess_result()) as run_mock:
            ok = tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert ok is True
        called_cmd = run_mock.call_args[0][0]
        assert called_cmd[0] == "aplay"
    finally:
        tts.shutdown()


def test_play_routes_via_pipewire_when_bluetooth_mac_configured(fake_config):
    fake_config.bluetooth_speaker_mac = "B3:BB:BE:7F:9B:1A"
    tts = PersistentPiperTTS(fake_config)
    try:
        with mock.patch("voice.tts.pipewire_playback.find_bluez_sink_target",
                         return_value="bluez_output.B3_BB_BE_7F_9B_1A.1") as find_mock, \
             mock.patch("voice.tts.pipewire_playback.play_via_pipewire",
                         return_value=True) as play_mock:
            ok = tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert ok is True
        find_mock.assert_called_once_with("B3:BB:BE:7F:9B:1A")
        play_mock.assert_called_once()
        assert play_mock.call_args[0][2] == "bluez_output.B3_BB_BE_7F_9B_1A.1"
    finally:
        tts.shutdown()


def test_play_fails_visibly_when_bluetooth_sink_not_found_not_silent_alsa_fallback(fake_config):
    """If the configured Bluetooth speaker isn't currently a PipeWire
    sink (e.g. disconnected), play() must fail and say so -- not silently
    fall back to playing on the built-in ALSA device nobody's listening
    to, which would look like success while producing no audible output
    on the actual production speaker."""
    fake_config.bluetooth_speaker_mac = "B3:BB:BE:7F:9B:1A"
    tts = PersistentPiperTTS(fake_config)
    try:
        with mock.patch("voice.tts.pipewire_playback.find_bluez_sink_target", return_value=None), \
             mock.patch("voice.subprocess_utils.run_with_group_kill") as aplay_mock:
            ok = tts.play(b"\x00\x00" * 100, alsa_device="plughw:CARD=Test,DEV=3")
        assert ok is False
        aplay_mock.assert_not_called()
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
