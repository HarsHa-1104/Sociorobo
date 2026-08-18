from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from voice.config import TTSConfig
from voice.tts.piper_tts import PiperTTS


@pytest.fixture
def tts_paths(tmp_path):
    binary = tmp_path / "piper"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    model = tmp_path / "en_US-lessac-medium.onnx"
    model.write_bytes(b"fake-voice-model")
    model_json = tmp_path / "en_US-lessac-medium.onnx.json"
    model_json.write_text("{}")
    return binary, model


def test_missing_binary_raises(tmp_path):
    cfg = TTSConfig(binary_path=str(tmp_path / "nope"), model_path=str(tmp_path / "nope.onnx"))
    with pytest.raises(FileNotFoundError):
        PiperTTS(cfg)


def test_synthesize_empty_text_returns_empty_bytes(tts_paths):
    binary, model = tts_paths
    tts = PiperTTS(TTSConfig(binary_path=str(binary), model_path=str(model)))
    assert tts.synthesize("") == b""


def test_synthesize_returns_stdout_pcm(tts_paths):
    binary, model = tts_paths
    tts = PiperTTS(TTSConfig(binary_path=str(binary), model_path=str(model)))
    fake_pcm = b"\x01\x02" * 100
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=fake_pcm, stderr=b"")
    with mock.patch("voice.tts.piper_tts.run_with_group_kill", return_value=fake_result):
        pcm = tts.synthesize("hello")
    assert pcm == fake_pcm


def test_play_blocks_until_aplay_exits_and_reports_success(tts_paths):
    binary, model = tts_paths
    tts = PiperTTS(TTSConfig(binary_path=str(binary), model_path=str(model)))
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
    with mock.patch("voice.tts.piper_tts.run_with_group_kill", return_value=fake_result) as run_mock:
        ok = tts.play(b"\x00\x00" * 100, alsa_device="default")
    assert ok is True
    run_mock.assert_called_once()


def test_play_reports_failure_on_nonzero_exit(tts_paths):
    binary, model = tts_paths
    tts = PiperTTS(TTSConfig(binary_path=str(binary), model_path=str(model)))
    fake_result = subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"device busy")
    with mock.patch("voice.tts.piper_tts.run_with_group_kill", return_value=fake_result):
        ok = tts.play(b"\x00\x00" * 100, alsa_device="default")
    assert ok is False


def test_play_empty_pcm_is_a_noop_success(tts_paths):
    binary, model = tts_paths
    tts = PiperTTS(TTSConfig(binary_path=str(binary), model_path=str(model)))
    assert tts.play(b"", alsa_device="default") is True


def test_play_missing_aplay_reports_failure(tts_paths):
    binary, model = tts_paths
    tts = PiperTTS(TTSConfig(binary_path=str(binary), model_path=str(model)))
    with mock.patch("voice.tts.piper_tts.run_with_group_kill", side_effect=FileNotFoundError()):
        ok = tts.play(b"\x00\x00" * 100, alsa_device="default")
    assert ok is False
