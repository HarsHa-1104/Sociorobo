from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from voice.config import STTConfig
from voice.stt.whisper_cpp import WhisperCppSTT


@pytest.fixture
def stt_paths(tmp_path):
    binary = tmp_path / "whisper-cli"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    model = tmp_path / "ggml-base.en-q5_0.bin"
    model.write_bytes(b"fake-model-bytes")
    return binary, model


def test_missing_binary_raises_filenotfound(tmp_path):
    cfg = STTConfig(binary_path=str(tmp_path / "nope"), model_path=str(tmp_path / "also_nope.bin"))
    with pytest.raises(FileNotFoundError):
        WhisperCppSTT(cfg)


def test_missing_model_raises_filenotfound(tmp_path, stt_paths):
    binary, _ = stt_paths
    cfg = STTConfig(binary_path=str(binary), model_path=str(tmp_path / "missing.bin"))
    with pytest.raises(FileNotFoundError):
        WhisperCppSTT(cfg)


def test_run_stt_returns_empty_on_empty_input(stt_paths):
    binary, model = stt_paths
    cfg = STTConfig(binary_path=str(binary), model_path=str(model))
    stt = WhisperCppSTT(cfg)
    assert stt.run_stt(b"", sample_rate=16000) == ""


def test_run_stt_parses_clean_transcript(stt_paths):
    binary, model = stt_paths
    cfg = STTConfig(binary_path=str(binary), model_path=str(model))
    stt = WhisperCppSTT(cfg)

    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=" what is the weather tomorrow \n", stderr=""
    )
    with mock.patch("subprocess.run", return_value=fake_result) as run_mock:
        text = stt.run_stt(b"\x00\x00" * 16000, sample_rate=16000)
    assert text == "what is the weather tomorrow"
    run_mock.assert_called_once()


def test_run_stt_strips_blank_audio_markers_and_ansi(stt_paths):
    binary, model = stt_paths
    cfg = STTConfig(binary_path=str(binary), model_path=str(model))
    stt = WhisperCppSTT(cfg)

    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout="[BLANK_AUDIO]\n\x1b[32mhello there\x1b[0m\n",
        stderr="",
    )
    with mock.patch("subprocess.run", return_value=fake_result):
        text = stt.run_stt(b"\x00\x00" * 16000, sample_rate=16000)
    assert text == "hello there"


def test_run_stt_returns_empty_on_nonzero_exit(stt_paths):
    binary, model = stt_paths
    cfg = STTConfig(binary_path=str(binary), model_path=str(model))
    stt = WhisperCppSTT(cfg)

    fake_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
    with mock.patch("subprocess.run", return_value=fake_result):
        text = stt.run_stt(b"\x00\x00" * 16000, sample_rate=16000)
    assert text == ""


def test_run_stt_returns_empty_on_timeout(stt_paths):
    binary, model = stt_paths
    cfg = STTConfig(binary_path=str(binary), model_path=str(model), timeout_s=1.0)
    stt = WhisperCppSTT(cfg)

    with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="whisper-cli", timeout=1.0)):
        text = stt.run_stt(b"\x00\x00" * 16000, sample_rate=16000)
    assert text == ""
