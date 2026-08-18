from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from voice.config import STTConfig
from voice.stt.whisper_cpp import WhisperCppSTT, _audio_ctx_for_duration


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
    with mock.patch("voice.stt.whisper_cpp.run_with_group_kill", return_value=fake_result) as run_mock:
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
    with mock.patch("voice.stt.whisper_cpp.run_with_group_kill", return_value=fake_result):
        text = stt.run_stt(b"\x00\x00" * 16000, sample_rate=16000)
    assert text == "hello there"


def test_run_stt_returns_empty_on_nonzero_exit(stt_paths):
    binary, model = stt_paths
    cfg = STTConfig(binary_path=str(binary), model_path=str(model))
    stt = WhisperCppSTT(cfg)

    fake_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
    with mock.patch("voice.stt.whisper_cpp.run_with_group_kill", return_value=fake_result):
        text = stt.run_stt(b"\x00\x00" * 16000, sample_rate=16000)
    assert text == ""


def test_run_stt_returns_empty_on_timeout(stt_paths):
    binary, model = stt_paths
    cfg = STTConfig(binary_path=str(binary), model_path=str(model), timeout_s=1.0)
    stt = WhisperCppSTT(cfg)

    with mock.patch("voice.stt.whisper_cpp.run_with_group_kill", side_effect=subprocess.TimeoutExpired(cmd="whisper-cli", timeout=1.0)):
        text = stt.run_stt(b"\x00\x00" * 16000, sample_rate=16000)
    assert text == ""


def test_run_stt_passes_duration_scaled_audio_ctx(stt_paths):
    """Regression test for the encode-time fix: -ac must be computed from
    the real clip duration (with margin), never omitted or hard-coded --
    an undersized value doesn't error, it makes whisper.cpp's decoder loop
    on repeated garbage (confirmed on real UNO Q hardware, see
    voice/stt/whisper_cpp.py's module docstring for the measurements)."""
    binary, model = stt_paths
    cfg = STTConfig(binary_path=str(binary), model_path=str(model))
    stt = WhisperCppSTT(cfg)

    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr="")
    with mock.patch("voice.stt.whisper_cpp.run_with_group_kill", return_value=fake_result) as run_mock:
        # 1 second of 16kHz mono int16 audio.
        stt.run_stt(b"\x00\x00" * 16000, sample_rate=16000)

    called_args = run_mock.call_args[0][0]
    assert "-ac" in called_args
    ac_value = int(called_args[called_args.index("-ac") + 1])
    assert ac_value == _audio_ctx_for_duration(1.0)


@pytest.mark.parametrize("duration_s,expected", [
    (0.1, 70),
    (1.5, 161),
    (6.0, 454),
    (10.15, 723),
    (20.0, 1364),
    (100.0, 1500),  # clamped to whisper's own 30s/1500-frame hard cap
])
def test_audio_ctx_formula_matches_benchmarked_values(duration_s, expected):
    """Pins the exact values measured/validated on real UNO Q hardware --
    a 10.15s clip with an undersized context (256 frames, ~5s worth) made
    whisper.cpp repeat the first few seconds of transcript 3x instead of
    erroring; -ac 723 (this formula's output for 10.15s) produced the
    correct full transcript. See docs/MODEL_DECISION.md-style evidence in
    the Milestone 3 commit for the raw benchmark."""
    assert _audio_ctx_for_duration(duration_s) == expected
