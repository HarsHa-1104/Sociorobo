"""Unit tests for voice/wake/wake_word.py using a fake openWakeWord model
(no real ONNX model file or network access required -- see
docs/MODEL_DECISION.md for the real-model integration results that
surfaced the bug this test suite now guards against: passing no explicit
``wakeword_models`` list makes openWakeWord silently load its entire
bundled stock model set instead of just the one configured keyword).
"""

from __future__ import annotations

from unittest import mock

import pytest

from voice.config import WakeConfig
from voice.wake.wake_word import WakeWordDetector


class _FakeOWWModel:
    """Records what it was constructed with; predict() reads off a script."""

    last_kwargs = None

    def __init__(self, **kwargs):
        _FakeOWWModel.last_kwargs = kwargs
        self.scripted_scores = []
        self.reset_calls = 0

    def predict(self, frame):
        score = self.scripted_scores.pop(0) if self.scripted_scores else 0.0
        return {"hey_jarvis": score}

    def reset(self):
        self.reset_calls += 1


@pytest.fixture
def fake_oww(monkeypatch):
    _FakeOWWModel.last_kwargs = None
    monkeypatch.setattr("voice.wake.wake_word._OWWModel", _FakeOWWModel)
    return _FakeOWWModel


def test_disabled_detector_never_loads_a_model_or_fires(fake_oww):
    det = WakeWordDetector(WakeConfig(enabled=False), sample_rate=16000)
    assert det.process_frame(b"\x00\x00" * 10) is False
    assert fake_oww.last_kwargs is None


def test_constructor_always_passes_explicit_single_model_list(fake_oww):
    """Regression test for the real bug found during Phase 2 integration
    validation: omitting wakeword_models makes openWakeWord load its
    entire bundled stock set instead of just the configured keyword.
    """
    WakeWordDetector(WakeConfig(model_name="hey_jarvis", model_path=None), sample_rate=16000)
    assert fake_oww.last_kwargs["wakeword_models"] == ["hey_jarvis"]

    WakeWordDetector(WakeConfig(model_path="/opt/wake/hey_arduino.onnx"), sample_rate=16000)
    assert fake_oww.last_kwargs["wakeword_models"] == ["/opt/wake/hey_arduino.onnx"]


# WakeWordDetector.process_frame() buffers input and only calls the model
# once _audio_buffer reaches 1280 samples (2560 bytes) -- matching
# openWakeWord's own documented chunk-size contract (model.py: "audio data
# ... (1280 samples), with longer lengths reducing overall CPU usage"). A
# frame this size drives exactly one model.predict() call per
# process_frame() call below, so scripted_scores lines up 1:1 with calls.
_MODEL_FRAME = b"\x00\x00" * 1280


def test_fires_only_after_trigger_level_consecutive_hits(fake_oww):
    cfg = WakeConfig(model_name="hey_jarvis", threshold=0.5, trigger_level=3)
    det = WakeWordDetector(cfg, sample_rate=16000)
    det._model.scripted_scores = [0.9, 0.9]  # only 2 consecutive hits -- below trigger_level=3
    assert det.process_frame(_MODEL_FRAME) is False
    assert det.process_frame(_MODEL_FRAME) is False

    det._model.scripted_scores = [0.9]  # 3rd consecutive hit -- fires now
    assert det.process_frame(_MODEL_FRAME) is True


def test_single_low_score_frame_resets_the_debounce_counter(fake_oww):
    cfg = WakeConfig(model_name="hey_jarvis", threshold=0.5, trigger_level=3)
    det = WakeWordDetector(cfg, sample_rate=16000)
    det._model.scripted_scores = [0.9, 0.9, 0.1, 0.9, 0.9]
    results = [det.process_frame(_MODEL_FRAME) for _ in range(5)]
    # The dip below threshold at index 2 must reset the streak, so 2
    # trailing high scores (indices 3-4) are not enough to fire.
    assert results == [False, False, False, False, False]


def test_reset_clears_debounce_and_forwards_to_model(fake_oww):
    cfg = WakeConfig(model_name="hey_jarvis", threshold=0.5, trigger_level=2)
    det = WakeWordDetector(cfg, sample_rate=16000)
    det._model.scripted_scores = [0.9]
    det.process_frame(_MODEL_FRAME)  # 1 hit, not yet fired
    det.reset()
    assert det._consecutive_hits == 0
    assert det._model.reset_calls == 1
