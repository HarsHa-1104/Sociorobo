"""Lightweight, always-on wake-word detection.

Phase 1 found no wake-word implementation anywhere in the old project --
this is entirely new. See docs/MODEL_DECISION.md for the full evaluation;
summary of the decision encoded here:

  * Engine: openWakeWord (ONNX runtime, CPU-only, small per-keyword models
    ~1-2 MB, designed for always-on embedded use). This is the only
    wake-word engine actually evaluated/installed in this repo -- if a
    different engine is substituted later, only this file should need to
    change (VoiceManager only depends on the ``WakeWordDetector`` interface
    below, not on openWakeWord specifically).
  * Default shipped model: a stock, pre-trained keyword ("hey_jarvis" by
    default) known to have a validated low false-accept rate, because it
    ships with the openWakeWord project and has real usage behind it.
  * Custom "hey_arduino": openWakeWord supports training a same-architecture
    custom model from synthesized speech, which is cost-neutral at
    inference time (identical model shape/size to a stock keyword -- see
    docs/MODEL_DECISION.md). Training was NOT executed in this repo (it
    needs a synthetic-data generation pipeline and GPU-friendly training
    time that a cloud sandbox audit session isn't the right place to run
    unsupervised). scripts/train_custom_wake_word.md documents the exact
    procedure. Ship with the stock model, swap in a custom one only after
    it clears the false-accept/false-reject validation test in
    tests/test_wake_word_validation.py (manual, on-device, real audio).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from voice.config import WakeConfig

logger = logging.getLogger(__name__)

try:
    from openwakeword.model import Model as _OWWModel  # type: ignore
except ImportError:  # pragma: no cover - optional until installed
    _OWWModel = None


class WakeWordDetector:
    """Wraps an openWakeWord model behind a simple frame-in/bool-out interface.

    This class intentionally does *not* know anything about audio devices
    or sessions -- ``AudioManager`` supplies frames, ``VoiceManager`` decides
    when this detector is even being fed (Section 7: it must be fully
    disabled during an active voice session, never just ignored).
    """

    def __init__(self, config: WakeConfig, sample_rate: int) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self._model: Optional["_OWWModel"] = None
        self._consecutive_hits = 0
        self._audio_buffer = bytearray()
        self._model_frame_samples = 1280

        if not config.enabled:
            logger.info("Wake-word detection disabled by configuration.")
            return

        if _OWWModel is None:
            raise RuntimeError(
                "openwakeword is not installed. Install it via "
                "scripts/setup_uno_q.sh before enabling wake-word detection, "
                "or set wake.enabled=false to run STT/LLM/TTS-only for testing."
            )

        # IMPORTANT: always pass an explicit wakeword_models list. Leaving it
        # unset makes openWakeWord load its *entire* bundled stock model set
        # (alexa, hey_jarvis, hey_mycroft, timer, weather, ...) -- multiple
        # models resident and being evaluated on every frame, which wastes
        # RAM/CPU and is not what a single-keyword, always-on embedded
        # detector should be doing (Section 6: keep this component as light
        # as possible, it's the only thing running continuously).
        selector = config.model_path or config.model_name
        kwargs = {
            "inference_framework": config.inference_framework,
            "wakeword_models": [selector],
        }
        self._model = _OWWModel(**kwargs)
        logger.info(
            "WakeWordDetector ready: engine=%s model=%s threshold=%.2f trigger_level=%d",
            config.engine,
            config.model_path or config.model_name,
            config.threshold,
            config.trigger_level,
        )

    # ------------------------------------------------------------------
    def process_frame(self, frame: bytes) -> bool:
        """Feed one raw int16 mono PCM frame. Returns True on a confirmed wake.

        Confirmation requires ``trigger_level`` consecutive frames above
        ``threshold`` -- a simple debounce so a single noisy frame can't
        fire a false wake (Section 6: false-positive risk matters as much
        as raw detection).
        """
        if not self.config.enabled or self._model is None:
            return False

        # AudioManager supplies 30 ms / 480-sample frames.
        # openWakeWord is most reliable when fed 80 ms / 1280-sample chunks.
        self._audio_buffer.extend(frame)

        bytes_per_model_frame = self._model_frame_samples * 2

        while len(self._audio_buffer) >= bytes_per_model_frame:
            chunk = bytes(
                self._audio_buffer[:bytes_per_model_frame]
            )
            del self._audio_buffer[:bytes_per_model_frame]

            audio = np.frombuffer(chunk, dtype=np.int16)
            predictions = self._model.predict(audio)

            score = 0.0
            for name, value in predictions.items():
                if (
                    self.config.model_name in name
                    or self.config.model_path in (None, name)
                ):
                    score = max(score, float(value))

            if not predictions:
                continue

            if score == 0.0:
                score = max(float(v) for v in predictions.values())

            logger.debug("wake score=%.4f", score)

            if score >= self.config.threshold:
                self._consecutive_hits += 1
            else:
                self._consecutive_hits = 0

            if self._consecutive_hits >= self.config.trigger_level:
                self._consecutive_hits = 0
                self.reset()
                return True

        return False

    def reset(self) -> None:
        """Clear internal debounce/model buffers, e.g. after a wake fires."""
        self._consecutive_hits = 0
        self._audio_buffer.clear()
        if self._model is not None and hasattr(self._model, "reset"):
            self._model.reset()


__all__ = ["WakeWordDetector"]
