"""Text-to-speech via a Piper subprocess -- headless, deterministic completion.

Ported from the old project's ``audio/tts_piper.py``. Phase 1 classified
that file's amplitude-callback plumbing as REFACTOR, not KEEP: the RMS
computation and the ``for level in levels: callback(level); time.sleep(...)``
loop existed solely to drive lip-sync animation, and that per-chunk sleep
loop was never actually the source of truth for "playback finished" -- the
real completion signal was always the ``aplay`` subprocess exiting.

This version removes the amplitude callback and the chunking-for-animation
entirely. Playback completion is now determined the direct way: write the
full PCM buffer to ``aplay``'s stdin and block on ``proc.wait()``. This is
both simpler and *more* correct for Section 13 of the spec ("playback
completion must be deterministic, not inferred from a GUI animation
callback").
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from voice.config import TTSConfig
from voice.subprocess_utils import run_with_group_kill

logger = logging.getLogger(__name__)


class TTSError(RuntimeError):
    pass


class PiperTTS:
    """Synthesise speech with Piper and play it back synchronously via ALSA."""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config
        self.binary = Path(config.binary_path)
        self.model = Path(config.model_path)
        self.model_json = (
            Path(config.model_json_path) if config.model_json_path
            else Path(str(self.model) + ".json")
        )

        if not self.binary.exists():
            raise FileNotFoundError(
                f"Piper binary not found at '{self.binary}'. "
                "Run scripts/setup_uno_q.sh, or set tts.binary_path."
            )
        if not self.model.exists():
            raise FileNotFoundError(
                f"Piper voice model not found at '{self.model}'. "
                "Run scripts/pull_models.sh, or set tts.model_path."
            )

        logger.info(
            "PiperTTS ready: model=%s rate=%dHz (output ALSA device is supplied per-call)",
            self.model.name, config.sample_rate,
        )

    # ------------------------------------------------------------------
    def synthesize(self, text: str) -> bytes:
        """Return raw int16 mono PCM bytes for *text*, or b'' on failure/empty input."""
        if not text.strip():
            return b""

        cmd = [
            str(self.binary),
            "--model", str(self.model),
            "--config", str(self.model_json),
            "--output-raw",
            "--quiet",
        ]
        try:
            # run_with_group_kill: Piper's espeak-ng phonemizer backend can
            # shell out to a separate process -- a plain subprocess.run
            # timeout would only kill the piper process itself and orphan
            # that child (confirmed on real hardware during Milestone 7 with
            # an equivalent hung-shell-script scenario).
            result = run_with_group_kill(
                cmd,
                input=text.encode("utf-8"),
                timeout=self.config.timeout_s,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Piper synthesis timed out after %.1fs.", self.config.timeout_s)
            return b""

        if result.returncode != 0:
            logger.error("Piper error: %s", result.stderr.decode(errors="replace").strip())
            return b""

        return result.stdout

    # ------------------------------------------------------------------
    def play(self, pcm_bytes: bytes, alsa_device: str) -> bool:
        """Play raw PCM through ALSA and block until playback is fully finished.

        Returns True if playback completed cleanly, False on any failure.
        This is the ONLY signal the Voice Manager state machine trusts for
        "TTS is done" -- see Section 13/28: resume must never happen before
        this returns.
        """
        if not pcm_bytes:
            return True

        aplay_cmd = [
            "aplay",
            "-D", alsa_device,
            "-f", "S16_LE",
            "-r", str(self.config.sample_rate),
            "-c", "1",
            "-t", "raw",
            "-q",
        ]
        try:
            proc = run_with_group_kill(
                aplay_cmd,
                input=pcm_bytes,
                timeout=self.config.timeout_s,
            )
        except FileNotFoundError:
            logger.error("`aplay` not found -- is alsa-utils installed?")
            return False
        except subprocess.TimeoutExpired:
            logger.error("aplay playback exceeded timeout -- killed.")
            return False

        if proc.returncode != 0:
            logger.error("aplay exited %d: %s", proc.returncode, proc.stderr.decode(errors="replace"))
            return False
        return True

    def synthesize_and_play(self, text: str, alsa_device: str) -> bool:
        """Convenience wrapper: synthesize, then block until playback finishes.

        Logs the synth/play split explicitly (Milestone 8 baseline
        profiling) -- VoiceManager only sees the combined TTS stage
        duration, and Piper's per-call subprocess/model-load overhead was
        flagged back in Milestone 5 as a real, unconfirmed suspicion; this
        makes it measurable instead of assumed.
        """
        t0 = time.perf_counter()
        pcm = self.synthesize(text)
        synth_s = time.perf_counter() - t0
        if not pcm:
            logger.info("TTS timing: synth=%.2fs play=skipped (empty synthesis)", synth_s)
            return False

        t0 = time.perf_counter()
        ok = self.play(pcm, alsa_device)
        play_s = time.perf_counter() - t0
        audio_s = len(pcm) / 2 / self.config.sample_rate
        logger.info(
            "TTS timing: synth=%.2fs play=%.2fs audio_len=%.2fs synth_rtf=%.2f",
            synth_s, play_s, audio_s, (synth_s / audio_s if audio_s else 0.0),
        )
        return ok


__all__ = ["PiperTTS", "TTSError"]
