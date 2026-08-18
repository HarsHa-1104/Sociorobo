"""Speech-to-text via a whisper.cpp subprocess.

Ported from the old project's ``audio/stt_cpp.py``, which Phase 1 found to
be clean, GUI-free, and already well error-handled -- kept close to
verbatim. Changes made here:

  * No hard-coded Jetson-era defaults beyond the same ``/opt/...``
    convention (which was already environment-overridable, not Jetson-
    specific) -- paths come from ``STTConfig`` now instead of module-level
    constants, so tests can inject fakes without touching env vars.
  * Timeout shortened per Section 11/25 (configurable, default 15s instead
    of 30s) -- a short spoken command should never legitimately take that
    long to transcribe, and a robot standing still waiting for its own
    turn should fail fast rather than hang.
  * Model choice (base.en vs tiny.en) is now a config value, not a code
    change -- see docs/MODEL_DECISION.md for the evidence behind the
    default.
"""

from __future__ import annotations

import logging
import re
import struct
import subprocess
import tempfile
from pathlib import Path

from voice.config import STTConfig
from voice.subprocess_utils import run_with_group_kill

logger = logging.getLogger(__name__)


class STTError(RuntimeError):
    pass


def _write_wav_header(f, num_samples: int, sample_rate: int) -> None:
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    chunk_size = 36 + data_size

    f.write(b"RIFF")
    f.write(struct.pack("<I", chunk_size))
    f.write(b"WAVE")
    f.write(b"fmt ")
    f.write(struct.pack("<I", 16))
    f.write(struct.pack("<H", 1))
    f.write(struct.pack("<H", num_channels))
    f.write(struct.pack("<I", sample_rate))
    f.write(struct.pack("<I", byte_rate))
    f.write(struct.pack("<H", block_align))
    f.write(struct.pack("<H", bits_per_sample))
    f.write(b"data")
    f.write(struct.pack("<I", data_size))


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# whisper.cpp's encoder always processes a fixed context window sized in
# 20ms mel-spectrogram frames (50 frames/sec) regardless of actual audio
# length -- n_audio_ctx=1500 is the model's built-in 30s cap. Benchmarked
# on this UNO Q (base.en-q5_1, 4 threads): a 1.5s clip at the default full
# context takes ~10s to encode; explicitly capping the context to roughly
# match the real clip length (via whisper-cli's -ac flag) cuts that to
# ~1.3s with an identical transcript -- encode time is the dominant cost
# in the whole STT call (>85% of wall time), so this is the single biggest
# lever found so far. Capping too aggressively is a *silent* failure mode,
# not an error: a 10s clip fed -ac 256 (only enough context for ~5s)
# didn't error, it made the decoder loop and repeat the first few seconds
# of transcript three times. Must always be computed from the real audio
# duration, with margin -- never a fixed constant.
_MEL_FRAMES_PER_SECOND = 50
_AUDIO_CTX_MARGIN = 1.3   # 30% headroom over the raw duration
_AUDIO_CTX_MIN_PAD_FRAMES = 64  # +~1.3s flat pad, so very short clips still get a safe cushion
_AUDIO_CTX_MAX_FRAMES = 1500    # whisper's own hard cap (30s)


def _audio_ctx_for_duration(duration_s: float) -> int:
    frames = int(duration_s * _MEL_FRAMES_PER_SECOND * _AUDIO_CTX_MARGIN) + _AUDIO_CTX_MIN_PAD_FRAMES
    return min(frames, _AUDIO_CTX_MAX_FRAMES)


class WhisperCppSTT:
    """Transcribe a speech segment using a local whisper.cpp binary."""

    def __init__(self, config: STTConfig) -> None:
        self.config = config
        self.binary = Path(config.binary_path)
        self.model = Path(config.model_path)

        if not self.binary.exists():
            raise FileNotFoundError(
                f"whisper.cpp binary not found at '{self.binary}'. "
                "Run scripts/setup_uno_q.sh, or set stt.binary_path."
            )
        if not self.model.exists():
            raise FileNotFoundError(
                f"whisper.cpp model not found at '{self.model}'. "
                "Run scripts/pull_models.sh, or set stt.model_path."
            )

        logger.info(
            "WhisperCppSTT ready: binary=%s model=%s threads=%d",
            self.binary.name, self.model.name, config.threads,
        )

    # ------------------------------------------------------------------
    def run_stt(self, raw_bytes: bytes, sample_rate: int) -> str:
        """Transcribe raw int16 mono PCM. Returns '' on empty input or failure."""
        if not raw_bytes:
            return ""

        num_samples = len(raw_bytes) // 2
        duration_s = num_samples / sample_rate
        audio_ctx = _audio_ctx_for_duration(duration_s)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            _write_wav_header(tmp, num_samples, sample_rate)
            tmp.write(raw_bytes)

        try:
            # run_with_group_kill, not subprocess.run: a plain subprocess.run
            # timeout only kills whisper-cli itself, not any child process it
            # might spawn -- confirmed on real hardware during Milestone 7
            # that a timeout-killed process can leave orphaned children
            # running indefinitely.
            result = run_with_group_kill(
                [
                    str(self.binary),
                    "-m", str(self.model),
                    "-f", str(tmp_path),
                    "-t", str(self.config.threads),
                    "-l", self.config.language,
                    "-ac", str(audio_ctx),
                    "--no-timestamps",
                    "-np",
                ],
                text=True,
                timeout=self.config.timeout_s,
            )
        except subprocess.TimeoutExpired:
            logger.warning("whisper.cpp timed out after %.1fs -- skipping segment.", self.config.timeout_s)
            return ""
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("whisper.cpp invocation failed: %s", exc)
            return ""
        finally:
            tmp_path.unlink(missing_ok=True)

        if result.returncode != 0:
            logger.warning("whisper.cpp error: %s", result.stderr.strip())
            return ""

        cleaned = []
        for line in result.stdout.splitlines():
            line = _ANSI_RE.sub("", line).strip()
            if not line or (line.startswith("[") and line.endswith("]")):
                continue
            cleaned.append(line)

        return " ".join(cleaned).strip()


__all__ = ["WhisperCppSTT", "STTError"]
