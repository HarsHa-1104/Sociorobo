"""Text-to-speech via a long-lived Piper subprocess (Milestone 8).

Measured directly on real UNO Q hardware (Piper's own --debug output,
not assumed): loading the ONNX voice model costs ~1.9-2.4s, every single
time, regardless of how short the text is. The original PiperTTS
(voice/tts/piper_tts.py) spawns a fresh Piper process per call, so every
synthesis pays that cost -- confirmed by a 3-run repeat benchmark showing
zero improvement run-to-run at any text length (see the Milestone 8
commit for the full baseline).

This class keeps ONE Piper process alive for the process's lifetime,
using Piper's own `--json-input` + `--output_file -` mode (verified
directly: a second request to an already-running process took 0.83s
with no reload, versus ~2-3s the first request paid). Falls back to
restarting the persistent process if it ever dies or a request times
out, so one bad request can't permanently break TTS -- and every restart
still goes through the same model-load cost exactly once, not per call.

Kept as a separate class rather than replacing PiperTTS outright, so the
original one-shot-per-call implementation stays available and this can
be reverted to it in one line if real-world testing ever shows a
regression (see config.tts.persistent).
"""

from __future__ import annotations

import json
import logging
import os
import select
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from voice.audio.combination import ComboGuard
from voice.audio.discovery import discover_output_devices, pipewire_reachable
from voice.audio.selection import SelectionError, make_output_selector
from voice.config import TTSConfig

logger = logging.getLogger(__name__)


class PersistentPiperTTS:
    """Synthesise speech with a long-lived Piper subprocess; play back via
    a fresh `aplay` call per response (aplay itself is cheap -- see
    Milestone 8 baseline, `play` duration is essentially just the audio's
    own length, not meaningful subprocess overhead, so only the Piper
    side benefits from staying resident).
    """

    def __init__(self, config: TTSConfig, combo_guard: Optional[ComboGuard] = None) -> None:
        self.config = config
        # Shared with AudioManager (via build_voice_manager) so speaker
        # selection can see which backend the microphone is currently
        # using and enforce the "at most one side may be Bluetooth"
        # product rule (voice/audio/combination.py). None (the default)
        # means no cross-role constraint is applied -- used by tests and
        # any standalone use of this class that doesn't need it.
        self._combo_guard = combo_guard
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

        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        # Plug-and-play Phase 1: one output selector for this TTS instance's
        # lifetime, so "sticky previous selection" (see voice/audio/selection.py)
        # actually means something across calls instead of resetting every time.
        self._output_selector = make_output_selector()

        # Load eagerly at construction time -- the ~2s model-load cost
        # happens once at app startup, not on the first real user request.
        self._ensure_process()

        logger.info(
            "PersistentPiperTTS ready: model=%s rate=%dHz (Piper process stays "
            "resident across requests; output ALSA device is supplied per-call)",
            self.model.name, config.sample_rate,
        )

    # ------------------------------------------------------------------
    def _spawn(self) -> subprocess.Popen:
        cmd = [
            str(self.binary),
            "--model", str(self.model),
            "--config", str(self.model_json),
            "--output_file", "-",
            "--json-input",
            "--quiet",
        ]
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # matches run_with_group_kill's Milestone 7 rationale
        )

    def _ensure_process(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            if self._proc is not None:
                logger.warning("Persistent Piper process was not alive -- restarting it "
                                "(this pays the model-load cost again, once).")
            self._proc = self._spawn()

    def _kill_process(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.kill()
            self._proc.wait(timeout=2.0)
        except Exception:
            pass
        self._proc = None

    # ------------------------------------------------------------------
    @staticmethod
    def _read_exact(fd: int, n: int, deadline: float) -> Optional[bytes]:
        """Read exactly n bytes from fd, bounded by deadline (monotonic
        time). Returns None on timeout or EOF (process died) -- never
        blocks past the deadline, unlike a plain stream.read(n)."""
        buf = bytearray()
        while len(buf) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            r, _, _ = select.select([fd], [], [], remaining)
            if not r:
                return None
            chunk = os.read(fd, n - len(buf))
            if not chunk:
                return None  # EOF
            buf.extend(chunk)
        return bytes(buf)

    def _read_one_wav(self, timeout_s: float) -> Optional[bytes]:
        deadline = time.monotonic() + timeout_s
        fd = self._proc.stdout.fileno()
        header = self._read_exact(fd, 12, deadline)
        if header is None or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            return None
        riff_size = struct.unpack("<I", header[4:8])[0]
        rest = self._read_exact(fd, riff_size - 4, deadline)
        if rest is None:
            return None
        return header + rest

    @staticmethod
    def _extract_pcm(wav_bytes: bytes) -> bytes:
        """Parses RIFF/WAVE chunks properly to find 'data' rather than
        assuming a fixed offset -- chunk order isn't format-guaranteed
        even though Piper's own output is consistent in practice."""
        pos = 12
        while pos + 8 <= len(wav_bytes):
            chunk_id = wav_bytes[pos:pos + 4]
            chunk_size = struct.unpack("<I", wav_bytes[pos + 4:pos + 8])[0]
            data_start = pos + 8
            if chunk_id == b"data":
                return wav_bytes[data_start:data_start + chunk_size]
            pos = data_start + chunk_size + (chunk_size % 2)  # chunks are word-aligned
        return b""

    # ------------------------------------------------------------------
    def synthesize(self, text: str) -> bytes:
        """Return raw int16 mono PCM bytes for *text*, or b'' on failure/empty input."""
        if not text.strip():
            return b""

        self._ensure_process()
        with self._lock:
            try:
                request = json.dumps({"text": text}) + "\n"
                self._proc.stdin.write(request.encode("utf-8"))
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                logger.warning("Persistent Piper stdin write failed (%s) -- restarting process.", exc)
                self._kill_process()
                return b""

            wav = self._read_one_wav(self.config.timeout_s)

        if wav is None:
            logger.warning(
                "Persistent Piper produced no valid response within %.1fs -- "
                "restarting process for the next request.", self.config.timeout_s,
            )
            self._kill_process()
            return b""

        return self._extract_pcm(wav)

    # ------------------------------------------------------------------
    def play(self, pcm_bytes: bytes, alsa_device: str) -> bool:
        """Plug-and-play Phase 1: routes to whichever output device
        voice/audio/discovery.py + selection.py currently resolve as the
        best available PipeWire sink -- Bluetooth or ALSA/USB alike, never
        a hardcoded MAC or sink id. `config.tts.speaker_mode` controls
        whether that's a free automatic pick ("auto", the default) or
        restricted to one explicitly pinned device ("pinned").

        `alsa_device` is kept only as a last-resort fallback for the rare
        case PipeWire itself is unreachable (not just "no sinks right
        now") -- see _resolve_output_target(). It is never used while
        PipeWire is working, even if PipeWire currently reports zero
        sinks: that case fails clearly instead of silently playing to a
        fixed ALSA device nobody may be listening to (confirmed on real
        hardware, Milestone 8: the built-in HPH jack is unused/off).
        """
        if not pcm_bytes:
            return True

        target_node, failure_reason = self._resolve_output_target()

        if target_node is not None:
            from voice.tts.pipewire_playback import play_via_pipewire
            return play_via_pipewire(pcm_bytes, self.config.sample_rate, target_node, self.config.timeout_s)

        if failure_reason == "pipewire_unreachable" and alsa_device:
            logger.warning(
                "PipeWire itself is unreachable (pw-dump missing/erroring) -- "
                "falling back to the configured ALSA device %s as a last "
                "resort. This is a degraded path, not normal operation.",
                alsa_device,
            )
            return self._play_via_aplay(pcm_bytes, alsa_device)

        logger.error(
            "No usable output device selected (%s) -- not falling back to "
            "an unintended device.", failure_reason,
        )
        return False

    def _resolve_output_target(self) -> tuple[Optional[str], Optional[str]]:
        """Returns (pipewire_node_name, None) on success, or
        (None, failure_reason) on failure. failure_reason is one of:
        "pipewire_unreachable" (pw-dump itself is missing/erroring/timing
        out -- discovery can't even ask PipeWire), "no_devices" (PipeWire
        is fine, it just currently has zero sinks), "bluetooth_conflict"
        (every currently available output is Bluetooth, but the
        microphone is already using Bluetooth -- Bluetooth mic + Bluetooth
        speaker is not a supported combination, see
        voice/audio/combination.py), or "selection_failed" (candidates
        exist but the configured pin doesn't match any of them, or none
        pass some other constraint). Distinguishing "pipewire_unreachable"
        from the rest matters because only that one is eligible for the
        aplay fallback in play() -- see that method's docstring. None of
        the others fall back to aplay: they all mean a real device
        landscape was seen and rejected for a specific reason, not that
        PipeWire itself is unusable, so silently degrading to a fixed
        ALSA device would be exactly the "unintended device" this design
        exists to avoid.
        """
        candidates = discover_output_devices()
        if not candidates:
            if pipewire_reachable():
                return None, "no_devices"
            return None, "pipewire_unreachable"

        is_allowed = (
            (lambda d: self._combo_guard.speaker_allowed(d.backend))
            if self._combo_guard is not None else None
        )
        pin = self.config.speaker_pin or self.config.bluetooth_speaker_mac
        try:
            chosen = self._output_selector.select(
                candidates, mode=self.config.speaker_mode, pin=pin, is_allowed=is_allowed,
            )
        except SelectionError as exc:
            if (
                self._combo_guard is not None
                and self._combo_guard.microphone_backend == "bluez5"
                and all(d.backend == "bluez5" for d in candidates)
            ):
                logger.error(
                    "Speaker selection failed: every currently available output "
                    "is Bluetooth, but the microphone is already using Bluetooth -- "
                    "Bluetooth mic + Bluetooth speaker is not a supported "
                    "combination. (%s)", exc,
                )
                return None, "bluetooth_conflict"
            logger.error("Speaker selection failed: %s", exc)
            return None, "selection_failed"

        if self._combo_guard is not None:
            self._combo_guard.set_speaker_backend(chosen.backend)
        return chosen.pipewire_node_name, None

    def _play_via_aplay(self, pcm_bytes: bytes, alsa_device: str) -> bool:
        from voice.subprocess_utils import run_with_group_kill

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
            proc = run_with_group_kill(aplay_cmd, input=pcm_bytes, timeout=self.config.timeout_s)
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
        t0 = time.perf_counter()
        pcm = self.synthesize(text)
        synth_s = time.perf_counter() - t0
        if not pcm:
            logger.info("TTS timing (persistent): synth=%.2fs play=skipped (empty synthesis)", synth_s)
            return False

        t0 = time.perf_counter()
        ok = self.play(pcm, alsa_device)
        play_s = time.perf_counter() - t0
        audio_s = len(pcm) / 2 / self.config.sample_rate
        logger.info(
            "TTS timing (persistent): synth=%.2fs play=%.2fs audio_len=%.2fs synth_rtf=%.2f",
            synth_s, play_s, audio_s, (synth_s / audio_s if audio_s else 0.0),
        )
        return ok

    def shutdown(self) -> None:
        """Explicitly stop the persistent Piper process. Not called
        automatically -- the process is a daemon of sorts for the
        VoiceManager's lifetime; call this on clean app shutdown if you
        want to avoid leaving it running until the parent process exits."""
        with self._lock:
            self._kill_process()


__all__ = ["PersistentPiperTTS"]
