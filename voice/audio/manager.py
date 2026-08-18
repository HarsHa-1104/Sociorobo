"""Shared Audio Manager -- the single owner of the microphone."""

from __future__ import annotations

import logging
import threading
from typing import Iterator, Optional

import numpy as np
import samplerate

from voice.config import AudioConfig

logger = logging.getLogger(__name__)

try:
    import pyaudio
except ImportError:  # pragma: no cover
    pyaudio = None  # type: ignore


class MicrophoneUnavailableError(RuntimeError):
    """Raised when the configured input device cannot be opened."""


class MicrophoneRecoveryFailedError(RuntimeError):
    """Raised when the mic stream broke mid-session and bounded recovery
    (reopen attempts + backoff) was exhausted without success.

    This is a deliberate, visible failure -- see the Milestone 7 fix in
    frames()/_attempt_recovery() for why silently retrying forever instead
    is not acceptable (confirmed on real hardware: it also prevents
    shutdown from propagating).
    """


class AudioManager:
    """Owns one microphone stream and outputs 16 kHz PCM frames.

    The UNO Q USB microphone runs at 48 kHz through PortAudio.
    The voice pipeline expects 16 kHz, so audio is resampled here.
    """

    PIPELINE_SAMPLE_RATE = 16000

    # Bounded fault-tolerance for a broken mic stream (Milestone 7). Confirmed
    # on real UNO Q hardware with a physical USB unplug/replug: without these
    # bounds, frames() enters an unbounded ~31ms-spaced retry loop that never
    # recovers even after the device is reconnected -- and because that inner
    # loop only re-checks its own condition every iteration rather than
    # blocking on the real stop event, a mic failure could make the whole
    # process ignore a shutdown signal and hang indefinitely.
    MAX_CONSECUTIVE_READ_ERRORS = 10  # ~300ms of continuous failure at 30ms frames before escalating to recovery
    MAX_REOPEN_ATTEMPTS = 5
    REOPEN_BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 8.0)  # bounded backoff, capped at 8s per attempt
    REOPEN_TIMEOUT_S = 3.0  # see _open_stream_on_existing_pa: pa.open() against a genuinely-
                             # absent device does not raise, it blocks forever -- confirmed on
                             # real hardware, a plain try/except cannot catch a hang.

    # Real hardware finding (Milestone 7): repeatedly tearing down and
    # recreating pyaudio.PyAudio() itself -- a full PortAudio/ALSA host
    # re-init -- while the USB mic was still physically absent was
    # confirmed to kill the whole process with no Python-level exception
    # or traceback at all (consistent with a native crash, not a hang;
    # journalctl -k showed the kernel's own USB disconnect/reconnect
    # timeline straddling the exact moment the process went silent).
    # Recovery therefore never calls pyaudio.PyAudio() again after the
    # first successful start() -- only pa.open() for a fresh *stream* on
    # the SAME already-initialized PyAudio host instance -- and gates even
    # that behind a PortAudio-free presence check via /proc/asound/cards,
    # so PyAudio is never touched at all while the device is confirmed
    # absent at the kernel level.
    ASOUND_CARDS_PATH = "/proc/asound/cards"

    def __init__(self, config: AudioConfig) -> None:
        if pyaudio is None:
            raise RuntimeError(
                "pyaudio is not installed. Install PortAudio + PyAudio first."
            )

        self.config = config

        # Hardware frame size.
        self.capture_frame_size = int(
            config.sample_rate * config.frame_duration_ms / 1000
        )

        # Pipeline frame size: always 16 kHz.
        self.frame_size = int(
            self.PIPELINE_SAMPLE_RATE * config.frame_duration_ms / 1000
        )

        self._pa: Optional["pyaudio.PyAudio"] = None
        self._stream = None
        # Captured on a successful start()/recovery -- the USB-Audio card
        # line(s) from /proc/asound/cards, used as a cheap, PortAudio-free
        # "is the mic physically present" check during recovery. None if
        # the configured device isn't backed by a USB-Audio card (e.g. a
        # non-USB input) -- the presence check then always passes through,
        # so this is a pure optimization/safeguard, never a hard blocker
        # for non-USB setups.
        self._usb_audio_signature: Optional[str] = None

        # Keep one resampler alive for the entire microphone stream.
        # libsamplerate needs its internal state preserved across chunks.
        self._resampler = samplerate.Resampler(
            converter_type="sinc_fastest",
            channels=1,
        )
        self._resample_ratio = (
            self.PIPELINE_SAMPLE_RATE / self.config.sample_rate
        )
        self._suspended = threading.Event()
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return

            self._pa = pyaudio.PyAudio()

            try:
                self._stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    frames_per_buffer=self.capture_frame_size,
                    input_device_index=self.config.input_device_index,
                )
            except Exception as exc:
                self._pa.terminate()
                self._pa = None
                raise MicrophoneUnavailableError(
                    f"Could not open microphone "
                    f"(device_index={self.config.input_device_index}, "
                    f"rate={self.config.sample_rate}): {exc}"
                ) from exc

            self._stop_flag.clear()
            self._usb_audio_signature = self._capture_usb_audio_signature()

            logger.info(
                "AudioManager started: device=%s capture=%dHz -> pipeline=%dHz "
                "frame=%dms (%d samples)",
                self.config.input_device_index,
                self.config.sample_rate,
                self.PIPELINE_SAMPLE_RATE,
                self.config.frame_duration_ms,
                self.frame_size,
            )

    def stop(self) -> None:
        with self._lock:
            self._stop_flag.set()

            if self._stream is not None:
                try:
                    if self._stream.is_active():
                        self._stream.stop_stream()
                    self._stream.close()
                except Exception:
                    logger.exception("Error closing audio stream")

                self._stream = None

            if self._pa is not None:
                self._pa.terminate()
                self._pa = None

            logger.info("AudioManager stopped")

    def suspend(self) -> None:
        self._suspended.set()

        if self._stream is not None and self._stream.is_active():
            self._stream.stop_stream()

    def resume(self) -> None:
        self._suspended.clear()

        if self._stream is not None and not self._stream.is_active():
            self._stream.start_stream()

    @property
    def is_suspended(self) -> bool:
        return self._suspended.is_set()

    def frames(self) -> Iterator[bytes]:
        """Yield 16 kHz mono int16 PCM frames.

        Raises MicrophoneRecoveryFailedError if the stream breaks and
        bounded recovery (see _attempt_recovery) is exhausted -- this is a
        deliberate, visible failure so the caller/process can restart
        cleanly, not a silent infinite retry.
        """

        if self._stream is None:
            raise RuntimeError(
                "AudioManager.start() must be called before frames()"
            )

        frame_interval = self.config.frame_duration_ms / 1000.0
        consecutive_errors = 0

        while not self._stop_flag.is_set():
            if self._suspended.is_set():
                # .wait() on the REAL stop flag, not a throwaway Event --
                # returns immediately if stop() is called during this wait,
                # instead of always blocking the full frame_interval.
                self._stop_flag.wait(frame_interval)
                continue

            try:
                raw = self._stream.read(
                    self.capture_frame_size,
                    exception_on_overflow=False,
                )
            except OSError as exc:
                consecutive_errors += 1
                logger.warning(
                    "Audio read error (%d/%d before recovery attempt): %s",
                    consecutive_errors, self.MAX_CONSECUTIVE_READ_ERRORS, exc,
                )
                if consecutive_errors < self.MAX_CONSECUTIVE_READ_ERRORS:
                    self._stop_flag.wait(frame_interval)
                    continue

                if self._stop_flag.is_set():
                    return
                if self._attempt_recovery():
                    consecutive_errors = 0
                    continue
                raise MicrophoneRecoveryFailedError(
                    f"Microphone (device_index={self.config.input_device_index}) "
                    f"did not recover after {self.MAX_REOPEN_ATTEMPTS} reopen attempts."
                )

            consecutive_errors = 0

            # Convert int16 PCM -> float32.
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

            # Resample 48 kHz -> 16 kHz using the persistent resampler.
            # Do NOT create a new resampler for every audio frame.
            resampled = self._resampler.process(
                samples,
                self._resample_ratio,
                end_of_input=False,
            )

            # Keep exactly one pipeline frame.
            if len(resampled) < self.frame_size:
                resampled = np.pad(
                    resampled,
                    (0, self.frame_size - len(resampled)),
                )
            elif len(resampled) > self.frame_size:
                resampled = resampled[:self.frame_size]

            # float32 -> int16 PCM.
            output = np.clip(
                resampled,
                -32768,
                32767,
            ).astype(np.int16)

            yield output.tobytes()

    # ------------------------------------------------------------------
    def _attempt_recovery(self) -> bool:
        """Bounded attempts to reopen the mic *stream* on the existing,
        already-initialized PyAudio host instance -- deliberately never
        calls pyaudio.PyAudio() again (see the class docstring comment for
        why: repeated full PortAudio/ALSA re-init while the device was
        absent was confirmed to kill the process with no Python-level
        exception at all). Gated behind a PortAudio-free OS-level presence
        check so PyAudio is never touched while the device is confirmed
        absent.

        Returns True only once a reopened stream proves it can actually
        deliver a real read (open() succeeding is not enough), False if
        every attempt is exhausted.

        Always responsive to stop(): checks _stop_flag before each attempt
        and uses _stop_flag.wait() for backoff and presence polling, so a
        shutdown request returns promptly instead of finishing out any wait.
        """
        if self._pa is None:
            logger.warning("Cannot attempt recovery: no PyAudio host instance (was start() ever called?).")
            return False

        for attempt in range(1, self.MAX_REOPEN_ATTEMPTS + 1):
            if self._stop_flag.is_set():
                return False

            backoff = self.REOPEN_BACKOFF_S[min(attempt - 1, len(self.REOPEN_BACKOFF_S) - 1)]
            logger.warning(
                "Attempting microphone recovery (%d/%d), waiting %.1fs first...",
                attempt, self.MAX_REOPEN_ATTEMPTS, backoff,
            )
            if self._stop_flag.wait(backoff):
                return False  # stop() was called during backoff

            if not self._mic_physically_present():
                logger.warning(
                    "Recovery attempt %d/%d: microphone not yet visible to the OS "
                    "(checked %s) -- skipping PyAudio entirely this attempt.",
                    attempt, self.MAX_REOPEN_ATTEMPTS, self.ASOUND_CARDS_PATH,
                )
                continue

            with self._lock:
                if self._stream is not None:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None

                try:
                    new_stream = self._open_stream_on_existing_pa(self.REOPEN_TIMEOUT_S)
                except Exception as exc:
                    logger.warning(
                        "Recovery attempt %d/%d failed to reopen: %s",
                        attempt, self.MAX_REOPEN_ATTEMPTS, exc,
                    )
                    continue
                if new_stream is None:
                    continue  # timed out -- _open_stream_on_existing_pa already logged why
                self._stream = new_stream

            # Prove the reopened stream actually delivers data -- do not
            # declare recovery successful just because open() didn't raise.
            try:
                self._stream.read(self.capture_frame_size, exception_on_overflow=False)
            except OSError as exc:
                logger.warning(
                    "Recovery attempt %d/%d: reopened but still failing reads: %s",
                    attempt, self.MAX_REOPEN_ATTEMPTS, exc,
                )
                continue

            logger.info("Microphone recovered after %d attempt(s).", attempt)
            return True

        return False

    def _open_stream_on_existing_pa(self, timeout_s: float):
        """Opens a fresh stream on self._pa (the SAME PyAudio host instance
        from the original start() -- never recreated here) in a background
        thread, waiting at most timeout_s. Confirmed on real UNO Q
        hardware: pa.open() against a genuinely-absent device does not
        raise, it blocks -- only a bounded wait catches that, a try/except
        cannot. Returns the new stream on success, None on timeout.

        A thread that times out is abandoned rather than force-killed --
        Python has no safe way to kill a thread blocked in a C call. It's a
        daemon thread, so it can't block process shutdown; if it does
        eventually complete in the background, its stream object is simply
        never referenced or closed. That's a narrow, accepted tradeoff
        against hanging the entire pipeline thread indefinitely. This is
        now reached only after _mic_physically_present() already confirmed
        the device is visible to the OS, so it should be rare in practice.
        """
        result: dict = {}

        def _do_open():
            try:
                result["stream"] = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    frames_per_buffer=self.capture_frame_size,
                    input_device_index=self.config.input_device_index,
                )
            except Exception as exc:
                result["error"] = exc

        t = threading.Thread(target=_do_open, daemon=True)
        t.start()
        t.join(timeout=timeout_s)

        if t.is_alive():
            logger.warning(
                "pa.open() did not return within %.1fs even though the device "
                "appeared present -- abandoning this attempt rather than waiting further.",
                timeout_s,
            )
            return None
        if "error" in result:
            raise result["error"]
        return result["stream"]

    # ------------------------------------------------------------------
    def _read_asound_cards(self) -> str:
        try:
            with open(self.ASOUND_CARDS_PATH) as f:
                return f.read()
        except OSError:
            return ""

    def _capture_usb_audio_signature(self) -> Optional[str]:
        usb_lines = [line for line in self._read_asound_cards().splitlines() if "USB" in line]
        return "\n".join(usb_lines) if usb_lines else None

    def _mic_physically_present(self) -> bool:
        """Cheap, PortAudio-free presence check via /proc/asound/cards --
        lets recovery wait for the device to actually reappear at the
        kernel/ALSA level before ever touching PyAudio again. Not tied to
        a specific ALSA card index (USB reconnection can renumber it),
        only to the USB-Audio card name signature captured at the last
        successful start(). Returns True (never blocks) if no such
        signature was ever captured -- e.g. a non-USB input device -- since
        this check is a safeguard for the USB-disconnect case specifically,
        not a general precondition for every configuration.
        """
        if not self._usb_audio_signature:
            return True
        return self._usb_audio_signature in self._read_asound_cards()


__all__ = [
    "AudioManager",
    "MicrophoneUnavailableError",
    "MicrophoneRecoveryFailedError",
]
