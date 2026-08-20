"""Shared Audio Manager -- the single owner of the microphone."""

from __future__ import annotations

import logging
import threading
from typing import Iterator, Optional

import numpy as np
import samplerate

from voice.audio.discovery import discover_input_devices
from voice.audio.selection import SelectionError, make_input_selector
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

        # Plug-and-play Phase 2: which physical device is actually open
        # right now, resolved via discovery/selection rather than taken
        # directly from config.input_device_index (see _resolve_microphone).
        # Both are set by a successful start() or a successful recovery
        # (either tier).
        self._resolved_pyaudio_index: Optional[int] = None
        self._resolved_stable_id: Optional[str] = None
        self._input_selector = make_input_selector()

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
                resolved_index, resolved_stable_id = self._resolve_microphone()
            except MicrophoneUnavailableError:
                self._pa.terminate()
                self._pa = None
                raise

            try:
                self._stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    frames_per_buffer=self.capture_frame_size,
                    input_device_index=resolved_index,
                )
            except Exception as exc:
                self._pa.terminate()
                self._pa = None
                raise MicrophoneUnavailableError(
                    f"Could not open microphone "
                    f"(stable_id={resolved_stable_id!r}, pyaudio_index={resolved_index}, "
                    f"rate={self.config.sample_rate}): {exc}"
                ) from exc

            self._resolved_pyaudio_index = resolved_index
            self._resolved_stable_id = resolved_stable_id
            self._stop_flag.clear()
            self._usb_audio_signature = self._capture_usb_audio_signature()

            logger.info(
                "AudioManager started: mic=%r (pyaudio_index=%s, mode=%s) "
                "capture=%dHz -> pipeline=%dHz frame=%dms (%d samples)",
                resolved_stable_id, resolved_index, self.config.microphone_mode,
                self.config.sample_rate,
                self.PIPELINE_SAMPLE_RATE,
                self.config.frame_duration_ms,
                self.frame_size,
            )

    def _resolve_microphone(self) -> tuple[int, str]:
        """Discovers and selects which physical microphone to open,
        reusing self._pa (just constructed by the caller, so it reflects
        the current device list -- see discover_input_devices' docstring
        for why reuse matters) rather than a second, independent PyAudio
        instance. Raises MicrophoneUnavailableError if no PyAudio-openable
        candidate can be selected: discovery found nothing, the configured
        pin doesn't match anything present (microphone_mode == "pinned"),
        or the only candidate(s) found are Bluetooth sources (backend ==
        "bluez5") -- those aren't openable through this PyAudio-based
        capture path at all (see voice/audio/pw_capture.py, Phase 3).
        """
        candidates = discover_input_devices(pyaudio_host=self._pa)
        try:
            chosen = self._input_selector.select(
                candidates, mode=self.config.microphone_mode, pin=self.config.microphone_pin,
            )
        except SelectionError as exc:
            raise MicrophoneUnavailableError(str(exc)) from exc

        if chosen.pyaudio_index is None:
            raise MicrophoneUnavailableError(
                f"Selected microphone {chosen.stable_id!r} (backend={chosen.backend!r}) "
                f"has no PyAudio-openable stream -- Bluetooth microphone capture is "
                f"not supported by this capture path."
            )

        return chosen.pyaudio_index, chosen.stable_id

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
                    f"Microphone (stable_id={self._resolved_stable_id!r}, "
                    f"pyaudio_index={self._resolved_pyaudio_index}) did not recover "
                    f"after {self.MAX_REOPEN_ATTEMPTS} same-device attempts"
                    + (" plus one rediscovery attempt" if self.config.microphone_mode == "auto" else "")
                    + "."
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
        """Tier 1: bounded attempts to reopen the mic *stream* on the
        existing, already-initialized PyAudio host instance -- deliberately
        never calls pyaudio.PyAudio() again in THIS loop (see the class
        docstring comment for why: repeated full PortAudio/ALSA re-init
        while the device was absent was confirmed to kill the process with
        no Python-level exception at all). Gated behind a PortAudio-free
        presence check so PyAudio is never touched while the ORIGINAL
        device is confirmed absent.

        If Tier 1 is exhausted and microphone_mode == "auto", this method
        falls through to _attempt_device_rediscovery() (Tier 2), which -- by
        necessity, not oversight -- DOES reconstruct pyaudio.PyAudio() under
        its own, separate safety gate. See that method's docstring.

        Returns True once a stream (same device or, via Tier 2, a different
        one) proves it can actually deliver a real read (open() succeeding
        is not enough), False if every attempt across both tiers fails.

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

        # Tier 1 (retrying the SAME device on the SAME PyAudio host) is
        # exhausted. In "auto" mode only, make one further attempt: maybe a
        # DIFFERENT compatible microphone is now available. Never attempted
        # in "pinned" mode -- pinned means deliberate control, and silently
        # substituting a different physical device would defeat that (same
        # philosophy as voice/audio/selection.py's pinned-mode behavior).
        if self.config.microphone_mode == "auto":
            return self._attempt_device_rediscovery()

        return False

    def _attempt_device_rediscovery(self) -> bool:
        """Tier 2 of recovery -- only reached after Tier 1 is exhausted.

        Unlike Tier 1 and every other PyAudio interaction in this class,
        this DOES reconstruct pyaudio.PyAudio(). There is no way around
        that: PortAudio's ALSA backend snapshots its device list once at
        construction time and does not refresh it live, so reusing the
        existing self._pa for discovery here would only ever see the
        ORIGINAL device list -- never a genuinely different or newly
        arrived microphone -- which would make this tier silently useless
        (it would never actually find anything new, while looking like it
        was implemented). Tier 1 doesn't have this problem because it only
        reopens a stream at an ALREADY-KNOWN index, which doesn't require
        re-enumeration at all.

        This is therefore the single riskiest operation in this class --
        Milestone 7 found that reconstructing pyaudio.PyAudio() while a USB
        device was physically absent crashed the whole process with no
        Python-level exception. Mitigated, not eliminated: gated behind
        _any_usb_audio_card_present(), a PortAudio-free OS-level check, so
        reconstruction is never attempted while the OS itself shows no USB
        audio hardware at all (the specific condition Milestone 7 found
        fatal). This does not prove reconstruction is always safe, only
        that the one confirmed-fatal condition doesn't apply.
        """
        if self._stop_flag.is_set():
            return False

        if not self._any_usb_audio_card_present():
            logger.warning(
                "Rediscovery: no USB audio hardware visible to the OS at all "
                "(checked %s) -- not attempting to reconstruct the PyAudio "
                "host (Milestone 7: doing so while nothing is present "
                "crashed the process).",
                self.ASOUND_CARDS_PATH,
            )
            return False

        logger.warning(
            "Tier 1 recovery exhausted -- attempting rediscovery of a "
            "different microphone (reconstructs the PyAudio host; see "
            "_attempt_device_rediscovery docstring)."
        )

        try:
            new_pa = pyaudio.PyAudio()
        except Exception as exc:
            logger.error("Rediscovery: failed to construct a new PyAudio host: %s", exc)
            return False

        try:
            candidates = discover_input_devices(pyaudio_host=new_pa)
        except Exception:
            logger.exception("Rediscovery: device enumeration raised unexpectedly.")
            new_pa.terminate()
            return False

        if not candidates:
            logger.warning("Rediscovery: no usable microphones found.")
            new_pa.terminate()
            return False

        try:
            chosen = self._input_selector.select(candidates, mode="auto")
        except SelectionError as exc:
            logger.warning("Rediscovery: selection failed: %s", exc)
            new_pa.terminate()
            return False

        if chosen.pyaudio_index is None:
            logger.warning(
                "Rediscovery selected %r but it has no PyAudio-openable "
                "stream (backend=%s) -- Bluetooth microphone capture is not "
                "supported by this capture path.",
                chosen.stable_id, chosen.backend,
            )
            new_pa.terminate()
            return False

        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

            # Keep the OLD host around until the new stream is proven to
            # actually work -- a failed swap must not leave us with neither
            # a working old host nor a working new one.
            old_pa = self._pa
            self._pa = new_pa

            try:
                self._stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    frames_per_buffer=self.capture_frame_size,
                    input_device_index=chosen.pyaudio_index,
                )
            except Exception as exc:
                logger.warning("Rediscovery: failed to open the new device: %s", exc)
                self._pa = old_pa
                new_pa.terminate()
                return False

        try:
            self._stream.read(self.capture_frame_size, exception_on_overflow=False)
        except OSError as exc:
            logger.warning("Rediscovery: new device opened but failed reads: %s", exc)
            with self._lock:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
                self._pa = old_pa
            new_pa.terminate()
            return False

        old_pa.terminate()
        self._resolved_pyaudio_index = chosen.pyaudio_index
        self._resolved_stable_id = chosen.stable_id
        self._usb_audio_signature = self._capture_usb_audio_signature()
        logger.info(
            "Microphone recovered via rediscovery: now using %r (pyaudio_index=%d).",
            chosen.stable_id, chosen.pyaudio_index,
        )
        return True

    def _open_stream_on_existing_pa(self, timeout_s: float):
        """Opens a fresh stream on self._pa (the SAME PyAudio host instance
        from the original start() -- never recreated here) at the currently
        resolved device index, in a background thread, waiting at most
        timeout_s. Confirmed on real UNO Q hardware: pa.open() against a
        genuinely-absent device does not raise, it blocks -- only a bounded
        wait catches that, a plain try/except cannot. Returns the new
        stream on success, None on timeout.

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
                    input_device_index=self._resolved_pyaudio_index,
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

    def _any_usb_audio_card_present(self) -> bool:
        """PortAudio-free, OS-level check: is ANY USB-Audio card currently
        visible to ALSA -- regardless of whether it matches the originally
        configured device's signature? Unlike _mic_physically_present()
        (which checks for the SAME device coming back), this is used to
        gate Tier 2 recovery (_attempt_device_rediscovery), which is
        specifically looking for a DIFFERENT device. Confirms only that
        the OS-level precondition Milestone 7 found fatal (reconstructing
        pyaudio.PyAudio() while NO USB device is present) does not apply --
        not a general guarantee that reconstruction is safe.
        """
        return "USB" in self._read_asound_cards()


__all__ = [
    "AudioManager",
    "MicrophoneUnavailableError",
    "MicrophoneRecoveryFailedError",
]
