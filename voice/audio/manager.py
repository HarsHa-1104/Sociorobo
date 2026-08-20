"""Shared Audio Manager -- the single owner of the microphone."""

from __future__ import annotations

import logging
import threading
from typing import Iterator, Optional

import numpy as np
import samplerate

from voice.audio.combination import ComboGuard
from voice.audio.discovery import DeviceDescriptor, discover_input_devices
from voice.audio.pw_capture import CaptureStartError, PipeWireCapture
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

    Dual-backend since the combination-support extension: a wired
    (USB/ALSA) microphone is captured via PyAudio/PortAudio exactly as
    before; a Bluetooth microphone is captured via PipeWire's pw-record
    (voice/audio/pw_capture.py, Phase 3). Which one is active is decided
    by discovery/selection (voice/audio/discovery.py +
    voice/audio/selection.py), never hardcoded, and is exposed as
    `self._active_backend` ("alsa" | "bluez5"). The two backends have
    genuinely different risk profiles -- see _attempt_alsa_recovery vs
    _attempt_bluetooth_recovery -- and are kept in clearly separate code
    paths throughout this class specifically so the already-hardware-
    validated ALSA/PyAudio path (Milestone 7) is never at risk of an
    editing mistake in the newer Bluetooth path bleeding into it.

    A microphone selected as Bluetooth is only ever chosen if it does not
    conflict with the speaker's own backend -- see
    voice/audio/combination.py, ComboGuard: at most one of {microphone,
    speaker} may be Bluetooth at a time (product requirement, not a
    technical limitation of this class specifically).

    The UNO Q USB microphone runs at 48 kHz through PortAudio. The voice
    pipeline expects 16 kHz, so audio is resampled here -- for BOTH
    backends: a Bluetooth capture requests this SAME config.sample_rate
    from pw-record (never the Bluetooth transport's actual native rate,
    typically 8-16kHz narrowband HFP/HSP -- see pw_capture.py), so
    capture_frame_size and the resampler below are identical regardless
    of which backend is active; PipeWire's own graph handles converting
    from the transport's real rate up to config.sample_rate transparently.
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

    # Bluetooth-capture-specific timeouts (voice/audio/pw_capture.py).
    # Spawning pw-record carries none of the PortAudio reconstruction risk
    # above -- it's a plain subprocess -- so these are just ordinary
    # bounded-wait values, not safety gates.
    BLUETOOTH_OPEN_PROOF_TIMEOUT_S = 3.0  # must actually deliver data before start()/handoff declares success
    BLUETOOTH_READ_TIMEOUT_S = 1.0  # per-frame read bound during normal frames() operation

    def __init__(self, config: AudioConfig, combo_guard: Optional[ComboGuard] = None) -> None:
        if pyaudio is None:
            raise RuntimeError(
                "pyaudio is not installed. Install PortAudio + PyAudio first."
            )

        self.config = config
        # Combination requirement: at most one of {microphone, speaker}
        # may be Bluetooth (voice/audio/combination.py). None (the
        # default) means no cross-role constraint -- used by tests and any
        # standalone use of this class that doesn't need it.
        self._combo_guard = combo_guard

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
        # Captured on a successful ALSA start()/recovery -- the USB-Audio
        # card line(s) from /proc/asound/cards, used as a cheap,
        # PortAudio-free "is the mic physically present" check during
        # recovery. None if the configured device isn't backed by a
        # USB-Audio card (e.g. a non-USB input) -- the presence check then
        # always passes through, so this is a pure optimization/safeguard,
        # never a hard blocker for non-USB setups.
        self._usb_audio_signature: Optional[str] = None

        # Bluetooth backend state (Phase 3 wiring).
        self._pw_capture: Optional[PipeWireCapture] = None
        self._active_backend: Optional[str] = None  # "alsa" | "bluez5"

        # Plug-and-play Phase 2: which physical device is actually open
        # right now, resolved via discovery/selection rather than taken
        # directly from config.input_device_index (see _resolve_microphone).
        # pyaudio_index is only meaningful for the ALSA backend.
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
            if self._stream is not None or self._pw_capture is not None:
                return

            self._pa = pyaudio.PyAudio()

            try:
                chosen = self._resolve_microphone()
            except MicrophoneUnavailableError:
                self._pa.terminate()
                self._pa = None
                raise

            if chosen.backend == "bluez5":
                self._pa.terminate()  # not needed for a Bluetooth capture
                self._pa = None
                self._start_bluetooth_backend(chosen)  # raises MicrophoneUnavailableError on failure
                self._resolved_pyaudio_index = None
            else:
                self._start_alsa_backend(chosen)  # raises + cleans up self._pa itself on failure
                self._resolved_pyaudio_index = chosen.pyaudio_index

            self._resolved_stable_id = chosen.stable_id
            self._active_backend = chosen.backend
            self._stop_flag.clear()
            if self._combo_guard is not None:
                self._combo_guard.set_microphone_backend(chosen.backend)

            logger.info(
                "AudioManager started: mic=%r backend=%s (pyaudio_index=%s, mode=%s) "
                "capture=%dHz -> pipeline=%dHz frame=%dms (%d samples)",
                chosen.stable_id, chosen.backend, self._resolved_pyaudio_index,
                self.config.microphone_mode, self.config.sample_rate,
                self.PIPELINE_SAMPLE_RATE, self.config.frame_duration_ms, self.frame_size,
            )

    def _resolve_microphone(self) -> DeviceDescriptor:
        """Discovers and selects which physical microphone to open,
        reusing self._pa (just constructed by the caller) for the
        ALSA/PyAudio side of discovery (see discover_input_devices'
        docstring for why reuse matters) -- Bluetooth candidates are
        visible via the same discovery call regardless (PipeWire-sourced,
        not PyAudio-sourced). Applies the combination guard's constraint
        (voice/audio/combination.py) as a hard filter, even in "pinned"
        mode -- a pinned device that would conflict with the speaker's
        current backend is treated exactly like a pinned device that
        isn't present: SelectionError, never a silent substitution.
        Raises MicrophoneUnavailableError if nothing survives selection.
        """
        candidates = discover_input_devices(pyaudio_host=self._pa)
        try:
            chosen = self._input_selector.select(
                candidates, mode=self.config.microphone_mode, pin=self.config.microphone_pin,
                is_allowed=self._combined_is_allowed,
            )
        except SelectionError as exc:
            raise MicrophoneUnavailableError(str(exc)) from exc

        return chosen

    def _combined_is_allowed(self, d: DeviceDescriptor) -> bool:
        """The single `is_allowed` predicate used at every selection call
        site in this class -- both conditions below must hold:

        1. The device must actually be openable BY THIS MANAGER. An
           "alsa"-backend candidate with pyaudio_index is None is a
           PipeWire-visible ALSA source discovery could not match to any
           currently-enumerated PyAudio device (e.g. a merge/timing edge
           case) -- passing None as PyAudio's input_device_index does NOT
           raise, it silently opens the SYSTEM DEFAULT input device
           instead, which would be exactly the kind of unintended-device
           selection this whole design exists to prevent. Excluded here,
           not left for pa.open() to paper over. Bluetooth candidates are
           unaffected -- they never have a pyaudio_index at all and are
           opened via PipeWireCapture instead.
        2. The combination guard (voice/audio/combination.py), if one was
           given -- at most one of {microphone, speaker} may be Bluetooth.
        """
        if d.backend == "alsa" and d.pyaudio_index is None:
            return False
        if self._combo_guard is not None:
            return self._combo_guard.microphone_allowed(d.backend)
        return True

    def _start_alsa_backend(self, chosen: DeviceDescriptor) -> None:
        """Opens the wired/ALSA capture stream on self._pa (already
        constructed by the caller). On failure, terminates self._pa and
        raises -- callers must not assume self._pa is still valid after
        an exception from this method."""
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
            self._pa.terminate()
            self._pa = None
            raise MicrophoneUnavailableError(
                f"Could not open microphone (stable_id={chosen.stable_id!r}, "
                f"pyaudio_index={chosen.pyaudio_index}, rate={self.config.sample_rate}): {exc}"
            ) from exc

        self._usb_audio_signature = self._capture_usb_audio_signature()

    def _start_bluetooth_backend(self, chosen: DeviceDescriptor) -> None:
        """Opens Bluetooth microphone capture via pw-record. Requests
        config.sample_rate (not the transport's native rate) -- see class
        docstring. Proves the capture actually delivers data before
        declaring success, same "don't just trust open()" philosophy as
        the ALSA path's recovery logic uses (the plain start() path for
        ALSA doesn't prove a read, but this one does, since -- unlike the
        wired path -- there's real, not-yet-hardware-validated uncertainty
        about whether a given Bluetooth source will actually deliver
        audio once opened; see voice/audio/pw_capture.py's docstring).
        """
        capture = PipeWireCapture(
            chosen.pipewire_node_name, rate=self.config.sample_rate, channels=self.config.channels,
        )
        try:
            capture.start()
        except CaptureStartError as exc:
            raise MicrophoneUnavailableError(
                f"Could not open Bluetooth microphone (stable_id={chosen.stable_id!r}, "
                f"node={chosen.pipewire_node_name}): {exc}"
            ) from exc

        proof = capture.read(self.capture_frame_size * 2, timeout_s=self.BLUETOOTH_OPEN_PROOF_TIMEOUT_S)
        if proof is None:
            capture.stop()
            raise MicrophoneUnavailableError(
                f"Bluetooth microphone {chosen.stable_id!r} opened but produced no "
                f"data within {self.BLUETOOTH_OPEN_PROOF_TIMEOUT_S}s."
            )

        self._pw_capture = capture

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

            if self._pw_capture is not None:
                self._pw_capture.stop()
                self._pw_capture = None

            logger.info("AudioManager stopped")

    def suspend(self) -> None:
        self._suspended.set()

        if self._active_backend == "alsa" and self._stream is not None and self._stream.is_active():
            self._stream.stop_stream()
        # Bluetooth backend: no equivalent pause primitive in
        # PipeWireCapture -- frames() already stops reading while
        # _suspended is set, and pw-record keeps running in the
        # background. PipeWire's own buffering absorbs a bounded pause; an
        # unusually long suspend could mean a small burst of slightly-stale
        # audio on resume, a real but minor tradeoff, not a crash or
        # data-corruption risk. Not validated against real Bluetooth
        # hardware -- flagged in the combination-support report.

    def resume(self) -> None:
        self._suspended.clear()

        if self._active_backend == "alsa" and self._stream is not None and not self._stream.is_active():
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

        if self._stream is None and self._pw_capture is None:
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
                raw = self._read_raw_capture_frame()
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
                    f"backend={self._active_backend}, "
                    f"pyaudio_index={self._resolved_pyaudio_index}) did not recover "
                    f"after {self.MAX_REOPEN_ATTEMPTS} attempts."
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

    def _read_raw_capture_frame(self) -> bytes:
        """Reads capture_frame_size samples of raw S16LE PCM from
        whichever backend is currently active, raising OSError on
        failure/timeout in both cases -- unifying both backends into the
        single exception type frames()'s error-handling loop above
        already expects, so that loop (backoff, MAX_CONSECUTIVE_READ_ERRORS,
        escalation to recovery) needs no backend-specific branching.
        """
        if self._active_backend == "bluez5":
            data = self._pw_capture.read(self.capture_frame_size * 2, timeout_s=self.BLUETOOTH_READ_TIMEOUT_S)
            if data is None:
                raise OSError("PipeWire Bluetooth microphone capture read failed or timed out")
            return data
        return self._stream.read(self.capture_frame_size, exception_on_overflow=False)

    # ------------------------------------------------------------------
    def _attempt_recovery(self) -> bool:
        """Dispatches to the recovery strategy for whichever backend is
        currently active. The two have genuinely different risk profiles
        -- see _attempt_alsa_recovery vs _attempt_bluetooth_recovery --
        and are kept as separate methods rather than one branchy method so
        neither backend's logic risks an editing mistake bleeding into the
        other."""
        if self._active_backend == "bluez5":
            return self._attempt_bluetooth_recovery()
        return self._attempt_alsa_recovery()

    def _attempt_alsa_recovery(self) -> bool:
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
        its own, separate safety gate, and may hand off to the Bluetooth
        backend if that's the best remaining candidate. See that method's
        docstring.

        Returns True once a stream (same device or, via Tier 2, a
        different one -- possibly even a different backend) proves it can
        actually deliver a real read (open() succeeding is not enough),
        False if every attempt across both tiers fails.

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
        """Tier 2 of ALSA recovery -- only reached after Tier 1 is
        exhausted.

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

        Combination-support extension: if the best remaining candidate
        (after applying the combination guard's constraint) is Bluetooth,
        hands off to the Bluetooth capture backend instead of failing --
        see _switch_to_bluetooth_backend.
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

        # Reset sticky state before selecting -- see _attempt_bluetooth_recovery's
        # docstring for why: the device we're recovering FROM could in
        # principle still be a candidate here, and sticky re-confirming a
        # demonstrably-broken pick would defeat the purpose of rediscovery.
        self._input_selector.reset_sticky()
        try:
            chosen = self._input_selector.select(candidates, mode="auto", is_allowed=self._combined_is_allowed)
        except SelectionError as exc:
            logger.warning("Rediscovery: selection failed: %s", exc)
            new_pa.terminate()
            return False

        if chosen.backend == "bluez5":
            logger.info(
                "Rediscovery: best available candidate %r is Bluetooth -- "
                "handing off from the ALSA/PyAudio capture path to the "
                "PipeWire capture path.", chosen.stable_id,
            )
            new_pa.terminate()  # not needed for a Bluetooth capture
            return self._switch_to_bluetooth_backend(chosen)

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
        self._active_backend = "alsa"
        self._usb_audio_signature = self._capture_usb_audio_signature()
        if self._combo_guard is not None:
            self._combo_guard.set_microphone_backend("alsa")
        logger.info(
            "Microphone recovered via rediscovery: now using %r (pyaudio_index=%d).",
            chosen.stable_id, chosen.pyaudio_index,
        )
        return True

    def _attempt_bluetooth_recovery(self) -> bool:
        """Recovery for the Bluetooth/PipeWire capture backend.

        Unlike _attempt_alsa_recovery/_attempt_device_rediscovery, this is
        a single unified loop, not two tiers: spawning a fresh pw-record
        subprocess carries none of PortAudio's reconstruct-while-absent
        crash risk (Milestone 7) -- pw-record is a plain CLI subprocess,
        not a PortAudio host -- so every attempt is free to fully
        re-resolve (same Bluetooth device, a different one, or hand off to
        a wired device if one is now available and the combination guard
        allows it) and reopen, with no special-cased "same device only"
        first phase needed.

        Resets the selector's sticky state before every attempt's
        selection. The device we're actively recovering FROM is often
        still discoverable (its Bluetooth connection can still be up even
        though its audio stream has gone bad -- unlike a device that's
        physically disappeared entirely), so without this, "sticky
        previous selection" (voice/audio/selection.py) would keep
        re-confirming the very device that's currently broken, before
        class-priority ever gets a chance to prefer a newly-available
        wired microphone. Sticky's job is to prevent gratuitous switching
        during NORMAL operation; re-affirming a demonstrably-broken pick
        during active recovery would defeat the purpose of recovering.
        """
        for attempt in range(1, self.MAX_REOPEN_ATTEMPTS + 1):
            if self._stop_flag.is_set():
                return False

            backoff = self.REOPEN_BACKOFF_S[min(attempt - 1, len(self.REOPEN_BACKOFF_S) - 1)]
            logger.warning(
                "Attempting Bluetooth microphone recovery (%d/%d), waiting %.1fs first...",
                attempt, self.MAX_REOPEN_ATTEMPTS, backoff,
            )
            if self._stop_flag.wait(backoff):
                return False

            # Only construct a PyAudio host (needed to see ALSA candidates
            # alongside Bluetooth ones for correct ranking) when the OS
            # itself shows some USB audio hardware present -- the same
            # Milestone 7-motivated gate _attempt_device_rediscovery uses,
            # applied here too since this loop also constructs a fresh
            # PyAudio() on every attempt. When nothing is present, discover
            # Bluetooth-only via discover_input_devices(include_alsa=False),
            # which never touches PyAudio at all.
            pa = None
            try:
                if self._any_usb_audio_card_present():
                    pa = pyaudio.PyAudio()
                    candidates = discover_input_devices(pyaudio_host=pa)
                else:
                    candidates = discover_input_devices(include_alsa=False)
            except Exception:
                logger.exception(
                    "Bluetooth recovery attempt %d/%d: device enumeration raised unexpectedly.",
                    attempt, self.MAX_REOPEN_ATTEMPTS,
                )
                if pa is not None:
                    pa.terminate()
                continue

            self._input_selector.reset_sticky()
            try:
                chosen = self._input_selector.select(
                    candidates, mode=self.config.microphone_mode, pin=self.config.microphone_pin,
                    is_allowed=self._combined_is_allowed,
                )
            except SelectionError as exc:
                logger.warning(
                    "Bluetooth recovery attempt %d/%d: selection failed: %s",
                    attempt, self.MAX_REOPEN_ATTEMPTS, exc,
                )
                if pa is not None:
                    pa.terminate()
                continue

            if chosen.backend == "alsa":
                # Only reachable when `pa` was actually constructed above
                # (an ALSA candidate needs a real pyaudio_index, which
                # requires it) -- never None here.
                logger.info(
                    "Bluetooth recovery attempt %d/%d: a wired microphone (%r) "
                    "is now available -- handing off to the ALSA capture path.",
                    attempt, self.MAX_REOPEN_ATTEMPTS, chosen.stable_id,
                )
                if self._switch_to_alsa_backend(chosen, pa):
                    return True
                continue  # handoff failure already cleaned up `pa` internally

            if pa is not None:
                pa.terminate()  # not needed for a Bluetooth capture
            if self._switch_to_bluetooth_backend(chosen):
                return True

        return False

    def _switch_to_bluetooth_backend(self, chosen: DeviceDescriptor) -> bool:
        """Shared cross-backend handoff: stop whatever capture is
        currently active (ALSA stream + its PyAudio host, or a previous
        Bluetooth capture), start Bluetooth capture for `chosen`, and
        adopt it on success. Returns False (never raises) on any failure,
        matching the bool contract both recovery loops expect.
        """
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None
            if self._pw_capture is not None:
                self._pw_capture.stop()
                self._pw_capture = None

            try:
                self._start_bluetooth_backend(chosen)
            except MicrophoneUnavailableError as exc:
                logger.warning("Cross-backend handoff to Bluetooth failed: %s", exc)
                return False

            self._resolved_pyaudio_index = None
            self._resolved_stable_id = chosen.stable_id
            self._active_backend = "bluez5"

        if self._combo_guard is not None:
            self._combo_guard.set_microphone_backend("bluez5")
        logger.info("Microphone switched to Bluetooth via cross-backend handoff: %r.", chosen.stable_id)
        return True

    def _switch_to_alsa_backend(self, chosen: DeviceDescriptor, pa: "pyaudio.PyAudio") -> bool:
        """Shared cross-backend handoff, the reverse direction: adopt an
        already-discovered wired/ALSA candidate, using the
        ALREADY-CONSTRUCTED PyAudio host `pa` the caller used to discover
        it (never constructs a second one here). Returns False (never
        raises) on any failure; `pa` is guaranteed terminated on failure
        (either by this method or by _start_alsa_backend internally).
        """
        with self._lock:
            if self._pw_capture is not None:
                self._pw_capture.stop()
                self._pw_capture = None
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            if self._pa is not None and self._pa is not pa:
                self._pa.terminate()
            self._pa = pa

            try:
                self._start_alsa_backend(chosen)
            except MicrophoneUnavailableError as exc:
                logger.warning("Cross-backend handoff to ALSA failed: %s", exc)
                return False  # _start_alsa_backend already terminated self._pa and set it to None

        # Prove it actually delivers data -- same "don't just trust open()"
        # philosophy as every other recovery path in this class.
        try:
            self._stream.read(self.capture_frame_size, exception_on_overflow=False)
        except OSError as exc:
            logger.warning("Cross-backend handoff to ALSA: opened but failed reads: %s", exc)
            with self._lock:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
                if self._pa is not None:
                    self._pa.terminate()
                    self._pa = None
            return False

        self._resolved_pyaudio_index = chosen.pyaudio_index
        self._resolved_stable_id = chosen.stable_id
        self._active_backend = "alsa"
        self._usb_audio_signature = self._capture_usb_audio_signature()
        if self._combo_guard is not None:
            self._combo_guard.set_microphone_backend("alsa")
        logger.info("Microphone switched to ALSA/wired via cross-backend handoff: %r.", chosen.stable_id)
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
