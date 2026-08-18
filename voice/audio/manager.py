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


class AudioManager:
    """Owns one microphone stream and outputs 16 kHz PCM frames.

    The UNO Q USB microphone runs at 48 kHz through PortAudio.
    The voice pipeline expects 16 kHz, so audio is resampled here.
    """

    PIPELINE_SAMPLE_RATE = 16000

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
        """Yield 16 kHz mono int16 PCM frames."""

        if self._stream is None:
            raise RuntimeError(
                "AudioManager.start() must be called before frames()"
            )

        frame_interval = self.config.frame_duration_ms / 1000.0

        while not self._stop_flag.is_set():
            if self._suspended.is_set():
                threading.Event().wait(frame_interval)
                continue

            try:
                raw = self._stream.read(
                    self.capture_frame_size,
                    exception_on_overflow=False,
                )
            except OSError as exc:
                logger.warning(
                    "Audio read error (device likely disconnected): %s",
                    exc,
                )
                threading.Event().wait(frame_interval)
                continue

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


__all__ = [
    "AudioManager",
    "MicrophoneUnavailableError",
]
