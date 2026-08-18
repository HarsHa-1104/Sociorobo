"""Voice-activity / end-of-speech segmentation.

Phase 1 finding this preserves: the old project's WebRTC-VAD ring-buffer
design (activation/deactivation hysteresis over a padding window) was found
to be sound and was explicitly recommended to be kept, not replaced. The
core algorithm below is a direct, deliberate port of that logic.

What changed from the old ``VADListener``:

  * It no longer owns a PyAudio stream itself -- ``AudioManager`` does that
    now (Section 10: one microphone, one owner). This class is a pure frame
    -> event state machine, which makes it trivially unit-testable with
    synthetic frames and no audio hardware at all.
  * It no longer starts/stops the mic on enable/disable -- that's an
    ``AudioManager.suspend()/resume()`` concern now.
  * The two-tier timeout behaviour from Section 8 of the Phase 2 spec
    (5-7s no-speech-after-wake, 20s hard session ceiling, early exit on
    end-of-speech) is implemented here via wall-clock checks driven by the
    caller feeding frames in real time -- see ``SpeechSegmenter.tick()``.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, Optional, Tuple

from voice.config import VADConfig

try:
    import webrtcvad
except ImportError:  # pragma: no cover
    webrtcvad = None  # type: ignore


Frame = Tuple[bytes, bool]


class SegmentEvent(Enum):
    NONE = auto()
    SPEECH_STARTED = auto()
    SPEECH_ENDED = auto()          # normal end-of-speech, carries audio
    NO_SPEECH_TIMEOUT = auto()     # Section 8A: nothing heard within no_speech_timeout_s
    MAX_DURATION_TIMEOUT = auto()  # Section 8B: hard 20s ceiling reached, possibly mid-speech


@dataclass
class SegmentResult:
    event: SegmentEvent
    audio: Optional[bytes] = None


class SpeechSegmenter:
    """Feed it frames in real time; it tells you when a spoken turn is done.

    Two clocks matter here, matching Section 8 of the spec exactly:

    * ``no_speech_timeout_s`` -- measured from the moment :meth:`start`
      is called (i.e. from wake-word detection) until the *first* frame of
      real speech is confirmed. If that fires before any speech starts,
      the caller gets ``NO_SPEECH_TIMEOUT`` and should end the session
      without ever calling STT.
    * ``max_session_duration_s`` -- measured from the same start point, as
      an absolute ceiling regardless of what's happening. If speech is
      still ongoing when this fires, the caller gets
      ``MAX_DURATION_TIMEOUT`` with whatever audio was captured so far
      (best effort) rather than waiting longer.

    Neither timer waits around once real end-of-speech is detected --
    ``SPEECH_ENDED`` fires as soon as the deactivation hysteresis trips,
    which is the whole point of Section 8B ("don't waste the remaining
    17 seconds").
    """

    def __init__(self, config: VADConfig, sample_rate: int, frame_duration_ms: int) -> None:
        if webrtcvad is None:
            raise RuntimeError("webrtcvad is not installed. pip install webrtcvad.")

        self.config = config
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms

        self.padding_frames = max(1, int(config.padding_duration_ms / frame_duration_ms))
        self.activation_count = max(1, int(self.padding_frames * config.activation_ratio))
        self.deactivation_count = max(1, int(self.padding_frames * config.deactivation_ratio))

        self._vad = webrtcvad.Vad(config.aggressiveness)

        self._ring: Deque[Frame] = collections.deque(maxlen=self.padding_frames)
        self._voiced: Deque[bytes] = collections.deque()
        self._triggered = False
        self._heard_any_speech = False
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Begin a new listening session. Call once per wake-word event."""
        self._ring.clear()
        self._voiced.clear()
        self._triggered = False
        self._heard_any_speech = False
        self._start_time = time.monotonic()

    def reset(self) -> None:
        self.start()

    # ------------------------------------------------------------------
    def process_frame(self, frame: bytes) -> SegmentResult:
        """Feed one frame; returns what happened, if anything, this frame."""
        if self._start_time is None:
            raise RuntimeError("SpeechSegmenter.start() must be called before process_frame()")

        elapsed = time.monotonic() - self._start_time

        if elapsed >= self.config.max_session_duration_s:
            audio = b"".join(self._voiced) if self._voiced else None
            return SegmentResult(SegmentEvent.MAX_DURATION_TIMEOUT, audio)

        if not self._heard_any_speech and elapsed >= self.config.no_speech_timeout_s:
            return SegmentResult(SegmentEvent.NO_SPEECH_TIMEOUT)

        is_speech = self._vad.is_speech(frame, self.sample_rate)
        self._ring.append((frame, is_speech))

        if not self._triggered:
            if len(self._ring) == self._ring.maxlen:
                num_voiced = sum(1 for _, speech in self._ring if speech)
                if num_voiced >= self.activation_count:
                    self._triggered = True
                    self._heard_any_speech = True
                    while self._ring:
                        self._voiced.append(self._ring.popleft()[0])
                    return SegmentResult(SegmentEvent.SPEECH_STARTED)
            return SegmentResult(SegmentEvent.NONE)

        # Currently triggered (mid-speech).
        self._voiced.append(frame)
        if len(self._ring) == self._ring.maxlen:
            num_unvoiced = sum(1 for _, speech in self._ring if not speech)
            if num_unvoiced >= self.deactivation_count:
                self._triggered = False
                audio = b"".join(self._voiced)
                self._voiced.clear()
                self._ring.clear()
                return SegmentResult(SegmentEvent.SPEECH_ENDED, audio)

        return SegmentResult(SegmentEvent.NONE)


__all__ = ["SpeechSegmenter", "SegmentEvent", "SegmentResult"]
