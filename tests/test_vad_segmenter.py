"""Tests for voice/audio/vad.py -- the two-tier timeout logic from Section 8
is the highest-value thing to get right here, since it's easy to get subtly
wrong (e.g. waiting out the full 20s after real end-of-speech).
"""

from __future__ import annotations

import types

import pytest

from voice.audio.vad import SegmentEvent, SpeechSegmenter
from voice.config import VADConfig


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


def _make_segmenter(monkeypatch, fake_webrtcvad, clock, **overrides):
    base = dict(
        aggressiveness=2,
        padding_duration_ms=150,   # -> padding_frames = 5 at 30ms frames
        activation_ratio=0.6,      # -> activation_count = 3
        deactivation_ratio=0.8,    # -> deactivation_count = 4
        no_speech_timeout_s=6.0,
        max_session_duration_s=20.0,
    )
    base.update(overrides)
    cfg = VADConfig(**base)
    import voice.audio.vad as vad_mod
    monkeypatch.setattr(vad_mod.time, "monotonic", clock.now)
    seg = SpeechSegmenter(cfg, sample_rate=16000, frame_duration_ms=30)
    return seg


def test_speech_start_and_end_with_hangover(monkeypatch, fake_webrtcvad):
    clock = FakeClock()
    seg = _make_segmenter(monkeypatch, fake_webrtcvad, clock)
    seg.start()

    # Fill the ring buffer with 5 voiced frames -> activation_count=3 met -> SPEECH_STARTED
    fake_webrtcvad.extend([True, True, True, True, True])
    events = [seg.process_frame(b"\x00\x00" * 10) for _ in range(5)]
    assert SegmentEvent.SPEECH_STARTED in [e.event for e in events]

    # Now feed silence. deactivation_count=4 unvoiced frames out of a full
    # ring (5) needed to end. Feed 5 silent frames.
    fake_webrtcvad.extend([False, False, False, False, False])
    events = [seg.process_frame(b"\x00\x00" * 10) for _ in range(5)]
    ended = [e for e in events if e.event == SegmentEvent.SPEECH_ENDED]
    assert len(ended) == 1
    assert ended[0].audio is not None and len(ended[0].audio) > 0


def test_brief_pause_mid_speech_does_not_end_segment(monkeypatch, fake_webrtcvad):
    """Section 9: 'Hey Arduino, what is the... [pause] ...weather tomorrow?'
    must not be cut off by a short pause that doesn't reach the
    deactivation hysteresis threshold.
    """
    clock = FakeClock()
    seg = _make_segmenter(monkeypatch, fake_webrtcvad, clock)
    seg.start()

    fake_webrtcvad.extend([True] * 5)
    for _ in range(5):
        seg.process_frame(b"\x00\x00" * 10)
    assert seg._triggered is True

    # Brief pause: only 2 unvoiced frames (below deactivation_count=4),
    # then speech resumes.
    fake_webrtcvad.extend([False, False, True, True, True])
    events = [seg.process_frame(b"\x00\x00" * 10) for _ in range(5)]
    assert all(e.event != SegmentEvent.SPEECH_ENDED for e in events)
    assert seg._triggered is True  # still mid-speech, not ended


def test_no_speech_timeout_fires_before_speech_starts(monkeypatch, fake_webrtcvad):
    clock = FakeClock()
    seg = _make_segmenter(monkeypatch, fake_webrtcvad, clock, no_speech_timeout_s=5.0)
    seg.start()

    fake_webrtcvad.extend([False])
    result = seg.process_frame(b"\x00\x00" * 10)
    assert result.event == SegmentEvent.NONE  # not timed out yet

    clock.advance(5.1)
    fake_webrtcvad.extend([False])
    result = seg.process_frame(b"\x00\x00" * 10)
    assert result.event == SegmentEvent.NO_SPEECH_TIMEOUT


def test_no_speech_timeout_does_not_fire_once_speech_heard(monkeypatch, fake_webrtcvad):
    """Once real speech has started, the 5-7s no-speech timer must never
    fire again -- only the 20s hard ceiling applies from then on.
    """
    clock = FakeClock()
    seg = _make_segmenter(monkeypatch, fake_webrtcvad, clock, no_speech_timeout_s=2.0,
                           max_session_duration_s=20.0)
    seg.start()

    fake_webrtcvad.extend([True] * 5)
    for _ in range(5):
        seg.process_frame(b"\x00\x00" * 10)
    assert seg._heard_any_speech is True

    # Advance well past no_speech_timeout_s -- must NOT report NO_SPEECH_TIMEOUT
    clock.advance(3.0)
    fake_webrtcvad.extend([False])
    result = seg.process_frame(b"\x00\x00" * 10)
    assert result.event != SegmentEvent.NO_SPEECH_TIMEOUT


def test_max_duration_ceiling_does_not_wait_out_full_20s_after_speech_ends(monkeypatch, fake_webrtcvad):
    """Section 8B: if the user finishes speaking after 3s, STT must run
    immediately -- the 20s value must never be an artificial wait.
    """
    clock = FakeClock()
    seg = _make_segmenter(monkeypatch, fake_webrtcvad, clock, max_session_duration_s=20.0)
    seg.start()

    clock.advance(3.0)  # user starts speaking at t=3s
    fake_webrtcvad.extend([True] * 5)
    for _ in range(5):
        seg.process_frame(b"\x00\x00" * 10)

    clock.advance(0.5)  # short utterance
    fake_webrtcvad.extend([False] * 5)
    events = [seg.process_frame(b"\x00\x00" * 10) for _ in range(5)]
    ended = [e for e in events if e.event == SegmentEvent.SPEECH_ENDED]
    assert len(ended) == 1
    # Total elapsed time when this fired should be nowhere near 20s.
    # (We advanced clock by 3.5s total before the segment ended.)


def test_max_duration_hard_ceiling_fires_mid_speech(monkeypatch, fake_webrtcvad):
    clock = FakeClock()
    seg = _make_segmenter(monkeypatch, fake_webrtcvad, clock, max_session_duration_s=10.0)
    seg.start()

    fake_webrtcvad.extend([True] * 5)
    for _ in range(5):
        seg.process_frame(b"\x00\x00" * 10)
    assert seg._triggered is True

    clock.advance(10.5)
    fake_webrtcvad.extend([True])
    result = seg.process_frame(b"\x00\x00" * 10)
    assert result.event == SegmentEvent.MAX_DURATION_TIMEOUT
    assert result.audio is not None  # partial capture, per Section 8B intent
