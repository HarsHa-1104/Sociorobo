"""End-to-end orchestration tests for VoiceManager, using fakes for every
hardware-facing collaborator (audio, wake word, VAD, STT, LLM, TTS) plus a
real HumanFollowerLink talking to a real ReferenceHumanFollowerServer over
an actual Unix socket. This is the closest thing in the repo to "does the
whole wake -> pause -> listen -> STT -> LLM -> TTS -> resume cycle actually
work", without needing real audio hardware or model binaries.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import pytest

from voice.audio.vad import SegmentEvent, SegmentResult
from voice.config import VoiceSystemConfig
from voice.ipc.server_stub import ReferenceHumanFollowerServer
from voice.manager.state_machine import VoiceState
from voice.manager.voice_manager import VoiceManager


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeAudioManager:
    """Yields pre-scripted frames; tracks suspend/resume calls for assertions."""

    def __init__(self, frame_batches: List[List[bytes]]):
        # Each call to frames() pops the next batch off this list. This
        # mirrors the real AudioManager's behaviour where each phase
        # (wake-wait, then listen) consumes a fresh generator.
        self._batches = frame_batches
        self.suspend_calls = 0
        self.resume_calls = 0
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def suspend(self):
        self.suspend_calls += 1

    def resume(self):
        self.resume_calls += 1

    def frames(self):
        batch = self._batches.pop(0) if self._batches else []
        for f in batch:
            yield f


class FakeWakeWordDetector:
    def __init__(self, fire_on_frame: bytes = b"WAKE"):
        self.fire_on_frame = fire_on_frame
        self.reset_calls = 0

    def process_frame(self, frame: bytes) -> bool:
        return frame == self.fire_on_frame

    def reset(self):
        self.reset_calls += 1


class FakeSegmenter:
    """Replays a pre-scripted sequence of SegmentResults, one per process_frame call."""

    def __init__(self, script: List[SegmentResult]):
        self.script = list(script)
        self.start_calls = 0

    def start(self):
        self.start_calls += 1

    def process_frame(self, frame: bytes) -> SegmentResult:
        if self.script:
            return self.script.pop(0)
        return SegmentResult(SegmentEvent.NONE)


class FakeSTT:
    def __init__(self, transcript: str = "what is the weather"):
        self.transcript = transcript
        self.calls = 0

    def run_stt(self, audio: bytes, sample_rate: int) -> str:
        self.calls += 1
        return self.transcript


class FakeLLM:
    def __init__(self, reply: str = "It is sunny today."):
        self.reply = reply
        self.queries: List[str] = []

    def query(self, text: str) -> str:
        self.queries.append(text)
        return self.reply


class FakeTTS:
    def __init__(self, ok: bool = True):
        self.ok = ok
        self.played: List[str] = []

    def synthesize_and_play(self, text: str, alsa_device: str) -> bool:
        self.played.append(text)
        return self.ok


# ---------------------------------------------------------------------------
@pytest.fixture
def config(tmp_path):
    cfg = VoiceSystemConfig()
    cfg.ipc.socket_path = str(tmp_path / "vm_test.sock")
    cfg.ipc.pause_confirm_timeout_s = 2.0
    cfg.ipc.heartbeat_interval_s = 0.1
    cfg.vad.post_tts_cooldown_s = 0.01  # keep tests fast
    return cfg


@pytest.fixture
def running_reference_server(config):
    events = {"pauses": 0, "resumes": 0, "outcomes": []}

    def on_pause():
        events["pauses"] += 1
        return True

    def on_resume():
        events["resumes"] += 1

    server = ReferenceHumanFollowerServer(
        config.ipc, config.watchdog, on_pause=on_pause, on_resume=on_resume,
    )
    server.start()
    yield server, events
    server.stop()


def _build_manager(config, segmenter_script, stt="what is the weather",
                    llm_reply="It is sunny today.", tts_ok=True):
    # Batch 1 feeds _wait_for_wake (one WAKE frame is enough to fire it).
    # Batch 2 feeds _listen_for_utterance -- it must supply at least as
    # many frames as segmenter_script has entries, since FakeSegmenter
    # pops one scripted result per process_frame() call and the listening
    # loop only breaks once it receives a terminal event (SPEECH_ENDED /
    # NO_SPEECH_TIMEOUT / MAX_DURATION_TIMEOUT), which by construction is
    # always the last entry in the script.
    wake_frame_batches = [
        [b"WAKE"],
        [f"F{i}".encode() for i in range(len(segmenter_script))],
    ]
    audio = FakeAudioManager(wake_frame_batches)
    wake = FakeWakeWordDetector()
    segmenter = FakeSegmenter(segmenter_script)
    stt_fake = FakeSTT(stt)
    llm_fake = FakeLLM(llm_reply)
    tts_fake = FakeTTS(tts_ok)
    vm = VoiceManager(config, audio, wake, segmenter, stt_fake, llm_fake, tts_fake)
    return vm, audio, wake, segmenter, stt_fake, llm_fake, tts_fake


def _run_n_cycles_in_thread(vm: VoiceManager, n: int, timeout: float = 5.0):
    """Like _run_one_cycle_in_thread, but waits for `n` full sessions and
    reports whether run_forever's thread crashed (as opposed to just not
    finishing in time).

    Regression coverage for a real crash found on UNO Q hardware during
    Milestone 6 end-to-end testing: with wake.enabled=True (the real
    code path, not the debug bypass), nothing transitioned
    SESSION_COMPLETE -> WAKE_LISTENING before the second wake fire, so
    _run_session()'s first move (-> PAUSE_PENDING) crashed with
    IllegalTransitionError on the second cycle. Every other test in this
    file only runs one cycle, which structurally could never have caught
    this -- see voice/manager/voice_manager.py's _wait_for_wake() fix.

    Returns (thread_still_alive, thread_crashed).
    """
    crashed = threading.Event()

    def _run():
        try:
            vm.run_forever()
        except Exception:
            crashed.set()
            raise

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if crashed.is_set():
            break
        if vm._session_count >= n and vm.state == VoiceState.WAKE_LISTENING:
            break
        if not t.is_alive():
            break
        time.sleep(0.02)
    vm.stop()
    t.join(timeout=2.0)
    return t.is_alive(), crashed.is_set()


def _run_one_cycle_in_thread(vm: VoiceManager, timeout: float = 3.0):
    """Runs run_forever() in a background thread and stops it shortly after
    the manager returns to WAKE_LISTENING following one full session (or
    after a timeout, to avoid hanging tests on a bug).
    """
    t = threading.Thread(target=vm.run_forever, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    # Wait for at least one full session to complete: state cycles back to
    # WAKE_LISTENING with session_count >= 1.
    while time.monotonic() < deadline:
        if vm._session_count >= 1 and vm.state == VoiceState.WAKE_LISTENING:
            break
        time.sleep(0.02)
    vm.stop()
    t.join(timeout=2.0)
    return t.is_alive()


# ---------------------------------------------------------------------------
def test_full_happy_path_answers_and_resumes(config, running_reference_server):
    server, events = running_reference_server

    segmenter_script = [
        SegmentResult(SegmentEvent.SPEECH_STARTED),
        SegmentResult(SegmentEvent.SPEECH_ENDED, audio=b"\x00\x00" * 1000),
    ]
    vm, audio, wake, seg, stt, llm, tts = _build_manager(config, segmenter_script)

    still_alive = _run_one_cycle_in_thread(vm)
    assert not still_alive, "run_forever thread did not stop cleanly"

    assert stt.calls == 1
    assert llm.queries == ["what is the weather"]
    assert tts.played == ["It is sunny today."]
    assert events["pauses"] == 1
    assert events["resumes"] == 1
    assert wake.reset_calls == 1
    # Wake detector must never be fed frames during the session itself --
    # enforced structurally, but we can at least assert suspend was used
    # around the point where TTS/processing happens.
    assert audio.suspend_calls >= 1
    assert audio.resume_calls >= 1


def test_no_speech_after_wake_ends_session_without_stt(config, running_reference_server):
    server, events = running_reference_server
    segmenter_script = [SegmentResult(SegmentEvent.NO_SPEECH_TIMEOUT)]
    vm, audio, wake, seg, stt, llm, tts = _build_manager(config, segmenter_script)

    still_alive = _run_one_cycle_in_thread(vm)
    assert not still_alive

    assert stt.calls == 0
    assert llm.queries == []
    assert tts.played == []
    assert events["resumes"] == 1  # must still resume HumanFollower


def test_stt_failure_ends_session_and_resumes(config, running_reference_server):
    server, events = running_reference_server
    segmenter_script = [SegmentResult(SegmentEvent.SPEECH_ENDED, audio=b"\x00\x00" * 100)]
    vm, audio, wake, seg, stt, llm, tts = _build_manager(
        config, segmenter_script, stt="",  # empty transcript = STT "failure"
    )

    still_alive = _run_one_cycle_in_thread(vm)
    assert not still_alive
    assert llm.queries == []  # never reached
    assert tts.played == []
    assert events["resumes"] == 1


def test_llm_failure_ends_session_and_resumes(config, running_reference_server):
    server, events = running_reference_server
    segmenter_script = [SegmentResult(SegmentEvent.SPEECH_ENDED, audio=b"\x00\x00" * 100)]
    vm, audio, wake, seg, stt, llm, tts = _build_manager(
        config, segmenter_script, llm_reply="",
    )

    still_alive = _run_one_cycle_in_thread(vm)
    assert not still_alive
    assert stt.calls == 1
    assert tts.played == []
    assert events["resumes"] == 1


def test_tts_failure_still_resumes(config, running_reference_server):
    server, events = running_reference_server
    segmenter_script = [SegmentResult(SegmentEvent.SPEECH_ENDED, audio=b"\x00\x00" * 100)]
    vm, audio, wake, seg, stt, llm, tts = _build_manager(
        config, segmenter_script, tts_ok=False,
    )

    still_alive = _run_one_cycle_in_thread(vm)
    assert not still_alive
    assert tts.played == ["It is sunny today."]  # attempted
    assert events["resumes"] == 1  # resumes even though TTS "failed" to play cleanly


def test_humanfollower_unreachable_still_completes_session(config):
    """No IPC server running at all -- Voice Manager must still run the
    full session rather than hanging forever waiting on PAUSE_CONFIRMED.
    """
    segmenter_script = [SegmentResult(SegmentEvent.SPEECH_ENDED, audio=b"\x00\x00" * 100)]
    vm, audio, wake, seg, stt, llm, tts = _build_manager(config, segmenter_script)

    still_alive = _run_one_cycle_in_thread(vm, timeout=5.0)
    assert not still_alive
    assert tts.played == ["It is sunny today."]


def test_two_consecutive_sessions_with_wake_enabled_do_not_crash(config, running_reference_server):
    """Regression test for a real crash found on UNO Q hardware: the
    second wake-word fire (real detection path, not the wake-disabled
    bypass) used to attempt SESSION_COMPLETE -> PAUSE_PENDING directly,
    which the state graph rejects and which crashed run_forever's whole
    thread with an unhandled IllegalTransitionError. See
    voice/manager/voice_manager.py's _wait_for_wake() fix and its
    docstring for the full story.
    """
    server, events = running_reference_server
    assert config.wake.enabled is True  # this must exercise the real detection path

    segmenter_script = [SegmentResult(SegmentEvent.SPEECH_ENDED, audio=b"\x00\x00" * 100)]
    # Two full cycles: [wake1, listen1, wake2, listen2].
    wake_frame_batches = [
        [b"WAKE"],
        [f"F{i}".encode() for i in range(len(segmenter_script))],
        [b"WAKE"],
        [f"G{i}".encode() for i in range(len(segmenter_script))],
    ]
    audio = FakeAudioManager(wake_frame_batches)
    wake = FakeWakeWordDetector()
    segmenter = FakeSegmenter(segmenter_script * 2)
    stt = FakeSTT("what is the weather")
    llm = FakeLLM("It is sunny today.")
    tts = FakeTTS(True)
    vm = VoiceManager(config, audio, wake, segmenter, stt, llm, tts)

    still_alive, crashed = _run_n_cycles_in_thread(vm, n=2)
    assert not crashed, "run_forever crashed instead of completing 2 sessions"
    assert not still_alive
    assert vm._session_count == 2
    assert vm.state == VoiceState.WAKE_LISTENING
    assert stt.calls == 2
    assert tts.played == ["It is sunny today.", "It is sunny today."]


def test_max_duration_timeout_with_partial_audio_still_runs_stt(config, running_reference_server):
    server, events = running_reference_server
    segmenter_script = [SegmentResult(SegmentEvent.MAX_DURATION_TIMEOUT, audio=b"\x00\x00" * 500)]
    vm, audio, wake, seg, stt, llm, tts = _build_manager(config, segmenter_script)

    still_alive = _run_one_cycle_in_thread(vm)
    assert not still_alive
    assert stt.calls == 1
    assert tts.played == ["It is sunny today."]
