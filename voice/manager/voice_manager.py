"""VoiceManager -- the orchestrator that ties every module together.

This is the only place in the codebase that sequences wake -> pause ->
listen -> STT -> LLM -> TTS -> resume. Every collaborator is injected via
the constructor rather than constructed internally, specifically so this
class can be unit- and integration-tested with fakes/mocks standing in for
real audio hardware, whisper.cpp, Ollama, Piper, and the IPC link (see
tests/test_voice_manager.py). Use :func:`build_voice_manager` to wire up
the real, hardware-backed collaborators for actual on-device use.

Hard rules enforced by this class's structure, not just by comments
(Sections 6/7/15/16 of the spec):

  * The wake-word detector's ``process_frame`` is only ever called from
    :meth:`_wait_for_wake`. Nowhere else in this class references it. That
    means wake-word detection being "off" during a session isn't a flag
    that could be forgotten -- it's a code path that simply doesn't exist
    during a session.
  * This class never issues a motor command, imports anything motor-
    related, or accepts a motor-control callback. Its only channel to
    HumanFollower is ``HumanFollowerLink``, which only carries the
    Section 15 message set.
  * ``SPEAKING`` always resolves to ``SESSION_COMPLETE`` -- there is no
    path back to LISTENING from within a single wake cycle (Section 5: one
    wake word, one question, one response).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from voice.audio.manager import AudioManager
from voice.audio.vad import SegmentEvent, SpeechSegmenter
from voice.config import VoiceSystemConfig
from voice.ipc.client import HumanFollowerLink
from voice.llm.ollama_client import OllamaClient
from voice.manager.state_machine import VoiceState, validate_transition
from voice.stt.whisper_cpp import WhisperCppSTT
from voice.tts.piper_tts import PiperTTS
from voice.wake.wake_word import WakeWordDetector

logger = logging.getLogger(__name__)


class VoiceManager:
    def __init__(
        self,
        config: VoiceSystemConfig,
        audio: AudioManager,
        wake_detector: WakeWordDetector,
        segmenter: SpeechSegmenter,
        stt: WhisperCppSTT,
        llm: OllamaClient,
        tts: PiperTTS,
    ) -> None:
        self.config = config
        self.audio = audio
        self.wake_detector = wake_detector
        self.segmenter = segmenter
        self.stt = stt
        self.llm = llm
        self.tts = tts

        self._state = VoiceState.WAKE_LISTENING
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._session_count = 0

    # ------------------------------------------------------------------
    @property
    def state(self) -> VoiceState:
        with self._state_lock:
            return self._state

    def _transition(self, to: VoiceState) -> None:
        with self._state_lock:
            validate_transition(self._state, to)
            logger.debug("State: %s -> %s", self._state.name, to.name)
            self._state = to

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        self.audio.start()
        # WAKE_LISTENING is the resting state; __init__ already starts here
        # and _end_session() always returns here via a validated
        # SESSION_COMPLETE -> WAKE_LISTENING transition, so this loop never
        # needs to (and must not) re-assert a WAKE_LISTENING -> WAKE_LISTENING
        # self-transition, which the state graph correctly rejects as a
        # no-op bug magnet.
        try:
            while not self._stop.is_set():
                if self._wait_for_wake():
                    self._session_count += 1
                    logger.info("=== Voice session #%d starting ===", self._session_count)
                    self._run_session()
        finally:
            self.audio.stop()

    # ------------------------------------------------------------------
    def _wait_for_wake(self) -> bool:
        """Wait for wake word, or immediately start a session when disabled."""

        # Must happen unconditionally, before either branch below: a real
        # wake-word fire (the loop just below) never transitioned state
        # itself -- it only returned True -- so _run_session()'s first move
        # (WAKE_LISTENING -> PAUSE_PENDING) would otherwise be attempted
        # from SESSION_COMPLETE on the second and every later cycle, which
        # the state graph correctly rejects and crashes the whole process.
        # Confirmed on real UNO Q hardware: this was masked in every prior
        # test because none of them ran a second cycle, and masked in
        # manual testing while wake was disabled, because the bypass branch
        # below happened to do this transition itself already.
        if self._state == VoiceState.SESSION_COMPLETE:
            self._transition(VoiceState.WAKE_LISTENING)

        if not self.config.wake.enabled:
            logger.info("Wake disabled -- starting voice session directly.")
            return True

        for frame in self.audio.frames():
            if self._stop.is_set():
                return False
            if self.wake_detector.process_frame(frame):
                logger.info("Wake word detected.")
                return True
        return False

    # ------------------------------------------------------------------
    def _run_session(self) -> None:
        # Milestone 8: per-stage wall-clock timing, logged as a single
        # structured summary at the end of every session (INFO level, not
        # DEBUG -- this is the baseline profiling data, meant to be always
        # available for future optimization work, not a one-off measurement).
        session_t0 = time.perf_counter()
        stage_durations_s: dict = {}

        # Wake word must not be able to re-fire on our own audio for the
        # rest of this cycle -- suspend immediately (Section 7).
        self.audio.suspend()

        link = HumanFollowerLink(self.config.ipc)
        self._transition(VoiceState.PAUSE_PENDING)
        t0 = time.perf_counter()
        confirmed = link.request_pause()
        stage_durations_s["pause_handshake"] = time.perf_counter() - t0
        if not confirmed:
            logger.warning(
                "Proceeding into LISTENING without PAUSE_CONFIRMED "
                "(HumanFollower unreachable or slow to confirm)."
            )
        link.start_heartbeat()

        t0 = time.perf_counter()
        outcome, speech_audio = self._listen_for_utterance()
        stage_durations_s["listening"] = time.perf_counter() - t0
        if speech_audio is None:
            self._log_session_timing(stage_durations_s, session_t0, outcome)
            self._end_session(link, outcome)
            return

        t0 = time.perf_counter()
        transcript = self._run_stt(speech_audio)
        stage_durations_s["stt"] = time.perf_counter() - t0
        if not transcript:
            self._log_session_timing(stage_durations_s, session_t0, "stt_empty_or_failed")
            self._end_session(link, "stt_empty_or_failed")
            return

        t0 = time.perf_counter()
        reply = self._run_llm(transcript)
        stage_durations_s["llm"] = time.perf_counter() - t0
        if not reply:
            self._log_session_timing(stage_durations_s, session_t0, "llm_empty_or_failed")
            self._end_session(link, "llm_empty_or_failed")
            return

        t0 = time.perf_counter()
        outcome = self._run_tts(reply)
        stage_durations_s["tts"] = time.perf_counter() - t0

        self._log_session_timing(stage_durations_s, session_t0, outcome)
        self._end_session(link, outcome)

    def _log_session_timing(self, stages: dict, session_t0: float, outcome: str) -> None:
        total = time.perf_counter() - session_t0
        breakdown = " ".join(f"{name}={dur:.2f}s" for name, dur in stages.items())
        logger.info(
            "Session timing: total=%.2fs outcome=%s [%s]",
            total, outcome, breakdown,
        )

    # ------------------------------------------------------------------
    def _listen_for_utterance(self):
        """Returns (outcome_tag, audio_bytes_or_None).

        audio is None whenever there is nothing worth sending to STT
        (no-speech timeout, or a max-duration timeout with zero captured
        speech) -- the caller ends the session immediately in that case.
        """
        self._transition(VoiceState.LISTENING)
        self.audio.resume()  # we need to actually hear the question now
        self.segmenter.start()

        outcome = "listening_aborted"
        audio: Optional[bytes] = None

        for frame in self.audio.frames():
            if self._stop.is_set():
                outcome = "shutdown"
                break

            result = self.segmenter.process_frame(frame)
            if result.event == SegmentEvent.SPEECH_ENDED:
                outcome = "speech_captured"
                audio = result.audio
                break
            if result.event == SegmentEvent.NO_SPEECH_TIMEOUT:
                logger.info("No speech detected within no_speech_timeout_s -- ending session.")
                outcome = "no_speech_timeout"
                break
            if result.event == SegmentEvent.MAX_DURATION_TIMEOUT:
                if result.audio:
                    logger.info("Hit 20s hard ceiling mid-speech -- sending partial capture to STT.")
                    outcome = "max_duration_partial_capture"
                    audio = result.audio
                else:
                    logger.info("Hit 20s hard ceiling with no confirmed speech -- ending session.")
                    outcome = "max_duration_no_speech"
                break

        # Mic not needed again until the next wake cycle -- suspend through
        # STT/LLM/TTS regardless of how this phase ended (Section 7).
        self.audio.suspend()
        return outcome, audio

    # ------------------------------------------------------------------
    def _run_stt(self, speech_audio: bytes) -> str:
        self._transition(VoiceState.PROCESSING_STT)
        try:
            # speech_audio comes from AudioManager.frames() via the segmenter,
            # which always yields PIPELINE_SAMPLE_RATE (16kHz) audio -- NOT
            # config.audio.sample_rate (the raw mic capture rate, 48kHz on
            # this board). Passing the wrong rate here mislabels the WAV
            # header STT sees, which is a real bug found on hardware during
            # Milestone 6 (confirmed via saved temp WAVs: header said 48kHz
            # for genuinely-16kHz data), not just cosmetic -- it corrupts
            # whisper's duration/pitch interpretation of the audio entirely.
            transcript = self.stt.run_stt(speech_audio, sample_rate=AudioManager.PIPELINE_SAMPLE_RATE)
        except Exception:
            logger.exception("STT raised unexpectedly")
            return ""
        logger.info("Transcript: %r", transcript)
        return transcript.strip()

    def _run_llm(self, transcript: str) -> str:
        self._transition(VoiceState.PROCESSING_LLM)
        try:
            reply = self.llm.query(transcript)
        except Exception:
            logger.exception("LLM query raised unexpectedly")
            return ""
        logger.info("LLM reply: %r", reply)
        return reply.strip()

    def _run_tts(self, reply: str) -> str:
        self._transition(VoiceState.SPEAKING)
        try:
            ok = self.tts.synthesize_and_play(reply, alsa_device=self.config.audio.output_device)
        except Exception:
            logger.exception("TTS raised unexpectedly")
            ok = False
        # Section 11: never resume before playback is completely finished.
        # synthesize_and_play() already blocks on subprocess completion, so
        # by the time we get here playback is genuinely done, not inferred.
        time.sleep(self.config.vad.post_tts_cooldown_s)
        return "answered" if ok else "tts_failed"

    def _end_session(self, link: HumanFollowerLink, outcome: str) -> None:
        self._transition(VoiceState.SESSION_COMPLETE)
        link.session_complete(outcome)
        link.close()
        self.audio.resume()  # ready for wake-word listening again
        self.wake_detector.reset()
        logger.info("=== Voice session ended: outcome=%s ===", outcome)


# ---------------------------------------------------------------------------
def build_voice_manager(config: VoiceSystemConfig) -> VoiceManager:
    """Wire up real, hardware-backed collaborators for on-device use.

    Kept separate from VoiceManager.__init__ so tests never accidentally
    depend on real audio hardware / subprocess binaries just by
    constructing a VoiceManager.
    """
    audio = AudioManager(config.audio)
    # Both of these consume frames from audio.frames(), which always yields
    # PIPELINE_SAMPLE_RATE (16kHz) audio regardless of the raw mic capture
    # rate (config.audio.sample_rate, 48kHz on this board) -- see the note
    # in _run_stt() above for the real bug this class of mistake caused.
    # SpeechSegmenter in particular feeds this straight into webrtcvad's
    # is_speech(frame, sample_rate), which silently accepts a wrong-but-
    # byte-length-compatible rate rather than raising, so this was corrupting
    # every VAD decision on real hardware without ever erroring.
    wake_detector = WakeWordDetector(config.wake, sample_rate=AudioManager.PIPELINE_SAMPLE_RATE)
    segmenter = SpeechSegmenter(config.vad, sample_rate=AudioManager.PIPELINE_SAMPLE_RATE,
                                 frame_duration_ms=config.audio.frame_duration_ms)
    stt = WhisperCppSTT(config.stt)
    llm = OllamaClient(config.llm)
    # Milestone 8: config.tts.persistent selects PersistentPiperTTS (keeps
    # one Piper process alive across requests, ~2-5x faster on real-hardware
    # benchmarks -- see docs/MILESTONE_8_PIPER_OPTIMIZATION.md) over the
    # original per-call PiperTTS. Both implement the same interface, so this
    # is the only place the choice is made.
    if config.tts.persistent:
        from voice.tts.persistent_piper_tts import PersistentPiperTTS
        tts = PersistentPiperTTS(config.tts)
    else:
        tts = PiperTTS(config.tts)

    return VoiceManager(config, audio, wake_detector, segmenter, stt, llm, tts)


__all__ = ["VoiceManager", "build_voice_manager"]
