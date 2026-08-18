from __future__ import annotations

import pytest

from voice.manager.state_machine import IllegalTransitionError, VoiceState, validate_transition


def test_full_happy_path_is_legal():
    path = [
        VoiceState.WAKE_LISTENING,
        VoiceState.PAUSE_PENDING,
        VoiceState.LISTENING,
        VoiceState.PROCESSING_STT,
        VoiceState.PROCESSING_LLM,
        VoiceState.SPEAKING,
        VoiceState.SESSION_COMPLETE,
        VoiceState.WAKE_LISTENING,
    ]
    for a, b in zip(path, path[1:]):
        validate_transition(a, b)  # must not raise


@pytest.mark.parametrize("frm,to", [
    (VoiceState.LISTENING, VoiceState.SESSION_COMPLETE),           # no-speech / max-duration-no-speech
    (VoiceState.PROCESSING_STT, VoiceState.SESSION_COMPLETE),      # empty/failed transcript
    (VoiceState.PROCESSING_LLM, VoiceState.SESSION_COMPLETE),      # empty/failed LLM reply
])
def test_early_exit_paths_are_legal(frm, to):
    validate_transition(frm, to)


def test_cannot_skip_from_wake_listening_straight_to_listening():
    """Every session must go through PAUSE_PENDING -- you cannot start
    listening without at least attempting to pause HumanFollower first.
    """
    with pytest.raises(IllegalTransitionError):
        validate_transition(VoiceState.WAKE_LISTENING, VoiceState.LISTENING)


def test_cannot_go_from_speaking_back_to_listening():
    """Section 5: one wake word = one question = one response. There is no
    legal path from SPEAKING back into LISTENING within the same cycle.
    """
    with pytest.raises(IllegalTransitionError):
        validate_transition(VoiceState.SPEAKING, VoiceState.LISTENING)


def test_cannot_go_directly_from_wake_listening_to_speaking():
    with pytest.raises(IllegalTransitionError):
        validate_transition(VoiceState.WAKE_LISTENING, VoiceState.SPEAKING)
