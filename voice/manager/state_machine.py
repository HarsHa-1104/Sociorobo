"""The voice interaction state machine (Section 13/28 of the spec).

This module defines the states and the *legal* transitions between them as
plain data, separate from the orchestration logic in ``voice_manager.py``.
Keeping it separate means the state graph itself -- which is what a
reviewer actually needs to sign off on -- can be read, tested, and diagrammed
without wading through I/O code.

Ownership (restated from Section 13, enforced procedurally in
voice_manager.py + the HumanFollower reference server, not by this module
directly -- this module just defines what's *legal*, not who's allowed to
trigger it):

    FOLLOWING, PAUSING, RESUMING, SAFE_STOP   -> HumanFollower
    LISTENING, PROCESSING_STT, PROCESSING_LLM,
    SPEAKING                                  -> Voice Manager

VoiceManager itself only ever lives in the Voice-Manager-owned states plus
a virtual WAKE_LISTENING state (wake word armed, everything else idle) and
PAUSE_PENDING (waiting on PAUSE_CONFIRMED). It never models FOLLOWING/
SAFE_STOP directly -- those belong to HumanFollower's own state, which this
codebase does not have access to.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, Set


class VoiceState(Enum):
    WAKE_LISTENING = auto()     # wake word armed; this is the only continuously-active heavy... actually light... component
    PAUSE_PENDING = auto()      # PAUSE_REQUEST sent, waiting (bounded) for PAUSE_CONFIRMED
    LISTENING = auto()          # session audio capture active, VAD running
    PROCESSING_STT = auto()
    PROCESSING_LLM = auto()
    SPEAKING = auto()           # TTS synthesizing and/or playing
    SESSION_COMPLETE = auto()   # terminal per-cycle state; loops back to WAKE_LISTENING


# Legal transitions. Anything not listed here is a bug if it happens.
_TRANSITIONS: Dict[VoiceState, Set[VoiceState]] = {
    VoiceState.WAKE_LISTENING: {VoiceState.PAUSE_PENDING},
    VoiceState.PAUSE_PENDING: {VoiceState.LISTENING},  # proceeds whether or not PAUSE_CONFIRMED arrived -- see HumanFollowerLink docstring
    VoiceState.LISTENING: {VoiceState.PROCESSING_STT, VoiceState.SESSION_COMPLETE},  # SESSION_COMPLETE = no-speech or max-duration timeout
    VoiceState.PROCESSING_STT: {VoiceState.PROCESSING_LLM, VoiceState.SESSION_COMPLETE},  # SESSION_COMPLETE = empty transcript or STT failure
    VoiceState.PROCESSING_LLM: {VoiceState.SPEAKING, VoiceState.SESSION_COMPLETE},  # SESSION_COMPLETE = empty/failed LLM response
    VoiceState.SPEAKING: {VoiceState.SESSION_COMPLETE},  # always -- whether TTS succeeded or failed, the session ends here
    VoiceState.SESSION_COMPLETE: {VoiceState.WAKE_LISTENING},
}


class IllegalTransitionError(RuntimeError):
    pass


def validate_transition(frm: VoiceState, to: VoiceState) -> None:
    if to not in _TRANSITIONS.get(frm, set()):
        raise IllegalTransitionError(f"{frm.name} -> {to.name} is not a legal transition")


__all__ = ["VoiceState", "IllegalTransitionError", "validate_transition"]
