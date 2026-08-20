"""Cross-role device-combination policy: at most one of {microphone,
speaker} may be Bluetooth at a time (product requirement, not a technical
limitation of any specific device).

Why this exists: microphone and speaker selection happen independently
(voice/audio/manager.py and voice/tts/persistent_piper_tts.py don't
reference each other) and at different times -- the microphone is resolved
once at AudioManager.start(), the speaker is re-resolved on every TTS call.
Enforcing a rule that spans both roles needs a small piece of state they
both see. ComboGuard is that shared state: both roles report which backend
they're currently using, and both roles ask it whether a candidate backend
would be allowed before selecting it.

This does not implement any Bluetooth-specific logic itself (profile
switching, codec negotiation, etc.) -- it is a pure, tiny coordination
primitive. The actual "why can't both sides be Bluetooth" reasoning
(A2DP/HFP profile mutual exclusion, confirmed on real hardware against
HBTS001 during the Phase 3 investigation) lives in the callers' docs, not
here -- this module doesn't need to know why, only that the rule applies.
"""

from __future__ import annotations

import threading
from typing import Optional

BLUETOOTH_BACKEND = "bluez5"


class ComboGuard:
    """Thread-safe. One instance is shared by the microphone role
    (AudioManager) and the speaker role (PersistentPiperTTS) for the
    lifetime of a VoiceManager -- see build_voice_manager(). Never share
    one instance across unrelated VoiceManager instances/tests; each
    needs its own, the same way each role has its own DeviceSelector.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._microphone_backend: Optional[str] = None
        self._speaker_backend: Optional[str] = None

    def set_microphone_backend(self, backend: Optional[str]) -> None:
        with self._lock:
            self._microphone_backend = backend

    def set_speaker_backend(self, backend: Optional[str]) -> None:
        with self._lock:
            self._speaker_backend = backend

    @property
    def microphone_backend(self) -> Optional[str]:
        with self._lock:
            return self._microphone_backend

    @property
    def speaker_backend(self) -> Optional[str]:
        with self._lock:
            return self._speaker_backend

    def microphone_allowed(self, backend: str) -> bool:
        """May the microphone role select a candidate of this backend,
        given the speaker's currently reported backend? Non-Bluetooth
        candidates are always allowed -- the rule only ever restricts a
        Bluetooth pick when the OTHER role is already Bluetooth."""
        if backend != BLUETOOTH_BACKEND:
            return True
        with self._lock:
            return self._speaker_backend != BLUETOOTH_BACKEND

    def speaker_allowed(self, backend: str) -> bool:
        if backend != BLUETOOTH_BACKEND:
            return True
        with self._lock:
            return self._microphone_backend != BLUETOOTH_BACKEND

    def is_bluetooth_conflict(self) -> bool:
        """True if both roles currently report Bluetooth -- should never
        happen if callers correctly consult microphone_allowed/
        speaker_allowed before selecting, but exposed as an explicit,
        checkable invariant rather than trusted implicitly."""
        with self._lock:
            return (
                self._microphone_backend == BLUETOOTH_BACKEND
                and self._speaker_backend == BLUETOOTH_BACKEND
            )


__all__ = ["ComboGuard", "BLUETOOTH_BACKEND"]
