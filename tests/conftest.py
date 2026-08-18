"""Shared pytest fixtures.

Deliberately avoids depending on real audio hardware, webrtcvad's compiled
extension, openwakeword/onnxruntime, or real whisper.cpp/Piper/Ollama
binaries -- every test in this suite exercises real logic (state machine
rules, timeout arithmetic, IPC wire format, orchestration sequencing)
against fakes/mocks for the hardware-facing edges. This is what lets the
suite run in any CI environment, not just on a UNO Q.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def fake_webrtcvad(monkeypatch):
    """Install a deterministic fake in place of the webrtcvad C extension.

    The fake's ``is_speech`` reads truth values off a queue the test
    controls, so VAD tests can script exact speech/silence sequences
    instead of depending on real audio classification.
    """
    calls = {"queue": []}

    class _FakeVad:
        def __init__(self, aggressiveness):
            self.aggressiveness = aggressiveness

        def is_speech(self, frame, sample_rate):
            if calls["queue"]:
                return calls["queue"].pop(0)
            return False

    fake_module = types.SimpleNamespace(Vad=_FakeVad)
    monkeypatch.setitem(sys.modules, "webrtcvad", fake_module)
    import voice.audio.vad as vad_mod
    monkeypatch.setattr(vad_mod, "webrtcvad", fake_module)
    return calls["queue"]
