"""Wire protocol between Voice Manager and HumanFollower.

Deliberately simple, per Section 15's own instruction not to over-engineer
this: newline-delimited JSON over a Unix domain socket. One message per
line, one connection per voice session (opened on wake, closed after the
session completes or is abandoned).

Message types (Section 15/16/17 of the spec):

    PAUSE_REQUEST        Voice Manager -> HumanFollower.  Wake word fired;
                          please decelerate and stop.
    PAUSE_CONFIRMED      HumanFollower -> Voice Manager.  Motors are
                          stopped; safe to start listening. Optional but
                          recommended -- see IPCConfig.pause_confirm_timeout_s
                          for what Voice Manager does if this never arrives.
    VOICE_SESSION_COMPLETE  Voice Manager -> HumanFollower.  TTS playback
                          (or an early-abort path) has fully finished;
                          resume following.
    HEARTBEAT             Voice Manager -> HumanFollower, periodic during
                          a session. HumanFollower's watchdog uses the
                          *absence* of these to detect a dead/hung Voice
                          Manager (Section 17) -- see docs/ARCHITECTURE.md.
    ERROR                  Either direction. Carries a human-readable reason
                          field; never implies any motor action by itself --
                          the receiver decides what an ERROR means for its
                          own state.

No message in this protocol ever contains a motor command. That is
intentional and load-bearing: Section 15/16 require that Voice Manager
(and by extension the LLM) can never direct motors, even indirectly through
a permissive message schema.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MessageType(str, Enum):
    PAUSE_REQUEST = "PAUSE_REQUEST"
    PAUSE_CONFIRMED = "PAUSE_CONFIRMED"
    VOICE_SESSION_COMPLETE = "VOICE_SESSION_COMPLETE"
    HEARTBEAT = "HEARTBEAT"
    ERROR = "ERROR"


@dataclass
class Message:
    type: MessageType
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)
    reason: Optional[str] = None       # populated on ERROR
    extra: Optional[dict] = None       # small optional payload, e.g. {"outcome": "no_speech_timeout"}

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "type": self.type.value,
            "session_id": self.session_id,
            "ts": self.ts,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.extra is not None:
            payload["extra"] = self.extra
        return json.dumps(payload, separators=(",", ":"))

    @staticmethod
    def from_json(raw: str) -> "Message":
        data = json.loads(raw)
        return Message(
            type=MessageType(data["type"]),
            session_id=data.get("session_id", ""),
            ts=data.get("ts", time.time()),
            reason=data.get("reason"),
            extra=data.get("extra"),
        )


def encode(msg: Message) -> bytes:
    return (msg.to_json() + "\n").encode("utf-8")


def decode_lines(buf: bytes) -> tuple[list[Message], bytes]:
    """Split buffered bytes into complete newline-delimited messages.

    Returns (messages, remainder) -- remainder is any trailing partial line
    that should be prepended to the next recv() chunk.
    """
    messages: list[Message] = []
    *lines, remainder = buf.split(b"\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            messages.append(Message.from_json(line.decode("utf-8")))
        except (json.JSONDecodeError, KeyError, ValueError):
            # Malformed message -- drop it rather than crash the IPC loop.
            # A dropped message is exactly the kind of thing the watchdog
            # (Section 17) exists to be robust against.
            continue
    return messages, remainder


__all__ = ["MessageType", "Message", "encode", "decode_lines"]
