"""Voice Manager's side of the IPC channel to HumanFollower.

This is the ONLY module in the whole Voice Manager that talks to
HumanFollower. It never issues a motor command -- it sends the small,
explicit message set defined in ``protocol.py`` and nothing else
(Section 15/16: Voice Manager requests, HumanFollower controls).

Connects to a Unix domain socket that HumanFollower's process is expected
to be listening on (see ``voice/ipc/server_stub.py`` for a reference
implementation of that listener, provided because the real HumanFollower
source was not available to integrate against directly -- see
docs/ARCHITECTURE.md "HumanFollower integration" section for the exact
contract this expects).
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Optional

from voice.config import IPCConfig
from voice.ipc.protocol import Message, MessageType, decode_lines, encode

logger = logging.getLogger(__name__)


class IPCError(RuntimeError):
    pass


class HumanFollowerLink:
    """Manages one voice-session's worth of IPC with HumanFollower.

    Usage pattern (mirrors the state machine in voice/manager/state_machine.py)::

        link = HumanFollowerLink(config.ipc)
        confirmed = link.request_pause(timeout=config.ipc.pause_confirm_timeout_s)
        link.start_heartbeat()
        ...
        link.stop_heartbeat()
        link.session_complete(outcome="answered")
        link.close()

    If the socket can't be reached at all (HumanFollower not running, or
    not yet listening), every method fails soft (returns False / logs and
    continues) rather than raising -- per Section 17, a Voice Manager that
    can't reach HumanFollower must not itself become a source of
    instability. The state machine treats "can't confirm pause" as "assume
    the worst, proceed cautiously and briefly" per the documented failure
    policy in docs/ARCHITECTURE.md.
    """

    def __init__(self, config: IPCConfig) -> None:
        self.config = config
        self._sock: Optional[socket.socket] = None
        self._recv_buf = b""
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        self._session_id: Optional[str] = None

    # ------------------------------------------------------------------
    def connect(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.settimeout(self.config.connect_timeout_s)
            self._sock.connect(self.config.socket_path)
            self._sock.settimeout(self.config.message_timeout_s)
            return True
        except OSError as exc:
            logger.error("Could not connect to HumanFollower IPC socket %s: %s",
                         self.config.socket_path, exc)
            self._sock = None
            return False

    def close(self) -> None:
        self.stop_heartbeat()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # ------------------------------------------------------------------
    def _send(self, msg: Message) -> bool:
        if self._sock is None:
            return False
        try:
            self._sock.sendall(encode(msg))
            return True
        except OSError as exc:
            logger.error("IPC send failed (%s): %s", msg.type.value, exc)
            return False

    def _recv_one(self, expected: MessageType, timeout: float) -> Optional[Message]:
        if self._sock is None:
            return None
        deadline = time.monotonic() + timeout
        try:
            self._sock.settimeout(max(0.05, deadline - time.monotonic()))
            while time.monotonic() < deadline:
                chunk = self._sock.recv(4096)
                if not chunk:
                    return None
                messages, self._recv_buf = decode_lines(self._recv_buf + chunk)
                for m in messages:
                    if m.type == expected:
                        return m
                    # Anything else (e.g. an ERROR) is logged and skipped --
                    # the caller's timeout still governs.
                    logger.info("IPC: received %s while waiting for %s", m.type.value, expected.value)
        except socket.timeout:
            return None
        except OSError as exc:
            logger.error("IPC recv failed: %s", exc)
            return None
        return None

    # ------------------------------------------------------------------
    def request_pause(self) -> bool:
        """Send PAUSE_REQUEST and wait (bounded) for PAUSE_CONFIRMED.

        Returns True if confirmed, False otherwise (timeout, no
        connection, or explicit ERROR). The state machine proceeds into
        LISTENING either way after this call returns -- PAUSE_CONFIRMED is
        a nice-to-have synchronization point, not a hard precondition,
        because HumanFollower's own deceleration logic is the actual source
        of truth for whether motors are stopped, and Voice Manager has no
        way to verify that directly by design (it has no motor authority).
        """
        if self._sock is None and not self.connect():
            return False

        msg = Message(type=MessageType.PAUSE_REQUEST)
        self._session_id = msg.session_id
        if not self._send(msg):
            return False

        reply = self._recv_one(MessageType.PAUSE_CONFIRMED, self.config.pause_confirm_timeout_s)
        if reply is None:
            logger.warning(
                "No PAUSE_CONFIRMED within %.1fs -- proceeding without confirmation "
                "(see docs/ARCHITECTURE.md failure policy).",
                self.config.pause_confirm_timeout_s,
            )
            return False
        return True

    def session_complete(self, outcome: str) -> bool:
        """Send VOICE_SESSION_COMPLETE. `outcome` is a short machine-readable
        tag (e.g. "answered", "no_speech_timeout", "stt_failed",
        "llm_failed", "tts_failed") logged on both sides for diagnostics.
        """
        self.stop_heartbeat()
        msg = Message(
            type=MessageType.VOICE_SESSION_COMPLETE,
            session_id=self._session_id or "",
            extra={"outcome": outcome},
        )
        return self._send(msg)

    def send_error(self, reason: str) -> bool:
        msg = Message(type=MessageType.ERROR, session_id=self._session_id or "", reason=reason)
        return self._send(msg)

    # ------------------------------------------------------------------
    def start_heartbeat(self) -> None:
        """Begin sending periodic HEARTBEAT messages until stop_heartbeat().

        This is what lets HumanFollower's watchdog (Section 17) tell the
        difference between "voice session still legitimately in progress"
        and "Voice Manager process died mid-session" -- see
        WatchdogConfig.heartbeat_timeout_s.
        """
        if self._heartbeat_thread is not None:
            return
        self._stop_heartbeat.clear()

        def _loop() -> None:
            while not self._stop_heartbeat.is_set():
                self._send(Message(type=MessageType.HEARTBEAT, session_id=self._session_id or ""))
                self._stop_heartbeat.wait(self.config.heartbeat_interval_s)

        self._heartbeat_thread = threading.Thread(target=_loop, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
            self._heartbeat_thread = None


__all__ = ["HumanFollowerLink", "IPCError"]
