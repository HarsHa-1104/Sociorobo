"""Reference HumanFollower-side IPC server + watchdog.

IMPORTANT CONTEXT: the real HumanFollower source code was not included in
the material provided for this project, so it could not be edited
directly. This module is a REFERENCE IMPLEMENTATION -- a small, complete,
runnable stand-in for "whatever HumanFollower's process does with these
messages" -- provided for two purposes:

  1. It lets the Voice Manager be integration-tested end-to-end (Section 23)
     without needing the real HumanFollower process.
  2. It documents, in executable form, exactly what HumanFollower needs to
     implement: listen on a Unix socket, respond to PAUSE_REQUEST with
     PAUSE_CONFIRMED once motors are actually stopped, track HEARTBEATs,
     and enforce the watchdog timeout independently of Voice Manager's own
     behavior.

Whoever owns the real HumanFollower codebase should port the
``_on_pause_request`` / ``_on_session_complete`` / ``_watchdog_loop`` logic
into HumanFollower's actual control loop, calling its real deceleration and
resume functions where this stub only calls the injected
``on_pause``/``on_resume`` callbacks. Nothing here should be run verbatim
in production -- it has no real motor underneath it.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Callable, Optional

from voice.config import IPCConfig, WatchdogConfig
from voice.ipc.protocol import Message, MessageType, decode_lines, encode

logger = logging.getLogger(__name__)


class ReferenceHumanFollowerServer:
    """A minimal, honest stand-in for HumanFollower's IPC + watchdog logic.

    Parameters
    ----------
    on_pause:
        Called when a PAUSE_REQUEST arrives. In the real system this is
        where HumanFollower's existing controlled-deceleration routine
        gets invoked. Must return True once motors are confirmed stopped
        (this stub calls it synchronously and treats any return as
        immediate confirmation -- the real implementation may need to
        poll motor state instead).
    on_resume:
        Called when a VOICE_SESSION_COMPLETE arrives, OR when the watchdog
        fires. In the real system this resumes HumanFollower's normal
        following behavior via its existing controller, never via any
        motor command invented here.
    on_watchdog_forced_resume:
        Optional callback fired specifically when the watchdog -- not a
        normal VOICE_SESSION_COMPLETE -- is what triggered the resume.
        Useful for logging/alerting that Voice Manager appears to have
        failed.
    """

    def __init__(
        self,
        ipc_config: IPCConfig,
        watchdog_config: WatchdogConfig,
        on_pause: Callable[[], bool],
        on_resume: Callable[[], None],
        on_watchdog_forced_resume: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.ipc_config = ipc_config
        self.watchdog_config = watchdog_config
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.on_watchdog_forced_resume = on_watchdog_forced_resume

        self._server_sock: Optional[socket.socket] = None
        self._running = threading.Event()
        self._accept_thread: Optional[threading.Thread] = None

        self._paused_since: Optional[float] = None
        self._last_heartbeat: Optional[float] = None
        self._session_active = False
        self._state_lock = threading.Lock()
        self._watchdog_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        path = self.ipc_config.socket_path
        if os.path.exists(path):
            os.unlink(path)
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(path)
        self._server_sock.listen(1)
        self._running.set()

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

        logger.info("Reference HumanFollower IPC server listening on %s", path)

    def stop(self) -> None:
        self._running.clear()
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
        if os.path.exists(self.ipc_config.socket_path):
            try:
                os.unlink(self.ipc_config.socket_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    def _accept_loop(self) -> None:
        while self._running.is_set():
            try:
                self._server_sock.settimeout(0.5)
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

    def _handle_conn(self, conn: socket.socket) -> None:
        buf = b""
        conn.settimeout(1.0)
        while self._running.is_set():
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            messages, buf = decode_lines(buf + chunk)
            for msg in messages:
                self._dispatch(msg, conn)
        try:
            conn.close()
        except OSError:
            pass

    def _dispatch(self, msg: Message, conn: socket.socket) -> None:
        if msg.type == MessageType.PAUSE_REQUEST:
            logger.info("[HumanFollower ref] PAUSE_REQUEST (session=%s)", msg.session_id)
            confirmed = False
            try:
                confirmed = bool(self.on_pause())
            except Exception:  # pragma: no cover - defensive
                logger.exception("on_pause callback raised")

            with self._state_lock:
                self._paused_since = time.monotonic()
                self._last_heartbeat = time.monotonic()
                self._session_active = True

            if confirmed:
                try:
                    conn.sendall(encode(Message(type=MessageType.PAUSE_CONFIRMED, session_id=msg.session_id)))
                except OSError:
                    pass

        elif msg.type == MessageType.HEARTBEAT:
            with self._state_lock:
                self._last_heartbeat = time.monotonic()

        elif msg.type == MessageType.VOICE_SESSION_COMPLETE:
            outcome = (msg.extra or {}).get("outcome", "unknown")
            logger.info("[HumanFollower ref] VOICE_SESSION_COMPLETE (session=%s outcome=%s)",
                        msg.session_id, outcome)
            with self._state_lock:
                self._session_active = False
                self._paused_since = None
            self.on_resume()

        elif msg.type == MessageType.ERROR:
            logger.warning("[HumanFollower ref] ERROR from Voice Manager: %s", msg.reason)

    # ------------------------------------------------------------------
    def _watchdog_loop(self) -> None:
        """Section 17: HumanFollower must not wait forever on Voice Manager.

        Two independent tripwires, either of which forces a resume:

          * ``max_paused_duration_s`` -- absolute ceiling from PAUSE_REQUEST,
            regardless of heartbeats. Catches a Voice Manager that is alive
            and heartbeating but stuck in a genuine infinite loop.
          * ``heartbeat_timeout_s`` -- no heartbeat for this long during an
            active session strongly suggests the Voice Manager process
            died outright.

        This loop is the deterministic safety net Section 17 asks for:
        "if there is a case where the robot should remain stopped instead
        of automatically resuming, use the safest deterministic behavior
        and document it." The documented policy here is the opposite
        choice, made deliberately: prefer resuming known-good autonomous
        following over leaving the robot motionless indefinitely, because
        an unresponsive stationary robot is not obviously safer than one
        that resumes a well-tested following behavior, and a robot stuck
        forever is a worse failure mode for its owner. If your deployment
        's safety analysis concludes the opposite (e.g. resuming into a
        cluttered/hazardous environment is worse than staying put), flip
        this policy to enter SAFE_STOP instead of RESUMING here -- that is
        a one-line change, isolated to this method.
        """
        while self._running.is_set():
            time.sleep(0.5)
            with self._state_lock:
                if not self._session_active or self._paused_since is None:
                    continue
                paused_elapsed = time.monotonic() - self._paused_since
                heartbeat_elapsed = (
                    time.monotonic() - self._last_heartbeat
                    if self._last_heartbeat is not None else paused_elapsed
                )
                trip_reason = None
                if paused_elapsed >= self.watchdog_config.max_paused_duration_s:
                    trip_reason = f"max_paused_duration_s ({self.watchdog_config.max_paused_duration_s}s) exceeded"
                elif heartbeat_elapsed >= self.watchdog_config.heartbeat_timeout_s:
                    trip_reason = f"heartbeat_timeout_s ({self.watchdog_config.heartbeat_timeout_s}s) exceeded"

                if trip_reason is None:
                    continue

                self._session_active = False
                self._paused_since = None

            logger.error("[HumanFollower ref] WATCHDOG TRIPPED: %s -- forcing resume.", trip_reason)
            if self.on_watchdog_forced_resume is not None:
                try:
                    self.on_watchdog_forced_resume(trip_reason)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("on_watchdog_forced_resume callback raised")
            self.on_resume()


__all__ = ["ReferenceHumanFollowerServer"]
