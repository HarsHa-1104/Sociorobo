"""HC05Controller: reads single-character movement commands from an HC-05
(or any SPP-compatible Bluetooth-serial) module and forwards them to a
MovementSafetyGate.

IMPORTANT -- wiring/protocol honesty: no HC-05 wiring, serial device path,
or command character set was found anywhere in this repository or on the
live board when this was built (see the motion-integration report for the
exact commands run). Nothing here should be read as a confirmed fact about
your specific hardware/app -- both the serial port and the command mapping
are constructor parameters with no hardcoded default device path, and the
default command map below is a documented, overridable best-guess based on
the near-universal convention used by "Bluetooth RC Car"-style Android
apps, not a verified fact about whichever specific app you use.

Runs its own background reader thread -- start()/stop() never block the
caller. Designed to be tested without any real serial hardware or the
`serial` package's actual I/O: pass `serial_factory` to inject a fake.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from motion.movement_controller import DEFAULT_SPEED, MAX_SPEED
from motion.safety_gate import MovementSafetyGate

logger = logging.getLogger(__name__)


class SerialPort:
    """The minimal interface HC05Controller needs from a serial connection
    -- matches the subset of pyserial's Serial class actually used, so a
    test can inject a fake without depending on the `serial` package or
    real hardware at all."""

    def read(self, size: int = 1) -> bytes: ...  # pragma: no cover - protocol only
    def close(self) -> None: ...  # pragma: no cover - protocol only


def _default_serial_factory(port: str, baudrate: int, read_timeout_s: float) -> SerialPort:
    import serial  # imported lazily -- only needed for the real hardware path
    return serial.Serial(port=port, baudrate=baudrate, timeout=read_timeout_s)


class HC05Controller:
    """Command mapping defaults to single uppercase ASCII characters --
    F/B/L/R for the four cardinal directions, G/H/I/J for the four
    diagonals (forward_right/backward_right/backward_left/forward_left
    respectively), S for stop. This F/B/L/R/G/H/I/J/S letter scheme is a
    real, documented convention used by several widely-forked open-source
    "Bluetooth RC Car" Arduino/Android app pairs -- but it is NOT verified
    against any specific app until physically tested (see
    docs/HC05_COMMAND_DISCOVERY.md's capture procedure). Override `command_map` if
    your app sends something different (some apps use digits/a numpad
    layout for direction instead of letters -- see the module docstring's
    note about the default speed_map using digits, which would collide
    with a digit-based direction scheme; if your app uses digits for
    direction, override BOTH command_map and speed_map to avoid a clash).

    Speed defaults to a separate `speed_map`: single ASCII digits '0'-'9'
    map to a 10-level PWM speed (0..MAX_SPEED), the convention used by
    many Bluetooth RC apps' speed sliders (which typically quantise to a
    small number of discrete levels before sending a single byte). Also
    NOT verified against any specific app -- override if different.
    Updates the speed used by the NEXT direction command; sending a speed
    byte alone never moves the car by itself.

    9600 baud is the HC-05 factory default (a documented hardware fact,
    not a guess) -- override `baudrate` if the module was reconfigured.
    """

    DEFAULT_COMMAND_MAP: dict[str, str] = {
        "F": "forward",
        "B": "backward",
        "L": "left",
        "R": "right",
        "G": "forward_right",
        "H": "backward_right",
        "I": "backward_left",
        "J": "forward_left",
        "S": "stop",
    }

    # 10-level speed convention (digit '0'..'9' -> 0..MAX_SPEED), the
    # default only -- override via `speed_map` if your app sends speed
    # differently (e.g. a raw 0-255 byte, a different digit range, or no
    # separate speed byte at all).
    DEFAULT_SPEED_MAP: dict[str, int] = {str(i): (i * MAX_SPEED) // 9 for i in range(10)}

    READ_TIMEOUT_S = 0.5  # bounds each read() call so the reader thread can observe stop_flag promptly
    RECONNECT_BACKOFF_S = 1.0  # pause between reconnect attempts after a read failure, to avoid a hot error loop

    def __init__(
        self,
        gate: MovementSafetyGate,
        port: str,
        baudrate: int = 9600,
        command_map: Optional[dict[str, str]] = None,
        speed_map: Optional[dict[str, int]] = None,
        serial_factory: Optional[Callable[[str, int, float], SerialPort]] = None,
    ) -> None:
        self._gate = gate
        self._port = port
        self._baudrate = baudrate
        self._command_map = dict(command_map) if command_map is not None else dict(self.DEFAULT_COMMAND_MAP)
        self._speed_map = dict(speed_map) if speed_map is not None else dict(self.DEFAULT_SPEED_MAP)
        self._serial_factory = serial_factory or _default_serial_factory
        self._current_speed = DEFAULT_SPEED

        self._ser: Optional[SerialPort] = None
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._ser = self._serial_factory(self._port, self._baudrate, self.READ_TIMEOUT_S)
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info("HC05Controller started: port=%s baudrate=%d", self._port, self._baudrate)

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                logger.exception("Error closing HC-05 serial port")
            self._ser = None
        logger.info("HC05Controller stopped.")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _read_loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                raw = self._ser.read(1)
            except Exception as exc:
                # HC-05 connection loss / read failure: issue a defensive
                # stop immediately (the link may have dropped mid-command),
                # then attempt to reconnect -- retrying read() on the SAME
                # now-broken port object would keep detecting the failure
                # safely forever but never actually recover once the
                # device reappears.
                logger.error("HC-05 serial read failed (%s) -- issuing a defensive stop.", exc)
                self._gate.stop()
                self._reconnect()
                continue

            if not raw:
                continue  # read timeout, no byte available -- normal, not an error

            self._handle_byte(raw)

    def _reconnect(self) -> None:
        """Bounded, patient reconnect after a read failure: closes the
        stale port object (best-effort) and retries opening a fresh one
        with backoff, remaining responsive to stop() throughout. Mirrors
        the reconnect pattern already established elsewhere in this
        project (voice/audio/manager.py's recovery loops) -- never
        busy-loop on a persistent error."""
        try:
            self._ser.close()
        except Exception:
            pass

        while not self._stop_flag.is_set():
            if self._stop_flag.wait(self.RECONNECT_BACKOFF_S):
                return  # stop() was called during backoff
            try:
                self._ser = self._serial_factory(self._port, self._baudrate, self.READ_TIMEOUT_S)
                logger.info("HC-05 serial port reopened successfully.")
                return
            except Exception as exc:
                logger.warning("HC-05 reconnect attempt failed (%s) -- retrying.", exc)

    def _handle_byte(self, raw: bytes) -> None:
        char = raw.decode("ascii", errors="ignore").strip().upper()
        if not char:
            return  # whitespace/newline/non-ASCII noise -- ignore silently, not an error

        # Speed bytes are checked first and never fall through to the
        # direction map -- updating speed must never itself move the car.
        if char in self._speed_map:
            self._current_speed = self._speed_map[char]
            logger.info("HC-05: speed set to %d.", self._current_speed)
            return

        action = self._command_map.get(char)
        if action is None:
            logger.warning("HC-05: unrecognised command byte %r -- ignoring (no unsafe fallback).", char)
            return

        dispatch = {
            "forward": self._gate.request_forward,
            "backward": self._gate.request_backward,
            "left": self._gate.request_left,
            "right": self._gate.request_right,
            "forward_left": self._gate.request_forward_left,
            "forward_right": self._gate.request_forward_right,
            "backward_left": self._gate.request_backward_left,
            "backward_right": self._gate.request_backward_right,
        }
        if action == "stop":
            self._gate.stop()
        elif action in dispatch:
            dispatch[action](self._current_speed)
        else:  # pragma: no cover - defensive; only reachable via a misconfigured command_map
            logger.error("HC-05: command_map entry %r -> %r is not a recognised action -- ignoring.", char, action)


__all__ = ["HC05Controller", "SerialPort"]
