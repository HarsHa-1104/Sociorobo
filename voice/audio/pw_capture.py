"""Bluetooth (or any PipeWire Audio/Source) microphone capture via
pw-record -- Phase 3 of the plug-and-play design.

STATUS: standalone and unit/integration-tested, but deliberately NOT wired
into AudioManager or the production voice pipeline yet. See the Phase 3
report for the full reasoning; summarized here because it shapes this
module's design:

Getting a microphone from THIS board's actual production Bluetooth speaker
(HBTS001) would require switching its BlueZ profile away from a2dp-sink to
headset-head-unit -- confirmed directly from `pw-dump`'s EnumProfile data
on that device, not assumed. A Bluetooth audio connection offers A2DP
(playback only, high fidelity -- stereo, ~44.1kHz+, SBC) XOR HSP/HFP
(bidirectional, but narrowband -- mono, 8kHz CVSD or 16kHz mSBC) for a
given device, never both at once. Switching HBTS001 to get a microphone
would therefore also downgrade its speaker output from the
already-validated, human-confirmed-clear A2DP audio (Phase 1) to
narrowband HFP/HSP quality, for both directions, for as long as that
profile is active -- a real, permanent product tradeoff, not an
implementation detail, and not something to decide unilaterally inside a
capture module.

This module is therefore intentionally generic and profile-agnostic: it
can capture from ANY PipeWire Audio/Source node by name, including a
DIFFERENT Bluetooth device's microphone -- a separate physical device has
its own independent BlueZ connection and profile, with no conflict against
HBTS001's A2DP speaker role. No second Bluetooth device was available to
validate this against on real hardware in this session; only a fake
subprocess (see tests/fake_pw_record_stub.py) and unit tests cover the
logic here. Wiring this into AudioManager's live capture path -- and
deciding whether/how to ever touch HBTS001's own profile -- is left as an
explicitly separate, not-yet-authorized step.

Required backend: pw-record, not PyAudio/PortAudio. Confirmed in Phase 0:
this board has no PipeWire-ALSA compatibility plugin, so PyAudio cannot
open a Bluetooth source at all -- the exact same reason `aplay` cannot
reach a Bluetooth sink (see voice/tts/pipewire_playback.py).

Whatever device this module ends up capturing from will be mono, 8-16kHz
narrowband (HFP/HSP; see above) -- NOT the 48kHz the wired USB mic uses.
A future integration must read the actual negotiated rate/channels rather
than assuming today's USB values, and downstream VAD timing constants
(voice/audio/vad.py, tuned against the wired USB mic's audio
characteristics) should be re-validated for narrowband audio before being
trusted for a Bluetooth mic in production -- both flagged, neither
addressed here.
"""

from __future__ import annotations

import logging
import os
import select
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


class CaptureStartError(RuntimeError):
    """Raised when the pw-record subprocess cannot be started at all
    (e.g. the binary is missing)."""


class PipeWireCapture:
    """Streams raw S16LE PCM from a PipeWire Audio/Source node via a
    long-lived `pw-record` subprocess -- the capture-side mirror of
    voice/tts/pipewire_playback.py's play_via_pipewire, but continuous
    (one long-lived process feeding repeated reads) rather than one-shot.
    """

    def __init__(
        self,
        target_node: str,
        rate: int,
        channels: int = 1,
        binary_path: str = "pw-record",
    ) -> None:
        self.target_node = target_node
        self.rate = rate
        self.channels = channels
        self.binary_path = binary_path
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if self._proc is not None:
            return

        cmd = [
            self.binary_path,
            "--target", self.target_node,
            "--rate", str(self.rate),
            "--channels", str(self.channels),
            "--format", "s16",
            "--raw",
            "-",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # matches run_with_group_kill's Milestone 7 rationale
            )
        except FileNotFoundError as exc:
            raise CaptureStartError(
                f"`{self.binary_path}` not found -- is pipewire-bin installed?"
            ) from exc

        logger.info(
            "PipeWireCapture started: target=%s rate=%dHz channels=%d",
            self.target_node, self.rate, self.channels,
        )

    def read(self, num_bytes: int, timeout_s: float = 1.0) -> Optional[bytes]:
        """Bounded read of exactly num_bytes, waiting at most timeout_s
        total. Returns None on EOF, timeout, or if capture was never
        started -- in every case the caller should treat this the same
        way AudioManager treats a PyAudio read error: a signal to attempt
        recovery, never a reason to block indefinitely. Mirrors
        PersistentPiperTTS._read_exact's bounded-read discipline
        (voice/tts/persistent_piper_tts.py), for the same real-hardware
        motivated reason: a hung or silently-dead capture source must
        never be able to hang the whole voice pipeline waiting on it.
        """
        if self._proc is None or self._proc.stdout is None:
            return None

        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + timeout_s
        buf = bytearray()
        while len(buf) < num_bytes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            r, _, _ = select.select([fd], [], [], remaining)
            if not r:
                return None
            chunk = os.read(fd, num_bytes - len(buf))
            if not chunk:
                return None  # EOF -- process likely died
            buf.extend(chunk)
        return bytes(buf)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=2.0)
        except Exception:
            logger.exception("Error stopping pw-record process")
        finally:
            self._proc = None
            logger.info("PipeWireCapture stopped: target=%s", self.target_node)


__all__ = ["PipeWireCapture", "CaptureStartError"]
