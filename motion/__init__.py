"""Motion subsystem: manual Bluetooth (HC-05) car control, coordinated
with the voice pipeline through the SAME IPC boundary HumanFollower was
always designed to use (voice/ipc/). See motion/README.md for the full
picture; the short version:

    HC05Controller --> MovementSafetyGate --> MovementController
                              ^
                              |
                   voice/ipc/server_stub.py's ReferenceHumanFollowerServer
                   (reused unmodified -- PAUSE_REQUEST/VOICE_SESSION_COMPLETE/
                   watchdog), driven by scripts/run_motion_controller.py

This package has zero dependency on anything under voice/ except the
already-existing, already-validated IPC config/server classes -- and
voice/ has zero dependency on this package. They communicate only through
the Unix-socket IPC protocol, exactly like the "two separate OS processes"
architecture documented in docs/ARCHITECTURE.md always intended, with this
package now standing in for what that document called "HumanFollower" (a
manual-control implementation of it, not the autonomous-following one --
see MotionSafetyGate's docstring for why that's a deliberate, easy-to-
extend design, not a placeholder to rewrite later).
"""
