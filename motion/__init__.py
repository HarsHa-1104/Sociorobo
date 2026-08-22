"""Motion subsystem: manual Bluetooth car control, coordinated with the
voice pipeline through the SAME IPC boundary HumanFollower was always
designed to use (voice/ipc/). TWO deployments exist, sharing every class
in this package except how HC-05 bytes are read and how motors are
ultimately driven:

  1. Pyserial deployment (HC-05 on a Linux-visible /dev/tty* adapter):

         HC05Controller --> MovementSafetyGate --> MovementController
                                   ^                (LoggingMovementController --
                                   |                 no real backend for this
                        voice/ipc/server_stub.py's   deployment exists yet)
                        ReferenceHumanFollowerServer
                        (reused unmodified), driven by
                        scripts/run_motion_controller.py

  2. Real UNO Q hardware deployment (the one this board's actual wiring
     needs -- HC-05 is on the MCU's own primary-serial hardware UART,
     D0/D1, not readable from Linux at all): HC-05 parsing AND all motor
     GPIO/PWM driving happen entirely in
     arduino_app/motion_controller/sketch/sketch.ino, an Arduino App run
     by arduino-app-cli. This package's role shrinks to relaying the
     voice pipeline's pause/resume signals to that sketch:

         MovementSafetyGate --> BridgeMovementController --> Bridge.call(...)
               ^                (motion/bridge_motor_controller.py)   |
               |                                                       v
         voice/ipc/server_stub.py's                          sketch.ino's
         ReferenceHumanFollowerServer                        voice_suppress()/
         (reused unmodified), driven by                      voice_release()/
         arduino_app/motion_controller/python/main.py        stop()/set_motion()

     HC05Controller (pyserial) is NOT used in this deployment -- the
     sketch parses HC-05 bytes itself, locally, at MCU speed. See
     motion/bridge_motor_controller.py's and sketch.ino's docstrings for
     the full reasoning.

Both deployments share MovementSafetyGate's "wake word always wins" logic
and MovementController's interface unchanged -- only the two ends
(HC-05 reader, motor driver) differ per deployment.

This package has zero dependency on anything under voice/ except the
already-existing, already-validated IPC config/server classes -- and
voice/ has zero dependency on this package. They communicate only through
the Unix-socket IPC protocol, exactly like the "two separate OS processes"
architecture documented in docs/ARCHITECTURE.md always intended, with this
package now standing in for what that document called "HumanFollower" (a
manual-control implementation of it, not the autonomous-following one --
see MovementSafetyGate's docstring for why that's a deliberate, easy-to-
extend design, not a placeholder to rewrite later).
"""
