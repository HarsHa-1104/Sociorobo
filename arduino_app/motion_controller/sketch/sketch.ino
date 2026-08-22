// Motion Controller sketch: HC-05 Bluetooth manual driving for a
// four-wheel mecanum car, coordinated with the voice pipeline's
// wake-word safety stop via the Router Bridge.
//
// ARCHITECTURE (see ../../../motion/__init__.py and motion/bridge_motor_controller.py
// for the Python side):
//
//   HC-05 (Serial, D0/D1) --byte--> parseCommand() --> setMotion()/stopAllMotors()
//                                                              |
//                                          voiceSuppressed gate (local, this file)
//                                                              |
//                                                     the 4 wheels (GPIO/PWM)
//
//   Python (voice pipeline, via IPC -> this App's python/main.py ->
//   Bridge.call) ------------------------------------------------> voiceSuppress()/
//                                                                   voiceRelease()/
//                                                                   stop()
//
// WHY suppression lives here, not only in Python: HC-05 is wired to this
// board's PRIMARY serial pins (D0/D1) -- the MCU's own hardware UART, not
// something Linux can open as a /dev/tty* device. Bluetooth bytes are
// parsed and acted on entirely in THIS sketch's loop(); Python never sees
// them. So Python's MovementSafetyGate (voice/pipeline side) cannot
// suppress manual commands by refusing to forward them -- it has nothing
// to forward. Instead, Python tells THIS sketch (via voice_suppress/
// voice_release, below) to hold off, and this sketch enforces that
// locally, at MCU speed, with no further round-trip needed per suppressed
// command.
//
// PIN MAPPING -- confirmed, not guessed (do not reuse pin numbers from
// any older/different sketch):
//
//   Wheel        PWM/EN   DIR1   DIR2
//   Front Left   D9       D12    D13
//   Back Left    D6       D10    D11
//   Front Right  D3       D2     D4
//   Back Right   D5       D7     D8
//
// CALIBRATION -- per-wheel PWM offsets, applied AFTER the base speed, all
// default to 0 (i.e. "trust the wiring/motors are matched until measured
// otherwise"). Do NOT assume the old sketch's calibration numbers apply
// to this wiring -- these must be re-measured on this exact car. See the
// physical validation section of the motion-controller integration report
// for the calibration procedure.
//
// HC-05 COMMAND PROTOCOL -- STATUS: ASSUMED, NOT VERIFIED. F/B/L/R for
// cardinal directions, G/H/I/J for the four diagonals
// (forward_right/backward_right/backward_left/forward_left), S for an
// immediate stop, and a single digit '0'-'9' for a 10-level speed setting
// -- a real, documented convention used by several widely-forked
// open-source "Bluetooth RC Car" Arduino/Android app pairs, but NOT
// confirmed against whichever specific app you actually use. See the
// integration report's HC-05 command-capture procedure before relying on
// this for real driving; COMMAND_MAP below is the ONE place to change it
// if your app sends something different.
//
// TWO DISTINCT STOP BEHAVIORS (do not conflate them):
//   * stop() / voiceSuppress() / an 'S' byte from HC-05: IMMEDIATE hard
//     stop -- both direction pins LOW, PWM 0, no ramp, applied to all
//     four wheels unconditionally, right away.
//   * Manual release (the app stops sending movement bytes / the
//     joystick returns to center): GRADUAL deceleration, ramping PWM
//     toward zero over RAMP_STEP_PWM/RAMP_INTERVAL_MS -- see loop()'s
//     idle-timeout ramp logic. Configurable, and deliberately NEVER used
//     for the wake-word path -- the two must stay independent so a
//     future change to one (e.g. a slower ramp for a heavier car) can
//     never accidentally slow down the safety stop.

#include "Arduino_RouterBridge.h"

// ---------------------------------------------------------------------
// Pin mapping (confirmed wiring -- see header comment).
// ---------------------------------------------------------------------
struct Wheel {
  const char *name;
  uint8_t pwmPin;
  uint8_t dir1Pin;
  uint8_t dir2Pin;
  int calibrationOffset;  // added to the commanded PWM magnitude before
                          // constraining to [0, 255] -- may be negative.
                          // ALL DEFAULT TO 0 -- see header comment.
};

enum WheelIndex { FL = 0, BL = 1, FR = 2, BR = 3 };

Wheel wheels[4] = {
  /* FL */ { "FL", 9,  12, 13, 0 },
  /* BL */ { "BL", 6,  10, 11, 0 },
  /* FR */ { "FR", 3,  2,  4,  0 },
  /* BR */ { "BR", 5,  7,  8,  0 },
};

const int PWM_MIN = 0;
const int PWM_MAX = 255;

// ---------------------------------------------------------------------
// Mecanum kinematics: per-direction signed multiplier (-1/0/+1) for each
// of the four wheels, derived from the standard mecanum drive equations
// FL=vy+vx+w, FR=vy-vx-w, BL=vy-vx+w, BR=vy+vx-w (vy=forward/back,
// vx=strafe right/left, w=rotate clockwise/counter-clockwise), then
// normalised to -1/0/+1. "left"/"right" are IN-PLACE ROTATION (matching
// what a plain Left/Right button on a basic RC car app is expected to
// do), NOT sideways strafing -- the four diagonal commands are true
// mecanum diagonal moves (exactly two opposite-corner wheels active).
// This is a deliberate, documented, EASILY ADJUSTABLE design choice --
// change only the table below (order: FL, BL, FR, BR) to redefine what
// any direction means; nothing else in this sketch, the HC-05 parsing,
// or the voice/Bridge integration needs to change.
// ---------------------------------------------------------------------
struct DirectionSigns { int fl, bl, fr, br; };

DirectionSigns signsFor(const String &direction) {
  if (direction == "forward")        return { 1,  1,  1,  1 };
  if (direction == "backward")       return {-1, -1, -1, -1 };
  if (direction == "left")           return {-1, -1,  1,  1 };  // rotate counter-clockwise
  if (direction == "right")          return { 1,  1, -1, -1 };  // rotate clockwise
  if (direction == "forward_left")   return { 0,  1,  1,  0 };
  if (direction == "forward_right")  return { 1,  0,  0,  1 };
  if (direction == "backward_left")  return {-1,  0,  0, -1 };
  if (direction == "backward_right") return { 0, -1, -1,  0 };
  return { 0, 0, 0, 0 };  // unrecognised direction -- safe default: no motion
}

// Tracks the ACTUAL currently-applied signed PWM per wheel (post-
// calibration), indexed the same way as `wheels[]` (FL, BL, FR, BR) --
// separate from the caller's requested speed. This is what the
// idle-timeout ramp steps toward zero, and what a NEW command
// immediately overrides (see loop()). Declared before applyWheel() below,
// which writes to it, since this is C++ and there's no hoisting.
int currentWheelSignedPWM[4] = { 0, 0, 0, 0 };

// ---------------------------------------------------------------------
// The one master motor-setting function every code path (HC-05 parsing,
// Bridge-driven set_motion, stop, the idle-timeout ramp) goes through --
// mirrors the reference sketch's "all motor commands flow through one
// master function" style. Takes an explicit wheel index (matching
// WheelIndex/wheels[]) rather than deriving it from a reference, to keep
// the indexing simple and unambiguous at every call site.
// ---------------------------------------------------------------------
void applyWheel(int wheelIndex, int signedSpeed) {
  Wheel &wheel = wheels[wheelIndex];
  int magnitude = abs(signedSpeed);
  int pwm = constrain(magnitude + wheel.calibrationOffset, PWM_MIN, PWM_MAX);

  if (signedSpeed > 0) {
    digitalWrite(wheel.dir1Pin, HIGH);
    digitalWrite(wheel.dir2Pin, LOW);
  } else if (signedSpeed < 0) {
    digitalWrite(wheel.dir1Pin, LOW);
    digitalWrite(wheel.dir2Pin, HIGH);
  } else {
    // Stop means both direction pins LOW and PWM 0 -- not just PWM 0
    // with a direction pin left driving, which some motor drivers would
    // interpret as active braking or a half-driven state.
    digitalWrite(wheel.dir1Pin, LOW);
    digitalWrite(wheel.dir2Pin, LOW);
    pwm = 0;
  }

  analogWrite(wheel.pwmPin, pwm);
  currentWheelSignedPWM[wheelIndex] = (signedSpeed == 0) ? 0 : (signedSpeed > 0 ? pwm : -pwm);
}

void setMotion(const String &direction, int speed) {
  speed = constrain(speed, PWM_MIN, PWM_MAX);
  DirectionSigns s = signsFor(direction);
  applyWheel(FL, s.fl * speed);
  applyWheel(BL, s.bl * speed);
  applyWheel(FR, s.fr * speed);
  applyWheel(BR, s.br * speed);
}

// IMMEDIATE hard stop -- all four wheels, no ramp. Used by the wake-word
// safety path (voiceSuppress()/stop()) and by an explicit 'S' byte from
// HC-05. Always safe to call even if nothing is moving.
void stopAllMotors() {
  for (int i = 0; i < 4; i++) {
    applyWheel(i, 0);
  }
}

// ---------------------------------------------------------------------
// Voice-interaction suppression (local to this sketch -- see header
// comment for why this can't live only in Python for this board's
// wiring).
// ---------------------------------------------------------------------
bool voiceSuppressed = false;

void voiceSuppress() {
  voiceSuppressed = true;
  stopAllMotors();  // immediate -- see header comment on the two stop behaviors
  Monitor.println("voice_suppress: motors stopped, manual commands suppressed.");
}

void voiceRelease() {
  voiceSuppressed = false;
  // Deliberately does NOT call setMotion() or otherwise move the car --
  // "do not automatically resume previous movement" applies here exactly
  // as it does in MovementSafetyGate.release() on the Python side. A new
  // HC-05 byte (or a new Bridge set_motion call) is required to move.
  Monitor.println("voice_release: manual control available again (car remains stopped).");
}

// Exposed to Python for the abstract MovementController.stop() -- same
// immediate behavior as voiceSuppress()'s stop, but does NOT touch
// voiceSuppressed (BridgeMovementController.stop() and .suppress() are
// separate calls on the Python side; MovementSafetyGate.suppress_and_stop()
// always calls both, in that order, so in practice this is always
// preceded by voice_suppress -- see motion/safety_gate.py).
void bridgeStop() {
  stopAllMotors();
}

// Exposed to Python for any FUTURE Python-driven movement (e.g.
// autonomous following, explicitly not implemented in this phase) --
// deliberately does NOT check voiceSuppressed: that flag exists to gate
// the LOCAL/manual (HC-05) command source specifically. A Bridge-issued
// set_motion call only ever happens because Python's own
// MovementSafetyGate already decided it's allowed (see
// motion/safety_gate.py's request_*() gating) -- checking voiceSuppressed
// here too would be redundant, not incorrect, but the single source of
// truth for "is a Bridge-issued command allowed" is Python's gate, not
// this flag.
void bridgeSetMotion(String direction, int speed) {
  setMotion(direction, speed);
}

// ---------------------------------------------------------------------
// HC-05 manual driving: command parsing (Serial = D0/D1, hardware UART).
// STATUS: command bytes are ASSUMED, not verified -- see header comment.
// ---------------------------------------------------------------------
const long HC05_BAUD = 9600;  // HC-05 factory default -- confirmed hardware fact, not a guess.
                                // Change ONLY if the module was reconfigured via AT commands.

int currentManualSpeed = 200;  // matches motion/movement_controller.py's DEFAULT_SPEED,
                                 // for consistency between the two (independent) parsers.
unsigned long lastManualCommandMillis = 0;

// 10-level speed convention: digit '0'..'9' -> 0..255. Matches
// motion/hc05_controller.py's DEFAULT_SPEED_MAP for documentation
// consistency -- the two parsers are independent code (Python's is not
// used for the real board's wiring, see header comment) but describe the
// SAME assumed protocol, so keeping them in sync avoids confusing
// documentation drift if the real protocol is later confirmed and this
// table is updated.
int speedForDigit(char c) {
  int level = c - '0';  // '0'-'9' -> 0-9
  return (level * PWM_MAX) / 9;
}

void handleHC05Byte(char c) {
  c = toupper(c);

  if (c >= '0' && c <= '9') {
    currentManualSpeed = speedForDigit(c);
    return;  // setting speed never itself moves the car
  }

  String direction;
  bool isStop = false;
  switch (c) {
    case 'F': direction = "forward"; break;
    case 'B': direction = "backward"; break;
    case 'L': direction = "left"; break;
    case 'R': direction = "right"; break;
    case 'G': direction = "forward_right"; break;
    case 'H': direction = "backward_right"; break;
    case 'I': direction = "backward_left"; break;
    case 'J': direction = "forward_left"; break;
    case 'S': isStop = true; break;
    default:
      // Unrecognised byte (including whitespace/newline/protocol noise)
      // -- ignore silently. No unsafe fallback.
      return;
  }

  lastManualCommandMillis = millis();

  if (voiceSuppressed) {
    // HC-05 stays connected and this sketch keeps reading/parsing bytes
    // throughout voice interaction -- "logically active" -- but a
    // suppressed command must never reach the physical motors.
    return;
  }

  if (isStop) {
    stopAllMotors();  // an explicit Stop button is also an immediate hard stop, not a ramp
  } else {
    setMotion(direction, currentManualSpeed);
  }
}

// ---------------------------------------------------------------------
// Manual-release gradual deceleration: only when NOT voice-suppressed
// (suppression's stop is always immediate, handled separately above) and
// no new HC-05 byte has arrived within MANUAL_IDLE_TIMEOUT_MS -- ramps
// every wheel's currently-applied PWM toward zero in fixed steps. A new
// command arriving resets lastManualCommandMillis and overwrites the
// wheel's PWM directly via setMotion()/applyWheel(), so the ramp never
// interferes with (or delays) a new command -- it only ever acts during
// genuine idle time.
// ---------------------------------------------------------------------
const unsigned long MANUAL_IDLE_TIMEOUT_MS = 300;  // configurable: how long without a new
                                                     // command before deceleration begins
const int RAMP_STEP_PWM = 15;                       // configurable: PWM reduced by this much per step
const unsigned long RAMP_INTERVAL_MS = 20;           // configurable: time between ramp steps
unsigned long lastRampStepMillis = 0;

void rampTowardZeroIfDue() {
  if (millis() - lastManualCommandMillis < MANUAL_IDLE_TIMEOUT_MS) {
    return;  // still within the "recently commanded" window -- not idle yet
  }
  if (millis() - lastRampStepMillis < RAMP_INTERVAL_MS) {
    return;  // not time for the next ramp step yet
  }
  lastRampStepMillis = millis();

  bool anyMoving = false;
  for (int i = 0; i < 4; i++) {
    int current = currentWheelSignedPWM[i];
    if (current == 0) continue;
    anyMoving = true;
    int step = (current > 0) ? -RAMP_STEP_PWM : RAMP_STEP_PWM;
    int next = current + step;
    // Don't overshoot past zero.
    if ((current > 0 && next < 0) || (current < 0 && next > 0)) {
      next = 0;
    }
    applyWheel(i, next);
  }
  if (anyMoving) {
    Monitor.println("Manual-release deceleration: ramping toward stop.");
  }
}

// ---------------------------------------------------------------------
void setup() {
  Monitor.begin(115200);

  for (int i = 0; i < 4; i++) {
    pinMode(wheels[i].pwmPin, OUTPUT);
    pinMode(wheels[i].dir1Pin, OUTPUT);
    pinMode(wheels[i].dir2Pin, OUTPUT);
  }
  // SAFETY: motors default to stopped at startup -- before anything else
  // runs, including Bridge/HC-05 initialization.
  stopAllMotors();

  Serial.begin(HC05_BAUD);  // HC-05 on the primary serial pins (D0/D1) -- see header comment
                              // on why this must not also be used for Serial.print() debugging;
                              // use Monitor (a separate channel) for sketch-side logging instead.

  Bridge.begin();
  Bridge.provide("set_motion", bridgeSetMotion);
  Bridge.provide("stop", bridgeStop);
  Bridge.provide("voice_suppress", voiceSuppress);
  Bridge.provide("voice_release", voiceRelease);

  lastManualCommandMillis = millis();
  Monitor.println("Motion Controller sketch ready: motors stopped, HC-05 listening, Bridge armed.");
}

void loop() {
  while (Serial.available() > 0) {
    handleHC05Byte((char)Serial.read());
  }

  if (!voiceSuppressed) {
    rampTowardZeroIfDue();
  }
}
