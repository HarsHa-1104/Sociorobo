# HC-05 command discovery: finding out what your Bluetooth RC app actually sends

**Status of the default mapping in this repository: ASSUMED, NOT VERIFIED.**
`sketch.ino`'s `handleHC05Byte()` and `motion/hc05_controller.py`'s
`DEFAULT_COMMAND_MAP`/`DEFAULT_SPEED_MAP` both implement a common,
widely-forked "Bluetooth RC Car" convention (single ASCII bytes `F`/`B`/
`L`/`R` for cardinal moves, `G`/`H`/`I`/`J` for the four diagonals, `S`
for stop, `0`-`9` for a 10-level speed slider) -- but this has NOT been
confirmed against the specific mobile app you're using. **Do not drive
the real car under the assumption that this mapping is correct until
you've completed the capture procedure below at least once.** If it turns
out wrong, only `handleHC05Byte()` in `sketch.ino` (and, if you also use
the pyserial deployment, `DEFAULT_COMMAND_MAP`/`DEFAULT_SPEED_MAP` in
`motion/hc05_controller.py`) need to change -- nothing else in the
architecture depends on the specific byte values.

## Why this can't be verified in advance

The exact bytes a "Bluetooth RC Controller" app sends are a property of
that specific app, not of the HC-05 module or this codebase -- different
apps in this genre use different byte sets (some use `F`/`B`/`L`/`R`,
some use `1`-`8` for all eight directions plus a separate stop byte, some
send multi-byte strings like `"FWD\n"`). There is no way to determine
this without capturing real traffic from the app you intend to use.

## Capture procedure (safe -- no motors need to be connected/powered)

This procedure reads and logs raw bytes from the HC-05 module without
ever driving a motor, so it's safe to run with the L298N drivers
unpowered or even fully disconnected.

### Step 1 -- isolate the HC-05 module for capture

Wire the HC-05 module to the UNO Q's **primary serial pins exactly as
specified** (TX -> UNO Q RX/D0, RX -> UNO Q TX/D1, VCC -> 5V, GND -> GND).
Leave the L298N motor drivers **disconnected or unpowered** for this step
-- this step only needs the HC-05 link, not the motors.

### Step 2 -- flash a byte-logging sketch (temporary, capture-only)

Do **not** use the real `motion_controller` App for this step -- its
`handleHC05Byte()` already assumes a mapping you haven't verified yet.
Instead, create a small, separate, temporary capture App:

```bash
arduino-app-cli app new hc05_capture -d "Temporary HC-05 byte capture/logging tool" -i "🔍"
```

Replace its `sketch/sketch.ino` with:

```cpp
#include "Arduino_RouterBridge.h"

void setup() {
  Monitor.begin(115200);
  Serial.begin(9600);  // HC-05 factory default baud -- change only if you've
                        // reconfigured the module via AT commands
  Bridge.begin();
  Monitor.println("HC-05 capture ready. Press buttons/move the slider in your app now.");
}

void loop() {
  while (Serial.available() > 0) {
    int b = Serial.read();
    Monitor.print("byte=");
    Monitor.print(b);            // decimal value
    Monitor.print(" ('");
    if (b >= 32 && b < 127) {
      Monitor.print((char)b);    // printable ASCII, if it is one
    } else {
      Monitor.print("?");
    }
    Monitor.println("')");
  }
}
```

This logs every raw byte the HC-05 module receives over Bluetooth,
decimal value and printable character both, with no assumptions about
what it means.

### Step 3 -- run it and capture real button presses

```bash
arduino-app-cli app start ~/ArduinoApps/hc05_capture
arduino-app-cli monitor
```

With `monitor` running, pair your phone to the HC-05 module in the app
and, **one at a time**, press: Up, Down, Left, Right, each diagonal (if
the app exposes them directly), Stop/Release, and move the speed slider
through a few positions. Note which logged byte(s) correspond to which
button press -- write them down as you go (e.g. "Up -> byte=70 ('F')").
Also note whether a button sends **once** (on press) or **repeatedly**
(streamed while held) -- this affects nothing in the current design
(both work fine with the existing idle-timeout ramp) but is useful to
know.

### Step 4 -- update the real mapping if it differs

If the captured bytes match this repository's defaults (`F`/`B`/`L`/`R`/
`G`/`H`/`I`/`J`/`S`/digits), no change is needed -- the assumption was
correct for your app. Otherwise, edit the `switch` in
`arduino_app/motion_controller/sketch/sketch.ino`'s `handleHC05Byte()`
(and, if you also plan to use the pyserial-based deployment, the
`DEFAULT_COMMAND_MAP`/`DEFAULT_SPEED_MAP` dicts in
`motion/hc05_controller.py`, or pass a custom `command_map`/`speed_map`
to `HC05Controller(...)` instead of editing the defaults) to match what
you actually captured. If your app sends multi-byte strings rather than
single bytes, `handleHC05Byte()`'s single-character `switch` will need to
become a small line-buffering parser instead -- ask for that change
explicitly if the capture reveals this, since it's a different (still
small) design, not a one-line edit.

### Step 5 -- clean up

```bash
arduino-app-cli app stop ~/ArduinoApps/hc05_capture
```

The temporary `hc05_capture` App can be left in place (it drives no
motors and does nothing unless started) or removed with
`arduino-app-cli app destroy ~/ArduinoApps/hc05_capture` once you're
confident in the real mapping -- your call.

## After this procedure

Once the mapping is confirmed (or corrected) this way, physical driving
tests can proceed per the motion-controller integration report's physical
validation section -- do not skip straight to full-speed driving tests
before completing this capture at least once.
