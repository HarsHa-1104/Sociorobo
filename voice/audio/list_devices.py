"""Installation utility: enumerate audio input/output devices on this board.

Run this on the actual UNO Q during setup to find the correct
``audio.input_device_index`` and ALSA ``audio.output_device`` values for
config/voice_config.yaml -- do not assume the old Jetson project's device
indices or ALSA strings apply (Phase 1 finding: they were Jetson/HDMI
specific and will not transfer).

Usage:
    python3 -m voice.audio.list_devices
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import pyaudio
    except ImportError:
        print("pyaudio is not installed -- run scripts/setup_uno_q.sh first.")
        return 1

    pa = pyaudio.PyAudio()
    print(f"PortAudio version: {pyaudio.get_portaudio_version_text()}\n")
    print(f"{'idx':>3}  {'in':>3}  {'out':>3}  {'default_sr':>10}  name")
    print("-" * 70)
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(
            f"{i:>3}  {int(info['maxInputChannels']):>3}  "
            f"{int(info['maxOutputChannels']):>3}  "
            f"{int(info['defaultSampleRate']):>10}  {info['name']}"
        )

    try:
        default_in = pa.get_default_input_device_info()
        print(f"\nDefault input device index: {default_in['index']} ({default_in['name']})")
    except OSError:
        print("\nNo default input device found -- microphone may not be connected.")

    try:
        default_out = pa.get_default_output_device_info()
        print(f"Default output device index: {default_out['index']} ({default_out['name']})")
    except OSError:
        print("No default output device found -- speaker may not be connected.")

    pa.terminate()

    print(
        "\nSet audio.input_device_index in config/voice_config.yaml to the "
        "index of your microphone above.\n"
        "For audio.output_device, run `aplay -L` separately to see ALSA "
        "device strings (e.g. 'default', 'plughw:1,0') -- PyAudio device "
        "indices and ALSA device strings are two different namespaces; "
        "playback goes through `aplay`, not PyAudio."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
