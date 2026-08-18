#!/usr/bin/env bash
# Installation script for a FRESH Arduino UNO Q board.
#
# Section 19 of the Phase 2 spec: the new UNO Q has NONE of this installed.
# This script does not assume otherwise -- every step checks before acting,
# and nothing here was blindly copied from the old Jetson project (which
# used apt packages tuned for a GPU-equipped desktop-Linux board, not this
# board's actual environment).
#
# What this script does NOT do:
#   - It does not install unnecessary packages "just in case."
#   - It does not download large models until you've read
#     docs/MODEL_DECISION.md and confirmed the choice (pull_models.sh is a
#     separate, explicit step).
#   - It does not touch anything related to HumanFollower.
#
# Usage:
#   bash scripts/setup_uno_q.sh
#
# Run this AS THE USER who will run the Voice Manager, not as root, except
# where sudo is explicitly needed for apt.

set -euo pipefail

echo "=== Step 1: Environment discovery (Section 19) ==="
echo "Do not assume anything about this board -- print what's actually here."
echo
echo "OS:"; cat /etc/os-release 2>/dev/null | grep -E "^(NAME|VERSION)=" || echo "  UNKNOWN -- /etc/os-release not found"
echo "Kernel: $(uname -a)"
echo "CPU architecture: $(uname -m)"
echo "CPU info:"; lscpu 2>/dev/null | grep -E "Model name|CPU\(s\)|Architecture" || echo "  lscpu not available"
echo "RAM:"; free -h 2>/dev/null || echo "  free not available"
echo "Disk:"; df -h / 2>/dev/null
echo "Python: $(python3 --version 2>&1 || echo 'NOT FOUND')"
echo
echo "^^ REVIEW THE ABOVE BEFORE CONTINUING. In particular, confirm the RAM"
echo "figure against docs/MODEL_DECISION.md's 2GB-budget assumption --"
echo "Phase 1 could not confirm whether this board is the 2GB or 4GB SKU."
echo
read -p "Press Enter to continue, or Ctrl+C to stop and investigate first... " _

echo
echo "=== Step 2: System package dependencies ==="
echo "Installing only what this project actually needs: audio (PortAudio/"
echo "ALSA), Python venv/build tooling, and whisper.cpp's build dependencies."
echo "NOT installing: SDL2, libpng, or anything GUI-related -- there is no"
echo "GUI in this project (Section 14)."

sudo apt-get update
sudo apt-get install -y \
    git curl python3-venv python3-dev python3-pip build-essential \
    cmake \
    libportaudio2 portaudio19-dev libasound2-dev alsa-utils \
    pkg-config

echo
echo "=== Step 3: Python virtual environment ==="
VENV_DIR="${VENV_DIR:-$HOME/Downloads/SocialRobot-UNOQ/.venv}"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created venv at $VENV_DIR"
else
    echo "Venv already exists at $VENV_DIR -- reusing."
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$(dirname "$0")/../requirements.txt"

echo
echo "=== Step 4: whisper.cpp (build from source) ==="
WHISPER_DIR="$HOME/Downloads/SocialRobot-UNOQ/runtime/whisper.cpp"
if [ ! -d "$WHISPER_DIR" ]; then
    sudo git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
    sudo chown -R "$(id -u):$(id -g)" "$WHISPER_DIR"
else
    echo "$WHISPER_DIR already exists -- reusing (git pull manually to update)."
fi
cmake -B "$WHISPER_DIR/build" -S "$WHISPER_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$WHISPER_DIR/build" --config Release -j"$(nproc)" --target whisper-cli
echo "whisper.cpp built: $WHISPER_DIR/build/bin/whisper-cli"

echo
echo "=== Step 5: Piper TTS (prebuilt binary) ==="
PIPER_DIR="$HOME/Downloads/SocialRobot-UNOQ/runtime/piper"
if [ ! -d "$PIPER_DIR" ]; then
    ARCH="$(uname -m)"
    case "$ARCH" in
        aarch64|arm64) PIPER_ASSET="piper_linux_aarch64.tar.gz" ;;
        x86_64)        PIPER_ASSET="piper_linux_x86_64.tar.gz" ;;
        *) echo "Unrecognized architecture '$ARCH' -- check https://github.com/rhasspy/piper/releases and download manually."; exit 1 ;;
    esac
    echo "Detected architecture $ARCH -> $PIPER_ASSET"
    # Pin a known-good release tag rather than "latest" so this script is
    # reproducible (Section 20: the final project must be reproducible).
    PIPER_TAG="2023.11.14-2"
    sudo mkdir -p "$PIPER_DIR"
    curl -fsSL -o /tmp/piper.tar.gz \
        "https://github.com/rhasspy/piper/releases/download/${PIPER_TAG}/${PIPER_ASSET}"
    sudo tar -xzf /tmp/piper.tar.gz -C "$PIPER_DIR" --strip-components=1
    sudo chown -R "$(id -u):$(id -g)" "$PIPER_DIR"
    rm /tmp/piper.tar.gz
else
    echo "$PIPER_DIR already exists -- reusing."
fi
mkdir -p "$PIPER_DIR/voices"
echo "Piper installed: $PIPER_DIR/piper"

echo
echo "=== Step 6: Ollama (local LLM runtime) ==="
if command -v ollama >/dev/null 2>&1; then
    echo "ollama already installed: $(ollama --version)"
else
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo "Starting ollama service check..."
if ! pgrep -x "ollama" >/dev/null 2>&1; then
    echo "ollama is not running. Start it with 'ollama serve' (or enable the"
    echo "systemd service if the installer set one up) before running the"
    echo "Voice Manager."
fi

echo
echo "=== Step 7: Audio device discovery (Section 10/19) ==="
echo "Do NOT assume the old Jetson project's HDMI ALSA device applies here."
echo "Listing devices now -- update config/voice_config.yaml with the"
echo "correct input_device_index / output_device before first run:"
echo
"$VENV_DIR/bin/python3" -m voice.audio.list_devices || echo "(list_devices failed -- run it manually after this script finishes)"
echo
echo "Also check ALSA-level output devices with: aplay -L"

echo
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Review docs/MODEL_DECISION.md, then run scripts/pull_models.sh"
echo "     to fetch the STT/LLM/TTS models it documents."
echo "  2. Edit config/voice_config.yaml with the audio devices discovered"
echo "     above."
echo "  3. Run scripts/benchmark_voice_pipeline.py to get REAL on-device"
echo "     RAM/CPU/latency numbers (Section 24) -- do not trust any number"
echo "     in this repo's docs that isn't explicitly marked as measured on"
echo "     this exact board."
echo "  4. Start the Voice Manager: python3 main_voice.py"
