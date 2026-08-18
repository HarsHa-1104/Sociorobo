#!/usr/bin/env python3
"""Voice Manager entrypoint -- run this as its own OS process, separate
from HumanFollower (Section 15: separate processes, IPC only, no shared
motor authority).

    python3 main_voice.py [--config path/to/voice_config.yaml]

This process:
  1. Loads configuration (defaults <- YAML <- environment variables).
  2. Builds the real audio/wake/VAD/STT/LLM/TTS collaborators.
  3. Runs the wake -> pause -> listen -> STT -> LLM -> TTS -> resume loop
     forever, exiting cleanly on SIGINT/SIGTERM.

It does NOT start a GUI, open a display, or import pygame -- there is no
GUI code left in this project at all (see docs/ARCHITECTURE.md "GUI
removal").

It does NOT implement HumanFollower's side of the IPC contract -- that is
either the real HumanFollower process (see docs/ARCHITECTURE.md
"HumanFollower integration") or, for local testing, the reference server
in voice/ipc/server_stub.py (see scripts/run_reference_humanfollower.py).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from voice.config import load_config
from voice.manager.voice_manager import build_voice_manager


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="HumanFollower Voice Manager")
    parser.add_argument("--config", type=str, default=None, help="Path to voice_config.yaml")
    args = parser.parse_args()

    # Config is loaded before logging is configured so --config/env can
    # control the log level too.
    config = load_config(args.config)
    _setup_logging(config.logging.level)
    logger = logging.getLogger("main_voice")

    logger.info("Building Voice Manager (wake=%s stt_model=%s llm_model=%s tts_model=%s)",
                config.wake.model_name,
                Path(config.stt.model_path).name,
                config.llm.model,
                Path(config.tts.model_path).name)

    try:
        manager = build_voice_manager(config)
    except FileNotFoundError as exc:
        logger.error("Missing required binary/model, cannot start: %s", exc)
        logger.error("Run scripts/setup_uno_q.sh and scripts/pull_models.sh first.")
        return 1
    except RuntimeError as exc:
        logger.error("Cannot start Voice Manager: %s", exc)
        return 1

    def _handle_signal(signum, _frame):
        logger.info("Received signal %d, shutting down...", signum)
        manager.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        manager.run_forever()
    except Exception:
        logger.exception("Voice Manager crashed")
        return 1

    logger.info("Voice Manager stopped cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
