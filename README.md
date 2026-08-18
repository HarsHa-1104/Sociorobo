# Voice Manager -- headless, local voice pipeline for HumanFollower

A lightweight, fully local/offline, headless voice-interaction subsystem
designed to run alongside a person-following robot (HumanFollower) on an
Arduino UNO Q, within a hard 2GB RAM budget shared with camera, person
detection, tracking, and motor control.

```
FOLLOWING -> wake word -> smooth stop -> listen -> STT -> local LLM -> TTS
   -> wait for playback to finish -> resume following -> wait for wake word
```

One wake word = one question = one response (Section 5 of the
implementation brief -- no multi-turn conversation in this version). No
GUI, no display, no face/avatar animation -- see `docs/ARCHITECTURE.md`
for the record of what was removed and why.

## What this is (and isn't)

This repository evolved from `OminousIndustries/SocialRobot`, an existing
Jetson Orin Nano voice-pipeline project with a GUI face/mouth animation
that Phase 1 of this project audited (see the forensic audit delivered
separately). This repo is the Phase 2 implementation: the same core
audio/STT/LLM/TTS logic, ported and hardened for a resource-constrained,
headless, multi-process robot integration -- **not** a rewrite from
scratch (Section 27: change only what's necessary).

It does **not** include HumanFollower's source (motor control, camera,
person detection) -- that wasn't available to integrate against directly.
See `humanfollower_integration/README.md` for the IPC contract
HumanFollower needs to implement, and `voice/ipc/server_stub.py` for a
runnable reference implementation of that contract used for testing this
repo standalone.

## Quick start

```bash
bash scripts/setup_uno_q.sh      # install everything, on the real UNO Q
bash scripts/pull_models.sh      # fetch STT/LLM/TTS model candidates
# edit config/voice_config.yaml with your board's audio devices
python3 scripts/benchmark_voice_pipeline.py --simulate-load  # real on-device numbers
python3 scripts/run_reference_humanfollower.py &  # terminal 1, or your real HumanFollower once integrated
python3 main_voice.py                              # terminal 2
```

Full walkthrough: `docs/INSTALL.md`.

## Documentation map

| Doc | Covers |
|---|---|
| `docs/ARCHITECTURE.md` | Final architecture, state machine, IPC, failure handling, GUI removal record |
| `docs/MODEL_DECISION.md` | Evidence-based reasoning for every wake-word/STT/LLM/TTS choice |
| `docs/INSTALL.md` | Step-by-step fresh-board installation |
| `docs/CONFIGURATION.md` | Every configurable value, its default, and why |
| `docs/TESTING.md` | Automated test suite + real-device validation checklist |
| `docs/TROUBLESHOOTING.md` | Common problems and fixes |
| `humanfollower_integration/README.md` | What HumanFollower's maintainer needs to implement |
| `scripts/train_custom_wake_word.md` | The "hey_arduino" custom wake-word path, documented but not executed |

## Project structure

```
voice/
  config.py            central configuration (defaults <- YAML <- env vars)
  audio/
    manager.py          single shared microphone stream (AudioManager)
    vad.py               VAD + two-tier session timeout logic (SpeechSegmenter)
    list_devices.py      installation utility: enumerate audio devices
  wake/
    wake_word.py          openWakeWord wrapper (WakeWordDetector)
  stt/
    whisper_cpp.py        whisper.cpp subprocess wrapper
  llm/
    ollama_client.py       local Ollama client, single-turn/stateless
  tts/
    piper_tts.py            Piper subprocess wrapper, deterministic playback completion
  manager/
    state_machine.py        the enforced voice-session state graph
    voice_manager.py         the orchestrator tying everything together
  ipc/
    protocol.py               wire format (5 message types, JSON)
    client.py                  Voice Manager's side (HumanFollowerLink)
    server_stub.py              reference HumanFollower-side server + watchdog
humanfollower_integration/
  README.md              integration checklist for HumanFollower's maintainer
config/
  voice_config.yaml      default configuration
scripts/
  setup_uno_q.sh          fresh-board installation
  pull_models.sh           model fetching
  benchmark_voice_pipeline.py   real on-device RAM/CPU/latency measurement
  run_reference_humanfollower.py   standalone test double for HumanFollower
  train_custom_wake_word.md   documented (not executed) custom wake-word procedure
tests/                    58 automated tests (pytest), see docs/TESTING.md
main_voice.py             Voice Manager entrypoint (run as its own OS process)
```

## Running the tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. python3 -m pytest tests/ -v
```
