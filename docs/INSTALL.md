# Installation (fresh UNO Q)

Assumes nothing is installed yet (Section 19) -- this is the actual order
to run things in.

## 1. Environment discovery + system setup

```bash
bash scripts/setup_uno_q.sh
```

This prints the board's actual OS/kernel/CPU/RAM/disk before doing
anything else (verify the RAM figure against `docs/MODEL_DECISION.md`'s
2GB-budget assumption -- Phase 1 could not confirm the exact SKU), then
installs only what's needed: PortAudio/ALSA dev libraries, build tooling,
a Python venv with `requirements.txt`, whisper.cpp (built from source),
Piper (prebuilt binary for the detected architecture), and Ollama.

At the end it runs `python -m voice.audio.list_devices` -- note the
microphone index and speaker ALSA device it reports; you'll need them in
step 3.

## 2. Pull models

```bash
bash scripts/pull_models.sh
```

Fetches the STT/LLM/TTS candidates documented in `docs/MODEL_DECISION.md`
-- multiple quantizations/tiers of each, specifically so step 4's
benchmark can compare them on this exact board rather than guessing.
**Deliberately does not pull `qwen2.5:3b-instruct`** -- see
`docs/MODEL_DECISION.md` for why.

## 3. Configure

Edit `config/voice_config.yaml`:

```yaml
audio:
  input_device_index: <from step 1's device list>
  output_device: "<ALSA device string, check with `aplay -L`>"
```

Every other value has a documented default (see `docs/CONFIGURATION.md`)
-- nothing else needs to change to get a first run working.

## 4. Benchmark on THIS board

```bash
python3 scripts/benchmark_voice_pipeline.py --simulate-load --json bench_results.json
```

This produces the real RAM/CPU/latency numbers this repo's documentation
cannot -- it was written in a cloud sandbox with no ARM CPU and no access
to the real model registries (see `docs/MODEL_DECISION.md`'s network-access
note). Use the output to pick final `stt.model_path`, `llm.model`, and
`tts.model_path` values in `config/voice_config.yaml` if they should
differ from the shipped defaults.

## 5. Test the Voice Manager standalone (no real HumanFollower needed yet)

Terminal 1:
```bash
python3 scripts/run_reference_humanfollower.py
```

Terminal 2:
```bash
python3 main_voice.py
```

Say "hey jarvis" (the default stock wake word -- see
`docs/MODEL_DECISION.md` on why `hey_arduino` isn't shipped yet), then ask
a short question. You should see the simulated "decelerating/stopped"
message in terminal 1, then the transcript/LLM reply/TTS playback in
terminal 2, then "resuming following" back in terminal 1.

## 6. Integrate with the real HumanFollower

See `humanfollower_integration/README.md` -- this is not automatic,
because the real HumanFollower source was not available to integrate
directly (see `docs/ARCHITECTURE.md`).

## 7. Run for real

```bash
# Terminal 1: HumanFollower, once it implements the IPC contract
python3 -m human_follower.main   # (illustrative -- actual entrypoint is HumanFollower's own)

# Terminal 2: Voice Manager
python3 main_voice.py
```

Consider running both under `systemd` units (or your process supervisor of
choice) with `Restart=on-failure` so a Voice Manager crash comes back on
its own -- not required for safety (HumanFollower's watchdog already
guarantees the robot doesn't get stuck, see `docs/ARCHITECTURE.md`), but
it's an operational nicety for the wake-word listener resuming without a
human noticing it went down.
