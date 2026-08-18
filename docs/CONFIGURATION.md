# Configuration reference

All values live in `voice/config.py` as dataclass defaults, can be
overridden in `config/voice_config.yaml`, and can be overridden again by
environment variables named `VOICE_<SECTION>_<FIELD>` (highest priority --
useful for a systemd unit or container without editing the shipped YAML).

```bash
export VOICE_LLM_MODEL=qwen2.5:1.5b-instruct
export VOICE_VAD_MAX_SESSION_DURATION_S=25
python3 main_voice.py
```

## audio

| Field | Default | Notes |
|---|---|---|
| `input_device_index` | `null` | MUST be set per-board -- run `python -m voice.audio.list_devices`. Do not assume the old Jetson project's device index. |
| `output_device` | `"default"` | ALSA device string for `aplay`. Check `aplay -L` on the real board -- do not assume the old project's `hw:0,3` (that was a Jetson HDMI convention). |
| `sample_rate` | `16000` | Shared by wake word, VAD, and STT. |
| `frame_duration_ms` | `30` | Must be 10/20/30 for webrtcvad. |

## wake

| Field | Default | Notes |
|---|---|---|
| `enabled` | `true` | Set `false` to run STT/LLM/TTS-only for bench-testing without a wake gate. |
| `model_name` | `"hey_jarvis"` | Stock, validated keyword. See `docs/MODEL_DECISION.md` for the "hey_arduino" evaluation. |
| `model_path` | `null` | Set to a custom `.onnx` path once a validated custom model exists (`scripts/train_custom_wake_word.md`). |
| `threshold` | `0.5` | Per-frame confidence threshold, openWakeWord's native [0,1] scale. |
| `trigger_level` | `4` | Consecutive above-threshold frames required to fire -- raises this to reduce false positives at the cost of a few extra frames of latency. |

## vad

| Field | Default | Spec section |
|---|---|---|
| `no_speech_timeout_s` | `6.0` | Section 8A (5-7s recommended) |
| `max_session_duration_s` | `20.0` | Section 8B (hard ceiling, not a target) |
| `padding_duration_ms` | `400` | Hysteresis window; raise if legitimate brief pauses are getting cut off (Section 9) |
| `aggressiveness` | `2` | webrtcvad 0-3; higher cuts non-speech more aggressively |
| `post_tts_cooldown_s` | `0.5` | Grace period before wake-word re-arms after TTS playback ends |

## stt

| Field | Default | Notes |
|---|---|---|
| `binary_path` | `/opt/whisper.cpp/build/bin/whisper-cli` | |
| `model_path` | `/opt/whisper.cpp/models/ggml-base.en-q5_0.bin` | See `docs/MODEL_DECISION.md`; swap to `tiny.en` variants if benchmarking shows contention |
| `threads` | `2` | Leaves CPU headroom for the rest of the stack |
| `timeout_s` | `15.0` | Shortened from the old project's 30s |

## llm

| Field | Default | Notes |
|---|---|---|
| `model` | `"gemma3:270m"` | See `docs/MODEL_DECISION.md` -- do not use `qwen2.5:3b-instruct` on a 2GB target |
| `timeout_s` | `20.0` | Shortened hard from the old project's 120s |
| `num_predict` | `96` | Caps response length structurally |
| `keep_alive` | `"10m"` | Explicit Ollama residency window (Section 18D) |
| `system_prompt` | *(see config.py)* | Tuned for short, direct answers |

## tts

| Field | Default | Notes |
|---|---|---|
| `model_path` | `/opt/piper/voices/en_US-lessac-medium.onnx` | Downgraded from the old project's `-high` tier -- see `docs/MODEL_DECISION.md` |
| `timeout_s` | `20.0` | |

## ipc

| Field | Default | Notes |
|---|---|---|
| `socket_path` | `/tmp/humanfollower_voice.sock` | Must match whatever HumanFollower binds to |
| `pause_confirm_timeout_s` | `8.0` | How long Voice Manager waits for `PAUSE_CONFIRMED` before proceeding anyway |
| `heartbeat_interval_s` | `2.0` | |

## watchdog

| Field | Default | Notes |
|---|---|---|
| `max_paused_duration_s` | `45.0` | Absolute ceiling regardless of heartbeats |
| `heartbeat_timeout_s` | `6.0` | No heartbeat for this long => assume Voice Manager is dead |

These two values are enforced by whatever process implements
HumanFollower's side of the IPC contract -- `voice/ipc/server_stub.py` is
the reference implementation; the real HumanFollower process needs its own
copy of this config (or to import `voice.config` directly, since it's a
plain Python package with no hardware dependencies at import time).
