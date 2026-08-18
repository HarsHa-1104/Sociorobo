# Troubleshooting

**"Voice Manager exits immediately with a `FileNotFoundError`"**
A binary or model path in `config/voice_config.yaml` doesn't exist on this
board yet. Run `scripts/setup_uno_q.sh` (binaries) and
`scripts/pull_models.sh` (models) if you haven't. This is intentional
fail-fast behavior (Section 17) -- it will not silently start in a broken
state.

**"MicrophoneUnavailableError at startup"**
`audio.input_device_index` in `config/voice_config.yaml` is wrong or the
mic isn't connected. Run `python -m voice.audio.list_devices` and update
the config.

**"Nothing happens when I say the wake word"**
- Confirm `wake.enabled: true` in config.
- Confirm the mic index is correct (`voice.audio.list_devices`).
- Try lowering `wake.threshold` or `wake.trigger_level` slightly -- but
  re-run the false-accept test in `scripts/train_custom_wake_word.md`'s
  validation gate section after any change, since lowering these raises
  false-positive risk.
- Check logs at `INFO` level (`logging.level: "INFO"`) for
  `"Wake word detected."` -- if it's not printing, the detector isn't
  seeing speech-like input at all (mic issue), not a threshold issue.

**"It wakes up when nobody said anything" (false wake)**
- Raise `wake.threshold` and/or `wake.trigger_level`.
- Confirm the wake-word listener is actually suspended during TTS
  playback (it should be, structurally -- see
  `docs/ARCHITECTURE.md`'s "Audio architecture" section). If this is
  happening right after the robot finishes speaking, first check
  `vad.post_tts_cooldown_s` isn't too short for your speaker/mic
  placement.

**"It cuts me off mid-sentence"**
Raise `vad.padding_duration_ms` (more hysteresis before declaring
end-of-speech) and/or lower `vad.deactivation_ratio` slightly. See
`docs/CONFIGURATION.md`.

**"It waits a long time after I stop talking before answering"**
The opposite problem -- lower `vad.padding_duration_ms` and/or raise
`vad.deactivation_ratio`. There's a real tradeoff here against the
"don't cut off mid-sentence" case above; tune against real speech, not
just one test utterance.

**"Ollama connection error" in the logs**
`ollama serve` (or the systemd service) isn't running, or
`llm.url`/port doesn't match. `curl http://localhost:11434/api/tags` to
check the daemon is up.

**"LLM responses take a long time" / "robot stands still for a while after asking"**
- Check `scripts/benchmark_voice_pipeline.py`'s output for this model's
  measured latency on this board.
- Consider `gemma3:270m` over `qwen2.5:1.5b-instruct` if the latency
  difference matters more than response quality for your use case --
  see `docs/MODEL_DECISION.md`.
- Confirm `llm.keep_alive` is set (avoids a cold model-load on every
  question) -- default is `"10m"`.

**"aplay: command not found" / TTS produces no sound**
`alsa-utils` isn't installed (`scripts/setup_uno_q.sh` installs it) or
`audio.output_device` in config doesn't match a real ALSA sink. Run
`aplay -L` on the board to see valid device strings.

**"ALSA device busy" errors**
This is exactly the class of bug the single-persistent-stream
`AudioManager` design (see `docs/ARCHITECTURE.md`) exists to prevent for
the *microphone* side. If you're seeing this on the *speaker* side,
confirm nothing else (another process, a leftover `aplay` from a crashed
previous run) is holding the output device open.

**"Voice Manager process died and the robot just... stopped forever"**
This should not happen if HumanFollower implements the documented
watchdog (`voice/ipc/server_stub.py::_watchdog_loop`,
`humanfollower_integration/README.md`). If it's happening, HumanFollower's
side of the integration is either not implemented yet or not enforcing
`watchdog.max_paused_duration_s`/`watchdog.heartbeat_timeout_s`. This is
the single most safety-critical piece of the whole integration -- verify
it explicitly with `scripts/run_reference_humanfollower.py` (kill
`main_voice.py` mid-session with `kill -9` and confirm the reference
server's log shows `WATCHDOG TRIPPED`) before trusting the real
integration.

**"pytest fails on my machine but not in the sandbox"**
Check `webrtcvad`/`pyaudio` built correctly for your Python version/OS --
these are C extensions. The test suite itself doesn't need real
`webrtcvad`/`pyaudio`/`openwakeword` installed (everything hardware-facing
is mocked, see `docs/TESTING.md`), so a failure to even *collect* the
tests usually means an import-time issue in `voice/`, not a test logic
issue -- run `python3 -c "import voice.manager.voice_manager"` to isolate
it.
