# Testing

## What's already automated (58 tests, run in this implementation session)

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. python3 -m pytest tests/ -v
```

| File | Covers |
|---|---|
| `test_config.py` | defaults / YAML / env-var override precedence |
| `test_state_machine.py` | every legal transition, several illegal ones rejected |
| `test_vad_segmenter.py` | Section 8's two-tier timeout logic: no-speech timeout fires only before speech starts; max-duration ceiling fires mid-speech with partial audio; brief mid-sentence pauses don't end a segment early; end-of-speech never waits out the remaining time |
| `test_wake_word.py` | debounce/trigger_level logic; the real bug this integration testing found (loading openWakeWord's entire stock set instead of one model) is now a regression test |
| `test_stt_wrapper.py`, `test_tts_wrapper.py`, `test_llm_client.py` | subprocess/HTTP mocked at the boundary; missing-binary errors, timeouts, malformed output all covered |
| `test_ipc_protocol.py` | wire format round-trip, partial-buffer handling, malformed-message resilience |
| `test_ipc_client_server.py` | **real Unix socket** between `HumanFollowerLink` and `ReferenceHumanFollowerServer` -- no mocks on the wire; includes the watchdog actually force-resuming on a missing heartbeat and on an absolute-duration ceiling |
| `test_voice_manager.py` | full session orchestration with fakes for every hardware-facing collaborator: happy path, no-speech timeout, STT/LLM/TTS failure paths, max-duration partial capture, and HumanFollower being completely unreachable -- every path ends in `VOICE_SESSION_COMPLETE` and audio resumed |

This suite runs anywhere (no audio hardware, no real model binaries, no
network) and should stay green through any future change -- it's the
fast feedback loop. It does **not** prove the system works on the UNO Q;
it proves the logic is correct given whatever the hardware-facing edges
report.

## Real integration validation already performed (this session, x86_64 sandbox)

Documented in full in `docs/MODEL_DECISION.md`. Summary: whisper.cpp built
from source and run end-to-end through the real `WhisperCppSTT` wrapper;
Piper's real binary run end-to-end through `PiperTTS`, synthesizing real
English speech; openWakeWord's real, production `hey_jarvis` model loaded
and run through `WakeWordDetector`, including a live false-positive check
(0/300 on random noise) that also caught and fixed a real bug. None of
this used ARM hardware or the real target LLM/STT/TTS weights (network
access to huggingface.co/ollama.com was blocked in that sandbox) -- treat
timing numbers from that session as "the code path works," not as UNO Q
performance data.

## What still needs to happen on the real board (Section 23)

Test independently first:
1. Wake word -- say it repeatedly in the actual deployment environment; check `scripts/benchmark_voice_pipeline.py`'s false-accept/reject behavior isn't just a sandbox artifact.
2. Microphone -- `python -m voice.audio.list_devices`, then a short manual capture.
3. VAD -- speak with a mid-sentence pause; confirm it isn't cut off (Section 9).
4. STT -- compare `base.en` vs `tiny.en` transcripts on real short commands.
5. LLM -- compare `gemma3:270m` vs `qwen2.5:1.5b-instruct` response quality/latency for real questions.
6. TTS -- compare `-medium` vs `-high` voice tiers by ear.
7. Speaker -- confirm the ALSA output device is actually the robot's speaker, not some other sink.
8. IPC -- run `scripts/run_reference_humanfollower.py` + `main_voice.py` together (Section 5 of `docs/INSTALL.md`).
9. HumanFollower pause -- once wired up (`humanfollower_integration/README.md`), confirm real deceleration happens on `PAUSE_REQUEST`.
10. HumanFollower resume -- confirm it only happens after `VOICE_SESSION_COMPLETE`, never before TTS finishes.

Then together:
- Full cycle: following -> wake word -> smooth stop -> listen -> STT -> LLM -> TTS -> playback complete -> resume following.
- No speech after wake word -> confirm ~6s timeout (not the full 20s), resume.
- Speech well under 20s -> confirm STT starts immediately on end-of-speech, not after waiting out the ceiling.
- Speech approaching/exceeding 20s -> confirm the hard ceiling fires with partial audio, not a hang.
- TTS output not re-triggering the wake word (Section 17's feedback-loop concern) -- speak a question, let it answer, confirm no spurious second session starts from the robot's own voice.
- Repeated wake cycles back-to-back -- watch RSS over 20-30 cycles for any growth (`scripts/benchmark_voice_pipeline.py`'s RSS helpers, or plain `ps`/`top`).
- Ollama unavailable (`systemctl stop ollama` or equivalent) -- confirm graceful `llm_empty_or_failed` outcome and resume, not a hang.
- STT/TTS binary or model temporarily renamed -- confirm the startup-time `FileNotFoundError` path is hit cleanly, not a crash mid-session.
- Voice Manager killed (`kill -9`) mid-session -- confirm HumanFollower's watchdog force-resumes within `watchdog.heartbeat_timeout_s`.
- Microphone/speaker physically unplugged -- confirm the documented failure behavior (see `docs/ARCHITECTURE.md`'s failure table), not a silent hang.
- High CPU load -- run `scripts/benchmark_voice_pipeline.py --simulate-load` alongside a real HumanFollower run; watch for detection-loop/control-loop degradation.
- Low-memory conditions -- deliberately load a heavier LLM candidate and watch for swap activity (`vmstat`) rather than guessing.

Do not run destructive physical stress tests (e.g. forcing OOM kills on a
production robot mid-operation near people) without controlled, supervised
conditions.
