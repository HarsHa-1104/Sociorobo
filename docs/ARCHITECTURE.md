# Architecture

## Final design

```
                                    ROBOT (UNO Q, headless)
                                             |
                     +-----------------------+----------------------+
                     |                                               |
             HUMAN FOLLOWER PROCESS                          VOICE MANAGER PROCESS
             (owns all motor authority --                    (no motor authority, ever --
              NOT part of this repo;                          this repo)
              see "HumanFollower integration" below)
                     |                                               |
                 Camera                                    AudioManager (single
                     |                                      persistent mic stream --
             Person Detection                                voice/audio/manager.py)
                     |                                                |
            Follow Controller <---- PAUSE_REQUEST -----+          +---+---+
                     |                                  |          |       |
                 Motors                                 |     WakeWordDetector  SpeechSegmenter
                     |                                  |     (always active     (active only
              (existing deceleration/                    |      while WAKE_       during LISTENING)
               stop logic -- never                        |      LISTENING)
               below 60 min-speed)                          |          |
                     |                                    |     wake fires
                     |<-- VOICE_SESSION_COMPLETE ----------+          |
                     |    (Unix socket, JSON messages --              v
                     |     voice/ipc/)                          PAUSE_REQUEST (see left)
              Resume Following                                        |
                                                                 LISTENING (VAD, two-tier
                                                                  timeout, early exit on
                                                                  end-of-speech)
                                                                        |
                                                                 WhisperCppSTT
                                                                        |
                                                                 OllamaClient (local,
                                                                  single-turn, no history)
                                                                        |
                                                                 PiperTTS (synthesize +
                                                                  block until aplay exits)
                                                                        |
                                                                 VOICE_SESSION_COMPLETE
                                                                  (see left)

WATCHDOG: HumanFollower's side of the IPC contract independently
force-resumes if no VOICE_SESSION_COMPLETE / HEARTBEAT arrives within a
bounded window after PAUSE_REQUEST -- see voice/ipc/server_stub.py's
_watchdog_loop for the reference implementation and its documented
failure policy. This is what makes "voice failure must never cause
uncontrolled motor behavior" true even when Voice Manager crashes outright.

NO GUI. NO DISPLAY. NO PYGAME. NO X SERVER. (see "GUI removal" below)
```

## HumanFollower integration

**The real HumanFollower source code was not available when this was
built** -- the material provided was the voice pipeline only. Nothing in
this repo modifies or assumes internal knowledge of HumanFollower's code.
Instead:

- `voice/ipc/protocol.py` defines the exact wire contract (message types,
  JSON shape).
- `voice/ipc/server_stub.py` is a **reference implementation** of what
  HumanFollower's side needs to do: listen on the Unix socket, call its
  own real deceleration/stop routine when `PAUSE_REQUEST` arrives, call its
  own real resume routine on `VOICE_SESSION_COMPLETE`, and run the
  watchdog loop that force-resumes on a missing heartbeat or an absolute
  timeout.
- `scripts/run_reference_humanfollower.py` runs that reference server
  standalone (with simulated deceleration, no real motors) so the Voice
  Manager can be manually exercised end-to-end before it's wired into the
  real HumanFollower process.
- `humanfollower_integration/README.md` is the integration checklist for
  whoever owns HumanFollower's code -- port the callback logic from
  `server_stub.py`'s `_dispatch`/`_watchdog_loop` into HumanFollower's real
  control loop, calling real deceleration/resume functions in place of the
  stub's placeholders.

## Process architecture (Section 11/15)

Two separate OS processes, never threads within one process:

- **HumanFollower** -- camera, detection, follow control, motor authority.
  Not part of this repo.
- **Voice Manager** (`main_voice.py`) -- wake word, audio capture, VAD,
  STT, LLM, TTS. Has *zero* motor authority and no import of anything
  motor-related.

Why separate processes rather than a thread inside HumanFollower: fault
isolation. A hang in the LLM call, a crash from a missing model file, or
an unhandled exception anywhere in the voice stack cannot corrupt
HumanFollower's memory or control loop if it's a different process address
space entirely. The IPC boundary (Unix domain socket, Section 15's
explicit recommendation, deliberately not over-engineered into a message
queue or pub/sub system) is the only channel between them, and it carries
only the five message types in `voice/ipc/protocol.py` -- never a motor
command, by construction (Voice Manager's code has no path to emit one).

## Voice state machine (Section 13/28)

See `voice/manager/state_machine.py` for the enforced transition graph
(illegal transitions raise `IllegalTransitionError` -- this is checked at
runtime by `VoiceManager._transition`, not just documented in prose).

```
WAKE_LISTENING --(wake word fires)--> PAUSE_PENDING
PAUSE_PENDING  --(proceeds regardless of PAUSE_CONFIRMED, see below)--> LISTENING
LISTENING      --(speech captured)--> PROCESSING_STT
LISTENING      --(no-speech timeout / max-duration timeout w/ no speech)--> SESSION_COMPLETE
PROCESSING_STT --(non-empty transcript)--> PROCESSING_LLM
PROCESSING_STT --(empty/failed transcript)--> SESSION_COMPLETE
PROCESSING_LLM --(non-empty reply)--> SPEAKING
PROCESSING_LLM --(empty/failed reply)--> SESSION_COMPLETE
SPEAKING       --(always, success or failure)--> SESSION_COMPLETE
SESSION_COMPLETE --> WAKE_LISTENING
```

Ownership: `WAKE_LISTENING`/`PAUSE_PENDING`/`LISTENING`/`PROCESSING_STT`/
`PROCESSING_LLM`/`SPEAKING`/`SESSION_COMPLETE` are all Voice Manager
states. `FOLLOWING`/`PAUSING`/`RESUMING`/`SAFE_STOP` are HumanFollower
states that this repo does not model directly (it has no visibility into
HumanFollower's internals) -- Voice Manager only ever *requests*
transitions on HumanFollower's side via `PAUSE_REQUEST` /
`VOICE_SESSION_COMPLETE`.

**Why proceeding into LISTENING doesn't require PAUSE_CONFIRMED**: Voice
Manager has no way to independently verify motors are stopped (it has no
motor authority or telemetry). Waiting indefinitely for a confirmation
that might never arrive (HumanFollower busy, crashed, or simply not
implementing this optional message yet) would make Voice Manager's own
liveness depend on HumanFollower's, in the wrong direction -- so it logs a
warning and proceeds after `ipc.pause_confirm_timeout_s` (default 8s)
either way. HumanFollower's own deceleration logic is the actual source of
truth for whether it's safe to listen, not this handshake.

## Audio architecture (Section 10/14)

One `AudioManager` (`voice/audio/manager.py`) owns exactly one PyAudio
stream for the process's lifetime, gated by `suspend()`/`resume()`
(stop_stream/start_stream), never repeated open/close -- this preserves
the one thing Phase 1 found the old project already did right, and it's
the direct mitigation for the ALSA device-busy issues previously
encountered. `WakeWordDetector` and `SpeechSegmenter` are both pure
frame-processing objects with no direct hardware access -- `VoiceManager`
is the only thing that decides which one gets fed frames at any moment,
and it does so structurally: `wake_detector.process_frame()` is *only*
ever called from `_wait_for_wake()`, never during a session (see
`voice/manager/voice_manager.py`'s class docstring for why this is
enforced by code structure, not a flag).

TTS-into-mic feedback (Section 17 of the Phase 1 audit) is handled by
suspending the entire audio stream (not just the wake-word detector) for
the whole `PAUSE_PENDING` -> `SPEAKING` span, with a configurable cooldown
(`vad.post_tts_cooldown_s`, default 0.5s) after playback before resuming
capture.

## GUI removal (Section 14)

Confirmed removed, not just hidden:

| Old file | Status |
|---|---|
| `face_animation/face.py` (pygame renderer) | Not carried into this project at all |
| `face.png`, `mouth.png`, `images/Desktop/*`, `images/Robot/*` | Not carried into this project |
| `pygame` dependency | Removed from `requirements.txt` |
| Face-thread wiring in `main.py` | `main_voice.py` has no thread, import, or reference to any rendering code |
| Amplitude-callback plumbing in the old `tts_piper.py` | Replaced by `PiperTTS.play()`, which blocks on `subprocess.run(...).wait()` -- deterministic completion, no animation-driven timing loop |

`main_voice.py` never imports anything display-related and never requires
`DISPLAY`/X11/Wayland/a virtual framebuffer to run.

## Failure handling (Section 17)

| Failure | Behavior | Where |
|---|---|---|
| Wake-word engine fails to load (missing dependency/model) | `build_voice_manager()` raises, `main_voice.py` logs and exits non-zero rather than starting in a broken state | `main_voice.py` |
| Microphone unavailable | `AudioManager.start()` raises `MicrophoneUnavailableError`; same as above at startup | `voice/audio/manager.py` |
| STT binary/model missing | Raises `FileNotFoundError` at construction, same startup-fail-fast behavior | `voice/stt/whisper_cpp.py` |
| STT fails/times out mid-session | Caught, logged, empty transcript -> session ends with `VOICE_SESSION_COMPLETE(outcome="stt_empty_or_failed")` -- HumanFollower resumes | `voice/manager/voice_manager.py::_run_stt` |
| Ollama unavailable/times out | Caught, logged, empty reply -> session ends the same way (`outcome="llm_empty_or_failed"`) | `voice/manager/voice_manager.py::_run_llm` |
| TTS fails | Caught, logged -> session still ends and reports `outcome="tts_failed"`, HumanFollower still resumes (never blocks resume on TTS success) | `voice/manager/voice_manager.py::_run_tts` |
| HumanFollower unreachable (no socket) | `HumanFollowerLink` fails soft (returns False/logs), Voice Manager proceeds through the session anyway rather than hanging | `voice/ipc/client.py` |
| Voice Manager crashes/hangs mid-session | HumanFollower's watchdog (reference implementation in `voice/ipc/server_stub.py`) force-resumes after `watchdog.heartbeat_timeout_s` (default 6s) of missing heartbeats, or `watchdog.max_paused_duration_s` (default 45s) absolute ceiling regardless | `voice/ipc/server_stub.py::_watchdog_loop` |
| Any unhandled exception inside a session | Caught at the `_run_stt`/`_run_llm`/`_run_tts` boundary via broad `except Exception` + logging, never propagates to crash the whole process mid-session | `voice/manager/voice_manager.py` |

The documented default policy when HumanFollower can't confirm Voice
Manager is still alive: **force-resume known-good autonomous following**,
not "stay stopped forever." See `server_stub.py::_watchdog_loop`'s
docstring for the reasoning and for how to flip this to a `SAFE_STOP`
policy instead if a specific deployment's safety analysis calls for it.

## Resource requirements

See `docs/MODEL_DECISION.md` for the full evidence trail behind each
model choice, and run `scripts/benchmark_voice_pipeline.py` on the real
board for authoritative RAM/CPU/latency numbers -- nothing in this repo's
documentation should be treated as a substitute for that on-device run.
