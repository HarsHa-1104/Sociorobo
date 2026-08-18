# Milestone 7 — Microphone Disconnect Recovery

Real on-device evidence for the mic-disconnect fault-injection work,
including a design iteration that was itself found unsafe and replaced.
All measurements below are from actual physical USB unplug/replug tests
on the real UNO Q, not simulated.

## The original bug (pre-Milestone 7)

`AudioManager.frames()` caught a broken-stream `OSError` on every read but
just logged a warning and retried forever at ~31ms intervals -- confirmed
on real hardware: it never recovered even after the device was physically
reconnected, and because the retry wait didn't block on the real stop
event, a mic failure could make the whole process ignore `SIGTERM` and
hang indefinitely.

## Iteration 1 — bounded retry + reopen via a fresh `pyaudio.PyAudio()`

Added bounded consecutive-error retry, then bounded reopen attempts with
backoff, recreating the PyAudio host instance each attempt. Unit-tested
cleanly. Real-hardware validation found a new, more serious problem:

- `pa.open()` against a genuinely-absent device does not raise -- it
  blocks indefinitely. Confirmed via an isolated diagnostic
  (`pyaudio.PyAudio()` returned in 0.04s; the following `pa.open()` never
  returned within a 20s hard timeout).
- Worse: repeatedly tearing down and recreating `pyaudio.PyAudio()` itself
  while the device was still absent was confirmed to kill the *entire
  process* with no Python-level exception or traceback at all --
  consistent with a native crash inside PortAudio/ALSA, not a hang.
  `journalctl -k` showed the kernel's own USB disconnect/reconnect
  timeline straddling the exact moment the process went silent, with zero
  Python output after the crash point. A thread-level timeout cannot catch
  this: a native crash kills every thread simultaneously, including the
  watcher thread.

## Iteration 2 (shipped) — reuse the existing PyAudio instance + OS-level presence check

Per explicit direction: no process-boundary isolation yet, narrow the fix
first. Recovery now:

1. Never calls `pyaudio.PyAudio()` again after the original `start()`.
   Only reopens a *stream* on the same already-initialized host instance.
2. Gates even that behind a PortAudio-free presence check: reads
   `/proc/asound/cards` (pure kernel/ALSA state, zero PortAudio
   involvement) and compares against a USB-Audio card-name signature
   captured at `start()` time. Recovery attempts skip `pa.open()` entirely
   while the device isn't visible at the OS level.
3. Still timeout-bounded (`REOPEN_TIMEOUT_S`, via a background thread) as
   defense-in-depth for the now-rare case where `pa.open()` hangs even
   after presence is confirmed.
4. Every wait (backoff, presence re-check) blocks on the real stop event,
   so shutdown remains responsive throughout.

## Real hardware validation (2026-08-18)

Multiple physical unplug/replug attempts were needed to get a clean,
complete run -- documented honestly rather than cherry-picked:

- Two attempts were **inconclusive**, not successful: the unplug never
  registered a real OSError at all (confirmed via `journalctl -k` showing
  no kernel-level USB disconnect event during those windows). No claim was
  made from those runs.
- One attempt exhausted all 5 bounded recovery attempts (~23s total
  backoff) before the device was replugged, correctly raising
  `MicrophoneRecoveryFailedError` -- expected, correct "fail visibly, not
  forever" behavior per the design, not a bug, but not the full recovery
  cycle either.
- The full cycle was captured cleanly in the run below.

```
Audio read error (10/10 before recovery attempt): [Errno -9988] Stream closed
Attempting microphone recovery (1/5) ... microphone not yet visible -- skipping PyAudio entirely this attempt.
Attempting microphone recovery (2/5) ... microphone not yet visible -- skipping PyAudio entirely this attempt.
Attempting microphone recovery (3/5) ... microphone not yet visible -- skipping PyAudio entirely this attempt.
Attempting microphone recovery (4/5) ... microphone not yet visible -- skipping PyAudio entirely this attempt.
Attempting microphone recovery (5/5) ... Microphone recovered after 5 attempt(s).
```

Confirmed after recovery: frame delivery (`frames_seen`) continued
incrementing cleanly for 60+ more seconds with zero further errors, and
the process never restarted (identical PID throughout). This is the full
cycle: **working -> unplug -> failure detected -> bounded recovery
correctly refuses to touch PyAudio while absent (no crash) -> mic
reconnected -> recovery succeeds -> capture resumes -- all in the same
process.**

## Tests

`tests/test_audio_manager.py` (9 tests, fake PyAudio + a real temp file
standing in for `/proc/asound/cards`): brief sub-threshold errors don't
trigger reopen; presence-gated recovery never reconstructs PyAudio;
exhausted recovery raises visibly with zero PyAudio touches while absent;
`stop()` is responsive during initial retry, backoff, and presence
polling; a failing-to-open reopen is bounded; a hanging `pa.open()` is
bounded by timeout without reconstructing PyAudio; a config with no
captured signature doesn't get permanently blocked. Full suite: 76/76
passing.

## Known remaining limitation

If `pa.open()` itself ever crashes the process (rather than hangs) even
after the presence check confirms the device is visible, this iteration
does not protect against that -- only Option 1 (process-level isolation)
would. No evidence of that specific failure mode was found in this
milestone's testing; the crash previously observed was specifically tied
to calling `pyaudio.PyAudio()` (full host re-init) while the device was
absent, which this iteration eliminates entirely.
