# HumanFollower integration checklist

The real HumanFollower source code was not available when this Voice
Manager was built (the material provided was the voice pipeline project
only), so this repository cannot wire into it directly. This is the
checklist for whoever owns HumanFollower's code to complete the
integration.

## What HumanFollower needs to implement

1. **Listen on a Unix domain socket** at the path in
   `config/voice_config.yaml`'s `ipc.socket_path` (default
   `/tmp/humanfollower_voice.sock`). One connection per voice session --
   Voice Manager opens a new connection on each wake-word detection and
   closes it at the end of the session.

2. **Parse newline-delimited JSON messages** per `voice/ipc/protocol.py`.
   You do not need to import that module (though you can -- it's a plain
   Python file with no hardware dependencies) as long as your
   implementation produces/consumes the same wire format:
   ```json
   {"type": "PAUSE_REQUEST", "session_id": "a1b2c3", "ts": 1234567.89}
   ```

3. **On `PAUSE_REQUEST`**: call your existing controlled-deceleration
   routine. Once motors are confirmed stopped, send back
   `{"type": "PAUSE_CONFIRMED", "session_id": "<same id>", "ts": ...}`.
   This is optional but recommended -- Voice Manager proceeds into
   listening after a bounded timeout (`ipc.pause_confirm_timeout_s`,
   default 8s) even without it, but confirming lets it start listening as
   soon as motors are actually stopped rather than guessing.
   **Do not introduce any motor command below 60** as part of this
   deceleration -- use your existing stop mechanism, not a new
   intermediate crawl speed (project constraint).

4. **Track `HEARTBEAT` messages** sent periodically during the session
   (`ipc.heartbeat_interval_s`, default every 2s). Note the timestamp of
   the last one received.

5. **On `VOICE_SESSION_COMPLETE`**: call your existing resume-following
   routine. The message's `extra.outcome` field tells you why the session
   ended (`"answered"`, `"no_speech_timeout"`, `"stt_empty_or_failed"`,
   `"llm_empty_or_failed"`, `"tts_failed"`) -- useful for logging, not
   something you need to branch on; the correct action is the same
   (resume) regardless of outcome.

6. **Run an independent watchdog** that force-resumes if:
   - No `HEARTBEAT` has arrived for `watchdog.heartbeat_timeout_s`
     (default 6s) since a `PAUSE_REQUEST` was accepted and no
     `VOICE_SESSION_COMPLETE` has arrived yet, OR
   - `watchdog.max_paused_duration_s` (default 45s) has elapsed since
     `PAUSE_REQUEST`, regardless of heartbeats.

   **This is the single most safety-critical piece of the integration.**
   It's what guarantees "voice failure must never cause uncontrolled
   motor behavior" even if the Voice Manager process crashes outright. Do
   not skip it or treat it as optional hardening -- see
   `voice/ipc/server_stub.py::_watchdog_loop` for a complete, runnable
   reference implementation (with its reasoning documented in its
   docstring) to port logic from.

## What HumanFollower must NOT do

- Must not accept any message type as a motor command. The protocol has
  no such message, by design (Section 15/16 of the implementation brief).
- Must not block its own control loop waiting synchronously on Voice
  Manager -- run the socket listener and watchdog on a separate
  thread/process from the real-time control loop, the same way
  `server_stub.py` does it (accept loop + watchdog loop, both daemon
  threads, neither blocking the caller).

## Fastest way to validate your implementation

1. Get `main_voice.py` running against `scripts/run_reference_humanfollower.py`
   first (see `docs/INSTALL.md` step 5) to confirm the Voice Manager side
   works.
2. Swap in your real HumanFollower's socket listener in place of the
   reference server, keeping everything else the same.
3. Re-run the manual test: say the wake word, ask a question, confirm real
   deceleration/stop/resume happens instead of the simulated log lines.
4. Run the watchdog validation from `docs/TROUBLESHOOTING.md`'s last
   entry: `kill -9` the Voice Manager process mid-session and confirm your
   watchdog fires and resumes following on its own.
