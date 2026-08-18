# Milestone 6 Validation — End-to-End Integration

Real on-device evidence that the full `wake -> VAD -> STT -> LLM -> TTS ->
speaker -> ready for next interaction` cycle works repeatedly, per the
Phase 1 definition of done. All measurements below are from actual
`main_voice.py` runs on the real UNO Q, not simulated.

## Bugs found and fixed en route

Two crash-class bugs were found only by actually running the process and
letting real, unscripted usage happen -- neither was reachable by the
existing unit test suite before this milestone (see the two commits below
for full root-cause writeups and the regression tests added):

1. **`SESSION_COMPLETE -> PAUSE_PENDING` crash on the second wake cycle.**
   A prior fix (see project history) only patched the wake-disabled debug
   bypass; re-enabling real wake detection in Milestone 2 exposed the same
   gap in the real detection code path. Fixed in
   `voice/manager/voice_manager.py`.
2. **Sample-rate mismatch corrupting both VAD and STT.** Three call sites
   passed `config.audio.sample_rate` (48kHz, the raw mic capture rate)
   into code that only ever receives `AudioManager`'s already-resampled
   16kHz output. This mislabeled every WAV header STT built and fed the
   wrong rate into every `webrtcvad.is_speech()` call, which silently
   accepts a wrong-but-byte-length-compatible rate rather than raising.
   Directly caused both the garbage transcripts ("(music)", repeated-word
   decoder loops) and the ~800ms premature listening cutoff seen in early
   testing. Fixed in the same file.

## Seven-session live run (2026-08-18, 10:04-10:28)

One continuous `main_voice.py` process, real speech, real Ollama, real
Piper, real speaker. No restarts between sessions.

| # | Transcript | Outcome | Wake-to-end latency |
|---|---|---|---|
| 1 | "What do you do for living?" | answered | 14.23s |
| 2 | *(silence)* | no_speech_timeout | 6.87s |
| 3 | "tell me" | answered | 20.93s |
| 4 | "Do you know anything about Arduino? You know Q?" | answered | 15.56s |
| 5 | "I'm not saying about ordino. I'm saying about ordino. You know you." | llm_empty_or_failed | 12.82s |
| 6 | "Tell me about yourself." | answered | 18.87s |
| 7 | "Do you know anything about or do you know you know Q?" | answered | 19.16s |

- **Zero crashes across 7 consecutive sessions.**
- All three of the state machine's failure/completion branches fired
  for real, not just in a mock: `answered` (5x), `no_speech_timeout`
  (1x, correctly hit the 6.0s ceiling), `llm_empty_or_failed` (1x,
  `PROCESSING_LLM -> SESSION_COMPLETE` correctly skipped `SPEAKING`).
- Every session correctly returned `SESSION_COMPLETE -> WAKE_LISTENING`
  and re-armed for the next wake word.
- Clean shutdown verified separately: `SIGTERM` -> `AudioManager stopped`
  -> `Voice Manager stopped cleanly.`, no hang, no traceback.
- Average latency for a full "answered" turn: **17.75s** (range
  14.2-20.9s) -- comfortably inside the 60-75s acceptable Phase 1
  baseline and already trending toward the ~40s long-term target ahead
  of Milestone 8's dedicated optimization pass.

## A real, known STT limitation surfaced here (not yet fixed)

Sessions 4 and 7 show whisper.cpp `base.en-q5_1` consistently mangling
"UNO Q" ("You know Q?", "you know you", "know Q"). This is a plausible,
known weakness of general STT models on uncommon proper nouns/product
names, not related to the sample-rate or context-window bugs above (both
already fixed). Not blocking Phase 1; worth revisiting in Milestone 8 or
later (e.g. a whisper.cpp prompt/vocabulary hint) if it proves a real
usability problem in practice.

## Conclusion

Milestone 6's completion criterion -- "the pipeline must survive multiple
consecutive interactions without crashing or entering an invalid state" --
is met, with real evidence, not a synthetic test. Proceeding to Milestone 7
(production hardening).
