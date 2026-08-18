# Model Decision Record

This document is the evidence trail behind every model choice in
`config/voice_config.yaml`. Per the Phase 2 brief: no model was chosen
because it has fewer parameters, and no model was carried over from the
old project just because it was already there. Three kinds of evidence are
used below, and each claim is labeled with which kind it is:

- **[MEASURED-SANDBOX]** -- actually run, in this implementation session, in
  the x86_64 cloud sandbox used to write this code. Real binaries, real (or
  real-format) models, real timings. **Not** the UNO Q -- treat these as
  "does the code path work, and what's the rough order of magnitude,"
  not as UNO Q performance numbers.
- **[SOURCED-PUBLIC]** -- a published, citable fact (a model's file size on
  the Ollama library, a vendor spec sheet), not measured by us.
- **[REQUIRES-ON-DEVICE]** -- cannot be determined without running on the
  actual UNO Q; flagged explicitly rather than guessed.

A critical environment note up front: the cloud sandbox this code was
written in has network access to source-code hosts (PyPI, `git clone`
against `github.com`, GitHub release-asset downloads) but **not** to
`ollama.com`, `huggingface.co`, or the Ollama model registry
(`registry.ollama.ai`) -- all returned `403` at the network layer. That
blocked pulling real Ollama LLM weights or real production Whisper/Piper
models for direct measurement here. The real UNO Q, during installation,
will have normal internet access and will not have this restriction --
`scripts/pull_models.sh` handles the real pulls there. What follows is the
most honest evidence obtainable without that access, clearly labeled.

---

## Wake word: openWakeWord, stock `hey_jarvis` model, "hey_arduino" evaluated but not shipped

**[MEASURED-SANDBOX]** openWakeWord installs cleanly via pip
(`openwakeword==0.6.0`, `onnxruntime==1.18.1`). Its per-keyword models are
downloaded separately (not bundled in the wheel) via
`openwakeword.utils.download_models()`, which succeeded in this sandbox
(that host, unlike ollama.com/huggingface.co, was reachable). Real,
production `hey_jarvis_v0.1.onnx` measured:

| Metric | Value |
|---|---|
| Model file size | 1.27 MB (`hey_jarvis_v0.1.onnx`) |
| Shared feature-extraction models (embedding + melspectrogram, loaded once, shared across all keywords) | 1.33 MB + 1.09 MB = 2.42 MB |
| Total on-disk footprint for one keyword | ~3.7 MB |
| Model load time | 0.118 s |
| Inference time per 80ms audio chunk | 3.43 ms average (300-sample run) |
| False positives on 300 frames of pure random noise | 0 |

Even accounting for a Cortex-A53 core being several times slower per-thread
than the x86_64 core this ran on, 3.43ms of compute per 80ms of audio
leaves a very large real-time margin (roughly 20x here) -- this is
credible evidence that wake-word inference cost is not the constraint on
the UNO Q; RAM is (this component's ~4MB footprint is trivial either way).

**A real bug this integration testing caught, worth calling out explicitly**:
the first version of `voice/wake/wake_word.py` omitted `wakeword_models`
from the openWakeWord constructor call when `config.model_path` was unset,
which makes openWakeWord silently load its **entire bundled stock model
set** (alexa, hey_jarvis, hey_mycroft, timer, weather, ...) instead of just
the configured keyword -- multiple models resident and evaluated every
frame, for no reason. This would never have been caught by a mocked unit
test; it only surfaced by actually constructing a real `Model()`. Fixed --
see `tests/test_wake_word.py::test_constructor_always_passes_explicit_single_model_list`
for the regression test.

**On "hey_arduino"**: [SOURCED-PUBLIC, from openWakeWord's own
documentation] custom keywords are trained from synthesized speech (a TTS
model generates thousands of training utterances) into the *same*
architecture as the stock models -- so a custom "hey_arduino" model would
be the same ~1.3MB, same ~3.4ms/chunk cost as `hey_jarvis` above. Cost is
not the blocker. Reliability is: false-accept/false-reject rate for a
custom keyword depends entirely on training data coverage of real acoustic
conditions (this robot's own motor/servo noise, room echo, distance), and
that can only be established by actually training a candidate and running
it against real audio -- which requires either GPU-accelerated training
time or a longer CPU training run, plus a synthetic-data generation
pipeline (`piper-sample-generator`), neither of which is something to run
unsupervised inside this audit/implementation session. **Decision: ship
`hey_jarvis` (validated, real-world-tested, zero training risk) as the
default and documented fallback; `scripts/train_custom_wake_word.md`
documents the exact procedure to attempt "hey_arduino" as a follow-up, to
be validated with a real false-accept/false-reject test
(`tests/test_wake_word_validation.py`, manual/on-device) before ever
becoming the shipped default.** This is not "don't do it" -- it's "don't
ship it unvalidated."

---

## STT: whisper.cpp, default model `base.en` (quantization TBD on-device), `tiny.en` as the documented fallback

**[MEASURED-SANDBOX]** whisper.cpp builds cleanly from source with cmake
(`whisper-cli` target, ~2 minutes on this sandbox's CPU count). The full
`WhisperCppSTT` wrapper was run end-to-end against a real compiled binary
using whisper.cpp's own bundled CI smoke-test model
(`models/for-tests-ggml-base.en.bin`, a deliberately truncated/stub model
the whisper.cpp project ships in-repo for pipeline testing, not real
transcription quality) and a synthetically generated 3-second 16kHz PCM
clip: subprocess invocation, WAV header construction, stdout parsing, and
ANSI/blank-audio-marker stripping all worked correctly end-to-end (2.26s
wall time for the 3s clip on this sandbox, 2 threads). This validates the
*code path*, not transcription quality -- the stub model correctly
produced no usable text, and the wrapper correctly returned `""` rather
than garbage, exactly as designed.

**[SOURCED-PUBLIC unreachable for a real base.en-q5_0 vs tiny.en-q5_1 size/latency
comparison in this sandbox** -- both huggingface.co (where whisper.cpp's
real GGML weights are hosted) and the Ollama-adjacent registries were
network-blocked here. The qualitative tradeoff from Phase 1 stands
unchanged: `base.en` gives meaningfully better accuracy on short spoken
commands/questions than `tiny.en`, at a real but bounded CPU/RAM cost, and
because STT here runs occasionally on short (<20s) clips rather than
continuously, the *absolute* latency delta between the two tiers matters
less than it would for sustained transcription. **Decision: keep `base.en`
as the shipped default (unchanged from Phase 1's reasoning), with `tiny.en`
as a one-line config change (`stt.model_path`) if on-device benchmarking
via `scripts/benchmark_voice_pipeline.py` shows real contention with the
vision pipeline.** The exact quantization level (q5_0 vs q5_1 vs q8_0) is
marked **[REQUIRES-ON-DEVICE]** -- pull all three during setup, compare
transcription latency and a handful of real test utterances, and adjust
`stt.model_path` accordingly; `scripts/pull_models.sh` fetches all three
so this comparison costs nothing extra during installation.

---

## LLM: `gemma3:270m` default, `qwen2.5:1.5b-instruct` as the documented "if RAM allows" upgrade, `qwen2.5:3b-instruct` explicitly NOT recommended for a 2GB target

This is the highest-stakes decision in the whole report, and the one the
brief asks to be most rigorous about. Real, sourced numbers first:

**[SOURCED-PUBLIC, Ollama model library, fetched during this session]**

| Model | Ollama tag | Published file size | Parameters |
|---|---|---|---:|
| Gemma 3 | `gemma3:270m` | **292 MB** (default Q8_0 quantization) | 268M |
| Qwen 2.5 Instruct | `qwen2.5:1.5b-instruct-q4_K_M` | **986 MB** | 1.5B |
| Qwen 2.5 Instruct | `qwen2.5:3b-instruct-q4_K_M` | **1.9 GB** | 3B |

Sources:
[gemma3:270m](https://ollama.com/library/gemma3:270m),
[qwen2.5:1.5b-instruct-q4_K_M](https://ollama.com/library/qwen2.5:1.5b-instruct-q4_K_M),
[qwen2.5:3b-instruct-q4_K_M](https://ollama.com/library/qwen2.5:3b-instruct-q4_K_M)

These are on-disk weight sizes, not runtime RAM -- actual resident RAM
while serving is the weight size plus KV-cache/context buffers plus the
Ollama runtime's own overhead, typically pushing the effective figure
higher than the file size, more so at longer context lengths. Exact
resident RAM is **[REQUIRES-ON-DEVICE]** (`ollama ps` while a request is
in flight) -- but the file sizes alone are already decisive reasoning
material against the spec's stated **hard 2GB budget shared with camera,
person detection, tracking, motor control, and an always-on wake word**:

- `qwen2.5:3b-instruct` at **1.9GB on disk alone** would consume the large
  majority of a 2GB ceiling before accounting for runtime overhead,
  camera/detection/motor software, the wake-word detector, and STT -- there
  is no realistic way this coexists with the rest of the stack inside a 2GB
  budget without severe swap risk. This was the old Jetson project's
  choice, made on hardware with a completely different RAM envelope; it
  does not transfer. **Explicitly not recommended for the UNO Q under this
  budget.**
- `gemma3:270m` at **292MB** leaves the most headroom by a wide margin --
  consistent with what the project owner reports actually having run
  before on a previous UNO Q. The known tradeoff (from general knowledge of
  270M-class instruction-tuned models, not measured here) is weaker
  instruction-following and topic coherence than a 1B+ model, which could
  show up as the model ignoring the "keep it to one short sentence" system
  prompt more often, or giving less useful answers to open-ended questions.
  Given the system prompt already constrains response length and
  `num_predict=96` caps it structurally regardless, this weakness is
  partially mitigated by construction.
- `qwen2.5:1.5b-instruct` at **986MB** is a genuine middle point: roughly
  3.4x `gemma3:270m`'s footprint, but still under half of the 2GB ceiling
  on disk alone, and a meaningfully more capable model class for following
  a "answer concisely" instruction reliably. This is the recommended
  upgrade path if on-device measurement shows the rest of the stack
  (camera + detection + motors + wake word + STT resident/warm cost) leaves
  enough real headroom above 986MB plus overhead.

**Decision: ship `gemma3:270m` as the safe default (matches the hard 2GB
budget with the most margin, and matches what the project owner already
successfully ran on a prior UNO Q), with `qwen2.5:1.5b-instruct` documented
and pre-staged by `scripts/pull_models.sh` as a one-line config change
(`llm.model`) once real on-device RAM measurement confirms headroom. Do
not use `qwen2.5:3b-instruct` on this hardware target.** This reverses the
old project's default, on RAM-budget grounds specific to this hardware,
not because smaller is inherently better -- the file-size evidence above is
the actual reason.

**[REQUIRES-ON-DEVICE]**: run `scripts/benchmark_voice_pipeline.py` for
all three models with the full stack under simulated concurrent load
before finalizing -- this record is the reasoning that produced the
*default*, not a substitute for that measurement.

---

## TTS: Piper, default voice tier `medium` (not the old project's `high`)

**[MEASURED-SANDBOX]** Piper's official prebuilt Linux x86_64 binary runs
correctly. Using Piper's own bundled CI test voice (`etc/test_voice.onnx`,
a real (if small, 27MB) functioning voice model, 16kHz), the actual
`PiperTTS.synthesize()` wrapper produced correct PCM audio for a real
English sentence ("The weather tomorrow will be sunny with a light
breeze."): 0.503s wall-clock synthesis time for ~2.84s of resulting audio
on this sandbox's CPU -- a real-time factor around 0.18 (i.e., synthesis
was about 5.6x faster than the audio's own playback duration on this
x86_64 core). This confirms the subprocess plumbing, stdout PCM capture,
and float/int16 handling all work correctly end-to-end.

**[SOURCED-PUBLIC unreachable for a direct medium-vs-high `en_US-lessac`
timing comparison** -- Piper's real production voices are hosted on
huggingface.co, which was network-blocked in this sandbox. The Phase 1
reasoning stands: "high" tier voices are the heaviest/slowest of Piper's
three quality tiers, and since the robot sits physically stopped for the
entire TTS duration (Section 11), synthesis latency directly extends how
long the robot is motionless -- for a task like "speed and quality
roughly equal priority," the safer default on a resource-constrained board
is `medium`, not `high`. **Decision: ship `en_US-lessac-medium` as the
default (already reflected in `config/voice_config.yaml`), fetch both
`-medium` and `-high` during setup via `scripts/pull_models.sh`, and
benchmark both for real synthesis latency + a real listening test before
finalizing -- this is a one-line config change either direction.**

---

## Summary table

| Component | Old project | Phase 2 default | Rationale |
|---|---|---|---|
| Wake word | none | openWakeWord `hey_jarvis` (stock) | Only validated option ready to ship; "hey_arduino" documented as a follow-up requiring on-device false-accept/reject validation |
| STT | whisper.cpp `base.en-q5_0` (unverified whether still current) | whisper.cpp `base.en` (exact quant TBD on-device) | Real subprocess pipeline validated end-to-end in-sandbox; exact quantization needs real weight files, which need on-device network access |
| LLM | `qwen2.5:3b-instruct` (1.9GB, Jetson-era) | `gemma3:270m` (292MB) | 1.9GB alone is incompatible with a hard 2GB shared budget; 292MB leaves the most real headroom; `qwen2.5:1.5b-instruct` (986MB) documented as the upgrade path pending on-device RAM headroom measurement |
| TTS | Piper `en_US-lessac-high` | Piper `en_US-lessac-medium` | Robot stays stopped for the full TTS duration; medium trades some naturalness for lower latency/CPU, both tiers staged for a real on-device comparison |
