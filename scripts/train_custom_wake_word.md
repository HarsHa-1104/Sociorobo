# Training a custom "hey_arduino" wake word (documented, not executed)

This procedure was **not run** as part of this implementation session --
see docs/MODEL_DECISION.md for why (it needs a synthetic-data generation
pipeline and a training run that shouldn't happen unsupervised inside a
cloud audit session). This document is the exact path to follow when
ready to attempt it for real, plus the validation gate before it can
replace the stock `hey_jarvis` model.

## Why this is realistic (recap from docs/MODEL_DECISION.md)

openWakeWord trains custom keyword models from **synthesized speech**, not
recordings of a real person -- a TTS engine generates thousands of
variations of the target phrase, which are then mixed with background
noise/room-impulse-response samples and used to train a small classifier
head on top of openWakeWord's shared, frozen feature-extraction models.
The resulting model is the same architecture and size (~1.3MB) as any
stock keyword -- so there is no runtime cost difference between
"hey_jarvis" and a custom "hey_arduino" once trained. The only question is
whether the trained model is reliable enough, and that can only be
answered empirically.

## Procedure

1. Clone `dscripka/openWakeWord` and follow its
   `notebooks/automatic_model_training.ipynb` pipeline (or the
   equivalent scripted version, `openwakeword.train`). This requires:
   - `piper-sample-generator` (or another TTS-based synthetic speech
     generator) to produce several thousand positive examples of "hey
     arduino" spoken with varied prosody/speed/pitch.
   - A negative-example set: openWakeWord's project provides pre-built
     negative feature sets (general speech + noise) suitable for reuse.
   - A GPU is strongly recommended for training speed; CPU-only training
     is possible but slow.
2. Export the trained model to ONNX (openWakeWord's training pipeline does
   this as a final step).
3. Copy the resulting `.onnx` file to this project, e.g.
   `/opt/wake/hey_arduino.onnx`, and set in `config/voice_config.yaml`:
   ```yaml
   wake:
     model_path: "/opt/wake/hey_arduino.onnx"
     model_name: "hey_arduino"
   ```

## Validation gate (do not skip)

Before `hey_arduino` replaces `hey_jarvis` as the default, it must pass
both of the following, run on the real UNO Q with the real microphone in
the real intended environment (not a quiet room, if the robot won't
operate in one):

1. **False-reject test**: say "hey arduino" naturally, in the robot's
   actual operating environment, at least 30 times across varied distances
   and speaking styles. Target: comparable detection rate to `hey_jarvis`'s
   real-world track record (openWakeWord's stock models are widely used
   and well-validated; the custom model should not be noticeably worse).
2. **False-accept test**: run the detector continuously for at least
   30-60 minutes of the robot's normal operating audio environment
   (ambient conversation, TV/media in the background, the robot's own
   motor noise, etc. -- whatever it will actually hear) with the wake word
   never spoken, and count false triggers. Target: zero, or a rate low
   enough that it won't meaningfully disrupt normal following behavior.

If either test fails to clear a bar comparable to the stock model,
keep `hey_jarvis` as the shipped default rather than trading reliability
for a more on-brand keyword -- this is the explicit priority ordering
from Section 6 of the spec ("this preference is NOT more important than
computational efficiency and reliability").
