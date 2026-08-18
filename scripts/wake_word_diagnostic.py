#!/usr/bin/env python3
"""Controlled comparison for the wake-word score-variance investigation.

Investigates why openWakeWord scores were observed to vary widely
(~0.45-0.52 in some runs, ~0.99 in others) by isolating variables instead
of guessing. Three paths, same underlying utterance:

  A. Clean digital: Piper TTS synthesizes "hey jarvis", resampled straight
     to 16kHz in-memory (no speaker, no mic, no room). This is the
     best-case ceiling for this exact model/onnxruntime install.
  B. Real acoustic loopback: the same synthesized utterance is played out
     the verified speaker (hw:0,3) and captured back in *real time* through
     the actual AudioManager code path (48kHz capture -> resample -> 480-
     sample frames) -- speaker frequency response, room acoustics, and the
     USB mic's ADC are all genuinely in the loop, not simulated.
  C. Chunk-cadence check: the same clean digital audio from A fed to
     WakeWordDetector.process_frame() in two different external chunk
     sizes (1280 samples at once vs. 480 samples at a time, matching how
     AudioManager really delivers frames). Since process_frame() buffers
     internally before ever calling the model, these two feeding patterns
     should produce IDENTICAL score sequences if the buffering code is
     correct -- this directly tests the "chunk cadence" hypothesis from
     the investigation brief.

IMPORTANT CAVEAT: this uses a synthetic TTS voice (Piper en_US-lessac),
not a real human voice. It is useful for isolating pipeline-vs-acoustic
effects and for a rough sanity check, but is NOT a substitute for real
human false-accept/false-reject validation (Milestone 2's remaining human
step -- see the script's own output for what to test manually).

Usage:
    python3 scripts/wake_word_diagnostic.py
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voice.audio.manager import AudioManager
from voice.config import load_config
from voice.tts.piper_tts import PiperTTS
from voice.wake.wake_word import WakeWordDetector


def _resample(pcm_int16: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    import samplerate
    ratio = dst_rate / src_rate
    resampler = samplerate.Resampler(converter_type="sinc_best", channels=1)
    out = resampler.process(pcm_int16.astype(np.float32), ratio, end_of_input=True)
    return np.clip(out, -32768, 32767).astype(np.int16)


def _score_sequence(detector: WakeWordDetector, pcm_int16: np.ndarray, feed_chunk: int) -> list:
    """Feed audio through process_frame() in `feed_chunk`-sample pieces,
    returning the raw score for every model-internal 1280-sample chunk by
    monkeypatching predict() to record scores as they happen."""
    scores = []
    real_predict = detector._model.predict

    def _spy_predict(audio):
        result = real_predict(audio)
        scores.append(max(result.values()) if result else 0.0)
        return result

    detector._model.predict = _spy_predict
    detector.reset()
    for start in range(0, len(pcm_int16), feed_chunk):
        chunk = pcm_int16[start:start + feed_chunk].tobytes()
        detector.process_frame(chunk)
    detector._model.predict = real_predict
    return scores


def main() -> int:
    config = load_config()
    wake_cfg = config.wake
    # Force-enable for this diagnostic even if disabled in the live config.
    wake_cfg.enabled = True

    print(f"Model: {wake_cfg.model_name}  threshold={wake_cfg.threshold}  "
          f"trigger_level={wake_cfg.trigger_level}\n")

    # --- Synthesize the wake phrase with Piper (already-verified TTS) ---
    tts = PiperTTS(config.tts)
    phrase = "hey jarvis"
    print(f"Synthesizing {phrase!r} with Piper (rate={config.tts.sample_rate}Hz)...")
    pcm_22k = tts.synthesize(phrase)
    if not pcm_22k:
        print("Piper synthesis FAILED -- cannot run this diagnostic.")
        return 1
    samples_22k = np.frombuffer(pcm_22k, dtype=np.int16)
    print(f"  -> {len(samples_22k)} samples ({len(samples_22k)/config.tts.sample_rate:.2f}s)\n")

    # =====================================================================
    # PATH A: clean digital, resampled straight to 16kHz, no speaker/mic.
    # =====================================================================
    print("=== PATH A: clean digital (TTS -> resample -> detector, no hardware) ===")
    samples_16k_clean = _resample(samples_22k, config.tts.sample_rate, 16000)
    # Pad with silence front/back so the model's own 5-frame warm-up
    # zeroing (openwakeword/model.py) doesn't eat into the real phrase.
    pad = np.zeros(16000, dtype=np.int16)  # 1s silence each side
    padded_clean = np.concatenate([pad, samples_16k_clean, pad])

    det_a = WakeWordDetector(wake_cfg, sample_rate=16000)
    scores_a = _score_sequence(det_a, padded_clean, feed_chunk=1280)
    peak_a = max(scores_a) if scores_a else 0.0
    print(f"  Score sequence: {[round(s, 3) for s in scores_a]}")
    print(f"  PEAK score: {peak_a:.4f}\n")

    # =====================================================================
    # PATH C: same clean audio, two different external feed chunk sizes.
    # =====================================================================
    print("=== PATH C: chunk-cadence check (1280-at-once vs 480-at-a-time) ===")
    det_c1 = WakeWordDetector(wake_cfg, sample_rate=16000)
    scores_c1 = _score_sequence(det_c1, padded_clean, feed_chunk=1280)
    det_c2 = WakeWordDetector(wake_cfg, sample_rate=16000)
    scores_c2 = _score_sequence(det_c2, padded_clean, feed_chunk=480)
    identical = scores_c1 == scores_c2
    print(f"  1280-sample feed peak: {max(scores_c1):.4f}")
    print(f"  480-sample feed peak:  {max(scores_c2):.4f}")
    print(f"  Score sequences IDENTICAL: {identical} "
          f"({'confirms buffering makes external chunk size irrelevant' if identical else 'UNEXPECTED -- investigate buffering code'})\n")

    # =====================================================================
    # PATH B: real acoustic loopback through hw:0,3 speaker + USB mic.
    # =====================================================================
    print("=== PATH B: real acoustic loopback (speaker hw:0,3 -> air -> USB mic) ===")
    print("  Playing synthesized phrase through the speaker while recording live...")

    audio = AudioManager(config.audio)
    audio.start()
    captured = []
    import subprocess
    import threading

    def _play():
        time.sleep(0.5)  # let capture start first
        subprocess.run(
            ["aplay", "-D", config.audio.output_device, "-f", "S16_LE",
             "-r", str(config.tts.sample_rate), "-c", "1", "-t", "raw", "-q"],
            input=pcm_22k,
        )

    player = threading.Thread(target=_play)
    player.start()

    t0 = time.monotonic()
    for frame in audio.frames():
        captured.append(frame)
        if time.monotonic() - t0 > 4.0:  # 4s capture window
            break
    player.join()
    audio.stop()

    captured_pcm = np.frombuffer(b"".join(captured), dtype=np.int16)
    det_b = WakeWordDetector(wake_cfg, sample_rate=16000)
    scores_b = _score_sequence(det_b, captured_pcm, feed_chunk=1280)
    peak_b = max(scores_b) if scores_b else 0.0
    print(f"  Captured {len(captured_pcm)} samples ({len(captured_pcm)/16000:.2f}s)")
    print(f"  Score sequence: {[round(s, 3) for s in scores_b]}")
    print(f"  PEAK score: {peak_b:.4f}\n")

    # --- Summary ---
    print("=== SUMMARY ===")
    print(f"  Path A (clean digital):     peak={peak_a:.4f}")
    print(f"  Path B (real acoustic loop): peak={peak_b:.4f}")
    print(f"  Delta A-B: {peak_a - peak_b:+.4f}")
    print(f"  Chunk cadence affects score: {'NO (confirmed identical)' if identical else 'YES -- unexpected, needs investigation'}")
    print()
    print("CAVEAT: Path A/B use a synthetic TTS voice (Piper en_US-lessac), not a")
    print("real human voice or your actual room acoustic conditions. This isolates")
    print("pipeline-vs-acoustic-path effects; it does NOT replace real human")
    print("false-accept/false-reject testing (say 'hey jarvis' yourself, at varying")
    print("distances, and watch `arduino-app-cli`-style logs / this project's own")
    print("logs for the live score).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
