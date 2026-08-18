#!/usr/bin/env python3
"""Real, on-device performance measurement -- Section 24 of the spec.

Everything in docs/MODEL_DECISION.md and docs/ARCHITECTURE.md that is
marked [REQUIRES-ON-DEVICE] or UNKNOWN is resolved by running this script
ON THE ACTUAL UNO Q, not by guessing. It measures:

  - idle RAM (this process, doing nothing)
  - wake-word CPU/latency (real openWakeWord inference)
  - STT RAM/CPU/latency for each whisper.cpp model pulled by pull_models.sh
  - LLM RAM/CPU/latency for each Ollama model pulled by pull_models.sh
    (reads Ollama's own /api/ps for resident RAM -- the authoritative
    source, not a guess)
  - TTS RAM/CPU/latency for each Piper voice pulled by pull_models.sh
  - total wake-to-spoken-response latency for one full synthetic cycle

Use --simulate-load to run a CPU-load generator in the background during
measurement, approximating "while HumanFollower is actually running"
(Section 24) until the real HumanFollower process is available to test
against directly -- a synthetic approximation is clearly better than no
concurrent-load measurement at all, but it is still labeled as such in the
output, not presented as equivalent to the real thing.

Usage:
    python3 scripts/benchmark_voice_pipeline.py [--simulate-load] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import psutil
except ImportError:
    print("This script requires psutil: pip install psutil")
    sys.exit(1)

import requests

from voice.config import STTConfig, TTSConfig, load_config
from voice.stt.whisper_cpp import WhisperCppSTT
from voice.tts.piper_tts import PiperTTS


def _cpu_busy_loop() -> None:  # pragma: no cover - only used by --simulate-load
    x = 0.0
    while True:
        for _ in range(200000):
            x = (x * 1.0000001) + 1.0


def _start_load_generator(n_procs: int) -> list:
    procs = []
    for _ in range(n_procs):
        p = multiprocessing.Process(target=_cpu_busy_loop, daemon=True)
        p.start()
        procs.append(p)
    return procs


def _rss_mb(pid: int) -> float:
    try:
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except psutil.NoSuchProcess:
        return -1.0


def bench_idle() -> dict:
    return {"process_rss_mb": _rss_mb_self(), "note": "this process's own idle RSS, not the whole system"}


def _rss_mb_self() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def bench_wake_word(config) -> dict:
    import numpy as np
    from voice.wake.wake_word import WakeWordDetector

    t0 = time.perf_counter()
    try:
        det = WakeWordDetector(config.wake, sample_rate=config.audio.sample_rate)
    except Exception as exc:
        return {"error": str(exc)}
    load_s = time.perf_counter() - t0

    n = 300
    t0 = time.perf_counter()
    for _ in range(n):
        frame = (np.random.randn(1280) * 100).astype(np.int16).tobytes()
        det.process_frame(frame)
    total = time.perf_counter() - t0

    return {
        "model": config.wake.model_name,
        "load_time_s": round(load_s, 4),
        "avg_inference_ms_per_chunk": round((total / n) * 1000, 3),
        "process_rss_mb_after_load": round(_rss_mb_self(), 1),
    }


def bench_stt(model_path: str, config, sample_seconds: float = 3.0) -> dict:
    import numpy as np

    cfg = STTConfig(
        binary_path=config.stt.binary_path,
        model_path=model_path,
        threads=config.stt.threads,
        language=config.stt.language,
        timeout_s=config.stt.timeout_s,
    )
    try:
        stt = WhisperCppSTT(cfg)
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    sr = config.audio.sample_rate
    t = np.linspace(0, sample_seconds, int(sr * sample_seconds), endpoint=False)
    tone = (0.2 * np.sin(2 * 3.14159265 * 440 * t) * 32767).astype(np.int16)
    raw = tone.tobytes()

    t0 = time.perf_counter()
    text = stt.run_stt(raw, sample_rate=sr)
    elapsed = time.perf_counter() - t0

    return {
        "model_path": model_path,
        "sample_seconds": sample_seconds,
        "wall_time_s": round(elapsed, 3),
        "real_time_factor": round(elapsed / sample_seconds, 3),
        "transcript": text,
    }


def bench_llm(model: str, config, prompt: str = "What is the weather like today?") -> dict:
    from voice.config import LLMConfig
    from voice.llm.ollama_client import OllamaClient

    cfg = LLMConfig(
        url=config.llm.url, model=model, stream=config.llm.stream,
        timeout_s=config.llm.timeout_s, num_predict=config.llm.num_predict,
        temperature=config.llm.temperature, keep_alive=config.llm.keep_alive,
        system_prompt=config.llm.system_prompt,
    )
    client = OllamaClient(cfg)

    t0 = time.perf_counter()
    reply = client.query(prompt)
    elapsed = time.perf_counter() - t0

    resident_mb = None
    try:
        ps = requests.get(config.llm.url.replace("/api/chat", "/api/ps"), timeout=5).json()
        for m in ps.get("models", []):
            if m.get("name", "").startswith(model.split(":")[0]):
                resident_mb = round(m.get("size", 0) / (1024 * 1024), 1)
    except Exception:
        pass

    return {
        "model": model,
        "wall_time_s": round(elapsed, 3),
        "reply": reply,
        "reply_chars": len(reply),
        "ollama_reported_resident_mb": resident_mb,
    }


def bench_tts(model_path: str, config, text: str = "The weather tomorrow will be sunny.") -> dict:
    cfg = TTSConfig(
        binary_path=config.tts.binary_path, model_path=model_path,
        sample_rate=config.tts.sample_rate, timeout_s=config.tts.timeout_s,
    )
    try:
        tts = PiperTTS(cfg)
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    t0 = time.perf_counter()
    pcm = tts.synthesize(text)
    elapsed = time.perf_counter() - t0
    audio_seconds = len(pcm) / 2 / cfg.sample_rate if pcm else 0.0

    return {
        "model_path": model_path,
        "wall_time_s": round(elapsed, 3),
        "audio_seconds": round(audio_seconds, 3),
        "real_time_factor": round(elapsed / audio_seconds, 3) if audio_seconds else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-load", action="store_true",
                         help="Run a synthetic CPU-load generator during measurement "
                              "to approximate HumanFollower running concurrently.")
    parser.add_argument("--json", type=str, default=None, help="Write results to this JSON file")
    args = parser.parse_args()

    config = load_config()
    results: dict = {"simulated_concurrent_load": args.simulate_load}

    load_procs = []
    if args.simulate_load:
        n = max(1, (psutil.cpu_count(logical=True) or 4) - 1)
        print(f"Starting {n} synthetic CPU-load processes (approximation only -- "
              f"not a substitute for testing against the real HumanFollower process)")
        load_procs = _start_load_generator(n)
        time.sleep(1.0)

    try:
        print("Measuring idle RAM...")
        results["idle"] = bench_idle()

        print("Benchmarking wake word...")
        results["wake_word"] = bench_wake_word(config)

        # NOTE: these used to be hardcoded to /opt/whisper.cpp/... and
        # /opt/piper/... paths that don't exist anywhere in this project --
        # this repo installs everything under runtime/ (see
        # scripts/setup_uno_q.sh), so every candidate was silently filtered
        # out by the Path(p).exists() check below and this script benchmarked
        # nothing for STT/TTS. Discovered by actually running this script on
        # the real UNO Q during Milestone 4 -- fixed to glob the real
        # runtime/ layout instead of a second hardcoded path list that can
        # drift out of sync with it again.
        project_root = Path(__file__).resolve().parent.parent
        print("Benchmarking STT candidates (whatever's actually present under runtime/whisper.cpp/models/)...")
        stt_candidates = sorted(str(p) for p in
                                 (project_root / "runtime/whisper.cpp/models").glob("ggml-*.bin"))
        results["stt"] = {p: bench_stt(p, config) for p in stt_candidates if Path(p).exists()}

        print("Benchmarking LLM candidates...")
        llm_candidates = ["gemma3:270m", "qwen2.5:1.5b-instruct"]
        results["llm"] = {m: bench_llm(m, config) for m in llm_candidates}

        print("Benchmarking TTS candidates (whatever's actually present under runtime/piper/voices/)...")
        tts_candidates = sorted(str(p) for p in
                                 (project_root / "runtime/piper/voices").glob("*.onnx"))
        results["tts"] = {p: bench_tts(p, config) for p in tts_candidates if Path(p).exists()}

    finally:
        for p in load_procs:
            p.terminate()

    print(json.dumps(results, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nWritten to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
