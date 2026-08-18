#!/usr/bin/env bash
# Fetch the model files docs/MODEL_DECISION.md recommends -- and the
# candidates it recommends benchmarking before finalizing. Run this AFTER
# scripts/setup_uno_q.sh, on the real UNO Q (this sandbox could not reach
# huggingface.co or the Ollama registry to run this itself -- see
# docs/MODEL_DECISION.md's network-access note).
#
# Section 20: models are documented here (name, quantization, source, size)
# so the installation stays reproducible.

set -euo pipefail

WHISPER_MODEL_DIR="/opt/whisper.cpp/models"
PIPER_VOICE_DIR="/opt/piper/voices"

echo "=== STT: whisper.cpp GGML models ==="
echo "Pulling base.en (shipped default) and tiny.en (documented fallback)"
echo "in a couple of quantizations each, so the on-device benchmark in"
echo "scripts/benchmark_voice_pipeline.py can compare them directly."
cd /opt/whisper.cpp
for model in base.en-q5_0 base.en-q5_1 tiny.en-q5_1 tiny.en-q8_0; do
    echo "--- ggml-${model}.bin ---"
    bash ./models/download-ggml-model.sh "$model" || echo "  FAILED -- check network access to huggingface.co"
done
ls -la "$WHISPER_MODEL_DIR"/ggml-*.bin 2>/dev/null || true

echo
echo "=== LLM: Ollama models ==="
echo "Default (docs/MODEL_DECISION.md): gemma3:270m"
echo "Upgrade candidate (pending on-device RAM headroom check): qwen2.5:1.5b-instruct"
echo "NOT pulling qwen2.5:3b-instruct by default -- docs/MODEL_DECISION.md"
echo "recommends against it for a 2GB target. Uncomment below only if you"
echo "have specifically confirmed real headroom via the benchmark script."
ollama pull gemma3:270m
ollama pull qwen2.5:1.5b-instruct
# ollama pull qwen2.5:3b-instruct   # NOT recommended -- see docs/MODEL_DECISION.md
ollama list

echo
echo "=== TTS: Piper voices ==="
echo "Fetching both medium (shipped default) and high (old project's"
echo "choice) so scripts/benchmark_voice_pipeline.py can compare them."
mkdir -p "$PIPER_VOICE_DIR"
cd "$PIPER_VOICE_DIR"
BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac"
for tier in medium high; do
    for ext in onnx onnx.json; do
        f="en_US-lessac-${tier}.${ext}"
        echo "--- $f ---"
        curl -fsSL -o "$f" "${BASE_URL}/${tier}/${f}" || echo "  FAILED -- check network access to huggingface.co"
    done
done
ls -la "$PIPER_VOICE_DIR"

echo
echo "=== Done ==="
echo "Model inventory (Section 20 documentation):"
echo "  STT candidates: $WHISPER_MODEL_DIR/ggml-{base.en-q5_0,base.en-q5_1,tiny.en-q5_1,tiny.en-q8_0}.bin"
echo "  LLM candidates: gemma3:270m, qwen2.5:1.5b-instruct (via 'ollama list')"
echo "  TTS candidates: $PIPER_VOICE_DIR/en_US-lessac-{medium,high}.onnx"
echo
echo "Next: run scripts/benchmark_voice_pipeline.py to measure real RAM/CPU/"
echo "latency for each candidate on THIS board, then set the winners in"
echo "config/voice_config.yaml."
