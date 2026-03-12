#!/bin/bash
# llama.cpp ROCm gfx803 smoke test
#
# Downloads a tiny GGUF model and runs GPU inference.
# Requires: llama-cpp-rocm-polaris installed, ~400MB disk for model.
#
# Models that fit in 2GB VRAM (WX 2100):
#   Qwen2.5-0.5B-Instruct Q4_K_M  ~400MB  (recommended for smoke test)
#   TinyLlama-1.1B-Chat   Q4_K_M  ~670MB
#   SmolLM2-360M-Instruct Q8_0    ~390MB
#
# Usage: ./smoke-test.sh

set -euo pipefail

MODEL_DIR="${HOME}/.cache/llama-polaris"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_FILE="$MODEL_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"

mkdir -p "$MODEL_DIR"

# Download model if not cached
if [ ! -f "$MODEL_FILE" ]; then
    echo "=== Downloading Qwen2.5-0.5B-Instruct Q4_K_M (~400MB) ==="
    curl -L -o "$MODEL_FILE" "$MODEL_URL"
fi

echo "=== llama.cpp ROCm Smoke Test ==="
echo "Model: $(basename "$MODEL_FILE")"
echo "GPU:   $(rocminfo 2>/dev/null | grep -m1 'Marketing Name' | sed 's/.*: *//')"
echo ""

# Test 1: Verify GPU is detected
echo "--- Test 1: GPU detection ---"
HSA_ENABLE_INTERRUPT=0 llama-cli \
    -m "$MODEL_FILE" \
    -n 1 -p "x" \
    --no-display-prompt \
    -ngl 99 \
    2>&1 | grep -iE 'hip|rocm|gfx|gpu|device|vram|offload' | head -10
echo ""

# Test 2: Short generation (GPU offload all layers)
echo "--- Test 2: Short generation (GPU) ---"
HSA_ENABLE_INTERRUPT=0 llama-cli \
    -m "$MODEL_FILE" \
    -n 32 \
    -p "The capital of France is" \
    --no-display-prompt \
    -ngl 99 \
    --temp 0 \
    2>&1
echo ""

# Test 3: Timing stats
echo "--- Test 3: Timing ---"
HSA_ENABLE_INTERRUPT=0 llama-cli \
    -m "$MODEL_FILE" \
    -n 64 \
    -p "Explain what a GPU is in one sentence." \
    --no-display-prompt \
    -ngl 99 \
    --temp 0 \
    2>&1 | grep -E 'eval|timing|token'

echo ""
echo "=== Done ==="
