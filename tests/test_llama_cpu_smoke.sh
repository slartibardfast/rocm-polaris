#!/bin/bash
# CPU smoke test for llama.cpp with TinyLlama Q8_0
# Validates that model loading, tokenization, and inference work.
# Uses llama-completion (non-interactive text completion mode).
#
# Prerequisites:
#   - llama-cpp-rocm-polaris installed
#   - /home/llm/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf present
#
# Expected: generates tokens, exits cleanly, reports t/s

set -euo pipefail

MODEL="/home/llm/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
PROMPT="The capital of France is"
N_PREDICT=10
TIMEOUT=30

echo "=== llama.cpp CPU Smoke Test ==="
echo "Model: $(basename $MODEL)"
echo "Prompt: $PROMPT"
echo "Tokens: $N_PREDICT"
echo ""

if [ ! -f "$MODEL" ]; then
    echo "FAIL: Model not found at $MODEL"
    echo "Download with: python3 -c \"from huggingface_hub import hf_hub_download; hf_hub_download('TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', 'tinyllama-1.1b-chat-v1.0.Q8_0.gguf', local_dir='/home/llm/models')\""
    exit 1
fi

# Run CPU-only completion, capture output
OUTPUT=$(timeout -k 5 $TIMEOUT llama-completion \
    -m "$MODEL" \
    -ngl 0 \
    -p "$PROMPT" \
    -n $N_PREDICT \
    -c 64 \
    2>&1)

RC=$?

# Extract perf stats
PROMPT_TS=$(echo "$OUTPUT" | grep "prompt eval time" | grep -oP '[\d.]+(?= tokens per second)')
EVAL_TS=$(echo "$OUTPUT" | grep "eval time.*runs" | grep -oP '[\d.]+(?= tokens per second)')
GENERATED=$(echo "$OUTPUT" | grep "eval time.*runs" | grep -oP '\d+(?= runs)')

echo "Exit code: $RC"
echo "Prompt eval: ${PROMPT_TS:-?} t/s"
echo "Generation:  ${EVAL_TS:-?} t/s"
echo "Tokens gen:  ${GENERATED:-0}"

if [ $RC -eq 0 ] && [ -n "$EVAL_TS" ]; then
    echo ""
    echo "PASS"
else
    echo ""
    echo "FAIL (RC=$RC)"
    exit 1
fi
