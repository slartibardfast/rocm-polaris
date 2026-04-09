#!/usr/bin/env bash
# Wait for the first chain script's perplexity step to complete
# (signaled by tests/phase26_perplexity.done), then run an F16 baseline
# perplexity for comparison. The chain runs the TQ_KV_1B+TQ_V_4B config;
# this script runs F16+F16 so we can compute the PPL delta.
#
# Acceptance per PHASE26.md #2: TQ PPL within 1% of F16 baseline.

set -uo pipefail

cd /home/llm/rocm-polaris

DONE_FILE=tests/phase26_perplexity.done
BIN_P="llama.cpp/src/llama.cpp-b8508/build-tq/bin/llama-perplexity"
MODEL="/home/llm/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"
LOG=tests/phase26_perplexity_baseline_chain.log

echo "[ppl-baseline] waiting for $DONE_FILE" > "$LOG"

while [[ ! -f "$DONE_FILE" ]]; do
    sleep 30
done

echo "[ppl-baseline] $DONE_FILE appeared at $(date)" >> "$LOG"

# Cooldown
sleep 30

echo "[ppl-baseline] starting F16 baseline perplexity at $(date)" >> "$LOG"

echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true

VK_ICD_FILENAMES=/dev/null \
  numactl --membind=0,1 --cpunodebind=0,1 \
  "$BIN_P" \
    -t 12 --no-mmap -ngl 0 \
    -c 2048 \
    --numa mirror \
    -ctk f16 -ctv f16 \
    -m "$MODEL" \
    -f tests/wikitext-2-test.txt \
    --chunks 4 \
    > tests/phase26_perplexity_f16baseline.out 2> tests/phase26_perplexity_f16baseline.err
RC=$?

echo "[ppl-baseline] f16 baseline exit=$RC at $(date)" >> "$LOG"
echo "EXIT=$RC" > tests/phase26_perplexity_f16baseline.done
