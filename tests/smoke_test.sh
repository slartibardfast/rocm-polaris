#!/usr/bin/env bash
# Fast smoke test: does this llama-completion binary load the model,
# run the hybrid SSM/MoE forward pass, and produce coherent output?
#
# Budget: ~15 seconds (mmap load + small generation).
# NOT a benchmark — just a go/no-go gate before committing 20+ minutes
# to the realistic OpenClaw fixture.
#
# Checks performed:
#   1. Binary exists and is executable
#   2. Model loads without assertion failure
#   3. 16-token generation completes without segfault or abort
#   4. Output contains at least one alphabetic word (not all garbage)
#   5. Generation speed is at least 1.0 t/s (sanity floor — a real
#      regression or scalar fallback would drop much lower)
#
# Usage:
#   tests/smoke_test.sh <binary> [extra llama-completion args]
#
# Example:
#   tests/smoke_test.sh build-tq/bin/llama-completion -ctk tq_kv_1b -ctv q4_0

set -euo pipefail

BINARY="${1:?Usage: $0 <binary> [extra llama-completion args]}"
shift

MODEL="${MODEL:-/home/llm/models/Qwen3.5-35B-A3B-Q4_K_M.gguf}"
NUMA_NODE="${NUMA_NODE:-1}"
NTHREADS="${NTHREADS:-6}"
MIN_TPS="${MIN_TPS:-1.0}"

if [[ ! -x "$BINARY" ]]; then
    echo "SMOKE FAIL: binary not found or not executable: $BINARY" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "SMOKE FAIL: model not found: $MODEL" >&2
    exit 1
fi

# Drop caches so the smoke timing is reproducible (doesn't matter for
# go/no-go but keeps behaviour consistent with the full benchmark).
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

LOG=$(mktemp /tmp/smoke-XXXXXX.log)
trap 'rm -f "$LOG"' EXIT

# Use mmap (not --no-mmap) for fast load — smoke is about go/no-go
# correctness, not reproducible performance. ~5s load vs ~45s.
# Small context (c512) to keep KV allocation cheap.
# 16 tokens is enough to confirm the forward pass works and the
# sampler produces non-garbage output.
set +e
VK_ICD_FILENAMES=/dev/null \
  numactl --membind="$NUMA_NODE" --cpunodebind="$NUMA_NODE" \
  "$BINARY" \
    -t "$NTHREADS" -ngl 0 \
    -c 512 -n 16 \
    --simple-io -no-cnv \
    -m "$MODEL" \
    -p "The theory of general relativity" \
    "$@" > "$LOG" 2>&1
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
    echo "SMOKE FAIL: binary exited non-zero ($RC)" >&2
    tail -20 "$LOG" >&2
    exit 1
fi

# Extract generation speed from the perf summary
TPS=$(grep -oP 'eval time.*?\K[0-9.]+(?= tokens per second)' "$LOG" | tail -1)
if [[ -z "$TPS" ]]; then
    echo "SMOKE FAIL: no eval time line in output" >&2
    tail -20 "$LOG" >&2
    exit 1
fi

# Sanity floor check
AWK_OK=$(awk -v t="$TPS" -v min="$MIN_TPS" 'BEGIN { print (t >= min) ? "ok" : "fail" }')
if [[ "$AWK_OK" != "ok" ]]; then
    echo "SMOKE FAIL: gen speed $TPS t/s below floor $MIN_TPS t/s" >&2
    tail -30 "$LOG" >&2
    exit 1
fi

# Coherence check: look for at least one 4+ letter alphabetic word
# in the generation output (rejects all-punctuation / all-garbage).
# We strip the [MTP-FINDSLOT] debug lines first since they're on
# every token and contain "find_slot".
GEN_OUT=$(grep -v '^\[MTP-FINDSLOT' "$LOG" | \
          grep -v '^common_perf_print' | \
          grep -v '^llama_' | \
          grep -v '^load:' | \
          grep -v '^print_info:' | \
          grep -v '^sampler' | \
          grep -v '^main:' | \
          grep -v '^generate:' | \
          grep -v '^sched_' | \
          grep -v '^warning:' | \
          grep -v '^system_info:' | \
          tail -20)
WORD_COUNT=$(echo "$GEN_OUT" | grep -oE '[A-Za-z]{4,}' | wc -l)
if [[ $WORD_COUNT -lt 3 ]]; then
    echo "SMOKE FAIL: generation output has <3 alphabetic words (likely garbage)" >&2
    echo "--- generated text ---" >&2
    echo "$GEN_OUT" >&2
    exit 1
fi

echo "SMOKE PASS: gen=$TPS t/s, $WORD_COUNT alphabetic words in output"
echo "--- sample output ---"
echo "$GEN_OUT" | head -5
exit 0
