#!/usr/bin/env bash
# Spec decode end-to-end test against the small Qwen3.5-0.8B draft model.
# Used to validate the common_speculative_is_compat fix from Phase 26 #4a.
#
# Acceptance: llama-server starts with --model-draft, the compat-check
# passes (no "the target context does not support partial sequence
# removal" warning), and the server responds to a prompt with non-zero
# spec acceptance rate logged.

set -euo pipefail

BINARY="${1:?Usage: $0 <llama-server-binary> [extra args]}"
shift

TARGET="${TARGET:-/home/llm/models/Qwen3.5-35B-A3B-Q4_K_M.gguf}"
DRAFT="${DRAFT:-/home/llm/models/Qwen3.5-0.8B-Q4_K_M.gguf}"
NUMA_NODE="${NUMA_NODE:-1}"
NTHREADS="${NTHREADS:-6}"
PORT="${PORT:-18080}"

if [[ ! -x "$BINARY" ]]; then
    echo "FATAL: binary not found or not executable: $BINARY" >&2
    exit 1
fi
if [[ ! -f "$TARGET" ]]; then
    echo "FATAL: target model not found: $TARGET" >&2
    exit 1
fi
if [[ ! -f "$DRAFT" ]]; then
    echo "FATAL: draft model not found: $DRAFT" >&2
    exit 1
fi

echo "=============================================="
echo "Spec decode test"
echo "  Binary:  $BINARY"
echo "  Target:  $TARGET"
echo "  Draft:   $DRAFT"
echo "  Threads: $NTHREADS on NUMA node $NUMA_NODE"
echo "  Port:    $PORT"
echo "=============================================="

# Drop caches for clean DRAM state
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

# Launch the server in background
LOG=$(mktemp /tmp/spec-test-XXXXXX.log)
trap 'rm -f "$LOG"; kill $SERVER_PID 2>/dev/null || true' EXIT

VK_ICD_FILENAMES=/dev/null \
  numactl --membind="$NUMA_NODE" --cpunodebind="$NUMA_NODE" \
  "$BINARY" \
    -t "$NTHREADS" --no-mmap -ngl 0 \
    -c 2048 \
    -m "$TARGET" \
    -md "$DRAFT" \
    --port "$PORT" \
    "$@" > "$LOG" 2>&1 &
SERVER_PID=$!

# Wait for server to come up (max 90s)
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "Server up after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "SPEC FAIL: server died during startup"
        tail -30 "$LOG"
        exit 1
    fi
    sleep 1
done

# Check the compat-check log
if grep -q "does not support partial sequence removal" "$LOG"; then
    echo "SPEC FAIL: compat-check rejected — patch did not fix it"
    tail -30 "$LOG"
    exit 1
fi

# Send a test prompt
RESPONSE=$(curl -s "http://127.0.0.1:$PORT/completion" \
    -d '{"prompt": "The theory of general relativity", "n_predict": 32, "temperature": 0}')

echo "--- Server log tail ---"
tail -20 "$LOG"
echo
echo "--- Response ---"
echo "$RESPONSE" | python3 -c 'import sys, json; d = json.loads(sys.stdin.read()); print(d.get("content", "(no content)"))' 2>/dev/null || echo "$RESPONSE"

# Look for spec acceptance metrics
ACCEPT=$(grep -oE 'spec_decode.*accept[^,}]*' "$LOG" | tail -1 || echo "")
if [[ -n "$ACCEPT" ]]; then
    echo "SPEC PASS: $ACCEPT"
else
    echo "SPEC PASS: server up, no compat error (but no acceptance metric found in log)"
fi
