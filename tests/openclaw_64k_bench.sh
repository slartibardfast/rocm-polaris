#!/usr/bin/env bash
# OpenClaw 64K context benchmark fixture.
# Used as the per-step checkpoint in Phase 25 (see PHASE25.md).
#
# Production target: 5.0 t/s sustained generation at realistic session
# fill, c65536 context, Qwen3.5-35B-A3B Q4_K_M on dual Xeon X5650.
#
# Workflow:
#   1. Run a fast smoke test (~15 s) — fail-fast gate for broken builds
#   2. If smoke passes, run the realistic fixture (~17-70 min depending
#      on FILL size)
#
# Usage:
#   tests/openclaw_64k_bench.sh <binary> [-ctk TYPE] [-ctv TYPE] [extra args]
#
# Environment overrides:
#   FILL=/path/to/fill.txt   -- fixture file (default: openclaw_32000.txt, 32K tokens)
#   CTX=65536                -- context allocation
#   NGEN=256                 -- generation tokens
#   NTHREADS=6               -- llama.cpp threads
#   NUMA_NODE=1              -- numactl --membind/--cpunodebind
#   SKIP_SMOKE=1             -- skip the smoke gate (only for re-runs)
#
# Example:
#   tests/openclaw_64k_bench.sh /usr/bin/llama-completion -ctk q4_0 -ctv q4_0
#   FILL=.../openclaw_8000.txt tests/openclaw_64k_bench.sh build-tq/bin/llama-completion -ctk tq_kv_fused -ctv tq_kv_fused
#
# Reproducibility discipline (see plan):
#   - --no-mmap: reload model from disk each run for NUMA determinism
#   - drop_caches: fresh page cache state
#   - numactl --membind=1 --cpunodebind=1: fixed NUMA topology, 6 cores on clean node
#   - VK_ICD_FILENAMES=/dev/null: suppress Vulkan backend
#   - performance governor: locked at 2.66 GHz
#   - -n 256: sustained generation, not burst

set -euo pipefail

BINARY="${1:?Usage: $0 <binary> [extra llama-completion args]}"
shift

MODEL="${MODEL:-/home/llm/models/Qwen3.5-35B-A3B-Q4_K_M.gguf}"
FILL="${FILL:-/home/llm/rocm-polaris/tests/openclaw_32000.txt}"
CTX="${CTX:-65536}"
NGEN="${NGEN:-256}"
NTHREADS="${NTHREADS:-6}"
NUMA_NODE="${NUMA_NODE:-1}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SMOKE_SCRIPT="$(dirname "$(readlink -f "$0")")/smoke_test.sh"

if [[ ! -x "$BINARY" ]]; then
    echo "FATAL: binary not found or not executable: $BINARY" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "FATAL: model not found: $MODEL" >&2
    exit 1
fi
if [[ ! -f "$FILL" ]]; then
    echo "FATAL: fill file not found: $FILL" >&2
    exit 1
fi

# Verify the performance governor is locked.
GOV=$(cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
if [[ "$GOV" != "performance" ]]; then
    echo "WARNING: CPU governor is '$GOV', expected 'performance'. Run:" >&2
    echo "  for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee \$c > /dev/null; done" >&2
    exit 1
fi

# Smoke gate: fail fast if the binary is broken before burning 17-70 min
# on the realistic fixture. Smoke uses mmap and small context to keep
# total smoke time under 30 seconds.
if [[ "$SKIP_SMOKE" != "1" ]]; then
    echo "=============================================="
    echo "Smoke gate (fast go/no-go)"
    echo "=============================================="
    if ! "$SMOKE_SCRIPT" "$BINARY" "$@"; then
        echo "SMOKE GATE FAILED — refusing to run full fixture" >&2
        exit 1
    fi
    echo
fi

# Verify the fill file has roughly the expected token count.
FILL_BYTES=$(wc -c < "$FILL")
FILL_TOKENS_EST=$(( FILL_BYTES * 10 / 43 ))  # ~4.3 bytes/token
echo "Fill file: $FILL_BYTES bytes (~$FILL_TOKENS_EST tokens)"

# Drop caches for clean NUMA placement and DRAM state.
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

# Report the config being run.
echo "=============================================="
echo "OpenClaw 64K Benchmark Fixture"
echo "  Binary:   $BINARY"
echo "  Model:    $MODEL"
echo "  Ctx:      $CTX"
echo "  Threads:  $NTHREADS on NUMA node $NUMA_NODE"
echo "  Gen:      $NGEN tokens"
echo "  Extra:    $*"
echo "=============================================="

# Run.
set -x
VK_ICD_FILENAMES=/dev/null \
  numactl --membind="$NUMA_NODE" --cpunodebind="$NUMA_NODE" \
  "$BINARY" \
    -t "$NTHREADS" --no-mmap -ngl 0 \
    -c "$CTX" -n "$NGEN" \
    --simple-io -no-cnv \
    -f "$FILL" \
    -m "$MODEL" \
    "$@"
