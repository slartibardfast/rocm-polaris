#!/usr/bin/env bash
# Overnight 64K production bench with all Phase 26 optimizations.
# Runs 3 configs sequentially, captures results.
#
# Config A: CPU-only, no draft, q4_0 K + tq_v_4b V, --numa mirror
# Config B: CPU-only, no draft, q4_0 K + tq_v_4b V, single socket
# Config C: GPU draft spec decode, q4_0 K + tq_v_4b V, single socket

set -uo pipefail
cd /home/llm/rocm-polaris

BIN_C="llama.cpp/src/llama.cpp-b8508/build-tq/bin/llama-completion"
BIN_S="llama.cpp/src/llama.cpp-b8508/build-tq/bin/llama-server"
MODEL="/home/llm/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"
DRAFT="/home/llm/models/Qwen3.5-0.8B-Q4_K_M.gguf"
LOG=tests/overnight_64k.log

echo "[overnight] starting at $(date)" > "$LOG"

# === Config A: NUMA mirror, 12 threads, no draft ===
echo "[overnight] Config A: numa mirror, no draft at $(date)" >> "$LOG"
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
NTHREADS=12 NUMA_NODE=0,1 SKIP_SMOKE=1 \
FILL=tests/fill-64k.txt CTX=65536 NGEN=128 \
tests/openclaw_64k_bench.sh "$BIN_C" \
  --numa mirror -ctk q4_0 -ctv tq_v_4b \
  > tests/overnight_64k_A.out 2> tests/overnight_64k_A.err
grep -E "eval time =.*runs|prompt eval time" tests/overnight_64k_A.err >> "$LOG"

sleep 30

# === Config B: Single socket, 6 threads, no draft ===
echo "[overnight] Config B: single socket, no draft at $(date)" >> "$LOG"
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
NTHREADS=6 NUMA_NODE=1 SKIP_SMOKE=1 \
FILL=tests/fill-64k.txt CTX=65536 NGEN=128 \
tests/openclaw_64k_bench.sh "$BIN_C" \
  -ctk q4_0 -ctv tq_v_4b \
  > tests/overnight_64k_B.out 2> tests/overnight_64k_B.err
grep -E "eval time =.*runs|prompt eval time" tests/overnight_64k_B.err >> "$LOG"

sleep 30

# === Config C: GPU draft spec decode, single socket ===
echo "[overnight] Config C: GPU draft spec decode at $(date)" >> "$LOG"
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
numactl --membind=1 --cpunodebind=1 \
  "$BIN_S" -t 6 --no-mmap -ngl 0 -dev none \
  -ngld 99 -devd Vulkan0 \
  -c 65536 \
  -m "$MODEL" -md "$DRAFT" \
  --port 18095 > tests/overnight_64k_C_server.log 2>&1 &
SRVPID=$!
for i in $(seq 1 300); do
  curl -sf http://127.0.0.1:18095/health > /dev/null 2>&1 && break
  kill -0 $SRVPID 2>/dev/null || { echo "Server died" >> "$LOG"; break; }
  sleep 1
done
# Send the 64K fill as prompt + 128 gen
if kill -0 $SRVPID 2>/dev/null; then
  curl -s http://127.0.0.1:18095/completion \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "import json; print(json.dumps({'prompt': open('tests/fill-64k.txt').read(), 'n_predict': 128, 'temperature': 0}))")" \
    > tests/overnight_64k_C_resp.json 2>&1
  sleep 2
  grep -E "perf_context.*eval time\|acceptance rate\|statistics draft" tests/overnight_64k_C_server.log >> "$LOG"
fi
kill $SRVPID 2>/dev/null; wait $SRVPID 2>/dev/null

echo "[overnight] all done at $(date)" >> "$LOG"
echo "EXIT=0" > tests/overnight_64k.done

# Summary
echo
echo "=== OVERNIGHT 64K RESULTS ==="
cat "$LOG"
