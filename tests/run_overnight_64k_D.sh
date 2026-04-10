#!/usr/bin/env bash
# Config D: GPU draft spec decode, 8 threads (4 per socket), NUMA mirror.
# Tests dual-socket mirrored weights + GPU draft at 64K. Fewer threads
# per socket than Config A (4 vs 6) may reduce contention while the
# mirror keeps reads local on both sockets.
#
# Waits for Config C to finish before starting.

set -uo pipefail
cd /home/llm/rocm-polaris

DONE_FILE=tests/overnight_64k_C.done
while [[ ! -f "$DONE_FILE" ]]; do sleep 30; done
echo "[D] C completed, cooldown..." && sleep 30

BIN_S="llama.cpp/src/llama.cpp-b8508/build-tq/bin/llama-server"
MODEL="/home/llm/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"
DRAFT="/home/llm/models/Qwen3.5-0.8B-Q4_K_M.gguf"
LOG=tests/overnight_64k_D.log
PORT=18097

echo "[D] starting at $(date)" > "$LOG"
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

# 8 threads across both sockets (4 per socket)
numactl --membind=0,1 --cpunodebind=0,1 \
  "$BIN_S" -t 8 --no-mmap -ngl 0 -dev none \
  --numa mirror \
  -ngld 99 -devd Vulkan0 \
  -c 65536 \
  -m "$MODEL" -md "$DRAFT" \
  --port $PORT > tests/overnight_64k_D_server.log 2>&1 &
SRVPID=$!

echo "[D] server PID=$SRVPID, waiting..." >> "$LOG"
for i in $(seq 1 300); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && echo "[D] server up after ${i}s" >> "$LOG" && break
  kill -0 $SRVPID 2>/dev/null || { echo "[D] server died during startup" >> "$LOG"; exit 1; }
  sleep 1
done

REQFILE=$(mktemp /tmp/spec64k-D-XXXXXX.json)
python3 -c "
import json
prompt = open('tests/fill-64k.txt').read()
json.dump({'prompt': prompt, 'n_predict': 128, 'temperature': 0}, open('$REQFILE', 'w'))
"
echo "[D] request file: $(wc -c < $REQFILE) bytes" >> "$LOG"
echo "[D] sending prompt at $(date)" >> "$LOG"

curl -s http://127.0.0.1:$PORT/completion \
  -H "Content-Type: application/json" \
  --data-binary @"$REQFILE" \
  > tests/overnight_64k_D_resp.json 2>&1
echo "[D] response received at $(date)" >> "$LOG"

rm -f "$REQFILE"

grep -E "perf_context.*eval time|acceptance rate|statistics draft" tests/overnight_64k_D_server.log >> "$LOG"
python3 -c '
import sys,json
d=json.loads(open("tests/overnight_64k_D_resp.json").read())
t=d.get("timings",{})
dn=t.get("draft_n",0); da=t.get("draft_n_accepted",0)
print(f"PP: {t.get(\"prompt_per_second\",0):.2f} t/s ({t.get(\"prompt_n\",0)} tokens)")
print(f"Gen: {t.get(\"predicted_per_second\",0):.2f} t/s ({t.get(\"predicted_n\",0)} tokens)")
print(f"Draft: gen={dn}, accepted={da}" + (f", rate={da/dn*100:.0f}%" if dn else ""))
' >> "$LOG" 2>&1

kill $SRVPID 2>/dev/null; wait $SRVPID 2>/dev/null
echo "[D] done at $(date)" >> "$LOG"
echo "EXIT=0" > tests/overnight_64k_D.done

echo "=== Config D Results ==="
cat "$LOG"
