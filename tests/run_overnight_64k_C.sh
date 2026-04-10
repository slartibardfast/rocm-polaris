#!/usr/bin/env bash
# Config C: GPU draft spec decode at 64K fill.
# Sends the prompt via a temp file piped through curl --data-binary @file
# to avoid the "Argument list too long" error.

set -uo pipefail
cd /home/llm/rocm-polaris

BIN_S="llama.cpp/src/llama.cpp-b8508/build-tq/bin/llama-server"
MODEL="/home/llm/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"
DRAFT="/home/llm/models/Qwen3.5-0.8B-Q4_K_M.gguf"
LOG=tests/overnight_64k_C.log
PORT=18096

echo "[C] starting at $(date)" > "$LOG"
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

numactl --membind=1 --cpunodebind=1 \
  "$BIN_S" -t 6 --no-mmap -ngl 0 -dev none \
  -ngld 99 -devd Vulkan0 \
  -c 65536 \
  -m "$MODEL" -md "$DRAFT" \
  --port $PORT > tests/overnight_64k_C_server.log 2>&1 &
SRVPID=$!

echo "[C] server PID=$SRVPID, waiting..." >> "$LOG"
for i in $(seq 1 300); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && echo "[C] server up after ${i}s" >> "$LOG" && break
  kill -0 $SRVPID 2>/dev/null || { echo "[C] server died during startup" >> "$LOG"; exit 1; }
  sleep 1
done

# Build the JSON request body as a temp file (avoids arg length limit)
REQFILE=$(mktemp /tmp/spec64k-XXXXXX.json)
python3 -c "
import json
prompt = open('tests/fill-64k.txt').read()
json.dump({'prompt': prompt, 'n_predict': 128, 'temperature': 0}, open('$REQFILE', 'w'))
"
echo "[C] request file: $(wc -c < $REQFILE) bytes" >> "$LOG"

# Send via --data-binary @file (no arg length limit)
echo "[C] sending prompt at $(date)" >> "$LOG"
curl -s http://127.0.0.1:$PORT/completion \
  -H "Content-Type: application/json" \
  --data-binary @"$REQFILE" \
  > tests/overnight_64k_C_resp.json 2>&1
echo "[C] response received at $(date)" >> "$LOG"

rm -f "$REQFILE"

# Extract results
grep -E "perf_context.*eval time|acceptance rate|statistics draft" tests/overnight_64k_C_server.log >> "$LOG"
python3 -c '
import sys,json
d=json.loads(open("tests/overnight_64k_C_resp.json").read())
t=d.get("timings",{})
dn=t.get("draft_n",0); da=t.get("draft_n_accepted",0)
print(f"PP: {t.get(\"prompt_per_second\",0):.2f} t/s ({t.get(\"prompt_n\",0)} tokens)")
print(f"Gen: {t.get(\"predicted_per_second\",0):.2f} t/s ({t.get(\"predicted_n\",0)} tokens)")
print(f"Draft: gen={dn}, accepted={da}" + (f", rate={da/dn*100:.0f}%" if dn else ""))
' >> "$LOG" 2>&1

kill $SRVPID 2>/dev/null; wait $SRVPID 2>/dev/null
echo "[C] done at $(date)" >> "$LOG"
echo "EXIT=0" > tests/overnight_64k_C.done

echo "=== Config C Results ==="
cat "$LOG"
