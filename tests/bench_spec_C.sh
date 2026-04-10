#!/usr/bin/env bash
# Config C: single socket 6t, GPU draft, q4_0+tq_v_4b, 8K real fill, 512 gen
set -uo pipefail
cd /home/llm/rocm-polaris

PORT=18098
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

llama.cpp/src/llama.cpp-b8508/build-tq/bin/llama-server \
  -t 6 --no-mmap -ngl 0 -dev none \
  -ngld 99 -devd Vulkan0 \
  -c 8192 \
  -ctk q4_0 -ctv tq_v_4b \
  -m /home/llm/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -md /home/llm/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --port $PORT > tests/bench_spec_C_server.log 2>&1 &
PID=$!

for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || exit 1
  sleep 1
done

python3 -c "
import json
p = open('tests/openclaw_8k_fill.txt').read()
json.dump({'prompt': p, 'n_predict': 512, 'temperature': 0.7}, open('tests/bench_spec_C_req.json', 'w'))
"

curl -s http://127.0.0.1:$PORT/completion \
  -H 'Content-Type: application/json' \
  --data-binary @tests/bench_spec_C_req.json \
  --max-time 1800 \
  -o tests/bench_spec_C_resp.json

python3 -c "
import json
d = json.loads(open('tests/bench_spec_C_resp.json').read())
t = d.get('timings', {})
dn = t.get('draft_n', 0)
da = t.get('draft_n_accepted', 0)
print(f'PP: {t.get(\"prompt_n\",0)} tok, {t.get(\"prompt_per_second\",0):.1f} t/s')
print(f'Gen: {t.get(\"predicted_n\",0)} tok, {t.get(\"predicted_per_second\",0):.2f} t/s')
if dn: print(f'Draft: gen={dn}, accepted={da}, rate={da/dn*100:.0f}%')
"

kill $PID 2>/dev/null; wait $PID 2>/dev/null
rm -f tests/bench_spec_C_req.json
