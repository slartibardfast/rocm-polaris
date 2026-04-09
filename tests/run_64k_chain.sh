#!/usr/bin/env bash
# Overnight chain runner for Phase 26 #1 close-out + #2 quality + #4a
# spec decode prep validation. Sequence:
#
#   1. Wait for the running default 64K bench (PID $1) to finish.
#   2. 30-second cooldown to let CPU thermal/cache settle.
#   3. Opt-in KV mirror 64K bench (LLAMA_NUMA_MIRROR_KV=1).
#   4. 30-second cooldown.
#   5. Spec decode end-to-end test (llama-server with --model-draft).
#   6. 30-second cooldown.
#   7. Perplexity smoke (small wikitext slice, default config).
#
# Each step writes its own .out / .err / .done file in tests/ so we
# can see what completed even if a later step fails.

set -uo pipefail

WAIT_PID="${1:?Usage: $0 <PID-of-default-bench>}"

cd /home/llm/rocm-polaris

BIN_C="llama.cpp/src/llama.cpp-b8508/build-tq/bin/llama-completion"
BIN_S="llama.cpp/src/llama.cpp-b8508/build-tq/bin/llama-server"
BIN_P="llama.cpp/src/llama.cpp-b8508/build-tq/bin/llama-perplexity"
MODEL="/home/llm/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"
DRAFT="/home/llm/models/Qwen3.5-0.8B-Q4_K_M.gguf"

# 1. Wait for the previous bench process to exit
echo "[chain] waiting for PID $WAIT_PID to exit" > tests/run_64k_chain.log
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
done
echo "[chain] PID $WAIT_PID exited at $(date)" >> tests/run_64k_chain.log

# 2. Cooldown
sleep 30

# 3. Opt-in KV mirror 64K bench
echo "[chain] starting opt-in 64K bench at $(date)" >> tests/run_64k_chain.log
LLAMA_NUMA_MIRROR_KV=1 NTHREADS=12 NUMA_NODE=0,1 SKIP_SMOKE=1 \
FILL=tests/fill-64k.txt CTX=65536 NGEN=128 \
tests/openclaw_64k_bench.sh \
  "$BIN_C" \
  --numa mirror -ctk tq_kv_1b -ctv tq_v_4b \
  > tests/phase26_64k_optin.out 2> tests/phase26_64k_optin.err
RC1=$?
echo "[chain] opt-in 64K bench exit=$RC1 at $(date)" >> tests/run_64k_chain.log
echo "EXIT=$RC1" > tests/phase26_64k_optin.done

# 4. Cooldown
sleep 30

# 5. Spec decode end-to-end test (llama-server -md against 0.8B draft)
echo "[chain] starting spec decode test at $(date)" >> tests/run_64k_chain.log
if [[ -f "$BIN_S" && -f "$DRAFT" ]]; then
    NTHREADS=6 NUMA_NODE=1 PORT=18080 \
    tests/spec_decode_test.sh "$BIN_S" \
      > tests/phase26_spec_decode.out 2> tests/phase26_spec_decode.err
    RC2=$?
else
    echo "skipped: missing $BIN_S or $DRAFT" > tests/phase26_spec_decode.out
    RC2=99
fi
echo "[chain] spec decode test exit=$RC2 at $(date)" >> tests/run_64k_chain.log
echo "EXIT=$RC2" > tests/phase26_spec_decode.done

# 6. Cooldown
sleep 30

# 7. Perplexity smoke (200-token slice of wikitext, default mirror config)
echo "[chain] starting perplexity smoke at $(date)" >> tests/run_64k_chain.log
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
VK_ICD_FILENAMES=/dev/null \
  numactl --membind=0,1 --cpunodebind=0,1 \
  "$BIN_P" \
    -t 12 --no-mmap -ngl 0 \
    -c 2048 \
    --numa mirror \
    -ctk tq_kv_1b -ctv tq_v_4b \
    -m "$MODEL" \
    -f tests/wikitext-2-test.txt \
    --chunks 4 \
    > tests/phase26_perplexity.out 2> tests/phase26_perplexity.err
RC3=$?
echo "[chain] perplexity exit=$RC3 at $(date)" >> tests/run_64k_chain.log
echo "EXIT=$RC3" > tests/phase26_perplexity.done

echo "[chain] all steps complete at $(date)" >> tests/run_64k_chain.log
echo "EXIT=$RC1+$RC2+$RC3" > tests/run_64k_chain.done
