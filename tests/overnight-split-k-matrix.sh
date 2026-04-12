#!/usr/bin/env bash
# overnight-split-k-matrix.sh — Complete validation suite for split K cache
#
# Runs sequentially (NEVER parallel GPU tests):
#   1. PPL matrix: all K×V cache type combos on wikitext-2 (c=512)
#   2. Throughput matrix: generation speed at c=512 and c=2048
#   3. Coherence gate: counting test at temp=0 for each config
#   4. Tool-call battery: server-based, 12 tests per config
#
# Output: tests/overnight_split_k_results.txt (tee'd)
#
# Usage:
#   bash tests/overnight-split-k-matrix.sh 2>&1 | tee tests/overnight_split_k.log
#
# Expected runtime: ~6-10 hours on dual Xeon X5650, CPU-only

set -euo pipefail

cd "$(dirname "$0")/.."
LLAMA_DIR="llama.cpp/src/llama.cpp-b8508"
BIN="$LLAMA_DIR/build-tq/bin"
MODEL="/home/llm/models/Qwen3.5-35B-A3B-mtp-q4km.gguf"
WIKITEXT="tests/wikitext-2-test.txt"
WIKITEXT_SHORT="/tmp/wikitext-overnight.txt"
RESULTS="tests/overnight_split_k_results.txt"
PORT=9099

# Create a ~50K sample for PPL (longer than smoke test, shorter than full)
head -c 50000 "$WIKITEXT" > "$WIKITEXT_SHORT"

THREADS=12
SEED=42

# K cache configs: f16 baseline, then turbo_kv_4b split variants
K_CONFIGS=(
    "f16"
    "f16:turbo_kv_4b"
    "q8_0:turbo_kv_4b"
    "q8_0:q4_0"
)

# V cache configs: f16 baseline + turbo_kv_4b
V_CONFIGS=(
    "f16"
    "turbo_kv_4b"
)

echo "============================================================"
echo "  Split K Cache Overnight Validation Matrix"
echo "  $(date)"
echo "  Model: $(basename $MODEL)"
echo "  Host: $(hostname)"
echo "============================================================"
echo ""

> "$RESULTS"

# ------------------------------------------------------------------
# 1. COHERENCE GATE — counting test at temp=0
# ------------------------------------------------------------------
echo "=== 1. COHERENCE GATE (counting, temp=0) ==="
echo ""

PROMPT="One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
COHERENCE_PASS=0
COHERENCE_TOTAL=0

for ck in "${K_CONFIGS[@]}"; do
    for cv in "${V_CONFIGS[@]}"; do
        COHERENCE_TOTAL=$((COHERENCE_TOTAL + 1))
        printf "  K=%-25s V=%-15s → " "$ck" "$cv"

        OUTPUT=$("$BIN/llama-completion" \
            -m "$MODEL" \
            -p "$PROMPT" -n 20 -c 512 --simple-io -no-cnv --temp 0 \
            --cache-type-k "$ck" --cache-type-v "$cv" \
            --no-warmup -ngl 0 -t $THREADS --seed $SEED 2>/dev/null || echo "CRASH")

        if echo "$OUTPUT" | grep -q "seventeen.*eighteen.*nineteen.*twenty"; then
            echo "PASS (counts correctly)"
            COHERENCE_PASS=$((COHERENCE_PASS + 1))
        elif echo "$OUTPUT" | grep -q "CRASH"; then
            echo "CRASH"
        else
            TAIL=$(echo "$OUTPUT" | tail -c 60 | tr '\n' ' ')
            echo "FAIL: ...$TAIL"
        fi
    done
done

echo ""
echo "  Coherence: $COHERENCE_PASS/$COHERENCE_TOTAL passed"
echo "  Coherence: $COHERENCE_PASS/$COHERENCE_TOTAL passed" >> "$RESULTS"
echo ""

# ------------------------------------------------------------------
# 2. PPL MATRIX — wikitext-2 perplexity
# ------------------------------------------------------------------
echo "=== 2. PPL MATRIX (wikitext-2, c=512) ==="
echo ""

for ck in "${K_CONFIGS[@]}"; do
    for cv in "${V_CONFIGS[@]}"; do
        printf "  K=%-25s V=%-15s → " "$ck" "$cv"

        PPL=$("$BIN/llama-perplexity" \
            -m "$MODEL" \
            -f "$WIKITEXT_SHORT" \
            --cache-type-k "$ck" --cache-type-v "$cv" \
            -c 512 -ngl 0 -t $THREADS 2>&1 | grep "Final" | grep -oP 'PPL = \K[0-9.]+' || echo "CRASH")

        echo "PPL=$PPL"
        echo "K=$ck V=$cv PPL=$PPL" >> "$RESULTS"
    done
done

echo ""

# ------------------------------------------------------------------
# 3. THROUGHPUT MATRIX — generation speed
# ------------------------------------------------------------------
echo "=== 3. THROUGHPUT MATRIX (n=20, c=512) ==="
echo ""

PROMPT_TPUT="The quick brown fox jumps over the lazy dog"

for ck in "${K_CONFIGS[@]}"; do
    for cv in "${V_CONFIGS[@]}"; do
        printf "  K=%-25s V=%-15s → " "$ck" "$cv"

        SPEED=$("$BIN/llama-completion" \
            -m "$MODEL" \
            -p "$PROMPT_TPUT" -n 20 -c 512 --simple-io -no-cnv \
            --cache-type-k "$ck" --cache-type-v "$cv" \
            --no-warmup -ngl 0 -t $THREADS --seed $SEED --temp 0 2>&1 | \
            grep "eval time" | grep -oP '(\d+\.\d+) ms per token' | head -1 || echo "CRASH")

        echo "$SPEED"
        echo "K=$ck V=$cv $SPEED" >> "$RESULTS"
    done
done

echo ""

# ------------------------------------------------------------------
# 4. THREE-WAY SERVER FACE-OFF
#    A) Standard model, f16 KV, CPU-only (baseline)
#    B) Standard model, split K + turbo V, CPU-only
#    C) MTP model, split K + turbo V, CPU-only
# ------------------------------------------------------------------
echo "=== 4. THREE-WAY SERVER FACE-OFF ==="
echo ""

MODEL_DRAFT="/home/llm/models/Qwen3.5-0.8B-mtp-q6k.gguf"

declare -A FACEOFF_LABEL
declare -A FACEOFF_MODEL
declare -A FACEOFF_CK
declare -A FACEOFF_CV
declare -A FACEOFF_NGL
declare -A FACEOFF_CTX

FACEOFF_LABEL[A]="MTP Baseline (f16/f16, CPU)"
FACEOFF_MODEL[A]="$MODEL"
FACEOFF_CK[A]="f16"
FACEOFF_CV[A]="f16"
FACEOFF_NGL[A]=0
FACEOFF_CTX[A]=2048

FACEOFF_LABEL[B]="MTP + Split K + turbo V (CPU)"
FACEOFF_MODEL[B]="$MODEL"
FACEOFF_CK[B]="q8_0:q4_0"
FACEOFF_CV[B]="turbo_kv_4b"
FACEOFF_NGL[B]=0
FACEOFF_CTX[B]=2048

# C: 0.8B MTP standalone on GPU (lightweight fast endpoint)
FACEOFF_LABEL[C]="0.8B MTP standalone (GPU)"
FACEOFF_MODEL[C]="/home/llm/models/Qwen3.5-0.8B-mtp-q6k.gguf"
FACEOFF_CK[C]="f16"
FACEOFF_CV[C]="f16"
FACEOFF_NGL[C]=99
FACEOFF_CTX[C]=2048

# D: 35B MTP CPU + 0.8B MTP GPU draft (cross-model spec decode)
FACEOFF_LABEL[D]="35B MTP + 0.8B MTP draft (CPU+GPU)"
FACEOFF_MODEL[D]="$MODEL"
FACEOFF_CK[D]="q8_0:q4_0"
FACEOFF_CV[D]="turbo_kv_4b"
FACEOFF_NGL[D]=0
FACEOFF_CTX[D]=2048

for tag in A B C D; do
    echo "  --- Config $tag: ${FACEOFF_LABEL[$tag]} ---"
    echo "  Starting server..."

    DRAFT_ARGS=""
    if [ "$tag" = "D" ]; then
        DRAFT_MODEL="/home/llm/models/Qwen3.5-0.8B-mtp-q6k.gguf"
        DRAFT_ARGS="-md $DRAFT_MODEL --ngld 99"
    fi

    "$BIN/llama-server" \
        -m "${FACEOFF_MODEL[$tag]}" \
        --cache-type-k "${FACEOFF_CK[$tag]}" --cache-type-v "${FACEOFF_CV[$tag]}" \
        --jinja -c "${FACEOFF_CTX[$tag]}" -ngl "${FACEOFF_NGL[$tag]}" \
        -t $THREADS --port $PORT $DRAFT_ARGS 2>/dev/null &
    SRV=$!

    READY=0
    for i in $(seq 1 120); do
        if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
            echo "  Server ready after ${i}s"
            READY=1
            break
        fi
        if ! kill -0 $SRV 2>/dev/null; then
            echo "  Server died during startup"
            break
        fi
        sleep 1
    done

    if [ "$READY" = "1" ]; then
        # Throughput: single long generation
        echo "  Running throughput test (n=64)..."
        TPUT=$(curl -s "http://localhost:$PORT/completion" \
            -d "{\"prompt\":\"Write a detailed essay about the history of computing.\",\"n_predict\":64,\"temperature\":0,\"seed\":42}" 2>/dev/null | \
            python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d.get(\"timings\",{}).get(\"predicted_per_second\",0):.2f} t/s')" 2>/dev/null || echo "ERROR")
        echo "  Throughput: $TPUT"
        echo "FACEOFF $tag ${FACEOFF_LABEL[$tag]}: $TPUT" >> "$RESULTS"

        # Tool-call battery
        echo "  Running tool-call battery..."
        BATTERY=$(python3 tests/tool_call_battery.py --port $PORT 2>&1 | tail -1)
        echo "  $BATTERY"
        echo "FACEOFF $tag tool_call: $BATTERY" >> "$RESULTS"

        kill $SRV 2>/dev/null || true
        wait $SRV 2>/dev/null || true
    else
        echo "  SKIP: server failed"
        echo "FACEOFF $tag: SKIP (server failed)" >> "$RESULTS"
    fi

    echo ""
    sleep 2
done

echo ""

# ------------------------------------------------------------------
# 5. CONTEXT SCALING — throughput as a function of context length
#    Measures max useful context window for each config.
# ------------------------------------------------------------------
echo "=== 5. CONTEXT SCALING (throughput vs context length) ==="
echo ""

# Fill prompt to force long context
FILL_PROMPT=$(python3 -c "print('The ' * 4000)")

SCALE_CONFIGS=(
    "f16/f16"
    "q8_0:q4_0/turbo_kv_4b"
)

for CTX in 512 1024 2048 4096 8192; do
    for conf in "${SCALE_CONFIGS[@]}"; do
        IFS='/' read -r ck cv <<< "$conf"
        printf "  c=%-5d K=%-20s V=%-15s → " "$CTX" "$ck" "$cv"

        # Use a prompt that fills the context, generate 32 tokens
        PROMPT_LEN=$((CTX - 64))
        FILL=$(python3 -c "print('The quick brown fox. ' * ($PROMPT_LEN // 5))")

        RESULT=$("$BIN/llama-completion" \
            -m "$MODEL" \
            -p "$FILL" -n 32 -c "$CTX" --simple-io -no-cnv \
            --cache-type-k "$ck" --cache-type-v "$cv" \
            --no-warmup -ngl 0 -t $THREADS --seed $SEED --temp 0 2>&1 | \
            grep "eval time" | grep -oP '(\d+\.\d+) ms per token' | head -1 || echo "FAILED")

        echo "$RESULT"
        echo "SCALE c=$CTX K=$ck V=$cv $RESULT" >> "$RESULTS"
    done
done

echo ""

# ------------------------------------------------------------------
# 6. OUTPUT LENGTH SCALING — generation time as a function of output
# ------------------------------------------------------------------
echo "=== 6. OUTPUT LENGTH SCALING ==="
echo ""

for N_PREDICT in 32 64 128 256 512; do
    for conf in "${SCALE_CONFIGS[@]}"; do
        IFS='/' read -r ck cv <<< "$conf"
        printf "  n=%-4d K=%-20s V=%-15s → " "$N_PREDICT" "$ck" "$cv"

        RESULT=$("$BIN/llama-completion" \
            -m "$MODEL" \
            -p "Write a very long and detailed essay about the complete history of computing from the abacus to artificial intelligence." \
            -n "$N_PREDICT" -c 4096 --simple-io -no-cnv \
            --cache-type-k "$ck" --cache-type-v "$cv" \
            --no-warmup -ngl 0 -t $THREADS --seed $SEED --temp 0 2>&1 | \
            grep "eval time" | grep -oP '(\d+\.\d+) ms per token.*?(\d+\.\d+) tokens per second' | head -1 || echo "FAILED")

        echo "$RESULT"
        echo "OUTPUT n=$N_PREDICT K=$ck V=$cv $RESULT" >> "$RESULTS"
    done
done

echo ""

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------
echo "============================================================"
echo "  Overnight run complete: $(date)"
echo "  Results: $RESULTS"
echo "============================================================"
echo ""
cat "$RESULTS"
