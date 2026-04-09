#!/usr/bin/env bash
# Wait for the entire overnight chain (default 64K bench, opt-in 64K
# bench, spec decode test, TQ perplexity, F16 baseline perplexity) to
# complete, then extract the key numbers from each .err file and write
# a single summary at tests/phase26_chain_summary.md.
#
# This is the file the next session should read first to see what
# happened overnight.

set -uo pipefail

cd /home/llm/rocm-polaris

LOG=tests/run_chain_summary.log
SUMMARY=tests/phase26_chain_summary.md

echo "[summary] starting at $(date)" > "$LOG"

# Wait for all .done markers to appear
for marker in tests/run_64k_chain.done tests/phase26_perplexity_f16baseline.done; do
    echo "[summary] waiting for $marker" >> "$LOG"
    while [[ ! -f "$marker" ]]; do
        sleep 30
    done
    echo "[summary] $marker appeared at $(date)" >> "$LOG"
done

# Cooldown
sleep 5

extract() {
    local label=$1
    local errfile=$2
    if [[ ! -f "$errfile" ]]; then
        echo "  $label: (file missing: $errfile)"
        return
    fi
    local pp=$(grep "prompt eval time" "$errfile" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+ tokens per second' | head -1)
    local gen=$(grep "eval time =.*runs" "$errfile" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+ tokens per second' | head -1)
    local kv=$(grep "KV buffer" "$errfile" 2>/dev/null | head -1 | sed 's/^.*: //')
    local rs=$(grep "RS buffer" "$errfile" 2>/dev/null | head -1 | sed 's/^.*: //')
    local ppl=$(grep -oE '\[1\].*\[2\]' "$errfile" 2>/dev/null | head -1)
    if [[ -z "$ppl" ]]; then
        ppl=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$errfile" 2>/dev/null | head -1)
    fi
    cat <<TBL
- **$label:**
  - PP: ${pp:-(none)}
  - Gen: ${gen:-(none)}
  - KV: ${kv:-(none)}
  - RS: ${rs:-(none)}
  - PPL: ${ppl:-(none)}
TBL
}

cat > "$SUMMARY" <<EOF
# Phase 26 overnight chain summary ($(date '+%Y-%m-%d %H:%M'))

Branch tip when chain started: \`7e657cc3b\` (KV mirror landed,
default OFF) plus follow-up commits \`5f7ba33fb\`,
\`583e6bedf\` (uncommitted to running binary, but on disk for the
chained opt-in bench).

## 64K production bench (default config — KV on regular CPU buft)
$(extract "default" tests/phase26_64k_default.err)

## 64K production bench (opt-in KV mirror — LLAMA_NUMA_MIRROR_KV=1)
$(extract "opt-in" tests/phase26_64k_optin.err)

## Spec decode end-to-end (Phase 26 #4a validation)
$(if [[ -f tests/phase26_spec_decode.err ]]; then
    grep -E "SPEC PASS|SPEC FAIL|partial sequence removal" tests/phase26_spec_decode.err 2>/dev/null | head -3
    if [[ -f tests/phase26_spec_decode.out ]]; then
        echo
        echo "Sample response:"
        tail -5 tests/phase26_spec_decode.out
    fi
else
    echo "(file missing)"
fi)

## Perplexity (TQ_KV_1B + TQ_V_4B vs F16 baseline)
$(extract "TQ" tests/phase26_perplexity.err)
$(extract "F16 baseline" tests/phase26_perplexity_f16baseline.err)

## Status of .done markers
$(for d in tests/phase26_64k_optin.done tests/phase26_spec_decode.done tests/phase26_perplexity.done tests/phase26_perplexity_f16baseline.done tests/run_64k_chain.done; do
    if [[ -f "$d" ]]; then
        echo "  - $d: $(cat "$d")"
    else
        echo "  - $d: MISSING"
    fi
done)

## Next steps for the next session

1. Read tests/phase26_64k_default.err and tests/phase26_64k_optin.err
   to confirm the numbers above. Update PHASE26.md "Current standing"
   block with the 64K production decode rate.
2. If TQ PPL is within 1% of F16 PPL, mark Phase 26 #2 (quality
   validation) DONE. If not, escalate.
3. If spec decode test passed, the compat-check patch is validated
   and Phase 26 #4a is DONE. Decide whether to also re-convert the
   GGUF for MTP support (#4b).
4. Otherwise: investigate why the chain step failed by reading the
   corresponding .err file.
EOF

echo "[summary] wrote $SUMMARY at $(date)" >> "$LOG"
echo "EXIT=0" > tests/phase26_chain_summary.done
