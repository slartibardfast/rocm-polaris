#!/usr/bin/env bash
# Build a realistic OpenClaw session fixture by concatenating:
#   - agent system prompt (constructed, ~2K tokens)
#   - real C++ source (ggml-cpu ops.cpp, ~5K lines / 40K tokens)
#   - real Python script from llama.cpp/scripts
#   - a section of the CHANGELOG / documentation
#   - simulated multi-turn user/assistant messages
#
# Output: tests/openclaw_<target>k_fill.txt
#
# Usage: ./make_openclaw_fixture.sh <target_tokens>  (e.g. 16000, 32000, 48000)

set -euo pipefail

TARGET="${1:?Usage: $0 <target_tokens>}"
OUT="/home/llm/rocm-polaris/tests/openclaw_${TARGET}.txt"
LLAMA_SRC="/home/llm/rocm-polaris/llama.cpp/src/llama.cpp-b8508"

# Realistic OpenClaw session anatomy (approx bytes, tuned for ~4 bytes/token on
# code, ~4.4 bytes/token on prose):
#   25% system/setup/docs
#   50% code files (C/C++/Python mixed)
#   25% simulated conversation + tool outputs

TARGET_BYTES=$(( TARGET * 43 / 10 ))   # ~4.3 bytes/token average

SYS_PROMPT=$(cat << 'SYS_EOF'
You are OpenClaw, an autonomous coding agent with access to the llama.cpp
ggml codebase. You have file read/write/search, shell execution, and git
tools available. Always prefer surgical edits over rewrites. When asked
to debug, form a hypothesis, write a test that can falsify it, run the
test, then act on the result. Never introduce malloc in a hot loop. If
you find scalar fallbacks on a target that supports SIMD, flag them as
regressions rather than working around them. Patches should minimize
the blast radius: change only what the task requires.

The user is David, a systems engineer working on ROCm support for
Polaris-era AMD GPUs on Westmere-class dual-Xeon hosts. Assume:
  - Target hardware lacks AVX/AVX2/F16C but has SSE4.1/SSE4.2/POPCNT.
  - The KV cache is memory-bandwidth bound; compressed formats help.
  - Build system is cmake; the package is llama-cpp-rocm-polaris.
  - Performance targets are measured in sustained tokens/second
    at realistic context fill (OpenClaw sessions are 8-48K tokens).
SYS_EOF
)

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

{
    printf '=== SYSTEM ===\n%s\n\n' "$SYS_PROMPT"
    printf '=== USER (turn 1) ===\n'
    printf 'Please audit ggml/src/ggml-cpu/ops.cpp for any hot-loop\n'
    printf 'allocations and paths that fall through to scalar generic\n'
    printf 'implementations on Westmere-class x86 hosts. Start with\n'
    printf 'the flash attention and quantized matmul paths.\n\n'
    printf '=== ASSISTANT (turn 1) ===\n'
    printf "I'll start by reading ops.cpp to get oriented, then trace\n"
    printf 'the flash attention and matmul kernels.\n\n'
    printf '<tool_call>Read</tool_call>\n'
    printf '=== TOOL OUTPUT ===\n'
    printf 'File: ggml/src/ggml-cpu/ops.cpp (first 2500 lines)\n\n'
    head -n 2500 "$LLAMA_SRC/ggml/src/ggml-cpu/ops.cpp" 2>/dev/null || true
    printf '\n=== ASSISTANT (turn 2) ===\n'
    printf 'Now let me check the quantized matmul path in arch/x86/quants.c\n'
    printf 'for SSSE3/SSE4.1 coverage of the K-quant formats.\n\n'
    printf '<tool_call>Read</tool_call>\n'
    printf '=== TOOL OUTPUT ===\n'
    printf 'File: ggml/src/ggml-cpu/arch/x86/quants.c (first 1500 lines)\n\n'
    head -n 1500 "$LLAMA_SRC/ggml/src/ggml-cpu/arch/x86/quants.c" 2>/dev/null || true
    printf '\n=== USER (turn 2) ===\n'
    printf 'Also grep for malloc and alloca in the ggml-cpu directory\n'
    printf 'to make sure no hot paths are allocating per-call.\n\n'
    printf '=== TOOL OUTPUT ===\n'
    printf 'Grep results for "malloc\\|alloca" in ggml-cpu/:\n'
    grep -rn 'malloc\|alloca' "$LLAMA_SRC/ggml/src/ggml-cpu/" 2>/dev/null | head -200 || true
    printf '\n=== ASSISTANT (turn 3) ===\n'
    printf 'Let me also pull in the CMakeLists for ggml-cpu to see how\n'
    printf 'SSE4.2, F16C, and repack are gated at build time.\n\n'
    printf '=== TOOL OUTPUT ===\n'
    cat "$LLAMA_SRC/ggml/src/ggml-cpu/CMakeLists.txt" 2>/dev/null || true
    # Pad with additional source if we're still under target
    printf '\n=== USER (turn 3) ===\n'
    printf 'Please also show the llama-kv-cache.cpp handling of the\n'
    printf 'flash_attn_ext call so I can understand the read path.\n\n'
    printf '=== TOOL OUTPUT ===\n'
    head -n 800 "$LLAMA_SRC/src/llama-kv-cache.cpp" 2>/dev/null || true
    printf '\n=== USER (turn 4) ===\n'
    printf 'Finally, pull in wikitext-2 to pad our context to a realistic\n'
    printf 'session length for the benchmark.\n\n'
    printf '=== TOOL OUTPUT ===\n'
    cat "$LLAMA_SRC/../../../tests/wikitext-2-test.txt" 2>/dev/null || cat /home/llm/rocm-polaris/tests/wikitext-2-test.txt 2>/dev/null || true
} > "$TMPDIR/full.txt"

# Truncate to target byte count
head -c "$TARGET_BYTES" "$TMPDIR/full.txt" > "$OUT"
wc -c "$OUT"
echo "Wrote $OUT (target $TARGET tokens, approx $(wc -c < "$OUT") bytes)"
