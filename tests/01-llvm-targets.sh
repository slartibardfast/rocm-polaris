#!/bin/bash
# Verify Arch's rocm-llvm includes gfx801/802/803 targets.
# Requires: rocm-llvm
set -euo pipefail

LLC=/opt/rocm/lib/llvm/bin/llc

if [ ! -x "$LLC" ]; then
  echo "SKIP: $LLC not found (install rocm-llvm)"
  exit 2
fi

targets=$("$LLC" --version 2>&1)
fail=0
for gfx in gfx801 gfx802 gfx803; do
  if echo "$targets" | grep -q "$gfx"; then
    echo "PASS: $gfx present in llc targets"
  else
    echo "FAIL: $gfx missing from llc targets"
    fail=1
  fi
done
exit $fail
