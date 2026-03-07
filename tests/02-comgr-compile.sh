#!/bin/bash
# Verify comgr/hipcc can compile a trivial kernel for gfx803.
# Requires: rocm-llvm, comgr, hip-runtime-amd
set -euo pipefail

HIPCC=/opt/rocm/bin/hipcc

if [ ! -x "$HIPCC" ]; then
  echo "SKIP: $HIPCC not found (install hip-runtime-amd)"
  exit 2
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

cat > "$tmpdir/test.cpp" << 'KERNEL'
#include <hip/hip_runtime.h>
__global__ void null_kernel() {}
int main() { return 0; }
KERNEL

for gfx in gfx801 gfx803; do
  if "$HIPCC" --offload-arch=$gfx -c "$tmpdir/test.cpp" -o "$tmpdir/test_${gfx}.o" 2>&1; then
    echo "PASS: hipcc compiled for $gfx"
  else
    echo "FAIL: hipcc failed to compile for $gfx"
    exit 1
  fi
done
