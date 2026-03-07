#!/bin/bash
# Build and run a HIP device query for gfx803.
# Requires: hip-runtime-amd, hardware present
set -euo pipefail

HIPCC=/opt/rocm/bin/hipcc

if [ ! -x "$HIPCC" ]; then
  echo "SKIP: $HIPCC not found (install hip-runtime-amd)"
  exit 2
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

cat > "$tmpdir/devquery.cpp" << 'SRC'
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count == 0) {
        printf("FAIL: no HIP devices found (err=%d)\n", err);
        return 1;
    }
    int pass = 0;
    for (int i = 0; i < count; i++) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);
        printf("Device %d: %s (gfx%x%x%x)\n", i, props.name,
               props.gcnArchMajor, props.gcnArchMinor, props.gcnArchPatch);
        if (props.gcnArchMajor == 8)
            pass = 1;
    }
    if (pass)
        printf("PASS: gfx8 device found\n");
    else
        printf("FAIL: no gfx8 device found\n");
    return !pass;
}
SRC

"$HIPCC" "$tmpdir/devquery.cpp" -o "$tmpdir/devquery" 2>&1
"$tmpdir/devquery"
