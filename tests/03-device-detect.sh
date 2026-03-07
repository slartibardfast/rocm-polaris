#!/bin/bash
# Verify rocminfo detects a gfx8xx device.
# Requires: rocminfo, hardware present
set -euo pipefail

ROCMINFO=/opt/rocm/bin/rocminfo

if [ ! -x "$ROCMINFO" ]; then
  echo "SKIP: $ROCMINFO not found (install rocminfo)"
  exit 2
fi

output=$("$ROCMINFO" 2>&1) || true

if echo "$output" | grep -qi "gfx80[1-3]"; then
  gfx=$(echo "$output" | grep -oi "gfx80[1-3]" | head -1)
  echo "PASS: detected $gfx device"
  echo "$output" | grep -A2 "Marketing Name"
else
  echo "FAIL: no gfx801/802/803 device detected"
  echo "Available agents:"
  echo "$output" | grep "Name:" | head -10
  exit 1
fi
