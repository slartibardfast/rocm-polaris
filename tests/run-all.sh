#!/bin/bash
# Run all validation tests in order.
# Tests exit 0=PASS, 1=FAIL, 2=SKIP
set -uo pipefail
dir=$(dirname "$0")
total=0 pass=0 fail=0 skip=0

for test in "$dir"/[0-9]*.sh; do
  name=$(basename "$test" .sh)
  echo "=== $name ==="
  total=$((total + 1))
  bash "$test"
  rc=$?
  case $rc in
    0) pass=$((pass + 1)) ;;
    2) skip=$((skip + 1)) ;;
    *) fail=$((fail + 1)) ;;
  esac
  echo
done

echo "--- Results: $pass pass, $fail fail, $skip skip / $total total ---"
[ $fail -eq 0 ]
