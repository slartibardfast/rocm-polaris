#!/bin/bash
# run_acceptance.sh — Full acceptance test suite for rocm-polaris
#
# Runs all test tiers including cross-process isolation.
# Each test binary is a separate process invocation.
# Checks dmesg for GPU faults after all tests.
#
# Usage: ./run_acceptance.sh [RUNS=5]
#
# Exit code: 0 = all pass, 1 = failures detected

set -euo pipefail

RUNS=${1:-5}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PASS=0
FAIL=0
TOTAL=0

run_test() {
    local name="$1"
    local cmd="$2"
    local timeout_s="${3:-120}"
    TOTAL=$((TOTAL + 1))
    printf "%-50s " "$name"
    local t0=$(date +%s%N)
    if output=$(timeout "$timeout_s" bash -c "$cmd" 2>&1); then
        local t1=$(date +%s%N)
        local ms=$(( (t1 - t0) / 1000000 ))
        echo "PASS (${ms}ms)"
        PASS=$((PASS + 1))
    else
        local exit_code=$?
        local t1=$(date +%s%N)
        local ms=$(( (t1 - t0) / 1000000 ))
        if [ $exit_code -eq 124 ]; then
            echo "TIMEOUT (${timeout_s}s)"
        else
            echo "FAIL (exit=$exit_code, ${ms}ms)"
        fi
        # Show first failure line
        echo "$output" | grep -m1 'FAIL' | sed 's/^/  /'
        FAIL=$((FAIL + 1))
    fi
}

echo "=========================================="
echo "  rocm-polaris acceptance test suite"
echo "  $(date)"
echo "  Kernel: $(uname -r)"
echo "  Runs per cross-process test: $RUNS"
echo "=========================================="
echo

# Rebuild all test binaries
echo "Building tests..."
make -j$(nproc) 2>&1 | tail -1
hipcc --offload-arch=gfx803 -w -o test_gpu_isolation test_gpu_isolation.cpp 2>/dev/null
hipcc --offload-arch=gfx803 -w -o test_kernarg_stability test_kernarg_stability.cpp 2>/dev/null
hipcc --offload-arch=gfx803 -w -o test_alloc_free_race test_alloc_free_race.cpp 2>/dev/null
hipcc --offload-arch=gfx803 -w -o test_mixed_stress test_mixed_stress.cpp 2>/dev/null
hipcc --offload-arch=gfx803 -w -o hip_inference_test hip_inference_test.cpp 2>/dev/null
echo

# ============================================================
echo "=== Tier 1: Regression ==="
# ============================================================
run_test "1.1 H2D kernel (17 subtests)" "./test_h2d_kernel"
run_test "1.2 Memset verify" "./test_memset_verify" 30
echo

# ============================================================
echo "=== Tier 2: Resource isolation ==="
# ============================================================
run_test "2.1 VA reuse + staging + signals + mixed" "./test_gpu_isolation" 600
echo

# ============================================================
echo "=== Tier 3: Kernarg integrity ==="
# ============================================================
run_test "3.1 Kernarg stability (9 subtests)" "./test_kernarg_stability" 300
run_test "3.2 Alloc/free race diagnostic" "./test_alloc_free_race" 60
run_test "3.3 Mixed stress (500 ops)" "./test_mixed_stress" 120
echo

# ============================================================
echo "=== Tier 4: Barrier handling ==="
# ============================================================
run_test "4.1 Barrier test suite (15 subtests)" "./hip_barrier_test" 900
echo

# ============================================================
echo "=== Tier 5: Inference patterns ==="
# ============================================================
run_test "5.1 Inference test suite (9 subtests)" "./hip_inference_test" 600
echo

# ============================================================
echo "=== Tier 6: Cross-process isolation ==="
# ============================================================
echo "Running each test $RUNS times as separate processes..."
cross_pass=0
cross_fail=0
for test_bin in test_kernarg_stability test_gpu_isolation; do
    for run in $(seq 1 "$RUNS"); do
        name="6.x $test_bin run $run/$RUNS"
        run_test "$name" "./$test_bin" 120
    done
done
echo

# ============================================================
echo "=== Tier 7: dmesg audit ==="
# ============================================================
TOTAL=$((TOTAL + 1))
printf "%-50s " "7.1 Zero GPU faults in dmesg"
fault_count=$(sudo dmesg 2>/dev/null | grep -c -i 'fault VA\|protection_fault' || true)
segfault_count=$(sudo dmesg 2>/dev/null | grep -c 'segfault.*libhsa\|segfault.*libamdhip' || true)
if [ "$fault_count" -eq 0 ] && [ "$segfault_count" -eq 0 ]; then
    echo "PASS (0 faults)"
    PASS=$((PASS + 1))
else
    echo "FAIL ($fault_count GPU faults, $segfault_count ROCR segfaults)"
    sudo dmesg | grep -i 'fault VA\|protection_fault\|segfault.*libhsa\|segfault.*libamdhip' | tail -5 | sed 's/^/  /'
    FAIL=$((FAIL + 1))
fi
echo

# ============================================================
echo "=========================================="
echo "  RESULTS: $PASS/$TOTAL PASS, $FAIL FAIL"
echo "=========================================="

exit $((FAIL > 0 ? 1 : 0))
