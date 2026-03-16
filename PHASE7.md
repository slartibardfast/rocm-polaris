# Phase 7: llama.cpp GPU Inference

## Status: IN PROGRESS — compute queue WORKS on clean boot, llama.cpp next

## Prerequisite

Phase 6 (Signal Completion) is COMPLETE. All GPU operations work: D2H, H2D, kernel dispatch, mixed kernel+memcpy. The interrupt-based bounce buffer handles signal completion correctly on no-atomics platforms.

## Problem Statement

llama.cpp with `-ngl 1` (1 GPU layer) loads the model successfully but hangs during process exit. CPU-only (`-ngl 0`) works and enters interactive mode. The hang is NOT in signal completion — it's in ggml-hip's interaction with CLR during cleanup.

### GDB Analysis

```
#0  sched_yield ()                           ← spinning
#1  libamdhip64.so (CLR yield loop)          ← AcquireWriteIndex or WaitCurrent
#2  libamdhip64.so (CLR queue operation)
#3  libamdhip64.so (CLR device init)
#4  libamdhip64.so (CLR lazy init)
#5  libc.so.6 (__cxa_finalize)               ← process exit path
#6  pthread_once ()                           ← lazy init guard
#7  libamdhip64.so (CLR init entry)
#8  libggml-hip.so.0 (ggml-hip destructor)   ← static destructor
#9  libc.so.6 (__cxa_finalize)               ← atexit cleanup
```

The ggml-hip library has a static destructor that calls into CLR during process exit. CLR's lazy initialization (via `pthread_once`) tries to perform a GPU operation, which gets stuck in a yield loop.

## Hypotheses

**H1: Inference never starts.** The model loads (blit queue works), but the first compute dispatch triggers CLR lazy init for the compute queue, which gets stuck. The `__cxa_finalize` + `pthread_once` in the backtrace is CLR's lazy init, not process exit.

**H2: Inference runs but hangs.** The model loads, inference starts, but a GPU kernel completion signal doesn't get decremented (bounce buffer issue on the compute queue).

**H3: Inference completes, exit hangs.** Everything works, but ggml-hip's static destructor triggers a stuck CLR operation during cleanup.

**H1 is most likely** because: (a) no tokens appear in output, (b) CPU is at 100% in a yield loop (not 0% as a signal wait would show), (c) `pthread_once` is a one-time init guard.

## Investigation Plan

### Step 1: Confirm which hypothesis (5 min)

Run llama.cpp in background, write output to file, sample GDB at 10s and 30s. If the backtrace changes between samples → inference is progressing. If identical → stuck at same point.

Also check: does `hipDeviceSynchronize` work standalone after model load? Write a tiny HIP program that does `hipMalloc + hipMemcpy + kernel + hipDeviceSynchronize` with the same ROCR/CLR stack.

### Step 2: Decode the CLR yield loop (10 min)

Install `hip-runtime-amd-polaris-debug` package. Re-attach GDB with debug symbols to decode the exact CLR function in the yield loop. Distinguish between:
- `AcquireWriteIndex` (queue full)
- `WaitCurrent` / signal wait
- `releaseGpuMemoryFence` (fence wait)
- Device init / `pthread_once` (first-use init)

### Step 3: Apply the simplest fix (15 min)

Based on findings:

| If | Then | How |
|----|------|-----|
| Queue full (H1) | The NOP kick drifted `write_dispatch_id` | Remove NOP kick, rely purely on interrupts |
| Signal wait (H2) | Bounce buffer not running for compute queue | Verify `InterruptSignal` used for compute signals |
| Init stuck (H1) | CLR compute queue creation hangs | Check if blit queue's init state blocks compute queue init |
| Exit cleanup (H3) | ggml-hip destructor calls GPU ops | Patch ggml-hip to skip cleanup, or use `_exit()` |

### Step 4: Verify with token generation (5 min)

After fix, run `llama-cli -ngl 1 -p "2+2=" -n 8` and verify tokens appear. Measure t/s. Compare to CPU baseline (13.3 t/s).

## Fix Options

| Option | Description | Risk |
|--------|-------------|------|
| A: Remove NOP kick entirely | With interrupts working, NOP kick may be unnecessary and harmful | Low — all tests passed without NOP kick before the segfault |
| B: Patch ggml-hip destructor | Skip GPU cleanup on process exit | Low — OS reclaims everything |
| C: Fix CLR init ordering | Ensure compute queue init doesn't deadlock with blit queue | Medium |
| D: Add `ProcessAllBounceBuffers` to CLR yield loops | The yield loop doesn't call the bounce buffer | Medium — broad change |

## Success Criteria

- `llama-cli -ngl 1 -p "Hello" -n 8` produces 8 tokens on GPU
- Process exits cleanly (no hang in destructor)
- Token generation rate measured (compare to CPU-only 13.3 t/s baseline)
- No regressions in D2H, H2D, stress, or mixed tests

## Dependencies

- Phase 6 COMPLETE ✅
- `llama-cpp-rocm-polaris` b7376-1 installed ✅
- `hsa-rocr-polaris` 7.2.0-9 with interrupt path ✅
- `hip-runtime-amd-polaris` 7.2.0-2 with CLR fixes ✅
