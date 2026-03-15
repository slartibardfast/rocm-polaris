# Phase 7: llama.cpp GPU Inference

## Status: NOT STARTED

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

## Investigation Plan

### Step 1: Determine if inference actually runs

The CPU-only test showed the interactive prompt appeared. With `-ngl 1`, the model loads (spinner completes) but no tokens are generated. Question: does inference START before the hang, or does the hang happen BEFORE inference?

**Test:** Use `-v` (verbose) to see timing messages. Use `AMD_LOG_LEVEL=3` for CLR-level tracing. Redirect output to file to avoid pipe blocking.

### Step 2: Identify the ggml-hip static destructor

**Test:** `nm -DC /usr/lib/libggml-hip.so.0 | grep -i "destructor\|__cxa_atexit\|fini"` to find what's registered.

### Step 3: Determine if the yield loop is queue-space or signal-wait

The `sched_yield` in CLR could be:
- `AcquireWriteIndex` busy-wait (queue full — `write_dispatch_id - read_dispatch_id >= queue_size`)
- `WaitCurrent` (waiting for signal completion — bounce buffer issue)
- `releaseGpuMemoryFence` (waiting for GPU fence during cleanup)

**Test:** GDB with CLR debug symbols, or decode the libamdhip64 offsets.

### Step 4: Fix options

| Option | Description | Risk |
|--------|-------------|------|
| A: Skip GPU cleanup on no-atomics | ggml-hip destructor checks platform, skips GPU calls | Low — cleanup isn't critical if process is exiting |
| B: Force CLR init before ggml-hip | Ensure CLR is fully initialized during ggml-hip module load | Medium — load order dependencies |
| C: Fix CLR lazy init to handle exit | Make CLR's pthread_once aware of process exit state | Medium — CLR code change |
| D: Rebuild llama-cpp-rocm-polaris with fix | Patch ggml-hip or link order | Low — our package, our control |

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
