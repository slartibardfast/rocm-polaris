# Phase 6: Signal Completion on No-Atomics Platforms

## Status: IN PROGRESS

## Problem Statement

On platforms without PCIe AtomicOps (Westmere Xeon + Polaris GPU), the GPU Command Processor (CP) cannot decrement AQL packet completion signals. The AtomicDec TLP is silently dropped by the root complex. Every other GPU operation (packet fetch, kernel execution, DMA, RPTR writes) works correctly.

This single broken primitive — **GPU → system memory atomic write** — breaks the entire ROCm signal completion chain.

## Layer-by-Layer Analysis

### Layer 1: Hardware (gfx8 CP)
- CP processes AQL packets correctly
- CP writes RPTR to `rptr_report_addr` via PCIe posted writes (WORKS)
- CP attempts signal AtomicDec via PCIe AtomicOp TLP (DROPPED)
- CP evaluates barrier dep_signals by reading from system memory (WORKS — signals are coherent)
- **CP goes idle after evaluating unsatisfied barrier dep with SLOT_BASED_WPTR=0**
- SLOT_BASED_WPTR=2 (memory polling) dead — CP cannot GPUVM-read poll address without ATC/UTCL2

### Layer 2: Kernel (KFD)
- MQD setup: SLOT_BASED_WPTR=0, NO_UPDATE_RPTR=0, RPTR_BLOCK_SIZE=4 for no-atomics
- Queue creation, doorbell mapping, RPTR buffer — all correct
- No kernel-level fix possible for the signal completion issue

### Layer 3: ROCR (libhsa-runtime64)
- **Bounce buffer** (patch 0004): monitors RPTR advancement, decrements signals from CPU
- **RPTR tracking** (patch 0003): GPU-visible RPTR buffer, dword→dispatch_id conversion
- **BlitKernel** creates AQL barrier packets from dep_signals — these stall the idle CP
- NOP kick hack wakes idle CP but accumulates write_dispatch_id drift

### Layer 4: CLR (libamdhip64)
- `WaitingSignal()` creates inter-operation barrier deps — stall the idle CP
- `rocrCopyBuffer()` passes dep_signals to BlitKernel — creates more barriers
- `cpu_wait_for_signal_` flag: CPU-waits instead of barrier deps (eliminates CLR-level barriers)
- H2D staging loop: WaitCurrent between chunks eliminates inter-chunk barriers

### Layer 5: Application
- hipMemcpy, hipLaunchKernel — no changes needed
- llama.cpp model loading works, inference hangs during graph compute

## Current Workaround Stack

| Fix | Layer | Purpose | Limitation |
|-----|-------|---------|------------|
| RPTR_BLOCK_SIZE=4 | Kernel | Per-packet RPTR writes | None |
| SpinMutex + mfence | ROCR | Thread-safe UpdateReadDispatchId | None |
| Retain/Release | ROCR | Signal use-after-free prevention | Crashes without NOP kick (latent bug) |
| NOP barrier kick | ROCR | Wakes idle CP | Drifts at ~286 ops |
| Fine-grain staging | CLR | D2H data integrity | None |
| clflush on buffers | CLR | CPU cache coherency for D2H | None |
| WaitCurrent H2D chunks | CLR | Eliminates H2D inter-chunk barriers | H2D only |
| cpu_wait_for_signal | CLR | Eliminates CLR inter-op barriers | Doesn't cover ROCR barriers |
| BlitKernel CPU-wait | ROCR | Eliminates ROCR blit barriers | Blit path only |

## What's Wrong With This Stack

1. **Too many layers.** 9 fixes across 3 layers. Stacked workarounds interact unpredictably.
2. **NOP kick masks a crash.** Removing it causes segfault. We don't know why.
3. **Barrier elimination is incomplete.** CLR + ROCR blit barriers eliminated, but compute dispatch barriers may still exist.
4. **NOP drift limits operation count.** ~286 ops before write_dispatch_id accumulation hangs.

## Correct Fix (Proposed)

The ONLY broken primitive is **signal AtomicDec**. The correct fix is to make the bounce buffer handle ALL signal completions reliably, WITHOUT injecting packets or modifying barrier behavior.

### Step 1: Understand the segfault
Strip the bounce buffer to minimum: just RPTR→dispatch_id conversion + signal SubRelaxed. No Retain/Release, no clflush, no NOP kicks. Find and fix why this crashes.

### Step 2: Add Retain/Release correctly
Once the base works, add signal lifecycle protection. Verify no crash.

### Step 3: Remove all barrier workarounds
If the bounce buffer works reliably, barriers should resolve naturally (CP re-evaluates on its own). The barrier workarounds (cpu_wait_for_signal, WaitCurrent, BlitKernel CPU-wait) may be unnecessary.

### Step 4: Verify at scale
Run llama.cpp inference end-to-end. If barrier stalls return, add ONLY the minimum necessary workaround.

## Success Criteria
- llama.cpp GPU inference produces correct tokens
- No hangs at any operation count
- No segfaults or signal lifecycle issues
- Maximum 3 patches total (kernel + ROCR + CLR), no PKGBUILD sed hacks
