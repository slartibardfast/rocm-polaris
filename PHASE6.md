# Phase 6: Signal Completion on No-Atomics Platforms

## Status: COMPLETE

## Problem Statement

On platforms without PCIe AtomicOps (Westmere Xeon + Polaris GPU), the GPU Command Processor (CP) cannot decrement AQL packet completion signals. The AtomicDec TLP is silently dropped by the root complex. Every other GPU operation (packet fetch, kernel execution, DMA, RPTR writes) works correctly.

This single broken primitive — **GPU → system memory atomic write** — breaks the entire ROCm signal completion chain.

## Key Discovery: The Interrupt Path Works

The CP firmware does THREE things when completing an AQL packet:

1. **AtomicDec on `signal_.value`** — PCIe AtomicOp → **DROPPED** on Westmere
2. **Write `event_id` to `event_mailbox_ptr`** — regular PCIe MWr → **WORKS**
3. **`s_sendmsg` interrupt** — MSI → **WORKS**

Steps 2 and 3 are regular PCIe operations that work without AtomicOps. The interrupt fires even when the atomic fails. Verified: 68-100 interrupts per dispatch operation on our hardware (`/proc/interrupts` delta).

**Prior art:** The SDMA no-atomics path (`amd_blit_sdma.cpp:548-579`) proves the pattern: replace atomic with fence write + mailbox write + trap. The DRM compute ring (`gfx_v8_0_ring_emit_fence_compute`) uses RELEASE_MEM (regular write) + optional MSI, never PCIe atomics. Intel GPUs, RDMA NICs, and NVMe all use the same pattern: device writes value → MSI interrupt → CPU reads value.

## Solution: Interrupt-Based Bounce Buffer

1. **Re-enable `g_use_interrupt_wait = true`** — our patch 0004 incorrectly forced it false
2. **Add `ProcessAllBounceBuffers()` to `InterruptSignal::WaitRelaxed`** — when the interrupt wakes `hsaKmtWaitOnEvent`, the bounce buffer reads RPTR (already advanced by PCIe ordering guarantee), converts to dispatch_id, and decrements the signal from CPU
3. **Re-read `signal_.value` after bounce buffer call** — critical: the bounce buffer modifies the value, so we must re-read before checking the condition
4. **Keep BlitKernel CPU-wait for dep_signals** — barrier packets still stall the idle CP; CPU-waiting eliminates them

## Changes Applied

| Change | File | Method |
|--------|------|--------|
| Remove `g_use_interrupt_wait = false` | `runtime.cpp` | PKGBUILD sed |
| Add `ProcessAllBounceBuffers()` + re-read to `InterruptSignal::WaitRelaxed` | `interrupt_signal.cpp` | PKGBUILD sed |
| Add `#include "core/inc/amd_aql_queue.h"` | `interrupt_signal.cpp` | PKGBUILD sed |
| CPU-wait dep_signals in `BlitKernel::SubmitLinearCopyCommand` | `amd_blit_kernel.cpp` | PKGBUILD sed |
| CPU-wait in CLR `WaitingSignal()` for no-atomics | `rocvirtual.cpp` | PKGBUILD sed |
| WaitCurrent between H2D staging chunks | `rocblit.cpp` | Patch 0001 |

## Test Results (all WITHOUT `HSA_ENABLE_INTERRUPT=0`)

| Test | Result |
|------|--------|
| `test_interrupt_signal` (4 subtests) | 4/4 PASS |
| `event_mailbox_ptr` non-zero | 0x208020 ✅ |
| HSA barrier dispatch | PASS |
| D2H sweep (13 sizes) | 13/13 PASS |
| Long stress (550 ops, 11 sizes x 50 rounds) | 550/550 PASS |
| Mixed kernel + memcpy (200 ops) | 200/200 PASS |

## What Was Eliminated

The interrupt approach made several earlier workarounds unnecessary or defense-in-depth:

- ~~`HSA_ENABLE_INTERRUPT=0`~~ — no longer needed (interrupts work!)
- NOP barrier kick — still present in patch 0004 as defense-in-depth, but the interrupt handles the primary signal completion path
- `ROC_CPU_WAIT_FOR_SIGNAL` env var — not needed (CLR PKGBUILD injection handles it)

## Packages

| Package | Version | Key changes |
|---------|---------|-------------|
| `linux-lts-rocm-polaris` | 6.18.16-10 | RPTR_BLOCK_SIZE=4, SLOT_BASED_WPTR=0, NO_UPDATE_RPTR=0 |
| `hsa-rocr-polaris` | 7.2.0-9 | Patches 0001-0005 + PKGBUILD sed (interrupt re-enable, BlitKernel CPU-wait) |
| `hip-runtime-amd-polaris` | 7.2.0-2 | Patch 0001 (fine-grain staging, clflush, H2D WaitCurrent) + PKGBUILD sed (cpu_wait_for_signal) |

## Remaining Issue → Phase 7

llama.cpp model loads successfully but hangs during process exit. GDB shows ggml-hip's static destructor triggers CLR lazy init via `pthread_once` which gets stuck in a `sched_yield` loop. This is a ggml-hip/CLR interaction bug during cleanup — NOT a signal completion issue. All GPU operations complete correctly.
