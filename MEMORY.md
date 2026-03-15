# MEMORY.md — Lessons Learned and Key Decisions

Append-only. Do not delete or rewrite old entries.

---

## 2026-03-07: Project inception
- Target: ROCm 7.2.0 on Arch Linux for gfx801/802/803 (GCN 1.2)
- Test hardware: Radeon Pro WX 2100 (Polaris 12, gfx803, 2GB GDDR5)
- Host: dual Xeon X5650 (Westmere) — no PCIe atomics, no ATC/IOMMU
- Community prior art: xuhuisheng/rocm-gfx803 (ROCm 5.3), NULL0xFF (ROCm 6.1.5 on EPYC)

## 2026-03-07: ROCR DoorbellType check is the first gate
- `amd_gpu_agent.cpp:124` rejects Polaris (DoorbellType 1, not 2). Single-line patch.

## 2026-03-08: KFD AQL queue ring buffer size mismatch
- `kfd_queue.c:250` halves `expected_queue_size` for AQL on GFX7/8, but ROCR allocates full-size BO. Fix: remove halving.

## 2026-03-09: MQD comparison VI vs V9 — missing AQL bits
- VI's `kfd_mqd_manager_vi.c` doesn't set DOORBELL_BIF_DROP or UNORD_DISPATCH for AQL queues. V9 does.

## 2026-03-11: WPTR polling broken on no-atomics gfx8
- SLOT_BASED_WPTR=2 polls a CPU heap VA via GPUVM — unreachable without UTCL2/ATC
- Fix: SLOT_BASED_WPTR=0 (direct doorbell). Proven by DRM gfx8 compute queues.

## 2026-03-11: NO_UPDATE_RPTR incompatible with SLOT_BASED_WPTR=0
- With both flags, CP doesn't write RPTR at all. Clear NO_UPDATE_RPTR for direct doorbell mode.

## 2026-03-11: Phase 3c verified — read_dispatch_id advances via GPU-visible RPTR buffer
- rptr_gpu_buf_ from kernarg pool (system_allocator). CP writes dword RPTR to it.

## 2026-03-11: Bounce buffer completion_signal offset is 56, not 24
- All AQL packet types have completion_signal at byte offset 56.

## 2026-03-12: CPU cache coherency — GPU PCIe writes bypass CPU cache
- On Westmere without ATC, GPU DMA writes go to DRAM, not CPU cache. Need clflush after GPU writes.
- Fine-grain staging (MTYPE=CC) makes GPU L2 flush correctly on system-scope release.

## 2026-03-13: Race in UpdateReadDispatchId — SpinMutex + mfence
- Concurrent callers (CLR dispatch, signal polling, scratch reclaim) race on last_rptr_dwords_.
- Fix: SpinMutex serialization + _mm_mfence() for clflush ordering + atomic::Add for dispatch ID.

## 2026-03-13: RPTR_BLOCK_SIZE=5 causes stale RPTR for single-packet dispatches
- CP writes RPTR every 2^5=32 dwords (2 AQL packets). Single-packet blit may not trigger write.
- Fix: RPTR_BLOCK_SIZE=4 (every 16 dwords = 1 AQL packet) for no-atomics path.

## 2026-03-14: Signal coherency theory DISPROVEN
- Signals ARE coherent (fine-grain pool, KFD_IOC_ALLOC_MEM_FLAGS_COHERENT). 14/14 barrier tests pass.
- GPU L2 caching was NOT the root cause. Wasted time on this before testing.

## 2026-03-14: CP idle stall root cause confirmed
- CP goes idle after evaluating unsatisfied barrier dep with SLOT_BASED_WPTR=0
- Same-WPTR doorbell re-ring does NOT generate DOORBELL_HIT
- STALL logging: Queue C RPTR frozen for 3.5M iterations while dep signal is 0

## 2026-03-14: SLOT_BASED_WPTR=2 definitively dead on gfx8
- Tested with kernarg pool (GPU-visible) address. CP_HQD_PQ_WPTR stays at 0.
- CP RPTR writes work (PCIe posted writes, GPU→system) but CP WPTR reads fail (GPUVM translation without ATC).

## 2026-03-14: NOP barrier kick — temporary workaround
- Injecting NOP barrier-AND at write_dispatch_id + ringing doorbell with WPTR+1 wakes idle CP.
- Passes 100/100 alternating 512B+2MB. Drifts at ~286 ops from write_dispatch_id accumulation.

## 2026-03-15: WaitCurrent between H2D staging chunks — clean fix for H2D
- 1-line CLR change. Matches D2H path behavior. 550/550 stress test.

## 2026-03-15: cpu_wait_for_signal for no-atomics — eliminates CLR barrier deps
- Existing `ROC_CPU_WAIT_FOR_SIGNAL` env var / `cpu_wait_for_signal_` setting.
- Inject check for `!pcie_atomics_` in WaitingSignal() via PKGBUILD sed.

## 2026-03-15: Phase 6a — BlitKernel CPU-wait dep_signals
- CPU-wait in SubmitLinearCopyCommand instead of AQL barrier packets. 550/550 stress test.

## 2026-03-15: Clean Retain/Release without NOP kick SEGFAULTS
- Removing NOP kick from 0004 causes segfault in simple H2D test. NOP kick masks a latent signal lifecycle bug.
- **The NOP kick is not just a CP wake mechanism — it prevents a crash.** Root cause unknown.

## 2026-03-15: Need systematic layer-by-layer review
- User feedback: we've been chasing symptoms, not understanding the system.
- Must map the full signal lifecycle (creation → AQL packet → CP dispatch → AtomicDec → bounce buffer → Release) before fixing further.
- The right fix is at ONE layer, not stacked workarounds across 3 layers.
