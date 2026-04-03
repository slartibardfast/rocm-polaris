# GCN3 Polaris — Complete Technical Record

## Status: COMPLETE — 49.13 t/s bandwidth ceiling reached

This document captures everything learned restoring ROCm on GCN3 (Polaris 12,
gfx803) and optimizing Vulkan inference to the hardware limit. Written as a
transfer document for GCN5 (Vega) work.

---

## Hardware Under Test

| Spec | Value |
|------|-------|
| GPU | Radeon Pro WX 2100 (Polaris 12) |
| ISA | gfx803 (GCN 1.2 / GCN3) |
| CUs | 10 |
| VRAM | 2 GB GDDR5, 64-bit bus |
| Bandwidth | ~24 GB/s theoretical |
| L2 Cache | 512 KB |
| VGPRs | 256 per SIMD, 4 SIMDs per CU |
| Wave size | 64 (native, no wave32) |
| PCIe | 2.0 x16 (host: dual Xeon X5650, no atomics) |
| BAR | 256 MB (small BAR) |
| Driver | RADV (Mesa 26.0.1), amdgpu kernel module |
| ROCm | 7.2.0 (patched for gfx803) |

**Model**: Qwen3.5-0.8B Q4_K_M (497 MiB, hybrid SSM+attention, 18 SSM + 6 attention layers)

---

## What We Built

### ROCm Package Stack (Arch Linux)

| Package | Version | Purpose |
|---------|---------|---------|
| `linux-lts-rocm-polaris` | 6.18.16-40 | Kernel with SNOOPED, TC_CFG, RPTR fixes |
| `hsa-rocr-polaris` | 7.2.0-37 | HSA runtime with Phase 10 shared EOP event |
| `hip-runtime-amd-polaris` | 7.2.0-37 | HIP/CLR with barrier fix, D2H safe threshold |
| `rocblas-gfx803` | 7.2.0-2 | rocBLAS with gfx803 target re-enabled |

### Vulkan Inference (Primary Path)

llama.cpp fork at `slartibardfast/llama.cpp:polaris-jit` with:
- Phase 18: GDN shader occupancy fix (LANES=64)
- Phase 19: f16 BDA state cache for SSM layers
- Phase 20: SSM state I/O elimination (skip GET_ROWS/CPY)
- Phase 21: Multi-WG JIT interpreter (confirmed dead, code preserved)

---

## Performance History

| Phase | ms/token | t/s | Key Change |
|-------|----------|-----|------------|
| Phase 10 (HIP) | 16.94 | 59.05 | SmolLM2-135M, interrupt signals |
| Phase 16 baseline | 22.60 | 44.24 | Qwen3.5-0.8B, shader tuning |
| Phase 18 | ~21.5 | ~46.5 | GDN occupancy 25%→100% |
| Phase 19 | 22.16 | 45.13 | f16 BDA state cache |
| Phase 20 | 20.44 | 48.91 | State I/O elimination |
| **Phase 20 (final)** | **20.35** | **49.13** | Best measurement |
| Phase 21 (JIT 1-WG) | 23.20 | 43.11 | 12% slower than standard |

**Bandwidth utilization**: 497 MiB / 20.35 ms = 24.4 GB/s ≈ **100% of GDDR5 theoretical**

---

## GCN3 Architecture — What We Proved

### Memory Model (VI / gfx8)

1. **No MTYPE in PTE.** gfx9+ has MTYPE at PTE bits 57:58. gfx8 uses only the
   SNOOPED bit (PTE bit 2) for GPU L2 coherency. Without SNOOPED=1, GPU L2
   acts non-coherently for system memory — dirty lines never reach DRAM.

2. **No GPU L2 flush instruction.** System-scope release is just `s_waitcnt`.
   System-scope acquire is L1-only (`BUFFER_WBINVL1_VOL`). GPU L2 ↔ DRAM
   coherency relies entirely on PCIe snoop protocol with SNOOPED=1.

3. **TC_CFG L2 load policy matters.** Default `MISS_LRU` for UC memory causes
   GPU L2 to cache system memory reads. Kernarg corruption at ~256 dispatches
   was traced to stale L2 entries. Fix: `MISS_EVICT` policy.

4. **ACQUIRE_MEM TC_WB is outside gfx8 ISA model.** Attempting PM4 ACQUIRE_MEM
   with TC_WB_ACTION_ENA causes VM faults on gfx8. Removed.

5. **CPU→GPU L2 coherency is one-directional.** GPU can snoop CPU caches
   (SNOOPED=1). CPU cannot snoop GPU L2. CPU must use UC/WC mappings for
   GPU-written data.

### Compute Model

1. **No preemption.** Compute shaders run to completion. No context switching.
   No TDR on Linux. A hung shader requires kexec to recover.

2. **No cross-WG synchronization.** `atomicOr` polling for inter-WG barriers
   deadlocks — L1 caches stale values and never sees updates from other CUs.
   `coherent` qualifier doesn't help on RADV/GCN3. The hardware was designed
   for independent dispatches, not persistent compute.

3. **Dispatch overhead is minimal.** At realistic context sizes (n_ctx≥512),
   per-dispatch overhead is ~13µs — too small to justify megakernel approaches.
   The JIT interpreter (1-WG) adds 12% overhead from SSBO reads, BDA
   indirection, and barrier cost.

4. **Occupancy is critical.** GCN3 hides memory latency through wave occupancy,
   not caches. GDDR5 latency is ~200 cycles; need ≥4 waves/SIMD to hide it.
   Target ≤64 VGPRs (4 waves) or ≤32 VGPRs (8 waves) per shader.

5. **Wave64 only.** No wave32 mode. Subgroup operations operate on 64 lanes.
   `subgroupAdd` replaces manual DPP reduction. `LANES_PER_COLUMN=64` for
   GDN shader enables full wave utilization.

### PCIe / Host Interaction

1. **No PCIe atomics** (Westmere host). KFD `pcie_atomics_` flag is false.
   Affects CLR staging buffer strategy — fine-grain coherency unavailable.

2. **Small BAR (256 MB).** Not all VRAM is host-visible. `eDeviceLocal |
   eHostVisible | eHostCoherent` allocations compete for BAR space.

3. **D2H blit L2 writeback bug.** hipMemcpy D2H via blit kernel writes through
   GPU L2, but L2 dirty lines aren't written back before CPU reads. Fix:
   hipHostMalloc for small allocations (≤1MB), bypassing blit kernel.

4. **CP idle stall.** Command Processor goes idle when WPTR polling fails.
   SLOT_BASED_WPTR=2 is dead on gfx8 (CP poll engine uses ATC, not GPUVM).
   Fix: CPU-managed barriers (Phase 8).

### RADV / Vulkan Specifics

1. **`GGML_VK_*` env vars collide with RADV.** Setting any env var starting
   with `GGML_VK_` causes GPU hangs on RADV/Polaris. The Vulkan loader or
   Mesa interprets these. Use `LLAMA_VULKAN_*` or `POLARIS_*` prefix.

2. **Pipeline creation during graph_compute hangs after warmup.** Creating a
   VkPipeline (via `vkCreateComputePipeline`) inside `graph_compute` after a
   warmup graph_compute has run causes subsequent dispatches to hang.
   Workaround: auto-skip warmup when JIT enabled.

3. **BDA (VK_KHR_buffer_device_address) works well.** Used for f16 state cache,
   JIT interpreter SSBO program, and direct tensor access. No driver issues.
   BDA is faster than descriptors on AMD (1 vs 2 indirections).

4. **Flash attention, conv2d pipelines compile lazily without issues.** Only the
   JIT interpreter pipeline triggers the warmup hang — likely related to
   shaderc runtime compilation vs pre-compiled SPIR-V.

---

## Kernel Patches (linux-lts-rocm-polaris)

| Patch | Purpose | Transfers to Vega? |
|-------|---------|--------------------|
| 0004: amdgpu gfx8 targets | Re-enable gfx8 ISA in amdgpu | No (Vega = gfx9) |
| 0005: KFD gfx8 support | KFD device enumeration for gfx8 | No |
| 0006: SDMA fixes | SDMA v2/v3 removed, v4 compat | No (Vega has SDMA v4) |
| **0008: RPTR_BLOCK_SIZE** | CP writes RPTR every 32→4 dwords | **Maybe** (check Vega CP) |
| **0009: SNOOPED=1 for ttm_uncached** | GPU L2 coherency for system mem | **Probably not** (gfx9 has MTYPE) |

**Vega notes:**
- gfx9 has MTYPE in PTE — can set UC/WC/CC per-page without SNOOPED bit hack
- Vega SDMA v4 should work out of box
- CP behavior may differ — test RPTR_BLOCK_SIZE on Vega before patching

---

## ROCR Patches (hsa-rocr-polaris)

| Patch | Purpose | Transfers to Vega? |
|-------|---------|--------------------|
| 0001: gfx8 device enumeration | HSA agent for gfx803 | No (Vega = gfx900/906) |
| 0002: firmware paths | Microcode loading for gfx8 | No |
| 0003: signal handling | Signal create/wait for gfx8 | **Maybe** (if signal path differs) |
| **0004: bounce buffer + NOP kick** | Retain/Release in bounce buf | **Review** (may apply to Vega) |
| **0005: FlushCpuCache in CopyMemory** | clflush before GPU reads | **Yes if no-atomics host** |

**Vega notes:**
- 0004 (bounce buffer) addresses a generic ROCR issue, not gfx8-specific
- 0005 (FlushCpuCache) is needed on any no-atomics platform (Westmere)
- If Vega is tested on a platform WITH PCIe atomics, 0005 may be unnecessary

---

## HIP/CLR Patches (hip-runtime-amd-polaris)

| Patch | Purpose | Transfers to Vega? |
|-------|---------|--------------------|
| **0001: fine-grain staging + clflush** | D2H coherency fix | **Yes if no-atomics host** |
| Phase 8 barrier fix | hsa_signal_wait_scacquire | **Review** |
| D2H safe threshold | hipHostMalloc for small D2H | **Yes if L2 writeback issue persists** |

**Vega notes:**
- gfx9 HAS L2 flush instruction (`ACQUIRE_MEM` with TC_WB works on gfx9)
- D2H blit L2 writeback fix may be unnecessary if ACQUIRE_MEM TC_WB works
- Test D2H correctness first before applying patches

---

## Vulkan Inference Optimizations — Transferability

| Optimization | Transfers? | Notes |
|-------------|-----------|-------|
| **GDN LANES=64** | Yes | wave64 native on both GCN3 and GCN5 |
| **f16 BDA state cache** | Yes | BDA works on any VK 1.2 device |
| **State I/O elimination** | Yes | Skip GET_ROWS/CPY is graph-level |
| **Occupancy tuning** | **Review** | Vega has same 256 VGPRs but more CUs |
| JIT interpreter | No | 12% slower, not worth pursuing |
| Multi-WG megakernel | **Maybe** | Vega has L2 flush + better atomics |

**Key difference for Vega inference:**
- HBM2 bandwidth: ~484 GB/s (Vega 64) vs 24 GB/s (WX 2100) = **20x**
- At 20x bandwidth, the 497 MiB model loads in ~1ms instead of 20ms
- **Dispatch overhead becomes dominant** — the megakernel might actually help!
- Cross-WG atomics may work on gfx9 (has proper L2 cache coherency)

---

## Debugging Techniques That Transfer

1. **LD_PRELOAD ioctl interceptor** — Match on ioctl type/nr to dump KFD structs
2. **Kernel dynamic debug** — `echo 'file kfd_queue.c +p' > /proc/dynamic_debug/control`
3. **PKGBUILD sed injection** — `sed -i '/pattern/r /tmp/inject.c'` for debug instrumentation
4. **Perf logger** — `GGML_VK_PERF_LOGGER=1` for per-op GPU timing
5. **strace for GPU hangs** — `futex(FUTEX_WAIT)` = fence wait = GPU hung
6. **Never read `amdgpu_regs`** — debugfs MMIO reads can hang GPU/crash host
7. **kexec recovery** — Only way to recover from GPU hang without full reboot

---

## Known Pitfalls for Vega Transition

1. **ROCm ISA string changes.** gfx803 → gfx900/gfx906. Grep all submodules for
   target strings: `gfx900`, `gfx906`, `GFX9`, `VEGA`, `vega10`, `vega20`.

2. **Tensile/rocBLAS.** Vega gfx906 has dot product instructions (v_dot2_f32_f16).
   rocBLAS Tensile configs may need gfx906-specific logic files. gfx900 lacks
   these instructions.

3. **PCIe atomics.** If Vega is tested on the same Westmere host, same no-atomics
   constraints apply. If tested on a Ryzen/EPYC host WITH atomics, many patches
   become unnecessary.

4. **HBM2 vs GDDR5 memory model.** HBM2 has different latency characteristics.
   Occupancy requirements may differ (HBM2 latency ~100 cycles vs GDDR5 ~200).

5. **SDMA availability.** Vega's SDMA v4 should work. Don't apply gfx8 SDMA
   removal patches.

6. **Multiple VCN/UVD engines.** Vega has different multimedia engines. Not
   relevant for compute but affects device enumeration.

---

## Files & Repos

| Location | Content |
|----------|---------|
| `github.com/slartibardfast/rocm-polaris` | Parent repo, PKGBUILDs, patches, phase docs |
| `github.com/slartibardfast/llama.cpp:polaris-jit` | Vulkan inference fork |
| `PHASE17.md` | JIT ubershader — abandoned |
| `PHASE18-20.md` | Occupancy + BDA cache + state I/O elimination |
| `PHASE21.md` | Multi-WG megakernel — confirmed dead on GCN3 |
| `PLAN.md` | Overall project plan and status |
| `.claude/projects/.../memory/` | Session memory files |

---

## Final Numbers

```
Qwen3.5-0.8B Q4_K_M on Polaris 12 (WX 2100, 2GB GDDR5):
  Generation: 49.13 t/s (20.35 ms/token)
  Prompt eval: 93.46 t/s (10.70 ms/token)
  Bandwidth:   24.4 GB/s = 100% of GDDR5 theoretical
  PPL:         16.13 (zero degradation from optimizations)

SmolLM2-135M on same hardware (HIP path):
  Generation: 59.05 t/s (16.94 ms/token)
```

Every byte accounted for. Every optimization proven or disproven on hardware.
