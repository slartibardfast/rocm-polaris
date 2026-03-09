# PLAN.md — ROCm GCN 1.2 Restoration

## Current Target

- **ROCm version:** 7.2.0
- **Target ISAs:** gfx801 (Carrizo), gfx802 (Tonga/Iceland), gfx803 (Fiji/Polaris)
- **Primary test hardware:** Radeon Pro WX 2100 (Polaris 12, gfx803, 2GB GDDR5)
- **Use case:** Single-slot micro-LLM inference
- **Host OS:** Arch Linux (matches `extra/` ROCm 7.2.0 packages)

## Why 7.2.0

- Latest upstream release as of 2026-03-07
- Matches Arch `extra/` ROCm packages exactly — clean `provides`/`conflicts`
- Inference frameworks (ollama-rocm, llama.cpp) in Arch link against 7.2.0
- Avoids ABI mismatch pain from targeting an older ROCm

## Scope

### In scope
- **`linux-lts-rocm-polaris`** kernel: disable `needs_pci_atomics` for gfx8, add `rmmio_remap` for VI (**DONE**: PCIe atomics patch; **TODO**: MMIO remap)
- **`hsa-rocr-polaris`**: patch DoorbellType check to accept type 1 (pre-Vega) GPUs
- **`hip-runtime-amd-polaris`** (optional): remove OpenCL gfx8 gate in `runtimeRocSupported()`
- **`rocblas-gfx803`**: rebuild with gfx803 in target list (**PKGBUILD ready**)
- PKGBUILD packaging for all above with `provides`/`conflicts` against `extra/` packages

### Out of scope
- GCN 1.0 (gfx6xx) and GCN 1.1 (gfx7xx)
- Performance tuning tables (Tensile, MIOpen) unless explicitly requested
- Test infrastructure or CI config patches

## Known Caveats

- **2GB VRAM:** Only quantized micro-models fit (Q4 ~1-2B parameter models)
- **PCIe atomics:** gfx803 on platforms without PCIe atomics (pre-Haswell Intel, pre-Zen AMD) is blocked by KFD at the kernel level. The check is in `kfd_device.c`: `needs_pci_atomics=true` for all non-Hawaii gfx8 chips, with no firmware version override. **Requires kernel patch or a platform with PCIe atomics.** Test host (dual Xeon X5650, Westmere) triggers this: `kfd: skipped device 1002:6995, PCI rejects atomics 730<0`
- **Arch `extra/` conflicts:** Our packages must declare `provides`/`conflicts` to coexist or replace official packages

## Phase 1: Assessment (COMPLETE — revised after testing)

Cloned ROCm 7.2.0 submodules and grepped all repos for gfx8xx support status.

### Findings

**Initial discovery: AMD removed gfx8 from build targets, not from source code.** However, deeper testing revealed additional runtime gates beyond build targets.

| Component | Source Code | Build Targets | Runtime Gates | Patches Needed |
|-----------|-------------|---------------|---------------|----------------|
| llvm-project (LLVM/Clang/comgr) | Fully present | Included | None | **None** |
| ROCR-Runtime (HSA) | Fully present | Included | **DoorbellType!=2 rejects pre-Vega** | **amd_gpu_agent.cpp:124** |
| clr (HIP/ROCclr) | Fully present | HIP path open | OpenCL gated by `runtimeRocSupported()` | **device.hpp:1457** (OpenCL only) |
| HIP (API headers) | Delegates to clr | N/A | None | **None** |
| rocBLAS | Runtime code present | Dropped from CMake at 6.0 | None | **CMake target list** |
| Kernel (KFD) | Fully present | N/A | `needs_pci_atomics`, missing `rmmio_remap` | **kfd_device.c, vi.c** |

### Runtime Gates Discovered During Testing

1. **KFD `needs_pci_atomics`** (`kfd_device.c:260`): Blocks GPU on platforms without PCIe atomic ops. **FIXED** in `linux-lts-rocm-polaris`.

2. **ROCR DoorbellType check** (`amd_gpu_agent.cpp:124`): `if (node_props.Capability.ui32.DoorbellType != 2)` throws `HSA_STATUS_ERROR`. Polaris uses DoorbellType=1 (pre-Vega). This is the primary blocker for `hsa_init()`. **Requires hsa-rocr rebuild.**

3. **MMIO remap missing for VI** (`kfd_chardev.c:1134`): `rmmio_remap.bus_addr` is never set for Volcanic Islands GPUs (no NBIO subsystem). KFD returns `-ENOMEM` for `KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP`. The thunk prints "Failed to map remapped mmio page" but continues — **non-fatal warning**, may need kernel patch for full functionality.

4. **CLR OpenCL gate** (`device.hpp:1457`): `!IS_HIP && versionMajor_ == 8` returns false. **HIP path unaffected.** Only matters if OpenCL is needed.

#### Detail: llvm-project
- `GCNProcessors.td`: gfx801 (carrizo), gfx802 (iceland/tonga), gfx803 (fiji/polaris10/polaris11) all defined
- `AMDGPU.td`: FeatureISAVersion8_0_{1,2,3} and FeatureVolcanicIslands fully present
- `clang/lib/Basic/OffloadArch.cpp`: GFX(801), GFX(802), GFX(803) present
- `amd/comgr/src/comgr-isa-metadata.def`: gfx801/802/803 metadata complete
- ELF machine code mappings present in AMDGPUTargetStreamer.cpp

#### Detail: ROCR-Runtime
- `isa.cpp`: ISA registry entries for gfx801/802/803
- `topology.c`: Device ID mappings — Carrizo (0x9870-0x9877), Tonga (0x6920-0x6939), Fiji (0x7300/0x730F)
- `amd_hsa_code.cpp`: ELF machine type mappings
- `blit_kernel.cpp`: Blit kernel objects for gfx801/802/803
- PMC, addrlib, memory management all present

#### Detail: clr (HIP + ROCclr)
- `device.cpp`: ISA definitions in supportedIsas_ array for gfx801/802/803
- `amd_hsa_elf.hpp`: ELF format definitions present
- `device.hpp`: `runtimeRocSupported()` gates gfx8 for non-HIP (`!IS_HIP && versionMajor_ == 8`). HIP path is unaffected.
- Device enumeration in rocdevice.cpp functional

#### Detail: rocBLAS
- `handle.hpp`: Processor enum includes `gfx803 = 803`
- `handle.cpp`: Runtime detection code for gfx803 present
- `tensile_host.cpp`: Tensile lazy loading for gfx803 present
- 16 Tensile YAML logic files for R9 Nano (gfx803) present in `blas3/Tensile/Logic/`
- `CMakeLists.txt`: gfx803 present in TARGET_LIST_ROCM_5.6 and 5.7, **removed from 6.0+**
- Patch: add `gfx803` to TARGET_LIST_ROCM_7.1 (or whichever list 7.2 selects)

### Assessment Summary (Revised)

Initial assessment was overly optimistic — "no runtime patches needed" was wrong. While the source code is intact, ROCm 7.2.0 has multiple runtime checks that reject pre-Vega GPUs. The project requires:

1. **Kernel patches** (3): PCIe atomics bypass (done), AQL queue size fix (done), MMIO remap for VI (TODO)
2. **hsa-rocr patch** (1): Accept DoorbellType 1 — this is the primary blocker
3. **rocBLAS patch** (1): Re-enable gfx803 build target (done)
4. **clr patch** (1, optional): Remove OpenCL gfx8 gate

No LLVM patches needed. No HIP API header patches needed.

## Strategy (Revised — Comprehensive `-polaris` Builds)

Build custom `-polaris` packages for every component with a gfx8 gate:

### Phase 2a: hsa-rocr-polaris (DONE)
- Patch `amd_gpu_agent.cpp:124`: accept DoorbellType 1 for gfx8
- Patch `image_runtime.cpp`: skip image dimension query when image_manager is NULL
- Built `hsa-rocr-polaris` 7.2.0-1: `provides=('hsa-rocr=7.2.0')` / `conflicts=('hsa-rocr')`
- **Result:** `hsa_init()` works, `rocminfo` detects WX 2100, HIP device enumeration works

### Phase 2b: Kernel AQL queue fix (DONE)
- `kfd_queue.c:250` halved expected ring buffer size for GFX7/8 AQL — mismatch with ROCR allocation
- Patch `0005-kfd-fix-aql-queue-ring-buffer-size-check-for-gfx8.patch` removes the halving
- Built `linux-lts-rocm-polaris` 6.18.16-2
- **Result:** Both queue creations succeed (`hsa_queue_create` returns 0)

### Phase 2c: Kernel MMIO remap (MAY BE NEEDED)
- Add `rmmio_remap.bus_addr` initialization for VI/Polaris in kernel
- Without this, HDP flush via MMIO is unavailable — may cause performance issues or runtime errors
- Can defer unless it turns out to be the GPU dispatch hang cause

### Phase 2d: rocBLAS (PKGBUILD READY)
- Apply existing `0001-re-enable-gfx803-target.patch`
- Build with `makepkg -s`

### Phase 3: GPU dispatch hang debugging (CURRENT)

**Symptom:** HIP test (`hipcc --offload-arch=gfx803`) hangs in runtime static init. Both `CREATE_QUEUE` ioctls return 0, many `ALLOC_MEMORY`/`MAP_MEMORY` succeed, then no more ioctls — HIP spins at 100% CPU in userspace. Pure HSA queue test (`hsa_queue_create` + `hsa_queue_destroy`) works fine — the hang is in GPU command dispatch, not queue creation.

**Suspected causes (ordered by likelihood):**

1. **Blit kernel dispatch hang.** HIP init uses blit kernels (precompiled gfx803 code objects in ROCR) for internal memory operations (fill, copy). If the blit kernel dispatch writes to the doorbell but the GPU never completes the packet, ROCR's `WaitOnSignal` spins forever. The blit kernel objects exist in `blit_kernel.cpp` for gfx803 — but they may have been compiled with assumptions about features unavailable on our hardware (e.g., flat scratch, GWS).

2. **HDP flush / cache coherency.** Polaris has no NBIO, so `rmmio_remap.bus_addr` is never set. ROCR's `HdpFlush()` may silently no-op or write to an invalid address. If the GPU doesn't see updated memory (AQL packets written by the CPU), it processes stale data or never sees the dispatch. This connects to the Phase 2c MMIO remap TODO.

3. **Doorbell write semantics.** Polaris uses 32-bit doorbells (DoorbellType=1), Vega+ uses 64-bit (DoorbellType=2). Our ROCR patch accepts DoorbellType=1, but downstream code may still write 64-bit doorbell values. If the doorbell stride or write size is wrong, the CP never wakes up to process the queue. Check `amd_gpu_agent.cpp` doorbell stride calculation and `AqlQueue::SubmitPacket` doorbell write path.

4. **CWSR (Compute Wave Save/Restore) trap handler.** KFD installs a trap handler for context switching. The CWSR image may be compiled for gfx9+ only, or the `ctx_save_restore_size` negotiation may assume features not present on gfx8. If the trap handler is invalid, the first wave dispatch could hang or fault silently. Check `kfd_device.c` CWSR firmware loading path for gfx8.

5. **Scratch memory setup.** If the `AqlQueue` constructor configures scratch (private memory) incorrectly for gfx8 — wrong wave size, wrong scratch segment size, or wrong flat scratch init — the first kernel dispatch will hang on scratch access. ROCR's `ScratchCache` and queue scratch setup may have gfx9-only assumptions.

6. **Signal/completion mechanism.** ROCR creates HSA signals for synchronization. If signal completion uses interrupts (`KFD_IOC_WAIT_EVENTS`) and the interrupt path for gfx8 is broken, the CPU side won't detect completion. However, the 100% CPU spin suggests ROCR is polling (not interrupt-waiting), so this is less likely.

**Investigation plan:**
- Write a minimal HSA dispatch test: create queue, submit a trivial no-op AQL packet, ring doorbell, poll completion signal — isolates whether *any* GPU dispatch works
- If no-op dispatch hangs: problem is doorbell/CP/HDP (causes 2-3)
- If no-op dispatch works but blit kernel hangs: problem is kernel code or scratch (causes 1, 5)
- Check `HSA_ENABLE_INTERRUPT=0` env var to force polling mode (rules out cause 6)
- Add `HSA_DEBUG` or ROCR debug logging to see where init stalls
- Inspect MMIO remap / HDP flush path for Polaris specifically

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-07 | Target ROCm 7.2.0 | Matches Arch extra/, latest upstream |
| 2026-03-07 | Scope limited to gfx801/802/803 | GCN 1.2 only, matching project charter |
| 2026-03-07 | WX 2100 as primary test hardware | Available single-slot gfx803 card for micro-LLM use case |
| 2026-03-07 | Kernel patch needed for test platform | Xeon X5650 lacks PCIe atomics; KFD skips GPU. Patch kfd_device.c |
| 2026-03-07 | Test 01 confirmed: Arch rocm-llvm has gfx8 | `llc -march=amdgcn -mcpu=help` shows gfx801/802/803 subtargets |
| 2026-03-07 | **hsa-rocr needs patching** | DoorbellType!=2 check in `amd_gpu_agent.cpp:124` rejects Polaris (type 1). Root cause of `hsa_init` failure. |
| 2026-03-07 | **MMIO remap missing for VI** | `rmmio_remap.bus_addr` never set for Polaris in kernel. KFD returns -ENOMEM for MMIO_REMAP alloc. Non-fatal but may affect HDP flush perf. |
| 2026-03-07 | Revised strategy: comprehensive -polaris builds | Stock Arch packages have multiple runtime gates; build custom packages for all gated components |
| 2026-03-07 | Community prior art: xuhuisheng/rocm-gfx803 | Built custom hsa-rocr starting at ROCm 5.3.0; confirms runtime patching needed. NULL0xFF/rocm-gfx803 uses ROCm 6.1.5 on EPYC (has PCIe atomics). |
| 2026-03-07 | **KFD AQL queue ring buffer size mismatch** (see Debugging Notes below) | `kfd_queue.c:250` halves `expected_queue_size` for AQL on GFX7/8, but ROCR allocates full-size ring buffer BO. `kfd_queue_buffer_get` requires exact BO size match → EINVAL on any queue > 2KB. Utility queue passes by accident (2048-byte expected → 0 pages → size check skipped). Fix: remove the halving — it's a CP register encoding detail, not a BO allocation convention. Only affects GFX7/8 code path. |
| 2026-03-08 | **Queue creation confirmed fixed** | After kernel 6.18.16-2, both CREATE_QUEUE ioctls return 0. Pure HSA test creates queue and destroys it successfully. HIP test hangs in runtime static init after all ioctls succeed — GPU dispatch execution issue, not queue creation. |

## Debugging Notes

### AQL Queue EINVAL — Root Cause Analysis

**Symptom:** Any HIP operation that dispatches GPU work (hipMemset, kernel launch) crashes with SIGSEGV. The segfault occurs in `ScratchCache::freeMain` (`scratch_cache.h:177`) during error cleanup after `AqlQueue` constructor throws — the real error is `AMDKFD_IOC_CREATE_QUEUE` returning EINVAL.

**Observation:** The *first* CREATE_QUEUE (utility/internal queue, ring_size=4096) succeeds. The *second* (user queue, ring_size=65536) fails. Both have identical ctl_stack_size, eop_size, ctx_save_restore_size, priority, queue_type.

**Debugging approach:**

1. **strace** confirmed the ioctl failure: `ioctl(3, AMDKFD_IOC_CREATE_QUEUE, ...) = -1 EINVAL`. But strace can't decode KFD ioctl struct fields — it just shows the pointer address.

2. **LD_PRELOAD ioctl interceptor** — wrote a small C shared library that intercepts `ioctl()`, checks if the request matches `AMDKFD_IOC_CREATE_QUEUE` (by matching the ioctl type/nr bytes `'K'/0x02`), and dumps all struct fields before and after the real call. This revealed the exact parameter values for both calls:
   ```
   Queue 1 (OK):   ring_size=4096,  eop=4096, ctx_size=2789376, ctl_stack=4096
   Queue 2 (FAIL): ring_size=65536, eop=4096, ctx_size=2789376, ctl_stack=4096
   ```
   All fields identical except ring_base address and ring_size. This narrowed the problem to ring buffer validation.

3. **Kernel source analysis** — traced the ioctl handler chain:
   - `kfd_ioctl_create_queue` → `set_queue_properties_from_user` (basic validation — all params pass)
   - → `kfd_queue_acquire_buffers` (BO ownership validation — **newer code, added after GFX8 was dropped**)
   - → `kfd_queue_buffer_get` (looks up VM mapping by address, requires exact size match)

4. **Found the halving** at `kfd_queue.c:245-250`:
   ```c
   /* AQL queues on GFX7 and GFX8 appear twice their actual size */
   if (format == AQL && gfx_version < 90000)
       expected_queue_size = queue_size / 2;
   ```
   This halves the expected BO size for the ring buffer lookup on GFX7/8.

5. **Traced the allocation side** — ROCR's `AqlQueue::AllocRegisteredRingBuffer` allocates `queue_size_pkts * sizeof(AqlPacket)` = full size. The BO mapping in the GPU VM is the full allocation.

6. **Size mismatch confirmed:**
   - Queue 1: `expected = 4096/2 = 2048 bytes = 0 pages` → size==0 skips the check → **passes by accident**
   - Queue 2: `expected = 65536/2 = 32768 = 8 pages`, but BO mapping is 16 pages → **mismatch → EINVAL**

**Key insight:** The BO validation (`kfd_queue_acquire_buffers` / `kfd_queue_buffer_get`) was added to prevent userspace from passing arbitrary addresses. It was implemented after GFX8 was already dropped from ROCm, so the AQL size halving was never tested against real GFX8 hardware. The halving is correct for the CP hardware register encoding but wrong for BO validation — the BO is always the full size.

**Fix:** Remove the GFX7/8 AQL size halving in `kfd_queue_acquire_buffers`. Kernel patch `0005-kfd-fix-aql-queue-ring-buffer-size-check-for-gfx8.patch`. Only affects code paths where `gfx_target_version < 90000`, which is exclusively GFX7/8 — no impact on GFX9+.
