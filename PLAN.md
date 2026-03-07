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

### Phase 2a: hsa-rocr-polaris (CRITICAL PATH)
- Patch `amd_gpu_agent.cpp:124`: accept DoorbellType 1 for gfx8
- Build `hsa-rocr-polaris` package: `provides=('hsa-rocr=7.2.0')` / `conflicts=('hsa-rocr')`
- This unblocks `hsa_init()` and `rocminfo`

### Phase 2b: Kernel MMIO remap (MAY BE NEEDED)
- Add `rmmio_remap.bus_addr` initialization for VI/Polaris in kernel
- Without this, HDP flush via MMIO is unavailable — may cause performance issues or runtime errors
- Can defer if hsa-rocr-polaris alone gets rocminfo working

### Phase 2c: rocBLAS (PKGBUILD READY)
- Apply existing `0001-re-enable-gfx803-target.patch`
- Build with `makepkg -s`

### Phase 3: Verify + Test
- Install hsa-rocr-polaris, verify `rocminfo` detects WX 2100
- Run `hipcc` compile test for gfx803
- Install rocblas-gfx803, test inference workload

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
| 2026-03-07 | **KFD AQL queue ring buffer size mismatch** | `kfd_queue.c:250` halves `expected_queue_size` for AQL on GFX7/8, but ROCR allocates full-size ring buffer BO. `kfd_queue_buffer_get` requires exact BO size match → EINVAL on any queue > 2KB. Utility queue passes by accident (2048-byte expected → 0 pages → size check skipped). Fix: remove the halving — it's a CP register encoding detail, not a BO allocation convention. Only affects GFX7/8 code path. |
