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
- LLVM/Clang AMDGPU backend: re-enable gfx801/gfx802/gfx803 targets
- ROCR-Runtime (HSA): device enumeration and firmware loading for gfx8xx
- HIP runtime: device-to-ISA mapping for gfx8xx
- ROCclr / amd_comgr: compiler runtime support
- rocBLAS: only if needed for inference workloads
- PKGBUILD packaging for Arch with `provides`/`conflicts` against `extra/`

### Out of scope
- GCN 1.0 (gfx6xx) and GCN 1.1 (gfx7xx)
- Performance tuning tables (Tensile, MIOpen) unless explicitly requested
- Kernel driver changes (amdgpu already supports these GPUs)
- Test infrastructure or CI config patches

## Known Caveats

- **2GB VRAM:** Only quantized micro-models fit (Q4 ~1-2B parameter models)
- **PCIe atomics:** gfx803 on some platforms lacks PCIe atomics support — BIOS/hardware limitation, not patchable
- **Arch `extra/` conflicts:** Our packages must declare `provides`/`conflicts` to coexist or replace official packages

## Phase 1: Assessment (COMPLETE)

Cloned ROCm 7.2.0 submodules and grepped all repos for gfx8xx support status.

### Findings

**Key discovery: AMD removed gfx8 from build targets, not from source code.** The code supporting gfx801/802/803 is still present across all components.

| Component | Source Code | Build Targets | Patches Needed |
|-----------|-------------|---------------|----------------|
| llvm-project (LLVM/Clang/comgr) | Fully present | Included | **None** |
| ROCR-Runtime (HSA) | Fully present | Included | **None** |
| clr (HIP/ROCclr) | Fully present | HIP path open, OpenCL gated | **None** for HIP |
| HIP (API headers) | Delegates to clr | N/A | **None** |
| rocBLAS | Runtime code present | Dropped from CMake at 6.0 | **CMake target list** |

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

### Assessment Summary

This project is much simpler than anticipated. The primary work is:
1. **PKGBUILD** that builds from source with gfx803 in GPU_TARGETS
2. **rocBLAS CMake patch** to add gfx803 back to the target list
3. **Verification** on WX 2100 hardware

No LLVM patches. No runtime patches. No HIP patches.

## Phases (Revised)

### Phase 2: PKGBUILD + rocBLAS patch
- Write PKGBUILD that builds llvm-project, ROCR-Runtime, clr, HIP, rocBLAS from source
- Pass `-DGPU_TARGETS="gfx801;gfx802;gfx803"` (or append to default list)
- Patch rocBLAS CMakeLists.txt to include gfx803 in target list
- Set up `provides`/`conflicts` against Arch `extra/` packages

### Phase 3: Build + Test
- `makepkg -s` clean build
- Verify `llc --version | grep gfx803`
- Verify `rocminfo` detects WX 2100 as gfx803
- Verify comgr can compile trivial HIP kernel for gfx803
- Test inference workload (e.g., llama.cpp with small quantized model)

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-07 | Target ROCm 7.2.0 | Matches Arch extra/, latest upstream |
| 2026-03-07 | Scope limited to gfx801/802/803 | GCN 1.2 only, matching project charter |
| 2026-03-07 | WX 2100 as primary test hardware | Available single-slot gfx803 card for micro-LLM use case |
| 2026-03-07 | No LLVM/ROCR/HIP patches needed | Source code fully supports gfx8; only build targets removed |
| 2026-03-07 | Only rocBLAS needs a patch | CMake target list dropped gfx803 at ROCm 6.0; runtime code intact |
