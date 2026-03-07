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

## Phases

### Phase 1: Assessment
- Clone ROCm 7.2.0 submodules
- Grep all repos for gfx8xx removal points
- Catalog required patches per component
- Estimate patch complexity

### Phase 2: LLVM/Clang
- Re-enable gfx801/gfx802/gfx803 in AMDGPU backend
- Verify `llc --version` lists targets
- Run AMDGPU lit tests if feasible

### Phase 3: Runtime stack
- Patch ROCR-Runtime, HIP, ROCclr, amd_comgr
- Verify `rocminfo` detects gfx803 (requires hardware)
- Verify comgr can compile trivial HIP kernel for gfx803

### Phase 4: Packaging
- Write PKGBUILD with proper split packages
- Test `makepkg -s` clean build
- Validate installed package with inference workload on WX 2100

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-07 | Target ROCm 7.2.0 | Matches Arch extra/, latest upstream |
| 2026-03-07 | Scope limited to gfx801/802/803 | GCN 1.2 only, matching project charter |
| 2026-03-07 | WX 2100 as primary test hardware | Available single-slot gfx803 card for micro-LLM use case |
