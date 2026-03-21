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

### Phase 2c: Kernel MMIO remap (NOT NEEDED)
- `nbio_v2_3.c:set_reg_remap()` already sets `rmmio_remap.bus_addr` for standard PAGE_SIZE (4096)
- ROCR doesn't use SDMA HDP flush for gfx8 anyway (gfx9+ only)
- Compute shader blits (the path gfx8 uses) don't require HDP flush
- **Eliminated as a concern during Phase 3 deep review**

### Phase 2d: rocBLAS (VERIFIED)
- Applied `0001-re-enable-gfx803-target.patch` (adds gfx803 to `TARGET_LIST_ROCM_7.1`)
- Built with `AMDGPU_TARGETS=gfx803` for smoke test (gfx803-only, ~1hr vs ~17hrs for all arches)
- 72 Tensile kernel files installed, `librocblas.so.5.2` (189MB)
- **SGEMM 64×64×64 verified on hardware** — correct results

#### rocBLAS atomics caveat
rocBLAS queries HSA platform atomics support and defaults to `atomics_not_allowed` on platforms without PCIe atomics (our Westmere Xeon). With atomics disabled, Tensile kernels silently produce zeros — the kernel dispatches but uses a no-op code path.

**Fix:** Applications must call `rocblas_set_atomics_mode(handle, rocblas_atomics_allowed)` after `rocblas_create_handle()`. The atomics rocBLAS cares about are GPU global memory atomics (for parallel reduction), not PCIe atomics — they work fine on gfx803. This is a false positive from `platform_atomic_support_` in ROCR being repurposed by rocBLAS.

Downstream consumers (llama.cpp, PyTorch) will need this unless we patch rocBLAS to not gate on platform atomics. Deferring that decision until we test actual inference workloads.

### Phase 3: GPU dispatch hang debugging (CURRENT)

**Symptom:** HIP test (`hipcc --offload-arch=gfx803`) hangs in runtime static init. Both `CREATE_QUEUE` ioctls return 0, many `ALLOC_MEMORY`/`MAP_MEMORY` succeed, then no more ioctls — HIP spins at 100% CPU in userspace. Pure HSA queue test (`hsa_queue_create` + `hsa_queue_destroy`) works fine — the hang is in GPU command dispatch, not queue creation.

**Note:** The pure HSA test only tested queue lifecycle, not actual dispatch. `tests/hsa_dispatch_test.c` exists with a barrier packet test but has not been run since the kernel AQL fix.

#### Deep codebase review (2026-03-09)

Audited all six suspected causes against the ROCR-Runtime 7.2.0 and kernel source. Findings below.

**VALIDATED — not the cause:**

1. **~~HDP flush / cache coherency~~** — ELIMINATED.
   - ROCR explicitly disables SDMA HDP flush for gfx8 (`amd_blit_sdma.cpp:159`: `GetMajorVersion() >= 9` guard).
   - MMIO remap IS correctly set up for Polaris when `PAGE_SIZE <= 4096` (`nbio_v2_3.c:set_reg_remap()`).
   - Compute shader blits (used for gfx8) don't use HDP flush at all.
   - Phase 2c MMIO remap kernel patch is NOT needed.

2. **~~CWSR trap handler~~** — ELIMINATED.
   - Kernel has ISA-specific `cwsr_trap_gfx8_hex` binary (`cwsr_trap_handler.h`).
   - KFD correctly selects it via `KFD_GC_VERSION(kfd) < IP_VERSION(9, 0, 1)`.
   - MQD manager for VI properly configures CWSR fields.
   - `ctx_save_restore_size` calculation uses correct `CNTL_STACK_BYTES_PER_WAVE = 8` for gfx8.

3. **~~Scratch memory setup~~** — ELIMINATED.
   - gfx8 shares scratch path with gfx9, only difference: `need_queue_scratch_base = (major > 8)` affects offset calc.
   - `FillBufRsrcWord{1,3}` SRD format identical for gfx8 and gfx9.
   - `FillComputeTmpRingSize` handles `main_size == 0` correctly (zeros the register).
   - Blit kernels use only flat memory ops, no scratch required (private_segment_size from code object metadata).

4. **~~Blit kernel ISA compatibility~~** — ELIMINATED.
   - Pre-compiled gfx8 binaries exist in `amd_blit_shaders.h` (V1 header, checked into source).
   - Assembly macros auto-select correct gfx8 instructions: `v_add_u32` (not `v_add_co_u32`).
   - All ops are basic: `flat_load/store`, `s_load`, `v_add`, `v_cmp`, `s_endpgm`. All valid on gfx803.
   - **Note:** gfx8 not in CMake TARGET_DEVS (blit shaders or trap handlers), but V1 header fallback provides the binaries.

5. **~~ROCR trap handler~~** — ELIMINATED.
   - `BindTrapHandler()` at line 2284: gfx8 gets `kCodeTrapHandler8` from V1 header.
   - `TrapHandlerKfdExceptions` for gfx8 also uses `kCodeTrapHandler8` (V1 fallback).
   - `SetTrapHandler()` KFD ioctl installs it correctly.

**CONFIRMED — relevant findings:**

6. **SDMA disabled for gfx8** (`amd_gpu_agent.cpp:806`).
   - `use_sdma = (GetMajorVersion() != 8)` — all blit operations use **compute shader dispatch** via AQL queue.
   - This means blit Fill/Copy kernels are dispatched as regular AQL kernel packets on the utility queue.
   - Every memory operation (hipMemset, hipMemcpy, internal fill) goes through `BlitKernel::SubmitLinearFillCommand()` → AQL dispatch → doorbell write.

7. **64-bit doorbell write on 32-bit doorbell hardware** (`amd_aql_queue.cpp:478`).
   - `*(signal_.hardware_doorbell_ptr) = uint64_t(value)` — unconditional 64-bit store.
   - `signal_.hardware_doorbell_ptr` is `volatile uint64_t*` (`amd_hsa_signal.h:65`).
   - `Queue_DoorBell` (uint32*) and `Queue_DoorBell_aql` (uint64*) are a **union** (`hsakmttypes.h:766-771`).
   - Thunk sets doorbell offset at `queue_id * DOORBELL_SIZE(gfxv)` = `queue_id * 4` for gfx8.
   - **Result:** 8-byte write to a 4-byte-strided doorbell aperture. The lower 4 bytes go to queue N's doorbell, the upper 4 bytes overflow into queue N+1's doorbell slot.
   - For small write indices (< 2^32), upper 4 bytes = 0, which may be benign (CP sees doorbell value 0 for adjacent queue, likely ignored if write_ptr ≤ read_ptr).
   - **But:** x86 `mov qword` to UC MMIO is a single 8-byte PCIe transaction. The gfx8 doorbell controller may or may not handle 8-byte writes to 4-byte registers correctly — hardware-dependent behavior.
   - **Status:** Possible cause, but needs empirical testing. The thunk's own test code (`AqlQueue.cpp:50`) also writes via uint64* for AQL queues, suggesting AMD may have designed the doorbell controller to handle this even on gfx8.

8. **HIP init does NOT dispatch during static init** — dispatches are deferred.
   - `InitDma()` sets up lazy pointers (`lazy_ptr<Queue>`) — queues created on first access.
   - `VirtualGPU::create()` calls `acquireQueue()` (creates HSA queue → CREATE_QUEUE ioctl) and `KernelBlitManager::create()` (loads blit kernel objects) but does NOT dispatch.
   - First actual GPU dispatch happens on **first HIP API call that moves data** (hipMemset, hipMemcpy, hipLaunchKernel).
   - The 100% CPU spin may be in the first blit dispatch triggered by such a call in our test program.

**Revised suspected causes (re-ranked after empirical testing):**

1. **~~First GPU dispatch never completes~~** — CONFIRMED. `hsa_dispatch_test.c` barrier packet times out. Queue `read_dispatch_id` stays at 0 — CP never fetches the packet.

2. **~~Doorbell write width~~** — ELIMINATED. Tested both 32-bit and 64-bit doorbell writes. Both time out identically.

3. **~~HWS not activating queues~~** — DISPROVEN. HQD register dump shows `CP_HQD_ACTIVE=1`, `DOORBELL_HIT=1`. Queue IS active. (Note: running in NO_HWS mode via `sched_policy=2`, so this is direct MMIO load, not MEC firmware.)

4. **Missing WPTR polling trigger in V8 kgd_hqd_load** — PRIME SUSPECT.

#### HQD register dump breakthrough (2026-03-09)

Used `tests/hsa_mqd_debug.c` to hold a queue open; dumped live HQD registers from `/sys/kernel/debug/kfd/hqds`:

| Register | Value | Meaning |
|----------|-------|---------|
| CP_HQD_ACTIVE | 1 | Queue IS active |
| DOORBELL_EN | 1 | Doorbell enabled |
| DOORBELL_HIT | 1 | Doorbell write received by hardware |
| DOORBELL_BIF_DROP | 0 | Not set (see analysis) |
| CP_HQD_PQ_WPTR | 0 | Write pointer never updated from poll |
| PQ_RPTR | 0 | No packets processed |
| WPTR_POLL_ADDR | 0x00221038 | GPU VA for WPTR polling |
| CP_HQD_VMID | 8 | Process VMID |

**Root cause:** CP received doorbell (`DOORBELL_HIT=1`) but WPTR stays at 0. The V8 `kgd_hqd_load()` in `amdgpu_amdkfd_gfx_v8.c` reads the initial WPTR from userspace and writes it to the register once, but **never triggers the CP's WPTR polling mechanism** for subsequent updates. V9's `kgd_hqd_load()` explicitly writes `CP_PQ_WPTR_POLL_CNTL1` with a queue bitmask to trigger a one-shot WPTR poll from memory. V8 omits this entirely.

With `SLOT_BASED_WPTR=2` (required for AQL queues), the doorbell is a notification to read WPTR from `WPTR_POLL_ADDR`. Without the polling trigger, the CP sees the doorbell but never reads the updated WPTR, so `PQ_WPTR` stays at 0 and no packets are fetched.

**Key registers:**
- `CP_PQ_WPTR_POLL_CNTL` (0x3083): Global enable, with `EN` bit [31] and `PERIOD` [7:0]
- `CP_PQ_WPTR_POLL_CNTL1` (0x3084): Per-queue bitmask — writing triggers one-shot poll
- Both exist on gfx8 (`gfx_8_0_d.h:295-296`). DRM init disables EN (`gfx_v8_0.c:4556`), matching V9 behavior. V9's kgd_hqd_load only writes CNTL1 (not CNTL), suggesting CNTL1 triggers polling regardless of EN.

**Fix:** Kernel patch `0007-kfd-gfx8-enable-wptr-polling-in-hqd-load.patch` — adds `CP_PQ_WPTR_POLL_CNTL1` write to V8 kgd_hqd_load, matching V9.

**Trace diagnostic (2026-03-10):** After kernel 6.18.16-5 (DOORBELL_BIF_DROP fix), doorbells reach CP (`DOORBELL_HIT=1`) but `CP_HQD_PQ_WPTR` stays 0. Added `pr_info` to patch 0007 to log: `poll_cntl` before/after, `CNTL1` mask, `WPTR_POLL_ADDR` from MQD, initial `WPTR` register value, and `PQ_CONTROL` bits. Kernel 6.18.16-6. If polling is fundamentally broken, Phase 2 fallback: remove `SLOT_BASED_WPTR=2` and use GFX7-style direct doorbell→WPTR (doorbell value IS the WPTR, no polling needed).

#### MQD comparison: VI vs V9 (2026-03-09)

Systematic field-by-field comparison of `kfd_mqd_manager_vi.c` vs `kfd_mqd_manager_v9.c` for AQL queues. Cross-referenced against gfx_8_0 hardware register headers to determine which bits actually exist on gfx8.

**Missing from VI MQD — bits that EXIST on gfx8 hardware:**

1. **`DOORBELL_BIF_DROP`** (bit 1 of `cp_hqd_pq_doorbell_control`) — set by V9/V10/V11/V12 `update_mqd` for AQL queues. The bit exists in `gfx_8_0_sh_mask.h` and `gfx_8_1_sh_mask.h`. VI MQD never sets it. This controls how the Bus Interface handles doorbell writes for AQL queues — likely critical for the MEC to properly process wptr updates via doorbell.

2. **`UNORD_DISPATCH`** (bit 28 of `cp_hqd_pq_control`) — set unconditionally by V9 `init_mqd`. Exists in `gfx_8_0_sh_mask.h`. VI never sets it. Enables unordered dispatch (packets can be processed out of order when dependencies allow).

**Missing from VI MQD — bits that do NOT exist on gfx8 (no action):**

3. `QUEUE_FULL_EN` (bit 14 of `cp_hqd_pq_control`) — V9+ only. Not in gfx_8_0 register headers.
4. `WPP_CLAMP_EN` (bit 13 of `cp_hqd_pq_control`) — V9+ only.
5. `cp_hqd_aql_control` register — V9+ only. Not in `vi_structs.h`.
6. `cp_hqd_hq_status0` bit 14 (DISPATCH_PTR) — bit 14 is reserved on gfx8 (only bits 0-9 defined).

**Identical between VI and V9:**

- `NO_UPDATE_RPTR` and `SLOT_BASED_WPTR` (AQL-specific `cp_hqd_pq_control` bits) — both set
- `RPTR_BLOCK_SIZE = 5` — both set
- EOP buffer setup — both configure `eop_control`, `eop_base_addr`
- Doorbell offset calculation — both set `DOORBELL_OFFSET` in doorbell control
- CWSR context save/restore — both configure identically
- Quantum settings — both identical
- `cp_hqd_persistent_state` — both set `PRELOAD_REQ` and `PRELOAD_SIZE=0x53`
- `cp_hqd_iq_rptr = 1` for AQL — both set

**Other VI vs V9 differences (not AQL-specific):**

- V9 sets `CP_MQD_CONTROL__PRIV_STATE` without `MTYPE`, VI sets both `PRIV_STATE` and `MTYPE_UC`
- V9 sets `IB_EXE_DISABLE`, VI sets `IB_ATC` and `MTYPE` in `cp_hqd_ib_control`
- V9 sets `cp_hqd_iq_timer = 0`, VI sets ATC and MTYPE bits in it
- These reflect legitimate V8↔V9 memory model differences (ATC/MTYPE vs UTCL2)

#### Attack plan (2026-03-09)

**Priority 1: Kernel patch — add missing AQL bits to VI MQD manager**

Patch `kfd_mqd_manager_vi.c` to add two missing AQL-specific bits:

```c
// In __update_mqd(), within the AQL format block:
if (q->format == KFD_QUEUE_FORMAT_AQL) {
    m->cp_hqd_pq_control |= CP_HQD_PQ_CONTROL__NO_UPDATE_RPTR_MASK |
            2 << CP_HQD_PQ_CONTROL__SLOT_BASED_WPTR__SHIFT;
    m->cp_hqd_pq_doorbell_control |= 1 <<
            CP_HQD_PQ_DOORBELL_CONTROL__DOORBELL_BIF_DROP__SHIFT;  // NEW
}

// In init_mqd(), add UNORD_DISPATCH to pq_control init:
m->cp_hqd_pq_control = 5 << CP_HQD_PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT |
        CP_HQD_PQ_CONTROL__UNORD_DISPATCH_MASK;  // NEW
```

Rationale: `DOORBELL_BIF_DROP` is set for AQL queues on every generation from V9 through V12. The bit exists on gfx8 hardware but VI's MQD manager never sets it. This is the most likely cause of the MEC firmware not activating AQL queues — without this bit, the doorbell/wptr mechanism for AQL may not function correctly, preventing the CP from fetching packets.

`UNORD_DISPATCH` is set unconditionally by V9 `init_mqd`. While less likely to be the root cause, it exists on gfx8 and should be set for correct AQL behavior.

**Priority 1b: ROCR patch — 32-bit doorbell write for gfx8** (DONE)

ROCR unconditionally writes 64 bits to the doorbell via `*(signal_.hardware_doorbell_ptr) = uint64_t(value)` (`amd_aql_queue.cpp:478`). On gfx8, doorbells are 4 bytes wide with 4-byte stride — the 8-byte write overflows into the adjacent queue's doorbell slot. Patch adds `legacy_doorbell_` flag (ISA major < 9) and uses 32-bit store for gfx8. Created `hsa-rocr/0003-use-32bit-doorbell-write-for-gfx8.patch`, bumped hsa-rocr-polaris to 7.2.0-2.

**Priority 2: MAP_QUEUES packet format audit** — ELIMINATED

Full audit of `kfd_packet_manager_vi.c` vs `kfd_packet_manager_v9.c` confirmed the MAP_QUEUES packet is structurally correct for VI. Differences are legitimate V8↔V9 hardware variations (no `extended_engine_sel`, 21-bit vs 26-bit doorbell offset field). `queue_type`, `engine_sel`, and all address fields are correctly set. Not a cause.

**Priority 3: Test with `sched_policy=1` (NO_HWS) if patches fail**

Bypasses HWS entirely, loads queues directly via MMIO registers (`kgd_hqd_load` in `amdgpu_amdkfd_gfx_v8.c`). The direct-load path explicitly sets `DOORBELL_EN` and `CP_HQD_ACTIVE` — fields that HWS normally manages. If dispatch works in NO_HWS but not CPSCH, the problem is in HWS queue activation, not MQD contents. Requires reboot with `amdgpu.sched_policy=1`.

**Priority 4: MEC firmware version investigation** — ELIMINATED

Binary analysis of Polaris 12 MEC firmware (`polaris12_mec_2.bin`, v0x2da, 65642 dwords). The MEC uses a proprietary undocumented microcode ISA (not GCN shader code). Key findings:

- **Register access patterns are nearly identical between Polaris 12 and Vega 10 MEC firmware.** Both touch `DOORBELL_CONTROL` (1 read, 2 writes), `PQ_WPTR` (3 reads, 13 writes), `PQ_CONTROL` (2-3 reads, 2 writes), `HQD_ACTIVE` (0 reads, 3 writes). The queue activation logic is functionally the same across generations.
- Both Polaris firmware variants (v0x2c1 and v0x2da) have identical register access patterns — the 72% binary diff is code motion/optimization, not functional changes to queue handling.
- The MEC firmware reads the full `CP_HQD_PQ_DOORBELL_CONTROL` register from the MQD — it will process whatever bits the kernel sets, including `DOORBELL_BIF_DROP`.
- **Conclusion:** The firmware is not the issue. It handles queue activation identically to Vega. The missing MQD bits (our kernel patch) are the problem — the firmware is faithfully loading what the kernel wrote, which was incomplete.

### Phase 3b: Direct Doorbell WPTR — Conditional SLOT_BASED_WPTR (CURRENT)

**Root cause:** WPTR polling is broken on gfx8 without PCIe atomics. With `SLOT_BASED_WPTR=2`, the CP polls `cp_hqd_pq_wptr_poll_addr` to read the write pointer. That address is a CPU heap VA (`&amd_queue_.write_dispatch_id`). With `PQ_ATC=0` and no UTCL2 (gfx8), the CP cannot translate it via GPUVM. Result: `CP_HQD_PQ_WPTR` stays at 0, no packets fetched — even though `DOORBELL_HIT=1`.

**Fix:** Conditional `SLOT_BASED_WPTR`: use 0 (direct doorbell) when platform lacks PCIe atomics, 2 (memory polling) when atomics are available. DRM's own gfx8 compute queues use `SLOT_BASED_WPTR=0` with `WDOORBELL32(ring->doorbell_index, lower_32_bits(ring->wptr))` — proven hardware path.

**Changes:**

1. **Kernel patch 0008** (`kfd_mqd_manager_vi.c`): Conditional `SLOT_BASED_WPTR` in `__update_mqd()`. Module parameter `gfx8_wptr_poll` (0=auto, -1=force direct, 1=force poll). Keys off `mm->dev->kfd->pci_atomic_requested`.

2. **Kernel patch 0007 removed from PKGBUILD.** WPTR polling register writes in `amdgpu_amdkfd_gfx_v8.c` are unnecessary: with `SLOT_BASED_WPTR=0` there's no polling, and with `SLOT_BASED_WPTR=2` the upstream code handles it.

3. **ROCR patch 0003 modified** (`amd_aql_queue.cpp`): Conditional WPTR encoding in `StoreRelaxed()`. With `no_atomics_` (SLOT_BASED_WPTR=0), doorbell value is dword offset = `(dispatch_id * 16) & mask`. Without `no_atomics_` (SLOT_BASED_WPTR=2), doorbell value is dispatch index (notification only). Also moves `no_atomics_` field + `NoPlatformAtomics()` agent detection from patch 0004 into 0003 (first consumer).

4. **ROCR patch 0004 modified**: Removes agent detection hunks (now in 0003). Adds `last_scanned_idx_` to constructor. All bounce buffer logic unchanged.

5. **PKGBUILDs**: kernel 6.18.16-7 (swap 0007→0008), hsa-rocr 7.2.0-4 (updated 0003/0004).

**Detection summary:**
| Layer | Flag | Source |
|-------|------|--------|
| Kernel | `kfd->pci_atomic_requested` | `amdgpu_amdkfd_have_atomics_support()` |
| Kernel | `gfx8_wptr_poll` | Module parameter (override) |
| ROCR | `no_platform_atomics_` (GpuAgent) | iolink `NoAtomics64bit` topology flag |
| ROCR | `no_atomics_` (AqlQueue) | Per-queue cache from agent |

**HQD register verification (2026-03-11):** After booting kernel 6.18.16-7 + hsa-rocr 7.2.0-4, used `hsa_wptr_debug` test tool (rings doorbell then sleeps 10s for register capture). `/sys/kernel/debug/kfd/hqds` dump of the user queue:

| Register | Value | Meaning |
|----------|-------|---------|
| CP_HQD_PQ_WPTR | 0x00000010 | 16 dwords = 1 AQL packet submitted |
| CP_HQD_PQ_RPTR | 0x00000010 | **RPTR caught up — packet consumed by CP** |
| DOORBELL_HIT | 1 | Doorbell ring received |
| PQ_CONTROL | 0x1801850d | SLOT_BASED_WPTR=0, NO_UPDATE_RPTR=1, UNORD_DISPATCH=1 |

**CP IS processing packets.** The direct doorbell WPTR fix works — RPTR matches WPTR, meaning the CP fetched and executed the barrier packet. The dispatch test still times out because:

1. **`read_dispatch_id` stays 0** — `NO_UPDATE_RPTR=1` prevents CP from writing RPTR back to `rptr_report_addr` (CPU VA `&amd_queue_.read_dispatch_id`). Same root cause as the WPTR poll address: it's a CPU heap VA unreachable via GPUVM on gfx8 without UTCL2. The hardware RPTR register advances (confirmed above), but ROCR never sees it.

2. **Signal completion fails** — CP's AtomicOp TLP for signal decrement is dropped by Westmere root complex (no PCIe atomics). This was the known issue the bounce buffer was designed for.

**Status:** Phase 3b VERIFIED. Kernel and ROCR patches confirmed working via HQD registers. Phase 3c needed to solve `read_dispatch_id` before bounce buffer can function.

### Phase 3c: read_dispatch_id — GPU-Visible RPTR Buffer (CURRENT)

**Problem:** `read_dispatch_id` stays 0, blocking signal completion, queue space detection, and scratch reclaim.

**Root cause discovery (iterative):**
1. First attempt: ROCR-only GPU-visible buffer with `NO_UPDATE_RPTR=1`. Allocated `rptr_gpu_buf_` from `system_allocator()` (kernarg pool, CPU VA = GPU VA) and pointed `rptr_report_addr` at it. **Result: `*rptr_gpu_buf_` stayed 0.** With `SLOT_BASED_WPTR=0` + `NO_UPDATE_RPTR=1`, gfx8 CP does NOT write to `rptr_report_addr` at all.
2. Key insight: DRM gfx8 compute queues use `SLOT_BASED_WPTR=0` WITHOUT `NO_UPDATE_RPTR`. With `NO_UPDATE_RPTR=0`, CP writes RPTR (dword offset) to `rptr_report_addr` per `RPTR_BLOCK_SIZE`.
3. Final approach: kernel patch 0008 clears `NO_UPDATE_RPTR` when `SLOT_BASED_WPTR=0` (no atomics). ROCR converts dword offsets to monotonic dispatch IDs.

**Approach: Kernel + ROCR combined fix.**
- **Kernel**: Patch 0008 makes `NO_UPDATE_RPTR` conditional — only set when `SLOT_BASED_WPTR=2` (has atomics). Without atomics (`SLOT_BASED_WPTR=0`), both flags are cleared, matching DRM gfx8 behavior.
- **ROCR**: `rptr_gpu_buf_` allocated from `system_allocator()` (kernarg pool). CP writes dword offset to it. `UpdateReadDispatchId()` converts to monotonic dispatch ID with wrap-around tracking (`last_rptr_dwords_`).

**Dword-to-dispatch-ID conversion:**
- CP writes RPTR as dword offset wrapping at `ring_size * 16` (each AQL packet = 64 bytes = 16 dwords)
- `UpdateReadDispatchId()` tracks delta from last seen offset, divides by 16, accumulates into `read_dispatch_id`
- Handles ring wrap-around correctly (delta can't exceed ring size)

**Changes:**

1. **Kernel patch 0008** (modify): Make `NO_UPDATE_RPTR_MASK` conditional — only set in the `SLOT_BASED_WPTR=2` branch (has atomics). When `SLOT_BASED_WPTR=0` (no atomics), neither flag is set.

2. **ROCR patch 0003** (extend): Add `rptr_gpu_buf_`, `last_rptr_dwords_`, `UpdateReadDispatchId()`. Allocate 4KB from `system_allocator()`. `UpdateReadDispatchId()` does dword→dispatch_id conversion with wrap tracking. Called on every `LoadReadIndex*`.

3. **ROCR patch 0004** (extend): Add `UpdateReadDispatchId()` call at top of `ProcessCompletions()`.

4. **PKGBUILDs:** kernel pkgrel 7→8, ROCR pkgrel 4→5.

**Risk: Low.** `NO_UPDATE_RPTR=0` + `SLOT_BASED_WPTR=0` is the proven DRM gfx8 compute queue configuration. Same `system_allocator()` path used for ring buffers (confirmed working). Wrap-around tracking is straightforward (ring buffer guarantees delta < ring_size).

### Phase 4: RPTR Bounce Buffer — Software Signal Completion

**Root cause (confirmed):** GFX8 CP writes completion signals to system memory using PCIe AtomicOp TLPs. Westmere root complex drops them silently. The bounce buffer depends on `read_dispatch_id` advancing to detect completion — blocked until Phase 3c resolves RPTR reporting.

**Solution:** ROCR patch `0004-rptr-bounce-buffer-for-no-atomics-platforms.patch` adds a per-queue RPTR bounce buffer:
1. `ScanNewPackets()` — called from doorbell write path (`StoreRelaxed`), reads newly submitted AQL packets and saves their `completion_signal` handles
2. `ProcessCompletions()` — called from BusyWaitSignal polling loop, checks `read_dispatch_id`, decrements saved signals from CPU when packets complete
3. `ProcessAllBounceBuffers()` — static method iterating all bounce-buffer queues, called from `default_signal.cpp`
4. Forces `g_use_interrupt_wait = false` for no-atomics platforms (event mailbox writes fail the same way)
5. No-atomics detection via iolink topology (`atomic_support_64bit` flag), same mechanism SDMA uses

**Files modified:** `amd_gpu_agent.{h,cpp}`, `amd_aql_queue.{h,cpp}`, `default_signal.cpp`, `runtime.cpp`

**Key design decisions:**
- Don't modify packets in ring buffer — let CP's failed signal write be harmless, bounce buffer handles it from CPU
- Use existing iolink `NoAtomics64bit` flag (kernel already sets it when `pci_atomic_requested=false`)
- Per-queue `bounce_lock_` mutex for thread safety; static `bounce_list_lock_` for global queue registry
- Double-checked locking pattern in `ScanNewPackets()` for fast-path (no-op when no new packets)

**Status: VERIFIED.** Barrier dispatch completes successfully after fixing `completion_signal` offset bug (was 24, correct is 56 — same for all AQL packet types). Full pipeline: CP processes packet → RPTR written to GPU buffer → `UpdateReadDispatchId()` converts dword→dispatch_id → `ProcessCompletions()` decrements signal from CPU → wait returns.

### Test expectations (Phase 3c+4)

After booting kernel 6.18.16-8 with ROCR 7.2.0-5:

1. **HQD PQ_CONTROL**: `NO_UPDATE_RPTR=0`, `SLOT_BASED_WPTR=0` (both cleared)
2. **`*rptr_gpu_buf_`**: non-zero after CP processes packets — dword offset (multiples of 16 for single packets)
3. **`read_dispatch_id`**: advances to match packet count via `UpdateReadDispatchId()` conversion
4. **Signal completion**: bounce buffer `ProcessCompletions()` decrements signals from CPU → barrier/dispatch returns
5. **`hsa_dispatch_test`**: barrier packet completes without timeout (HSA_ENABLE_INTERRUPT=0)

**Verification commands:**
```bash
# Quick: does dispatch complete?
HSA_ENABLE_INTERRUPT=0 timeout 10 ./tests/hsa_dispatch_test

# Debug: inspect RPTR buffer directly
HSA_ENABLE_INTERRUPT=0 timeout 10 ./tests/hsa_rptr_debug

# Multi-packet: exceed RPTR_BLOCK_SIZE threshold
HSA_ENABLE_INTERRUPT=0 timeout 10 ./tests/hsa_rptr_multi

# HQD registers: confirm NO_UPDATE_RPTR=0
sudo ./tests/hsa_hqd_check
```

**If `*rptr_gpu_buf_` stays 0:** CP still not writing RPTR. Check HQD PQ_CONTROL for unexpected bits. May need `RPTR_BLOCK_SIZE` to be non-zero (currently 5, set by patch 0006).

**If `read_dispatch_id` advances but signal never fires:** Bounce buffer scan/completion logic issue. Check `ScanNewPackets()` captures the signal, `ProcessCompletions()` sees the advanced index.

### Phase 5: HIP Runtime Smoke Test (NEXT)

**Goal:** Verify HIP userspace works end-to-end on gfx803, from device query through kernel dispatch. This determines whether we need a custom `hip-runtime-amd-polaris` package or can use stock Arch `hip-runtime-amd`.

**Prerequisite state (all VERIFIED):**
- HSA init, device enumeration, queue creation ✓
- AQL barrier dispatch with signal completion ✓
- Kernel: direct doorbell WPTR, RPTR reporting, bounce buffer signals ✓

**Test sequence (incremental, each depends on prior):**

| Step | Test | What it exercises | Pass criteria |
|------|------|-------------------|---------------|
| 5a | `hipInit(0)` + `hipGetDeviceCount` | HIP runtime init, CLR device enumeration | Returns `hipSuccess`, count ≥ 1 |
| 5b | `hipGetDeviceProperties(&props, 0)` | Device property population, ISA query | `props.gcnArchName` = `"gfx803"` |
| 5c | `hipSetDevice(0)` + `hipMalloc` + `hipFree` | VRAM allocation via KFD | No errors |
| 5d | `hipMemset(ptr, 0x42, N)` | Blit kernel dispatch (Fill shader on utility queue) | Memory filled correctly |
| 5e | `hipMemcpy(host, dev, N, D2H)` | Blit kernel dispatch (Copy shader) | Data matches |
| 5f | Trivial kernel launch (`__global__ void set(int *p) { *p = 42; }`) | Full dispatch: code object load, kernarg setup, kernel dispatch packet, scratch, signal | Output = 42 |

**Known risk areas:**

1. **CLR `runtimeRocSupported()`** (`device.hpp:1457`): blocks gfx8 for OpenCL but HIP path is open. Should not affect steps 5a-5f. If it does, we need `hip-runtime-amd-polaris`.

2. **Scratch memory allocation**: gfx8 scratch path differs from gfx9+ (`need_queue_scratch_base` = false for gfx8). Phase 3 review ELIMINATED this as a code issue, but we haven't tested it under real dispatch. Step 5f exercises this.

3. **Code object loading**: `hipcc --offload-arch=gfx803` must produce valid code objects. Arch `rocm-llvm` has gfx803 targets (verified Phase 1). But `amd_comgr` links against ROCR — if our patches changed any symbol visibility, comgr could fail at runtime.

4. **Blit kernel ISA**: Pre-compiled gfx8 blit binaries exist in `amd_blit_shaders.h` (V1 header). Steps 5d/5e confirm they actually execute correctly on hardware.

5. **Multi-queue interaction**: HIP creates an internal utility queue (for blits) and a user queue. Both share the doorbell aperture with 4-byte stride. Our 32-bit doorbell write (patch 0003) prevents overflow into adjacent slots. Steps 5d/5e implicitly test both queues working simultaneously.

6. **Bounce buffer under load**: Steps 5d-5f generate multiple AQL packets with completion signals. `ProcessCompletions()` must handle rapid signal accumulation and FIFO completion ordering.

**Build & test commands:**
```bash
# Compile test (uses stock hipcc from Arch)
hipcc --offload-arch=gfx803 -o tests/hip_smoke tests/hip_smoke.cpp

# Run (HSA_ENABLE_INTERRUPT=0 forces polling — required for bounce buffer)
HSA_ENABLE_INTERRUPT=0 ./tests/hip_smoke
```

**Failure diagnosis:**
- `hipInit` fails → CLR or ROCR init issue; check `HSA_ENABLE_SDMA=0` env var, `strace` for failing ioctls
- `hipGetDeviceProperties` fails → CLR device property population; may hit image dimension query (patched in 0002)
- `hipMalloc` fails → KFD memory allocation; check `dmesg` for KFD errors
- `hipMemset` hangs → Blit kernel dispatch broken; use `hsa_hqd_check` to see if CP is processing, `hsa_rptr_debug` for RPTR state
- `hipMemset` returns but data wrong → Blit shader ISA issue or memory coherency; dump buffer contents
- Kernel launch hangs → Code object load failure or kernarg issue; add `AMD_LOG_LEVEL=4` for CLR debug output
- Kernel launch returns but wrong result → Kernel ISA execution or memory mapping issue

**Decision point after Phase 5:**
- If 5a-5f pass with stock `hip-runtime-amd` → no custom CLR package needed, proceed to rocBLAS
- If 5a fails → need `hip-runtime-amd-polaris` with CLR gate patch
- If 5d-5f fail → deeper dispatch debugging, may need additional ROCR patches

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
| 2026-03-11 | **WPTR polling broken on no-atomics gfx8** | SLOT_BASED_WPTR=2 polls CPU VA via GPUVM — unreachable without UTCL2. Conditional: use direct doorbell (=0) without atomics, proven by DRM gfx8 compute queues. |
| 2026-03-11 | **Remove kernel patch 0007** | WPTR polling register writes unnecessary: direct doorbell mode doesn't poll, memory polling mode handled by upstream. Replace with conditional 0008. |
| 2026-03-11 | **Move no_atomics_ to ROCR patch 0003** | Doorbell WPTR encoding (0003) is the first consumer of no_atomics_; patch 0004 (bounce buffer) depends on it. Keeps patches independently testable. |
| 2026-03-11 | **CP packet processing VERIFIED** | HQD registers show RPTR=WPTR=0x10 — CP consumed the barrier packet. Direct doorbell WPTR fix confirmed working. |
| 2026-03-11 | **read_dispatch_id broken by NO_UPDATE_RPTR** | `rptr_report_addr` is CPU heap VA, unreachable via GPUVM (same root cause as WPTR poll). DRM gfx8 uses VRAM writeback buffer instead. Bounce buffer blocked until resolved. |
| 2026-03-11 | **GPU-visible RPTR buffer via system_allocator()** | Allocate rptr_report_addr from kernarg pool (CPU VA = GPU VA). Same proven path as ring_buf_ and pm4_ib_buf_. Copies to amd_queue_.read_dispatch_id on LoadReadIndex* for transparent consumer compatibility. |
| 2026-03-11 | **NO_UPDATE_RPTR=1 incompatible with SLOT_BASED_WPTR=0** | Testing showed `*rptr_gpu_buf_` stayed 0 — CP does not write to rptr_report_addr with NO_UPDATE_RPTR=1 + SLOT_BASED_WPTR=0. DRM gfx8 uses NO_UPDATE_RPTR=0 in this mode. |
| 2026-03-11 | **Kernel 0008: conditional NO_UPDATE_RPTR** | Both NO_UPDATE_RPTR and SLOT_BASED_WPTR now only set when platform has atomics. No-atomics path gets neither flag, matching DRM gfx8 compute queue config. CP writes dword offset RPTR to rptr_report_addr. |
| 2026-03-11 | **ROCR: dword→dispatch_id conversion** | With NO_UPDATE_RPTR=0, CP writes wrapping dword offset (not dispatch ID). ROCR tracks delta with wrap-around (last_rptr_dwords_), divides by 16 to get AQL packet count, accumulates into monotonic read_dispatch_id. |
| 2026-03-11 | **Phase 3c VERIFIED: read_dispatch_id advances** | After kernel 6.18.16-8 boot, `read_dispatch_id` correctly reads 1 after barrier packet. GPU-visible RPTR buffer + dword conversion working. |
| 2026-03-11 | **Bounce buffer signal offset bug** | `ScanNewPackets()` used offset 24 for barrier `completion_signal`, correct is 56. All AQL packet types have `completion_signal` at offset 56. Fixed, signal now decremented correctly. |
| 2026-03-11 | **Phase 3c+4 VERIFIED: barrier dispatch completes** | `hsa_dispatch_test` returns `signal=0` — full pipeline working: CP → RPTR buffer → dispatch_id → bounce buffer → signal. |

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

## Phase 4: CPU Cache Coherency for Non-Coherent Platforms

### Problem

On gfx8 without ATC/IOMMU (Polaris on Westmere Xeon), GPU PCIe writes to system memory bypass CPU cache. CPU reads see stale cached data. This breaks:
- `hsa_memory_copy` VRAM→system (ROCR path)
- HIP `hipMemcpy` D2H (ROCclr staging buffer path)
- Any GPU→system DMA where CPU reads the result

### Root Cause

On x86 without IOMMU/ATC, GPU PCIe writes to system memory bypass CPU cache (PCIe is not a cache-coherent participant for writes on these platforms). CPU reads after GPU writes return stale cached data.

### Key Discovery: HIP D2H Path Goes Through ROCR

Initial assumption was that ROCclr's `hipMemcpy(D2H)` used its own blit shader dispatch path, separate from ROCR. **This was wrong.** Code trace reveals:

```
hipMemcpy(D2H)
  → ihipMemcpyCommand() [hip_memory.cpp:576]
    → VirtualGPU::submitReadMemory() [rocvirtual.cpp:2161]
      → DmaBlitManager::readBuffer() [rocblit.cpp:53]
        → hsaCopyStagedOrPinned() [rocblit.cpp:691]
          → rocrCopyBuffer() [rocblit.cpp:740]
            → hsa_amd_memory_async_copy() [hsa_ext_amd.cpp:253]
              → Runtime::CopyMemory() [runtime.cpp:591]  ← OUR PATCH 0005 IS HERE
                → GpuAgent::DmaCopy()  (GPU SDMA engine)
```

ROCclr allocates a staging buffer, then calls **ROCR's `hsa_amd_memory_async_copy()`** for the actual GPU→staging DMA. This goes through `CopyMemory()` where our patch 0005 adds `FlushCpuCache()`. After the flush, ROCclr does `memcpy(host, staging, size)` which now sees clean data.

**ROCR patch 0005 alone fixes both ROCR and HIP paths.** No separate ROCclr patch needed.

### Patches

**ROCR patch 0005** (`0005-flush-cpu-cache-after-gpu-writes-to-system-memory.patch`): `FlushCpuCache()` after GPU→system DmaCopy in `CopyMemory()`, conditioned on `NoPlatformAtomics()`. This is the **primary fix** — one clflush per cache line for the copy destination. Fixes both `hsa_memory_copy` and `hipMemcpy` D2H.

**Kernel patch 0009** (`0009-drm-amdgpu-honor-uncached-flag-in-ttm-caching-mode.patch`): Check `AMDGPU_GEM_CREATE_UNCACHED` before `CPU_GTT_USWC` in `amdgpu_ttm_tt_new()`, use `ttm_uncached`. Defense-in-depth — makes kernarg pool memory truly uncached on CPU, so cache flushes become unnecessary. Flag propagation chain verified:

```
ROCR: mem_flag_.ui32.Uncached = 1  (amd_memory_region.cpp:109, kernarg regions)
  → thunk: KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED  (fmm.c:1822)
    → kernel KFD: flags passed through  (kfd_chardev.c:1141)
      → amdgpu_amdkfd_gpuvm.c:1755: alloc_flags |= AMDGPU_GEM_CREATE_UNCACHED
        → amdgpu_object.c:685: bo->flags = bp->flags
          → amdgpu_ttm.c:1134: OUR PATCH checks abo->flags → ttm_uncached
```

**Open question:** `hsa_cache_timing` test still shows 1.0x ratio (WB-cached behavior) after kernel 6.18.16-9 with patch 0009. Flag propagation is verified correct in source. Possible causes: KFD mmap path may bypass TTM pgprot, or test allocates via a path that doesn't trigger `amdgpu_ttm_tt_new()`. Not blocking — ROCR 0005 is the working fix. Worth investigating later for correctness.

### Verification Results (2026-03-12)

```
Kernel: 6.18.16-9-lts-rocm-polaris (patch 0009 applied)
ROCR:   hsa-rocr-polaris 7.2.0-6 (patches 0001-0005)
HIP:    stock Arch extra/ hipcc + libamdhip64 7.2.0

hsa_dispatch_test:       PASS  (barrier dispatch + signal completion)
hsa_memcopy_test:        PASS  (sys→sys, sys→vram→sys round-trip, memory_fill readback)
hsa_memcopy_test2:       PASS  (VRAM→system round-trip via ROCR path)
hip_smoke 5a (init):     PASS
hip_smoke 5b (props):    PASS  (AMD Radeon Pro WX 2100, gfx803)
hip_smoke 5c (malloc):   PASS
hip_smoke 5d (memcpy):   PASS  ← PREVIOUSLY FAILING, NOW FIXED
hip_smoke 5e (memset):   PASS
hip_smoke 5f (kernel):   PASS  (simple addition kernel returns 42)
hsa_cache_timing:        1.0x (kernarg pool still appears WB-cached — see open question)
```

### Pitfall: inline x86 assembly crashes ROCR

`hsa_memcopy_test` previously segfaulted during `hsa_init()` at GPU VRAM addresses (`SEGV_ACCERR` at `0x4000xxxxxx`). Root cause: inline `asm volatile("clflush (%0)")` in the binary. ROCR maps the process `.text` section into GPU-accessible address space during initialization; the inline assembly changes binary layout enough to cause a GPU VA mapping failure. Fix: removed the clflush diagnostic (no longer needed after patch 0005). All tests pass without it.

### Status

- [x] ROCR patch 0005: FlushCpuCache in CopyMemory — **THE FIX** (hsa-rocr-polaris 7.2.0-6)
- [x] Kernel patch 0009: honor UNCACHED in TTM (defense-in-depth, 6.18.16-9)
- [x] HIP hipMemcpy D2H: **VERIFIED WORKING**
- [x] HIP kernel launch: **VERIFIED WORKING**
- [x] hsa_memcopy_test: **ALL PASS** (was segfault from inline asm, now fixed)
- [ ] Investigate why cache_timing still shows WB despite kernel 0009 (non-blocking)

## Phase 5: HIP Runtime Smoke Test (COMPLETE)

All 6 steps pass with stock Arch `hip-runtime-amd` — no custom CLR package needed.
See Phase 4 verification results for full test output.

## Phase 6: rocBLAS (COMPLETE)

**Built and installed:** `rocblas-gfx803` 7.2.0-1 (gfx803-only, 55MB package)

**SGEMM verified on hardware:**
```
rocBLAS SGEMM 64x64x64: PASS
```
Requires `rocblas_set_atomics_mode(handle, rocblas_atomics_allowed)` — see Phase 2d caveat.

### Package inventory (all installed and verified)

| Package | Version | Patches | Status |
|---------|---------|---------|--------|
| `linux-lts-rocm-polaris` | 6.18.16-9 | 0004-0006, 0008-0009 | INSTALLED |
| `hsa-rocr-polaris` | 7.2.0-7 | 0001-0005 | INSTALLED |
| `rocblas-gfx803` | 7.2.0-2 | 0001-0002 | INSTALLED |
| `llama-cpp-rocm-polaris` | b7376-1 | none | INSTALLED (GPU inference blocked) |
| `hip-runtime-amd` | 7.2.0 (stock Arch) | none needed | INSTALLED |

### What works end-to-end

- HSA: init, device enumeration, queue creation, barrier dispatch, signal completion
- HSA: memory copy (sys→sys, sys→vram→sys, memory_fill readback)
- HIP: init, device props, malloc/free, memcpy (H2D + D2H ≤8MB), memset, kernel launch
- HIP: 300 consecutive 4KB memcpy round-trips (queue wraparound verified)
- rocBLAS: SGEMM on Tensile kernels (atomics_allowed default patched)
- llama.cpp: CPU-only inference works (Qwen2.5-0.5B at 10.4 t/s)

## Phase 5: Large Transfer & Inference Support (IN PROGRESS)

### Problem: hipMemcpy hangs at 16MB+

**Root cause:** AQL queue fill-up during chunked DMA transfers.

The BlitKernel (used for hipMemcpy on no-SDMA platforms) splits large transfers into chunks, each becoming an AQL packet. When the queue fills, `AcquireWriteIndex()` spins waiting for `read_dispatch_id` to advance:

```cpp
// amd_blit_kernel.cpp:855
while (write_index + num_packet - queue_->LoadReadIndexRelaxed() > queue_->public_handle()->size)
    os::YieldThread();
```

`LoadReadIndexRelaxed()` calls `UpdateReadDispatchId()` which reads `rptr_gpu_buf_` — a GPU-visible buffer where the CP writes RPTR. But on non-coherent platforms, GPU PCIe writes bypass CPU cache, so the CPU reads stale data.

**Fix applied (pkgrel 7):** Added `FlushCpuCache()` in `UpdateReadDispatchId()` before reading `rptr_gpu_buf_`.

**Result:** Threshold moved from 16MB → testing in progress. 64MB H2D confirmed working.

### Codepath audit: patch 0005 overfitted to sync path

**Discovery:** Patch 0005 only covers the sync 3-arg `CopyMemory(dst, src, size)` in `runtime.cpp`. The primary HIP hipMemcpy path uses the async 7-arg `CopyMemory()` → `hsa_amd_memory_async_copy()`, which is completely unpatched.

#### What patch 0005 covers (PROTECTED)

| Path | Location | Notes |
|------|----------|-------|
| Sync `hsa_memory_copy` | `runtime.cpp:574-585` | Flushes dst after DmaCopy |
| GPU→GPU via system staging | `runtime.cpp:597-607` | Flushes temp buffer between DMAs |

#### What patch 0005 misses (VULNERABLE)

| Path | Location | Impact |
|------|----------|--------|
| Async `hsa_amd_memory_async_copy` | `runtime.cpp:637` | **All CLR/HIP copies** |
| `CopyMemoryOnEngine` | `runtime.cpp:655` | Engine-specific async copies |
| CLR D2H staging buffer | `rocblit.cpp:736-754` | hipMemcpy D2H ≤ staging size |
| CLR D2H pinned path | `rocblit.cpp:742-748` | hipMemcpy D2H > staging size |
| BlitKernel sync 3-arg | `amd_blit_kernel.cpp:633` | After signal wait, before data read |

#### Two-layer root cause

**Layer 1 — GPU L2 not flushed (staging buffer path, 91.8% zeros):**
The staging buffer is allocated from `coarse_grain_pool` (CLR `rocvirtual.cpp:1854`, segment `kNoAtomics` → `rocdevice.cpp:1979`). Coarse-grain memory uses MTYPE=NC in GPU page tables. On gfx8, `FENCE_SCOPE_SYSTEM` release may not properly flush GPU L2 for NC system memory. Blit kernel writes stay in GPU L2; DRAM retains initialization zeros. CPU memcpy from staging reads zeros.

**Layer 2 — CPU cache stale (pinned path, 1.1% 0xBB stale):**
For transfers > `pinnedMinXferSize_` (1MB default), CLR pins the user's malloc'd buffer and DMA goes directly to it. Data mostly reaches DRAM (GPU L2 flushes for pinned/CC memory), but ~1% of CPU cache lines still hold stale values from the pre-copy `memset(0xBB)`.

#### Test evidence

| Size | Path | Result | Root Cause |
|------|------|--------|------------|
| 1MB (staging) | `hsaCopyStagedOrPinned` → staging | 91.8% zeros | GPU L2 not flushed to DRAM |
| 1MB (forced pin via `GPU_PINNED_MIN_XFER_SIZE=0`) | `hsaCopyStagedOrPinned` → pinned | 1.1% 0xBB stale | CPU cache stale |
| 64MB (default=pinned) | `hsaCopyStagedOrPinned` → pinned | 100% correct | Works (timing?) |
| H2D verified by kernel | `check_vram<<<1,1>>>` | 100% correct | H2D data reaches VRAM |

#### Key CLR code path (the gap)

```
hipMemcpy(D2H)
  → ihipMemcpy → ReadMemoryCommand → submitReadMemory
  → blitMgr().readBuffer()                    [rocblit.cpp:53-77]
  → hsaCopyStagedOrPinned()                   [rocblit.cpp:691-776]
    → rocrCopyBuffer()                        [rocblit.cpp:740] — async DMA via hsa_amd_memory_async_copy
    → gpu().Barriers().WaitCurrent()          [rocblit.cpp:744] — wait for signal
    → memcpy(hostDst, stagingBuffer, size)    [rocblit.cpp:747] — ← READS STALE DATA
```

Between WaitCurrent() and memcpy(), there is NO FlushCpuCache on the staging buffer. The async CopyMemory at runtime.cpp:637 can't flush because the DMA hasn't completed yet.

#### Performance analysis of fix options

The staging buffer (GPU L2) issue **cannot** be fixed with CPU-side `clflush` alone — data stuck in GPU L2 never reaches DRAM, so flushing CPU cache just reads zeros faster. Must fix at the GPU memory type level.

| Approach | GPU Write Cost | CPU Read Cost | Fixes Staging? | Fixes Pinned? |
|----------|---------------|---------------|----------------|---------------|
| **Fine-grain staging** (MTYPE NC→CC) | ~0% (L2 cached, release fence already present) | ~0% (DMA snoops CPU cache) | **Yes** | No |
| **UC staging** (bypass GPU L2) | **Bad** — every write = PCIe txn | ~0% | Yes | No |
| **clflush on staging after wait** | 0% | ~10%/chunk | **No** — data not in DRAM | Yes |
| **clflush on pinned after wait** | 0% | proportional to size | N/A | Yes |
| **Force pinning (no staging)** | 0% | varies | Bypasses issue | Partial (1.1% stale at 1MB) |

**Key insight:** Fine-grain memory on gfx8 without PCIe atomics still works for non-atomic DMA. The `FENCE_SCOPE_SYSTEM` release in the blit kernel dispatch packet already flushes GPU L2 — but only for CC (coherent) MTYPE pages, not NC (non-coherent). Changing the staging buffer from coarse-grain (NC) to fine-grain (CC) makes the existing release fence effective. Zero additional per-transfer overhead.

#### Chosen approach: Fine-grain staging + clflush defense-in-depth

**CLR patch (single patch, two changes):**

1. **Change staging buffer segment** `kNoAtomics` → `kAtomics` in `rocvirtual.cpp:1854`
   - Makes staging buffer use fine-grain pool (CC MTYPE)
   - GPU blit kernel writes are L2-cached with coherency
   - Release fence flushes L2 → data reaches DRAM → DMA snoops CPU cache
   - **Zero additional overhead** — same write pattern, fence already exists

2. **Add `clflush` on staging/pinned buffer after DMA wait** in `rocblit.cpp:744-747`
   - Defense-in-depth for CPU cache stale lines (handles pinned path's 1.1% issue)
   - Conditional on device lacking platform atomic support (no overhead on coherent platforms)
   - ~0.5ms per 1MB staging chunk (staging is max 1MB, so bounded)
   - For pinned path: proportional to chunk size (32MB default), but these are already large DMA-bound transfers where clflush is <10% overhead

**Total estimated overhead:** <1% for fine-grain staging (dominant path). <10% additional for pinned path clflush on large transfers (defense only, may not be necessary).

**Keep:** ROCR patch 0005 for `hsa_memory_copy` sync path (already deployed, covers different API).

### CLR patch implementation (DONE)

Created `hip-runtime-amd-polaris` 7.2.0-1 with CLR patch `0001-use-fine-grain-staging-and-flush-cpu-cache-for-d2h.patch`:
- `rocvirtual.cpp`: staging buffer `kNoAtomics` → `kAtomics` (fine-grain pool)
- `rocblit.cpp`: `_mm_clflush` + `_mm_mfence` after `WaitCurrent()` for staging and pinned paths, conditional on `!dev().info().pcie_atomics_`

### Verification results (2026-03-13)

**Data corruption: FIXED.** Every transfer that completes returns correct data — no more zeros or stale 0xBB bytes.

**Intermittent hangs: NEW ISSUE.** Fork-based sweep test (30s timeout per size, each size in isolated child process):

```
  1MB: PASS  (1.0s)
  2MB: PASS  (1.0s)
  4MB: PASS  (1.0s)
  8MB: PASS  (1.0s)
 16MB: HANG  (killed after 30s)
 24MB: HANG  (killed after 30s)
 29MB: PASS  (1.0s)
 30MB: HANG  (killed after 30s)
 31MB: HANG  (killed after 30s)
 32MB: PASS  (1.0s)
 33MB: PASS  (1.0s)
 48MB: PASS  (1.0s)
 64MB: PASS  (1.0s)

9 pass, 0 fail, 4 hang / 13 total
```

**Key observations:**
1. **No correlation with transfer size or dispatch count** — 16MB hangs (32 dispatches) but 64MB passes (128 dispatches)
2. **Non-deterministic** — 32MB hung in earlier runs but passed here; 16MB passed earlier but now hangs
3. **GPU recovers after SIGKILL** — subsequent tests pass after a killed child
4. **Queue capacity is not the issue** — queue size is 16384 packets, max dispatches tested is 128

**Root cause hypothesis: Race in RPTR read-back path.** The `UpdateReadDispatchId()` conversion (FlushCpuCache → read `rptr_gpu_buf_` → dword delta → dispatch_id accumulation) occasionally misses a CP write, causing `queue_load_read_index_scacquire` in CLR's `rocvirtual.cpp:1156` to spin forever waiting for queue space. This is the CLR-side consumer of `read_dispatch_id`, separate from the ROCR bounce buffer.

**The hang location:** CLR `rocvirtual.cpp:1156`:
```cpp
while ((index - Hsa::queue_load_read_index_scacquire(gpu_queue_)) >= sw_queue_size) {
    amd::Os::yield();
}
```
This busy-waits for queue space. If `read_dispatch_id` doesn't advance, and enough packets have been submitted (varies by allocation/timing), the loop spins forever.

### Race condition in UpdateReadDispatchId — analysis and fix (2026-03-13)

**Root cause confirmed:** `UpdateReadDispatchId()` is called concurrently from multiple threads without synchronization:
- CLR dispatch threads (via `queue_load_read_index_scacquire` at `rocvirtual.cpp:1156`)
- Signal wait/polling threads (via `ProcessCompletions()` at `amd_aql_queue.cpp:616`)
- Scratch reclaim (via `LoadReadIndexRelaxed()` at `amd_aql_queue.cpp:958`)

The function performs a read-modify-write on `last_rptr_dwords_` (plain uint32_t) and a non-atomic load+store on `read_dispatch_id`. When two threads race:
1. Both read the same `last_rptr_dwords_` value
2. Both compute the same delta
3. One overwrites `last_rptr_dwords_`, the other re-reads and sees cur_dw == prev_dw → no-op
4. Or: both store to `read_dispatch_id` — the second store uses a stale base, advancing by less than it should
5. `read_dispatch_id` stalls → CLR busy-wait at `rocvirtual.cpp:1156` spins forever

**Fix (patch 0003, pkgrel 8):**
1. Add `SpinMutex rptr_lock_` to `AqlQueue` — serializes `UpdateReadDispatchId()` across callers
2. Replace `atomic::Load` + `atomic::Store` on `read_dispatch_id` with `atomic::Add` — safe for consumers reading outside the lock
3. Early return for `!rptr_gpu_buf_` stays outside the lock (zero overhead on gfx9+)

**Why SpinMutex over CAS:** Critical section is ~5 instructions. CAS would require re-reading `rptr_gpu_buf_` (with clflush) on retry — potentially slower. SpinMutex already used in the file (`bounce_lock_`). `ScopedAcquire<SpinMutex>` is valid per `locks.h:222`.

### RPTR_BLOCK_SIZE root cause (2026-03-13)

SpinMutex + mfence fix improved results (12/13 on one run) but hangs persisted (9/13 on next run).

**Actual root cause:** `RPTR_BLOCK_SIZE=5` in `kfd_mqd_manager_vi.c` means CP only writes RPTR to `rptr_report_addr` every 2^5=32 dwords (= 2 AQL packets). When the blit kernel submits a single dispatch packet, RPTR may not be written before the next doorbell arrives. The bounce buffer reads stale RPTR → `read_dispatch_id` doesn't advance → CLR busy-wait spins forever.

This only affects the no-atomics path. With PCIe atomics (`NO_UPDATE_RPTR=1`), CP writes dispatch_id directly (not RPTR), so RPTR_BLOCK_SIZE is irrelevant.

**Fix (kernel patch 0008 update, pkgrel 10):** Add `else` clause for no-atomics path that reduces RPTR_BLOCK_SIZE from 5→4 (every 16 dwords = 1 AQL packet). This ensures RPTR is written after every packet retirement. Combined with the ROCR SpinMutex + mfence fix in patch 0003 (pkgrel 8), this should eliminate both the race and the stale-RPTR issues.

### D2H sweep verification (2026-03-13)

**Kernel 6.18.16-10 + ROCR 7.2.0-8: D2H sweep 13/13 passed, 3 consecutive runs, zero hangs.**

The two-part fix (RPTR_BLOCK_SIZE 5→4 + SpinMutex/mfence/atomic::Add) eliminates D2H transfer hangs completely.

### H2D multi-chunk hang — new issue (2026-03-13)

**Symptom:** llama.cpp GPU inference crashes during model loading. H2D `hipMemcpyAsync` + `hipStreamSynchronize` hangs after ~50 operations when transfer sizes exceed the CLR staging buffer (1MB default).

**Reproduction:** Alternating 512B and 2MB `hipMemcpy` H2D calls. Hangs on the 5th 2MB transfer (~10th op in sequence). Uniform-size tests pass (100x 2MB works, 100x 512B works). Mixed 512B+551936B alternation works (40 ops). The trigger is **multi-chunk staging** (>1MB transfers that require 2+ staging buffer fills).

**Key observations:**
1. CLR staging buffer is 1MB. Transfers >1MB are split into chunks, each submitting AQL packets.
2. CLR tries to pin host memory first for >1MB transfers. On our non-HMM platform, pinning fails (`DmaBlitManager::getBuffer failed to pin a resource!`). Falls back to staging.
3. The staging fallback for >1MB does multiple 1MB staging chunks → multiple blit kernel AQL packets per `hipMemcpy`.
4. After enough multi-chunk operations, `hipStreamSynchronize` hangs in `hsa_signal_wait_scacquire` → `ProcessAllBounceBuffers` → `UpdateReadDispatchId`.
5. A background ROCR `AsyncEventsLoop` thread crashes at `Signal::Convert` accessing freed signal memory (`0x5fff00`), confirming a signal lifetime issue.

#### Signal lifetime race — analysis (2026-03-13)

**Core dump evidence (PID 11955):**

| Thread | Location | State |
|--------|----------|-------|
| 11955 (main) | `ProcessAllBounceBuffers` → `UpdateReadDispatchId` → `atomic::Load(rptr_gpu_buf_)` | Spinning — RPTR not advancing |
| 11960 | `AsyncEventsLoop` → `Signal::Convert` → `SharedSignal::IsValid` on `this=0x5fff00` | Accessing freed signal |
| 11959 | Same as 11960 | Crashed → `std::terminate` → `abort` |

**What `AsyncEventsLoop` monitors:** Queue management signals registered via `hsa_amd_signal_async_handler`:
- `queue_inactive_signal` — set when queue errors occur (`DynamicQueueEventsHandler`)
- `exception_signal_` — hardware exception handler
- `queue_scratch_.queue_retry` — scratch memory reclaim signal

These are **permanent per-queue signals**, created in `AqlQueue()` constructor, destroyed in `~AqlQueue()`. They are NOT copy completion signals.

**Two bugs, possibly related:**

**Bug A — Main thread hang:** `UpdateReadDispatchId()` spins reading `rptr_gpu_buf_`. RPTR doesn't advance. This is the same class of issue as the D2H hang (RPTR stalling), but only triggers with mixed-size H2D transfers. Uniform 2MB x100 works. This suggests CP writes RPTR at packet boundaries but the timing changes when packet submission patterns vary.

**Bug B — AsyncEventsLoop crash:** `SharedSignal::IsValid()` fails on address `0x5fff00`. This means `async_events_.signal_[i]` contains a stale handle pointing to freed/recycled memory. Possible causes:

1. **Queue destruction during hang cleanup:** If the main thread's hang triggers a timeout or error handler that destroys a queue, the queue's signals are freed while AsyncEventsLoop still references them. But the main thread is blocked (not cleaning up), so this is unlikely during normal hang.

2. **Our bounce buffer corrupts memory:** `ProcessCompletions()` casts `entry.signal.handle` to `amd_signal_t*` and writes via `atomic::Sub`. If a stale or wrong signal handle is in `pending_completions_`, this write corrupts arbitrary memory — potentially the signal pool that AsyncEventsLoop reads from.

3. **Signal freed via `DestroySignal`→`Release`→`doDestroySignal`:** Something calls `hsa_signal_destroy` on a signal that's still in the `async_events_` list. The `DestroySignal()` path decrements `refcount_` then calls `Release()`. If `retained_` is 1 (default), `Release()` calls `doDestroySignal()` which deletes the signal. AsyncEventsLoop doesn't hold a Retain() on the signals it monitors.

**Most likely scenario:** Bug A (RPTR stall) causes the main thread to hang. Then, during the extended hang, some internal timer or error mechanism fires that destroys a queue. The queue's signals are freed. AsyncEventsLoop, still iterating, hits the freed signal → crash. **Bug B is a symptom of Bug A.**

Evidence supporting this: the 100x uniform 2MB test works (no Bug A → no Bug B). The mixed-size test triggers Bug A → then Bug B follows.

#### Proposed fixes

**Fix for Bug A (primary — RPTR stall in mixed-size H2D):**

Need to investigate WHY RPTR stalls specifically during mixed-size transfers. Hypotheses:
- Multi-chunk staging submits multiple AQL packets; RPTR_BLOCK_SIZE=4 writes RPTR every 16 dwords (1 packet), but consecutive packets from the same doorbell ring might batch differently
- The blit kernel for staging submits both a kernel dispatch packet AND a barrier packet per chunk — doubling the packet count and potentially grouping across RPTR block boundaries
- `ScanNewPackets` is called in `StoreRelaxed` (doorbell write), but CLR may write multiple packets before ringing the doorbell once, causing ScanNewPackets to process a batch — if the batch includes intermediate signalless packets, the FIFO ordering in ProcessCompletions could stall

**Investigation plan for Bug A:**
1. Count AQL packets per hipMemcpy call at various sizes using `write_dispatch_id` delta
2. Verify RPTR advancement matches packet count (add debug logging to `UpdateReadDispatchId`)
3. Check if CLR submits barrier packets between staging chunks (would affect packet count)

**Fix for Bug B (defense-in-depth — AsyncEventsLoop crash on freed signal):**

**Option 1: Retain/Release in bounce buffer** (preferred if Bug B is independent)
```cpp
// ScanNewPackets: retain signal before saving
if (sig.handle != 0) {
  core::Signal::Convert(sig)->Retain();
  pending_completions_.push_back({idx + 1, sig});
}

// ProcessCompletions: release after decrement
core::Signal* signal = core::Signal::Convert(entry.signal);
signal->SubRelaxed(1);  // use Signal API instead of raw atomic::Sub
signal->Release();
```
This prevents signal deallocation while our bounce buffer holds a reference. `Retain()` increments `retained_` (starts at 1); `Release()` decrements it and only calls `doDestroySignal()` when it hits 0. With our extra Retain, the signal survives one `DestroySignal()`→`Release()` cycle.

Risk: `Signal::Convert()` throws if the signal is already freed. Must be called while signal is still valid (before any race window). In `ScanNewPackets`, the signal was just written by CLR to the ring buffer — guaranteed valid at scan time. In `ProcessCompletions`, we hold the Retain, so it's still valid. **Safe.**

**Option 2: Validate before access** (simpler but weaker)
```cpp
// ProcessCompletions: check before decrement
SharedSignal* shared = SharedSignal::Convert(entry.signal);
if (shared->IsValid()) {
  amd_signal_t* sig = reinterpret_cast<amd_signal_t*>(entry.signal.handle);
  atomic::Sub(&sig->value, int64_t(1), std::memory_order_release);
}
```
Prevents crash on freed signals, but TOCTOU race remains — signal could be freed between `IsValid()` and `atomic::Sub`. Also doesn't prevent the freed signal's memory from being reused for a different signal, which we'd then corrupt.

**Option 3: Use SubRelaxed API only** (minimal change)
Replace raw `atomic::Sub` with `core::Signal::Convert(entry.signal)->SubRelaxed(1)`. This uses the proper Signal API which sets `signal_.kind` appropriately and wakes waiters. Doesn't fix lifetime but integrates better with the signal subsystem.

**Recommendation:** Fix Bug A first (root cause). Then apply Option 1 as defense-in-depth for Bug B.

### Bug A root cause confirmed — CP idle stall with SLOT_BASED_WPTR=0 (2026-03-14)

**Root cause (revised):** ~~GPU L2 caching~~ DISPROVEN — signals are coherent (fine-grain pool, `KFD_IOC_ALLOC_MEM_FLAGS_COHERENT`). 14/14 barrier tests pass.

**Actual root cause:** With `SLOT_BASED_WPTR=0` (direct doorbell), the CP goes idle after evaluating an AQL barrier whose dep_signal != 0. The CP only wakes on a doorbell with a HIGHER WPTR value. Same-WPTR re-ring doesn't generate `DOORBELL_HIT`. When the bounce buffer later decrements the dep signal from CPU, the CP never re-evaluates because it's idle.

Doorbell kick with WPTR+1 moved the hang from op 4 to op 10 (proved CP wakeup works) but corrupts queue state (can't inject packets behind CLR's back).

**Evidence:**
- Uniform 2MB x100 works (no inter-call barriers — each hipMemcpy is synchronous, bounced buffer drains signal between calls)
- Mixed 512B+2MB alternation hangs at 5th→7th operation (barrier accumulates from inter-call deps)
- WaitCurrent() drain at end of hsaCopyStagedOrPinned extends hang-free window (drains some barriers from CPU) but doesn't fully fix (other code paths create barrier deps too)
- The issue ONLY affects AQL barrier evaluation by the CP hardware — CPU polling (our bounce buffer) works fine

**Bug B (signal use-after-free): FIXED** with Retain/Release in patch 0004 (pkgrel 9).

### Fix: SLOT_BASED_WPTR=2 with GPU-visible poll address (2026-03-14)

Revert to `SLOT_BASED_WPTR=2` (memory polling) for the no-atomics path. The original Phase 3b failure with WPTR polling was because the poll address (`&amd_queue_.write_dispatch_id`) was a CPU heap VA unreachable by the GPU. Fix: allocate `wptr_gpu_buf_` from kernarg pool (system_allocator), same as `rptr_gpu_buf_`. Set `Queue_write_ptr_aql` to it. The CP continuously polls this GPU-visible address, so it naturally re-evaluates stalled barriers.

**Changes:**
- Kernel 0008: add `SLOT_BASED_WPTR=2` to no-atomics else branch (keep `NO_UPDATE_RPTR=0` + `RPTR_BLOCK_SIZE=4`)
- ROCR PKGBUILD: sed-inject `wptr_gpu_buf_` allocation, Queue_write_ptr_aql routing, and StoreRelaxed write to poll buffer + doorbell notification

**Why this fixes the hang:** With WPTR=2, the CP continuously polls the WPTR address (every few hundred cycles). When we write the wptr dword value, the CP sees it and re-reads the ring buffer, re-evaluating any stalled barriers. No doorbell needed for wakeup.

**Recommendation: Option A** — minimal change, leverages existing KFD coherency infrastructure.

### Test plan — upstream-idiomatic rocrtst tests

Tests follow the `rocrtst` framework pattern: `TestBase` subclass with `SetUp()`/`Run()`/`Close()`, registered via `TEST()` macros in `main.cc`.

#### Test 1: `SignalCpuWriteGpuBarrier`

Verifies that GPU CP correctly evaluates a barrier-AND packet whose dep_signal was decremented from CPU (mimicking bounce buffer behavior).

```
Setup:
  - hsa_init, find GPU agent, create queue
  - Create signal A (value=1)
  - Create signal B (value=1)

Run:
  - Submit barrier-AND packet: dep_signal[0] = A, completion_signal = B
  - Ring doorbell
  - From CPU: hsa_signal_store_screlease(A, 0)  // decrement A
  - Wait on B with 3-second timeout

Pass: B reaches 0 (barrier resolved after CPU write to A)
Fail: Timeout (GPU CP didn't see CPU write → MTYPE issue)
```

**Acceptance criteria:** Test passes on all platforms. On no-atomics gfx8, this specifically validates that the signal pool coherency fix makes CPU writes visible to the CP's barrier evaluation.

#### Test 2: `BounceBufferBarrierChain`

Verifies that a chain of barrier-dependent dispatches works when signals are decremented by the bounce buffer (not GPU hardware).

```
Setup:
  - hsa_init, find GPU agent, create queue with bounce buffer enabled
  - Allocate ring buffer, kernarg, code object for nop kernel

Run (10 iterations):
  - Submit dispatch packet with signal S[i]
  - Submit barrier-AND packet: dep_signal[0] = S[i], completion_signal = S[i+1]
  - Ring doorbell
  - Wait on S[i+1] with 3-second timeout

Pass: All 10 barriers resolve within timeout
Fail: Any timeout (bounce buffer + barrier chain broken)
```

#### Test 3: `MixedSizeH2DStress`

End-to-end HIP test: alternating small (512B) and large (2MB) hipMemcpy H2D transfers.

```
Setup:
  - hipMalloc 4MB device buffer
  - malloc 3MB host buffer, fill with pattern

Run:
  - 20 iterations of: hipMemcpy(512B, H2D) + hipMemcpy(2MB, H2D)
  - Verify data integrity after each transfer via D2H readback

Pass: All 40 transfers complete without hang, data correct
Fail: Hang (timeout) or data corruption
```

**Acceptance criteria:** Must complete within 30 seconds on gfx8 no-atomics hardware.

#### Test 4: `SignalRetainRelease`

Unit test for the bounce buffer's Retain/Release lifecycle.

```
Setup:
  - Create signal S (value=1)
  - Signal::Convert(S)->Retain()  // bounce buffer holds reference

Run:
  - hsa_signal_destroy(S)  // normal destruction
  - Verify SharedSignal::IsValid() still returns true (Retain prevents free)
  - Signal::Convert(S)->SubRelaxed(1)  // bounce buffer decrements
  - Signal::Convert(S)->Release()  // bounce buffer releases
  - Verify signal is now freed (IsValid returns false)

Pass: Signal survives destroy while retained, freed after release
Fail: Crash or premature free
```

### Signal coherency theory DISPROVEN (2026-03-13)

**Barrier test PASSED.** `barrier_test.cpp` confirms: the GPU CP correctly reads CPU-written signal values through AQL barrier-AND packets. `hsa_signal_store_screlease(dep, 0)` from CPU → CP sees 0 → barrier resolves. Signals ARE coherent.

**GPU L2 caching is NOT the root cause.** Signal memory is allocated from fine-grain system pool which KFD maps with `KFD_IOC_ALLOC_MEM_FLAGS_COHERENT` → MTYPE=CC on GPU.

**CLR WaitCurrent drain CAUSED REGRESSION.** Adding `WaitCurrent()` at the end of `hsaCopyStagedOrPinned` for H2D made uniform 2MB transfers hang at op 4 (down from 93 without it). Reverted.

### Root cause confirmed: CP idle stall with SLOT_BASED_WPTR=0 (2026-03-15)

**The CP goes idle after processing AQL packets and evaluating an unsatisfied barrier dep.** With `SLOT_BASED_WPTR=0`, the doorbell value IS the WPTR. Re-ringing with the same WPTR value does NOT generate `DOORBELL_HIT`. The CP never re-evaluates the barrier, even after the bounce buffer decrements the dep signal from CPU.

**Evidence:**
- 14/14 barrier tests PASS (same-queue, cross-queue, delays, sub+clflush — hardware is correct)
- STALL logging: Queue C RPTR frozen at 32 dwords for 3.5M iterations while dep signal is 0
- Doorbell kick with WPTR+1: moved hang from op 4 → op 10 (proved CP wake mechanism)
- NOP barrier injection: 100/100 alternating 512B+2MB, but drifts at ~286 ops

**SLOT_BASED_WPTR=2 dead:** CP cannot GPUVM-read poll address on gfx8 without ATC/UTCL2. Tested with kernarg pool (GPU-visible) address — CP_HQD_PQ_WPTR stays at 0. RPTR writes work (PCIe posted writes, GPU→system) but WPTR reads fail (require GPUVM VA translation).

### Current workaround stack (2026-03-15)

| Layer | Fix | What it does | Coverage |
|-------|-----|-------------|----------|
| Kernel 0008 | RPTR_BLOCK_SIZE=4 | Per-packet RPTR writes | All queues |
| ROCR 0003 | SpinMutex + mfence + atomic::Add | Thread-safe UpdateReadDispatchId | All queues |
| ROCR 0004 | Retain/Release on bounce signals | Prevents signal use-after-free | All queues |
| ROCR 0004 | NOP barrier kick (2 per stall) | Wakes idle CP on stalled queues | ~286 ops before drift |
| CLR 0001 | Fine-grain staging (kAtomics) | GPU L2 flushed for D2H | D2H path |
| CLR 0001 | clflush on staging/pinned buffers | CPU cache coherency for D2H | D2H path |
| CLR 0001 | WaitCurrent between H2D chunks | Eliminates inter-chunk barrier deps | H2D staging |
| CLR PKGBUILD | cpu_wait_for_signal for no-atomics | CPU-side waits in WaitingSignal() | CLR inter-op deps |

**Test results with full stack:**
- D2H sweep: 13/13 x3 PASS
- H2D stress (550 ops): PASS
- Mixed kernel+memcpy (200 ops): PASS
- GPU compute (500 sync + 1000 async): PASS
- llama.cpp model loading: COMPLETES
- Long stress (llama pattern, 550 ops): 286/550 before NOP drift hang
- llama.cpp inference: HANGS (needs 500+ ops without drift)

## Future Work

### Phase 7a: Fix CP Idle Stall — SLOT_BASED_WPTR=0 (DONE)

**Problem:** llama.cpp `-ngl 1` hangs during prompt eval (~100 rapid GPU dispatches).

**Root cause:** Kernel patch 0008's no-atomics else branch set `SLOT_BASED_WPTR=2`, telling CP to poll WPTR from a CPU VA. With `PQ_ATC=0` and no UTCL2 on gfx8, CP cannot translate the address. Multiple doorbells collapse into a single `DOORBELL_HIT` flag. CP wakes, tries to read poll address, fails, goes idle. Serial warmup worked because each op completed before the next doorbell.

**Fix:** Remove `SLOT_BASED_WPTR=2` from no-atomics else branch, leaving default 0 (direct doorbell). With SLOT_BASED_WPTR=0, doorbell value IS the WPTR in dwords. ROCR patch 0003's conversion `(dispatch_id * 16) & mask` provides the correct dword WPTR. Each doorbell directly updates `CP_HQD_PQ_WPTR`. No polling needed.

**Changes:**
- Kernel patch 0008: remove `m->cp_hqd_pq_control |= 2 << SLOT_BASED_WPTR__SHIFT` from else branch
- Keep RPTR_BLOCK_SIZE=4 and NO_UPDATE_RPTR=0 (unchanged)
- Bump kernel pkgrel 11→12

**Results:** 1000 rapid serial dispatches PASS, 4-stream multi-dispatch PASS, 500/500 interleaved compute+memcpy PASS (standalone). CP idle stall is fixed. However, llama.cpp model loading still hangs — see Phase 7b.

### Phase 7b: Uncached Shared Memory for Non-Coherent Platforms (CURRENT)

#### The Problem

On coherent platforms, PCIe atomics guarantee that GPU writes to system memory automatically snoop CPU caches — every CPU read sees the GPU's latest write, at zero software cost.

On Westmere (no PCIe atomics), GPU writes via PCIe bypass CPU caches entirely. The CPU sees stale cached copies. Our current mitigation (`clflush` + `mfence`) is a point-in-time invalidation — between the flush and the subsequent read, a new PCIe write can re-populate the cache line with in-flight data. Under sustained GPU dispatch, this race is wide enough to hit intermittently.

Three shared memory regions are affected:

| Region | Size | Allocation | Current caching | Flush? |
|--------|------|------------|----------------|--------|
| RPTR bounce buffer | 4 KiB | `system_allocator()(0x1000, 0x1000, 0)` | **WB** (flags=0) | clflush+mfence in `UpdateReadDispatchId` |
| Signal pool | 4 KiB/block | `allocate_()(block_size, align, AllocateNonPaged, 0)` | **WB** (NonPaged only) | **NONE** — signal polling has no flush |
| CLR staging buffer | 4 MiB | `hostAlloc(pool_size, 0, kAtomics)` → fine-grain pool | **WB** (no uncached flag) | clflush+mfence in D2H path |

Symptoms:
- D2H memcpy returns stale/zero data (iter 297-379 of 500)
- llama.cpp `hipHostFree` → `SyncAllStreams` → `awaitCompletion` spins forever (`command.cpp:247`)
- Signal polling loop reads cached signal value that never updates (no flush in hot path)

#### The Elegant Fix: UC Mapping

Map all GPU→CPU shared memory as **Uncacheable (UC)** on the CPU side. Every CPU read goes directly to DRAM, seeing whatever the GPU last wrote via PCIe. No flushes needed. No race windows. Same semantic guarantee as PCIe atomics, enforced by page table attributes instead of hardware snooping.

The `AllocateUncached` flag is **already plumbed end-to-end** but nobody sets it:

```
ROCR AllocateUncached (1 << 11)
  → KfdDriver: kmt_alloc_flags.ui32.Uncached = 1
    → KFD ioctl: KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED (1 << 25)
      → amdgpu_amdkfd_gpuvm.c: AMDGPU_GEM_CREATE_UNCACHED
        → amdgpu_ttm.c (patch 0009): ttm_uncached
          → x86 PAT: pgprot_noncached → _PAGE_CACHE_MODE_UC_MINUS
```

UC-MINUS is available on all x86 since Pentium Pro. Westmere fully supports it.

#### Performance Analysis

UC reads are ~100-200ns (DRAM latency) vs ~1-4ns (L1 hit). But:

1. **RPTR**: Changes on every GPU packet completion. Caching provides zero benefit — every read needs the latest value. With WB, we already pay clflush(~100ns) + mfence(~50ns) + load. UC load (~200ns) is comparable and eliminates the race.

2. **Signals**: Change on every GPU operation completion. Same argument — stale cached values are useless. The polling loop currently busy-waits anyway.

3. **Staging buffer**: Written by GPU, read once by CPU `memcpy`, then overwritten. Cache is polluted with data that's never re-read. UC + streaming reads (`movntdqa` via `memcpy` optimization) is ideal.

**Net impact on hot path:** Neutral to positive. Eliminates flush overhead and race window. The data is write-once-read-once with no temporal locality — UC is the correct caching policy.

#### Changes

##### 1. ROCR patch 0005 update: UC bounce buffer allocation
**File:** `hsa-rocr/src/ROCR-Runtime/.../amd_aql_queue.cpp:145`

```cpp
// Before:
rptr_gpu_buf_ = static_cast<uint64_t*>(
    agent_->system_allocator()(0x1000, 0x1000, 0));

// After:
rptr_gpu_buf_ = static_cast<uint64_t*>(
    agent_->system_allocator()(0x1000, 0x1000,
        core::MemoryRegion::AllocateUncached));
```

Then remove `FlushCpuCache` + `_mm_mfence()` from `UpdateReadDispatchId` — no longer needed with UC mapping. The `atomic::Load` reads directly from DRAM.

##### 2. ROCR new patch 0006: UC signal pool for no-atomics
**File:** `runtime.cpp` signal pool allocator setup (~line 213-230)

The signal pool is global (shared across all agents). If ANY GPU agent has `NoPlatformAtomics()`, the signal pool must be UC — signals are written by GPU and polled by CPU.

```cpp
// In Runtime::RegisterAgent, after detecting a no-atomics GPU:
// Set signal allocator to use AllocateUncached | AllocateNonPaged
```

Challenge: the signal pool is initialized once at first CPU agent registration, before GPU agents are registered. Two approaches:

**Option A (preferred):** Lazy reallocation. After first GPU agent with `NoPlatformAtomics()` registers, mark the pool as needing UC. Next allocation creates a new UC block. Existing signals (few at this point) continue with WB — acceptable since they're control signals, not completion signals.

**Option B:** Add a `system_allocator_uncached_` variant to GpuAgent that passes `AllocateUncached`. Wire it into the signal pool's allocate functor. Requires refactoring the pool to accept per-allocation flags.

**Option C (simplest):** At ROCR init, check if any GPU in the topology lacks atomics (query KFD node properties before agent creation). If so, create the signal pool with UC from the start.

##### 3. CLR patch 0001 update: UC staging buffer
**File:** `rocclr/device/rocm/rocvirtual.cpp:1899`

```cpp
// Before:
gpu().dev().hostAlloc(pool_size_, 0, mem_segment)

// After (when !dev().info().pcie_atomics_):
gpu().dev().hostAlloc(pool_size_, 0, mem_segment, /* uncached = */ true)
```

Requires plumbing an uncached flag through `hostAlloc` → `hsa_amd_memory_pool_allocate` → KFD. The fine-grain pool already goes through the same KFD path — just needs the `Uncached` bit set.

Then remove `clflush` + `mfence` loops from `hsaCopyStagedOrPinned()` D2H path — no longer needed.

##### 4. CLR patch 0001 update: UC pinned buffer path
**File:** `rocblit.cpp` lines 789-796

When GPU DMA writes directly to user's pinned host buffer (large transfers), the pinned mapping should also be UC. This goes through `hsa_amd_memory_lock` — check if it inherits the original page's caching mode or can be overridden.

If pinned pages inherit WB from userspace mapping (likely), the clflush path must remain for pinned transfers. Document as known limitation — pinned path is less common than staging path.

##### 5. Remove clflush code paths (cleanup)
After UC mappings are verified working:
- Remove `FlushCpuCache` call in `UpdateReadDispatchId`
- Remove `_mm_mfence` after flush in `UpdateReadDispatchId`
- Remove `_mm_clflush` loop in `hsaCopyStagedOrPinned` D2H path
- Remove `_mm_clflush` loop in pinned D2H path (only if pinned also UC)
- Keep `ROC_CPU_WAIT_FOR_SIGNAL` env var (orthogonal — controls polling vs interrupt)

#### Verification Plan

**Phase B1: RPTR bounce buffer UC**
```bash
# After ROCR patch, before CLR changes
# RPTR is the most critical — if bounce buffer works, signals fire
timeout 120 ./hip_torture_test   # Test 1 (1000 serial) + Test 6 (10K stress)
```

**Phase B2: Signal pool UC**
```bash
# After signal patch
# Signals now visible without flush — completion path reliable
timeout 120 ./hip_torture_test   # All 6 tests
```

**Phase B3: Staging buffer UC**
```bash
# After CLR staging patch
# D2H data now correct without flush
for i in $(seq 10); do timeout 60 /tmp/test_interleave2; done   # 10 runs, 500 each
```

**Phase B4: llama.cpp**
```bash
timeout 600 llama-cli -m ~/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -ngl 1 -p "2+2=" -n 8 --threads 4 --no-mmap
```

**Phase B5: Verify UC via /proc**
```bash
# Confirm pages are UC-MINUS
grep -c 'uncached\|write-combining' /proc/$(pgrep hip_torture)/smaps
# Or check PAT bits in page table entries via pagemap
```

#### Success Criteria
- HIP torture test: 6/6 PASS, zero data corruption across 10 consecutive runs
- llama.cpp `-ngl 1`: produces tokens, clean exit
- No `clflush` in hot path (grep patched source)
- `dmesg`: zero GPU resets

#### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| UC flag not reaching TTM | Low | High | Patch 0009 verified; trace with `pr_info` in `amdgpu_ttm_tt_new` |
| UC breaks GPU writes to same pages | None | N/A | UC only affects CPU caching; GPU writes via PCIe are unaffected |
| Signal pool UC causes regression on coherent platforms | None | N/A | Flag only set when `NoPlatformAtomics()` detected |
| Pinned path still WB | Medium | Low | Pinned path is fallback for >16KB; staging path handles most copies |
| Performance regression from UC reads | Low | Low | Data has no temporal locality; UC is correct policy |
| ROCR `system_allocator` doesn't forward flags | Medium | High | Verify `AllocateUncached` propagates through allocator lambda |

### Phase 7c: CLR Event Completion for No-Atomics Platforms (NEXT)

#### Problem

`hipHostFree` → `SyncAllStreams` → `HostQueue::finish()` → `Event::awaitCompletion()` spins forever at `command.cpp:247`. This blocks llama.cpp model loading and test cleanup.

#### Root Cause

CLR's event completion uses a two-stage mechanism:

```
GPU finishes AQL packet
  → decrements completion signal (via PCIe AtomicOp — FAILS on no-atomics)
    → ROCR async events loop detects signal change
      → calls HsaAmdSignalHandler (rocvirtual.cpp:265)
        → updateCommandsState() (rocvirtual.cpp:2088)
          → setStatus(CL_COMPLETE) (command.cpp:105)
            → awaitCompletion() unblocks
```

On no-atomics platforms, the GPU can't write signals via PCIe AtomicOps. Our ROCR bounce buffer CPU-decrements signals instead (via `SubRelaxed`). But `Event::awaitCompletion` spins on a C++ atomic `status_` field — it does NOT call `ProcessAllBounceBuffers()`. So the bounce buffer never runs during the spin, signals never fire, `HsaAmdSignalHandler` never executes, and `status_` stays at `CL_SUBMITTED` forever.

**Why the smoke test works:** Simple `hipMemcpy` is synchronous — CLR calls `HostQueue::finish()` with `cpu_wait=true`. The command's signal IS waited on via ROCR signal wait paths (which DO call `ProcessAllBounceBuffers`). The marker submitted by `finish()` has no signal of its own — it waits for all previous commands. If previous commands already completed (their signals fired via the ROCR wait path), the marker completes.

**Why llama.cpp hangs:** Model loading submits many H2D copies across multiple streams, then `hipHostFree` calls `SyncAllStreams`. Some streams have commands whose async handler hasn't fired yet (signal was CPU-decremented by bounce buffer, but the async events thread hasn't polled it). The marker in `finish()` waits for these stale commands, and the active-wait loop at `command.cpp:247` doesn't drive the bounce buffer.

#### Fix

Add `ProcessAllBounceBuffers()` to CLR's active-wait loop in `Event::awaitCompletion()`:

```cpp
// command.cpp:246-248, in the ActiveWait loop:
while (status() > CL_COMPLETE) {
    AMD::AqlQueue::ProcessAllBounceBuffers();  // ← ADD THIS
    amd::Os::yield();
}
```

This closes the completion loop: while CLR spins waiting for event status, it drives the bounce buffer, which fires signals, which triggers `HsaAmdSignalHandler`, which sets `CL_COMPLETE`.

**Why this is correct:** On coherent platforms, `ProcessAllBounceBuffers()` is a no-op (no bounce queues registered). On no-atomics platforms, it's the same call already present in `BusyWaitSignal::WaitRelaxed` and `InterruptSignal::WaitRelaxed` — we're just adding it to the CLR-level wait loop that wasn't going through ROCR signal wait.

#### Changes

**CLR patch (hip-runtime-amd PKGBUILD sed injection):**
```bash
# In command.cpp, add ProcessAllBounceBuffers to active-wait loop
sed -i '/amd::Os::yield();/i\
        AMD::AqlQueue::ProcessAllBounceBuffers();' \
    rocclr/platform/command.cpp

# Add include for AqlQueue header
sed -i '/#include "platform\/command.h"/a\
#include "core/inc/amd_aql_queue.h"' \
    rocclr/platform/command.cpp
```

Bump hip-runtime-amd-polaris pkgrel 3→4.

#### Verification

```bash
# Test 1: hipHostFree no longer hangs
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 30 ./hip_smoke   # exit 0, not 124

# Test 2: Torture test cleanup doesn't hang
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 600 ./hip_torture_test   # All 12 tests

# Test 3: llama.cpp (blocked by Phase 7d VM fault)
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 600 llama-cli \
  -m ~/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -ngl 1 -p "2+2=" -n 8 --threads 4 --no-mmap
```

**Note:** `HSA_OVERRIDE_GFX_VERSION=8.0.3` is NOT needed — our packages compile native gfx803. The override can cause NX faults and should not be used.

#### Success Criteria
- `hip_smoke` exits with code 0 (not 124/timeout)
- Torture test 12/12 PASS with clean exit
- llama.cpp produces tokens
- Zero GPU resets in dmesg

### Phase 7d: GPU VM Fault During Model Tensor Access (CURRENT)

#### Problem

llama.cpp with any `-ngl` value crashes during inference:
- `-ngl 1`: VM fault "Page not present" reading `hipHostMalloc`'d host memory (zero-copy)
- `-ngl 99`: crash in `ggml_cuda_op_rope_impl` → ROCR → `std::terminate`, CP preemption timeout

#### Phase 7d Findings

**GART size (resolved):** Default 256MB, now 2048MB via `amdgpu.gartsize=2048`. Eliminated the original >256MB mapping failure. Fault persists at ~32MB — within GART range. **Not a GART size issue.**

**Fault location:** GPU reads from **`hipHostMalloc`'d host memory** at offsets ~800KB into 1MB staging buffers. ggml-hip with `-ngl 1` keeps non-offloaded layers in host memory; GPU kernels read weights via zero-copy over PCIe. Some pages within the allocation are not mapped in the GPU's VMID page tables despite being within the 2GB GART aperture.

**UC patches NOT the cause:** `hipHostMalloc` uses `ihipMalloc`/SVM path, not `Device::hostAlloc`. Our UC injection only affects `hostAlloc`. Verified: UNCACHED flag does not affect GPU page table permissions (NX/execute bits independent of caching mode).

**`HSA_OVERRIDE_GFX_VERSION` NOT needed and harmful:**
- All our packages compile native gfx803: `libggml-hip.so` contains `amdgcn-amd-amdhsa--gfx803`, rocBLAS has 72 `.hsaco` files for gfx803
- The override is unnecessary for our packages and caused an NX fault with TinyLlama ("Execute access to a page marked NX")
- **Going forward: only `ROC_CPU_WAIT_FOR_SIGNAL=1`, drop `HSA_OVERRIDE_GFX_VERSION`**

**`-ngl 99` crash:** `ggml_cuda_op_rope_impl` → ROCR allocation path → `std::terminate`. CP queue preemption timeout. Separate from the VM fault — occurs during compute dispatch, not memory access. May be signal/kernarg exhaustion or a gfx803 ISA issue with the RoPE kernel.

#### Root Cause: H2D Blit Kernel Silent Data Corruption

**Not a VRAM mapping issue.** Isolated testing shows:
- `hipMalloc` + `hipMemset` + GPU kernel read: PASS at all sizes up to 512MB
- `hipHostMalloc` + GPU kernel read (zero-copy): PASS at all offsets up to 454MB
- `hipMalloc` + `hipMemcpy H2D` + GPU kernel verify: **1MB PASS, 16MB FAIL** (4M bad values)

The H2D `hipMemcpy` via CLR's staging/blit path reports `hipSuccess` but **silently fails to write data to VRAM** for transfers >1MB. The GPU then reads uninitialized VRAM (zeros or stale data), which llama.cpp interprets as model weights → garbage → eventual VM fault when a computed pointer lands outside mapped range.

**Mechanism:** Large H2D copies are chunked through a staging buffer (fine-grain host memory, now UC). Each chunk: CPU fills staging → blit kernel copies staging→VRAM → wait for completion → next chunk. If inter-chunk completion detection fails, the next chunk overwrites the staging buffer before the blit kernel finishes reading it, corrupting the transfer.

**Evidence:** 1MB transfers work (single chunk, fits in staging buffer). 16MB transfers fail (multiple chunks through 4MB staging buffer, ~4 chunks needed). The 4M bad values ≈ 16MB/4 = 4M ints, meaning essentially ALL data is wrong — consistent with staging buffer overwrite before blit completion.

**Connection to earlier phases:**
- Phase 7b (UC staging): Changed staging from WB→UC. This is correct for CPU→GPU visibility (staging data reaches DRAM). But UC may affect the blit kernel's read pattern — UC memory is not cached in GPU L2, so blit kernel reads go to DRAM every time.
- Phase 7c (synchronous flush): Forces `flush()` in `submitMarker`, but `hipMemcpy` internally uses `WaitCurrent()` between chunks via `releaseGpuMemoryFence()`. If `WaitCurrent()` returns before the blit kernel truly finishes (bounce buffer RPTR race), the next chunk overwrites staging.

#### Fix Avenues

**Avenue A: Verify WaitCurrent inter-chunk completion (most likely fix)**

The D2H path in `hsaCopyStagedOrPinned()` (rocblit.cpp) calls `gpu().Barriers().WaitCurrent()` between chunks. If `WaitCurrent()` relies on CLR's signal/event mechanism (which uses the async handler that doesn't fire on no-atomics), it may return prematurely.

Check: does `Barriers().WaitCurrent()` go through ROCR signal wait (which has bounce buffer) or CLR event wait (which we fixed only for markers in Phase 7c)?

If CLR's `WaitCurrent()` doesn't drive the bounce buffer, we need to extend the Phase 7c fix to cover all `releaseGpuMemoryFence()` paths, not just `submitMarker`.

**Avenue B: Bisect UC staging**

Test if the H2D corruption exists WITHOUT the UC staging buffer patch (Phase 7b). If H2D worked before UC and fails after, the issue is that the blit kernel reads UC memory differently. GPU L2 caching behavior for fine-grain UC system memory on gfx8 may differ from fine-grain WB.

Quick test: temporarily remove the UC `hostAlloc` injection from CLR PKGBUILD, rebuild, rerun `test_h2d_kernel`.

**Avenue C: Force single-chunk H2D**

Increase the staging buffer size or the per-chunk transfer size so that 16MB fits in a single chunk. This avoids the inter-chunk completion problem entirely. CLR's `StagingXferSize` defaults to ~2MB — increasing it to 64MB would make most model weight transfers single-chunk.

Downside: wastes memory. But for a 2GB card with 454MB model, this is acceptable.

**Avenue D: CPU memcpy fallback for H2D on no-atomics**

Instead of using the blit kernel (GPU-side copy via staging), do a direct CPU `memcpy` to VRAM through the BAR for H2D transfers. This bypasses the staging/blit/completion path entirely.

Problem: BAR is only 256MB (visible VRAM). Allocations in invisible VRAM (>256MB) can't be CPU-written through the BAR. Only works for small models that fit in visible VRAM.

**Avenue E: Per-chunk hipDeviceSynchronize in CLR blit**

Add an explicit `hipDeviceSynchronize` (or equivalent full GPU drain) between staging chunks in the H2D path. This is the nuclear option — guaranteed correct but serializes every chunk with a full GPU sync.

#### Avenue A Investigation Results

`WaitCurrent()` → `CpuWaitForSignal()` → `WaitForSignal()` → `hsa_signal_wait_scacquire()` → ROCR bounce buffer. **The completion chain IS correct.** The CLR profiling signal is placed as the AQL packet's `completion_signal`. The bounce buffer scans it, fires it, and `WaitCurrent()` returns.

Yet 16MB H2D still fails. Bounce buffer debug shows RPTR advancing across multiple queues (blit uses a separate ROCR queue from compute). The VM fault is at a VRAM address that should have been written by the blit kernel. This suggests the blit kernel ran but **wrote to the wrong destination** — possibly because kernarg was corrupted, or the blit queue's AQL packet had stale data.

**Avenue B removed** — UC is the correct policy, not a regression candidate.

#### Systematic Audit Needed

Tactical patches won't work. We need a complete audit of every function in the no-atomics data path. The remaining corruption could be in any of:

1. **Kernarg allocation for blit kernel** — is the kernarg pool UC? If not, the blit kernel reads stale CPU-cached kernarg (src/dst pointers)
2. **Blit queue signal pool** — blit uses its own ROCR AqlQueue; does its bounce buffer process correctly?
3. **Multi-queue interaction** — CLR creates separate queues for compute vs SDMA-emulated copy; signal tracking across queues may have gaps
4. **FENCE_SCOPE_SYSTEM in blit packet header** — line 889-890 of `amd_blit_kernel.cpp` sets `SCACQUIRE_FENCE_SCOPE=SYSTEM` and `SCRELEASE_FENCE_SCOPE=SYSTEM`. On gfx8 without coherency, system scope fences may not flush GPU L2 to DRAM. The blit kernel may write to GPU L2 but the data never reaches visible memory.
5. **Staging buffer address in blit kernarg** — if kernarg uses WB-cached host memory, the blit kernel may read a stale staging buffer pointer
6. **AQL ring buffer for blit queue** — if the ring buffer itself is WB-cached and the CPU writes aren't visible to the GPU, the AQL packet content may be stale

**Each of these requires UC or explicit flush treatment on no-atomics platforms.** The fix must be comprehensive — covering all shared memory between CPU and GPU in the data transfer path, not just the three regions we identified in Phase 7b.

#### Complete Shared Memory Audit

All allocations via `system_allocator()` return kernarg pool memory (fine-grain host, CPU VA = GPU VA). On coherent platforms this is safe. On no-atomics Westmere, CPU writes are WB-cached and GPU reads via PCIe see DRAM — not the CPU cache. Every region where CPU writes and GPU reads must be UC.

**Region 1: Blit kernel kernarg** (`amd_blit_kernel.cpp:566-568`)
```cpp
kernarg_async_ = system_allocator()(size, 16, AllocateNoFlags);
```
- **Flags:** `AllocateNoFlags` (= 0) → **WB cached**
- **Contains:** src pointer, dst pointer, copy size, work dimensions
- **CPU writes:** `args->copy_aligned.*` (lines 758-791), no clflush after
- **GPU reads:** blit kernel reads these as kernel arguments
- **Status: BROKEN.** Prime suspect for H2D corruption. GPU reads stale/zero kernarg → copies wrong src to wrong dst.
- **Fix:** Change to `AllocateUncached`

**Region 2: AQL ring buffer** (`amd_aql_queue.cpp:715-717`)
```cpp
ring_buf_ = system_allocator()(ring_buf_alloc_bytes_, 0x1000, AllocateExecutable);
```
- **Flags:** `AllocateExecutable` → **WB cached**
- **Contains:** AQL dispatch/barrier packets (64 bytes each)
- **CPU writes:** `PopulateQueue` (line 910-912) with acquire/release fences
- **GPU reads:** CP fetches packets from ring buffer via PCIe
- **Status: SUSPECT.** CPU writes AQL packet body, then atomically stores header. x86 `std::atomic_thread_fence(release)` ensures ordering within the CPU cache, but doesn't flush to DRAM. CP may read stale packet from ring buffer.
- **Fix:** Change to `AllocateExecutable | AllocateUncached`

**Region 3: Staging buffer** (CLR `hostAlloc` with `kAtomics` segment)
- **Status: FIXED** (Phase 7b). UC via `HSA_AMD_MEMORY_POOL_UNCACHED_FLAG`.

**Region 4: RPTR report buffer** (`amd_aql_queue.cpp:144-145`)
- **Status: FIXED** (Phase 7b). UC via `AllocateUncached`.

**Region 5: Signal pool** (`signal.cpp:82`)
- **Status: FIXED** (Phase 7b). UC via `AllocateNonPaged | AllocateUncached`.

**Region 6: Blit kernel code object** (`amd_gpu_agent.cpp:381-382`)
```cpp
code_buf = system_allocator()(code_buf_size, 0x1000,
    AllocateExecutable | AllocateExecutableBlitKernelObject);
```
- **Flags:** `AllocateExecutable` → **WB cached**
- **Contains:** compiled shader binary (read-only after load)
- **CPU writes:** once at initialization
- **GPU reads:** every blit dispatch (instruction fetch)
- **Status: PROBABLY OK.** Written once at init, read many times. By the time the first blit dispatch runs, the write has long since flushed from CPU cache to DRAM (time-based eviction). But technically unsafe on a strictly non-coherent platform. Low risk in practice.
- **Fix (belt and suspenders):** Add `AllocateUncached`

**Region 7: PM4 indirect buffer** (`amd_aql_queue.cpp:349-350`)
```cpp
pm4_ib_buf_ = system_allocator()(pm4_ib_size_b_, 0x1000, AllocateExecutable);
```
- **Flags:** `AllocateExecutable` → **WB cached**
- **Contains:** PM4 commands for icache invalidation
- **Status: PROBABLY OK.** Written once at queue creation, rarely changes. Same time-based reasoning as code object.
- **Fix (belt and suspenders):** Add `AllocateUncached`

**Region 8: Doorbell queue map** (`amd_gpu_agent.cpp:2310`)
```cpp
doorbell_queue_map_ = system_allocator()(size, 0x1000, 0);
```
- **Status: CPU-only.** Maps doorbell index to queue pointer. Not read by GPU.
- **Fix: None needed.**

#### FENCE_SCOPE_SYSTEM on gfx8 (Region 9 — Hardware Behavior)

The blit kernel AQL packet sets `SCRELEASE_FENCE_SCOPE=SYSTEM` (line 889-890 of `amd_blit_kernel.cpp`). Investigation of what this means on gfx8:

**MTYPE for system memory on gfx8:** The MQD sets `MTYPE_UC` (`kfd_mqd_manager_vi.c:128`), meaning GPU L2 does **not** cache system memory accesses. Reads/writes to system memory bypass L2 and go directly to the memory controller → PCIe. This is correct behavior — no GPU L2 coherency issue.

**MTYPE for VRAM on gfx8:** VRAM accesses go through GPU L2 with MTYPE dependent on allocation flags. For standard VRAM allocations, MTYPE defaults to a cached mode. The blit kernel writes to VRAM through GPU L2.

**SCRELEASE_FENCE_SCOPE=SYSTEM effect:** On gfx8, this is an **ordering fence**, not a cache flush. The CP ensures all prior memory operations from this wave are visible before marking the packet complete. With MTYPE=UC for system memory, there's nothing to flush (L2 never cached it). For VRAM writes through L2, the fence ensures the L2 write has completed before the completion signal.

**Bottom line for FENCE_SCOPE_SYSTEM:** Not a problem. The blit kernel's VRAM writes go through GPU L2, and the system-scope release fence ensures they're committed before completion. The blit kernel's reads from system memory (staging buffer, kernarg) bypass L2 (MTYPE=UC for system memory). **The issue is on the CPU side (WB cached writes not reaching DRAM), not the GPU side.**

**One caveat:** The above analysis assumes MTYPE=UC for system memory accessed by the compute queue. If the fine-grain pool's MTYPE is NC (non-coherent) rather than UC, GPU L2 **might** cache stale reads from system memory. This would need verification via `amdgpu_vm_get_pte_flags` in the kernel driver. However, even with MTYPE=NC, the blit kernel's `s_waitcnt vmcnt(0)` and the AQL release fence should drain any L2-cached reads.

#### Comprehensive Fix Plan

**Phase 7e: UC all CPU→GPU shared memory in ROCR**

Three ROCR sed injections needed (all in `amd_aql_queue.cpp` and `amd_blit_kernel.cpp`):

1. **Kernarg pool for blit kernel** — `AllocateNoFlags` → `AllocateUncached`
2. **AQL ring buffer** — `AllocateExecutable` → `AllocateExecutable | AllocateUncached`
3. **Code object** (belt and suspenders) — add `AllocateUncached`
4. **PM4 IB buffer** (belt and suspenders) — add `AllocateUncached`

The CLR `kKernArg` exclusion from UC in Phase 7b should be **removed** — user kernel kernarg has the same CPU→GPU visibility problem. The exclusion was wrong.

**Expected result:** With all CPU→GPU shared memory UC, the CPU's writes go directly to DRAM. The GPU reads from DRAM via PCIe (MTYPE=UC, bypassing L2). No stale data possible. The H2D blit kernel reads correct kernarg (src/dst/size) and AQL ring buffer (packet headers/dispatches), and the staging buffer data is also correct (already UC).

#### Phase 7e Results (post cold boot)

Test results with Phase 7e UC fixes on a cold-booted GPU:

| Test | 1MB | 4MB | 16MB | 32MB | 48MB | 64MB | 65MB |
|------|-----|-----|------|------|------|------|------|
| hipMemset + GPU verify | PASS | PASS | PASS | PASS | - | PASS | - |
| H2D + D2H (CPU verify) | PASS | PASS | PASS | PASS | - | PASS | - |
| H2D + GPU kernel verify | PASS | PASS | PASS | PASS | PASS | PASS | **FAIL** |

**16MB H2D + GPU verify: FIXED.** This was the original Phase 7d blocker.

**65MB: New boundary.** At 65MB, the GPU kernel verify shows `gpu_bad=0` (data correct in VRAM) but `cpu_bad=78K-278K` (D2H readback corrupt, nondeterministic count). On a faulted GPU, even Test A (H2D+D2H with no kernel) fails at 65MB. On a clean GPU, `test_h2d_boundary` passes at 512MB with CPU-only D2H verify.

**Remaining hypothesis — D2H MTYPE/L2 issue:**

The D2H blit kernel reads from VRAM (L2-cached, MTYPE=CC) and writes to the staging buffer (system memory). The staging buffer was made UC on the CPU side (via `HSA_AMD_MEMORY_POOL_UNCACHED_FLAG`). But what MTYPE does the **GPU** see for the staging buffer PTE?

- If the GPU PTE for staging has MTYPE=UC: GPU writes bypass L2, go directly to DRAM via PCIe. CPU reads DRAM. Correct.
- If the GPU PTE for staging has MTYPE=NC: GPU writes go to L2 but are not coherent with DRAM. CPU reads DRAM and sees stale data. **This is the bug.**
- If the GPU PTE for staging has MTYPE=CC: GPU writes go to L2, and L2 snoops/flushes on system scope release fence. Correct on coherent platforms. On no-atomics platforms, the release fence may be a no-op for CC pages — data stays in L2.

The `HSA_AMD_MEMORY_POOL_UNCACHED_FLAG` sets the CPU mapping to UC via `set_memory_uc()` or equivalent. But the GPU PTE MTYPE is set separately by KFD during `mmap`/`map_memory_to_gpu`. We need to verify that the KFD `kfd_ioctl_map_memory_to_gpu` path respects the uncached flag and sets MTYPE=UC in the GPU page table entry.

#### Phase 7f: GPU PTE SNOOPED fix (kernel patch)

**Root cause identified.** The D2H corruption and the Phase 7e 65MB boundary failure share the same root cause: our kernel patch 0009 (`AMDGPU_GEM_CREATE_UNCACHED` → `ttm_uncached`) inadvertently set SNOOPED=0 in the GPU PTE, disabling the hardware coherency protocol that GFX8 depends on.

**Evidence chain:**

1. **VI/gfx8 PTEs have NO MTYPE field** (only gfx9+ has MTYPE at bits 57:58). The ONLY per-page coherency control on VI is the SNOOPED bit (PTE bit 2).

2. **GFX8 ISA has NO L2 flush mechanism.** LLVM's `SIGfx7CacheControl::insertRelease()` (`SIMemoryLegalizer.cpp:1291-1298`) implements system-scope release as just `insertWait()` (= `s_waitcnt`). `SIGfx7CacheControl::insertAcquire()` (line 1300-1348) inserts `BUFFER_WBINVL1_VOL` — L1 invalidate only, not L2.

3. **GFX8 assumes hardware coherency for system memory.** Since the ISA provides no L2 flush, the hardware must handle L2→DRAM coherency. The mechanism is the SNOOPED bit: SNOOPED=1 makes GPU L2 participate in the PCIe coherency protocol. Writes through L2 are visible to the CPU via the snooping hardware in the root complex.

4. **Kernel code sets SNOOPED based on TTM caching mode** (`amdgpu_ttm.c:1377-1378`):
   ```c
   if (ttm && ttm->caching == ttm_cached)
       flags |= AMDGPU_PTE_SNOOPED;
   ```
   Our patch 0009 changed caching from `ttm_cached` to `ttm_uncached` → SNOOPED=0 → GPU L2 non-coherent → D2H writes to staging stay in GPU L2 → CPU reads stale DRAM.

5. **Precedent: ARM non-coherent platform work.** The [drm/ttm RFC](https://lore.kernel.org/linux-kernel/20240629052247.2653363-1-uwu@icenowy.me/T/) for devices without coherency and the [Raspberry Pi GPU testing](https://github.com/geerlingguy/raspberry-pi-pcie-devices/discussions/756) confirm that GPU drivers assume PCIe snooping works. Non-coherent platforms need special handling.

**Fix:** Set SNOOPED=1 for `ttm_uncached` system memory in `amdgpu_ttm_tt_pde_flags()`. This restores GPU L2 coherency while keeping CPU-side UC (no CPU cache lines to flush). The GPU snoops CPU cache (finds nothing since UC), writes propagate through the coherency protocol to DRAM.

**Phase 7f (ACQUIRE_MEM hack): REMOVED.** The PM4 TC_WB_ACTION_ENA is a hardware operation outside the GFX8 ISA memory model. It caused VM faults when ExecutePM4 was called on the blit queue. The kernel SNOOPED fix is the correct solution.

#### Phase 7g: hipMemcpy silently fails after sustained use

**Status: INVESTIGATING**

**Root cause: hipMemcpy H2D silently fails after ~7000+ GPU operations in a
single process.  VRAM retains stale fill data; GPU kernels execute on wrong
input; no HIP error returned.**

#### Evidence chain

1. `test_mixed_stress` (isolated alloc→H2D→GPU verify→free): **500/500 PASS**
   on clean boot with 292 VA reuses.  Simple single-buffer pattern works
   perfectly.

2. `hip_barrier_test` full suite (15 tests in one process):
   - Tests 2.1-4.1 all PASS (~7000 kernel dispatches, ~700 alloc/free cycles,
     multiple stream/event create/destroy)
   - Test 4.2 FAILS at iter 0: `expected 10, got 0x9A9A9A9A`
   - Tests 5.1 cascades (same garbage value)

3. Test 4.2 in isolation (separate process): **PASSES**.  Same test, fresh
   HIP/CLR state.  Runs correctly every time.

4. Test 4.2 after running tests 2.1-4.1 in the same process: **FAILS**.
   The prior tests' cumulative HIP/CLR state changes cause the failure.

#### Failure analysis

The value `0x9A9A9A9A` in test 4.2 is diagnostic:
- Test does: `hipMemcpy(d_a, h_init, ...)` where h_init = {0, 0, ...}
- Then runs 10 `add_one` kernels: a→b→a→b... (each adds 1)
- Expected: d_a[0] = 10
- Got: d_a[0] = 0x9A9A9A9A

Working backwards: `0x9A9A9A9A = 0x9A9A9A90 + 10`.  The kernels DID execute
(added 10).  But d_a started at `0x9A9A9A90` instead of 0.  **The hipMemcpy
H2D wrote zeros to d_a, but the GPU read stale VRAM content** — the H2D copy
silently failed or wrote to the wrong address.

`0x9A` is a VRAM fill pattern (uninitialized allocation content).  The GPU
is reading the allocation's original VRAM content, not the H2D-copied data.

#### What degrades after sustained use

The first ~7000 operations work perfectly.  After that, hipMemcpy H2D begins
silently failing.  Possible degradation mechanisms:

1. **Staging buffer pool exhaustion/corruption** — CLR uses a fixed staging
   buffer pool for H2D copies.  After thousands of copies, pool management
   state (free list, active chunk tracking) may become inconsistent.

2. **Bounce buffer signal leak** — ROCR's bounce buffer tracks completion
   via signals.  If signals are not properly released after ~7000 dispatches,
   the signal pool exhausts and new copies can't get completion signals.

3. **AQL queue ring buffer wrap race** — after ~7000 dispatches, the ring
   buffer has wrapped many times.  RPTR/WPTR tracking via the bounce buffer
   may accumulate small errors that eventually cause the queue to appear full
   or cause packets to be dropped.

4. **kernarg pool exhaustion** — each kernel dispatch allocates from the
   kernarg pool.  If the pool fills and recycling barriers (which our Phase 8
   fix modifies) don't work correctly, subsequent dispatches get wrong
   kernarg addresses.

5. **CLR command batch overflow** — CLR's VirtualGPU batches commands into
   a command list.  After thousands of operations, batch bookkeeping may
   overflow or wrap, causing commands to target wrong buffers.

#### Implications for llama.cpp

This directly explains the llama.cpp inference hang:
- Model loading does hundreds of hipMemcpy H2D (one per tensor)
- Each load also involves hipMalloc, hipDeviceSynchronize, barriers
- After enough operations, subsequent H2D copies silently fail
- GPU inference kernels read stale/uninitialized VRAM → wrong results
- If results are NaN/Inf, downstream kernels may produce garbage or hang

#### Test structure finding

Test degradation between runs of `hip_barrier_test` is NOT caused by:
- Fork test (5.2) — KFD blocks forked children at ioctl level
  (`process->lead_thread != current->group_leader` → `-EBADF`)
- GPU hardware corruption — clean kexec boot reproduces same pattern
- D2H coherency (Phase 7a-f) — the H2D path is failing, not D2H

The ROCR SIGSEGV at offset 0x62c23 in forked children is benign:
the signal validation reads from a VM_DONTCOPY page (unmapped in child).
KFD set VM_DONTCOPY on doorbell, event, and reserved memory VMAs.

#### Investigation plan

1. **Binary search for degradation point** — run N operations (varying N),
   then test H2D correctness.  Find the exact threshold where H2D fails.

2. **Monitor ROCR signal pool** — add logging to signal alloc/free to detect
   leaks.  Check if `ProcessAllBounceBuffers()` stops processing after
   sustained use.

3. **Monitor staging buffer state** — log CLR staging pool active chunk,
   free list, allocation count.  Check if pool recycling breaks.

4. **Monitor AQL queue state** — log RPTR, WPTR, dispatch_id, ring buffer
   wrap count.  Check for RPTR tracking drift.

5. **Bisect: test with hipDeviceSynchronize after every operation** — if
   forced serialization fixes the issue, it's a signal/completion race.
   If it doesn't, it's a resource exhaustion.

#### Phase 7g fix: CPU-verified resource fencing

**Tested and eliminated:**
- Option 3 (keep dep_signals): WORSE (80% failure rate — CP stalls on signals
  that are 0 but reads race with bounce buffer decrement)
- Option 5 fence in dispatchBarrierPacket: did NOT help (~same failure rate)
- WB memory instead of UC for kernarg: MUCH WORSE (100% failure rate — CPU
  WB-cached writes not reliably visible to GPU on Westmere PCIe 2.0)
- UC confirmed correct: DRAM writes are visible to GPU, but something else
  causes intermittent corruption

**KFD queue teardown audit (critical gaps):**

Audited the complete queue destroy path: `kfd_ioctl_destroy_queue` →
`pqm_destroy_queue` → `destroy_queue_nocpsch` → `kfd_destroy_mqd_cp` →
`kgd_hqd_destroy` (amdgpu_amdkfd_gfx_v8.c:391-493).

| Gap | What's missing | Impact |
|-----|---------------|--------|
| CP_HQD_PQ_WPTR not cleared | After HQD deactivate, WPTR register retains last value | Next queue on same HQD slot may read stale ring entries |
| No L2 cache flush | gfx8 has no L2 flush (Hawaii gets TC flush, gfx8 skipped) | GPU L2 dirty lines persist across queue destroy |
| MQD not zeroed | GTT page freed but not cleared; reuse gets old MQD values | Next alloc at same page reads stale MQD fields |
| Doorbell not cleared in HW | Software clears bitmap but BAR page retains old value | Stale doorbell writes could trigger old queue action |
| No memory barrier | No fence between dequeue and MQD free | Race between HW teardown and memory reclaim |

`kgd_hqd_destroy()` does:
1. Wait for IQ timer to settle ✓
2. Send DRAIN_PIPE or RESET_WAVES dequeue request ✓
3. Wait for CP_HQD_ACTIVE to clear ✓
4. Return — **no register cleanup, no cache flush, no barrier**

**Cross-process scenario:**
1. Process A destroys queue → CP_HQD_ACTIVE cleared
2. CP_HQD_PQ_WPTR, L2 cache, MQD contents all retain Process A's state
3. Process B creates queue on same HQD slot
4. `kgd_hqd_load()` reinitializes registers from MQD → should be clean
5. BUT: GPU L2 may cache stale data at reused VAs (kernarg, staging)

**Within-process scenario (the primary issue):**
The cross-process gaps may contribute to inter-run degradation, but the
corruption also occurs WITHIN a single process after ~200+ rapid
alloc/free+dispatch cycles.  This suggests an additional mechanism beyond
queue teardown — possibly in the CLR ManagedBuffer pool rotation or the
ROCR bounce buffer's signal management under sustained load.

**Key finding (2026-03-21):** Corruption happens on FIRST run from clean
kexec boot.  7/9 on run 1.  NOT a cross-process accumulation issue.  The
problem is inherent to Phase 8 barrier dep-clearing.

**Root cause confirmed:** Phase 8 clears dep_signals in ALL barriers
including resource recycling barriers.  When deps are cleared:

1. `ManagedBuffer::Acquire()` chunk rotation (line 1947 in rocvirtual.cpp):
   - Dispatches barrier with `completion_signal = pool_signal_[chunk]`
   - Our Phase 8 clears its dep_signals → CP completes barrier instantly
   - `pool_signal_[chunk]` goes to 0 via bounce buffer at RPTR advance
   - RPTR advances when CP **consumes** (launches) the preceding blit kernel
   - Preceding blit kernel wavefronts may still be **reading** the chunk
   - Next iteration overwrites the chunk → blit reads corrupted data

2. `releaseGpuMemoryFence()` (line 1695 in rocvirtual.cpp):
   - Dispatches barrier → `WaitCurrent()` → `ResetQueueStates()`
   - Same issue: barrier completes before preceding kernels finish
   - `resetKernArgPool()` reuses kernarg memory → kernel reads stale args

**The fix:** In these TWO specific recycling paths (not all barriers), add a
CPU-side drain that waits for ALL preceding wavefronts to retire before
recycling.  The drain uses `Barriers().WaitCurrent()` which CPU-waits
the barrier's completion signal.  The key insight: the barrier's completion
signal goes to 0 when the bounce buffer sees RPTR advance PAST the barrier
packet.  For a barrier with the BARRIER header bit, the CP can't advance
RPTR past it until all preceding wavefronts retire.

So the flow becomes:
1. Phase 8 clears dep_signals → barrier has no deps
2. Barrier is submitted to AQL queue with BARRIER bit
3. CP processes preceding dispatches (launches wavefronts)
4. CP reaches barrier → BARRIER bit forces wait for wavefront retirement
5. Wavefronts retire → CP advances RPTR past barrier
6. Bounce buffer sees RPTR advance → decrements completion signal
7. CPU `WaitCurrent()` returns → safe to recycle

The issue was that `WaitCurrent()` in `releaseGpuMemoryFence()` already
does this!  And `WaitForSignal(pool_signal_[next_chunk])` in Acquire()
should too.  So why doesn't it work?

**Hypothesis:** The `pool_signal_[active_chunk_]` in Acquire() is set to
1 and used as the barrier's completion_signal.  But on no-atomics, the
GPU can't decrement this signal — the bounce buffer does it.  The bounce
buffer only processes signals for packets in the AQL queue's ring buffer.
The `pool_signal_` is a SEPARATE signal created outside the AQL queue.
The bounce buffer may NOT process it.

If the bounce buffer doesn't decrement `pool_signal_[chunk]`, then
`WaitForSignal()` at line 1951 spins forever... unless there's a timeout
or the signal is processed by a different mechanism.

**Actually:** Looking at the bounce buffer code (ROCR ProcessCompletions),
it processes the completion_signal of each AQL packet it encounters in
the ring buffer.  The barrier packet has `completion_signal = pool_signal_[chunk]`.
When the bounce buffer processes the barrier packet, it decrements
pool_signal_[chunk].  So the bounce buffer DOES handle it.

**Revised hypothesis:** The bounce buffer processes packets in order.
If the barrier's RPTR hasn't advanced yet (CP waiting for wavefront
retirement), the bounce buffer hasn't reached the barrier packet.
Meanwhile, the NEXT Acquire() call finds the NEXT chunk's signal already
at 0 (from 16 rotations ago, or initial value).  So it proceeds to
recycle the next chunk — but the CURRENT chunk might still have in-flight
blit kernels reading from it.

Wait — Acquire() waits on `pool_signal_[NEXT_chunk]`, not the current
one.  The current chunk gets its signal set and barrier submitted.  The
next chunk was used 16 rotations ago.  If that barrier completed, its
signal is 0.

The issue is the CURRENT chunk's preceding blit kernel.  The blit kernel
reads from the CURRENT chunk.  Acquire() submits a barrier for the current
chunk, then moves to the NEXT chunk and returns an address in the NEXT
chunk.  The caller writes new data to the NEXT chunk.  The old blit kernel
reads from the CURRENT chunk.  These are DIFFERENT chunks — no overlap!

So where is the corruption?  It must be in the kernarg pool, not the
staging buffer.  The kernarg pool uses `releaseGpuMemoryFence()` which
resets the ENTIRE pool (`pool_cur_offset_ = 0`).  If the reset happens
while a kernel is still reading its kernarg, the next kernel's args
overwrite the in-use memory.

**Fix: add explicit CPU drain before kernarg pool reset.**

In `releaseGpuMemoryFence()`, `WaitCurrent()` should already provide
this.  But `WaitCurrent()` waits on the barrier's completion signal.
If the barrier was submitted with cleared deps (Phase 8), and the CP
completes it instantly (no deps, BARRIER bit doesn't help because the
CP considers itself "idle" after processing all preceding packets with
SLOT_BASED_WPTR=0)...

**NEW INSIGHT:** With SLOT_BASED_WPTR=0, when the CP finishes processing
all packets in the ring buffer, it goes idle.  When a new barrier arrives
(via doorbell), the CP wakes, reads the barrier.  The barrier has BARRIER
bit set.  Does the idle CP remember that preceding wavefronts haven't
retired?  Or does it consider the queue "drained" (all packets consumed)?

If the CP goes idle and "forgets" about in-flight wavefronts, then the
BARRIER bit on the newly-arrived barrier is useless — the CP thinks all
preceding work is done.

**This is the root cause.** The CP's idle state with SLOT_BASED_WPTR=0
loses track of in-flight wavefronts.  The BARRIER bit only checks the
CP's internal state, which resets on idle.

**Probability of fix: 70%** if we add an explicit `hipDeviceSynchronize`
(or `Barriers().WaitCurrent()` with a preceding NOP dispatch) before
every kernarg pool reset in `releaseGpuMemoryFence()`.  This forces the
CP to process a NOP dispatch (which resets its "all packets consumed"
state), then hit the barrier (which now correctly waits for the NOP and
all preceding wavefronts).

The 30% uncertainty is from: (a) this theory could be wrong — the CP
might actually track wavefronts correctly even when idle, and the real
issue is elsewhere; (b) the NOP dispatch workaround might have its own
timing issues.

**Breakthrough diagnostic (2026-03-21):**

Instrumented kernel records what arguments it ACTUALLY received:

```
FAIL iter 256:
  GPU val arg     = 0     ← kernel received val = 0
  GPU out ptr lo  = 0x0   ← kernel received out = NULL
  d ptr           = 0x4100600000   ← CPU passed this pointer
```

**ALL kernel arguments are ZERO at iter 256.**  Not stale, not partially
correct — completely zeroed.  The kernarg_address in the AQL dispatch
packet points to memory containing all zeros.

Earlier test showed sequential aliasing: iter 1035 reads values from iter 11
(offset = 1024 exactly).  With deeper instrumentation, iter 256 shows
all-zero args.

**Eliminated hypotheses (total 10):**

| # | Hypothesis | How tested | Result |
|---|-----------|-----------|--------|
| 1 | GPU L2 stale from UC system memory | VRAM kernarg via BAR | Still fails |
| 2 | GPU L2 stale from WB system memory | WB hostAlloc | Much worse (100%) |
| 3 | Premature recycling (single barrier) | Phase 8 dep-clear + WaitCurrent | Fails |
| 4 | Premature recycling (double barrier) | Two barriers + WaitCurrent | Fails |
| 5 | read_index fence | Spin on read_idx >= write_idx | Fails |
| 6 | Keep dep_signals in barrier | Don't clear after CPU-wait | Much worse (80%) |
| 7 | NT store torn writes | Regular memcpy instead | Fails |
| 8 | Cross-process HQD state | Clear HQD regs after destroy | VM faults; reverted |
| 9 | Cross-process accumulation | Test on clean kexec | Fails on first run |
| 10 | Ring buffer wrap stale AQL | Count packets vs ring size | No wrap at failure point |

**What we KNOW from the diagnostic:**
- Kernel receives `val=0`, `out=NULL` — ALL args zeroed
- The kernarg CPU-side write is correct (CPU passes `val=542`, `out=0x4100600000`)
- Failure at iter 256 (or 1035, 542 — varies) — no single clean boundary
- `allocKernArg` returns an address, CPU writes there, but GPU reads zeros

**Current investigation path:**

The kernarg pool offset and the AQL packet's `kernarg_address` may diverge
when `resetKernArgPool()` is called between `allocKernArg()` and the AQL
packet submission.  Sequence:

```
1. allocKernArg() returns address A (offset X in pool)
2. nontemporalMemcpy writes args to address A
3. releaseGpuMemoryFence() triggered by hipMemcpy or barrier
   → resetKernArgPool() → offset = 0
4. dispatchAqlPacket() constructs AQL packet with kernarg_address = A
5. BUT: another allocKernArg (from blit kernel) now returns offset 0
   in the SAME pool, potentially overwriting address A's data with zeros
```

**OR:** `allocKernArg` returns an address in a VRAM chunk that was just
rotated.  The chunk contains zeros (never initialized).  The CPU writes
correct data.  But the AQL packet's `kernarg_address` was set BEFORE
the write completed (PCIe BAR write latency vs AQL packet submission).

**Instrumentation results (completed):** Addresses are always correct.
Pool lifecycle is not the bug. CPU writes correct data to correct address.
GPU reads zeros from that address. This is a GPU L2 stale cache read.

**Root cause confirmed (2026-03-21):** GPU L2 retains stale cache lines
from previous dispatches at the same kernarg pool address. CPU writes to
system memory (UC) update DRAM, but do NOT invalidate GPU L2 on Westmere
PCIe 2.0 (no PCIe coherency extensions). The kernarg pool resets to
offset 0 on every `releaseGpuMemoryFence`, so every dispatch reuses the
same address → same L2 set → stale read.

**Evidence:**
- Offset between expected and got is ALWAYS a multiple of 256 iterations
  (= 128KB of L2 working set, exactly 1/4 of the 512KB L2, matching
  4-way set-associative aliasing stride)
- Increasing ring buffer to 8MB (max HW) doesn't help (ring doesn't wrap)
- Retry without hipMalloc/hipMemcpy trigger sometimes works (L2 eviction
  is probabilistic)
- hipMemcpy to the OUTPUT buffer (GPU-side blit write) before kernel
  dispatch nearly eliminates failures (GPU write updates L2)
- gfx803 has NO fine-grain GPU memory pool (only coarse-grain VRAM);
  fine-grain system pools on CPU agents are identical to coarse-grain
  on gfx8 (both SNOOPED=1 in PTE, no MTYPE field)

### Phase 7h: HDP flush for gfx8 no-atomics

#### Problem

CPU writes to system memory (kernarg pool, AQL ring buffer) are invisible
to the GPU L2 cache on Westmere PCIe 2.0. The GPU reads stale L2 entries
from previous dispatches at the same address. This causes intermittent
kernel argument corruption (~30% failure rate under sustained dispatch).

All software-level mitigations failed (10 hypotheses eliminated) because
the root cause is a HARDWARE coherency gap: PCIe 2.0 has no snoop
protocol for CPU→GPU L2 invalidation. The fix must use GPU hardware
mechanisms to invalidate L2.

#### Solution: HDP (Host Data Path) flush register

AMD GPUs have an HDP flush register (`HDP_MEM_COHERENCY_FLUSH_CNTL`)
that flushes the Host Data Path buffers, ensuring CPU writes to system
memory are visible to the GPU. Writing 1 to this register:
1. Flushes any pending HDP write-combining buffers
2. Invalidates GPU L2 cache lines for system memory
3. Ensures subsequent GPU reads fetch from DRAM (not stale L2)

On gfx9+ (NBIO v7+), this register is exposed to userspace via KFD's
MMIO_REMAP topology. On gfx8 (BIF v5), the register EXISTS in hardware
(`mmREMAP_HDP_MEM_FLUSH_CNTL = 0x1426` in BIF 5.1) but is NOT exposed
by the kernel driver. The fix has three parts:

#### Part 1: Kernel — expose HDP flush MMIO remap for gfx8

**File:** `kernel/PKGBUILD` (python injection targeting `vi.c` or
`amdgpu_device.c`)

Add `rmmio_remap` initialization for VI (Volcanic Islands / gfx8):

```c
// In vi_common_early_init() or equivalent:
adev->rmmio_remap.reg_offset = mmREMAP_HDP_MEM_FLUSH_CNTL * 4;
adev->rmmio_remap.bus_addr = adev->rmmio_base + mmREMAP_HDP_MEM_FLUSH_CNTL * 4;
```

This makes KFD expose the HDP flush register as a `HSA_HEAPTYPE_MMIO_REMAP`
topology node. ROCR's `GpuAgent` reads this and populates
`HDP_flush_.HDP_MEM_FLUSH_CNTL` with a userspace-accessible pointer.

**Verification:**
- After kernel install + kexec:
  `cat /sys/devices/virtual/kfd/kfd/topology/nodes/2/mem_banks/*/properties`
  should show `heap_type 3` (MMIO_REMAP) in addition to `heap_type 2`
- ROCR should report non-null HDP flush pointer

#### Part 2: CLR — use HDP flush after kernarg writes on no-atomics

**File:** `hip-runtime-amd/PKGBUILD` (python injection targeting
`rocvirtual.cpp`)

In `submitKernelInternal()`, after `nontemporalMemcpy(argBuffer, ...)`:

```cpp
if (!roc_device_.info().pcie_atomics_ && roc_device_.info().hdpMemFlushCntl) {
    *roc_device_.info().hdpMemFlushCntl = 1u;
}
```

This single MMIO write (~500ns) ensures the kernarg data written by
the CPU is visible to the GPU before the kernel dispatch reads it.

Also add HDP flush in:
- `releaseGpuMemoryFence()` before `ResetQueueStates()` — ensures all
  preceding CPU writes (kernarg, staging) are visible before pool reset
- `ManagedBuffer::Acquire()` chunk rotation — ensures chunk data is
  visible before GPU reads from the recycled chunk

#### Part 3: ROCR — use HDP flush for AQL ring buffer writes

**File:** `hsa-rocr/PKGBUILD` (sed injection targeting
`amd_aql_queue.cpp`)

In `StoreRelaxed()` (the doorbell write path), after writing the AQL
packet to the ring buffer and before ringing the doorbell:

```cpp
if (no_atomics_ && agent_->HDP_flush_.HDP_MEM_FLUSH_CNTL) {
    *agent_->HDP_flush_.HDP_MEM_FLUSH_CNTL = 1u;
}
```

This ensures the AQL packet data is visible to the CP before the doorbell
wakes it. Without this, the CP might read a stale AQL packet from L2
(though the ring buffer aliasing is less frequent than kernarg due to
larger ring size).

#### Part 4: Cleanup — remove workarounds made obsolete by HDP flush

With HDP flush providing proper CPU→GPU coherency, we can remove:
- Phase 7g double-drain barriers in `releaseGpuMemoryFence` (no longer
  needed — HDP flush ensures visibility)
- Phase 7g fine-grain kernarg pool change (ineffective on gfx8 — no
  fine-grain GPU pool)
- Phase 7g regular memcpy for kernarg (NT stores are fine with HDP flush)
- Phase 7g kernarg lifecycle instrumentation (debug logging)
- UC flag on hostAlloc may become optional (HDP flush handles coherency)

Keep:
- Phase 8 barrier dep-clearing (still needed — CP idle stall is separate)
- UC flag on hostAlloc (defense-in-depth — UC + HDP flush = belt and
  suspenders)
- ROCR Phase 7b UC signals and ring buffer (same defense-in-depth)

#### Build order

1. Kernel: add HDP MMIO remap → build → install → kexec
2. Verify: `cat /sys/.../mem_banks/*/properties` shows heap_type 3
3. ROCR: add HDP flush to ring buffer write → build → install
4. Verify: ROCR debug log shows non-null HDP flush pointer
5. HIP: add HDP flush after kernarg write → build → install
6. Verify: `test_kernarg_vram` 5/5 PASS on 10 consecutive runs

#### Verification plan

**Tier 1: HDP flush register accessible**
- KFD topology shows MMIO_REMAP node
- `check_hdp` test program gets non-null pointer
- Writing 1 to HDP flush doesn't crash

**Tier 2: Kernarg corruption eliminated**
- `test_kernarg_vram` 5/5 PASS on 10 consecutive runs (was 0-2/5)
- `test_kernarg_stability` 9/9 PASS on 10 consecutive runs
- `trace_aql` retry test: 0 failures on 10 runs

**Tier 3: Regression — existing tests still pass**
- `test_h2d_kernel` 17/17
- `hip_barrier_test` 15/15
- `hip_inference_test` 9/9

**Tier 4: Integration**
- `llama-completion --simple-io --log-disable --no-display-prompt
  -m models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf -ngl 1 -p "2+2=" -n 8`
  produces tokens without hang or fault

**Tier 5: Cross-process + dmesg**
- `run_acceptance.sh 10` — all tiers pass, zero dmesg faults

#### Risk assessment

**Probability of fix: 90%.** The HDP flush is the AMD-designed mechanism
for CPU→GPU memory visibility. It exists in gfx8 hardware. The only risk
is that the BIF 5.1 `mmREMAP_HDP_MEM_FLUSH_CNTL` register behaves
differently than the NBIO v7+ equivalent (different register semantics
or address space).

**Fallback if HDP flush doesn't work:** Direct MMIO write to the HDP
flush register from kernel space (via a custom KFD ioctl or debugfs
interface), bypassing the userspace MMIO mapping.

**Performance impact:** One MMIO write (~500ns) per kernel dispatch.
For llama.cpp inference with ~100 kernel dispatches per token at ~10ms
per token, the overhead is ~50μs per token = ~0.5% overhead. Negligible.

### Phase 8: CPU-managed barriers for no-atomics platforms

#### Problem: CP idle stall on AQL barrier deps

The Command Processor (CP) goes idle after encountering an unsatisfied AQL
barrier dependency and never re-evaluates.  With `SLOT_BASED_WPTR=0` (direct
doorbell), the CP only re-reads WPTR on a doorbell hit.  When the barrier dep
signal is decremented externally (by another queue or CPU), the CP doesn't
notice — it's idle, no new doorbell arrives (same WPTR), no `DOORBELL_HIT`
interrupt fires.

This causes:
- llama.cpp inference hangs (CP stalls on internal barrier, never resumes)
- Queue drain failures → scratch/context freed while GPU executing → VM faults
- All "use-after-free" VM faults trace back to failed queue drain

#### SLOT_BASED_WPTR=2 is not viable without ATC/IOMMU

`SLOT_BASED_WPTR=2` (CP polls WPTR from memory) would solve the idle stall by
making the CP periodically re-evaluate.  However, the CP polling engine reads
from `cp_hqd_pq_wptr_poll_addr` via ATC (Address Translation Cache), not via
GPUVM.  On gfx8 without ATC/IOMMU (our platform: Westmere PCIe 2.0, no ATS,
no PASID), the CP cannot resolve the poll address.

Evidence:
- Allocated poll buffer from kernarg pool (CPU VA = GPU VA, GPUVM-mapped)
- Set `CP_PQ_WPTR_POLL_CNTL.EN=1` and queue bit in `CNTL1`
- CP never fetches packets — 100% CPU spin, no VM fault, no dmesg errors
- DRM compute rings on gfx8 never use `SLOT_BASED_WPTR=2` (pure doorbell)
- KFD was designed for APUs with shared memory and ATC (Kaveri)
- IOMMU without ATS/PASID (our platform) cannot provide ATC translation

IOMMU alone (Intel VT-d) provides DMA remapping but NOT the on-demand VA→PA
translation that ATC needs.  ATS (PCIe 3.0) and PASID are required for the
GPU to request translations from the IOMMU.  Westmere is PCIe 2.0.

#### What's logically impossible on gfx8/no-atomics/no-ATC

| Capability | Status | Implication |
|---|---|---|
| GPU atomic write to system memory | **Impossible** | GPU can't decrement HSA signals |
| ATC address translation | **Impossible** | CP can't resolve CPU VAs for polling |
| SLOT_BASED_WPTR=2 polling | **Impossible** | Depends on ATC |
| System-scope L2 flush (ISA) | **Not in ISA** | gfx8 has no L2 flush instruction |
| GPU-side barrier dep evaluation | **Hangs** | CP goes idle, never re-evaluates |
| GPU-side signal completion | **Unreliable** | Needs PCIe atomics |

| Capability | Status | |
|---|---|---|
| GPUVM ring buffer fetch | **Works** | CP fetches AQL packets |
| GPUVM RPTR write | **Works** | Proven with rptr_gpu_buf_ |
| Direct doorbell (SLOT_BASED_WPTR=0) | **Works** | Doorbell value IS WPTR |
| CPU-side signal management | **Works** | Bounce buffer, CPU waits |
| Staging buffer copies | **Works** | All H2D/D2H sizes verified |

#### Root cause: CLR submits AQL barriers with GPU-side deps

ROCR's BlitKernel already has a CPU-wait bypass (Phase 6a): when
`NoPlatformAtomics()`, it CPU-waits all dep signals and clears them before
submitting the AQL packet.  This eliminates barrier deps from the blit copy
path.

CLR's `dispatchBarrierPacket()` has **no equivalent bypass**.  It submits
`BARRIER_AND` packets with dep_signals that require the CP to evaluate them on
the GPU.  When any dep is unsatisfied, the CP stalls permanently.

Complete barrier source inventory:

**ROCR (1 site, BYPASSED):**
- `BlitKernel::SubmitLinearCopyCommand()` — Phase 6a CPU-wait ✅

**CLR (10+ call sites, NOT BYPASSED):**

| Call site | Function | Dep signals? | Risk |
|---|---|---|---|
| `releaseGpuMemoryFence()` | Memory ordering | Yes (active signal) | **HANG** |
| `dispatchBlockingWait()` | Multi-signal wait | Yes (1-5 deps) | **HANG** |
| Marker (cache flush) | hipEvent/Stream sync | Yes (active signal) | **HANG** |
| Marker (no cache) | Lightweight sync | Sometimes | **HANG if deps** |
| KernelArgPool chunk | Pool flush | Possibly | **HANG if deps** |
| Stream write | hipStreamWriteValue | Yes | **HANG** |
| VM mapping | Virtual memory ops | Yes | **HANG** |
| Dynamic parallelism | Child dispatch sync | Yes | **HANG** |
| Accumulate | NOP barrier | No deps typically | Low risk |

#### Fix: CPU-wait barrier deps in CLR

Single change in `dispatchBarrierPacket()`: when `!dev().info().pcie_atomics_`,
CPU-wait each dep_signal to completion before submitting the AQL barrier.  The
barrier is still submitted (with completion_signal) but with all dep_signal
slots cleared — the CP sees it as an immediate barrier and completes it
instantly.

```
dispatchBarrierPacket(header, skipSignal):
  if (!dev().info().pcie_atomics_) {
      for each dep_signal[j] in barrier_packet_:
          if (dep_signal[j].handle != 0):
              while (hsa_signal_load_relaxed(dep_signal[j]) > 0):
                  yield()
              dep_signal[j] = {0}   // clear — no GPU-side eval needed
  }
  // ... normal AQL submission with completion_signal only
```

**Why this is safe on gfx8:**
- AQL queues are in-order — removing deps doesn't reorder execution
- gfx8 fence scopes are effectively NOPs (no ISA L2 flush; SNOOPED=1 handles
  coherency at PTE level via kernel patch 0009)
- Completion signal is set by CP after barrier completes (bounce buffer handles
  signal lifecycle)
- Pattern identical to proven Phase 6a BlitKernel fix

**Performance impact:** Barriers become synchronous CPU waits, serializing the
pipeline at barrier points.  For llama.cpp (many small kernel dispatches), this
may reduce throughput.  But correctness is required before optimization, and the
no-atomics platform is inherently CPU-bound for signal management.

#### Implementation

Single python injection in `hip-runtime-amd/PKGBUILD` targeting
`rocclr/device/rocm/rocvirtual.cpp` `dispatchBarrierPacket()`.  Add CPU-wait
loop before the existing AQL submission code, conditional on
`!dev().info().pcie_atomics_`.

#### Revert: SLOT_BASED_WPTR=2 changes

Revert kernel patch 0008 to the working `SLOT_BASED_WPTR=0` (direct doorbell)
for no-atomics.  Remove `wptr_gpu_buf_` from ROCR patch 0003.  Remove
`CP_PQ_WPTR_POLL_CNTL` injection from kernel PKGBUILD.  These changes are dead
code on platforms without ATC.

Keep the `gfx8_wptr_poll` module parameter as a debug/documentation artifact
(default=0 now means direct doorbell for no-atomics).

#### Acceptance tests

All tests must pass on clean boot (kexec).  Each test has a timeout; hang =
fail.  Tests are ordered from isolated/simple to integrated/complex.  A failure
in an early test means later tests are unreliable.

**Tier 1: Regression (existing tests, must not regress)**

| # | Test | What it covers | Timeout | Pass criteria |
|---|------|---------------|---------|---------------|
| 1.1 | `test_h2d_kernel` | H2D + GPU verify, alloc/free/realloc, 1-64MB | 120s | 17/17 |
| 1.2 | `test_h2d_boundary` | H2D boundary sizes near staging buffer edge | 60s | 10/10 |
| 1.3 | `test_memset_verify` | hipMemset + D2H readback | 30s | PASS |
| 1.4 | `hip_torture_test` tests 1-6 | Dispatch correctness (serial, multi-stream, interleaved) | 120s | 6/6 |
| 1.5 | `hip_torture_test` tests 7-12 | Completion visibility (signal storm, D2H sweep, ring wrap) | 120s | 6/6 |

**Tier 2: Barrier-specific (new tests targeting Phase 8 fix)**

| # | Test | What it covers | Timeout | Pass criteria |
|---|------|---------------|---------|---------------|
| 2.1 | Kernel→memcpy→kernel chain | `hipLaunchKernel` + `hipMemcpy(D2H)` + `hipLaunchKernel` interleaved 100x — exercises `releaseGpuMemoryFence()` barriers | 60s | 100 correct readbacks |
| 2.2 | hipEventRecord + hipEventSynchronize | Record event after kernel, sync on it, verify result — exercises marker barrier path | 30s | 500 iterations |
| 2.3 | hipStreamSynchronize under load | Launch 100 kernels, hipStreamSynchronize, verify all completed — exercises flush barrier | 30s | 10 rounds |
| 2.4 | Cross-stream event wait | Stream A records event, stream B waits on it via hipStreamWaitEvent, stream B kernel reads stream A result — exercises `dispatchBlockingWait()` | 60s | 100 iterations |
| 2.5 | hipMemcpyAsync + hipStreamSync | Async H2D on stream, sync, verify — exercises staging barrier + marker | 30s | 1-64MB sweep |
| 2.6 | Rapid hipMalloc/hipFree cycle | Alloc 1MB, H2D, kernel, D2H verify, free — repeat 200x — exercises queue drain barriers during free | 120s | 200/200 correct |
| 2.7 | Back-to-back hipDeviceSynchronize | Launch kernel, device sync, repeat 500x — exercises global barrier path | 60s | 500 syncs |

**Tier 3: Stress (sustained load, race conditions)**

| # | Test | What it covers | Timeout | Pass criteria |
|---|------|---------------|---------|---------------|
| 3.1 | Multi-stream sustained | 4 streams, each: 200 kernels + hipStreamSync, all concurrent — exercises multi-queue barrier interaction | 120s | All streams complete, correct results |
| 3.2 | Alloc/compute/free pipeline | Pipeline: alloc→H2D→kernel→D2H→verify→free, 500 iterations, overlapping allocs — exercises barrier + VM lifecycle | 300s | 500/500 correct |
| 3.3 | Mixed sizes sustained | Random sizes 4KB-16MB, random H2D/D2H direction, random kernel dispatch, 300 ops — exercises all copy + barrier paths | 180s | Zero corruption, zero hangs |
| 3.4 | Queue pressure | Launch 4096 tiny kernels without sync, then hipDeviceSynchronize — exercises ring buffer wrap + barrier accumulation | 60s | Correct final value |

**Tier 4: Integration (llama.cpp-like patterns)**

| # | Test | What it covers | Timeout | Pass criteria |
|---|------|---------------|---------|---------------|
| 4.1 | Model load pattern | 50x sequential: alloc VRAM, H2D 1MB, verify — simulates tensor loading | 120s | 50/50 correct |
| 4.2 | Inference pattern | Kernel dispatch chain: 10 kernels writing to shared buffer, readback, verify — simulates forward pass | 60s | 100 iterations correct |
| 4.3 | `llama-cli -ngl 1 -p "2+2=" -n 8` | Actual llama.cpp inference with 1 GPU layer | 120s | Tokens generated, no hang, no VM fault |
| 4.4 | `llama-cli -ngl 1 -n 32` longer generation | Sustained token generation | 300s | 32 tokens, clean exit |

**Tier 5: Cleanup (process exit, error paths)**

| # | Test | What it covers | Timeout | Pass criteria |
|---|------|---------------|---------|---------------|
| 5.1 | Clean process exit | Run any kernel, `_exit(0)` — verifies no hang during HIP shutdown | 10s | Exit code 0 |
| 5.2 | Exit with pending work | Launch 1000 async kernels, `_exit(0)` immediately — verifies queue destroy doesn't hang | 10s | Exit code 0 |
| 5.3 | `dmesg` audit | After all tests: `dmesg \| grep 'fault VA'` | — | Zero faults |

**Implementation:** Tests 1.x use existing binaries.  Tests 2.x-5.x will be a
new `hip_barrier_test.cpp` compiled with hipcc.  Each test function returns
bool, with alarm-based timeout.  Test runner prints per-test PASS/FAIL and
summary.

**Run protocol:**
1. Cold boot or kexec to clean GPU state
2. `make -C tests` to rebuild all test binaries
3. Run tiers in order; stop on first failure in tier 1 (fundamental regression)
4. Tiers 2-5 run even if some tests fail (collect all failures)
5. Final `dmesg` audit (tier 5.3)

### Phase 8b: llama.cpp GPU inference optimization

Once inference works, measure performance:
- Token generation rate (t/s) vs CPU-only baseline (13.3 t/s)
- VRAM usage with different `-ngl` values (2GB limit)
- Optimal context size for the 2GB card
- Compare Q4_K_M vs Q5_0 quantizations

### Phase 9: Upstream preparation

- Clean up patches: remove debug logging, consolidate PKGBUILD sed injections into proper patches
- Write rocrtst-idiomatic tests (SignalCpuWriteGpuBarrier, MixedSizeH2DStress, etc.)
- Document the no-atomics platform support story for upstream consideration
- Consider submitting kernel patches to LKML (kfd_mqd_manager_vi.c changes are clean and well-documented)

### Phase 10: Additional ROCm components (stretch)

- rocFFT, hipBLAS, MIOpen — evaluate if needed for micro-LLM inference
- hipSPARSE — might help with sparse attention patterns
- ROCm SMI monitoring — temperature/power tracking for the WX 2100

## Decisions Log (continued)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-13 | SpinMutex + mfence in UpdateReadDispatchId | Race condition between concurrent callers; mfence ensures clflush ordering on Intel |
| 2026-03-13 | RPTR_BLOCK_SIZE 5→4 for no-atomics | Per-packet RPTR writes needed for bounce buffer completion tracking |
| 2026-03-13 | Retain/Release on bounce buffer signals | Prevents AsyncEventsLoop crash on freed signals (use-after-free) |
| 2026-03-14 | GPU L2 signal caching theory DISPROVEN | 14/14 barrier tests pass; signals are coherent (fine-grain pool, COHERENT flag) |
| 2026-03-14 | CP idle stall root cause confirmed | STALL logging: RPTR frozen 3.5M iterations; doorbell kick with WPTR+1 wakes CP |
| 2026-03-14 | SLOT_BASED_WPTR=2 dead on gfx8 | CP can't GPUVM-read poll address without ATC/UTCL2; tested with kernarg pool |
| 2026-03-14 | NOP barrier kick as temporary workaround | Wakes idle CP, passes 100/100 mixed H2D, drifts at ~286 ops |
| 2026-03-15 | WaitCurrent between H2D staging chunks | Eliminates inter-chunk barrier deps; mirrors D2H path behavior |
| 2026-03-15 | cpu_wait_for_signal for no-atomics in CLR | Force CPU-side waits in WaitingSignal(); eliminates CLR-level barrier deps |
| 2026-03-15 | Combined NOP + cpu_wait: 286 ops stable | NOP handles CP wake, cpu_wait prevents CLR barriers; drift remains from ROCR barriers |
| 2026-03-16 | SLOT_BASED_WPTR=0 fixes CP idle stall | 1000 rapid serial dispatches pass; CP directly reads WPTR from doorbell, no polling needed |
| 2026-03-16 | D2H coherency: clflush insufficient under load | Iter 297-379 of 500: stale/partial data; llama.cpp hipHostFree hangs on SyncAllStreams |
| 2026-03-20 | VM fault root cause: pinned host memory path | CLR unpins (clears PTEs) before DMA reads complete; disabling pinning on no-atomics fixes all sizes 1-512MB |
| 2026-03-20 | kernel PT coherence patches were red herrings | CPU_ACCESS_REQUIRED, wait=true, vm_update_mode=3 — all unnecessary; PTEs were correct, just cleared too early by CLR unpin |
| 2026-03-20 | hip-runtime-amd-polaris 7.2.0-6 | Fix 4: `doHostPinning &&= dev().info().pcie_atomics_` in getBuffer(); forces staging pool path |
| 2026-03-21 | SLOT_BASED_WPTR=2 confirmed dead on gfx8/no-ATC | CP polling engine uses ATC, not GPUVM; kernarg pool address mapped but CP never reads; no VM fault confirms CP ignores poll addr |
| 2026-03-21 | Phase 8: CPU-managed barriers | Eliminate all GPU-side barrier dep evaluation; CPU-wait deps in CLR dispatchBarrierPacket(); same pattern as proven Phase 6a blit fix |
| 2026-03-21 | Revert SLOT_BASED_WPTR=2 changes | Dead code on no-ATC platforms; revert to SLOT_BASED_WPTR=0 (direct doorbell) which is proven working |
