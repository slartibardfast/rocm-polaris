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

### Phase 2d: rocBLAS (PKGBUILD READY)
- Apply existing `0001-re-enable-gfx803-target.patch`
- Build with `makepkg -s`

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

hsa_dispatch_test:   PASS  (barrier dispatch + signal completion)
hsa_memcopy_test2:   PASS  (VRAM→system round-trip via ROCR path)
hip_smoke 5a (init):     PASS
hip_smoke 5b (props):    PASS  (AMD Radeon Pro WX 2100, gfx803)
hip_smoke 5c (malloc):   PASS
hip_smoke 5d (memcpy):   PASS  ← PREVIOUSLY FAILING, NOW FIXED
hip_smoke 5e (memset):   PASS
hip_smoke 5f (kernel):   PASS  (simple addition kernel returns 42)
hsa_cache_timing:    1.0x (kernarg pool still appears WB-cached — see open question)
```

### Status

- [x] ROCR patch 0005: FlushCpuCache in CopyMemory — **THE FIX** (hsa-rocr-polaris 7.2.0-6)
- [x] Kernel patch 0009: honor UNCACHED in TTM (defense-in-depth, 6.18.16-9)
- [x] HIP hipMemcpy D2H: **VERIFIED WORKING**
- [x] HIP kernel launch: **VERIFIED WORKING**
- [ ] Investigate why cache_timing still shows WB despite kernel 0009 (non-blocking)
- [ ] `hsa_memcopy_test` fill path segfault (non-blocking)
