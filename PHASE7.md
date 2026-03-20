# Phase 7: llama.cpp GPU Inference

## Status: BLOCKED on CP idle stall — VM fault RESOLVED

## Progress

### Phase 7a-f: Signal completion & memory coherency (COMPLETE)
- 11/12 torture tests pass, clean process exit
- Interrupt-based bounce buffer handles signal completion on no-atomics
- Fine-grain staging (MTYPE=CC) fixes GPU L2 flush for system memory
- CPU cache flush (clflush) fixes stale reads after GPU DMA writes
- cpu_wait_for_signal_ avoids AQL barrier deps that stall the CP

### Phase 7g: VM fault root cause (RESOLVED)
- **Root cause**: CLR's pinned host memory path unpins (clears PTEs) before
  DMA engine finishes reading from the pinned GPU VA range
- **Fix**: Disable pinning on no-atomics platforms (`!pcie_atomics_` in getBuffer)
- **Result**: All H2D/D2H sizes 1MB-512MB pass with zero bad bytes
- See PHASE7g.md for full details

### Phase 7h: llama.cpp inference (BLOCKED)
- llama.cpp loads model H2D successfully (no VM fault with Fix 4)
- llama.cpp hangs during prompt eval (CP idle stall, NOT a memory issue)
- No VM faults during the hang — the GPU simply stops processing AQL packets
- This is the existing SLOT_BASED_WPTR / barrier dependency / CP idle stall
  problem that predates the VM fault investigation

## Remaining Blocker: CP Idle Stall

The Command Processor goes idle after encountering a barrier packet with
unsatisfied dependencies. With SLOT_BASED_WPTR=0, the CP does not re-check
the write pointer after going idle, so new packets are never processed.

Known mitigations attempted:
- SLOT_BASED_WPTR=0 (current): CP goes idle on unsatisfied barrier
- SLOT_BASED_WPTR=2: Requires GPU-visible poll address; crashes on Polaris
- NOP kick: Temporarily advances write pointer; drifts over time
- cpu_wait_for_signal_: Serializes operations (avoids barriers) but some
  CLR/HIP internal operations still emit barrier packets

## Packages (Current)

- `linux-lts-rocm-polaris` 6.18.16-24
- `hsa-rocr-polaris` 7.2.0-13
- `hip-runtime-amd-polaris` 7.2.0-6 (includes Fix 4: no pinning)
- `rocblas-gfx803` 7.2.0-2
- `llama-cpp-rocm-polaris` b7376-1

## Success Criteria

- [x] H2D/D2H copies work at all sizes (1MB-512MB)
- [x] No VM faults during model load
- [ ] `llama-cli -ngl 1 -p "Hello" -n 8` produces 8 tokens on GPU
- [ ] Process exits cleanly
- [ ] Token generation rate measured
