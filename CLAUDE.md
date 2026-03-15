# CLAUDE.md — ROCm GCN3 Restoration for Arch Linux

"Work as if you live in the early days of a better world."

## Project Purpose

Restore ROCm support for GCN 3rd-generation GPUs (gfx801, gfx802, gfx803) on Arch Linux. AMD dropped these targets from official ROCm builds. We maintain an Arch PKGBUILD that clones ROCm components as git submodules and applies minimal patches to re-enable these ISAs.

Target hardware examples: Fiji (gfx803), Tonga (gfx802), Carrizo (gfx801) — covers cards like R9 Fury/Nano, R9 285/380, and Carrizo APUs.

## Repository Structure

```
rocm-gfx80x/
├── CLAUDE.md              # This file
├── PLAN.md                # Current design decisions and status
├── PKGBUILD               # Arch makepkg build script
├── patches/               # Our patches, organized per-component
│   ├── llvm-project/      # AMDGPU backend target re-enablement
│   ├── ROCR-Runtime/      # Firmware/ISA loading for gfx80x
│   ├── ROCclr/            # Compiler runtime / device enumeration
│   ├── HIP/               # HIP runtime gfx80x paths
│   ├── rocBLAS/           # Optional: BLAS kernels/Tensile configs
│   └── ...                # Additional components as needed
├── submodules/            # ROCm git submodules (not committed, cloned by PKGBUILD)
└── keys/                  # GPG keys for source verification if needed
```

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### Think Before Coding
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- **Stop when confused.** Don't generate plausible-sounding code to mask uncertainty. Request clarification.
- When a theory is disproven by evidence, say so immediately and pivot. Don't defend dead hypotheses.

### Simplicity First
- No features beyond what was asked. No abstractions for single-use code.
- No speculative "flexibility" or error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.
- **Three similar lines are better than a premature abstraction.** Don't create helpers for one-time use.
- Don't design for hypothetical future requirements.

### Surgical Changes
- Don't "improve" adjacent code, comments, or formatting.
- Match existing style, even if you'd do it differently.
- Remove imports/variables that YOUR changes made unused, but don't touch pre-existing dead code.
- Every changed line should trace directly to the user's request.
- **When reviewing, only flag issues in OUR code. Never fix, reorder, or restyle pre-existing upstream code — even if it violates conventions. Our patches must be minimal diffs against upstream.**
- Don't add docstrings, comments, or type annotations to code you didn't change.
- Don't add backwards-compatibility shims, `// removed` comments, or renamed `_unused` vars. If something is unused, delete it completely.

### Goal-Driven Execution
- Transform tasks into verifiable goals with success criteria.
- For multi-step tasks, state a brief plan with verification steps.
- **Test-first when possible.** Write the verification command before writing the fix.
- When debugging, form a hypothesis, design a test that can disprove it, run the test, then act on the result. Don't skip straight to a fix based on speculation.

### Audited PLAN.md and PHASEx.md (MUST)
All changes to `PLAN.md` and `PHASEx.md` files MUST be committed immediately.
- Every edit to `PLAN.md` or any `PHASEx.md` triggers a git commit. Do not batch with code changes.
- After completing a plan step in code, update the relevant plan file to reflect what was actually implemented, then commit as a separate commit.
- `PLAN.md` and `PHASEx.md` files live in the top-level repository. Never inside submodules.
- Before any decision that changes design, architecture, or approach: update the plan FIRST, commit it, THEN write code. The record reflects intent, not reconstruction.

### Maintain MEMORY.md
`MEMORY.md` is an append-only persistent scratchpad recording key decisions, discovered constraints, and lessons learned.
- After completing a significant task, resolving a non-obvious bug, or discovering an unexpected constraint, add a short entry.
- Update `MEMORY.md` in a separate commit. Do not bundle with code changes.
- Write entries as you go — don't wait until end of session. If unsure whether to record something, record it.
- Do not delete or rewrite old entries. If an earlier entry is wrong, add a new entry correcting it.
- Reading `MEMORY.md` should be enough to avoid repeating past mistakes when starting a new session.

## Core Technical Rules

### Patch Philosophy
- **Minimal diffs only.** Every hunk in a patch must be necessary to re-enable gfx801/gfx802/gfx803. No drive-by fixes.
- Generate patches with `git diff` or `git format-patch` against the exact upstream tag we track as a submodule.
- Patches go in `patches/<component>/` named descriptively: `0001-re-enable-gfx803-target.patch`, not `fix.patch`.
- Each patch file should have a comment header explaining what it does and why. One logical change per patch.
- If upstream reorganizes code between versions, regenerate patches from scratch against the new base rather than trying to rebase old patches.

### PKGBUILD Rules
- Follow Arch packaging standards: https://wiki.archlinux.org/title/Creating_packages
- Pin every submodule to an exact ROCm release tag (e.g., `rocm-6.2.0`). Never track branches.
- Use `_rocm_version` variable at the top; all submodule tags derive from it.
- Apply patches in `prepare()` using `git apply` or `patch -p1` within the relevant submodule directory.
- `makedepends` and `depends` must be explicit and complete — no implicit reliance on the user's installed packages.
- Split packages where Arch convention expects it (e.g., `rocm-llvm` separate from `hip-runtime`).
- Include `check()` function if upstream provides tests that can validate gfx80x without hardware (e.g., lit tests for LLVM target availability).
- Use `sha256sums` for all non-git sources; submodules are verified by commit hash.

### GFX Target Details
When patching, the typical changes needed per component are:

**LLVM/Clang AMDGPU backend:**
- Ensure `gfx801`, `gfx802`, `gfx803` are present in target processor lists and not gated behind removal guards.
- Check `lib/Target/AMDGPU/` — GCN subtarget definitions, ISA feature maps, and the `getArchNameAMDGCN()` / processor table.

**ROCR-Runtime (HSA runtime):**
- Device enumeration: ensure gfx80x chip IDs map to valid `hsa_agent_t` structs.
- Firmware/microcode loading paths for these ISAs.

**HIP / ROCclr:**
- Device-to-ISA mapping tables that gate compilation and dispatch.
- `hipDeviceProp_t` population for these architectures.

**rocBLAS / rocFFT / MIOpen (optional, per user need):**
- Tensile logic files or kernel generator configs that list supported architectures.
- These are large; only patch if explicitly requested.

### Version Strategy
- We track a single ROCm release at a time. All submodules must be from the same release tag.
- Document the tracked version in `PLAN.md` under a `## Current Target` heading.
- When bumping versions: first update `PLAN.md`, then update `_rocm_version`, then regenerate all patches against new submodule state. Test build before committing patches.

## Build & Test Commands

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd rocm-gfx80x

# Build (Arch)
makepkg -s          # install makedepends and build
makepkg -si         # build and install

# Verify gfx803 target is present in built LLVM
./pkg/*/usr/bin/llc --version | grep gfx803

# Quick HIP device query (requires hardware)
./pkg/*/usr/bin/rocminfo | grep -i "gfx80"

# Run LLVM lit tests for AMDGPU (if check() is wired up)
makepkg -s --check
```

## Pitfalls & Prior Knowledge

- **ROCm's CMake is sprawling.** Don't chase transitive CMake issues — patch only the source files that gate ISA support. Build system changes should be last resort.
- **`amd_comgr` links against the ROCm LLVM.** If LLVM patches change symbol visibility or remove GCN3 code objects, comgr will fail at runtime. Always verify comgr can compile a trivial HIP kernel targeting gfx803 after patching.
- **ISA removal is rarely a single commit.** Upstream typically removes targets across multiple repos over several releases. When identifying what to patch, grep all submodules for the target strings (`gfx801`, `gfx802`, `gfx803`, `GFX8`, `GCN3`, `VI`, `volcanic islands`) — the naming is inconsistent.
- **Arch's `rocm-*` packages in `extra/` conflict.** Our PKGBUILD must either `provides=()` / `conflicts=()` with the official packages, or install to a non-conflicting prefix. Document the chosen approach in `PLAN.md`.
- **Kernel driver (amdgpu) still supports these GPUs.** The kernel side is fine — only userspace ROCm dropped support. Don't touch kernel modules.
- **PCIe atomics:** gfx803 on some platforms lacks PCIe atomics support. This is a hardware/BIOS limitation, not something we patch. Document it as a known caveat.

## Commit Messages

```
component: short description of what changed

Why this change is needed for gfx80x support.
Refs: upstream commit hash or issue if relevant.
```

Component prefixes: `pkgbuild`, `patch/llvm`, `patch/rocr`, `patch/hip`, `patch/rocblas`, `docs`, `ci`.

## What NOT To Do

- Don't fork entire ROCm repos. We use submodules + patches.
- Don't patch test infrastructure or CI configs — only production code paths.
- Don't add gfx80x to performance tuning tables (Tensile, MIOpen) unless specifically asked — these are enormous and fragile.
- Don't "future-proof" patches for GPUs we don't target. gfx801, gfx802, gfx803 only.
- Don't modify this CLAUDE.md without explicit instruction.
