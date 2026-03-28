# Phase 16 Patching Notes

## What's Done

### Infrastructure (COMPILES)
- 3 new `GGML_OP_*` enum entries in `ggml.h` (via sed after GATED_DELTA_NET)
- 3 op name strings in `ggml.c` (via sed after [GGML_OP_GATED_DELTA_NET])
- Static assert count updated (`== N` → `== N + 3`)
- 3 tensor creation functions in `ggml.c` (via sed before ggml_ssm_scan)
- 3 function declarations in `ggml.h` (via sed before ggml_ssm_scan)
- 3 pipeline fields in device struct (via sed after pipeline_ssm_conv_f32)
- 3 push constant structs (via sed before vk_op_ssm_conv_push_constants)
- 3 create_pipeline calls (via sed after ssm_conv create_pipeline)
- 3 dispatch functions (via sed before ggml_vk_ssm_conv)
- 3 pipeline selection cases (via sed after return pipeline_ssm_conv_f32)
- 3 graph handler cases (via sed after ggml_vk_ssm_conv dispatch call)
- 3 op support cases (via sed using SSM_CONV/SSM_SCAN pair anchor)
- 3 shader files registered in vulkan-shaders-gen.cpp
- 3 .comp files copied from patches/vulkan/
- Fence busy-wait → vkWaitForFences
- nodes_per_submit = 2000
- SSM_CONV + SILU fusion (sed on ssm_conv.comp + qwen35.cpp)
- sigmoid fusion into gated_delta_net.comp (sed on .comp + qwen35.cpp)

### What Failed / Needs Fixing

**qwen35.cpp gate_prep wiring (line 237):**
```
error: redeclaration of 'ggml_tensor* gate'
```
The sed that replaces add+softplus+mul with fused_gate_prep creates a
new `gate` variable, but the original `gate` variable still exists from
a later reshape. The sed range matching didn't delete all 3 original
lines cleanly.

**Fix needed:** The sed command at Step 14 should replace these 3 lines:
```cpp
ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);
```
With:
```cpp
ggml_tensor * gate = ggml_fused_gate_prep(ctx0, alpha, model.layers[il].ssm_dt, model.layers[il].ssm_a);
```

The current sed uses a range match that doesn't work cleanly. Replace
with 3 individual sed commands:
```bash
sed -i 's/ggml_tensor \* alpha_biased.*= ggml_add.*ssm_dt.*/\/\/ Phase 16: fused into gate_prep/' "$_q35"
sed -i 's/ggml_tensor \* alpha_softplus.*= ggml_softplus.*alpha_biased.*/\/\/ Phase 16: fused/' "$_q35"
sed -i 's/ggml_tensor \* gate.*= ggml_mul.*alpha_softplus.*ssm_a.*/ggml_tensor * gate = ggml_fused_gate_prep(ctx0, alpha, model.layers[il].ssm_dt, model.layers[il].ssm_a);/' "$_q35"
```

**fused_gated_norm and fused_dual_l2_norm model graph changes NOT YET DONE.**
These need similar sed replacements in qwen35.cpp:

For fused_gated_norm (replaces build_norm_gated):
```cpp
// Before:
ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
ggml_tensor * gated_silu = ggml_silu(ctx0, gate);
return ggml_mul(ctx0, normalized, gated_silu);

// After:
return ggml_fused_gated_norm(ctx0, input, weights, gate);
```

For fused_dual_l2_norm:
```cpp
// Before:
q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

// After — complex because output is a combined tensor that needs splitting
// May need to reconsider the dual_l2_norm approach
```

## Sed Anchoring Lessons

1. **`/pattern/a\` matches ALL occurrences.** Use `0,/pattern/{/pattern/a\...}`
   for first-match-only on GNU sed. But this breaks with multiline inserts.

2. **Use unique context patterns.** `pipeline_ssm_conv_f32;$` matches both
   the struct field and the return statement. Use `return.*pipeline_ssm_conv_f32`
   to match only the return.

3. **Multi-line function declarations break line-after insertion.** The
   `ggml_ssm_conv` declaration spans 3 lines. `/ggml_ssm_conv/a\` inserts
   after the FIRST line, breaking the declaration.

4. **Use BEFORE anchors (i\) instead of AFTER (a\)** when the target pattern
   appears at the START of a multi-line construct. `ggml_ssm_scan` is a clean
   insertion point because it's a standalone function definition start.

5. **For switch cases:** `case GGML_OP_SSM_CONV:` appears in 4 switch
   statements. Use adjacent case (`SSM_CONV` followed by `SSM_SCAN`) as a
   unique 2-line anchor: `/SSM_CONV:/{n;/SSM_SCAN:/i\...}`.

## Files Modified (via sed in PKGBUILD prepare())

| File | What | Anchor |
|------|------|--------|
| ggml.h | 3 enum entries | after GATED_DELTA_NET |
| ggml.h | 3 function decls | before ggml_ssm_scan |
| ggml.c | 3 op name strings | after [GGML_OP_GATED_DELTA_NET] |
| ggml.c | static_assert +3 | s/== N/== N + 3/ |
| ggml.c | 3 function impls | before ggml_ssm_scan |
| ggml-vulkan.cpp | pipeline fields | after pipeline_ssm_conv_f32 (struct) |
| ggml-vulkan.cpp | push const structs | before vk_op_ssm_conv_push_constants |
| ggml-vulkan.cpp | create_pipeline | after ssm_conv create_pipeline |
| ggml-vulkan.cpp | dispatch functions | before ggml_vk_ssm_conv |
| ggml-vulkan.cpp | pipeline select | after return pipeline_ssm_conv_f32 |
| ggml-vulkan.cpp | graph handler | after ggml_vk_ssm_conv call |
| ggml-vulkan.cpp | op support | SSM_CONV+SSM_SCAN pair |
| vulkan-shaders-gen.cpp | 3 shader regs | after ssm_conv_f32 |
| qwen35.cpp | graph changes | direct pattern match |

## Shader Files

| File | Location | Purpose |
|------|----------|---------|
| fused_gate_prep.comp | patches/vulkan/ | add + softplus + mul |
| fused_gated_norm.comp | patches/vulkan/ | rms_norm + silu + mul |
| fused_dual_l2_norm.comp | patches/vulkan/ | l2_norm(q) + l2_norm(k) |

## Next Steps

1. Fix qwen35.cpp gate_prep sed (variable redeclaration)
2. Add qwen35.cpp gated_norm sed
3. Test: does fused_gate_prep alone give measurable speedup?
4. If yes: add gated_norm and dual_l2_norm model graph wiring
5. Full benchmark: target 30+ t/s

## Current Package State

- Installed: `llama-cpp-vulkan-polaris` b8508-11 (SSM_CONV+SILU + sigmoid fusion)
- Building: b8508-12 (full fused ops — qwen35.cpp error, needs fix)
- Verified working: 23 t/s at ctx=8192 with fusions 1+2
