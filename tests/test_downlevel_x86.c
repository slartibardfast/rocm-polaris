/*
 * test_downlevel_x86.c — correctness test for Layer 1 shim helpers.
 *
 * Compiles standalone with gcc. Compares ggml_x86_cvtph_ps against
 * a scalar reference (IEEE 754 bit manipulation) and checks that
 * the result is byte-identical for the full 16-bit input range
 * (65536 fp16 values), plus spot-checks for known values.
 *
 * Build & run:
 *   gcc -O3 -march=native \
 *       -I/home/llm/rocm-polaris/llama.cpp/src/llama.cpp-b8508/ggml/src/ggml-cpu \
 *       tests/test_downlevel_x86.c -o /tmp/test_downlevel_x86
 *   /tmp/test_downlevel_x86
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "arch/x86/downlevel.h"

/* Scalar IEEE 754 fp16 → fp32 reference.
 * Full 16-bit table, no shortcuts, matches what F16C hardware does. */
static float ref_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                      /* ±0 */
        } else {
            /* Denormal: value = mant * 2^-24. Convert to normalised fp32. */
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            exp++;
            mant &= ~0x400u;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13); /* ±inf or NaN */
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* Bit-exact comparison that treats both NaN bit patterns as equal.
 * We allow the shim to produce a different NaN payload than the
 * reference (NaN propagation rules are underspecified). */
static int bits_equal(float a, float b) {
    if (isnan(a) && isnan(b)) return 1;
    uint32_t ua, ub;
    memcpy(&ua, &a, 4);
    memcpy(&ub, &b, 4);
    return ua == ub;
}

static int test_full_range(void) {
    int mismatches = 0;
    int first_bad  = -1;
    for (int h = 0; h < 65536; h++) {
        uint16_t in[4] = { (uint16_t)h, (uint16_t)h, (uint16_t)h, (uint16_t)h };
        __m128i v = _mm_loadl_epi64((const __m128i *) in);
        __m128 y = ggml_x86_cvtph_ps(v);
        float out[4];
        _mm_storeu_ps(out, y);

        float ref = ref_fp16_to_fp32((uint16_t) h);
        if (!bits_equal(out[0], ref)) {
            if (first_bad < 0) first_bad = h;
            mismatches++;
        }
    }
    printf("full-range test: %d mismatches (first at 0x%04x)\n",
           mismatches, first_bad);
    return mismatches;
}

static void test_spot_checks(void) {
    const struct { uint16_t h; float expected; const char * name; } cases[] = {
        { 0x0000,  0.0f,         "zero" },
        { 0x8000, -0.0f,         "neg zero" },
        { 0x3C00,  1.0f,         "one" },
        { 0xBC00, -1.0f,         "neg one" },
        { 0x7BFF,  65504.0f,     "fp16 max" },
        { 0xFBFF, -65504.0f,     "fp16 min (neg max)" },
        { 0x7C00,  INFINITY,     "+inf" },
        { 0xFC00, -INFINITY,     "-inf" },
        { 0x4200,  3.0f,         "three" },
        { 0x3555,  0.333252f,    "approx 1/3" },  /* nearest fp16 to 1/3 */
    };
    int n = (int)(sizeof(cases) / sizeof(cases[0]));
    int pass = 0;
    for (int i = 0; i < n; i++) {
        uint16_t in[4] = { cases[i].h, 0, 0, 0 };
        __m128i v = _mm_loadl_epi64((const __m128i *) in);
        __m128 y = ggml_x86_cvtph_ps(v);
        float out[4];
        _mm_storeu_ps(out, y);

        float diff = fabsf(out[0] - cases[i].expected);
        int ok = (diff < 1e-5f) ||
                 (isinf(cases[i].expected) && isinf(out[0]) &&
                  (cases[i].expected < 0) == (out[0] < 0));
        printf("  %-20s  in=0x%04x  expected=%g  got=%g  %s\n",
               cases[i].name, cases[i].h, cases[i].expected, out[0],
               ok ? "PASS" : "FAIL");
        if (ok) pass++;
    }
    printf("spot checks: %d/%d pass\n", pass, n);
}

static void test_q4_0_unpack(void) {
    /* Pack 32 signed values in [-8, 7] as nibbles, unpack, compare. */
    int8_t src[32];
    for (int i = 0; i < 32; i++) src[i] = (int8_t)((i % 16) - 8);

    uint8_t qs[16];
    for (int i = 0; i < 16; i++) {
        uint8_t lo = (uint8_t)(src[i] + 8);          /* low nibble */
        uint8_t hi = (uint8_t)(src[i + 16] + 8);     /* high nibble */
        qs[i] = (uint8_t)(lo | (hi << 4));
    }

    float out[32];
    ggml_x86_q4_0_unpack_32(qs, 1.0f, out);

    int fail = 0;
    for (int i = 0; i < 16; i++) {
        if (out[i] != (float)src[i]) fail++;
    }
    for (int i = 0; i < 16; i++) {
        if (out[i + 16] != (float)src[i + 16]) fail++;
    }
    printf("q4_0 unpack: %s (%d mismatches of 32)\n",
           fail == 0 ? "PASS" : "FAIL", fail);
}

int main(void) {
    int rc = 0;
    rc += test_full_range();
    test_spot_checks();
    test_q4_0_unpack();
    return rc == 0 ? 0 : 1;
}
