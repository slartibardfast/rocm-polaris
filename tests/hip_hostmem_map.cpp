// hip_hostmem_map.cpp — Test GPU access to hipHostMalloc'd host memory
//
// Reproduces the llama.cpp VM fault: GPU kernel reads from host-mapped
// memory at various offsets and sizes.  Identifies the mapping boundary
// where GPU page table entries go missing.
//
// Build: hipcc --offload-arch=gfx803 -o hip_hostmem_map hip_hostmem_map.cpp
// Run:   ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 120 ./hip_hostmem_map

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                e, hipGetErrorString(e), __FILE__, __LINE__); \
        return false; \
    } \
} while (0)

// GPU kernel: read one byte from host memory at given offset
__global__ void read_byte(const char *base, char *out, long offset) {
    *out = base[offset];
}

// GPU kernel: read N ints from host memory, sum them
__global__ void sum_region(const int *base, long start, long count, long *out) {
    long s = 0;
    for (long i = start + threadIdx.x; i < start + count; i += blockDim.x) {
        s += base[i];
    }
    atomicAdd((unsigned long long *)out, (unsigned long long)s);
}

// Test 1: Probe GPU access at exponentially-spaced offsets within a single
// large hipHostMalloc allocation.  Finds the first unmapped offset.
static bool test_probe_offsets(size_t alloc_size) {
    printf("Test 1: Probe offsets in %zuMB hipHostMalloc...\n", alloc_size >> 20);

    char *h_buf;
    HIP_CHECK(hipHostMalloc(&h_buf, alloc_size, 0));
    memset(h_buf, 0x42, alloc_size);
    printf("  hipHostMalloc: %p (%zuMB)\n", h_buf, alloc_size >> 20);

    char *d_out;
    HIP_CHECK(hipMalloc(&d_out, 1));

    // Probe at page boundaries: 0, 4K, 8K, ..., then 1MB, 2MB, ..., then fine-grain near failures
    size_t offsets[] = {
        0, 4096, 8192, 16384, 65536,
        1UL<<20, 2UL<<20, 4UL<<20, 8UL<<20, 16UL<<20,
        32UL<<20, 48UL<<20, 64UL<<20, 96UL<<20, 128UL<<20,
        192UL<<20, 256UL<<20, 384UL<<20, 448UL<<20,
        alloc_size - 4096, alloc_size - 1
    };

    int pass = 0, fail = 0;
    size_t first_fail = alloc_size;
    for (size_t off : offsets) {
        if (off >= alloc_size) continue;
        read_byte<<<1, 1>>>(h_buf, d_out, (long)off);
        hipError_t e = hipDeviceSynchronize();
        char result = 0;
        if (e == hipSuccess) {
            hipMemcpy(&result, d_out, 1, hipMemcpyDeviceToHost);
        }
        bool ok = (e == hipSuccess && result == 0x42);
        printf("  offset %10zu (%6zuMB + %5zuKB): %s",
               off, off >> 20, (off & 0xFFFFF) >> 10,
               ok ? "PASS" : "FAIL");
        if (!ok && e != hipSuccess)
            printf(" (hip error %d: %s)", e, hipGetErrorString(e));
        else if (!ok)
            printf(" (got 0x%02x, expected 0x42)", (unsigned char)result);
        printf("\n");

        if (ok) pass++; else { fail++; if (off < first_fail) first_fail = off; }
    }

    // If we found a failure, do a binary search for the exact boundary
    if (first_fail < alloc_size && first_fail > 0) {
        printf("\n  Binary search for mapping boundary near %zuMB...\n", first_fail >> 20);
        size_t lo = 0, hi = first_fail;
        // Find the last passing offset below first_fail
        for (size_t off : offsets) {
            if (off < first_fail) {
                read_byte<<<1, 1>>>(h_buf, d_out, (long)off);
                if (hipDeviceSynchronize() == hipSuccess) lo = off;
            }
        }
        // Binary search between lo and hi
        for (int i = 0; i < 20 && (hi - lo) > 4096; i++) {
            size_t mid = ((lo + hi) / 2) & ~4095UL;  // page-align
            read_byte<<<1, 1>>>(h_buf, d_out, (long)mid);
            if (hipDeviceSynchronize() == hipSuccess) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        printf("  Mapping boundary: PASS at %zu (%.3fMB), FAIL at %zu (%.3fMB)\n",
               lo, lo / (1024.0 * 1024.0), hi, hi / (1024.0 * 1024.0));
    }

    hipFree(d_out);
    hipHostFree(h_buf);
    printf("  Result: %d pass, %d fail\n\n", pass, fail);
    return fail == 0;
}

// Test 2: Multiple small hipHostMalloc allocations, test each.
// llama.cpp does: 1x 475MB + 3x 1MB hipHostMalloc.
static bool test_multiple_allocs() {
    printf("Test 2: Multiple hipHostMalloc (mimics llama.cpp pattern)...\n");

    struct Alloc { size_t size; void *ptr; const char *desc; };
    Alloc allocs[] = {
        {475735040, nullptr, "model buffer (454MB)"},
        {1048576,   nullptr, "staging 1 (1MB)"},
        {1048576,   nullptr, "staging 2 (1MB)"},
        {1048576,   nullptr, "staging 3 (1MB)"},
    };
    int nallocs = sizeof(allocs) / sizeof(allocs[0]);

    // Allocate all
    for (int i = 0; i < nallocs; i++) {
        hipError_t e = hipHostMalloc(&allocs[i].ptr, allocs[i].size, 0);
        printf("  alloc %d: %s = %p (%s)\n", i, allocs[i].desc, allocs[i].ptr,
               e == hipSuccess ? "OK" : "FAIL");
        if (e != hipSuccess) {
            for (int j = 0; j < i; j++) hipHostFree(allocs[j].ptr);
            return false;
        }
        memset(allocs[i].ptr, 0x42 + i, allocs[i].size);
    }

    char *d_out;
    hipMalloc(&d_out, 1);

    // Test GPU access to each allocation at various offsets
    int pass = 0, fail = 0;
    for (int i = 0; i < nallocs; i++) {
        char *base = (char *)allocs[i].ptr;
        size_t sz = allocs[i].size;
        char expected = 0x42 + i;
        size_t test_offsets[] = {0, sz/4, sz/2, sz*3/4, sz - 1};
        for (size_t off : test_offsets) {
            if (off >= sz) continue;
            read_byte<<<1, 1>>>(base, d_out, (long)off);
            hipError_t e = hipDeviceSynchronize();
            char result = 0;
            if (e == hipSuccess) hipMemcpy(&result, d_out, 1, hipMemcpyDeviceToHost);
            bool ok = (e == hipSuccess && result == expected);
            if (!ok) {
                printf("  FAIL: alloc %d (%s) offset %zu: ", i, allocs[i].desc, off);
                if (e != hipSuccess) printf("hip error %d\n", e);
                else printf("got 0x%02x expected 0x%02x\n", (unsigned char)result, (unsigned char)expected);
                fail++;
            } else {
                pass++;
            }
        }
    }

    hipFree(d_out);
    for (int i = 0; i < nallocs; i++) hipHostFree(allocs[i].ptr);
    printf("  Result: %d pass, %d fail\n\n", pass, fail);
    return fail == 0;
}

// Test 3: hipMalloc (VRAM) + hipMemcpy H2D + GPU kernel read.
// This is the -ngl 99 path: everything in VRAM, no zero-copy.
static bool test_vram_path() {
    printf("Test 3: VRAM path (hipMalloc + H2D + kernel read)...\n");

    size_t sizes[] = {1UL<<20, 16UL<<20, 64UL<<20, 256UL<<20, 400UL<<20};
    char *d_out;
    hipMalloc(&d_out, 1);

    int pass = 0, fail = 0;
    for (size_t sz : sizes) {
        int *d_buf;
        hipError_t e1 = hipMalloc(&d_buf, sz);
        if (e1 != hipSuccess) {
            printf("  %3zuMB: hipMalloc FAIL (%d)\n", sz >> 20, e1);
            fail++;
            continue;
        }
        // Fill from host
        int *h_buf = (int *)malloc(sz);
        if (!h_buf) { printf("  %3zuMB: host malloc fail\n", sz >> 20); hipFree(d_buf); fail++; continue; }
        int n = sz / sizeof(int);
        for (int i = 0; i < n; i++) h_buf[i] = i & 0xFF;
        hipMemcpy(d_buf, h_buf, sz, hipMemcpyHostToDevice);

        // GPU reads from VRAM
        long *d_sum;
        hipMalloc(&d_sum, sizeof(long));
        hipMemset(d_sum, 0, sizeof(long));
        // Read first 1024 and last 1024 ints
        sum_region<<<1, 256>>>(d_buf, 0, 1024, d_sum);
        sum_region<<<1, 256>>>(d_buf, n - 1024, 1024, d_sum);
        hipError_t e2 = hipDeviceSynchronize();

        printf("  %3zuMB: alloc OK, kernel %s\n", sz >> 20,
               e2 == hipSuccess ? "PASS" : "FAIL");
        if (e2 == hipSuccess) pass++; else fail++;

        hipFree(d_sum);
        free(h_buf);
        hipFree(d_buf);
    }

    hipFree(d_out);
    printf("  Result: %d pass, %d fail\n\n", pass, fail);
    return fail == 0;
}

int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device: %s (%s), VRAM: %zuMB\n\n", prop.name, prop.gcnArchName,
           prop.totalGlobalMem >> 20);

    bool p1 = test_probe_offsets(475735040);  // 454MB, same as llama.cpp
    bool p2 = test_multiple_allocs();
    bool p3 = test_vram_path();

    printf("=== Summary ===\n");
    printf("  Host memory probe:    %s\n", p1 ? "PASS" : "FAIL");
    printf("  Multiple host allocs: %s\n", p2 ? "PASS" : "FAIL");
    printf("  VRAM path:            %s\n", p3 ? "PASS" : "FAIL");

    return (p1 && p2 && p3) ? 0 : 1;
}
