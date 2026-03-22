// test_kernarg_inspect.cpp — Check if kernel receives correct pointer AND value
//
// The kernel inspects BOTH its arguments:
//   arg0: out (pointer) — writes the pointer value itself to out[1]
//   arg1: val (int)     — writes val to out[0]
//
// If the kernarg_address in the AQL packet is wrong, the pointer arg
// will be corrupted too. If only the val arg is corrupted (low byte),
// the issue is in how that specific field is loaded from kernarg.
//
// Build: hipcc --offload-arch=gfx803 -o test_kernarg_inspect test_kernarg_inspect.cpp
// Run:   ./test_kernarg_inspect [iters]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

// Kernel reports its own argument values back
__global__ void inspect_args(int *out, int val) {
    if (threadIdx.x == 0) {
        out[0] = val;                           // the scalar arg
        out[1] = (int)(uintptr_t)out;           // low 32 bits of pointer arg
        out[2] = (int)((uintptr_t)out >> 32);   // high 32 bits of pointer arg
    }
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 10000;

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    int *d_result;
    hipMalloc(&d_result, 3 * sizeof(int));

    // Get the expected pointer value
    uintptr_t expected_ptr = (uintptr_t)d_result;
    printf("Expected pointer: 0x%lx\n\n", expected_ptr);

    int val_fails = 0, ptr_fails = 0;

    printf("%-8s %-12s %-12s %-12s %-20s %-20s\n",
           "iter", "exp_val", "got_val", "val&0xFF?",
           "exp_ptr_lo", "got_ptr_lo");

    for (int i = 0; i < iters; i++) {
        int expected_val = i * 7 + 13;
        hipMemset(d_result, 0, 3 * sizeof(int));
        inspect_args<<<1, 1>>>(d_result, expected_val);
        hipDeviceSynchronize();

        int h[3];
        hipMemcpy(h, d_result, 3 * sizeof(int), hipMemcpyDeviceToHost);

        int got_val = h[0];
        uintptr_t got_ptr = ((uintptr_t)(unsigned)h[2] << 32) | (unsigned)h[1];

        bool val_bad = (got_val != expected_val);
        bool ptr_bad = (got_ptr != expected_ptr);

        if (val_bad || ptr_bad) {
            bool is_low_byte = (got_val == (expected_val & 0xFF));
            printf("%-8d %-12d %-12d %-12s 0x%-18lx 0x%-18lx%s\n",
                   i, expected_val, got_val,
                   is_low_byte ? "YES" : "no",
                   expected_ptr, got_ptr,
                   ptr_bad ? " PTR_CORRUPT" : "");
            if (val_bad) val_fails++;
            if (ptr_bad) ptr_fails++;
            if (val_fails + ptr_fails > 50) break;
        }
    }

    hipFree(d_result);
    printf("\nval_fails: %d, ptr_fails: %d / %d\n", val_fails, ptr_fails, iters);
    return (val_fails + ptr_fails) > 0 ? 1 : 0;
}
