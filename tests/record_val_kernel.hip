// Standalone kernel for raw HSA test
#include <hip/hip_runtime.h>
extern "C" __global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}
