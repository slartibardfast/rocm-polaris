#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(e) do { hipError_t _e = (e); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %d at %s:%d\n", _e, __FILE__, __LINE__); return 1; } } while(0)

int main() {
    for (size_t mb : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}) {
        size_t sz = mb << 20;
        int n = sz / sizeof(int);
        int *h = (int*)malloc(sz), *d = nullptr, *h2 = (int*)malloc(sz);
        if (!h || !h2) { printf("%3zuMB: SKIP (host OOM)\n", mb); break; }
        for (int i = 0; i < n; i++) h[i] = (int)(0xDEAD0000 | (i & 0xFFFF));
        hipError_t me = hipMalloc(&d, sz);
        if (me != hipSuccess) { printf("%3zuMB: SKIP (hipMalloc failed)\n", mb); free(h); free(h2); break; }
        CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(h2, d, sz, hipMemcpyDeviceToHost));
        int bad = 0, first = -1;
        for (int i = 0; i < n; i++) {
            if (h2[i] != h[i]) { bad++; if (first<0) first=i; }
        }
        printf("%3zuMB: %d/%d bad, first_bad_offset=%dKB (%s)\n",
               mb, bad, n, first>=0?first*4/1024:-1, bad==0?"PASS":"FAIL");
        hipFree(d); free(h); free(h2);
    }
    return 0;
}
