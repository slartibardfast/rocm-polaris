#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
int main() {
    for (size_t mb : {1, 2, 4, 8, 9, 10, 12, 14, 16, 20, 24, 32}) {
        size_t sz = mb << 20;
        int n = sz / sizeof(int);
        int *h = (int*)malloc(sz), *d, *h2 = (int*)malloc(sz);
        for (int i = 0; i < n; i++) h[i] = 0xDEAD;
        hipMalloc(&d, sz);
        hipMemcpy(d, h, sz, hipMemcpyHostToDevice);
        hipMemcpy(h2, d, sz, hipMemcpyDeviceToHost);
        int bad = 0, first = -1;
        for (int i = 0; i < n; i++) {
            if (h2[i] != 0xDEAD) { bad++; if (first<0) first=i; }
        }
        printf("%2zuMB: %d/%d bad, first_bad_offset=%dKB (%s)\n",
               mb, bad, n, first>=0?first*4/1024:-1, bad==0?"PASS":"FAIL");
        free(h); free(h2); hipFree(d);
        if (bad > 0) break;
    }
    return 0;
}
