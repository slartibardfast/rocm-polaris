/* Test CPU-visible VRAM allocation via KFD ioctl directly.
 * Verifies Phase 13 kernel patch allows small-BAR PUBLIC VRAM. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <linux/kfd_ioctl.h>

int main() {
    int fd = open("/dev/kfd", O_RDWR);
    if (fd < 0) { perror("open /dev/kfd"); return 1; }

    /* Get GPU ID from topology */
    uint32_t gpu_id = 0;
    char buf[256];
    for (int node = 0; node < 10; node++) {
        snprintf(buf, sizeof(buf),
                 "/sys/devices/virtual/kfd/kfd/topology/nodes/%d/gpu_id", node);
        FILE *f = fopen(buf, "r");
        if (!f) continue;
        uint32_t gid = 0;
        if (fscanf(f, "%u", &gid) == 1 && gid > 0) {
            gpu_id = gid;
            printf("GPU node %d: gpu_id=%u\n", node, gid);
        }
        fclose(f);
        if (gpu_id) break;
    }
    if (!gpu_id) { fprintf(stderr, "No GPU found\n"); return 1; }

    /* Allocate 4KB CPU-visible VRAM */
    struct kfd_ioctl_alloc_memory_of_gpu_args alloc = {0};
    alloc.gpu_id = gpu_id;
    alloc.size = 4096;
    alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                  KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC |
                  KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                  KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;

    int ret = ioctl(fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc);
    printf("ALLOC_MEMORY_OF_GPU(VRAM|PUBLIC, 4KB): ret=%d, handle=0x%llx, mmap_offset=0x%llx\n",
           ret, alloc.handle, alloc.mmap_offset);

    if (ret != 0) {
        perror("ioctl ALLOC_MEMORY");
        close(fd);
        return 1;
    }

    /* mmap for CPU access */
    void *ptr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, alloc.mmap_offset);
    printf("mmap: %s, ptr=%p\n", ptr != MAP_FAILED ? "OK" : "FAIL", ptr);

    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    /* Test CPU write + read */
    memset(ptr, 0xBB, 4096);
    __builtin_ia32_sfence();
    unsigned char *p = (unsigned char *)ptr;
    int ok = 1;
    for (int i = 0; i < 4096; i++) {
        if (p[i] != 0xBB) { ok = 0; printf("  MISMATCH at %d: got 0x%02x\n", i, p[i]); break; }
    }
    printf("CPU write+readback: %s\n", ok ? "PASS" : "FAIL");
    printf("VA: %p (handle=0x%llx)\n", ptr, alloc.handle);

    munmap(ptr, 4096);
    close(fd);
    printf("DONE\n");
    return 0;
}
