/* Phase 16: Vulkan VRAM placement test harness.
 * Allocates from each memory type and checks physical VRAM via sysfs.
 * Answers: does RADV actually put DEVICE_LOCAL in physical VRAM?
 *
 * Build: gcc -O2 -o test_vk_vram test_vk_vram.c -lvulkan
 * Run:   ./test_vk_vram
 */
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define ALLOC_SIZE (128 * 1024 * 1024)  /* 128 MB per test */

static size_t read_vram_used(void) {
    FILE *f = fopen("/sys/class/drm/card1/device/mem_info_vram_used", "r");
    if (!f) f = fopen("/sys/class/drm/card0/device/mem_info_vram_used", "r");
    if (!f) return 0;
    size_t val = 0;
    fscanf(f, "%zu", &val);
    fclose(f);
    return val;
}

static size_t read_gtt_used(void) {
    FILE *f = fopen("/sys/class/drm/card1/device/mem_info_gtt_used", "r");
    if (!f) f = fopen("/sys/class/drm/card0/device/mem_info_gtt_used", "r");
    if (!f) return 0;
    size_t val = 0;
    fscanf(f, "%zu", &val);
    fclose(f);
    return val;
}

static const char *prop_flags_str(VkMemoryPropertyFlags f) {
    static char buf[256];
    buf[0] = 0;
    if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) strcat(buf, "DEVICE_LOCAL ");
    if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) strcat(buf, "HOST_VISIBLE ");
    if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) strcat(buf, "HOST_COHERENT ");
    if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) strcat(buf, "HOST_CACHED ");
    if (buf[0] == 0) strcpy(buf, "(none)");
    return buf;
}

int main(void) {
    VkResult r;

    /* Create instance */
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    VkInstance instance;
    r = vkCreateInstance(&ici, NULL, &instance);
    if (r != VK_SUCCESS) { fprintf(stderr, "vkCreateInstance failed: %d\n", r); return 1; }

    /* Find physical device */
    uint32_t gpu_count = 0;
    vkEnumeratePhysicalDevices(instance, &gpu_count, NULL);
    if (gpu_count == 0) { fprintf(stderr, "No GPUs\n"); return 1; }
    VkPhysicalDevice *gpus = calloc(gpu_count, sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &gpu_count, gpus);

    /* Pick first discrete GPU */
    VkPhysicalDevice gpu = gpus[0];
    for (uint32_t i = 0; i < gpu_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpus[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            gpu = gpus[i];
            printf("GPU: %s\n", props.deviceName);
            break;
        }
    }
    free(gpus);

    /* Get memory properties */
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(gpu, &mem_props);

    printf("\n=== Memory Heaps ===\n");
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        printf("  Heap %u: %zu MB%s\n", i,
               (size_t)(mem_props.memoryHeaps[i].size / (1024*1024)),
               (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " [DEVICE_LOCAL]" : "");
    }

    printf("\n=== Memory Types ===\n");
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        printf("  Type %u: heap %u | %s\n", i,
               mem_props.memoryTypes[i].heapIndex,
               prop_flags_str(mem_props.memoryTypes[i].propertyFlags));
    }

    /* Find a queue family */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &qf_count, NULL);
    VkQueueFamilyProperties *qf_props = calloc(qf_count, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &qf_count, qf_props);
    uint32_t compute_qf = 0;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { compute_qf = i; break; }
    }
    free(qf_props);

    /* Create device */
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = compute_qf,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };
    VkDevice device;
    r = vkCreateDevice(gpu, &dci, NULL, &device);
    if (r != VK_SUCCESS) { fprintf(stderr, "vkCreateDevice failed: %d\n", r); return 1; }

    /* Test each memory type */
    printf("\n=== Allocation Test (%d MB each) ===\n", ALLOC_SIZE / (1024*1024));
    printf("%-6s %-6s %-30s %12s %12s %s\n",
           "Type", "Heap", "Flags", "VRAM delta", "GTT delta", "Placement");
    printf("----------------------------------------------------------------------\n");

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        /* Create a buffer to get memory requirements */
        VkBufferCreateInfo bci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = ALLOC_SIZE,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VkBuffer buffer;
        r = vkCreateBuffer(device, &bci, NULL, &buffer);
        if (r != VK_SUCCESS) {
            printf("%-6u %-6u %-30s %12s %12s SKIP (buffer create failed)\n",
                   i, mem_props.memoryTypes[i].heapIndex,
                   prop_flags_str(mem_props.memoryTypes[i].propertyFlags), "-", "-");
            continue;
        }

        VkMemoryRequirements mem_req;
        vkGetBufferMemoryRequirements(device, buffer, &mem_req);

        /* Check if this type is compatible */
        if (!(mem_req.memoryTypeBits & (1u << i))) {
            vkDestroyBuffer(device, buffer, NULL);
            printf("%-6u %-6u %-30s %12s %12s SKIP (incompatible)\n",
                   i, mem_props.memoryTypes[i].heapIndex,
                   prop_flags_str(mem_props.memoryTypes[i].propertyFlags), "-", "-");
            continue;
        }

        /* Check heap has enough space */
        if (mem_props.memoryHeaps[mem_props.memoryTypes[i].heapIndex].size < ALLOC_SIZE) {
            vkDestroyBuffer(device, buffer, NULL);
            printf("%-6u %-6u %-30s %12s %12s SKIP (heap too small)\n",
                   i, mem_props.memoryTypes[i].heapIndex,
                   prop_flags_str(mem_props.memoryTypes[i].propertyFlags), "-", "-");
            continue;
        }

        /* Measure VRAM before */
        size_t vram_before = read_vram_used();
        size_t gtt_before = read_gtt_used();

        /* Allocate */
        VkMemoryAllocateInfo mai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = ALLOC_SIZE,
            .memoryTypeIndex = i,
        };
        VkDeviceMemory mem;
        r = vkAllocateMemory(device, &mai, NULL, &mem);
        if (r != VK_SUCCESS) {
            vkDestroyBuffer(device, buffer, NULL);
            printf("%-6u %-6u %-30s %12s %12s FAIL (alloc: %d)\n",
                   i, mem_props.memoryTypes[i].heapIndex,
                   prop_flags_str(mem_props.memoryTypes[i].propertyFlags), "-", "-", r);
            continue;
        }

        /* Bind buffer to memory */
        vkBindBufferMemory(device, buffer, mem, 0);

        /* Measure VRAM after */
        size_t vram_after = read_vram_used();
        size_t gtt_after = read_gtt_used();
        long vram_delta = (long)(vram_after - vram_before) / (1024*1024);
        long gtt_delta = (long)(gtt_after - gtt_before) / (1024*1024);

        const char *placement;
        if (vram_delta > 100) placement = "*** PHYSICAL VRAM ***";
        else if (gtt_delta > 100) placement = "GTT (system RAM)";
        else placement = "unknown";

        printf("%-6u %-6u %-30s %10ld MB %10ld MB %s\n",
               i, mem_props.memoryTypes[i].heapIndex,
               prop_flags_str(mem_props.memoryTypes[i].propertyFlags),
               vram_delta, gtt_delta, placement);

        /* Cleanup */
        vkFreeMemory(device, mem, NULL);
        vkDestroyBuffer(device, buffer, NULL);

        /* Small delay for sysfs to settle */
        usleep(100000);
    }

    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    printf("\nDone.\n");
    return 0;
}
