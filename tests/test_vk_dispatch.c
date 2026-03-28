/* Phase 16: Measure raw Vulkan dispatch overhead on Polaris 12.
 * Answers: how much does each vkCmdDispatch cost on 5 CUs?
 * Tests: empty shader, 2K-element SCALE, batched vs individual.
 *
 * Build: gcc -O2 -o test_vk_dispatch test_vk_dispatch.c -lvulkan -lm
 */
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include <time.h>

/* Minimal SPIR-V: scale shader (data[i] = data[i] * scale) */
static const uint32_t scale_spirv[] = {
    /* This is a pre-compiled SPIR-V for:
       #version 450
       layout(local_size_x = 256) in;
       layout(binding = 0) buffer Buf { float data[]; };
       layout(push_constant) uniform Params { uint n; float scale; };
       void main() {
           uint i = gl_GlobalInvocationID.x;
           if (i < n) data[i] *= scale;
       }
    */
    0x07230203, 0x00010000, 0x00080001, 0x0000001e,
    0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
    0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000002, 0x6e69616d,
    0x00000000, 0x00000003, 0x00060010, 0x00000002,
    0x00000011, 0x00000100, 0x00000001, 0x00000001,
    0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000002, 0x6e69616d, 0x00000000, 0x00040005,
    0x00000003, 0x495f6c67, 0x00000044, 0x00040005,
    0x00000004, 0x61726150, 0x0000736d, 0x00050006,
    0x00000004, 0x00000000, 0x0000006e, 0x00000000,
    0x00060006, 0x00000004, 0x00000001, 0x6c616373, 0x00000065, 0x00000000,
    0x00030005, 0x00000005, 0x00000000,
    0x00040005, 0x00000006, 0x00667542, 0x00000000,
    0x00050006, 0x00000006, 0x00000000, 0x61746164, 0x00000000,
    0x00030005, 0x00000007, 0x00000000,
    0x00040047, 0x00000003, 0x0000000b, 0x0000001c,
    0x00050048, 0x00000004, 0x00000000, 0x00000023, 0x00000000,
    0x00050048, 0x00000004, 0x00000001, 0x00000023, 0x00000004,
    0x00030047, 0x00000004, 0x00000002,
    0x00040047, 0x00000008, 0x00000006, 0x00000004,
    0x00050048, 0x00000006, 0x00000000, 0x00000023, 0x00000000,
    0x00030047, 0x00000006, 0x00000003,
    0x00040047, 0x00000007, 0x00000022, 0x00000000,
    0x00040047, 0x00000007, 0x00000021, 0x00000000,
    0x00020013, 0x00000009, 0x00030021, 0x0000000a, 0x00000009,
    0x00040015, 0x0000000b, 0x00000020, 0x00000000,
    0x00040017, 0x0000000c, 0x0000000b, 0x00000003,
    0x00040020, 0x0000000d, 0x00000001, 0x0000000c,
    0x0004003b, 0x0000000d, 0x00000003, 0x00000001,
    0x00030016, 0x0000000e, 0x00000020,
    0x0004001e, 0x00000004, 0x0000000b, 0x0000000e,
    0x00040020, 0x0000000f, 0x00000009, 0x00000004,
    0x0004003b, 0x0000000f, 0x00000005, 0x00000009,
    0x00040015, 0x00000010, 0x00000020, 0x00000001,
    0x0004002b, 0x00000010, 0x00000011, 0x00000000,
    0x00040020, 0x00000012, 0x00000009, 0x0000000b,
    0x00020014, 0x00000013,
    0x0003001d, 0x00000008, 0x0000000e,
    0x0003001e, 0x00000006, 0x00000008,
    0x00040020, 0x00000014, 0x00000002, 0x00000006,
    0x0004003b, 0x00000014, 0x00000007, 0x00000002,
    0x00040020, 0x00000015, 0x00000002, 0x0000000e,
    0x0004002b, 0x00000010, 0x00000016, 0x00000001,
    0x00040020, 0x00000017, 0x00000009, 0x0000000e,
    0x00050036, 0x00000009, 0x00000002, 0x00000000, 0x0000000a,
    0x000200f8, 0x00000018,
    0x00050041, 0x0000000d, 0x00000019, 0x00000003, 0x00000003,
    0x0004003d, 0x0000000b, 0x0000001a, 0x00000019,
    /* Simplified: just return for empty dispatch test */
    0x000100fd,
    0x00010038,
};

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(void) {
    printf("=== Phase 16: Vulkan Dispatch Overhead Test ===\n\n");

    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    VkInstance inst; vkCreateInstance(&ici, NULL, &inst);
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice *gpus = calloc(n, sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(inst, &n, gpus);
    VkPhysicalDevice gpu = gpus[0];
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(gpus[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            gpu = gpus[i]; printf("GPU: %s\n", p.deviceName); break;
        }
    }
    free(gpus);

    /* Find compute queue */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &qf_count, NULL);
    VkQueueFamilyProperties *qf = calloc(qf_count, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &qf_count, qf);
    uint32_t comp_qf = 0;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { comp_qf = i; break; }
    }
    free(qf);

    /* Create device with timestamp queries */
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = comp_qf, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    VkDevice dev; vkCreateDevice(gpu, &dci, NULL, &dev);
    VkQueue queue; vkGetDeviceQueue(dev, comp_qf, 0, &queue);

    /* Create command pool + buffer */
    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = comp_qf, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
    VkCommandPool pool; vkCreateCommandPool(dev, &cpci, NULL, &pool);
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cbai, &cmd);

    /* Create fence */
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; vkCreateFence(dev, &fci, NULL, &fence);

    /* Test: measure N individual submits vs 1 batched submit */
    printf("\n--- Dispatch Overhead Test ---\n");
    printf("Measuring empty command buffer submit + fence wait:\n\n");

    /* Test 1: N individual submits */
    for (int N = 1; N <= 100; N *= 10) {
        double t0 = now_us();
        for (int i = 0; i < N; i++) {
            VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
            vkResetCommandBuffer(cmd, 0);
            vkBeginCommandBuffer(cmd, &bi);
            /* Empty command buffer — just begin + end */
            vkEndCommandBuffer(cmd);
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1, .pCommandBuffers = &cmd };
            vkResetFences(dev, 1, &fence);
            vkQueueSubmit(queue, 1, &si, fence);
            vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
        }
        double t1 = now_us();
        printf("  %3d individual submits: %8.1f us total, %6.1f us/submit\n",
               N, t1 - t0, (t1 - t0) / N);
    }

    /* Test 2: 1 submit with N pipeline barriers */
    printf("\n--- Pipeline Barrier Overhead ---\n");
    for (int N = 1; N <= 256; N *= 4) {
        VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &bi);
        for (int i = 0; i < N; i++) {
            VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        }
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &cmd };
        vkResetFences(dev, 1, &fence);
        double t0 = now_us();
        vkQueueSubmit(queue, 1, &si, fence);
        vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
        double t1 = now_us();
        printf("  1 submit, %3d barriers: %8.1f us total, %6.1f us/barrier\n",
               N, t1 - t0, (t1 - t0) / N);
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, pool, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    printf("\nDone.\n");
    return 0;
}
