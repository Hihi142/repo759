#include <cstddef>

#ifndef HW07_CPU_TEST
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#endif

#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
#ifdef HW07_CPU_TEST
    // The host-only harness runs this same kernel body with a block barrier.
    extern float *hw07_reduce_shared;
    float *shared = hw07_reduce_shared;
#else
    extern __shared__ float shared[];
#endif
    const unsigned int thread = threadIdx.x;
    const unsigned int width = blockDim.x;
    const std::size_t index =
        2 * static_cast<std::size_t>(blockIdx.x) * width + thread;

    // Reduction #4: add two global-memory values before storing in shared memory.
    float value = index < n ? g_idata[index] : 0.0f;
    if (index + width < n) {
        value += g_idata[index + width];
    }
    shared[thread] = value;
    __syncthreads();

    // For odd active counts, the unpaired middle element stays in the lower
    // half. This also supports block sizes that are not powers of two.
    for (unsigned int active = width; active > 1;) {
        const unsigned int next = (active + 1) / 2;
        if (thread < active / 2) {
            shared[thread] += shared[thread + next];
        }
        __syncthreads();
        active = next;
    }
    if (thread == 0) {
        g_odata[blockIdx.x] = shared[0];
    }
}

#ifndef HW07_CPU_TEST
namespace {
void check_reduce(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}
}  // namespace

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block) {
    if (N == 0 || threads_per_block == 0 || threads_per_block > 1024) {
        throw std::invalid_argument(
            "reduce requires N > 0 and 1 <= threads_per_block <= 1024");
    }
    if (input == nullptr || output == nullptr || *input == nullptr ||
        *output == nullptr || *input == *output) {
        throw std::invalid_argument("reduce requires two distinct device buffers");
    }

    // Alternate between the two existing allocations. The output allocation
    // holds the first pass's partial sums, and every later pass is smaller.
    // A pass always reads and writes different buffers, avoiding inter-block
    // races. Preserve the caller's allocation pointers for safe cleanup/reuse.
    float *source = *input;
    float *destination = *output;
    unsigned int remaining = N;
    const unsigned int elements_per_block = 2 * threads_per_block;
    while (remaining > 1) {
        const unsigned int blocks = (remaining - 1) / elements_per_block + 1;
        if (blocks > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("reduce exceeds CUDA's grid.x limit");
        }
        reduce_kernel<<<blocks, threads_per_block,
                        static_cast<std::size_t>(threads_per_block) * sizeof(float)>>>(
            source, destination, remaining);
        check_reduce(cudaGetLastError(), "reduce_kernel launch");
        float *old_source = source;
        source = destination;
        destination = old_source;
        remaining = blocks;
    }
    if (source != *input) {
        check_reduce(cudaMemcpy(*input, source, sizeof(float), cudaMemcpyDeviceToDevice),
                     "copy final sum into input[0]");
    }
    // This also establishes the required completion guarantee for N == 1.
    check_reduce(cudaDeviceSynchronize(), "reduce synchronization");
}
#endif
