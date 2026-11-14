#include <cstddef>

#ifndef HW06_CPU_TEST
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#endif

#include "stencil.cuh"

__global__ void stencil_kernel(const float *image, const float *mask,
                               float *output, unsigned int n, unsigned int R) {
#ifdef HW06_CPU_TEST
  // The CPU harness supplies an allocation of the same size as CUDA shared memory.
  extern float *hw06_dynamic_shared;
  float *shared = hw06_dynamic_shared;
#else
  extern __shared__ float shared[];
#endif
  const std::size_t thread = threadIdx.x;
  const std::size_t width = blockDim.x;
  const std::size_t mask_size = 2 * static_cast<std::size_t>(R) + 1;
  const std::size_t tile_size = width + 2 * static_cast<std::size_t>(R);
  float *shared_mask = shared;
  float *shared_image = shared_mask + mask_size;
  float *shared_output = shared_image + tile_size;

  const std::size_t block_start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x;
  for (std::size_t k = thread; k < mask_size; k += width) {
    shared_mask[k] = mask[k];
  }
  for (std::size_t k = thread; k < tile_size; k += width) {
    // Widen before subtracting R so the left halo cannot wrap around unsigned 0.
    const long long image_index = static_cast<long long>(block_start) +
                                  static_cast<long long>(k) -
                                  static_cast<long long>(R);
    shared_image[k] = image_index >= 0 && image_index < static_cast<long long>(n)
                          ? image[static_cast<std::size_t>(image_index)]
                          : 1.0f;
  }
  __syncthreads();

  const std::size_t index = block_start + thread;
  if (index < n) {
    float sum = 0.0f;
    for (std::size_t k = 0; k < mask_size; ++k) {
      sum += shared_image[thread + k] * shared_mask[k];
    }
    shared_output[thread] = sum;
  }
  // Tail-block threads also reach both barriers; all results first reside in shared memory.
  __syncthreads();
  if (index < n) {
    output[index] = shared_output[thread];
  }
}

#ifndef HW06_CPU_TEST
__host__ void stencil(const float *image, const float *mask, float *output,
                      unsigned int n, unsigned int R,
                      unsigned int threads_per_block) {
  if (n == 0) {
    return;
  }
  const std::size_t mask_size = 2 * static_cast<std::size_t>(R) + 1;
  if (threads_per_block < mask_size || threads_per_block > 1024) {
    throw std::invalid_argument(
        "threads_per_block must be at least 2 * R + 1 and at most 1024");
  }
  const unsigned int blocks = (n - 1) / threads_per_block + 1;
  if (blocks > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("stencil launch exceeds CUDA grid.x limit");
  }
  const std::size_t shared_bytes =
      (2 * static_cast<std::size_t>(threads_per_block) +
       4 * static_cast<std::size_t>(R) + 1) * sizeof(float);
  stencil_kernel<<<blocks, threads_per_block, shared_bytes>>>(image, mask, output,
                                                            n, R);
}
#endif
