#include <cstddef>

#ifndef HW06_CPU_TEST
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#endif

#include "matmul.cuh"

__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              std::size_t n) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= n * n) {
    return;
  }

  const std::size_t row = index / n;
  const std::size_t column = index % n;
  float sum = 0.0f;
  for (std::size_t k = 0; k < n; ++k) {
    sum += A[row * n + k] * B[k * n + column];
  }
  C[index] = sum;
}

#ifndef HW06_CPU_TEST
void matmul(const float *A, const float *B, float *C, std::size_t n,
            unsigned int threads_per_block) {
  if (n == 0) {
    return;
  }
  if (threads_per_block == 0 || threads_per_block > 1024) {
    throw std::invalid_argument("threads_per_block must be in [1, 1024]");
  }
  if (n > std::numeric_limits<std::size_t>::max() / n) {
    throw std::overflow_error("matrix element count overflows size_t");
  }
  const std::size_t blocks = (n * n - 1) / threads_per_block + 1;
  if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("matrix launch exceeds CUDA grid.x limit");
  }
  matmul_kernel<<<static_cast<unsigned int>(blocks), threads_per_block>>>(A, B,
                                                                       C, n);
}
#endif
