#include "vscale.cuh"
#include <cstddef>

__global__ void vscale(const float *a, float *b, unsigned int n) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    b[index] = a[index] * b[index];
  }
}
