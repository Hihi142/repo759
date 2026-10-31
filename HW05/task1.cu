#include <cstdio>

// The local CPU harness checks only arithmetic and indexing. Normal builds use
// the CUDA runtime and execute this kernel on a GPU.
#ifndef HW05_CPU_TEST
#include <cuda_runtime.h>
#include <cstdlib>
#endif

__global__ void factorial_kernel() {
  const unsigned int a = threadIdx.x + 1;
  unsigned int factorial = 1;
  for (unsigned int factor = 2; factor <= a; ++factor) {
    factorial *= factor;
  }
  std::printf("%u!=%u\n", a, factorial);
}

#ifndef HW05_CPU_TEST
int main() {
  factorial_kernel<<<1, 8>>>();
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "factorial kernel launch: %s\n",
                 cudaGetErrorString(status));
    return EXIT_FAILURE;
  }
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "factorial kernel execution: %s\n",
                 cudaGetErrorString(status));
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
#endif
