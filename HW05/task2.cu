// HW05_CPU_TEST enables arithmetic/index checks without claiming to emulate
// CUDA execution. Ordinary nvcc builds always include the real host program.
#ifndef HW05_CPU_TEST
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t status = (call);                                          \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status));       \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (false)
#endif

__global__ void affine_kernel(int *dA, int a) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  dA[index] = a * static_cast<int>(threadIdx.x) + static_cast<int>(blockIdx.x);
}

#ifndef HW05_CPU_TEST
int main() {
  // A fixed seed makes results reproducible while still generating a randomly.
  std::mt19937 generator(759);
  std::uniform_int_distribution<int> distribution(1, 100);
  const int a = distribution(generator);

  int *dA = nullptr;
  int hA[16] = {};
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dA), sizeof(hA)));
  affine_kernel<<<2, 8>>>(dA, a);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(hA, dA, sizeof(hA), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dA));

  for (unsigned int i = 0; i < 16; ++i) {
    std::printf("%s%d", i == 0 ? "" : " ", hA[i]);
  }
  std::printf("\n");
  return EXIT_SUCCESS;
}
#endif
