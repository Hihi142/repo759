// The CPU test build omits the host CUDA driver; it cannot validate GPU runtime
// behavior or measure GPU performance.
#ifndef HW05_CPU_TEST
#include <cuda_runtime.h>

#include "vscale.cuh"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <random>
#include <vector>

#ifndef HW05_THREADS_PER_BLOCK
#define HW05_THREADS_PER_BLOCK 512
#endif

static_assert(HW05_THREADS_PER_BLOCK == 512 || HW05_THREADS_PER_BLOCK == 16,
              "HW05 uses 512 or 16 threads per block");

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t status = (call);                                          \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status));       \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (false)

namespace {
bool parse_size(const char *text, unsigned int &n) {
  if (*text == '\0') {
    return false;
  }
  for (const char *p = text; *p != '\0'; ++p) {
    if (*p < '0' || *p > '9') {
      return false;
    }
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno == ERANGE || *end != '\0' || value == 0 ||
      value > std::numeric_limits<unsigned int>::max() ||
      value > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    return false;
  }
  n = static_cast<unsigned int>(value);
  return true;
}
} // namespace

int main(int argc, char **argv) {
  unsigned int n = 0;
  if (argc != 2 || !parse_size(argv[1], n)) {
    std::fprintf(stderr, "Usage: %s n (positive unsigned integer)\n", argv[0]);
    return EXIT_FAILURE;
  }

  try {
    std::vector<float> a(n);
    std::vector<float> b(n);
    std::mt19937 generator(759);
    std::uniform_real_distribution<float> a_distribution(-10.0f, 10.0f);
    std::uniform_real_distribution<float> b_distribution(0.0f, 1.0f);
    for (unsigned int i = 0; i < n; ++i) {
      a[i] = a_distribution(generator);
      b[i] = b_distribution(generator);
    }

    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    constexpr unsigned int threads = HW05_THREADS_PER_BLOCK;
    // This avoids overflowing n + threads - 1 when n is near UINT_MAX.
    const unsigned int blocks = n / threads + (n % threads != 0);
    float *d_a = nullptr;
    float *d_b = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up in this process, then restore b because vscale overwrites it.
    // Allocation, input copies, warm-up and restoration are outside timing.
    vscale<<<blocks, threads>>>(d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    vscale<<<blocks, threads>>>(d_a, d_b, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Only endpoints are needed for the assignment output. Full-array GPU
    // correctness is tested separately before collecting performance results.
    float first = 0.0f;
    float last = 0.0f;
    CUDA_CHECK(cudaMemcpy(&first, d_b, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last, d_b + (n - 1), sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    const float expected_first = a.front() * b.front();
    const float expected_last = a.back() * b.back();
    const auto matches = [](float actual, float expected) {
      return std::isfinite(actual) &&
             std::fabs(actual - expected) <=
                 1.0e-6f + 1.0e-5f * std::fabs(expected);
    };
    if (!matches(first, expected_first) || !matches(last, expected_last)) {
      std::fprintf(stderr,
                   "task3 endpoint verification failed: first %.9g "
                   "(expected %.9g), last %.9g (expected %.9g)\n",
                   first, expected_first, last, expected_last);
      return EXIT_FAILURE;
    }

    std::printf("%.6f\n%.9g\n%.9g\n", milliseconds, first, last);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "task3: %s\n", error.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
#endif
