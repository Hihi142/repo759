#include <cstddef>

#ifndef HW07_CPU_TEST
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#endif

#include "matmul.cuh"

// The CPU harness supplies per-block storage and a real thread barrier, so it
// executes the same indexing, loads, arithmetic, and synchronization as CUDA.
#ifdef HW07_CPU_TEST
extern void* hw07_matmul_shared;
#endif

template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C,
                             unsigned int n) {
#ifdef HW07_CPU_TEST
    T* tile_a = static_cast<T*>(hw07_matmul_shared);
#else
    // Byte storage gives every template specialization the same shared-memory
    // declaration while retaining the alignment required by double.
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    T* tile_a = reinterpret_cast<T*>(shared_bytes);
#endif
    const unsigned int tile = blockDim.x;
    T* tile_b = tile_a + tile * tile;
    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const unsigned int row = blockIdx.y * tile + y;
    const unsigned int column = blockIdx.x * tile + x;
    const unsigned int tiles = (n + tile - 1) / tile;
    T sum = T(0);

    for (unsigned int t = 0; t < tiles; ++t) {
        const unsigned int a_column = t * tile + x;
        const unsigned int b_row = t * tile + y;
        tile_a[y * tile + x] = row < n && a_column < n
            ? A[static_cast<std::size_t>(row) * n + a_column] : T(0);
        tile_b[y * tile + x] = b_row < n && column < n
            ? B[static_cast<std::size_t>(b_row) * n + column] : T(0);
        __syncthreads();
        for (unsigned int k = 0; k < tile; ++k) {
            sum += tile_a[y * tile + k] * tile_b[k * tile + x];
        }
        // Edge threads must also wait before the next tile overwrites storage.
        __syncthreads();
    }
    if (row < n && column < n) {
        C[static_cast<std::size_t>(row) * n + column] = sum;
    }
}

#ifndef HW07_CPU_TEST
namespace {
void check_matmul(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

template <typename T>
void launch_matmul(const T* A, const T* B, T* C, unsigned int n,
                   unsigned int block_dim) {
    if (A == nullptr || B == nullptr || C == nullptr) {
        throw std::invalid_argument("matmul requires non-null A, B, and C");
    }
    if (n == 0 || n > (1u << 14)) {
        throw std::invalid_argument("n must be in [1, 16384]");
    }
    if (block_dim == 0 || block_dim > 32) {
        throw std::invalid_argument("block_dim must be in [1, 32]");
    }
    int device = 0;
    cudaDeviceProp properties{};
    cudaFuncAttributes attributes{};
    check_matmul(cudaGetDevice(&device), "cudaGetDevice");
    check_matmul(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties");
    check_matmul(cudaFuncGetAttributes(&attributes, matmul_kernel<T>),
                 "cudaFuncGetAttributes");
    const unsigned int threads = block_dim * block_dim;
    const unsigned int blocks = (n + block_dim - 1) / block_dim;
    const std::size_t shared_bytes = 2 * threads * sizeof(T);
    if (threads > static_cast<unsigned int>(properties.maxThreadsPerBlock) ||
        threads > static_cast<unsigned int>(attributes.maxThreadsPerBlock) ||
        block_dim > static_cast<unsigned int>(properties.maxThreadsDim[0]) ||
        block_dim > static_cast<unsigned int>(properties.maxThreadsDim[1]) ||
        blocks > static_cast<unsigned int>(properties.maxGridSize[0]) ||
        blocks > static_cast<unsigned int>(properties.maxGridSize[1]) ||
        shared_bytes + attributes.sharedSizeBytes > properties.sharedMemPerBlock ||
        shared_bytes > static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes)) {
        throw std::runtime_error("Requested matmul launch exceeds this device's limits");
    }
    const dim3 block(block_dim, block_dim);
    const dim3 grid(blocks, blocks);
    matmul_kernel<T><<<grid, block, shared_bytes>>>(A, B, C, n);
    check_matmul(cudaGetLastError(), "matmul kernel launch");
    check_matmul(cudaDeviceSynchronize(), "matmul synchronization");
}
}  // namespace

__host__ void matmul_1(const int* A, const int* B, int* C, unsigned int n,
                       unsigned int block_dim) {
    launch_matmul(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float* A, const float* B, float* C, unsigned int n,
                       unsigned int block_dim) {
    launch_matmul(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double* A, const double* B, double* C, unsigned int n,
                       unsigned int block_dim) {
    launch_matmul(A, B, C, n, block_dim);
}
#endif
