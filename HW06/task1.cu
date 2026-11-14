#include <cuda_runtime.h>
#include <cstddef>

#include "matmul.cuh"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

bool positive(const char* text, unsigned int& value) {
    const char* end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end && value != 0;
}

class DeviceArray {
public:
    explicit DeviceArray(std::size_t bytes) {
        check(cudaMalloc(reinterpret_cast<void**>(&data_), bytes), "cudaMalloc");
    }
    ~DeviceArray() { if (data_ != nullptr) cudaFree(data_); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    float* get() const { return data_; }
    void close() {
        float* data = data_;
        data_ = nullptr;
        check(cudaFree(data), "cudaFree");
    }
private:
    float* data_ = nullptr;
};

class Event {
public:
    Event() { check(cudaEventCreate(&event), "cudaEventCreate"); }
    ~Event() { if (event != nullptr) cudaEventDestroy(event); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    void close() {
        const cudaEvent_t old = event;
        event = nullptr;
        check(cudaEventDestroy(old), "cudaEventDestroy");
    }
    cudaEvent_t event = nullptr;
};

void verify_last(float actual, const std::vector<float>& a,
                 const std::vector<float>& b, std::size_t n) {
    double expected = 0.0;
    double magnitude = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double product = static_cast<double>(a[(n - 1) * n + k]) * b[k * n + n - 1];
        expected += product;
        magnitude += std::abs(product);
    }
    const double roundoff = std::numeric_limits<float>::epsilon() / 2.0;
    const double factor = (2.0 * n + 1.0) * roundoff;
    const double tolerance = 1e-6 + factor / (1.0 - factor) * magnitude;
    if (!std::isfinite(actual) || std::abs(static_cast<double>(actual) - expected) > tolerance) {
        throw std::runtime_error("Last matrix element disagrees with the CPU reference");
    }
}
}  // namespace

int main(int argc, char** argv) {
    unsigned int input_n = 0;
    unsigned int threads = 0;
    if (argc != 3 || !positive(argv[1], input_n) || input_n > (1u << 14) ||
        !positive(argv[2], threads) || threads > 1024 || threads % 32 != 0) {
        std::fprintf(stderr, "Usage: %s n threads_per_block (1 <= n <= 16384; "
                     "threads is a multiple of 32 in [32, 1024])\n", argv[0]);
        return 1;
    }

    try {
        const std::size_t n = input_n;
        const std::size_t count = n * n;
        const std::size_t bytes = count * sizeof(float);
        int device = 0;
        cudaDeviceProp properties{};
        check(cudaGetDevice(&device), "cudaGetDevice");
        check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        cudaFuncAttributes attributes{};
        check(cudaFuncGetAttributes(&attributes, matmul_kernel), "cudaFuncGetAttributes");
        const std::size_t blocks = count / threads + (count % threads != 0);
        if (threads > static_cast<unsigned int>(properties.maxThreadsPerBlock) ||
            threads > static_cast<unsigned int>(properties.maxThreadsDim[0]) ||
            threads > static_cast<unsigned int>(attributes.maxThreadsPerBlock) ||
            blocks > static_cast<std::size_t>(properties.maxGridSize[0])) {
            throw std::runtime_error("Requested matrix launch exceeds this device's limits");
        }

        std::vector<float> a(count), b(count), c(count);
        std::mt19937 generator(759);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (std::size_t i = 0; i < count; ++i) {
            a[i] = distribution(generator);
            b[i] = distribution(generator);
        }
        DeviceArray device_a(bytes), device_b(bytes), device_c(bytes);
        check(cudaMemcpy(device_a.get(), a.data(), bytes, cudaMemcpyHostToDevice), "copy A to device");
        check(cudaMemcpy(device_b.get(), b.data(), bytes, cudaMemcpyHostToDevice), "copy B to device");
        Event start, stop;

        // Warm-up and allocation/transfers are outside the measured function call.
        matmul(device_a.get(), device_b.get(), device_c.get(), n, threads);
        check(cudaGetLastError(), "matmul warm-up launch");
        check(cudaDeviceSynchronize(), "matmul warm-up synchronization");
        check(cudaEventRecord(start.event), "record start event");
        matmul(device_a.get(), device_b.get(), device_c.get(), n, threads);
        check(cudaEventRecord(stop.event), "record stop event");
        check(cudaGetLastError(), "matmul launch");
        check(cudaEventSynchronize(stop.event), "synchronize stop event");
        float milliseconds = 0.0f;
        check(cudaEventElapsedTime(&milliseconds, start.event, stop.event), "read elapsed time");
        check(cudaMemcpy(c.data(), device_c.get(), bytes, cudaMemcpyDeviceToHost), "copy C to host");
        verify_last(c.back(), a, b, n);
        if (!std::isfinite(milliseconds) || milliseconds < 0.0f) {
            throw std::runtime_error("Invalid CUDA event time");
        }
        start.close();
        stop.close();
        device_a.close();
        device_b.close();
        device_c.close();
        // Required format: last output element, then CUDA-event milliseconds.
        std::printf("%.9g\n%.6f\n", c.back(), milliseconds);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "task1: %s\n", error.what());
        return 1;
    }
    return 0;
}
