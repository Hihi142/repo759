#include <cuda_runtime.h>

#include "reduce.cuh"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

bool positive(const char *text, unsigned int &value) {
    const char *end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end && value != 0;
}

class DeviceArray {
public:
    explicit DeviceArray(std::size_t bytes) {
        check(cudaMalloc(reinterpret_cast<void **>(&data_), bytes), "cudaMalloc");
    }
    ~DeviceArray() { if (data_ != nullptr) cudaFree(data_); }
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;
    float *get() const { return data_; }
    void close() {
        float *data = data_;
        data_ = nullptr;
        check(cudaFree(data), "cudaFree");
    }
private:
    float *data_ = nullptr;
};

class Event {
public:
    Event() { check(cudaEventCreate(&event), "cudaEventCreate"); }
    ~Event() { if (event != nullptr) cudaEventDestroy(event); }
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
    void close() {
        const cudaEvent_t old = event;
        event = nullptr;
        check(cudaEventDestroy(old), "cudaEventDestroy");
    }
    cudaEvent_t event = nullptr;
};

class CompensatedSum {
public:
    void add(long double value) {
        const long double corrected = value - correction_;
        const long double next = sum_ + corrected;
        correction_ = (next - sum_) - corrected;
        sum_ = next;
    }
    long double get() const { return sum_; }
private:
    long double sum_ = 0.0L;
    long double correction_ = 0.0L;
};

unsigned int reduction_depth(unsigned int count, unsigned int threads) {
    unsigned int levels = 0;
    for (unsigned int width = threads; width > 1; width = (width + 1) / 2) {
        ++levels;
    }
    unsigned int depth = 0;
    while (count > 1) {
        count = (count - 1) / (2 * threads) + 1;
        depth += 1 + levels;  // First add during load, then shared-memory tree.
    }
    return depth;
}

void verify_sum(float actual, long double expected, long double sum_abs,
                unsigned int count, unsigned int threads) {
    // Each input's path has at most depth rounded additions. The standard
    // forward-error bound is gamma_depth * sum(abs(input)), with u = 2^-24.
    // A factor of two covers bound evaluation; compensated long-double host
    // accumulation contributes the much smaller second term. Absolute error
    // is used because a valid sum can be close to zero through cancellation.
    const long double unit = std::numeric_limits<float>::epsilon() / 2.0L;
    const long double factor = reduction_depth(count, threads) * unit;
    const long double gamma = factor / (1.0L - factor);
    const long double tolerance =
        (2.0L * gamma + 16.0L * std::numeric_limits<long double>::epsilon()) * sum_abs;
    const long double error = std::abs(static_cast<long double>(actual) - expected);
    if (!std::isfinite(actual) || error > tolerance) {
        char detail[256];
        std::snprintf(detail, sizeof(detail),
                      "Reduction disagrees with CPU reference: GPU=%.9g, "
                      "CPU=%.12Lg, error=%.5Lg, tolerance=%.5Lg",
                      actual, expected, error, tolerance);
        throw std::runtime_error(detail);
    }
}
}  // namespace

int main(int argc, char **argv) {
    unsigned int count = 0;
    unsigned int threads = 0;
    if (argc != 3 || !positive(argv[1], count) || !positive(argv[2], threads) ||
        count > (1u << 30) || threads > 1024 ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        std::fprintf(stderr, "Usage: %s N threads_per_block "
                     "(positive integers; N <= 2^30; threads_per_block <= 1024)\n", argv[0]);
        return 1;
    }

    try {
        const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
        const unsigned int blocks = (count - 1) / (2 * threads) + 1;
        const std::size_t output_bytes = static_cast<std::size_t>(blocks) * sizeof(float);
        const std::size_t shared_bytes = static_cast<std::size_t>(threads) * sizeof(float);
        int device = 0;
        cudaDeviceProp properties{};
        check(cudaGetDevice(&device), "cudaGetDevice");
        check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        cudaFuncAttributes attributes{};
        check(cudaFuncGetAttributes(&attributes, reduce_kernel), "cudaFuncGetAttributes");
        if (threads > static_cast<unsigned int>(properties.maxThreadsPerBlock) ||
            threads > static_cast<unsigned int>(properties.maxThreadsDim[0]) ||
            threads > static_cast<unsigned int>(attributes.maxThreadsPerBlock) ||
            blocks > static_cast<unsigned int>(properties.maxGridSize[0]) ||
            shared_bytes + attributes.sharedSizeBytes > properties.sharedMemPerBlock) {
            throw std::runtime_error("Requested reduction launch exceeds this device's limits");
        }

        std::vector<float> values(count);
        std::mt19937 generator(759);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        CompensatedSum reference, magnitude;
        for (float &value : values) {
            value = distribution(generator);
            reference.add(static_cast<long double>(value));
            magnitude.add(std::abs(static_cast<long double>(value)));
        }
        DeviceArray device_input(bytes), device_output(output_bytes);
        float *input = device_input.get();
        float *output = device_output.get();
        check(cudaMemcpy(input, values.data(), bytes, cudaMemcpyHostToDevice), "copy input to device");
        Event start, stop;

        reduce(&input, &output, count, threads);
        // reduce overwrites its input with partial/final sums. Restore ALL
        // original elements before the timed call, outside the event interval.
        input = device_input.get();
        output = device_output.get();
        check(cudaMemcpy(input, values.data(), bytes, cudaMemcpyHostToDevice), "restore input after warm-up");
        check(cudaEventRecord(start.event), "record start event");
        reduce(&input, &output, count, threads);
        check(cudaEventRecord(stop.event), "record stop event");
        check(cudaEventSynchronize(stop.event), "synchronize stop event");
        float milliseconds = 0.0f;
        check(cudaEventElapsedTime(&milliseconds, start.event, stop.event), "read elapsed time");
        float result = 0.0f;
        check(cudaMemcpy(&result, input, sizeof(float), cudaMemcpyDeviceToHost), "copy sum to host");
        verify_sum(result, reference.get(), magnitude.get(), count, threads);
        if (!std::isfinite(milliseconds) || milliseconds < 0.0f) {
            throw std::runtime_error("Invalid CUDA event time");
        }
        start.close();
        stop.close();
        device_input.close();
        device_output.close();
        // Required format: final sum, followed by CUDA-event milliseconds.
        std::printf("%.9g\n%.6f\n", result, milliseconds);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "task2: %s\n", error.what());
        return 1;
    }
    return 0;
}
