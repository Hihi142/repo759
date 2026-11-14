#include <cuda_runtime.h>
#include <cstddef>

#include "stencil.cuh"

#include <charconv>
#include <cmath>
#include <cstdint>
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

void verify_last(float actual, const std::vector<float>& image,
                 const std::vector<float>& mask, unsigned int radius) {
    double expected = 0.0;
    double magnitude = 0.0;
    for (std::size_t k = 0; k < mask.size(); ++k) {
        const std::int64_t index = static_cast<std::int64_t>(image.size()) - 1 +
                                   static_cast<std::int64_t>(k) - radius;
        const double pixel = index < 0 || index >= static_cast<std::int64_t>(image.size())
                                 ? 1.0 : image[static_cast<std::size_t>(index)];
        const double product = pixel * mask[k];
        expected += product;
        magnitude += std::abs(product);
    }
    const double roundoff = std::numeric_limits<float>::epsilon() / 2.0;
    const double factor = (2.0 * mask.size() + 1.0) * roundoff;
    const double tolerance = 1e-6 + factor / (1.0 - factor) * magnitude;
    if (!std::isfinite(actual) || std::abs(static_cast<double>(actual) - expected) > tolerance) {
        throw std::runtime_error("Last stencil element disagrees with the CPU reference");
    }
}
}  // namespace

int main(int argc, char** argv) {
    unsigned int n = 0;
    unsigned int radius = 0;
    unsigned int threads = 0;
    if (argc != 4 || !positive(argv[1], n) || !positive(argv[2], radius) ||
        !positive(argv[3], threads) || threads > 1024 ||
        2ull * radius + 1 > threads ||
        n > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        std::fprintf(stderr, "Usage: %s n R threads_per_block (positive integers; "
                     "2*R+1 <= threads_per_block <= 1024)\n", argv[0]);
        return 1;
    }

    try {
        const std::size_t mask_count = 2ull * radius + 1;
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
        const std::size_t mask_bytes = mask_count * sizeof(float);
        const std::size_t shared_bytes = (2ull * threads + 4ull * radius + 1) * sizeof(float);
        const std::size_t blocks = n / threads + (n % threads != 0);
        int device = 0;
        cudaDeviceProp properties{};
        check(cudaGetDevice(&device), "cudaGetDevice");
        check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        cudaFuncAttributes attributes{};
        check(cudaFuncGetAttributes(&attributes, stencil_kernel), "cudaFuncGetAttributes");
        if (threads > static_cast<unsigned int>(properties.maxThreadsPerBlock) ||
            threads > static_cast<unsigned int>(properties.maxThreadsDim[0]) ||
            threads > static_cast<unsigned int>(attributes.maxThreadsPerBlock) ||
            blocks > static_cast<std::size_t>(properties.maxGridSize[0]) ||
            shared_bytes + attributes.sharedSizeBytes > properties.sharedMemPerBlock) {
            throw std::runtime_error("Requested stencil launch exceeds this device's limits");
        }

        std::vector<float> image(n), output(n), mask(mask_count);
        std::mt19937 generator(759);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (float& value : image) value = distribution(generator);
        for (float& value : mask) value = distribution(generator);
        DeviceArray device_image(bytes), device_mask(mask_bytes), device_output(bytes);
        check(cudaMemcpy(device_image.get(), image.data(), bytes, cudaMemcpyHostToDevice), "copy image to device");
        check(cudaMemcpy(device_mask.get(), mask.data(), mask_bytes, cudaMemcpyHostToDevice), "copy mask to device");
        Event start, stop;

        // The kernel overwrites every valid output, so no reset is needed after warm-up.
        stencil(device_image.get(), device_mask.get(), device_output.get(), n, radius, threads);
        check(cudaGetLastError(), "stencil warm-up launch");
        check(cudaDeviceSynchronize(), "stencil warm-up synchronization");
        check(cudaEventRecord(start.event), "record start event");
        stencil(device_image.get(), device_mask.get(), device_output.get(), n, radius, threads);
        check(cudaEventRecord(stop.event), "record stop event");
        check(cudaGetLastError(), "stencil launch");
        check(cudaEventSynchronize(stop.event), "synchronize stop event");
        float milliseconds = 0.0f;
        check(cudaEventElapsedTime(&milliseconds, start.event, stop.event), "read elapsed time");
        check(cudaMemcpy(output.data(), device_output.get(), bytes, cudaMemcpyDeviceToHost), "copy output to host");
        verify_last(output.back(), image, mask, radius);
        if (!std::isfinite(milliseconds) || milliseconds < 0.0f) {
            throw std::runtime_error("Invalid CUDA event time");
        }
        start.close();
        stop.close();
        device_image.close();
        device_mask.close();
        device_output.close();
        // Required format: last output element, then CUDA-event milliseconds.
        std::printf("%.9g\n%.6f\n", output.back(), milliseconds);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "task2: %s\n", error.what());
        return 1;
    }
    return 0;
}
