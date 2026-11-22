#include <cuda_runtime.h>
#include "matmul.cuh"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void cleanup_error(cudaError_t status, const char* operation) noexcept {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "task1 cleanup: %s: %s\n", operation,
                     cudaGetErrorString(status));
    }
}

bool positive(const char* text, unsigned int& value) {
    if (text == nullptr || *text == '\0') return false;
    const char* end = text + std::strlen(text);
    for (const char* p = text; p != end; ++p) {
        if (*p < '0' || *p > '9') return false;
    }
    const auto parsed = std::from_chars(text, end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end && value != 0;
}

template <typename T>
class ManagedArray {
public:
    explicit ManagedArray(std::size_t count) {
        check(cudaMallocManaged(reinterpret_cast<void**>(&data_), count * sizeof(T)),
              "cudaMallocManaged");
    }
    ~ManagedArray() {
        if (data_ != nullptr) cleanup_error(cudaFree(data_), "cudaFree");
    }
    ManagedArray(const ManagedArray&) = delete;
    ManagedArray& operator=(const ManagedArray&) = delete;
    T* get() const { return data_; }
    void close() {
        T* old = data_;
        data_ = nullptr;
        check(cudaFree(old), "cudaFree");
    }
private:
    T* data_ = nullptr;
};

class Event {
public:
    Event() { check(cudaEventCreate(&event), "cudaEventCreate"); }
    ~Event() {
        if (event != nullptr) cleanup_error(cudaEventDestroy(event), "cudaEventDestroy");
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    void close() {
        cudaEvent_t old = event;
        event = nullptr;
        check(cudaEventDestroy(old), "cudaEventDestroy");
    }
    cudaEvent_t event = nullptr;
};

struct Reference {
    long double value = 0.0L;
    long double magnitude = 0.0L;
};

template <typename T>
Reference reference_element(const T* a, const T* b, unsigned int n,
                            unsigned int row, unsigned int column) {
    Reference reference;
    for (unsigned int k = 0; k < n; ++k) {
        const long double product =
            static_cast<long double>(a[static_cast<std::size_t>(row) * n + k]) *
            static_cast<long double>(b[static_cast<std::size_t>(k) * n + column]);
        reference.value += product;
        reference.magnitude += std::abs(product);
    }
    return reference;
}

template <typename T>
void verify(T actual, Reference expected, unsigned int n, const char* position) {
    long double tolerance = 0.0L;
    if constexpr (!std::is_integral<T>::value) {
        // A conservative dot-product roundoff bound also allows CUDA FMA.
        const long double unit = std::numeric_limits<T>::epsilon() / 2.0L;
        const long double factor = (2.0L * n + 1.0L) * unit;
        tolerance = factor / (1.0L - factor) * expected.magnitude +
                    8.0L * std::numeric_limits<T>::epsilon();
    }
    if (!std::isfinite(static_cast<long double>(actual)) ||
        std::abs(static_cast<long double>(actual) - expected.value) > tolerance) {
        throw std::runtime_error(std::string(position) +
                                 " matrix element disagrees with CPU reference");
    }
}

template <typename T>
struct Result {
    T first;
    T last;
    float milliseconds;
};

template <typename T>
Result<T> run(unsigned int n, unsigned int block_dim,
               void (*multiply)(const T*, const T*, T*, unsigned int, unsigned int)) {
    const std::size_t count = static_cast<std::size_t>(n) * n;
    ManagedArray<T> a(count), b(count), c(count);
    std::mt19937 generator(759);
    if constexpr (std::is_integral<T>::value) {
        // |sum| <= 4*n <= 65536, safely below the int range.
        std::uniform_int_distribution<int> distribution(-2, 2);
        for (std::size_t i = 0; i < count; ++i) {
            a.get()[i] = distribution(generator);
            b.get()[i] = distribution(generator);
            c.get()[i] = 0;
        }
    } else {
        std::uniform_real_distribution<T> distribution(T(-1), T(1));
        for (std::size_t i = 0; i < count; ++i) {
            a.get()[i] = distribution(generator);
            b.get()[i] = distribution(generator);
            c.get()[i] = std::numeric_limits<T>::quiet_NaN();
        }
    }
    const Reference first_reference = reference_element(a.get(), b.get(), n, 0, 0);
    const Reference last_reference = reference_element(a.get(), b.get(), n, n - 1, n - 1);
    Event start, stop;

    // This also migrates managed input/output pages to the device. Do not touch
    // managed allocations on the CPU between warm-up and the timed call.
    multiply(a.get(), b.get(), c.get(), n, block_dim);
    check(cudaEventRecord(start.event), "record start event");
    multiply(a.get(), b.get(), c.get(), n, block_dim);
    check(cudaEventRecord(stop.event), "record stop event");
    check(cudaEventSynchronize(stop.event), "synchronize stop event");
    Result<T> result{};
    check(cudaEventElapsedTime(&result.milliseconds, start.event, stop.event),
          "read elapsed time");
    result.first = c.get()[0];
    result.last = c.get()[count - 1];
    verify(result.first, first_reference, n, "First");
    verify(result.last, last_reference, n, "Last");
    if (!std::isfinite(result.milliseconds) || result.milliseconds < 0.0f) {
        throw std::runtime_error("Invalid CUDA event time");
    }
    start.close();
    stop.close();
    a.close();
    b.close();
    c.close();
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    unsigned int n = 0;
    unsigned int block_dim = 0;
    if (argc != 3 || !positive(argv[1], n) || n > (1u << 14) ||
        !positive(argv[2], block_dim) || block_dim > 32) {
        std::fprintf(stderr, "Usage: %s n block_dim (1 <= n <= 16384; "
                     "1 <= block_dim <= 32)\n", argv[0]);
        return 1;
    }
    try {
        // Free each type before allocating the next: the largest case needs
        // three double matrices (6 GiB), rather than nine simultaneous arrays.
        const Result<int> integers = run<int>(n, block_dim, matmul_1);
        const Result<float> floats = run<float>(n, block_dim, matmul_2);
        const Result<double> doubles = run<double>(n, block_dim, matmul_3);
        std::printf("%d\n%d\n%.6f\n%.9g\n%.9g\n%.6f\n%.17g\n%.17g\n%.6f\n",
                     integers.first, integers.last, integers.milliseconds,
                     floats.first, floats.last, floats.milliseconds,
                     doubles.first, doubles.last, doubles.milliseconds);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "task1: %s\n", error.what());
        return 1;
    }
    return 0;
}
