#include "matmul.h"

#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <vector>

namespace {

bool parse_positive_size(const char* text, std::size_t& value) {
    const char* end = text + std::strlen(text);
    const auto [ptr, error] = std::from_chars(text, end, value);
    return error == std::errc{} && ptr == end && value > 0;
}

bool parse_thread_count(const char* text, int& threads) {
    const char* end = text + std::strlen(text);
    const auto [ptr, error] = std::from_chars(text, end, threads);
    return error == std::errc{} && ptr == end && threads >= 1 && threads <= 20;
}

bool square_count(std::size_t n, std::size_t& count) {
    if (n > std::numeric_limits<std::size_t>::max() / n) {
        return false;
    }
    count = n * n;
    return count <= std::numeric_limits<std::size_t>::max() / sizeof(float);
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n t\n";
        return 1;
    }

    std::size_t n = 0;
    int threads = 0;
    std::size_t count = 0;
    if (!parse_positive_size(argv[1], n) ||
        !parse_thread_count(argv[2], threads) || !square_count(n, count)) {
        std::cerr << "n must be positive and t must be an integer in [1, 20].\n";
        return 1;
    }

    try {
        std::vector<float> A(count);
        std::vector<float> B(count);
        std::vector<float> C(count);
        std::mt19937 generator(759);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (std::size_t i = 0; i < count; ++i) {
            A[i] = distribution(generator);
            B[i] = distribution(generator);
        }

        omp_set_num_threads(threads);
        const auto start = std::chrono::high_resolution_clock::now();
        mmul(A.data(), B.data(), C.data(), n);
        const auto stop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;

        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << C[0] << '\n'
                  << C[count - 1] << '\n'
                  << elapsed.count() << '\n';
    } catch (const std::bad_alloc&) {
        std::cerr << "Unable to allocate the matrices.\n";
        return 1;
    }
    return 0;
}
