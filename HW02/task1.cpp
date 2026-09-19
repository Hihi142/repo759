#include "scan.h"

#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <random>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " n\n";
        return 1;
    }

    std::size_t n = 0;
    const char* end = argv[1] + std::strlen(argv[1]);
    const auto [ptr, error] = std::from_chars(argv[1], end, n);
    const auto max_elements = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    if (error != std::errc{} || ptr != end || n == 0 || n > max_elements) {
        std::cerr << "n must be a positive integer with a representable array size.\n";
        return 1;
    }

    try {
        // unique_ptr releases each array with delete[].
        std::unique_ptr<float[]> input(new float[n]);
        std::unique_ptr<float[]> output(new float[n]);
        std::mt19937 generator(759);  // Reproducible random inputs.
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (std::size_t i = 0; i < n; ++i) {
            input[i] = distribution(generator);
        }

        const auto start = std::chrono::high_resolution_clock::now();
        scan(input.get(), output.get(), n);
        const auto stop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;

        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << elapsed.count() << '\n'
                  << output[0] << '\n'
                  << output[n - 1] << '\n';
    } catch (const std::bad_alloc&) {
        std::cerr << "Unable to allocate the input and output arrays.\n";
        return 1;
    }
    return 0;
}
