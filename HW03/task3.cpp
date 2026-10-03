#include "msort.h"

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

bool valid_array_size(std::size_t n) {
    return n <= std::numeric_limits<std::size_t>::max() / sizeof(int);
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " n t ts\n";
        return 1;
    }

    std::size_t n = 0;
    int threads = 0;
    std::size_t threshold = 0;
    if (!parse_positive_size(argv[1], n) ||
        !parse_thread_count(argv[2], threads) ||
        !parse_positive_size(argv[3], threshold) || !valid_array_size(n)) {
        std::cerr << "n and ts must be positive, and t must be an integer in [1, 20].\n";
        return 1;
    }

    try {
        std::vector<int> arr(n);
        std::mt19937 generator(759);
        std::uniform_int_distribution<int> distribution(-1000, 1000);
        for (int& value : arr) {
            value = distribution(generator);
        }

        omp_set_num_threads(threads);
        const auto start = std::chrono::high_resolution_clock::now();
        msort(arr.data(), arr.size(), threshold);
        const auto stop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;

        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << arr.front() << '\n'
                  << arr.back() << '\n'
                  << elapsed.count() << '\n';
    } catch (const std::bad_alloc&) {
        std::cerr << "Unable to allocate the input array.\n";
        return 1;
    }
    return 0;
}
