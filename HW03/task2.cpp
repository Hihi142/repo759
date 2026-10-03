#include "convolution.h"

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
        constexpr std::size_t mask_width = 3;
        constexpr std::size_t mask_count = mask_width * mask_width;
        std::vector<float> image(count);
        std::vector<float> mask(mask_count);
        std::vector<float> output(count);

        std::mt19937 generator(759);
        std::uniform_real_distribution<float> image_distribution(-10.0f, 10.0f);
        std::uniform_real_distribution<float> mask_distribution(-1.0f, 1.0f);
        for (float& value : image) {
            value = image_distribution(generator);
        }
        for (float& value : mask) {
            value = mask_distribution(generator);
        }

        omp_set_num_threads(threads);
        const auto start = std::chrono::high_resolution_clock::now();
        convolve(image.data(), output.data(), n, mask.data(), mask_width);
        const auto stop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;

        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << output[0] << '\n'
                  << output[count - 1] << '\n'
                  << elapsed.count() << '\n';
    } catch (const std::bad_alloc&) {
        std::cerr << "Unable to allocate the image, mask, and output arrays.\n";
        return 1;
    }
    return 0;
}
