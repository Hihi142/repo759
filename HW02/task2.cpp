#include "convolution.h"

#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <random>

namespace {

bool parse_dimension(const char* text, std::size_t& dimension) {
    const char* end = text + std::strlen(text);
    const auto [ptr, error] = std::from_chars(text, end, dimension);
    if (error != std::errc{} || ptr != end || dimension == 0) {
        return false;
    }
    const auto max_elements = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    return dimension <= max_elements / dimension;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n m\n";
        return 1;
    }

    std::size_t n = 0;
    std::size_t m = 0;
    if (!parse_dimension(argv[1], n) || !parse_dimension(argv[2], m) || m % 2 == 0) {
        std::cerr << "n and m must be positive integers with representable matrix "
                     "sizes, and m must be odd.\n";
        return 1;
    }

    try {
        const std::size_t image_size = n * n;
        const std::size_t mask_size = m * m;
        // unique_ptr releases each array with delete[].
        std::unique_ptr<float[]> image(new float[image_size]);
        std::unique_ptr<float[]> mask(new float[mask_size]);
        std::unique_ptr<float[]> output(new float[image_size]);
        std::mt19937 generator(759);
        std::uniform_real_distribution<float> image_distribution(-10.0f, 10.0f);
        std::uniform_real_distribution<float> mask_distribution(-1.0f, 1.0f);
        for (std::size_t i = 0; i < image_size; ++i) {
            image[i] = image_distribution(generator);
        }
        for (std::size_t i = 0; i < mask_size; ++i) {
            mask[i] = mask_distribution(generator);
        }

        const auto start = std::chrono::high_resolution_clock::now();
        convolve(image.get(), output.get(), n, mask.get(), m);
        const auto stop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;

        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << elapsed.count() << '\n'
                  << output[0] << '\n'
                  << output[image_size - 1] << '\n';
    } catch (const std::bad_alloc&) {
        std::cerr << "Unable to allocate the image, mask, and output arrays.\n";
        return 1;
    }
    return 0;
}
