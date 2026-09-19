#include "convolution.h"
#include "matmul.h"
#include "scan.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T, typename U>
void compare(const T* actual, const std::vector<U>& expected,
             long double tolerance, const std::string& label) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const long double error = std::abs(static_cast<long double>(actual[i]) -
                                           static_cast<long double>(expected[i]));
        require(std::isfinite(actual[i]) &&
                    error <= tolerance * (1.0L + std::abs(expected[i])),
                label + " at element " + std::to_string(i));
    }
}

void test_scan() {
    float sentinel = 17.0f;
    scan(nullptr, &sentinel, 0);
    require(sentinel == 17.0f, "empty scan wrote output");

    const std::vector<float> known{1.0f, -2.0f, 3.0f, 0.0f, -4.0f};
    std::vector<float> result(known.size());
    scan(known.data(), result.data(), known.size());
    compare(result.data(), std::vector<float>{1, -1, 2, 2, -2}, 0, "known scan");

    std::mt19937 generator(2);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (std::size_t n : {1, 2, 3, 17, 1024, 65537}) {
        std::vector<float> input(n);
        std::generate(input.begin(), input.end(), [&] { return distribution(generator); });
        const auto original = input;
        std::vector<float> expected(n);
        std::partial_sum(input.begin(), input.end(), expected.begin());
        std::vector<float> output(n + 2, 12345.0f);
        scan(input.data(), output.data() + 1, n);
        compare(output.data() + 1, expected, 0, "random scan");
        require(input == original, "scan modified input");
        require(output.front() == 12345.0f && output.back() == 12345.0f,
                "scan wrote beyond output");
        scan(input.data(), input.data(), n);
        compare(input.data(), expected, 0, "in-place scan");
    }
}

// Build a padded image explicitly, then take dot products with each window.
std::vector<long double> reference_convolution(const std::vector<float>& image,
                                              const std::vector<float>& mask,
                                              std::size_t n, std::size_t m) {
    const std::size_t radius = m / 2;
    const std::size_t width = n + 2 * radius;
    std::vector<long double> padded(width * width, 0.0L);
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            const bool row_inside = row >= radius && row < radius + n;
            const bool column_inside = column >= radius && column < radius + n;
            if (row_inside && column_inside) {
                padded[row * width + column] = image[(row - radius) * n + column - radius];
            } else if (row_inside != column_inside) {
                padded[row * width + column] = 1.0L;
            }
        }
    }
    std::vector<long double> expected(n * n, 0.0L);
    for (std::size_t index = 0; index < n * n; ++index) {
        for (std::size_t row = 0; row < m; ++row) {
            const auto begin = padded.begin() + (index / n + row) * width + index % n;
            expected[index] += std::inner_product(begin, begin + m,
                                                  mask.begin() + row * m, 0.0L);
        }
    }
    return expected;
}

void test_convolution() {
    float sentinel = 17.0f;
    const float unit = 1.0f;
    convolve(nullptr, &sentinel, 0, &unit, 1);
    require(sentinel == 17.0f, "empty convolution wrote output");

    const std::vector<float> image{1, 3, 4, 8, 6, 5, 2, 4, 3, 4, 6, 8, 1, 4, 5, 2};
    const std::vector<float> mask{0, 0, 1, 0, 1, 0, 1, 0, 0};
    std::vector<float> output(16);
    convolve(image.data(), output.data(), 4, mask.data(), 3);
    compare(output.data(), std::vector<float>{3, 10, 10, 10, 10, 12, 14, 11,
                                              9, 7, 14, 14, 5, 11, 14, 4},
            0, "PDF convolution example");

    // An asymmetric impulse detects a flipped mask and checks all edge types.
    const std::vector<float> impulse{1, 0, 0, 0, 0, 0, 0, 0, 0};
    convolve(image.data(), output.data(), 4, impulse.data(), 3);
    compare(output.data(), std::vector<float>{0, 1, 1, 1, 1, 1, 3, 4,
                                              1, 6, 5, 2, 1, 3, 4, 6},
            0, "convolution mask orientation");

    const float pixel = 7.0f;
    for (std::size_t m : {1, 3, 5, 9}) {
        const std::vector<float> ones(m * m, 1.0f);
        convolve(&pixel, output.data(), 1, ones.data(), m);
        require(output[0] == pixel + 2.0f * static_cast<float>(m - 1),
                "single-pixel edge and corner padding");
    }

    std::mt19937 generator(3);
    std::uniform_real_distribution<float> image_distribution(-10.0f, 10.0f);
    std::uniform_real_distribution<float> mask_distribution(-1.0f, 1.0f);
    for (std::size_t n : {1, 2, 3, 4, 7, 17}) {
        for (std::size_t m : {1, 3, 5, 9}) {
            std::vector<float> input(n * n);
            std::vector<float> kernel(m * m);
            std::generate(input.begin(), input.end(), [&] { return image_distribution(generator); });
            std::generate(kernel.begin(), kernel.end(), [&] { return mask_distribution(generator); });
            const auto original_input = input;
            const auto original_kernel = kernel;
            const auto expected = reference_convolution(input, kernel, n, m);
            std::vector<float> actual(n * n + 2, 12345.0f);
            convolve(input.data(), actual.data() + 1, n, kernel.data(), m);
            compare(actual.data() + 1, expected, 2e-5L, "random convolution");
            require(input == original_input && kernel == original_kernel,
                    "convolution modified inputs");
            require(actual.front() == 12345.0f && actual.back() == 12345.0f,
                    "convolution wrote beyond output");
        }
    }
}

void check_matmul(const std::vector<double>& A, const std::vector<double>& B,
                  unsigned int n, const std::vector<long double>& expected) {
    using Multiply = void (*)(const double*, const double*, double*, unsigned int);
    const Multiply functions[]{mmul1, mmul2, mmul3};
    const auto original_A = A;
    const auto original_B = B;
    for (int variant = 0; variant < 4; ++variant) {
        std::vector<double> output(expected.size() + 2, 12345.0);
        std::fill(output.begin() + 1, output.end() - 1,
                  std::numeric_limits<double>::quiet_NaN());
        for (int repeat = 0; repeat < 2; ++repeat) {
            if (variant == 3) {
                mmul4(A, B, output.data() + 1, n);
            } else {
                functions[variant](A.data(), B.data(), output.data() + 1, n);
            }
            compare(output.data() + 1, expected, 1e-12L,
                    "mmul" + std::to_string(variant + 1));
            require(output.front() == 12345.0 && output.back() == 12345.0,
                    "matmul wrote beyond output");
        }
        require(A == original_A && B == original_B, "matmul modified inputs");
    }
}

void test_matmul() {
    check_matmul({}, {}, 0, {});
    check_matmul({-2}, {3}, 1, {-6});
    check_matmul({1, 2, 3, 4}, {5, 6, 7, 8}, 2, {19, 22, 43, 50});
    check_matmul({1, 2, 3, 4}, {0, 0, 0, 0}, 2, {0, 0, 0, 0});
    check_matmul({1, 2, 3, 4}, {1, 0, 0, 1}, 2, {1, 2, 3, 4});

    std::mt19937 generator(4);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (unsigned int n : {2, 3, 7, 16, 31, 65}) {
        std::vector<double> A(n * n);
        std::vector<double> B(n * n);
        std::generate(A.begin(), A.end(), [&] { return distribution(generator); });
        std::generate(B.begin(), B.end(), [&] { return distribution(generator); });
        std::vector<long double> transposed_B(n * n);
        for (std::size_t index = 0; index < B.size(); ++index) {
            transposed_B[index % n * n + index / n] = B[index];
        }
        std::vector<long double> expected(n * n);
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const auto row = A.begin() + index / n * n;
            const auto column = transposed_B.begin() + index % n * n;
            expected[index] = std::inner_product(row, row + n, column, 0.0L);
        }
        check_matmul(A, B, n, expected);
    }
}

}  // namespace

int main() {
    try {
        test_scan();
        test_convolution();
        test_matmul();
        std::cout << "PASS: scan, convolution, and all four matrix products\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
