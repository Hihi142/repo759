#include "convolution.h"

void convolve(const float* image, float* output, std::size_t n,
              const float* mask, std::size_t m) {
    const auto radius = static_cast<std::ptrdiff_t>(m / 2);
    const auto dimension = static_cast<std::ptrdiff_t>(n);

    for (std::size_t x = 0; x < n; ++x) {
        for (std::size_t y = 0; y < n; ++y) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < m; ++i) {
                const auto row = static_cast<std::ptrdiff_t>(x) +
                                 static_cast<std::ptrdiff_t>(i) - radius;
                const bool row_inside = row >= 0 && row < dimension;
                for (std::size_t j = 0; j < m; ++j) {
                    const auto column = static_cast<std::ptrdiff_t>(y) +
                                        static_cast<std::ptrdiff_t>(j) - radius;
                    const bool column_inside = column >= 0 && column < dimension;
                    float value = 0.0f;
                    if (row_inside && column_inside) {
                        value = image[static_cast<std::size_t>(row) * n +
                                      static_cast<std::size_t>(column)];
                    } else if (row_inside || column_inside) {
                        // One coordinate outside: edge = 1; both outside: corner = 0.
                        value = 1.0f;
                    }
                    sum += mask[i * m + j] * value;
                }
            }
            output[x * n + y] = sum;
        }
    }
}
