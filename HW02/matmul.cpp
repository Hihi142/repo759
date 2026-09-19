#include "matmul.h"

#include <algorithm>

void mmul1(const double* A, const double* B, double* C, const unsigned int n) {
    const std::size_t width = n;
    std::fill_n(C, width * width, 0.0);
    for (std::size_t i = 0; i < width; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
            for (std::size_t k = 0; k < width; ++k) {
                C[i * width + j] += A[i * width + k] * B[k * width + j];
            }
        }
    }
}

void mmul2(const double* A, const double* B, double* C, const unsigned int n) {
    const std::size_t width = n;
    std::fill_n(C, width * width, 0.0);
    for (std::size_t i = 0; i < width; ++i) {
        for (std::size_t k = 0; k < width; ++k) {
            for (std::size_t j = 0; j < width; ++j) {
                C[i * width + j] += A[i * width + k] * B[k * width + j];
            }
        }
    }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n) {
    const std::size_t width = n;
    std::fill_n(C, width * width, 0.0);
    for (std::size_t j = 0; j < width; ++j) {
        for (std::size_t k = 0; k < width; ++k) {
            for (std::size_t i = 0; i < width; ++i) {
                C[i * width + j] += A[i * width + k] * B[k * width + j];
            }
        }
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B,
           double* C, const unsigned int n) {
    const std::size_t width = n;
    std::fill_n(C, width * width, 0.0);
    for (std::size_t i = 0; i < width; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
            for (std::size_t k = 0; k < width; ++k) {
                C[i * width + j] += A[i * width + k] * B[k * width + j];
            }
        }
    }
}
