#include "matmul.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <random>

int main() {
    constexpr unsigned int n = 1024;
    constexpr std::size_t count = static_cast<std::size_t>(n) * n;

    try {
        std::unique_ptr<double[]> A(new double[count]);
        std::unique_ptr<double[]> B(new double[count]);
        std::unique_ptr<double[]> C(new double[count]);
        std::mt19937 generator(759);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        for (std::size_t i = 0; i < count; ++i) {
            A[i] = distribution(generator);
            B[i] = distribution(generator);
        }
        const std::vector<double> vector_A(A.get(), A.get() + count);
        const std::vector<double> vector_B(B.get(), B.get() + count);

        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << n << '\n';
        const auto run = [&](auto multiply) {
            const auto start = std::chrono::high_resolution_clock::now();
            multiply();
            const auto stop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::milli> elapsed = stop - start;
            std::cout << elapsed.count() << '\n' << C[count - 1] << '\n';
        };

        run([&] { mmul1(A.get(), B.get(), C.get(), n); });
        run([&] { mmul2(A.get(), B.get(), C.get(), n); });
        run([&] { mmul3(A.get(), B.get(), C.get(), n); });
        run([&] { mmul4(vector_A, vector_B, C.get(), n); });
    } catch (const std::bad_alloc&) {
        std::cerr << "Unable to allocate the matrices.\n";
        return 1;
    }
    return 0;
}
