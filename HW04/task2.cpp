#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

// Parameters and array interfaces follow the supplied task2.cpp skeleton.
const double G = 1.0;
const double softening = 0.1;
const double dt = 0.01;
const double board_size = 4.0;

void getAcc(const double pos[][3], const double mass[], double acc[][3], int N) {

    for (int i = 0; i < N; ++i) {
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                continue;
            }
            const double dx = pos[j][0] - pos[i][0];
            const double dy = pos[j][1] - pos[i][1];
            const double dz = pos[j][2] - pos[i][2];
            const double r2 = dx * dx + dy * dy + dz * dz + softening * softening;
            const double inv_r = 1.0 / std::sqrt(r2);
            const double factor = G * mass[j] * inv_r * inv_r * inv_r;
            ax += dx * factor;
            ay += dy * factor;
            az += dz * factor;
        }
        // Each iteration owns one output particle; the input positions are read-only.
        acc[i][0] = ax;
        acc[i][1] = ay;
        acc[i][2] = az;
    }
}

void advanceSystem(double pos[][3], double vel[][3], const double mass[],
                   double acc[][3], int N) {

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            vel[i][k] += acc[i][k] * dt / 2.0;
            pos[i][k] += vel[i][k] * dt;
            pos[i][k] = std::clamp(pos[i][k], -board_size, board_size);
        }
    }
    // The parallel loop's implicit barrier completes all drifts before force evaluation.
    getAcc(pos, mass, acc, N);

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            vel[i][k] += acc[i][k] * dt / 2.0;
        }
    }
}

void initializeSystem(double pos[][3], double vel[][3], double mass[], int N) {
    // Fixed seed makes serial/OpenMP runs comparable. NumPy uses a different sampler.
    std::mt19937 generator(17);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    double momentum[3] = {0.0, 0.0, 0.0};
    double totalMass = 0.0;
    for (int i = 0; i < N; ++i) {
        mass[i] = uniform_dist(generator);
        for (int k = 0; k < 3; ++k) {
            pos[i][k] = normal_dist(generator);
        }
        for (int k = 0; k < 3; ++k) {
            vel[i][k] = normal_dist(generator);
            momentum[k] += mass[i] * vel[i][k];
        }
        totalMass += mass[i];
    }
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            vel[i][k] -= momentum[k] / totalMass;
        }
    }
}

void savePositionsToCSV(const double pos[][3], int N, int step,
                        const std::string& filename) {
    std::ofstream file(filename, std::ios::app);
    if (!file) {
        throw std::runtime_error("Unable to open CSV: " + filename);
    }
    file << std::setprecision(std::numeric_limits<double>::max_digits10);
    file << step << ",[";
    for (int i = 0; i < N; ++i) {
        if (i != 0) {
            file << ',';
        }
        file << '[' << pos[i][0] << ',' << pos[i][1] << ',' << pos[i][2] << ']';
    }
    file << "]\n";
    // Buffered writes can fail only when flushed or closed (for example, a full disk).
    file.close();
    if (!file) {
        throw std::runtime_error("Unable to write CSV: " + filename);
    }
}

int positiveInt(const char* text) {
    int value = 0;
    const char* end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, value);
    if (result.ec != std::errc{} || result.ptr != end || value <= 0) {
        throw std::invalid_argument("Expected a positive integer.");
    }
    return value;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <number_of_particles> <simulation_end_time>\n";
        return 1;
    }
    try {
        const int N = positiveInt(argv[1]);
        std::size_t consumed = 0;
        const double tEnd = std::stod(argv[2], &consumed);
        const double steps = std::ceil(tEnd / dt);
        if (consumed != std::strlen(argv[2]) || !std::isfinite(tEnd) || tEnd < 0.0 ||
            !std::isfinite(steps) || steps > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("End time must be finite, nonnegative and fit the step counter.");
        }
        const int Nt = static_cast<int>(steps);

        auto mass = std::make_unique<double[]>(N);
        std::unique_ptr<double[][3]> pos(new double[N][3]);
        std::unique_ptr<double[][3]> vel(new double[N][3]);
        std::unique_ptr<double[][3]> acc(new double[N][3]);
        initializeSystem(pos.get(), vel.get(), mass.get(), N);

        // Debug output is opt-in, and never created during default timing runs.
        const char* csv = std::getenv("NBODY_CSV");
        if (csv && *csv) {
            std::ofstream file(csv, std::ios::trunc);
            if (!file) {
                throw std::runtime_error("Unable to create CSV file.");
            }
            file.close();
            savePositionsToCSV(pos.get(), N, 0, csv);
        }
        const auto start = std::chrono::steady_clock::now();
        getAcc(pos.get(), mass.get(), acc.get(), N);
        for (int step = 0; step < Nt; ++step) {
            advanceSystem(pos.get(), vel.get(), mass.get(), acc.get(), N);
            if (csv && *csv) {
                savePositionsToCSV(pos.get(), N, step + 1, csv);
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << std::setprecision(10) << "time: " << elapsed.count() << "ms\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
