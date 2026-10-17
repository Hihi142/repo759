#include <algorithm>
#include <cerrno>
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
#include <vector>
#include <omp.h>
#ifdef __linux__
#include <sched.h>
#endif
// Parameters and array interfaces follow the supplied task2.cpp skeleton.
const double G = 1.0;
const double softening = 0.1;
const double dt = 0.01;
const double board_size = 4.0;

void getAcc(const double pos[][3], const double mass[], double acc[][3], int N) {
#pragma omp parallel for schedule(runtime)
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
#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            vel[i][k] += acc[i][k] * dt / 2.0;
            pos[i][k] += vel[i][k] * dt;
            pos[i][k] = std::clamp(pos[i][k], -board_size, board_size);
        }
    }
    // The parallel loop's implicit barrier completes all drifts before force evaluation.
    getAcc(pos, mass, acc, N);
#pragma omp parallel for schedule(runtime)
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


struct ThreadPlacement {
    int place = -1;
    int cpu = -1;
    int affinityError = 0;
#ifdef __linux__
    cpu_set_t affinity{};
#endif
};

const char* scheduleName(omp_sched_t schedule) {
    // The high bit can carry the monotonic modifier independently of the kind.
    switch (static_cast<unsigned int>(schedule) & 0x7fffffffU) {
        case omp_sched_static: return "static";
        case omp_sched_dynamic: return "dynamic";
        case omp_sched_guided: return "guided";
        case omp_sched_auto: return "auto";
        default: return "unknown";
    }
}

const char* bindingName(omp_proc_bind_t binding) {
    switch (binding) {
        case omp_proc_bind_false: return "false";
        case omp_proc_bind_true: return "true";
        case omp_proc_bind_master: return "primary";
        case omp_proc_bind_close: return "close";
        case omp_proc_bind_spread: return "spread";
        default: return "unknown";
    }
}

void verifyOpenMPRuntime(int requestedThreads, const char* diagnosticsPath) {
    const bool diagnostics = diagnosticsPath && *diagnosticsPath;
    std::vector<ThreadPlacement> placements(diagnostics ? requestedThreads : 0);
    int actualThreads = 0;
    // This uses the same top-level team settings as the timed parallel loops.
    // It also creates the worker pool in this process before timing starts.
#pragma omp parallel
    {
#pragma omp single
        actualThreads = omp_get_num_threads();
        if (diagnostics) {
            const int thread = omp_get_thread_num();
            if (thread < requestedThreads) {
                ThreadPlacement& placement = placements[thread];
                placement.place = omp_get_place_num();
#ifdef __linux__
                placement.cpu = sched_getcpu();
                if (sched_getaffinity(0, sizeof(placement.affinity),
                                      &placement.affinity) != 0) {
                    // Record a failure explicitly rather than claim an empty affinity.
                    placement.affinityError = errno;
                }
#endif
            }
        }
    }
    if (actualThreads != requestedThreads) {
        throw std::runtime_error(
            "OpenMP requested " + std::to_string(requestedThreads) +
            " threads but created " + std::to_string(actualThreads) +
            ". Check OMP_THREAD_LIMIT, OMP_MAX_ACTIVE_LEVELS and the runtime environment.");
    }
    if (!diagnostics) {
        return;
    }

    omp_sched_t schedule;
    int chunkSize = 0;
    omp_get_schedule(&schedule, &chunkSize);
    const omp_proc_bind_t binding = omp_get_proc_bind();
    const int numPlaces = omp_get_num_places();
    std::ofstream file(diagnosticsPath, std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Unable to create OpenMP diagnostics: " +
                                 std::string(diagnosticsPath));
    }
    file << "{\n  \"schema_version\": 1,\n  \"openmp_version\": " << _OPENMP
         << ",\n  \"requested_threads\": " << requestedThreads
         << ",\n  \"actual_threads\": " << actualThreads
         << ",\n  \"dynamic\": " << (omp_get_dynamic() ? "true" : "false")
         << ",\n  \"max_active_levels\": " << omp_get_max_active_levels()
         << ",\n  \"thread_limit\": " << omp_get_thread_limit()
         << ",\n  \"schedule\": {\"kind\": \"" << scheduleName(schedule)
         << "\", \"chunk_size\": " << chunkSize
         << ", \"raw_kind\": " << static_cast<unsigned int>(schedule) << "},"
         << "\n  \"proc_bind\": {\"raw_kind\": " << static_cast<int>(binding)
         << ", \"name\": \"" << bindingName(binding) << "\"},"
         << "\n  \"num_places\": " << numPlaces
         << ",\n  \"places\": [";
    for (int place = 0; place < numPlaces; ++place) {
        std::vector<int> cpus(omp_get_place_num_procs(place));
        omp_get_place_proc_ids(place, cpus.data());
        file << (place == 0 ? "\n    " : ",\n    ")
             << "{\"place\": " << place << ", \"cpu_ids\": [";
        for (std::size_t index = 0; index < cpus.size(); ++index) {
            file << (index == 0 ? "" : ", ") << cpus[index];
        }
        file << "]}";
    }
    file << "\n  ],\n  \"threads\": [";
    for (int thread = 0; thread < actualThreads; ++thread) {
        const ThreadPlacement& placement = placements[thread];
        file << (thread == 0 ? "\n    " : ",\n    ")
             << "{\"thread\": " << thread << ", \"place\": " << placement.place
             << ", \"cpu\": ";
        if (placement.cpu >= 0) {
            file << placement.cpu;
        } else {
            file << "null";
        }
        file << ", \"affinity\": [";
#ifdef __linux__
        bool first = true;
        if (placement.affinityError == 0) {
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                if (CPU_ISSET(cpu, &placement.affinity)) {
                    file << (first ? "" : ", ") << cpu;
                    first = false;
                }
            }
        }
#endif
        file << "], \"affinity_error\": ";
        if (placement.affinityError != 0) {
            file << placement.affinityError;
        } else {
            file << "null";
        }
        file << '}';
    }
    file << "\n  ]\n}\n";
    file.close();
    if (!file) {
        throw std::runtime_error("Unable to write OpenMP diagnostics: " +
                                 std::string(diagnosticsPath));
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <number_of_particles> <simulation_end_time> <num_threads>\n";
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
        const int threads = positiveInt(argv[3]);
        if (threads > 8) {
            throw std::invalid_argument("num_threads must be in [1, 8].");
        }
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        // Select static/dynamic/guided with OMP_SCHEDULE, without recompiling.
        if (std::getenv("OMP_SCHEDULE") == nullptr) {
            omp_set_schedule(omp_sched_static, 0);
        }
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
        verifyOpenMPRuntime(threads, std::getenv("NBODY_DIAGNOSTICS"));
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
