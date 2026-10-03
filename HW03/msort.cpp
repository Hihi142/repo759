#include "msort.h"

#include <algorithm>
#include <vector>

namespace {

void insertion_sort(int* begin, std::size_t n) {
    for (std::size_t i = 1; i < n; ++i) {
        const int value = begin[i];
        std::size_t j = i;
        while (j > 0 && begin[j - 1] > value) {
            begin[j] = begin[j - 1];
            --j;
        }
        begin[j] = value;
    }
}

void merge_ranges(int* arr, int* scratch, std::size_t left, std::size_t middle,
                  std::size_t right) {
    std::size_t first = left;
    std::size_t second = middle;
    std::size_t out = left;

    while (first < middle && second < right) {
        if (arr[second] < arr[first]) {
            scratch[out++] = arr[second++];
        } else {
            scratch[out++] = arr[first++];
        }
    }
    while (first < middle) {
        scratch[out++] = arr[first++];
    }
    while (second < right) {
        scratch[out++] = arr[second++];
    }
    std::copy(scratch + left, scratch + right, arr + left);
}

void parallel_merge_sort(int* arr, int* scratch, std::size_t left,
                         std::size_t right, std::size_t threshold) {
    const std::size_t n = right - left;
    if (n < 2) {
        return;
    }
    if (n <= threshold) {
        insertion_sort(arr + left, n);
        return;
    }

    const std::size_t middle = left + n / 2;

    #pragma omp task shared(arr, scratch) firstprivate(left, middle, threshold)
    parallel_merge_sort(arr, scratch, left, middle, threshold);

    #pragma omp task shared(arr, scratch) firstprivate(middle, right, threshold)
    parallel_merge_sort(arr, scratch, middle, right, threshold);

    #pragma omp taskwait
    merge_ranges(arr, scratch, left, middle, right);
}

}  // namespace

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    if (arr == nullptr || n < 2) {
        return;
    }

    const std::size_t cutoff = std::max<std::size_t>(threshold, 2);
    std::vector<int> scratch(n);

    #pragma omp parallel
    {
        #pragma omp single nowait
        parallel_merge_sort(arr, scratch.data(), 0, n, cutoff);
    }
}
