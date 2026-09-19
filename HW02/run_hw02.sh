#!/usr/bin/env bash
# Run from Euler's login shell: ./run_hw02.sh
# The default action submits a batch job; computation runs on its compute node.
set -Eeuo pipefail
on_error() {
    local exit_status=$?
    printf 'HW02 failed at script line %s (exit %s). Check the printed log directory.\n' "$1" "$exit_status" >&2
    exit "$exit_status"
}
trap 'on_error "$LINENO"' ERR

usage() {
    cat <<'HELP'
Usage: ./run_hw02.sh                 Submit the complete Euler assignment run.
       ./run_hw02.sh --local-check   Run a smaller, clearly labelled local check.
       ./run_hw02.sh --report DIR    Generate PDFs locally from a downloaded run.
       ./run_hw02.sh --help

Euler: instruction partition, 1 CPU, 16 GB RAM, 30 minutes.
Runs task1 for every power of two from 2^10 to 2^30, task2 with n=256/m=3,
and task3 with its required 1024 x 1024 matrices. Saves raw outputs, CSVs,
source snapshots, metadata, and completion status. No plotting on Euler.
Results: ../output/HW02/run-<timestamp>-<suffix>/ (relative to HW02).
The ../output/HW02/latest symlink points to the most recently submitted run.

After downloading a run directory, use --report DIR on your local Linux
machine to create task1.pdf and assignment2.pdf. Only this local report mode
installs plotting packages, in a dedicated cache (override: HW02_CACHE_DIR).
A local check uses only 2^10..2^16 for task1 and writes to a temporary
directory; reports made from it are labelled NOT FOR SUBMISSION.
HELP
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

mode=${1:---submit}
if [[ $mode == --help || $mode == -h ]]; then
    usage
    exit 0
fi
case $mode in
    --submit) (( $# <= 1 )) || fail 'Unexpected arguments.' ;;
    --local-check) (( $# == 1 )) || fail 'Unexpected arguments.' ;;
    --report) (( $# == 2 )) || fail '--report needs the downloaded run directory.' ;;
    --run) (( $# == 3 )) || fail '--run is reserved for the submitted Slurm job.' ;;
    *) usage >&2; exit 1 ;;
esac

if command -v module >/dev/null 2>&1; then
    module purge
fi
[[ -z ${LOADEDMODULES:-} ]] || fail 'Unload all modules before running this script.'
command -v python3 >/dev/null || fail 'python3 is required.'
if [[ $mode != --report ]]; then
    command -v g++ >/dev/null || fail 'g++ is required.'
fi

if [[ $mode == --report ]]; then
    [[ $(hostname) != *euler* ]] || fail 'Download results and generate PDFs on your local Linux machine.'
    result_dir=$(cd -- "$2" && pwd)
    [[ -f "$result_dir/metadata.json" ]] || fail 'Missing metadata.json in the run directory.'
    source_dir="$result_dir/source"
    run_kind=report
    python_version=$(python3 -c 'import sys; print("%s.%s" % sys.version_info[:2])')
    cache_dir=${HW02_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/ece759-hw02}
    mkdir -p -- "$cache_dir/py$python_version"
    cache_dir=$(cd -- "$cache_dir" && pwd)
    package_dir="$cache_dir/py$python_version/packages"
    if ! PYTHONPATH="$package_dir${PYTHONPATH:+:$PYTHONPATH}" python3 -c 'import matplotlib, reportlab' >/dev/null 2>&1; then
        printf 'Installing local plotting packages into %s\n' "$package_dir"
        python3 -m pip install --disable-pip-version-check --no-input --only-binary=:all: \
            --upgrade --target "$package_dir" 'matplotlib>=3.7,<4' 'reportlab>=4,<5'
    fi
    export PYTHONPATH="$package_dir${PYTHONPATH:+:$PYTHONPATH}"
    export MPLBACKEND=Agg
    MPLCONFIGDIR="$(dirname -- "$package_dir")/matplotlib"
    export MPLCONFIGDIR
elif [[ $mode == --run ]]; then
    [[ -n ${SLURM_JOB_ID:-} ]] || fail 'Benchmarks must run in a Slurm allocation.'
    source_dir=$2
    result_dir=$3
    run_kind=euler
else
    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    script_path="$script_dir/$(basename -- "${BASH_SOURCE[0]}")"
    repository_dir=$(cd -- "$script_dir/.." && pwd)
    required=(scan.cpp scan.h task1.cpp convolution.cpp convolution.h task2.cpp matmul.cpp matmul.h task3.cpp)
    for name in "${required[@]}"; do
        [[ -f "$script_dir/$name" ]] || fail "Missing source file: $script_dir/$name"
    done

    if [[ $mode == --submit ]]; then
        command -v sbatch >/dev/null || fail 'Run this command on Euler, or use --local-check locally.'
    else
        [[ $(hostname) != *euler* ]] || fail 'On Euler, run without --local-check to submit a job.'
    fi

    if [[ $mode == --submit ]]; then
        result_root="$repository_dir/output/HW02"
        mkdir -p -- "$result_root"
        result_dir=$(mktemp -d "$result_root/run-$(date +%Y%m%d-%H%M%S)-XXXXXX")
    else
        result_dir=$(mktemp -d "${TMPDIR:-/tmp}/hw02-local-check-XXXXXX")
    fi
    source_dir="$result_dir/source"
    mkdir -- "$source_dir"
    for name in "${required[@]}"; do
        cp -- "$script_dir/$name" "$source_dir/$name"
    done
    cp -- "$script_path" "$source_dir/run_hw02.sh"

    if [[ $mode == --submit ]]; then
        # The largest scan needs 8 GiB for the two float arrays alone.
        submission=$(sbatch --parsable --partition=instruction --nodes=1 --ntasks=1 \
            --cpus-per-task=1 --mem=16G --time=00:30:00 --job-name=HW02 \
            --chdir="$result_dir" --output="$result_dir/slurm-%j.out" \
            --error="$result_dir/slurm-%j.err" --export=ALL \
            "$source_dir/run_hw02.sh" --run "$source_dir" "$result_dir")
        job_id=${submission%%;*}
        [[ $job_id =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response: $submission"
        printf '%s\n' "$job_id" > "$result_dir/job-id.txt"
        ln -sfn -- "$(basename -- "$result_dir")" "$result_root/latest"
        printf 'Submitted HW02 job %s. It will continue if you disconnect SSH.\n' "$job_id"
        printf 'Results: %s\n' "$result_dir"
        printf 'Queue:   squeue -j %s\n' "$job_id"
        printf 'Log:     tail -f %q\n' "$result_dir/slurm-$job_id.out"
        printf 'Finished successfully only when this file exists: %s/SUCCESS.txt\n' "$result_dir"
        exit 0
    fi
    run_kind=local-check
    printf 'LOCAL CHECK ONLY: 2^10..2^16; outputs are not submission data.\n'
fi

build_dir="$result_dir/build"
printf 'Results: %s\n' "$result_dir"

# Embed the existing independent correctness tests to keep deployment to one script.
if [[ $run_kind != report ]]; then
mkdir -p -- "$build_dir"
cat > "$build_dir/correctness.cpp" <<'HW02_CORRECTNESS_CPP'
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
HW02_CORRECTNESS_CPP
fi

python3 -u - "$source_dir" "$result_dir" "$run_kind" <<'HW02_PYTHON'
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from xml.sax.saxutils import escape


source, output = map(Path, sys.argv[1:3])
report_only = sys.argv[3] == "report"
local_check = sys.argv[3] == "local-check"
build = output / "build"
raw = output / "raw"
if not report_only:
    raw.mkdir()
github_url = "https://github.com/Hihi142/repo759/tree/main/HW02"
exponents = list(range(10, 17 if local_check else 31))


def execute(command, name, cwd=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    with (raw / (name + ".out")).open("w") as stdout, (raw / (name + ".err")).open("w") as stderr:
        subprocess.run(list(map(str, command)), cwd=cwd, stdout=stdout, stderr=stderr, check=True)
    return (raw / (name + ".out")).read_text()


def numbers(text, count, name):
    lines = text.splitlines()
    if len(lines) != count or not text.endswith("\n"):
        raise ValueError("%s: expected exactly %s newline-terminated values" % (name, count))
    values = [float(line) for line in lines]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(name + ": non-finite output")
    return values


def write_csv(name, header, rows):
    with (output / name).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def make_pdfs(scan_rows, convolution, product, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    ns = [row[1] for row in scan_rows]
    times = [row[2] for row in scan_rows]
    fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
    ax.plot(ns, times, color="#176b87", marker="o", markersize=4, linewidth=1.6)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    stride = max(1, math.ceil(len(exponents) / 6))
    ticks = sorted(set(exponents[::stride] + [exponents[-1]]))
    ax.set_xticks([2 ** power for power in ticks], [r"$2^{%d}$" % power for power in ticks])
    ax.set_xlabel("Array length n (elements; logarithmic scale)")
    ax.set_ylabel("Scan execution time (milliseconds; logarithmic scale)")
    title = "Task 1: inclusive scan scaling on Euler"
    if local_check:
        title = "LOCAL CHECK - NOT FOR SUBMISSION\nTask 1: reduced scan sweep"
    ax.set_title(title, pad=14)
    ax.grid(True, which="major", color="#d6dce2", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0, -0.22, "%d sizes; one run per size; initialization excluded from timing."
            % len(scan_rows), transform=ax.transAxes, fontsize=9, color="#444444")
    fig.savefig(output / "task1.pdf", metadata={"Title": title})
    plt.close(fig)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="HWBody", fontName="Helvetica", fontSize=9.5,
                              leading=13, spaceAfter=7, textColor=colors.HexColor("#253444")))
    styles.add(ParagraphStyle(name="HWHeading", fontName="Helvetica-Bold", fontSize=11,
                              leading=14, spaceBefore=9, spaceAfter=6,
                              textColor=colors.HexColor("#176b87")))
    story = []

    def body(text):
        story.append(Paragraph(text, styles["HWBody"]))

    def heading(text):
        story.append(Paragraph(text, styles["HWHeading"]))

    story.append(Paragraph("ECE 759 - Assignment 2", styles["Title"]))
    if local_check:
        body("<b>LOCAL CHECK - NOT FOR SUBMISSION.</b> The scan sweep is reduced; "
             "all values in this document are local test data.")
    body('GitHub: <link href="%s" color="#176b87">%s</link>' % (github_url, github_url))
    body("Host: %s &nbsp; | &nbsp; Slurm job: %s<br/>UTC start: %s<br/>"
         "Compiler: %s<br/>Build flags: -Wall -O3 -std=c++17. "
         "No environment modules loaded. Timings cover only the algorithm calls."
         % tuple(escape(str(metadata[key])) for key in ("hostname", "slurm_job_id", "started_utc", "compiler")))

    heading("1. Inclusive scan")
    body("Tested all %d powers of two from 2^%d through 2^%d, with one run per size. "
         "The labelled scaling plot is supplied separately as <b>task1.pdf</b>; "
         "task1.csv contains the measured time and first/last output elements for every size."
         % (len(scan_rows), exponents[0], exponents[-1]))

    heading("2. Convolution")
    body("For n = 256 and m = 3: time = %.6f ms; first output = %.9g; last output = %.9g. "
         "The correctness checks include the assignment's example, an asymmetric mask, "
         "edge/corner padding, masks larger than the image, and random inputs."
         % tuple(convolution))

    heading("3. Matrix multiplication")
    body("A and B are %d x %d matrices. The four implementations passed independent "
         "elementwise correctness checks on smaller matrices; the full-size runs also "
         "have matching last elements within floating-point tolerance."
         % (int(product[0]), int(product[0])))
    timings = product[1::2]
    last = product[2::2]
    orders = ["i, j, k", "i, k, j", "j, k, i", "i, j, k (vector)"]
    rows = [["Function", "Loop order", "Time (ms)", "Last element of C"]]
    rows.extend([["mmul%d" % (i + 1), orders[i], "%.6f" % timings[i], "%.17g" % last[i]]
                 for i in range(4)])
    table = Table(rows, colWidths=[54, 102, 100, 236], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176b87")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eef4f6"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
    ]))
    story.extend([table, Spacer(1, 9)])
    ranking = sorted(range(4), key=lambda i: timings[i])
    body("Observed order, fastest to slowest: %s. Relative to mmul1, the elapsed-time "
         "ratios for mmul2, mmul3, and mmul4 are %.3f, %.3f, and %.3f, respectively."
         % (", ".join("mmul%d" % (i + 1) for i in ranking),
            timings[1] / timings[0], timings[2] / timings[0], timings[3] / timings[0]))
    body("With row-major storage, mmul1 traverses a row of A and a strided column of B "
         "in its inner loop. In mmul2, the inner j loop walks contiguous rows of B and C "
         "while reusing A[i,k], improving spatial locality and making vectorization easier. "
         "The inner i loop in mmul3 strides through both A and C by %d bytes, which can "
         "increase cache misses and memory traffic."
         % (int(product[0]) * 8))
    body("mmul4 uses the same loop order as mmul1, and std::vector also stores elements "
         "contiguously. Its vectors are passed by const reference, with conversion outside "
         "the timer, so similar performance is expected. A single-run timing difference "
         "does not isolate container overhead from compiler effects and system variability.")
    body("The raw outputs and metadata.json accompany these results. The maximum "
         "difference between the four reported last elements is %.3g, consistent with "
         "floating-point rounding." % (max(last) - min(last)))

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#667788"))
        canvas.drawString(0.8 * inch, 0.45 * inch,
                          "LOCAL CHECK - NOT FOR SUBMISSION" if local_check else "ECE 759 | Assignment 2")
        canvas.drawRightString(letter[0] - 0.8 * inch, 0.45 * inch, str(doc.page))

    SimpleDocTemplate(str(output / "assignment2.pdf"), pagesize=letter,
                      rightMargin=0.8 * inch, leftMargin=0.8 * inch,
                      topMargin=0.55 * inch, bottomMargin=0.65 * inch,
                      title="ECE 759 Assignment 2").build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    compiler = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    metadata = {
        "run_kind": "local-check-not-for-submission" if local_check else "euler",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "compiler": compiler, "flags": ["-Wall", "-O3", "-std=c++17"],
        "loaded_modules": os.environ.get("LOADEDMODULES", ""),
        "python": sys.version,
        "scan_exponents": exponents, "task2_n": 256, "task2_m": 3,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(source.iterdir()) if path.is_file()},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    flags = ["-Wall", "-O3", "-std=c++17"]
    for task, files in [("task1", ["scan.cpp", "task1.cpp"]),
                        ("task2", ["convolution.cpp", "task2.cpp"]),
                        ("task3", ["task3.cpp", "matmul.cpp"])]:
        execute(["g++", *files, *flags, "-o", build / task], "compile-" + task, cwd=source)
    execute(["g++", build / "correctness.cpp", source / "scan.cpp", source / "convolution.cpp",
             source / "matmul.cpp", "-I", source, *flags, "-o", build / "correctness"], "compile-checks")
    print(execute([build / "correctness"], "correctness"), end="", flush=True)

    scan_rows = []
    for exponent in exponents:
        n = 1 << exponent
        values = numbers(execute([build / "task1", n], "task1-2pow%d" % exponent), 3, "task1")
        if values[0] <= 0 or not -1 <= values[1] <= 1 or abs(values[2]) > n:
            raise ValueError("Unexpected task1 result for n=%s" % n)
        scan_rows.append([exponent, n, *values])
        write_csv("task1.csv", ["exponent", "n", "time_ms", "first", "last"], scan_rows)
        print("Task 1 completed: 2^%d (%d/%d)" % (exponent, len(scan_rows), len(exponents)), flush=True)
    if not local_check and [row[1] for row in scan_rows] != [1 << i for i in range(10, 31)]:
        raise ValueError("Incomplete assignment scan sweep")

    convolution = numbers(execute([build / "task2", 256, 3], "task2"), 3, "task2")
    if convolution[0] < 0:
        raise ValueError("Negative convolution time")
    write_csv("task2.csv", ["n", "m", "time_ms", "first", "last"], [[256, 3, *convolution]])

    product = numbers(execute([build / "task3"], "task3"), 9, "task3")
    if not product[0].is_integer() or product[0] < 1000 or any(t <= 0 for t in product[1::2]):
        raise ValueError("Invalid task3 dimensions or timings")
    if not all(math.isclose(product[2], value, rel_tol=1e-12, abs_tol=1e-12) for value in product[2::2]):
        raise ValueError("The four matrix products disagree")
    write_csv("task3.csv", ["function", "n", "time_ms", "last"],
              [["mmul%d" % (i + 1), int(product[0]), product[1 + 2 * i], product[2 + 2 * i]]
               for i in range(4)])
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["correctness_checks_passed"] = True
    metadata["scan_rows"] = len(scan_rows)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    status = "LOCAL_CHECK_SUCCESS.txt" if local_check else "SUCCESS.txt"
    (output / status).write_text(
        ("LOCAL CHECK ONLY - NOT FOR SUBMISSION\n" if local_check else "Euler HW02 run completed successfully.\n")
        + "Correctness checks and output validation passed.\n"
        + "Scan sizes completed: %d\n" % len(scan_rows)
        + "CSVs: task1.csv, task2.csv, task3.csv\n"
        + "Raw outputs and compiler logs: raw/\n"
        + "Source snapshot: source/; run metadata: metadata.json\n"
        + "Download this directory and run run_hw02.sh --report DIR on local Linux to generate PDFs.\n")
    print("Completed:", output, flush=True)
    print("Status:", output / status, flush=True)



def report():
    global local_check, exponents
    metadata = json.loads((output / "metadata.json").read_text())
    local_check = metadata["run_kind"] == "local-check-not-for-submission"
    if not local_check and metadata["run_kind"] != "euler":
        raise ValueError("Unrecognized data source")
    status = "LOCAL_CHECK_SUCCESS.txt" if local_check else "SUCCESS.txt"
    if not (output / status).is_file():
        raise ValueError("The run has not completed successfully: missing " + status)
    exponents = list(range(10, 17 if local_check else 31))

    def read_csv(name):
        with (output / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    scan_rows = [[int(row["exponent"]), int(row["n"]), float(row["time_ms"]),
                  float(row["first"]), float(row["last"])] for row in read_csv("task1.csv")]
    if [row[:2] for row in scan_rows] != [[i, 1 << i] for i in exponents]:
        raise ValueError("Missing or unexpected scan sizes")
    for row in scan_rows:
        actual = numbers((raw / ("task1-2pow%d.out" % row[0])).read_text(), 3, "task1")
        if actual != row[2:] or row[2] <= 0:
            raise ValueError("Scan CSV does not match the raw output")
    convolution = numbers((raw / "task2.out").read_text(), 3, "task2")
    product = numbers((raw / "task3.out").read_text(), 9, "task3")
    if (not product[0].is_integer() or product[0] < 1000
            or any(value <= 0 for value in product[1::2])
            or not all(math.isclose(product[2], value, rel_tol=1e-12, abs_tol=1e-12)
                       for value in product[2::2])):
        raise ValueError("Invalid matrix multiplication results")
    make_pdfs(scan_rows, convolution, product, metadata)
    for name in ["task1.pdf", "assignment2.pdf"]:
        if not (output / name).read_bytes().startswith(b"%PDF-"):
            raise ValueError("Invalid PDF: " + name)
    print("Created locally:", output / "task1.pdf", "and", output / "assignment2.pdf", flush=True)


try:
    if report_only:
        report()
    else:
        main()
except Exception:
    (output / ("REPORT_FAILED.txt" if report_only else "FAILED.txt")).write_text(traceback.format_exc())
    raise
HW02_PYTHON
