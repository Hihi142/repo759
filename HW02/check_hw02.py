#!/usr/bin/env python3
"""Build HW02 and check correctness; discard timing values and temporary binaries."""

import math
from pathlib import Path
import subprocess
import tempfile


HW02 = Path(__file__).resolve().parent


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def check_output(executable, arguments, line_count):
    result = run([str(executable), *map(str, arguments)], capture_output=True, timeout=180)
    assert not result.stderr, result.stderr
    assert result.stdout.endswith("\n"), "missing trailing newline"
    lines = result.stdout.splitlines()
    assert len(lines) == line_count, f"expected {line_count} output lines: {lines}"
    values = [float(line) for line in lines]
    assert all(math.isfinite(value) for value in values), "non-finite output"
    return values


def main():
    with tempfile.TemporaryDirectory(prefix="hw02-check-") as temporary:
        build = Path(temporary)
        algorithms = [HW02 / name for name in ("scan.cpp", "convolution.cpp", "matmul.cpp")]
        harness = HW02 / "hw02_correctness.cpp"
        for label, flags in (
            ("release", ["-O3"]),
            ("sanitized", ["-O1", "-g", "-fsanitize=address,undefined",
                           "-fno-omit-frame-pointer", "-fno-pie", "-no-pie"]),
        ):
            executable = build / label
            run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
                 *flags, "-I", str(HW02), str(harness), *map(str, algorithms),
                 "-o", str(executable)])
            run([str(executable)], timeout=60)
            print(f"PASS: {label} correctness checks", flush=True)

        for task, implementation in ((1, "scan.cpp"), (2, "convolution.cpp"), (3, "matmul.cpp")):
            # Same options and translation units as the assignment commands.
            run(["g++", f"task{task}.cpp", implementation, "-Wall", "-O3", "-std=c++17",
                 "-o", str(build / f"task{task}")], cwd=HW02)

        for n in (1, 2, 17, 1024, 65537):
            _, first, last = check_output(build / "task1", [n], 3)
            assert -1.0 <= first <= 1.0 and abs(last) <= n
            if n == 1:
                assert first == last
        for n, m in ((1, 1), (1, 3), (2, 5), (4, 3), (17, 9), (32, 1)):
            check_output(build / "task2", [n, m], 3)

        invalid_arguments = {
            "task1": [[], [0], [-1], [""], ["abc"], ["1.5"], ["12x"], [1, 2],
                      ["18446744073709551616"], ["18446744073709551615"]],
            "task2": [[], [1], [0, 3], [3, 0], [-1, 3], [3, -1], [3, 2],
                      ["abc", 3], [3, "1.5"], ["", 3], [3, "3x"], [1, 1, 1],
                      ["18446744073709551616", 3], ["2147483648", 3],
                      [3, "2147483649"]],
        }
        for task, cases in invalid_arguments.items():
            for arguments in cases:
                result = subprocess.run([str(build / task), *map(str, arguments)],
                                        text=True, capture_output=True, timeout=5)
                assert result.returncode == 1 and result.stderr and not result.stdout, (
                    task, arguments, result.returncode, result.stdout, result.stderr
                )
        print("PASS: task1/task2 command lines and invalid inputs", flush=True)

        values = check_output(build / "task3", [], 9)
        assert values[0] >= 1000 and values[0].is_integer()
        last_elements = values[2::2]
        assert all(math.isclose(last_elements[0], value, rel_tol=1e-12, abs_tol=1e-12)
                   for value in last_elements)
        print("PASS: task3 at 1024 x 1024, nine-line output and matching results", flush=True)
        print("All checks passed. Timing values were discarded.", flush=True)


if __name__ == "__main__":
    main()
