# [MODULE DOC]
# Smallest runnable example of the Timer API: times a broadcasting add with both timeit and
# blocked autorange, three rounds, printing the results. No import-time side effects.
"""Trivial use of Timer API:

$ python -m examples.simple_timeit
"""

import torch

import torch.utils.benchmark as benchmark_utils


def main() -> None:
    # [FUNCTION DOC] Creates one labeled Timer for an add of a 4x8 and a 1x8 tensor, then three
    # times prints a ten-thousand-iteration timeit result and an autorange result. Prints to
    # standard output; returns nothing.
    timer = benchmark_utils.Timer(
        stmt="x + y",
        globals={"x": torch.ones((4, 8)), "y": torch.ones((1, 8))},
        label="Broadcasting add (4x8)",
    )

    for i in range(3):
        print(f"Run: {i}\n{'-' * 40}")
        print(f"timeit:\n{timer.timeit(10000)}\n")
        print(f"autorange:\n{timer.blocked_autorange()}\n\n")


if __name__ == "__main__":
    main()
