"""Basic runner for the instruction count microbenchmarks.

The contents of this file are placeholders, and will be replaced by more
expressive and robust components (e.g. better runner and result display
components) in future iterations. However this allows us to exercise the
underlying benchmark generation infrastructure in the mean time.
"""

# mypy: ignore-errors
# [MODULE DOC] Placeholder command line runner for the instruction count benchmarks. As a script it
# takes a mode option, either debug (the default; runs main here) or ci (runs the CI module), and
# passes the remaining arguments to the chosen function. Import-time side effect: none beyond
# importing sibling packages, which need the benchmark directory on the import path.

import argparse
import sys

from applications import ci
from core.expand import materialize
from definitions.standard import BENCHMARKS
from execution.runner import Runner
from execution.work import WorkOrder


def main(argv: list[str]) -> None:
    # [FUNCTION DOC] Debug mode: materializes the standard benchmarks, wraps each into a work order
    # (timeout 600 seconds, two retries), runs them all with a default Runner, then prints the
    # label, auto labels, thread count and instruction counts of every order in order. argv is
    # unused. Returns None.
    work_orders = tuple(
        WorkOrder(label, autolabels, timer_args, timeout=600, retries=2)
        for label, autolabels, timer_args in materialize(BENCHMARKS)
    )

    results = Runner(work_orders).run()
    for work_order in work_orders:
        print(
            work_order.label,
            work_order.autolabels,
            work_order.timer_args.num_threads,
            results[work_order].instructions,
        )


if __name__ == "__main__":
    modes = {
        "debug": main,
        "ci": ci.main,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=list(modes.keys()), default="debug")

    args, remaining_args = parser.parse_known_args(sys.argv)
    modes[args.mode](remaining_args[1:])
