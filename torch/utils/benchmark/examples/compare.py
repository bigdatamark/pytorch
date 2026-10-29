# mypy: allow-untyped-defs
# [MODULE DOC]
# Runnable example of the Timer and Compare APIs. Builds many timers over sizes, thread counts
# and fake "branches" (emulated via a wrapper that adds artificial per-element sleep), measures
# them, round-trips the measurements through serialization, and prints an unformatted and a
# formatted comparison table. Running as a script calls main; no import-time side effects.
"""Example of Timer and Compare APIs:

$ python -m examples.compare
"""

import pickle
import sys
import time

import torch

import torch.utils.benchmark as benchmark_utils


class FauxTorch:
    # [CLASS DOC] HELPER: Wraps the real torch module and exposes add, mul, cat and matmul that
    # call the real functions and then sleep in proportion to the result size, to fake a slower
    # version of the library. State: the wrapped module and the extra nanoseconds per element.
    """Emulate different versions of pytorch.

    In normal circumstances this would be done with multiple processes
    writing serialized measurements, but this simplifies that model to
    make the example clearer.
    """
    def __init__(self, real_torch, extra_ns_per_element) -> None:
        # [FUNCTION DOC] HELPER: Stores the real torch module and the per-element delay.
        self._real_torch = real_torch
        self._extra_ns_per_element = extra_ns_per_element

    def extra_overhead(self, result):
        # [FUNCTION DOC] HELPER: Sleeps for the element count times the per-element delay when
        # the result has more than 5000 elements, then returns the result unchanged.
        # time.sleep has a ~65 us overhead, so only fake a
        # per-element overhead if numel is large enough.
        numel = int(result.numel())
        if numel > 5000:
            time.sleep(numel * self._extra_ns_per_element * 1e-9)
        return result

    def add(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Real add followed by the artificial delay.
        return self.extra_overhead(self._real_torch.add(*args, **kwargs))

    def mul(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Real mul followed by the artificial delay.
        return self.extra_overhead(self._real_torch.mul(*args, **kwargs))

    def cat(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Real cat followed by the artificial delay.
        return self.extra_overhead(self._real_torch.cat(*args, **kwargs))

    def matmul(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Real matmul followed by the artificial delay.
        return self.extra_overhead(self._real_torch.matmul(*args, **kwargs))


def main() -> None:
    # [FUNCTION DOC] Builds timers for every combination of fake branch, task, tensor size and
    # thread count, runs each twice with a short blocked autorange, serializes and deserializes
    # the measurements, and prints the Compare table both raw and with trimmed significant
    # figures and colors. Output goes to standard output; takes no arguments, returns nothing.
    tasks = [
        ("add", "add", "torch.add(x, y)"),
        ("add", "add (extra +0)", "torch.add(x, y + zero)"),
    ]

    serialized_results = []
    repeats = 2
    timers = [
        benchmark_utils.Timer(
            stmt=stmt,
            globals={
                "torch": torch if branch == "master" else FauxTorch(torch, overhead_ns),
                "x": torch.ones((size, 4)),
                "y": torch.ones((1, 4)),
                "zero": torch.zeros(()),
            },
            label=label,
            sub_label=sub_label,
            description=f"size: {size}",
            env=branch,
            num_threads=num_threads,
        )
        for branch, overhead_ns in [("master", None), ("my_branch", 1), ("severe_regression", 5)]
        for label, sub_label, stmt in tasks
        for size in [1, 10, 100, 1000, 10000, 50000]
        for num_threads in [1, 4]
    ]

    for i, timer in enumerate(timers * repeats):
        serialized_results.append(pickle.dumps(
            timer.blocked_autorange(min_run_time=0.05)
        ))
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()
    print()

    comparison = benchmark_utils.Compare([
        pickle.loads(i) for i in serialized_results
    ])

    print("== Unformatted " + "=" * 80 + "\n" + "/" * 95 + "\n")
    comparison.print()

    print("== Formatted " + "=" * 80 + "\n" + "/" * 93 + "\n")
    comparison.trim_significant_figures()
    comparison.colorize()
    comparison.print()


if __name__ == "__main__":
    main()
