# [MODULE DOC] Dynamo profiling: metric aggregation classes (ProfileMetrics, ProfileResult),
# the Profiler that analyzes torch profiler events to compute captured vs total operator
# counts and time, helpers for the TORCHDYNAMO_PRINT_MISSING environment variable, and
# fx_insert_profiling which wraps a graph module call in a profiler record region. No
# import-time side effects beyond imports.
"""
Dynamo profiling implementation.

This module provides profiling functionality for Dynamo, including:
- ProfileMetrics: Class for collecting and aggregating performance metrics like
  execution time, operator counts, and fusion statistics
- ProfileResult: Class for analyzing and reporting profiling results
- Utilities for tracking missed/uncaptured operations
- Functions for instrumenting FX graphs with profiling capabilities

The profiler helps measure and optimize the performance of Dynamo-compiled code
by tracking both captured and total operations, timing, and graph statistics.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any
from typing_extensions import Self

import torch

from .utils import print_once


@dataclasses.dataclass
class ProfileMetrics:
    # [CLASS DOC] Dataclass of aggregate profile numbers: microseconds, operators, fusions
    # and graphs (all default zero). Supports addition and element-wise division to form
    # ratios (e.g. captured over total); also formats as percentages or CSV.
    microseconds: float = 0.0
    operators: int = 0
    fusions: int = 0
    graphs: int = 0

    def __iadd__(self, other: Self) -> Self:
        # [FUNCTION DOC] In-place add of microseconds, operators and fusions from other;
        # returns self.
        # KNOWN ISSUE: the graphs field is not accumulated.
        self.microseconds += other.microseconds
        self.operators += other.operators
        self.fusions += other.fusions
        return self

    def __add__(self, other: ProfileMetrics) -> ProfileMetrics:
        # [FUNCTION DOC] Returns a new metrics object with summed microseconds, operators and
        # fusions. Raises AssertionError if other is not a ProfileMetrics.
        # KNOWN ISSUE: the graphs field is dropped (result has zero graphs).
        if not isinstance(other, ProfileMetrics):
            raise AssertionError(f"Expected ProfileMetrics, got {type(other)}")
        return ProfileMetrics(
            self.microseconds + other.microseconds,
            self.operators + other.operators,
            self.fusions + other.fusions,
        )

    def __truediv__(self, other: Any) -> ProfileMetrics:
        # [FUNCTION DOC] Element-wise ratio of microseconds, operators and fusions to those of
        # other, with each denominator floored at one to avoid division by zero. An int other
        # is first expanded to a metrics object with that value in the three fields. Returns
        # a new metrics object (graphs left at zero).
        if isinstance(other, int):
            other = ProfileMetrics(other, other, other)
        return ProfileMetrics(
            self.microseconds / max(1, other.microseconds),
            # pyrefly: ignore [bad-argument-type]
            self.operators / max(1, other.operators),
            # pyrefly: ignore [bad-argument-type]
            self.fusions / max(1, other.fusions),
        )

    def __str__(self) -> str:
        # [FUNCTION DOC] Formats operators and microseconds fields as percentages.
        return f"{self.operators:4.0%} ops {self.microseconds:4.0%} time"

    def tocsv(self) -> list[float]:
        # [FUNCTION DOC] Returns a list of the operators value followed by the microseconds.
        return [self.operators, self.microseconds]


class ProfileResult:
    # [CLASS DOC] Holds captured and total ProfileMetrics plus the unique graph count for a
    # profiling run; supports accumulation, percent computation, string and CSV output.
    def __init__(
        self, captured: ProfileMetrics, total: ProfileMetrics, unique_graphs: int
    ) -> None:
        # [FUNCTION DOC] Stores captured and total metrics (falling back to empty metrics if
        # falsy) and the unique graph count.
        self.captured: ProfileMetrics = captured or ProfileMetrics()
        self.total: ProfileMetrics = total or ProfileMetrics()
        self.unique_graphs: int = unique_graphs

    def __iadd__(self, other: Self) -> Self:
        # [FUNCTION DOC] Accumulates other's captured, total and unique graph count into
        # self; returns self.
        self.captured += other.captured
        self.total += other.total
        self.unique_graphs += other.unique_graphs
        return self

    def percent(self) -> ProfileMetrics:
        # [FUNCTION DOC] Returns captured divided by total, i.e. the captured fractions.
        return self.captured / self.total

    def __str__(self) -> str:
        # [FUNCTION DOC] Human-readable summary: graph count, graph calls, captured/total
        # operator counts and the percent string.
        return (
            f"{self.unique_graphs:2} graphs {self.captured.graphs:2} graph calls "
            f"{self.captured.operators:4}/{self.total.operators:4} = "
            + str(self.percent())
        )

    def tocsv(self) -> list[Any]:
        # [FUNCTION DOC] CSV row: unique graphs, graph calls, captured ops, total ops,
        # followed by the percent values.
        return [
            self.unique_graphs,
            self.captured.graphs,
            self.captured.operators,
            self.total.operators,
        ] + self.percent().tocsv()


def should_print_missing() -> bool:
    # [FUNCTION DOC] True when the environment variable TORCHDYNAMO_PRINT_MISSING equals 1.
    return os.environ.get("TORCHDYNAMO_PRINT_MISSING") == "1"


def print_missing(stack: list[str]) -> None:
    # [FUNCTION DOC] Prints once a "MISSING" message with the last three relevant stack
    # entries of an uncaptured op, after filtering out built-in and installed torch frames.
    # Does nothing if the stack involves the autograd profiler.
    if any("/torch/autograd/profiler.py" in x for x in stack):
        return
    stack = [
        x for x in stack if ("<built-in" not in x and "site-packages/torch/" not in x)
    ]
    print_once("MISSING", " >> ".join(stack[-3:]))


class Profiler:
    # [CLASS DOC] Wraps a CPU torch profiler. Class attribute unique_graphs counts graphs
    # instrumented since the last results() call (global state, reset by results).
    unique_graphs: int = 0

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates a CPU-activity torch profiler, recording stacks only when
        # missing-op printing is enabled.
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_stack=should_print_missing(),
        )

    def results(self) -> ProfileResult:
        # [FUNCTION DOC] Sorts profiler events by start time and walks them: TORCHDYNAMO
        # events mark captured regions; top-level ops ending within the latest region count
        # as captured, others are totals (and printed if enabled); nested ops are ignored.
        # Adjusts total ops for profiler setup ops. Reads and resets the class unique_graphs
        # counter. Returns a ProfileResult.
        captured_regions = 0
        captured_ops = 0
        captured_microseconds = 0
        total_ops = 0
        total_microseconds = 0

        last_op_end_time = -1
        captured_region_end_time = -1
        events = sorted(self.prof.events(), key=lambda x: x.time_range.start)
        for e in events:
            if e.name == "TORCHDYNAMO":
                captured_region_end_time = e.time_range.end
                captured_regions += 1
                # ignore `handle = torch.zeros(1)` in record_function.__init__()
                total_ops -= 1
            elif e.time_range.start >= last_op_end_time:
                last_op_end_time = e.time_range.end
                if e.time_range.end <= captured_region_end_time:
                    captured_ops += 1
                    captured_microseconds += e.time_range.elapsed_us()
                elif should_print_missing():
                    print_missing(e.stack)
                total_ops += 1
                total_microseconds += e.time_range.elapsed_us()
            else:
                pass  # ops recursively called from other ops (ignored)

        unique_graphs = Profiler.unique_graphs
        Profiler.unique_graphs = 0
        # we counted one extra op that is part of the profiler setup code
        total_ops -= 1

        return ProfileResult(
            captured=ProfileMetrics(
                microseconds=captured_microseconds,
                operators=captured_ops,
                fusions=captured_ops - captured_regions,
                graphs=captured_regions,
            ),
            total=ProfileMetrics(
                microseconds=total_microseconds,
                operators=total_ops,
                fusions=total_ops - 1,
            ),
            unique_graphs=unique_graphs,
        )


def fx_insert_profiling(gm: torch.fx.GraphModule, example_inputs: list[Any]) -> Any:
    # [FUNCTION DOC] Backend-style compiler: increments the global unique graph count and
    # returns a wrapper that runs the graph module's forward inside a profiler record
    # region named TORCHDYNAMO. example_inputs is unused.
    def _wrapped(*args: Any) -> Any:
        # [FUNCTION DOC] HELPER: runs gm.forward on args within the TORCHDYNAMO region.
        with torch.profiler.record_function("TORCHDYNAMO"):
            return gm.forward(*args)

    Profiler.unique_graphs += 1
    return _wrapped
