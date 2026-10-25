# [MODULE DOC] Profiler data model for Dynamo tracing time. Defines records for
# per-inlined-function timings and stack entries, and DynamoProfilerState which
# collects timings and converts them into a cProfile-compatible pstats object,
# an optional SVG call graph (via external gprof2dot and graphviz), and printed
# reports. No import-time side effects.
"""
Dynamo Profiler - tracks where Dynamo spends time during compilation.

This module provides profiling functionality for Dynamo tracing, showing per-function
cumtime (inclusive) and tottime (exclusive) in a cProfile-compatible format.
The output can be visualized with tools like snakeviz.

Usage:
    # Enable via config (prints pstats output):
    torch._dynamo.config.dynamo_profiler = True

    # Or save to file for snakeviz:
    torch._dynamo.config.dynamo_profiler = "/tmp/dynamo.prof"
    # Then: snakeviz /tmp/dynamo.prof
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import pstats


@dataclass
class FunctionTraceTiming:
    # [CLASS DOC] Dataclass record of one traced (inlined) function call:
    # identity (name, file, first line), inclusive and exclusive time in
    # nanoseconds, bytecode count, inline depth, optional caller identity,
    # whether the call is non-recursive (primitive), and the full call stack as
    # a tuple of (name, file, line) entries. Properties give millisecond views
    # and pstats-style keys.
    """
    Timing data for a single inlined function trace.

    Follows cProfile conventions:
    - cumtime: total time in function including all subcalls (inclusive)
    - tottime: time in function excluding subcalls (exclusive)
    - caller info: who called this function (for building call graph)
    """

    # Function identification
    func_name: str
    filename: str
    firstlineno: int
    # Timing data (in nanoseconds) - cProfile-style
    cumtime_ns: int  # Inclusive time (includes subcalls)
    tottime_ns: int  # Exclusive time (excludes subcalls)
    # Code stats (for comparing tracing overhead vs function complexity)
    bytecode_count: int
    # Nesting depth when this function was traced
    inline_depth: int
    # Caller information (for building call graph edges)
    caller_func_name: str | None = None
    caller_filename: str | None = None
    caller_firstlineno: int | None = None
    # Whether this is a primitive (non-recursive) call
    # A call is primitive if the function doesn't appear anywhere in the call stack
    is_primitive_call: bool = True
    # Full call stack at the time of this call (for proper snakeviz drill-down)
    # Each entry is (func_name, filename, firstlineno)
    call_stack: tuple[tuple[str, str, int], ...] = ()

    # Backwards compatibility alias
    @property
    def trace_time_ns(self) -> int:
        # [FUNCTION DOC] Backwards-compatible alias returning the inclusive
        # time in nanoseconds.
        return self.cumtime_ns

    @property
    def trace_time_ms(self) -> float:
        # [FUNCTION DOC] Inclusive time converted to milliseconds.
        return self.cumtime_ns / 1e6

    @property
    def cumtime_ms(self) -> float:
        # [FUNCTION DOC] Inclusive time (including subcalls) in milliseconds.
        return self.cumtime_ns / 1e6

    @property
    def tottime_ms(self) -> float:
        # [FUNCTION DOC] Exclusive time (excluding subcalls) in milliseconds.
        return self.tottime_ns / 1e6

    @property
    def caller_key(self) -> tuple[str, int, str] | None:
        # [FUNCTION DOC] Return the caller as a (filename, line, name) tuple,
        # substituting empty text / zero for missing file or line, or None if
        # there is no caller name.
        """Return caller as a pstats-compatible key tuple."""
        if self.caller_func_name is not None:
            return (
                self.caller_filename or "",
                self.caller_firstlineno or 0,
                self.caller_func_name,
            )
        return None

    @property
    def func_key(self) -> tuple[str, int, str]:
        # [FUNCTION DOC] Return this function as a (filename, line, name) tuple.
        """Return this function as a pstats-compatible key tuple."""
        return (self.filename, self.firstlineno, self.func_name)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Human-readable summary with name, location, cumulative
        # and total times in milliseconds, bytecode count and depth.
        return (
            f"FunctionTraceTiming({self.func_name} at {self.filename}:{self.firstlineno}, "
            f"cumtime={self.cumtime_ms:.2f}ms, tottime={self.tottime_ms:.2f}ms, "
            f"bytecode={self.bytecode_count}, depth={self.inline_depth})"
        )


@dataclass
class ProfilerStackEntry:
    # [CLASS DOC] Dataclass for one live frame on the profiler stack: function
    # identity, start time (ns), accumulated time of traced children (ns), and
    # whether the call is primitive (not already on the stack).
    """Stack entry for tracking function timing in the Dynamo profiler."""

    func_name: str
    filename: str
    firstlineno: int
    start_time_ns: int
    child_time_ns: int  # Accumulated time spent in traced children
    is_primitive_call: bool = True  # Whether this is a non-recursive call


class DynamoProfilerState:
    # [CLASS DOC] Mutable profiler state: timings (list of finished
    # FunctionTraceTiming records) and stack (list of ProfilerStackEntry for
    # in-progress traces). Methods maintain the stack, record results, and
    # export them as pstats, SVG, or printed output.
    """State for Dynamo profiler tracking function trace timings."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Create empty timings and stack lists.
        self.timings: list[FunctionTraceTiming] = []
        self.stack: list[ProfilerStackEntry] = []

    def record_timing(self, timing: FunctionTraceTiming) -> None:
        # [FUNCTION DOC] Append a finished timing record to the timings list.
        """Record timing data for a traced function."""
        self.timings.append(timing)

    def get_timings(self) -> list[FunctionTraceTiming]:
        # [FUNCTION DOC] Return the internal timings list itself (not a copy).
        """Get all recorded timings."""
        return self.timings

    def push(
        self, func_name: str, filename: str, firstlineno: int, start_time_ns: int
    ) -> None:
        # [FUNCTION DOC] Push a stack entry for a function starting at the given
        # time. The entry is marked primitive only if no entry with the same
        # name, file and first line is already on the stack; child time starts
        # at zero. Mutates the stack.
        """Push a new entry onto the timing stack."""
        # Check if this function already exists in the stack (indirect recursion)
        is_primitive = not any(
            entry.func_name == func_name
            and entry.filename == filename
            and entry.firstlineno == firstlineno
            for entry in self.stack
        )
        self.stack.append(
            ProfilerStackEntry(
                func_name=func_name,
                filename=filename,
                firstlineno=firstlineno,
                start_time_ns=start_time_ns,
                child_time_ns=0,
                is_primitive_call=is_primitive,
            )
        )

    def pop(self) -> ProfilerStackEntry | None:
        # [FUNCTION DOC] Remove and return the top stack entry, or None if the
        # stack is empty.
        """Pop the top entry from the timing stack."""
        if self.stack:
            return self.stack.pop()
        return None

    def add_child_time(self, child_cumtime_ns: int) -> None:
        # [FUNCTION DOC] Add a finished child's inclusive time (ns) to the top
        # stack entry's child-time accumulator; no-op on an empty stack.
        """Add the child's cumulative time to the parent's child_time accumulator."""
        if self.stack:
            self.stack[-1].child_time_ns += child_cumtime_ns

    def get_current_caller(self) -> tuple[str, str, int] | None:
        # [FUNCTION DOC] Return (name, file, first line) of the top stack entry,
        # or None if the stack is empty.
        """Get the current caller (top of stack) as (func_name, filename, firstlineno)."""
        if self.stack:
            entry = self.stack[-1]
            return (entry.func_name, entry.filename, entry.firstlineno)
        return None

    def get_call_stack(self) -> tuple[tuple[str, str, int], ...]:
        # [FUNCTION DOC] Return the whole stack, bottom to top, as a tuple of
        # (name, file, first line) tuples.
        """Get the full current call stack as tuple of (func_name, filename, firstlineno)."""
        return tuple(
            (entry.func_name, entry.filename, entry.firstlineno) for entry in self.stack
        )

    def generate_pstats(
        self, output_file: str | None = None, print_raw: bool = False
    ) -> pstats.Stats:
        # [FUNCTION DOC] Aggregate recorded timings per function (call count,
        # primitive-call count, total time, cumulative time counted only for
        # primitive calls) and per caller edge (edge cumulative time counts all
        # calls, for call-graph tools). Caller-only functions get zeroed
        # top-level entries. Builds a pstats.Stats from a dummy profile and
        # overwrites its stats and totals. Side effects: optionally prints a
        # raw aggregate table (print_raw), writes the stats file when
        # output_file is given and logs a message. Returns the Stats object.
        """Generate pstats.Stats object from recorded timings.

        Args:
            output_file: Optional file path to save the stats.
            print_raw: If True, print raw aggregate timings before returning.
        """
        import cProfile
        import io
        import logging
        import pstats

        log = logging.getLogger(__name__)

        # Aggregate by (filename, lineno, func_name)
        aggregated: dict[tuple[str, int, str], dict[str, Any]] = {}
        # caller_edges[callee_key][caller_key] -> edge stats
        caller_edges: dict[
            tuple[str, int, str], dict[tuple[str, int, str], dict[str, Any]]
        ] = {}

        for t in self.timings:
            key = (t.filename, t.firstlineno, t.func_name)

            if key not in aggregated:
                aggregated[key] = {
                    "ncalls": 0,
                    "pcalls": 0,
                    "tottime": 0.0,
                    "cumtime": 0.0,
                }
                caller_edges[key] = {}

            agg = aggregated[key]
            agg["ncalls"] += 1
            agg["tottime"] += t.tottime_ns / 1e9

            if t.is_primitive_call:
                agg["pcalls"] += 1
                agg["cumtime"] += t.cumtime_ns / 1e9

            # Build caller edge
            if t.caller_filename is not None:
                caller_key = (
                    t.caller_filename,
                    t.caller_firstlineno or 0,
                    t.caller_func_name or "",
                )
                if caller_key not in caller_edges[key]:
                    caller_edges[key][caller_key] = {
                        "ncalls": 0,
                        "pcalls": 0,
                        "tottime": 0.0,
                        "cumtime": 0.0,
                    }
                edge = caller_edges[key][caller_key]
                edge["ncalls"] += 1
                edge["tottime"] += t.tottime_ns / 1e9
                # Always add cumtime to edges for visualization (gprof2dot)
                # Function-level cumtime is already correct (only primitive calls)
                edge["cumtime"] += t.cumtime_ns / 1e9
                if t.is_primitive_call:
                    edge["pcalls"] += 1

        if print_raw:
            sorted_items = sorted(
                aggregated.items(), key=lambda x: x[1]["cumtime"], reverse=True
            )
            print("\n=== Aggregate Timings (raw) ===")
            print(
                f"{'ncalls':>8} {'pcalls':>8} {'tottime':>12} {'cumtime':>12}  function"
            )
            print("-" * 80)
            total_cumtime = 0.0
            total_tottime = 0.0
            for (filename, lineno, func_name), agg in sorted_items:
                ncalls = agg["ncalls"]
                pcalls = agg["pcalls"]
                tottime = agg["tottime"] * 1000  # convert to ms
                cumtime = agg["cumtime"] * 1000
                total_cumtime += cumtime
                total_tottime += tottime
                short_file = filename.split("/")[-1] if "/" in filename else filename
                print(
                    f"{ncalls:>8} {pcalls:>8} {tottime:>10.2f}ms {cumtime:>10.2f}ms  "
                    f"{func_name} ({short_file}:{lineno})"
                )
            print("-" * 80)
            print(
                f"Total timings: {len(self.timings)}, unique functions: {len(aggregated)}"
            )
            print(
                f"Sum tottime: {total_tottime:.2f}ms, Sum cumtime: {total_cumtime:.2f}ms"
            )

        # Ensure caller-only functions have a top-level entry.
        # gprof2dot expects every function referenced as a caller to also
        # exist as a top-level entry in the stats dict with timing data.
        for key in list(caller_edges.keys()):
            for caller_key in caller_edges[key]:
                if caller_key not in aggregated:
                    aggregated[caller_key] = {
                        "ncalls": 0,
                        "pcalls": 0,
                        "tottime": 0.0,
                        "cumtime": 0.0,
                    }
                    caller_edges[caller_key] = {}

        # Build the stats dict in pstats format
        stats_dict: dict[
            tuple[str, int, str], tuple[int, int, float, float, dict[Any, Any]]
        ] = {}

        for key, agg in aggregated.items():
            callers: dict[tuple[str, int, str], tuple[int, int, float, float]] = {}
            for caller_key, edge in caller_edges[key].items():
                callers[caller_key] = (
                    edge["ncalls"],
                    edge["pcalls"],
                    edge["tottime"],
                    edge["cumtime"],
                )

            stats_dict[key] = (
                agg["pcalls"],
                agg["ncalls"],
                agg["tottime"],
                agg["cumtime"],
                callers,
            )

        # Create a pstats.Stats object
        dummy_profile = cProfile.Profile()
        dummy_profile.enable()
        dummy_profile.disable()
        stats = pstats.Stats(dummy_profile, stream=io.StringIO())

        stats.stats = stats_dict  # type: ignore[attr-defined]
        stats.total_calls = sum(s[1] for s in stats_dict.values())  # type: ignore[attr-defined]
        stats.prim_calls = sum(s[0] for s in stats_dict.values())  # type: ignore[attr-defined]
        stats.total_tt = sum(s[2] for s in stats_dict.values())  # type: ignore[attr-defined]

        if output_file:
            stats.dump_stats(output_file)
            log.info(
                "Saved pstats to %s. Visualize with: snakeviz %s",
                output_file,
                output_file,
            )

        return stats

    def generate_svg(
        self, profile_file: str, svg_file: str | None = None
    ) -> str | None:
        # [FUNCTION DOC] Render a call-graph SVG from a pstats file by piping
        # gprof2dot output into graphviz dot via subprocesses. The output path
        # defaults to the profile path with an svg extension. Prints a message
        # and returns None if either tool is missing, a process fails, or the
        # file is not created, or on any exception; otherwise prints and returns
        # the SVG path.
        """Generate an SVG call graph from a profile file using gprof2dot and graphviz.

        Args:
            profile_file: Path to the pstats profile file.
            svg_file: Optional path for the output SVG. If not provided, uses
                profile_file with .svg extension.

        Returns:
            Path to the generated SVG file, or None if generation failed.
        """
        import os
        import shutil
        import subprocess

        if not shutil.which("gprof2dot"):
            print("gprof2dot not found. Install with: pip install gprof2dot")
            return None

        if not shutil.which("dot"):
            print("graphviz 'dot' not found. Install graphviz package.")
            return None

        if svg_file is None:
            svg_file = profile_file.rsplit(".", 1)[0] + ".svg"

        try:
            # gprof2dot -f pstats profile.prof | dot -Tsvg -o profile.svg
            gprof2dot = subprocess.Popen(
                [
                    "gprof2dot",
                    "-f",
                    "pstats",
                    "--node-label=total-time-percentage",
                    "--node-label=self-time-percentage",
                    "--node-label=total-time",
                    profile_file,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            dot = subprocess.Popen(
                ["dot", "-Tsvg", "-o", svg_file],
                stdin=gprof2dot.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            gprof2dot.stdout.close()  # type: ignore[union-attr]
            _, dot_err = dot.communicate()
            _, gprof2dot_err = gprof2dot.communicate()

            if gprof2dot.returncode != 0:
                print(f"gprof2dot failed: {gprof2dot_err.decode()}")
                return None

            if dot.returncode != 0:
                print(f"graphviz dot failed: {dot_err.decode()}")
                return None

            if not os.path.isfile(svg_file):
                print(f"SVG file was not created: {svg_file}")
                return None

            print(f"SVG call graph saved to: {svg_file}")
            return svg_file

        except Exception as e:
            print(f"Failed to generate SVG: {e}")
            return None

    def dump_stats(
        self, output_file: str | None = None, generate_svg: bool = True
    ) -> None:
        # [FUNCTION DOC] Print profiler results to stdout: does nothing if no
        # timings were recorded; otherwise builds pstats (with raw table), prints
        # it sorted by cumulative time, and if output_file is given saves it,
        # prints visualization hints, and optionally generates an SVG.
        """Print profiler stats to stdout and optionally save to file.

        Args:
            output_file: Optional path to save the pstats profile.
            generate_svg: If True and output_file is provided, also generate an SVG
                call graph using gprof2dot and graphviz.
        """
        import sys

        if not self.timings:
            return

        stats = self.generate_pstats(output_file, print_raw=True)
        print("\n=== Dynamo Profiler (pstats) ===")
        stats.stream = sys.stdout  # type: ignore[attr-defined]
        stats.sort_stats("cumulative").print_stats()

        if output_file:
            print(f"\nProfile saved to: {output_file}")
            print(f"Visualize with: snakeviz {output_file}")

            if generate_svg:
                self.generate_svg(output_file)
