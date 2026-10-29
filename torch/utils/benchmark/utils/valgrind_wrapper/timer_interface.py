# [MODULE DOC] Bridge between Timer and Valgrind/Callgrind. Defines result containers
# (FunctionCount, FunctionCounts, CallgrindStats), the globals serialization helpers
# (Serialization, CopyIfCallgrind, GlobalsBridge), and a private wrapper that spawns a
# valgrind subprocess running a generated script (or compiled C++ template) and parses
# callgrind_annotate output. A lazily created singleton wrapper is exposed through
# wrapper_singleton. Exports FunctionCount, FunctionCounts, CallgrindStats, CopyIfCallgrind.
# No import-time side effects beyond imports and defining module constants.
"""Intermediate layer between `Timer` and `valgrind`."""
from __future__ import annotations

import collections
import enum
import dataclasses
import itertools as it
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
from typing import (
    cast, Any, NamedTuple,
    TYPE_CHECKING)

import torch
from torch.utils.benchmark.utils import common, cpp_jit
import operator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from torch.utils.benchmark.utils._stubs import CallgrindModuleType


__all__ = ["FunctionCount", "FunctionCounts", "CallgrindStats", "CopyIfCallgrind"]


if TYPE_CHECKING:
    CompletedProcessType = subprocess.CompletedProcess[str]
else:
    CompletedProcessType = subprocess.CompletedProcess


class FunctionCount(NamedTuple):
    # [CLASS DOC] One Callgrind row: an instruction count and the function name
    # (file and function text) it belongs to.
    # TODO(#105471): Rename the count field
    count: int  # type: ignore[assignment]
    function: str


@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class FunctionCounts:
    # [CLASS DOC] Immutable, tuple-like collection of FunctionCount rows. Fields: the
    # data tuple, an inclusive flag (counts include callees), whether repr truncates
    # long output, and an optional line width override for tests. Supports iteration,
    # length, indexing/slicing, addition, subtraction, scaling, filter, transform,
    # sum and denoise. Merging requires matching inclusive flags.
    """Container for manipulating Callgrind results.

    It supports:
        1) Addition and subtraction to combine or diff results.
        2) Tuple-like indexing.
        3) A `denoise` function which strips CPython calls which are known to
           be non-deterministic and quite noisy.
        4) Two higher order methods (`filter` and `transform`) for custom
           manipulation.
    """
    _data: tuple[FunctionCount, ...]
    inclusive: bool
    truncate_rows: bool = True

    # For normal use, torch._tensor_str.PRINT_OPTS.linewidth determines
    # the print settings. This is simply to allow hermetic unit tests.
    _linewidth: int | None = None

    def __iter__(self) -> Iterator[FunctionCount]:
        # [FUNCTION DOC] Yields each FunctionCount row in stored order.
        yield from self._data

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the number of rows.
        return len(self._data)

    def __getitem__(self, item: Any) -> FunctionCount | FunctionCounts:
        # [FUNCTION DOC] Returns a single FunctionCount for an integer index, or a new
        # FunctionCounts (same inclusive flag, no row truncation in repr) for a slice.
        data: FunctionCount | tuple[FunctionCount, ...] = self._data[item]
        return (
            FunctionCounts(cast(tuple[FunctionCount, ...], data), self.inclusive, truncate_rows=False)
            if isinstance(data, tuple) else data
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC] Formats the default object repr followed by one right-aligned
        # count and function name per row. Long names are shortened in the middle to fit
        # the line width (tensor print width unless overridden). When truncating, shows
        # only the first and last 9 rows if over 18. Exclusive counts get a Total line.
        count_len = 0
        for c, _ in self:
            # Account for sign in string length.
            count_len = max(count_len, len(str(c)) + int(c < 0))

        lines = []
        linewidth = self._linewidth or torch._tensor_str.PRINT_OPTS.linewidth
        fn_str_len = max(linewidth - count_len - 4, 40)
        for c, fn in self:
            if len(fn) > fn_str_len:
                left_len = int((fn_str_len - 5) // 2)
                fn = fn[:left_len] + " ... " + fn[-(fn_str_len - left_len - 5):]
            lines.append(f"  {c:>{count_len}}  {fn}")

        if self.truncate_rows and len(lines) > 18:
            lines = lines[:9] + ["...".rjust(count_len + 2)] + lines[-9:]

        if not self.inclusive:
            lines.extend(["", f"Total: {self.sum()}"])

        return "\n".join([super().__repr__()] + lines)

    def __add__(
        self,
        other: FunctionCounts,
    ) -> FunctionCounts:
        # [FUNCTION DOC] Returns the per-function sum of two counts objects. Raises an
        # error if the inclusive flags differ.
        return self._merge(other, lambda c: c)

    def __sub__(
        self,
        other: FunctionCounts,
    ) -> FunctionCounts:
        # [FUNCTION DOC] Returns the per-function difference (self minus other). Raises
        # an error if the inclusive flags differ; zero results are dropped.
        return self._merge(other, operator.neg)

    def __mul__(self, other: int | float) -> FunctionCounts:
        # [FUNCTION DOC] Returns a new object with every count multiplied by other and
        # truncated to an integer; rows that become zero are dropped.
        return self._from_dict({
            fn: int(c * other) for c, fn in self._data
        }, self.inclusive)

    def transform(self, map_fn: Callable[[str], str]) -> FunctionCounts:
        # [FUNCTION DOC] Maps every function name through map_fn, summing counts of names
        # that collide, and returns the new sorted FunctionCounts.
        """Apply `map_fn` to all of the function names.

        This can be used to regularize function names (e.g. stripping irrelevant
        parts of the file path), coalesce entries by mapping multiple functions
        to the same name (in which case the counts are added together), etc.
        """
        counts: collections.defaultdict[str, int] = collections.defaultdict(int)
        for c, fn in self._data:
            counts[map_fn(fn)] += c

        return self._from_dict(counts, self.inclusive)

    def filter(self, filter_fn: Callable[[str], bool]) -> FunctionCounts:
        # [FUNCTION DOC] Returns a new FunctionCounts with only rows whose function name
        # makes filter_fn return true; order and inclusive flag are preserved.
        """Keep only the elements where `filter_fn` applied to function name returns True."""
        return FunctionCounts(tuple(i for i in self if filter_fn(i.function)), self.inclusive)

    def sum(self) -> int:
        # [FUNCTION DOC] Returns the total of all counts.
        return sum(c for c, _ in self)

    def denoise(self) -> FunctionCounts:
        # [FUNCTION DOC] Returns a copy without rows whose name mentions the CPython
        # unicode dictionary lookup function, a known source of nondeterministic counts.
        """Remove known noisy instructions.

        Several instructions in the CPython interpreter are rather noisy. These
        instructions involve unicode to dictionary lookups which Python uses to
        map variable names. FunctionCounts is generally a content agnostic
        container, however this is sufficiently important for obtaining
        reliable results to warrant an exception."""
        return self.filter(lambda fn: "dictobject.c:lookdict_unicode" not in fn)

    def _merge(
        self,
        second: FunctionCounts,
        merge_fn: Callable[[int], int]
    ) -> FunctionCounts:
        # [FUNCTION DOC] HELPER: Sums counts per function name, applying merge_fn to the
        # counts from second (identity to add, negation to subtract). Raises an error if
        # the inclusive flags differ. Returns a new FunctionCounts.
        if self.inclusive != second.inclusive:
            raise AssertionError("Cannot merge inclusive and exclusive counts.")
        counts: collections.defaultdict[str, int] = collections.defaultdict(int)
        for c, fn in self:
            counts[fn] += c

        for c, fn in second:
            counts[fn] += merge_fn(c)

        return self._from_dict(counts, self.inclusive)

    @staticmethod
    def _from_dict(counts: dict[str, int], inclusive: bool) -> FunctionCounts:
        # [FUNCTION DOC] HELPER: Builds a FunctionCounts from a name-to-count mapping,
        # dropping zero counts and sorting rows by count in descending order.
        flat_counts = (FunctionCount(c, fn) for fn, c in counts.items() if c)
        return FunctionCounts(tuple(sorted(flat_counts, reverse=True)), inclusive)


@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class CallgrindStats:
    # [CLASS DOC] Immutable result of one Callgrind collection. Holds the task spec,
    # iterations per run, whether PyTorch has debug symbols, inclusive and exclusive
    # FunctionCounts for both the statement and the empty baseline, and optionally the
    # raw callgrind output text. Key methods: stats, counts, delta, as_standardized.
    """Top level container for Callgrind results collected by Timer.

    Manipulation is generally done using the FunctionCounts class, which is
    obtained by calling `CallgrindStats.stats(...)`. Several convenience
    methods are provided as well; the most significant is
    `CallgrindStats.as_standardized()`.
    """
    task_spec: common.TaskSpec
    number_per_run: int
    built_with_debug_symbols: bool
    baseline_inclusive_stats: FunctionCounts
    baseline_exclusive_stats: FunctionCounts
    stmt_inclusive_stats: FunctionCounts
    stmt_exclusive_stats: FunctionCounts
    stmt_callgrind_out: str | None

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a summary: default repr, task spec summary, statement and
        # baseline instruction totals (raw and denoised), runs per measurement and thread
        # count, plus a warning if PyTorch lacks debug symbols.
        base_stats = self.baseline_exclusive_stats
        output = f"""
{super().__repr__()}
{self.task_spec.summarize()}
  {'':>25}All{'':>10}Noisy symbols removed
    Instructions: {self.counts(denoise=False):>12}{'':>15}{self.counts(denoise=True):>12}
    Baseline:     {base_stats.sum():>12}{'':>15}{base_stats.denoise().sum():>12}
{self.number_per_run} runs per measurement, {self.task_spec.num_threads} thread{'s' if self.task_spec.num_threads > 1 else ''}
""".strip()
        if not self.built_with_debug_symbols:
            output += textwrap.dedent("""
            Warning: PyTorch was not built with debug symbols.
                     Source information may be limited. Rebuild with
                     REL_WITH_DEB_INFO=1 for more detailed results.""")
        return output

    def stats(self, inclusive: bool = False) -> FunctionCounts:
        # [FUNCTION DOC] Returns the statement's inclusive or exclusive FunctionCounts
        # depending on the flag.
        """Returns detailed function counts.

        Conceptually, the FunctionCounts returned can be thought of as a tuple
        of (count, path_and_function_name) tuples.

        `inclusive` matches the semantics of callgrind. If True, the counts
        include instructions executed by children. `inclusive=True` is useful
        for identifying hot spots in code; `inclusive=False` is useful for
        reducing noise when diffing counts from two different runs. (See
        CallgrindStats.delta(...) for more details)
        """
        return self.stmt_inclusive_stats if inclusive else self.stmt_exclusive_stats

    def counts(self, *, denoise: bool = False) -> int:
        # [FUNCTION DOC] Returns the total exclusive instruction count of the statement,
        # optionally after removing noisy symbols.
        """Returns the total number of instructions executed.

        See `FunctionCounts.denoise()` for an explanation of the `denoise` arg.
        """
        stats = self.stmt_exclusive_stats
        return (stats.denoise() if denoise else stats).sum()

    # FIXME: Once 3.7 is the minimum version, type annotate `other` per PEP 563
    def delta(
        self,
        other: CallgrindStats,
        inclusive: bool = False,
    ) -> FunctionCounts:
        # [FUNCTION DOC] Returns the difference between this object's and other's
        # statement counts (inclusive or exclusive per the flag), as a FunctionCounts.
        """Diff two sets of counts.

        One common reason to collect instruction counts is to determine the
        effect that a particular change will have on the number of instructions
        needed to perform some unit of work. If a change increases that number, the
        next logical question is "why". This generally involves looking at what part
        if the code increased in instruction count. This function automates that
        process so that one can easily diff counts on both an inclusive and
        exclusive basis.
        """
        return self.stats(inclusive=inclusive) - other.stats(inclusive=inclusive)

    def as_standardized(self) -> CallgrindStats:
        # [FUNCTION DOC] Returns a new CallgrindStats whose function names have build-dir
        # prefixes, CPython path prefixes and trailing library names removed, so profiles
        # from different builds diff cleanly. The raw callgrind output is dropped (set to
        # none) since it no longer matches the names.
        """Strip library names and some prefixes from function strings.

        When comparing two different sets of instruction counts, on stumbling
        block can be path prefixes. Callgrind includes the full filepath
        when reporting a function (as it should). However, this can cause
        issues when diffing profiles. If a key component such as Python
        or PyTorch was built in separate locations in the two profiles, which
        can result in something resembling::

            23234231 /tmp/first_build_dir/thing.c:foo(...)
             9823794 /tmp/first_build_dir/thing.c:bar(...)
              ...
               53453 .../aten/src/Aten/...:function_that_actually_changed(...)
              ...
             -9823794 /tmp/second_build_dir/thing.c:bar(...)
            -23234231 /tmp/second_build_dir/thing.c:foo(...)

        Stripping prefixes can ameliorate this issue by regularizing the
        strings and causing better cancellation of equivalent call sites
        when diffing.
        """
        def strip(stats: FunctionCounts) -> FunctionCounts:
            # [FUNCTION DOC] HELPER: Applies the prefix and library-name regex
            # substitutions in order to every function name and returns the result.
            transforms = (
                # PyTorch may have been built in different locations.
                (r"^.+build/\.\./", "build/../"),
                (r"^.+/" + re.escape("build/aten/"), "build/aten/"),

                # "Python" and "Objects" come from CPython.
                (r"^.+/" + re.escape("Python/"), "Python/"),
                (r"^.+/" + re.escape("Objects/"), "Objects/"),

                # Strip library name. e.g. `libtorch.so`
                (r"\s\[.+\]$", ""),
            )

            for before, after in transforms:
                stats = stats.transform(lambda fn: re.sub(before, after, fn))

            return stats

        return CallgrindStats(
            task_spec=self.task_spec,
            number_per_run=self.number_per_run,
            built_with_debug_symbols=self.built_with_debug_symbols,
            baseline_inclusive_stats=strip(self.baseline_inclusive_stats),
            baseline_exclusive_stats=strip(self.baseline_exclusive_stats),
            stmt_inclusive_stats=strip(self.stmt_inclusive_stats),
            stmt_exclusive_stats=strip(self.stmt_exclusive_stats),

            # `as_standardized` will change symbol names, so the contents will
            # no longer map directly to `callgrind.out`
            stmt_callgrind_out=None,
        )


class Serialization(enum.Enum):
    # [CLASS DOC] HELPER: How a global is transferred to the Valgrind subprocess:
    # pickle, torch save/load, or TorchScript save/load.
    PICKLE = 0
    TORCH = 1
    TORCH_JIT = 2


_GLOBALS_ALLOWED_TYPES: dict[Serialization, tuple[Any, ...]] = {
    Serialization.PICKLE: (str, bytes, bool, int, float, complex),
    Serialization.TORCH_JIT: (torch.jit.ScriptFunction, torch.jit.ScriptModule),
    Serialization.TORCH: (torch.nn.Module,),
}


class CopyIfCallgrind:
    # [CLASS DOC] Wrapper marking a global as safe to replace by a deserialized copy
    # during Callgrind collection. Holds the value, an optional setup string for the
    # subprocess, and the Serialization method chosen from the value's type (builtin
    # scalars, strings and bytes, TorchScript objects, or nn.Modules).
    """Signal that a global may be replaced with a deserialized copy.

    See `GlobalsBridge` for why this matters.
    """
    def __init__(self, value: Any, *, setup: str | None = None) -> None:
        # [FUNCTION DOC] Picks the serialization method whose allowed types match value
        # and stores value, setup and method. Raises ValueError listing supported types
        # if the value type is not allowed.
        # KNOWN ISSUE: the supported-type list is built by chaining over the dict values
        # (tuples), so it prints the tuples' reprs rather than each type name.
        for method, supported_types in _GLOBALS_ALLOWED_TYPES.items():
            if any(isinstance(value, t) for t in supported_types):
                self._value: Any = value
                self._setup: str | None = setup
                self._serialization: Serialization = method
                break
        else:
            supported_str = "\n".join([
                getattr(t, "__name__", repr(t))
                for t in it.chain(_GLOBALS_ALLOWED_TYPES.values())])

            raise ValueError(
                f"Unsupported type: {type(value)}\n"
                f"`collect_callgrind` restricts globals to the following types:\n"
                f"{textwrap.indent(supported_str, '  ')}"
            )

    @property
    def value(self) -> Any:
        # [FUNCTION DOC] Returns the wrapped value.
        return self._value

    @property
    def setup(self) -> str | None:
        # [FUNCTION DOC] Returns the optional setup code run in the subprocess before loading.
        return self._setup

    @property
    def serialization(self) -> Serialization:
        # [FUNCTION DOC] Returns the Serialization method chosen for the value.
        return self._serialization

    @staticmethod
    def unwrap_all(globals: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC] Returns a new mapping where each CopyIfCallgrind entry is
        # replaced by its wrapped value and other entries are unchanged.
        return {
            k: (v.value if isinstance(v, CopyIfCallgrind) else v)
            for k, v in globals.items()
        }


class GlobalsBridge:
    # [CLASS DOC] Prepares user globals for the Valgrind subprocess. Holds the validated
    # mapping of names to CopyIfCallgrind wrappers and a data directory. construct writes
    # each value to disk and returns source code lines that reload them in the subprocess.
    """Handle the transfer of (certain) globals when collecting Callgrind statistics.

    Key takeaway: Any globals passed must be wrapped in `CopyIfCallgrind` to
                  work with `Timer.collect_callgrind`.

    Consider the following code snippet:
    ```
        import pickle
        import timeit

        class Counter:
            value = 0

            def __call__(self):
                self.value += 1

        counter = Counter()
        timeit.Timer("counter()", globals={"counter": counter}).timeit(10)
        print(counter.value)  # 10

        timeit.Timer(
            "counter()",
            globals={"counter": pickle.loads(pickle.dumps(counter))}
        ).timeit(20)
        print(counter.value)  # Still 10
    ```

    In the first case, `stmt` is executed using the objects in `globals`;
    however, the addition of serialization and deserialization changes the
    semantics and may meaningfully change behavior.

    This is a practical consideration when collecting Callgrind statistics.
    Unlike `exec` based execution (which `timeit` uses under the hood) which
    can share in-memory data structures with the caller, Callgrind collection
    requires an entirely new process in order to run under Valgrind. This means
    that any data structures used for statement execution will have to be
    serialized and deserialized in the subprocess.

    In order to avoid surprising semantics from (user invisible) process
    boundaries, what can be passed through `globals` is severely restricted
    for `Timer.collect_callgrind`. It is expected that most setup should be
    achievable (albeit perhaps less ergonomically) by passing a `setup`
    string.

    There are, however, exceptions. One such class are TorchScripted functions.
    Because they require a concrete file with source code it is not possible
    to define them using a `setup` string. Another group are torch.nn.Modules,
    whose construction can be complex and prohibitively cumbersome to coerce
    into a `setup` string. Finally, most builtin types are sufficiently well
    behaved and sufficiently common to warrant allowing as well. (e.g.
    `globals={"n": 1}` is very convenient.)

    Fortunately, all have well defined serialization semantics. This class
    is responsible for enabling the Valgrind subprocess to use elements in
    `globals` so long as they are an allowed type.

    Caveats:
        The user is required to acknowledge this serialization by wrapping
        elements in `globals` with `CopyIfCallgrind`.

        While ScriptFunction and ScriptModule are expected to save and load
        quite robustly, it is up to the user to ensure that an nn.Module can
        un-pickle successfully.

        `torch.Tensor` and `np.ndarray` are deliberately excluded. The
        serialization/deserialization process perturbs the representation of a
        tensor in ways that could result in incorrect measurements. For example,
        if a tensor lives in pinned CPU memory, this fact would not be preserved
        by a dump, and that will in turn change the performance of certain CUDA
        operations.
    """

    def __init__(self, globals: dict[str, Any], data_dir: str) -> None:
        # [FUNCTION DOC] Validates globals and keeps the wrapped entries, skipping torch
        # and the builtins entry. Creates data_dir if it does not exist (side effect).
        # Raises ValueError if torch is mocked or if any other entry is not wrapped in
        # CopyIfCallgrind.
        self._globals: dict[str, CopyIfCallgrind] = {}
        self._data_dir = data_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if globals.get("torch", torch) is not torch:
            raise ValueError("`collect_callgrind` does not support mocking out `torch`.")

        for name, value in globals.items():
            if name in ("torch", "__builtins__"):
                # Torch will be imported by the collection script, and
                # __builtins__ is added by Timer.
                continue

            if not isinstance(value, CopyIfCallgrind):
                raise ValueError(
                    "`collect_callgrind` requires that globals be wrapped in "
                    "`CopyIfCallgrind` so that serialization is explicit."
                )

            self._globals[name] = value

    def construct(self) -> str:
        # [FUNCTION DOC] Serializes every global into data_dir (pickle file, torch save
        # file or TorchScript save file by method) and returns newline-joined source
        # text that runs each setup string and then loads the value in the subprocess.
        # Raises NotImplementedError for an unknown serialization method. Writes files.
        load_lines = []
        for name, wrapped_value in self._globals.items():
            if wrapped_value.setup is not None:
                load_lines.append(textwrap.dedent(wrapped_value.setup))

            if wrapped_value.serialization == Serialization.PICKLE:
                path = os.path.join(self._data_dir, f"{name}.pkl")
                load_lines.append(
                    f"with open({repr(path)}, 'rb') as f:\n    {name} = pickle.load(f)")
                with open(path, "wb") as f:
                    pickle.dump(wrapped_value.value, f)

            elif wrapped_value.serialization == Serialization.TORCH:
                path = os.path.join(self._data_dir, f"{name}.pt")
                # TODO: Figure out if we can use torch.serialization.add_safe_globals here
                # Using weights_only=False after the change in
                # https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573
                load_lines.append(f"{name} = torch.load({repr(path)}, weights_only=False)")
                torch.save(wrapped_value.value, path)

            elif wrapped_value.serialization == Serialization.TORCH_JIT:
                path = os.path.join(self._data_dir, f"{name}.pt")
                load_lines.append(f"{name} = torch.jit.load({repr(path)})")
                with open(path, "wb") as f:
                    torch.jit.save(wrapped_value.value, f)  # type: ignore[no-untyped-call]

            else:
                raise NotImplementedError(
                    f"Unknown serialization method: {wrapped_value.serialization}")

        return "\n".join(load_lines)


class _ValgrindWrapper:
    # [CLASS DOC] HELPER: Runs Callgrind collection. State: optional JIT-compiled bindings
    # module, whether the platform supports valgrind, availability of the valgrind tool
    # commands, and the PyTorch build type. Methods: collect_callgrind (public entry),
    # _invoke (subprocess run and parsing) and _construct_script (script generation).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Detects Callgrind bindings in the torch extension module,
        # otherwise JIT-compiles them (prints a message). Records platform support, checks
        # the required commands on supported platforms via a subprocess, and extracts the
        # build type from the torch config. Raises an error if JIT bindings lack symbols.
        self._bindings_module: CallgrindModuleType | None = None
        valgrind_symbols = (
            "_valgrind_supported_platform",
            "_valgrind_toggle",
            "_valgrind_toggle_and_dump_stats",
        )
        if all(hasattr(torch._C, symbol) for symbol in valgrind_symbols):
            self._supported_platform: bool = torch._C._valgrind_supported_platform()

        else:
            print("Callgrind bindings are not present in `torch._C`. JIT-ing bindings.")
            self._bindings_module = cpp_jit.get_compat_bindings()
            if not all(hasattr(self._bindings_module, symbol) for symbol in valgrind_symbols):
                raise AssertionError("JIT-compiled callgrind bindings are missing required symbols")
            self._supported_platform = self._bindings_module._valgrind_supported_platform()

        self._commands_available: dict[str, bool] = {}
        if self._supported_platform:
            # Only bother checking on supported platforms.
            for cmd in ("valgrind", "callgrind_control", "callgrind_annotate"):
                self._commands_available[cmd] = not subprocess.run(
                    ["which", cmd],
                    capture_output=True,
                    check=False,
                ).returncode

        self._build_type: str | None = None
        build_search = re.search("BUILD_TYPE=(.+),", torch.__config__.show())  # type: ignore[no-untyped-call]
        if build_search is not None:
            self._build_type = build_search.groups()[0].split(",")[0]

    def _validate(self) -> None:
        # [FUNCTION DOC] HELPER: Raises OSError if the platform is unsupported or if any
        # required command (valgrind, callgrind_control, callgrind_annotate) is missing.
        if not self._supported_platform:
            raise OSError("Valgrind is not supported on this platform.")

        missing_cmds = [cmd for cmd, available in self._commands_available.items() if not available]
        if missing_cmds:
            raise OSError("Missing: " + ", ".join(missing_cmds))

    def collect_callgrind(
        self,
        task_spec: common.TaskSpec,
        globals: dict[str, Any],
        *,
        number: int,
        repeats: int,
        collect_baseline: bool,
        is_python: bool,
        retain_out_file: bool,
    ) -> tuple[CallgrindStats, ...]:
        # [FUNCTION DOC] HELPER: Validates the environment, runs the collection through
        # _invoke, splits off the baseline (last) result, and returns one CallgrindStats
        # per repeat that shares the baseline. Raises OSError if unsupported and an error
        # if a baseline is requested for non-Python timers or the result count is wrong.
        """Collect stats, and attach a reference run which can be used to filter interpreter overhead."""
        self._validate()
        if not is_python and collect_baseline:
            raise AssertionError("collect_baseline is only supported for Python timers")

        *task_stats, baseline_stats = self._invoke(
            task_spec=task_spec,
            globals=globals,
            number=number,
            repeats=repeats,
            collect_baseline=collect_baseline,
            is_python=is_python,
            retain_out_file=retain_out_file,
        )
        if len(task_stats) != repeats:
            raise AssertionError("Unexpected number of task stats returned from _invoke")

        return tuple(
            CallgrindStats(
                task_spec=task_spec,
                number_per_run=number,
                built_with_debug_symbols=self._build_type == "RelWithDebInfo",
                baseline_inclusive_stats=baseline_stats[0],
                baseline_exclusive_stats=baseline_stats[1],
                stmt_inclusive_stats=stmt_inclusive_stats,
                stmt_exclusive_stats=stmt_exclusive_stats,
                stmt_callgrind_out=out_contents,
            )
            for stmt_inclusive_stats, stmt_exclusive_stats, out_contents in task_stats
        )

    def _invoke(
        self,
        *,
        task_spec: common.TaskSpec,
        globals: dict[str, Any],
        number: int,
        repeats: int,
        collect_baseline: bool,
        is_python: bool,
        retain_out_file: bool,
    ) -> tuple[tuple[FunctionCounts, FunctionCounts, str | None], ...]:
        # [FUNCTION DOC] HELPER: Creates a temp directory, then for Python writes a
        # generated script (copying JIT bindings if used) or for C++ compiles a template,
        # runs it under valgrind's callgrind tool, and parses callgrind_annotate output.
        # Returns, for each repeat plus a baseline slot, (inclusive, exclusive, optional
        # raw output); the baseline is empty when not collected. Raises OSError if
        # valgrind fails (with the error log) and an error for bad arguments or parse
        # failures. The temp directory is always removed.
        """Core invocation method for Callgrind collection.

        Valgrind operates by effectively replacing the CPU with an emulated
        version which allows it to instrument any code at the cost of severe
        performance degradation. This has the practical effect that in order
        to collect Callgrind statistics, a new process has to be created
        running under `valgrind`. The steps for this process are:

        1) Create a scratch directory.
        2) Codegen a run script. (_ValgrindWrapper._construct_script)
            Inside the run script:
                * Validate that Python and torch match the parent process
                * Validate that it is indeed running under valgrind
                * Execute `setup` and warm up `stmt`
                * Begin collecting stats
                * Run the `stmt` loop
                * Stop collecting stats
        3) Parse the run results.
        4) Cleanup the scratch directory.
        """
        working_dir = common._make_temp_dir(prefix="callgrind")
        data_dir = os.path.join(working_dir, "data")
        script_file = os.path.join(working_dir, "timer_callgrind.py")
        callgrind_out = os.path.join(working_dir, "callgrind.out")
        error_log = os.path.join(working_dir, "error.txt")
        stat_log = os.path.join(working_dir, "callgrind_stat.txt")
        stdout_stderr_log = os.path.join(working_dir, "stdout_stderr.log")

        def run(args: list[str], **kwargs: Any) -> tuple[CompletedProcessType, str]:
            # [FUNCTION DOC] HELPER: Runs a subprocess with combined output redirected to a
            # log file (avoiding pipe deadlocks). Returns the completed process and the
            # log text; extra keyword arguments go to the subprocess call.
            # https://thraxil.org/users/anders/posts/2008/03/13/Subprocess-Hanging-PIPE-is-your-enemy/
            with open(stdout_stderr_log, "wb") as f_stdout_stderr:
                invocation = subprocess.run(
                    args,
                    stdout=f_stdout_stderr,
                    stderr=subprocess.STDOUT,
                    **kwargs,
                )
                with open(stdout_stderr_log) as f:
                    return invocation, f.read()

        try:
            if is_python:
                if self._bindings_module is not None:
                    shutil.copy(
                        self._bindings_module.__file__,
                        os.path.join(working_dir, os.path.split(self._bindings_module.__file__)[1])
                    )

                script_file = os.path.join(working_dir, "timer_callgrind.py")
                with open(script_file, "w") as f:
                    f.write(self._construct_script(
                        task_spec,
                        globals=GlobalsBridge(globals, data_dir),
                        number=number,
                        repeats=repeats,
                        collect_baseline=collect_baseline,
                        error_log=error_log,
                        stat_log=stat_log,
                        bindings=self._bindings_module))

                run_loop_cmd = ["python", script_file]
            else:
                if collect_baseline:
                    raise AssertionError("collect_baseline must be False for non-Python timers")
                run_loop_exec = cpp_jit.compile_callgrind_template(
                    stmt=task_spec.stmt,
                    setup=task_spec.setup,
                    global_setup=task_spec.global_setup,
                )
                run_loop_cmd = [
                    run_loop_exec,
                    "--number", str(number),
                    "--number-warmup", str(min(number, 10)),
                    "--repeats", str(repeats),
                    "--number-threads", str(task_spec.num_threads),
                ]

            valgrind_invocation, valgrind_invocation_output = run([
                "valgrind",
                "--tool=callgrind",
                f"--callgrind-out-file={callgrind_out}",
                "--dump-line=yes",
                "--dump-instr=yes",
                "--instr-atstart=yes",
                "--collect-atstart=no",
            ] + run_loop_cmd)

            if valgrind_invocation.returncode:
                error_report = ""
                if os.path.exists(error_log):
                    with open(error_log) as f:
                        error_report = f.read()
                if not error_report:
                    error_report = "Unknown error.\n" + valgrind_invocation_output

                raise OSError(f"Failed to collect callgrind profile:\n{error_report}")

            def parse_output(fpath: str, inclusive: bool) -> FunctionCounts:
                # [FUNCTION DOC] HELPER: Runs callgrind_annotate on a profile file and
                # scans its text with a small state machine: find the program total, find
                # the column header, then read count/function rows, skipping rows equal to
                # the total and separator lines, stopping at the first other line.
                # Returns a sorted FunctionCounts. Raises an error if parsing never starts.
                _annotate_invocation, annotate_invocation_output = run([
                    "callgrind_annotate",
                    f"--inclusive={'yes' if inclusive else 'no'}",
                    "--threshold=100",
                    "--show-percs=no",
                    fpath
                ], check=True)

                total_pattern = re.compile(r"^([0-9,]+)\s+PROGRAM TOTALS")
                begin_pattern = re.compile(r"Ir\s+file:function")
                function_pattern = re.compile(r"^\s*([0-9,]+)\s+(.+:.+)$")

                class ScanState(enum.Enum):
                    # [CLASS DOC] HELPER: Parser phases: looking for the totals line,
                    # looking for the column header, and reading function rows.
                    SCANNING_FOR_TOTAL = 0
                    SCANNING_FOR_START = 1
                    PARSING = 2

                scan_state = ScanState.SCANNING_FOR_TOTAL
                fn_counts = []
                for l in annotate_invocation_output.splitlines(keepends=False):
                    if scan_state == ScanState.SCANNING_FOR_TOTAL:
                        total_match = total_pattern.match(l)
                        if total_match:
                            program_totals = int(total_match.groups()[0].replace(",", ""))
                            scan_state = ScanState.SCANNING_FOR_START

                    elif scan_state == ScanState.SCANNING_FOR_START:
                        if begin_pattern.match(l):
                            scan_state = ScanState.PARSING

                    else:
                        if scan_state != ScanState.PARSING:
                            raise AssertionError("Failed to enter PARSING state while parsing callgrind_annotate output")
                        fn_match = function_pattern.match(l)
                        if fn_match:
                            ir_str, file_function = fn_match.groups()
                            ir = int(ir_str.replace(",", ""))
                            if ir == program_totals:  # type: ignore[possibly-undefined]
                                # Callgrind includes some top level red herring symbols when
                                # a program dumps multiple profiles.
                                continue
                            fn_counts.append(FunctionCount(ir, file_function))

                        elif re.match(r"-+", l):
                            # Ignore heading separator lines.
                            continue

                        else:
                            break

                if scan_state != ScanState.PARSING:
                    raise AssertionError(f"Failed to parse {fpath}")
                return FunctionCounts(tuple(sorted(fn_counts, reverse=True)), inclusive=inclusive)

            def read_results(i: int) -> tuple[FunctionCounts, FunctionCounts, str | None]:
                # [FUNCTION DOC] HELPER: Returns the inclusive counts, exclusive counts and
                # optional raw text for output file number i (one-indexed on disk). The
                # extra baseline slot without baseline collection yields empty counts.
                if i == repeats and not collect_baseline:
                    # Null baseline.
                    return (
                        FunctionCounts((), inclusive=True),
                        FunctionCounts((), inclusive=False),
                        None,
                    )

                fpath = f"{callgrind_out}.{i + 1}"  # Callgrind one-indexes files.
                callgrind_out_contents: str | None = None
                if retain_out_file:
                    with open(fpath) as f:
                        callgrind_out_contents = f.read()

                return (
                    parse_output(fpath, inclusive=True),
                    parse_output(fpath, inclusive=False),
                    callgrind_out_contents
                )

            return tuple(read_results(i) for i in range(repeats + 1))
        finally:
            shutil.rmtree(working_dir)

    @staticmethod
    def _construct_script(
        task_spec: common.TaskSpec,
        globals: GlobalsBridge,
        *,
        number: int,
        repeats: int,
        collect_baseline: bool,
        error_log: str,
        stat_log: str,
        bindings: CallgrindModuleType | None,
    ) -> str:
        # [FUNCTION DOC] HELPER: Returns the source of the Python script run under
        # valgrind. It verifies the interpreter and torch match the parent, loads globals,
        # runs setup and a short warmup, confirms callgrind is active, then for each repeat
        # toggles collection around a partially unrolled statement loop and dumps stats;
        # optionally measures an empty baseline. Failures are written to the error log.
        def block_stmt(stmt: str, indent: int = 0) -> str:
            # [FUNCTION DOC] HELPER: Builds the statement loop text: blocks of 100 copies
            # inside a loop plus the remainder copies (no loop when only one block),
            # indented by the given amount.
            """Partially unroll benchmark loop.

            The naive template looks something like:
                "for _ in range({number}): {stmt}"

            However a loop in Python is surprisingly expensive, and significantly
            increases the number of background Python instructions. So instead we
            partially unroll the loops, with a block size of 100 chosen to keep
            the instruction overhead from `range` low while also not ballooning
            the size of the generated file.
            """
            block_size = 100
            loop_count = number // block_size
            if loop_count == 1:
                # There is no point in having `for _ in range(1): ...` rather
                # than just `...`, and this lets us save shave a few background
                # instructions.
                loop_count = 0
            remainder = number - block_size * loop_count
            blocked_stmt = ""

            if loop_count:
                unrolled_stmts = textwrap.indent("\n".join([stmt] * block_size), " " * 4)
                blocked_stmt += f"for _ in range({loop_count}):\n{unrolled_stmts}\n"

            if remainder:
                blocked_stmt += "\n".join([stmt] * remainder)

            return textwrap.indent(blocked_stmt, " " * indent)

        pass_baseline = (
            "callgrind_bindings._valgrind_toggle()\n"
            f"{block_stmt('pass')}\n"
            "callgrind_bindings._valgrind_toggle_and_dump_stats()"
        )

        return textwrap.dedent(r"""
            import gc
            import os
            import pickle
            import subprocess
            import sys
            import time

            # Mitigate https://github.com/pytorch/pytorch/issues/37377
            # which can sometimes cause the subprocess call to fail.
            import numpy as np

            import torch
            torch.set_num_threads({num_threads})

            {bindings_import}

            PID = os.getpid()

            def log_failure(msg):
                with open({error_log_repr}, "wt") as f:
                    f.write(msg)
                sys.exit(1)

            def check_result(completed_process):
                if completed_process.returncode:
                    log_failure(f"Command failed: {{' '.join(completed_process.args)}}")
                return completed_process

            # =============================================================================
            # == Check that subprocess matches parent =====================================
            # =============================================================================
            if os.path.realpath(sys.executable) != "{parent_interpreter}":
                log_failure(
                    "Interpreter mismatch:\n"
                    f"  {{os.path.realpath(sys.executable)}}\n    vs.\n  {parent_interpreter}"
                )

            if torch.__file__ != "{torch_file}":
                log_failure(
                    "PyTorch does not match expected file:\n"
                    f"  {{torch.__file__}}\n    vs.\n  {torch_file}"
                )

            # =============================================================================
            # == User specified setup =====================================================
            # =============================================================================
            # Load serialized globals
            {load_globals}

            # User setup str
            {setup}

            for _ in range({warmup_number}):
            {indented_stmt}

            # =============================================================================
            # == Callgrind management =====================================================
            # =============================================================================
            with open("{stat_log}", "wb") as stat_file:
                # If many instances of callgrind are running at once, the output of
                # `callgrind_control` may exceed 16kb which would cause `subprocess.PIPE`
                # to deadlock. So instead we use a file.
                callgrind_stat = check_result(subprocess.run(
                    ["callgrind_control", "--stat"],
                    stdout=stat_file,
                    stderr=subprocess.STDOUT,
                ))

            with open("{stat_log}", "rt") as stat_file:
                stat_lines = stat_file.read().splitlines()

            if f"PID {{PID}}: python {{__file__}}" not in stat_lines:
                log_failure("Process does not appear to be running callgrind.")

            gc.collect()
            time.sleep(0.01)

            # =============================================================================
            # == User code block ==========================================================
            # =============================================================================
            for _ in range({repeats}):
                callgrind_bindings._valgrind_toggle()
            {blocked_stmt}
                callgrind_bindings._valgrind_toggle_and_dump_stats()
                gc.collect()

            {baseline}
        """).strip().format(
            indented_stmt=textwrap.indent(task_spec.stmt, " " * 4),
            blocked_stmt=block_stmt(task_spec.stmt, indent=4),
            baseline=(pass_baseline if collect_baseline else ""),
            number=number,
            repeats=repeats,
            load_globals=globals.construct(),
            setup=task_spec.setup,
            warmup_number=min(number, 10),
            num_threads=task_spec.num_threads,
            error_log_repr=repr(error_log),
            stat_log=stat_log,
            parent_interpreter=os.path.realpath(sys.executable),
            torch_file=torch.__file__,
            bindings_import=(
                "import torch._C as callgrind_bindings" if bindings is None
                else f"import {bindings.__name__} as callgrind_bindings"),
        )


CALLGRIND_SINGLETON: _ValgrindWrapper | None = None
def wrapper_singleton() -> _ValgrindWrapper:
    # [FUNCTION DOC] Returns the process-wide _ValgrindWrapper, creating it on first call
    # (global state; creation may print a message and JIT-compile bindings).
    global CALLGRIND_SINGLETON
    if CALLGRIND_SINGLETON is None:
        CALLGRIND_SINGLETON = _ValgrindWrapper()
    return CALLGRIND_SINGLETON
