# mypy: ignore-errors
# [MODULE DOC] Utilities for the instruction count benchmark: a lazily created temporary directory
# that is deleted at interpreter exit, flattening of nested benchmark definitions into a mapping
# keyed by tuples of strings, and a parser for side-by-side Python and C++ statement tables. Holds
# one module global caching the temp directory path.
import atexit
import re
import shutil
import textwrap

from core.api import GroupedBenchmark, TimerArgs
from core.types import Definition, FlatIntermediateDefinition, Label

from torch.utils.benchmark.utils.common import _make_temp_dir


_TEMPDIR: str | None = None


def get_temp_dir() -> str:
    # [FUNCTION DOC] Returns the path of a process-wide temporary directory, creating it on first
    # use (with an instruction count name prefix, also cleaning stale shared-memory directories)
    # and registering removal of the whole tree at interpreter exit. Later calls return the cached
    # path.
    global _TEMPDIR
    if _TEMPDIR is None:
        _TEMPDIR = _make_temp_dir(
            prefix="instruction_count_microbenchmarks", gc_dev_shm=True
        )
        atexit.register(shutil.rmtree, path=_TEMPDIR)
    return _TEMPDIR


def _flatten(
    key_prefix: Label, sub_schema: Definition, result: FlatIntermediateDefinition
) -> None:
        # [FUNCTION DOC] HELPER: Recursive worker of flatten. Walks the nested mapping, turning
        # each key into a tuple (a tuple key must contain only strings, None adds nothing, a string
        # becomes a one-element tuple) appended to the accumulated prefix. Leaf values (timer
        # arguments or grouped benchmarks) are stored in the result mapping, which is mutated;
        # nested mappings are recursed into. Raises AssertionError for bad key types, duplicate
        # flat keys or values that are neither leaves nor mappings.
    for k, value in sub_schema.items():
        if isinstance(k, tuple):
            if not all(isinstance(ki, str) for ki in k):
                raise AssertionError(
                    f"expected all elements of key tuple to be str, got {k}"
                )
            key_suffix: Label = k
        elif k is None:
            key_suffix = ()
        else:
            if not isinstance(k, str):
                raise AssertionError(f"expected key to be str, got {type(k)}")
            key_suffix = (k,)

        key: Label = key_prefix + key_suffix
        if isinstance(value, (TimerArgs, GroupedBenchmark)):
            if key in result:
                raise AssertionError(f"duplicate key: {key}")
            result[key] = value
        else:
            if not isinstance(value, dict):
                raise AssertionError(f"expected value to be dict, got {type(value)}")
            _flatten(key_prefix=key, sub_schema=value, result=result)


def flatten(schema: Definition) -> FlatIntermediateDefinition:
    # [FUNCTION DOC] Flattens a nested benchmark definition into a mapping from tuple keys to timer
    # arguments or grouped benchmarks (see the types module for the schema). Runs a final
    # validation that every key is a tuple of strings and every value is a leaf type, raising
    # AssertionError otherwise. Returns the new mapping.
    """See types.py for an explanation of nested vs. flat definitions."""
    result: FlatIntermediateDefinition = {}
    _flatten(key_prefix=(), sub_schema=schema, result=result)

    # Ensure that we produced a valid flat definition.
    for k, v in result.items():
        if not isinstance(k, tuple):
            raise AssertionError(f"expected key to be tuple, got {type(k)}")
        if not all(isinstance(ki, str) for ki in k):
            raise AssertionError(
                f"expected all elements of key tuple to be str, got {k}"
            )
        if not isinstance(v, (TimerArgs, GroupedBenchmark)):
            raise AssertionError(
                f"expected value to be TimerArgs or GroupedBenchmark, got {type(v)}"
            )
    return result


def parse_stmts(stmts: str) -> tuple[str, str]:
    # [FUNCTION DOC] Parses a side-by-side table of Python and C++ code into two code strings. The
    # text is dedented and stripped and needs at least three lines (AssertionError otherwise). The
    # first line must be the header with a 40 column Python column, a separator bar and C++
    # (ValueError otherwise); the second a separator line of dashes (AssertionError); each
    # following line is split at a fixed 40 character left column and a bar, with the right part
    # optional (ValueError for a line that does not fit). Each line must round trip back to the
    # same text, else AssertionError. Returns the Python lines and the C++ lines, each joined with
    # newlines.
    """Helper function for side-by-side Python and C++ stmts.

    For more complex statements, it can be useful to see Python and C++ code
    side by side. To this end, we provide an **extremely restricted** way
    to define Python and C++ code side-by-side. The schema should be mostly
    self explanatory, with the following non-obvious caveats:
      - Width for the left (Python) column MUST be 40 characters.
      - The column separator is " | ", not "|". Whitespace matters.
    """
    stmts = textwrap.dedent(stmts).strip()
    lines: list[str] = stmts.splitlines(keepends=False)
    if len(lines) < 3:
        raise AssertionError(f"Invalid string (expected at least 3 lines):\n{stmts}")

    column_header_pattern = r"^Python\s{35}\| C\+\+(\s*)$"
    signature_pattern = r"^: f\((.*)\)( -> (.+))?\s*$"  # noqa: F841
    separation_pattern = r"^[-]{40} | [-]{40}$"
    code_pattern = r"^(.{40}) \|($| (.*)$)"

    column_match = re.search(column_header_pattern, lines[0])
    if column_match is None:
        raise ValueError(
            f"Column header `{lines[0]}` "
            f"does not match pattern `{column_header_pattern}`"
        )

    if not re.search(separation_pattern, lines[1]):
        raise AssertionError(
            f"Separation line `{lines[1]}` does not match pattern `{separation_pattern}`"
        )

    py_lines: list[str] = []
    cpp_lines: list[str] = []
    for l in lines[2:]:
        l_match = re.search(code_pattern, l)
        if l_match is None:
            raise ValueError(f"Invalid line `{l}`")
        py_lines.append(l_match.groups()[0])
        cpp_lines.append(l_match.groups()[2] or "")

        # Make sure we can round trip for correctness.
        l_from_stmts = f"{py_lines[-1]:<40} | {cpp_lines[-1]:<40}".rstrip()
        if l_from_stmts != l.rstrip():
            raise AssertionError(f"Failed to round trip `{l}`")

    return "\n".join(py_lines), "\n".join(cpp_lines)
