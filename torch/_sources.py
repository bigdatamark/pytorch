# mypy: allow-untyped-defs
# [MODULE DOC]
# Helpers for TorchScript to obtain and parse Python function source: source retrieval, indentation
# normalization, SourceContext metadata and parse_def which yields the AST. No import-time side effects.
import ast
import functools
import inspect
from textwrap import dedent
from typing import Any, NamedTuple

from torch._C import ErrorReport
from torch._C._jit_tree_views import SourceRangeFactory


def get_source_lines_and_file(
    obj: Any,
    error_msg: str | None = None,
) -> tuple[list[str], int, str | None]:
    # [FUNCTION DOC] Returns (source lines, starting line number, filename) for obj using inspect. Raises OSError
    # with a TorchScript-specific message (plus optional error_msg) if source is unavailable.
    # filename may be None if it could not be determined.
    """
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lineno, filename)
    """
    filename = None  # in case getsourcefile throws
    try:
        filename = inspect.getsourcefile(obj)
        sourcelines, file_lineno = inspect.getsourcelines(obj)
    except OSError as e:
        msg = (
            f"Can't get source for {obj}. TorchScript requires source access in "
            "order to carry out compilation, make sure original .py files are "
            "available."
        )
        if error_msg:
            msg += "\n" + error_msg
        raise OSError(msg) from e

    return sourcelines, file_lineno, filename


def normalize_source_lines(sourcelines: list[str]) -> list[str]:
    # [FUNCTION DOC] Re-indents a function's source lines so every line before and after the def line starts with
    # the def line's leading whitespace (so comments or string continuations at lower indentation
    # do not break parsing). Returns the lines unchanged if no def line exists (e.g. lambda).
    """
    This helper function accepts a list of source lines. It finds the
    indentation level of the function definition (`def`), then it indents
    all lines in the function body to a point at or greater than that
    level. This allows for comments and continued string literals that
    are at a lower indentation than the rest of the code.
    Args:
        sourcelines: function source code, separated into lines by
                        the '\n' character
    Returns:
        A list of source lines that have been correctly aligned
    """

    def remove_prefix(text, prefix):
        # [FUNCTION DOC] HELPER: Returns text with prefix removed if it starts with it, else text unchanged.
        return text[text.startswith(prefix) and len(prefix) :]

    # Find the line and line number containing the function definition
    idx = None
    for i, l in enumerate(sourcelines):
        if l.lstrip().startswith("def"):
            idx = i
            break

    # This will happen when the function is a lambda- we won't find "def" anywhere in the source
    # lines in that case. Currently trying to JIT compile a lambda will throw an error up in
    # `parse_def()`, but we might want to handle this case in the future.
    if idx is None:
        return sourcelines

    # Get a string representing the amount of leading whitespace
    fn_def = sourcelines[idx]
    whitespace = fn_def.split("def")[0]

    # Add this leading whitespace to all lines before and after the `def`
    aligned_prefix = [
        whitespace + remove_prefix(s, whitespace) for s in sourcelines[:idx]
    ]
    aligned_suffix = [
        whitespace + remove_prefix(s, whitespace) for s in sourcelines[idx + 1 :]
    ]

    # Put it together again
    aligned_prefix.append(fn_def)
    return aligned_prefix + aligned_suffix


# Thin wrapper around SourceRangeFactory to store extra metadata
# about the function-to-be-compiled.
class SourceContext(SourceRangeFactory):
    # [CLASS DOC] Thin wrapper over SourceRangeFactory that also stores uses_true_division, filename and
    # funcname for the function being compiled by TorchScript.
    def __init__(
        self,
        source,
        filename,
        file_lineno,
        leading_whitespace_len,
        uses_true_division=True,
        funcname=None,
    ):
        # [FUNCTION DOC] Initializes the base factory with source, filename, file_lineno and leading_whitespace_len and
        # records uses_true_division (default True), filename and funcname.
        super().__init__(source, filename, file_lineno, leading_whitespace_len)
        self.uses_true_division = uses_true_division
        self.filename = filename
        self.funcname = funcname


@functools.cache
def make_source_context(*args):
    # [FUNCTION DOC] Cached constructor: returns a SourceContext for the given positional arguments (memoized).
    return SourceContext(*args)


def fake_range():
    # [FUNCTION DOC] Returns a placeholder source range (empty source, no file) covering one character, for
    # compiler-generated code.
    return SourceContext("", None, 0, 0).make_raw_range(0, 1)


class ParsedDef(NamedTuple):
    # [CLASS DOC] Named tuple result of parse_def: ast module, SourceContext, source text, filename, file_lineno.
    ast: ast.Module
    ctx: SourceContext
    source: str
    filename: str | None
    file_lineno: int


def parse_def(fn):
    # [FUNCTION DOC] Parses the source of function fn into a ParsedDef. Gets and normalizes source lines, dedents,
    # parses with ast, and raises RuntimeError unless the module holds exactly one function definition.
    # Computes the dedent width as the leading whitespace length and builds a SourceContext named
    # after fn.
    sourcelines, file_lineno, filename = get_source_lines_and_file(
        fn, ErrorReport.call_stack()
    )
    sourcelines = normalize_source_lines(sourcelines)
    source = "".join(sourcelines)
    dedent_src = dedent(source)
    py_ast = ast.parse(dedent_src)
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        raise RuntimeError(
            f"Expected a single top-level function: {filename}:{file_lineno}"
        )
    leading_whitespace_len = len(source.split("\n", 1)[0]) - len(
        dedent_src.split("\n", 1)[0]
    )
    ctx = make_source_context(
        source, filename, file_lineno, leading_whitespace_len, True, fn.__name__
    )
    return ParsedDef(py_ast, ctx, source, filename, file_lineno)
