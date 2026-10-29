# [MODULE DOC] Lintrunner adapter built on the shared file-linter framework that reports
# Python classes and functions which are longer than a line limit but have a docstring
# shorter than a minimum length. Supports a grandfather list (JSON of tolerated existing
# block sizes), a report mode that prints a JSON summary of all blocks, and rewriting the
# grandfather list. Import-time side effects: computes the module path list and imports the
# local _linter package either relatively or absolutely depending on sys.path; defines
# limits (100 lines per class, 80 per function, 50 docstring characters, 10 percent tolerance).
from __future__ import annotations

import itertools
import json
import sys
from functools import cached_property
from pathlib import Path
from typing import Any, TYPE_CHECKING


_FILE = Path(__file__).absolute()
_PATH = [Path(p).absolute() for p in sys.path]
_OVERRIDES = {"@override", "@typing_extensions.override", "@typing.override"}

if TYPE_CHECKING or _FILE.parent not in _PATH:
    from . import _linter
else:
    import _linter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


GRANDFATHER_LIST = _FILE.parent / "docstring_linter-grandfather.json"

# We tolerate a 10% increase in block size before demanding a docstring
TOLERANCE_PERCENT = 10

MAX_LINES = {"class": 100, "def": 80}

MIN_DOCSTRING = 50  # docstrings shorter than this are too short

DESCRIPTION = """
`docstring_linter` reports on long functions, methods or classes without docstrings
""".strip()

METHOD_OVERRIDE_HINT = (
    "If the method overrides a method on a parent class, adding the"
    " `@typing_extensions.override` decorator will make this error"
    " go away."
)


class DocstringLinter(_linter.FileLinter):
    # [CLASS DOC] File linter flagging over-long undocumented blocks. Class attributes name
    # the linter, give its description and mark it as not a fixer. State: path_to_blocks maps
    # each linted path to per-block data (status good, bad or grandfather) and path_to_errors
    # maps paths to error entries, both filled during linting. Key methods: lint_all (lint,
    # then report and optionally write the grandfather file), _lint, _is_bad_block.
    linter_name = "docstring_linter"
    description = DESCRIPTION
    is_fixer = False

    path_to_blocks: dict[str, list[dict[str, Any]]]
    path_to_errors: dict[str, list[dict[str, Any]]]

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        # [FUNCTION DOC] Initialize the base linter with argv, register this linter's extra
        # command line options on its parser, and start with empty block and error maps.
        super().__init__(argv)
        add_arguments(self.parser.add_argument)
        self.path_to_blocks = {}
        self.path_to_errors = {}

    def lint_all(self) -> bool:
        # [FUNCTION DOC] Lint all files via the base class, then print the optional report and
        # write the grandfather file if requested (side effects: stdout, file write). Returns
        # the base class success flag.
        success = super().lint_all()
        self._report()
        self._write_grandfather()
        return success

    def _lint(self, pf: _linter.PythonFile) -> Iterator[_linter.LintResult]:
        # [FUNCTION DOC] HELPER: lint one parsed file. A file already seen is reported on
        # stderr and skipped. Otherwise find bad blocks, drop class/constructor pairs where one
        # side is documented (unless separate linting is on), subtract grandfathered blocks,
        # and yield a result for each remaining block in sorted order. Records every block with
        # its status (grandfather, bad, good) in the per-path block map.
        if (p := str(pf.path)) in self.path_to_blocks:
            print("Repeated file", p, file=sys.stderr)
            return

        blocks = pf.blocks
        bad = {b for b in blocks if self._is_bad_block(b, pf)}
        bad = self._dont_require_constructor_and_class_docs(blocks, bad)
        gf = self._grandfathered(pf.path, bad)

        yield from (self._block_result(b, pf) for b in sorted(bad - gf))

        def as_data(b: _linter.Block) -> dict[str, Any]:
            # [FUNCTION DOC] HELPER: block data mapping with a status field of grandfather,
            # bad or good, prepended to the block's own data.
            status = "grandfather" if b in gf else "bad" if b in bad else "good"
            return {"status": status, **b.as_data()}

        self.path_to_blocks[p] = [as_data(b) for b in blocks]

    def _error(self, pf: _linter.PythonFile, result: _linter.LintResult) -> None:
        # [FUNCTION DOC] HELPER: record a single error entry (line number to message name) for
        # the file path, replacing any earlier entry for that path.
        self.path_to_errors[str(pf.path)] = [{str(result.line): result.name}]

    @cached_property
    def _grandfather(self) -> dict[str, dict[str, Any]]:
        # [FUNCTION DOC] HELPER: cached property loading the grandfather JSON file named by the
        # grandfather option; returns an empty mapping if it is missing. Other errors are
        # printed to stderr and re-raised.
        # KNOWN ISSUE: the error message names the default grandfather path constant, not the
        # path actually given by the option.
        try:
            with open(self.args.grandfather) as fp:
                return json.load(fp)  # type: ignore[no-any-return]
        except FileNotFoundError:
            return {}
        except Exception as e:
            print("ERROR:", e, "in", GRANDFATHER_LIST, file=sys.stderr)
            raise

    @cached_property
    def _max_lines(self) -> dict[str, int]:
        # [FUNCTION DOC] HELPER: cached property mapping block category (class, def) to the
        # configured maximum undocumented line count.
        return {"class": self.args.max_class, "def": self.args.max_def}

    def _grandfathered(
        self, path: Path | None, bad: set[_linter.Block]
    ) -> set[_linter.Block]:
        # [FUNCTION DOC] HELPER: return the subset of bad blocks that are grandfathered, i.e.
        # whose line count does not exceed the recorded size for that file and block name
        # grown by the tolerance percent (unknown blocks count as zero). Returns an empty set
        # when there is no path, grandfathering is disabled, or the list is being rewritten.
        if path is None or self.args.no_grandfather or self.args.write_grandfather:
            return set()

        grand: dict[str, int] = self._grandfather.get(str(path), {})
        tolerance_ratio = 1 + self.args.grandfather_tolerance / 100.0

        def grandfathered(b: _linter.Block) -> bool:
            # [FUNCTION DOC] HELPER: True if the block's line count is within the tolerated
            # grandfathered size for its display name.
            lines = int(grand.get(b.display_name, 0) * tolerance_ratio)
            return b.line_count <= lines

        return {b for b in bad if grandfathered(b)}

    def _block_result(
        self, b: _linter.Block, pf: _linter.PythonFile
    ) -> _linter.LintResult:
        # [FUNCTION DOC] HELPER: build the lint result for a bad block, positioned at the
        # block's first token. The message says no docstring was found (with an override hint
        # for methods) or that the docstring was too short, with its length and the minimum.
        def_name = "function" if b.category == "def" else "class"
        msg = f"docstring found for {def_name} '{b.name}' ({b.line_count} lines)"
        if len(b.docstring):
            s = "" if len(b.docstring) == 1 else "s"
            needed = f"needed {self.args.min_docstring}"
            msg = f"{msg} was too short ({len(b.docstring)} character{s}, {needed})"
        else:
            msg = f"No {msg}"
            if b.is_method:
                msg = f"{msg}. {METHOD_OVERRIDE_HINT}"
        return _linter.LintResult(msg, *pf.tokens[b.begin].start)

    def _display(
        self, pf: _linter.PythonFile, results: list[_linter.LintResult]
    ) -> Iterator[str]:
        # [FUNCTION DOC] HELPER: yield the base class display lines unless report mode is on,
        # in which case nothing is displayed.
        if not self.args.report:
            yield from super()._display(pf, results)

    def _dont_require_constructor_and_class_docs(
        self, blocks: Sequence[_linter.Block], bad: set[_linter.Block]
    ) -> set[_linter.Block]:
        # [FUNCTION DOC] HELPER: unless separate linting of constructors is enabled, remove
        # from the bad set every class whose constructor has a long enough docstring and every
        # constructor whose class has one, so documenting either side suffices.
        if self.args.lint_init:
            return bad

        good = {b for b in blocks if len(b.docstring) >= self.args.min_docstring}

        def has_class_init_doc(b: _linter.Block) -> bool:
            # [FUNCTION DOC] HELPER: for a class, True if any child constructor is documented;
            # for a constructor, True if its parent class is documented; otherwise False.
            if b.is_class:
                # Is it a class whose constructor is documented?
                children = (blocks[i] for i in b.children)
                return any(b.is_init and b in good for b in children)

            # Is it a constructor whose class is documented?
            return b.is_init and b.parent is not None and blocks[b.parent] in good

        return {b for b in bad if not has_class_init_doc(b)}

    def _is_bad_block(self, b: _linter.Block, pf: _linter.PythonFile) -> bool:
        # [FUNCTION DOC] HELPER: a block is bad when it is not an override or omitted by a
        # lint-suppression marker, exceeds the line limit for its category, has a docstring
        # shorter than the minimum, and (unless enabled by options) is neither a local
        # definition nor a name starting with an underscore.
        max_lines = self._max_lines[b.category]
        return (
            not (b.is_override or pf.omitted(pf.tokens, b.begin, b.end + 1))
            and b.line_count > max_lines
            and len(b.docstring) < self.args.min_docstring
            and (self.args.lint_local or not b.is_local)
            and (self.args.lint_protected or not b.name.startswith("_"))
        )

    def _report(self) -> None:
        # [FUNCTION DOC] HELPER: when report mode is on and not running under lintrunner, print
        # to stdout a sorted JSON summary of per-file non-good blocks plus recorded errors.
        if not self.args.lintrunner and self.path_to_blocks and self.args.report:
            report = {
                k: s for k, v in self.path_to_blocks.items() if (s := file_summary(v))
            } | self.path_to_errors
            print(json.dumps(report, sort_keys=True, indent=2))

    def _write_grandfather(self) -> None:
        # [FUNCTION DOC] HELPER: if rewriting is requested, overwrite the grandfather file with
        # a sorted JSON mapping of path to block display name to line count for all blocks with
        # bad status. Side effect: file write.
        if self.args.write_grandfather:
            results: dict[str, dict[str, int]] = {}

            for path, blocks in self.path_to_blocks.items():
                for block in blocks:
                    if block["status"] == "bad":
                        d = results.setdefault(path, {})
                        d[block["display_name"]] = block["line_count"]

            with open(self.args.grandfather, "w") as fp:
                json.dump(results, fp, sort_keys=True, indent=2)


def make_recursive(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Convert a flat list of block data mappings, whose children are index
    # lists, into a forest: copies of the top-level blocks (no parent) with children replaced
    # recursively by the child mappings.
    def rec(i: int) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: copy block i and replace its child indices by recursively
        # built child copies.
        d = dict(blocks[i])
        d["children"] = [rec(c) for c in d["children"]]
        return d

    return [rec(i) for i, b in enumerate(blocks) if b["parent"] is None]


def make_terse(
    blocks: Sequence[dict[str, Any]],
    index_by_line: bool = True,
) -> dict[str, dict[str, Any]]:
    # [FUNCTION DOC] Build a compact mapping describing blocks: for each block an entry with
    # docstring length, line count, status (default good) and, for nested blocks, a child
    # mapping. Keys are start lines padded to equal width (index_by_line true, name stored in
    # the entry) or unique names made by appending a counter (false, line stored in entry).
    # Raises AssertionError if the children of a block mix integers and mappings.
    result: dict[str, dict[str, Any]] = {}

    max_line = max(b["start_line"] for b in blocks) if blocks else 0
    line_field_width = len(str(max_line))

    for b in blocks:
        root = f"{b['category']} {b['full_name']}"
        for i in itertools.count():
            name = root + bool(i) * f"[{i + 1}]"
            if name not in result:
                break

        d = {
            "docstring_len": len(b["docstring"]),
            "lines": b["line_count"],
            "status": b.get("status", "good"),
        }

        start_line = b["start_line"]
        if index_by_line:
            d["name"] = name
            result[f"{start_line:>{line_field_width}}"] = d
        else:
            d["line"] = start_line
            result[name] = d

        if kids := b["children"]:
            if not all(isinstance(k, int) for k in kids):
                if not all(isinstance(k, dict) for k in kids):
                    raise AssertionError("children must be all int or all dict")
                d["children"] = make_terse(kids)

    return result


def file_summary(
    blocks: Sequence[dict[str, Any]], report_all: bool = False
) -> dict[str, str]:
    # [FUNCTION DOC] Summarize blocks as a mapping from start line label to a one-line text
    # with name, line count, docstring length and a FAIL or grandfathered marker. Good blocks
    # are omitted unless report_all is true. Common leading padding of the keys is removed.
    # Raises AssertionError on an unknown status.
    def to_line(v: dict[str, Any]) -> str | None:
        # [FUNCTION DOC] HELPER: format one terse entry as text, or None for a good block when
        # not reporting all. Raises AssertionError for a status other than good, grandfather
        # or bad.
        if (status := v["status"]) == "good":
            if not report_all:
                return None
            fail = ""
        elif status == "grandfather":
            fail = ": (grandfathered)"
        else:
            if status != "bad":
                raise AssertionError(f"Expected status 'bad', got '{status}'")
            fail = ": FAIL"
        name = v["name"]
        lines = v["lines"]
        docs = v["docstring_len"]
        parens = "()" if name.startswith("def ") else ""
        return f"{name}{parens}: {lines=}, {docs=}{fail}"

    t = make_terse(blocks)
    r = {k: line for k, v in t.items() if (line := to_line(v))}
    while r and all(k.startswith(" ") for k in r):
        r = {k[1:]: v for k, v in r.items()}
    return r


def add_arguments(add: Callable[..., Any]) -> None:
    # [FUNCTION DOC] Register this linter's options through the given add-argument callable:
    # grandfather file and tolerance, lint-init, lint-local, lint-protected, max class and
    # function lines, minimum docstring length, no-grandfather, report and write-grandfather.
    h = "Set the grandfather list"
    add("--grandfather", "-g", default=str(GRANDFATHER_LIST), type=str, help=h)

    h = "Tolerance for grandfather sizes, in percent"
    add("--grandfather-tolerance", "-t", default=TOLERANCE_PERCENT, type=float, help=h)

    h = "Lint __init__ and class separately"
    add("--lint-init", "-i", action="store_true", help=h)

    h = "Lint definitions inside other functions"
    add("--lint-local", "-o", action="store_true", help=h)

    h = "Lint functions, methods and classes that start with _"
    add("--lint-protected", "-p", action="store_true", help=h)

    h = "Maximum number of lines for an undocumented class"
    add("--max-class", "-c", default=MAX_LINES["class"], type=int, help=h)

    h = "Maximum number of lines for an undocumented function"
    add("--max-def", "-d", default=MAX_LINES["def"], type=int, help=h)

    h = "Minimum number of characters for a docstring"
    add("--min-docstring", "-s", default=MIN_DOCSTRING, type=int, help=h)

    h = "Disable the grandfather list"
    add("--no-grandfather", "-n", action="store_true", help=h)

    h = "Print a report on all classes and defs"
    add("--report", "-r", action="store_true", help=h)

    h = "Rewrite the grandfather list"
    add("--write-grandfather", "-w", action="store_true", help=h)


if __name__ == "__main__":
    DocstringLinter.run()
