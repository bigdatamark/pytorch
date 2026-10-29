# [MODULE DOC]
# Lintrunner adapter that flags uses of the builtin set type (calls and braced set literals)
# and proposes autofix replacements using OrderedSet from torch.utils._ordered_set.
# Main content: SetLinter, a FileLinter subclass, plus the message/description constants.
# Import-time side effect: picks the relative or absolute import of the shared _linter module
# depending on whether this directory is already on sys.path. Run as a script, it invokes
# SetLinter.run().
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _PARENT not in _PATH:
    from . import _linter
else:
    import _linter

if TYPE_CHECKING:
    from collections.abc import Iterator


ERROR = "Builtin `set` is deprecated"
IMPORT_LINE = "from torch.utils._ordered_set import OrderedSet\n\n"

DESCRIPTION = """`set_linter` is a lintrunner linter which finds usages of the
Python built-in class `set` in Python code, and optionally replaces them with
`OrderedSet`.
"""

EPILOG = """
`lintrunner` operates on whole commits. If you want to remove uses of `set`
from existing files not part of a commit, call `set_linter` directly:

    python tools/linter/adapters/set_linter.py --fix [... python files ...]

---

To omit a line of Python code from `set_linter` checking, append a comment:

    s = set()  # noqa: set_linter
    t = {  # noqa: set_linter
       "one",
       "two",
    }

---

Running set_linter in fix mode (though either `lintrunner -a` or `--fix`
should not significantly change the behavior of working code, but will still
usually needs some manual intervention:

1. Replacing `set` with `OrderedSet` will sometimes introduce new typechecking
errors because `OrderedSet` is imperfectly generic. Find a common type for its
elements (in the worst case, `typing.Any` always works), and use
`OrderedSet[YourCommonTypeHere]`.

2. The fix mode doesn't recognize generator expressions, so it replaces:

    s = {i for i in range(3)}

with

    s = OrderedSet([i for i in range(3)])

You can and should delete the square brackets in every such case.

3. There is a common pattern of set usage where a set is created and then only
used for testing inclusion. For small collections, up to around 12 elements, a
tuple is more time-efficient than an OrderedSet and also has less visual clutter
(see https://github.com/rec/test/blob/master/python/time_access.py).
"""


class SetLinter(_linter.FileLinter):
    # [CLASS DOC]
    # File linter that reports every use of the builtin set and every braced set literal.
    # It is stateless apart from class attributes (linter name, description, epilog) and
    # reports column numbers. The only overridden hook is _lint.
    linter_name = "set_linter"
    description = DESCRIPTION
    epilog = EPILOG
    report_column_numbers = True

    def _lint(self, pf: _linter.PythonFile) -> Iterator[_linter.LintResult]:
        # [FUNCTION DOC] HELPER: Generator of lint results for one parsed Python file.
        # If the file has any set usage and an import insertion point is known, first yields
        # a result adding the OrderedSet import line. Then for each braced set yields two
        # results rewriting the opening token to the start of an OrderedSet list call and the
        # closing token to the end of it, and for each use of the set name yields a rename to
        # OrderedSet. Does not mutate the file; no exceptions raised here.
        if (pf.sets or pf.braced_sets) and (ins := pf.insert_import_line) is not None:
            yield _linter.LintResult(
                "Add import for OrderedSet", ins, 0, IMPORT_LINE, 0
            )
        for b in pf.braced_sets:
            yield _linter.LintResult(ERROR, *b[0].start, "OrderedSet([", 1)
            yield _linter.LintResult(ERROR, *b[-1].start, "])", 1)

        for s in pf.sets:
            yield _linter.LintResult(ERROR, *s.start, "OrderedSet", 3)


if __name__ == "__main__":
    SetLinter.run()
