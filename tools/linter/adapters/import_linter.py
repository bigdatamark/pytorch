# [MODULE DOC] Lint adapter that flags import statements of modules outside an allow list
# (standard library modules plus a few approved third-party packages such as torch, numpy,
# sympy and triton). Uses the shared _linter package (imported relatively when type checking
# or when this directory is not on the module path, otherwise absolutely) to tokenize files.
# When run as a script, prints each finding as a JSON line.
"""
Checks files to make sure there are no imports from disallowed third party
libraries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import token
from enum import Enum
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _PARENT not in _PATH:
    from . import _linter
else:
    import _linter


class LintSeverity(str, Enum):
    # [CLASS DOC] HELPER: String-valued enum of the severities lintrunner understands: error,
    # warning, advice and disabled.
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    # [CLASS DOC] HELPER: Record for one lint finding in lintrunner's JSON schema (path, line,
    # char, code, severity, name, original, replacement, description; None means unset). Instances
    # are converted to a mapping and printed as one JSON object per line.
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


LINTER_CODE = "IMPORT_LINTER"
CURRENT_FILE_NAME = os.path.basename(__file__)
_MODULE_NAME_ALLOW_LIST: set[str] = set()

# Add builtin modules of python.
_MODULE_NAME_ALLOW_LIST.update(sys.stdlib_module_names)

# Add the allowed third party libraries. Please avoid updating this unless you
# understand the risks -- see `_ERROR_MESSAGE` for why.
_MODULE_NAME_ALLOW_LIST.update(
    [
        "sympy",
        "einops",
        "libfb",
        "torch",
        "tvm",
        "_pytest",
        "tabulate",
        "optree",
        "typing_extensions",
        "triton",
        "functorch",
        "torchrec",
        "numpy",
        "torch_xla",
        "annotationlib",  # added in python 3.14
    ]
)

_ERROR_MESSAGE = """
Please do not import third-party modules in PyTorch unless they're explicit
requirements of PyTorch. Imports of a third-party library may have side effects
and other unintentional behavior. If you're just checking if a module exists,
use sys.modules.get("torchrec") or the like.
"""


def check_file(filepath: str) -> list[LintMessage]:
    # [FUNCTION DOC] Tokenize the file with the shared PythonFile helper and, for each logical
    # line, skip leading indent tokens and check whether it starts with the import or from keyword
    # followed by a name token; if that first module name is not in the allow list, record an
    # error LintMessage with the standard explanatory text. Returns the list of messages. Relative
    # imports and the part after the first dotted component are not examined.
    # KNOWN ISSUE: the reported line is the zero-based token line index rather than a one-based
    # line number; the argparse description in the main block is copy-pasted from the native
    # functions linter.
    path = Path(filepath)
    file = _linter.PythonFile("import_linter", path=path)
    lint_messages = []
    for line_number, line_of_tokens in enumerate(file.token_lines):
        # Skip indents
        idx = 0
        for tok in line_of_tokens:
            if tok.type == token.INDENT:
                idx += 1
            else:
                break

        # Look for either "import foo..." or "from foo..."
        if idx + 1 < len(line_of_tokens):
            tok0 = line_of_tokens[idx]
            tok1 = line_of_tokens[idx + 1]
            if tok0.type == token.NAME and tok0.string in {"import", "from"}:
                if tok1.type == token.NAME:
                    module_name = tok1.string
                    if module_name not in _MODULE_NAME_ALLOW_LIST:
                        msg = LintMessage(
                            path=filepath,
                            line=line_number,
                            char=None,
                            code="IMPORT",
                            severity=LintSeverity.ERROR,
                            name="Disallowed import",
                            original=None,
                            replacement=None,
                            description=_ERROR_MESSAGE,
                        )
                        lint_messages.append(msg)
    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native functions linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filepaths",
        nargs="+",
        help="paths of files to lint",
    )
    args = parser.parse_args()

    # Check all files.
    all_lint_messages = []
    for filepath in args.filepaths:
        lint_messages = check_file(filepath)
        all_lint_messages.extend(lint_messages)

    # Print out lint messages.
    for lint_message in all_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
