# [MODULE DOC]
# Defines the command line parser shared by the token-based linters: an argparse parser
# preloaded with a files list and the fix, lintrunner and verbose flags, and with epilog
# printing that keeps line breaks. No import-time side effects.
from __future__ import annotations

import argparse
import sys
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from typing_extensions import Never


class ArgumentParser(argparse.ArgumentParser):
    # [CLASS DOC] Argument parser with the standard linter options already registered:
    # positional files, a fix flag (hidden in help unless the linter is a fixer), a lintrunner
    # flag and a verbose flag. Keeps the epilog privately so it can be printed verbatim.
    """
    Adds better help formatting and default arguments to argparse.ArgumentParser
    """

    def __init__(
        self,
        prog: str | None = None,
        usage: str | None = None,
        description: str | None = None,
        epilog: str | None = None,
        is_fixer: bool = False,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Builds the parser with the given program name, usage and description,
        # passing no epilog to the base class (it is stored privately instead), then adds the
        # files argument and the fix, lintrunner and verbose options. is_fixer controls whether
        # the fix option appears in help. Extra keyword arguments go to the base class.
        super().__init__(prog, usage, description, None, **kwargs)
        self._epilog = epilog

        help = "A list of files or directories to lint"
        self.add_argument("files", nargs="*", help=help)
        # TODO(rec): get fromfile_prefix_chars="@", type=argparse.FileType to work

        help = "Fix lint errors if possible" if is_fixer else argparse.SUPPRESS
        self.add_argument("-f", "--fix", action="store_true", help=help)

        help = "Run for lintrunner and print LintMessages which aren't edits"
        self.add_argument("-l", "--lintrunner", action="store_true", help=help)

        help = "Print more debug info"
        self.add_argument("-v", "--verbose", action="store_true", help=help)

    def exit(self, status: int = 0, message: str | None = None) -> Never:
        # [FUNCTION DOC] Overrides exit so the epilog is printed with its original line breaks
        # when help was requested on the command line, then defers to the base exit, which
        # terminates the process with the given status and message.
        # KNOWN ISSUE: operator precedence makes the condition "(epilog and status is zero and
        # -h given) or --help given", so the epilog (possibly None) is printed whenever --help
        # appears, even with a non-zero status or no epilog.
        """
        Overriding this method is a workaround for argparse throwing away all
        line breaks when printing the `epilog` section of the help message.
        """
        argv = sys.argv[1:]
        if self._epilog and not status and "-h" in argv or "--help" in argv:
            print(self._epilog)
        super().exit(status, message)
