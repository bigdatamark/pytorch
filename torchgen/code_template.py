# [MODULE DOC] Defines CodeTemplate, a small string template engine used by the code
# generators. Identifiers written with a dollar sign (optionally in braces) are replaced from
# an environment; list values are expanded as indented blocks or comma separated lists.
# No import-time side effects.
from __future__ import annotations

import itertools
import re
import textwrap
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# match $identifier or ${identifier} and replace with value in env
# If this identifier is at the beginning of whitespace on a line
# and its value is a list then it is treated as
# block substitution by indenting to that depth and putting each element
# of the list on its own line
# if the identifier is on a line starting with non-whitespace and a list
# then it is comma separated ${,foo} will insert a comma before the list
# if this list is not empty and ${foo,} will insert one after.


class CodeTemplate:
    # [CLASS DOC] A template pattern string plus an optional source filename. substitution is a
    # class-level compiled regex matching optional leading indentation and a dollar identifier.
    # Use from_file to load a template from disk and substitute to render it.
    substitution_str = r"(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})"
    substitution = re.compile(substitution_str, re.MULTILINE)

    pattern: str
    filename: str

    @staticmethod
    def from_file(filename: str) -> CodeTemplate:
        # [FUNCTION DOC] Read the whole file named filename and return a CodeTemplate holding its
        # text, with the filename recorded. Raises OSError if the file cannot be read.
        with open(filename) as f:
            return CodeTemplate(f.read(), filename)

    def __init__(self, pattern: str, filename: str = "") -> None:
        # [FUNCTION DOC] Store the pattern text and the filename (empty by default).
        self.pattern = pattern
        self.filename = filename

    def substitute(
        self, env: Mapping[str, object] | None = None, **kwargs: object
    ) -> str:
        # [FUNCTION DOC] Render the template. Each placeholder is looked up in kwargs first, then
        # in env; a missing key raises KeyError. A placeholder at the start of a line (after
        # whitespace) is replaced by the value: a list becomes one element per line, split on
        # newlines, indented to the same depth, trailing whitespace stripped. Elsewhere a list
        # becomes comma separated text, with the brace forms adding a leading or trailing comma
        # only when the list is nonempty; other values are converted with str. Returns the text.
        if env is None:
            env = {}

        def lookup(v: str) -> object:
            # [FUNCTION DOC] HELPER: Return the value for key v, preferring kwargs over env.
            if env is None:
                raise AssertionError("env must be non-None")
            return kwargs[v] if v in kwargs else env[v]

        def indent_lines(indent: str, v: Sequence[object]) -> str:
            # [FUNCTION DOC] HELPER: Indent all lines of all elements of v by indent and rstrip
            # each line, then join with newlines.
            content = "\n".join(
                itertools.chain.from_iterable(str(e).splitlines() for e in v)
            )
            content = textwrap.indent(content, prefix=indent)
            # Remove trailing whitespace on each line
            return "\n".join(map(str.rstrip, content.splitlines())).rstrip()

        def replace(match: re.Match[str]) -> str:
            # [FUNCTION DOC] HELPER: Regex callback computing the replacement text for one
            # placeholder match: handles brace and comma markers, block indent and list joins.
            indent = match.group(1)
            key = match.group(2)
            comma_before = ""
            comma_after = ""
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    comma_before = ", "
                    key = key[1:]
                if key[-1] == ",":
                    comma_after = ", "
                    key = key[:-1]
            v = lookup(key)
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]
                return indent_lines(indent, v)
            elif isinstance(v, list):
                middle = ", ".join([str(x) for x in v])
                if len(v) == 0:
                    return middle
                return comma_before + middle + comma_after
            else:
                return str(v)

        return self.substitution.sub(replace, self.pattern)


if __name__ == "__main__":
    c = CodeTemplate(
        """\
    int foo($args) {

        $bar
            $bar
        $a+$b
    }
    int commatest(int a${,stuff})
    int notest(int a${,empty,})
    """
    )
    print(
        c.substitute(
            args=["hi", 8],
            bar=["what", 7],
            a=3,
            b=4,
            stuff=["things...", "others"],
            empty=[],
        )
    )
