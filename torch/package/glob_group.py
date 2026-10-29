# mypy: allow-untyped-defs
# [MODULE DOC] Glob-style pattern matching for torch.package. Defines the GlobPattern type
# alias (a string or iterable of strings) and GlobGroup, which compiles include/exclude
# patterns to regular expressions. No import-time side effects beyond the alias.
from __future__ import annotations

import re
from collections.abc import Iterable


GlobPattern = str | Iterable[str]


class GlobGroup:
    # [CLASS DOC] Matcher built from include and exclude glob patterns. State: lists of
    # compiled include and exclude regular expressions, the segment separator and a debug
    # description string used for string conversion. Immutable after construction. The key
    # method is matches.
    """A set of patterns that candidate strings will be matched against.

    A candidate is composed of a list of segments separated by ``separator``, e.g. "foo.bar.baz".

    A pattern contains one or more segments. Segments can be:
        - A literal string (e.g. "foo"), which matches exactly.
        - A string containing a wildcard (e.g. "torch*", or "foo*baz*"). The wildcard matches
          any string, including the empty string.
        - A double wildcard ("**"). This matches against zero or more complete segments.

    Examples:
        ``torch.**``: matches ``torch`` and all its submodules, e.g. ``torch.nn`` and ``torch.nn.functional``.
        ``torch.*``: matches ``torch.nn`` or ``torch.functional``, but not ``torch.nn.functional``.
        ``torch*.**``: matches ``torch``, ``torchvision``, and all their submodules.

    A candidate will match the ``GlobGroup`` if it matches any of the ``include`` patterns and
    none of the ``exclude`` patterns.

    Args:
        include (str | Iterable[str]): A string or list of strings,
            each representing a pattern to be matched against. A candidate
            will match if it matches *any* include pattern
        exclude (str | Iterable[str]): A string or list of strings,
            each representing a pattern to be matched against. A candidate
            will be excluded from matching if it matches *any* exclude pattern.
        separator (str): A string that delimits segments in candidates and
            patterns. By default this is "." which corresponds to how modules are
            named in Python. Another common value for this is "/", which is
            the Unix path separator.
    """

    def __init__(
        self, include: GlobPattern, *, exclude: GlobPattern = (), separator: str = "."
    ):
        # [FUNCTION DOC] Compiles the include and exclude patterns (each a string or an
        # iterable of strings) into regular expressions for the given separator and stores
        # them with a description string. Raises ValueError if a double wildcard appears as
        # part of a larger segment.
        self._dbg = f"GlobGroup(include={include}, exclude={exclude})"
        self.include = GlobGroup._glob_list(include, separator)
        self.exclude = GlobGroup._glob_list(exclude, separator)
        self.separator = separator

    def __str__(self):
        # [FUNCTION DOC] Returns the description string listing the include and exclude
        # patterns.
        return self._dbg

    def __repr__(self):
        # [FUNCTION DOC] Returns the same description string as string conversion.
        return self._dbg

    def matches(self, candidate: str) -> bool:
        # [FUNCTION DOC] Returns True if the candidate (with a leading separator prepended)
        # fully matches at least one include regex and none of the exclude regexes. Pure.
        candidate = self.separator + candidate
        return any(p.fullmatch(candidate) for p in self.include) and all(
            not p.fullmatch(candidate) for p in self.exclude
        )

    @staticmethod
    def _glob_list(elems: GlobPattern, separator: str = "."):
        # [FUNCTION DOC] HELPER: Compiles one pattern string, or each string of an iterable,
        # and returns the list of compiled regular expressions.
        if isinstance(elems, str):
            return [GlobGroup._glob_to_re(elems, separator)]
        else:
            return [GlobGroup._glob_to_re(e, separator) for e in elems]

    @staticmethod
    def _glob_to_re(pattern: str, separator: str = "."):
        # [FUNCTION DOC] HELPER: Splits the pattern on the separator and converts each
        # segment to a regex fragment (see nested helper), joins them and compiles the result.
        # Raises ValueError for a double wildcard embedded in a larger segment.
        # to avoid corner cases for the first component, we prefix the candidate string
        # with '.' so `import torch` will regex against `.torch`, assuming '.' is the separator
        def component_to_re(component):
            # [FUNCTION DOC] HELPER: Converts one segment to a regex fragment starting with
            # the escaped separator. A lone double wildcard becomes a group matching zero or
            # more whole segments; a single wildcard matches any run of non-separator
            # characters within the segment; otherwise literal text. Raises ValueError if a
            # double wildcard is mixed with other text.
            # KNOWN ISSUE: the separator is placed unescaped inside a character class, so
            # characters special in classes (e.g. a caret or backslash) would misbehave.
            if "**" in component:
                if component == "**":
                    return "(" + re.escape(separator) + "[^" + separator + "]+)*"
                else:
                    raise ValueError("** can only appear as an entire path segment")
            else:
                return re.escape(separator) + ("[^" + separator + "]*").join(
                    re.escape(x) for x in component.split("*")
                )

        result = "".join(component_to_re(c) for c in pattern.split(separator))
        return re.compile(result)
