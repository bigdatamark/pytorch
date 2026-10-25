# [MODULE DOC] Maps (source file, line number) to the dotted qualified name of the enclosing
# function/class, computed lazily by tokenizing the file and memoized in a module-level cache.
# Main contents: cache, clearcache, get_funcname. No import-time side effects.
"""
This module provides functionality for caching and looking up fully qualified function
and class names from Python source files by line number.

It uses Python's tokenize module to parse source files and tracks function/class
definitions along with their nesting to build fully qualified names (e.g. 'class.method'
or 'module.function'). The results are cached in a two-level dictionary mapping:

    filename -> (line_number -> fully_qualified_name)

Example usage:
    name = get_funcname("myfile.py", 42)  # Returns name of function/class at line 42
    clearcache()  # Clear the cache if file contents have changed

The parsing is done lazily when a file is first accessed. Invalid Python files or
IO errors are handled gracefully by returning empty cache entries.
"""

import tokenize


cache: dict[str, dict[int, str]] = {}


def clearcache() -> None:
    # [FUNCTION DOC] Empties the module-level filename-to-line-names cache so files are re-parsed
    # on next lookup. Takes no arguments, returns nothing; mutates global state.
    cache.clear()


def _add_file(filename: str) -> None:
    # [FUNCTION DOC] HELPER: Tokenizes the file and stores in the cache a mapping from each
    # token start line to the dotted name of the innermost enclosing class/def (empty string at
    # top level). Tracks nesting through indent/dedent tokens. On OSError or TokenError the
    # file gets an empty mapping. Mutates the global cache; returns nothing.
    # KNOWN ISSUE: only the first token's line is recorded per token, so a line gets the name
    # in effect at its last token; behavior is undefined for invalid Python source.
    try:
        with tokenize.open(filename) as f:
            tokens = list(tokenize.generate_tokens(f.readline))
    except (OSError, tokenize.TokenError):
        cache[filename] = {}
        return

    # NOTE: undefined behavior if file is not valid Python source,
    # since tokenize will have undefined behavior.
    result: dict[int, str] = {}
    # current full funcname, e.g. xxx.yyy.zzz
    cur_name = ""
    cur_indent = 0
    significant_indents: list[int] = []

    for i, token in enumerate(tokens):
        if token.type == tokenize.INDENT:
            cur_indent += 1
        elif token.type == tokenize.DEDENT:
            cur_indent -= 1
            # possible end of function or class
            if significant_indents and cur_indent == significant_indents[-1]:
                significant_indents.pop()
                # pop the last name
                cur_name = cur_name.rpartition(".")[0]
        elif (
            token.type == tokenize.NAME
            and i + 1 < len(tokens)
            and tokens[i + 1].type == tokenize.NAME
            and (token.string == "class" or token.string == "def")
        ):
            # name of class/function always follows class/def token
            significant_indents.append(cur_indent)
            if cur_name:
                cur_name += "."
            cur_name += tokens[i + 1].string
        result[token.start[0]] = cur_name

    cache[filename] = result


def get_funcname(filename: str, lineno: int) -> str | None:
    # [FUNCTION DOC] Returns the dotted qualified function/class name enclosing the given line
    # of the given file, or None if the line has no recorded token (e.g. blank or beyond EOF).
    # Parses and caches the file on first access (file I/O, mutates the global cache). An
    # empty string means the line is at module level or the file was unreadable.
    if filename not in cache:
        _add_file(filename)
    return cache[filename].get(lineno, None)
