# [MODULE DOC]
# Stream filter that post-processes include lines of include-what-you-use output: reads
# standard input and writes to standard output, rewriting quoted includes to angle-bracket
# includes and mapping C standard headers to their C++ counterparts. Other lines pass
# through unchanged. Defines the two include regexes and the C-to-C++ header map.
import re
import sys


QUOTE_INCLUDE_RE = re.compile(r'^#include "(.*)"')
ANGLE_INCLUDE_RE = re.compile(r"^#include <(.*)>")

# By default iwyu will pick the C include, but we prefer the C++ headers
STD_C_HEADER_MAP = {
    "<assert.h>": "<cassert>",
    "<complex.h>": "<ccomplex>",
    "<ctype.h>": "<cctype>",
    "<errno.h>": "<cerrno>",
    "<fenv.h>": "<cfenv>",
    "<float.h>": "<cfloat>",
    "<inttypes.h>": "<cinttypes>",
    "<iso646.h>": "<ciso646>",
    "<limits.h>": "<climits>",
    "<locale.h>": "<clocale>",
    "<math.h>": "<cmath>",
    "<setjmp.h>": "<csetjmp>",
    "<signal.h>": "<csignal>",
    "<stdalign.h>": "<cstdalign>",
    "<stdarg.h>": "<cstdarg>",
    "<stdbool.h>": "<cstdbool>",
    "<stddef.h>": "<cstddef>",
    "<stdint.h>": "<cstdint>",
    "<stdio.h>": "<cstdio>",
    "<stdlib.h>": "<cstdlib>",
    "<string.h>": "<cstring>",
    "<tgmath.h>": "<ctgmath>",
    "<time.h>": "<ctime>",
    "<uchar.h>": "<cuchar>",
    "<wchar.h>": "<cwchar>",
    "<wctype.h>": "<cwctype>",
}


def main() -> None:
    # [FUNCTION DOC] Reads standard input line by line and echoes each line to standard
    # output after fixing include directives. A quoted include becomes an angle-bracket
    # include (its trailing text is kept as is). An angle-bracket include has its header
    # replaced by the C++ equivalent when one is in the map, and a single space is inserted
    # before any trailing text longer than a newline. All other lines are copied verbatim.
    # KNOWN ISSUE: a quoted include is converted without the C-header mapping, so a quoted C
    # header stays as the C header.
    for line in sys.stdin:
        # Convert all quoted includes to angle brackets
        match = QUOTE_INCLUDE_RE.match(line)
        if match is not None:
            print(f"#include <{match.group(1)}>{line[match.end(0) :]}", end="")
            continue

        match = ANGLE_INCLUDE_RE.match(line)
        if match is not None:
            path = f"<{match.group(1)}>"
            new_path = STD_C_HEADER_MAP.get(path, path)
            tail = line[match.end(0) :]
            if len(tail) > 1:
                tail = " " + tail
            print(f"#include {new_path}{tail}", end="")
            continue

        print(line, end="")


if __name__ == "__main__":
    main()
