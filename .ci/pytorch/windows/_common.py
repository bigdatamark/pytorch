# [MODULE DOC] HELPER: Utilities shared by the Windows CD scripts: shell quoting, writing an
# env file of export lines for a parent bash to source (with PATH converted to POSIX form via
# cygpath), and a retrying file download. Import-time effect: compiles one regex only.
"""Helpers shared between the Windows CD Python scripts.

build_env_setup.py and build_install_deps.py both hand env back to a
parent bash wrapper via a `--env-out` file and both stream binaries from
ossci-windows. Keeping the helpers here means a fix in one place (e.g.
the cygpath -up PATH conversion or the BASH_FUNC_retry%% filter) reaches
every writer.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import urllib.request
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


_BASH_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def shell_quote(value: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the string unchanged if it is non-empty and made only of
    # letters, digits and a few safe punctuation characters; otherwise wraps it in single
    # quotes, escaping embedded single quotes, so bash reads it back verbatim. Pure function.
    if value and all(c.isalnum() or c in "_-./:=" for c in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def _to_posix_path_list(windows_path_list: str) -> str:
    # [FUNCTION DOC] HELPER: Converts a semicolon-separated Windows path list to a
    # colon-separated POSIX list by running the cygpath tool once. Empty input is returned
    # as is. Raises CalledProcessError if cygpath fails (or OSError if it is missing).
    """Convert a Windows `;`-separated path list to POSIX `:`-separated.

    vcvarsall.bat (and other cmd-side env-setup scripts) write PATH in
    Windows format with `;` separators and `\\` directory separators. The
    parent bash uses `:` separators and POSIX-style paths to find
    executables; sourcing PATH unmodified leaves bash with a single
    bogus PATH entry and the next `python` lookup dies with exit 127.

    `cygpath -up` is the canonical translator (Git Bash / MSYS ship it
    and the rest of PyTorch's Windows CI already uses it). Calling it
    once on the whole list is fine for this hot path.
    """
    if not windows_path_list:
        return windows_path_list
    result = subprocess.run(
        ["cygpath", "-up", windows_path_list],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def write_env_exports(env: dict[str, str], path: Path | None) -> None:
    # [FUNCTION DOC] Writes one export line per environment entry to the file at path, for
    # a bash script to source. Does nothing if path is None. Entries whose names are not
    # valid bash identifiers are skipped; a PATH value containing a semicolon is converted
    # to POSIX form first; values are shell-quoted. Overwrites the file (side effect).
    """Write `export KEY=VALUE` lines for build.sh to source.

    PATH is converted from Windows-format (`;`-separated, backslashes)
    to POSIX-format (`:`-separated, forward slashes) so the parent bash
    can find executables in subsequent steps. Other path-like env vars
    (INCLUDE, LIB, LIBPATH, ...) are left in Windows format because the
    MSVC tools that consume them expect that.

    Skip keys that aren't valid bash identifiers. When CI bash exports a
    function (e.g. `export -f retry` in binary_populate_env.sh), bash
    serializes it into the env as `BASH_FUNC_retry%%=() { ... }`. That
    leaks into the Python interpreter's `os.environ` and from there into
    the diff captured by `_capture_cmd_env`, but bash cannot re-export an
    identifier containing `%`, so sourcing the env file would die on
    `not a valid identifier`.
    """
    if path is None:
        return
    lines = []
    for k, v in env.items():
        if not _BASH_IDENT_RE.match(k):
            continue
        if k.upper() == "PATH" and ";" in v:
            # ';' separator marks the captured PATH as Windows-form; if
            # some future caller already feeds POSIX form (no ';') we
            # leave it alone rather than double-convert.
            v = _to_posix_path_list(v)
        lines.append(f"export {k}={shell_quote(v)}")
    path.write_text("\n".join(lines) + "\n")


def download(url: str, dest: Path, attempts: int = 5) -> None:
    # [FUNCTION DOC] Streams the url to the file dest in 1 MiB chunks, printing progress.
    # On any exception it sleeps with exponentially growing delay and retries, up to
    # attempts tries; after the last failure it exits the process with an error message.
    # Overwrites dest. Returns nothing.
    """Stream `url` to `dest`, retrying with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading {url} -> {dest} (attempt {attempt}/{attempts})")
            with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            return
        except Exception as exc:
            if attempt == attempts:
                sys.exit(f"Failed to download {url}: {exc}")
            time.sleep(2**attempt)
