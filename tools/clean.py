# [MODULE DOC] Provides clean(), which removes build artifacts by expanding the patterns of
# the gitignore file in the current directory. No import-time side effects.

import glob
import os
import shutil
from pathlib import Path


def clean():
    # [FUNCTION DOC] Deletes every path matching a pattern from the gitignore file in the
    # current directory, reading lines until the BEGIN NOT-CLEAN-FILES marker comment and
    # skipping other comment lines. Leading dots and slashes are stripped so absolute patterns
    # become relative. Files are removed first; if removal fails, the path is removed as a
    # directory tree with errors ignored. Side effect: irreversibly deletes files.
    # KNOWN ISSUE: stripping dot/slash characters also changes patterns that start with a dot
    # (such as hidden names), and negation patterns are not interpreted.
    """Clean, that is remove all files in .gitignore except in the NOT-CLEAN-FILES section."""
    ignores = Path(".gitignore").read_text(encoding="utf-8")
    for wildcard in filter(None, ignores.splitlines()):
        if wildcard.strip().startswith("#"):
            if "BEGIN NOT-CLEAN-FILES" in wildcard:
                # Marker is found and stop reading .gitignore.
                break
            # Ignore lines which begin with '#'.
        else:
            # Don't remove absolute paths from the system
            wildcard = wildcard.lstrip("./")
            for filename in glob.iglob(wildcard):
                try:
                    os.remove(filename)
                except OSError:
                    shutil.rmtree(filename, ignore_errors=True)
