# [MODULE DOC] Mypy plugin entry point that only checks the installed mypy version against the
# version pinned in .ci/docker/requirements-ci.txt, printing upgrade instructions to stderr on
# mismatch. Import-time side effects: none beyond imports (requires mypy installed).
import re
import sys
from pathlib import Path

from mypy.plugin import Plugin


def get_correct_mypy_version():
    # [FUNCTION DOC] Reads .ci/docker/requirements-ci.txt (relative to this file's parent
    # directory) and returns the pinned mypy version string. Raises ValueError if the file
    # does not contain exactly one pinned mypy entry; file errors propagate.
    # there's probably a more elegant way to do this
    (match,) = re.finditer(
        r"mypy==(\d+(?:\.\d+)*)",
        (
            Path(__file__).parent.parent / ".ci" / "docker" / "requirements-ci.txt"
        ).read_text(),
    )
    (version,) = match.groups()
    return version


def plugin(version: str):
    # [FUNCTION DOC] Mypy plugin hook called with the running mypy version. If it differs from
    # the pinned version, prints a warning with pip and conda install instructions to stderr
    # (does not abort). Always returns the base Plugin class.
    correct_version = get_correct_mypy_version()
    if version != correct_version:
        print(
            f"""\
You are using mypy version {version}, which is not supported
in the PyTorch repo. Please switch to mypy version {correct_version}.

For example, if you installed mypy via pip, run this:

    pip install mypy=={correct_version}

Or if you installed mypy via conda, run this:

    conda install -c conda-forge mypy={correct_version}
""",
            file=sys.stderr,
        )
    return Plugin
