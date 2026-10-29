# [MODULE DOC] Legacy setup.py shim for PyTorch. The real build uses scikit-build-core via
# pyproject.toml; this file only handles direct "python setup.py <command>" calls.
# The commands install and develop are forwarded to pip (with a DeprecationWarning); any
# other command aborts with the deprecation notice. Defines the constants PIP_INSTALL,
# FORWARDS and DEPRECATION_NOTICE. Importing has no side effects; main runs only as a script.
# PyTorch is built with scikit-build-core through the PEP 517 interface
# declared in pyproject.toml; pip and `python -m build` never run this
# file. It exists only for direct `python setup.py <command>` calls:
# install and develop are forwarded to pip, everything else fails with
# instructions (#180248).
#
# Deprecation schedule:
#   PyTorch 2.14-2.15: install and develop forward to pip
#   PyTorch 2.16-2.17: all commands fail with instructions
#   PyTorch 2.18:      this file is removed
#
# See https://github.com/pytorch/pytorch/issues/152276 for background.

import shlex
import subprocess
import sys
import warnings


PIP_INSTALL = [sys.executable, "-m", "pip", "install", "-v", "--no-build-isolation"]
FORWARDS = {
    "install": [*PIP_INSTALL, "."],
    "develop": [*PIP_INSTALL, "-e", "."],
}

DEPRECATION_NOTICE = """\
`python setup.py {command}` is deprecated: PyTorch is built with
scikit-build-core via the standard PEP 517 interface (pyproject.toml),
and setup.py is no longer part of the build.

Deprecation schedule:
  PyTorch 2.14-2.15: install and develop forward to pip
                     (current behavior); other commands fail
  PyTorch 2.16-2.17: all commands fail with instructions
  PyTorch 2.18:      setup.py is removed

Replacement commands:
  install:           spin install  (or: pip install --no-build-isolation -v .)
  editable install:  spin develop  (or: pip install --no-build-isolation -v -e .)
  wheel:             python -m build --wheel --no-isolation
  sdist:             python -m build --sdist

Build customization through environment variables (DEBUG=1, USE_CUDA=0,
MAX_JOBS=..., etc.) works unchanged with the replacement commands. See
the "From Source" section of README.md for details and
https://github.com/pytorch/pytorch/issues/152276 for background.
"""


def main() -> int:
    # [FUNCTION DOC] Entry point. Reads the command name from the first command-line argument
    # (placeholder text if absent) and formats the deprecation notice with it. Unknown commands
    # exit via SystemExit carrying the notice. For install/develop: warns with the notice, warns
    # about extra arguments if any, prints the forwarded pip command to stderr, runs it as a
    # subprocess and returns its exit code (the process exit status when run as a script).
    command = sys.argv[1] if len(sys.argv) > 1 else "<command>"
    notice = DEPRECATION_NOTICE.format(command=command)
    replacement = FORWARDS.get(command)
    if replacement is None:
        raise SystemExit(f"error: {notice}")
    warnings.warn(notice, DeprecationWarning)
    if len(sys.argv) > 2:
        warnings.warn(f"ignoring extra arguments: {shlex.join(sys.argv[2:])}")
    print(f"Forwarding to `{shlex.join(replacement)}`.", file=sys.stderr)
    return subprocess.run(replacement).returncode


if __name__ == "__main__":
    sys.exit(main())
