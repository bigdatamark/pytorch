# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [MODULE DOC] Script run inside a vLLM checkout that edits the requirements files and
# pyproject.toml in place, deleting lines that pin torch, torchvision or torchaudio so the
# already installed torch build is used. Matches whole package names, or optionally only
# known line prefixes. No import-time side effects beyond compiling a regex.

import argparse
import glob
import os
import re
import sys


# Only strip targeted libraries when checking prefix
TORCH_LIB_PREFIXES = (
    # requirements/*.txt/in
    "torch=",
    "torchvision=",
    "torchaudio=",
    # pyproject.toml
    '"torch =',
    '"torchvision =',
    '"torchaudio =',
)

# Match lines where the package name is exactly torch/torchvision/torchaudio,
# not a substring of another package (e.g. terratorch, open_clip_torch).
_TORCH_PKG_RE = re.compile(
    r"""^\s*['"]?\s*(?:torchvision|torchaudio|torch)\s*(?:[=<>!;\[,\]'"@~#(]|$)""",
    re.IGNORECASE,
)


def main(argv):
    # [FUNCTION DOC] Parses the optional prefix-only flag from argv, then for each
    # requirements text/in file under the current directory plus pyproject.toml that exists
    # and mentions torch, rewrites it without the torch-family requirement lines, printing
    # each removed line. In prefix mode only lines beginning with a known prefix are
    # removed; otherwise a regex on the package name is used. Mutates files; returns None.
    parser = argparse.ArgumentParser(
        description="Strip torch lib requirements to use installed version."
    )
    parser.add_argument(
        "--prefix",
        action="store_true",
        help="Strip prefix matches only (default: False)",
    )
    args = parser.parse_args(argv)

    for file in (
        *glob.glob("requirements/**/*.txt", recursive=True),
        *glob.glob("requirements/**/*.in", recursive=True),
        "pyproject.toml",
    ):
        if not os.path.exists(file):
            continue
        with open(file) as f:
            lines = f.readlines()
        if "torch" in "".join(lines).lower():
            with open(file, "w") as f:
                for line in lines:
                    if (
                        args.prefix
                        and not line.lower().strip().startswith(TORCH_LIB_PREFIXES)
                        or not args.prefix
                        and not _TORCH_PKG_RE.match(line)
                    ):
                        f.write(line)
                    else:
                        print(f">>> removed from {file}:", line.strip())


if __name__ == "__main__":
    main(sys.argv[1:])
