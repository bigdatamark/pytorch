# [MODULE DOC] Script that regenerates the generated docstrings module of torch.masked.
# Run it from the repository root; torch must be importable. No import-time side effects.

"""This script updates the file torch/masked/_docs.py that contains
the generated doc-strings for various masked operations. The update
should be triggered whenever a new masked operation is introduced to
torch.masked package. Running the script requires that torch package
is functional.
"""

import os


def main() -> None:
    # [FUNCTION DOC] Imports torch (prints and returns on failure), builds new file content
    # with a header and one docstring variable per public masked operation (sorted by name),
    # and rewrites the target file only if the content changed. Prints the outcome. The target
    # path is relative to the current directory.
    target = os.path.join("torch", "masked", "_docs.py")

    try:
        import torch
    except ImportError as msg:
        print(f"Failed to import torch required to build {target}: {msg}")
        return

    if os.path.isfile(target):
        with open(target) as _f:
            current_content = _f.read()
    else:
        current_content = ""

    _new_content = []
    _new_content.append(
        """\
# -*- coding: utf-8 -*-
# This file is generated, do not modify it!
#
# To update this file, run the update masked docs script as follows:
#
#   python tools/update_masked_docs.py
#
# The script must be called from an environment where the development
# version of torch package can be imported and is functional.
#
"""
    )

    for func_name in sorted(torch.masked._ops.__all__):
        func = getattr(torch.masked._ops, func_name)
        func_doc = torch.masked._generate_docstring(func)  # type: ignore[no-untyped-call, attr-defined]
        _new_content.append(f'{func_name}_docstring = """{func_doc}"""\n')

    new_content = "\n".join(_new_content)

    if new_content == current_content:
        print(f"Nothing to update in {target}")
        return

    with open(target, "w") as _f:
        _f.write(new_content)

    print(f"Successfully updated {target}")


if __name__ == "__main__":
    main()
