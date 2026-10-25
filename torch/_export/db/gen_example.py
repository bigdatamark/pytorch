# [MODULE DOC] Command-line helper that scaffolds a new ExportDB example file
# from a template. Run as a script with the case name; nothing happens at
# import time except importing the examples package.
import os
import sys

import torch._export.db.examples as examples

TEMPLATE = '''import torch

def {case_name}(x):
    """
    """

    return
'''

if __name__ == "__main__":
    # Script entry: requires exactly one argument (the case name), else raises
    # an AssertionError. Derives the examples directory from the package name,
    # asserts it exists, and writes a new file named after the case containing
    # the template with the case name substituted; prints the target path.
    if len(sys.argv) != 2:
        raise AssertionError(f"expected 2 arguments, got {len(sys.argv)}")
    root_dir = examples.__name__.replace(".", "/")
    if not os.path.exists(root_dir):
        raise AssertionError(f"root_dir does not exist: {root_dir}")
    with open(os.path.join(root_dir, sys.argv[1] + ".py"), "w") as f:
        print("Writing to", f.name, "...")
        f.write(TEMPLATE.format(case_name=sys.argv[1]))
