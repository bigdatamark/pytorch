# [MODULE DOC] Command line utility that dumps the string form of every registered operator
# schema (builtin and custom class schemas known to the JIT) to a text file, one schema per
# line. The output is intended as input for the forward/backward compatibility check.
# Importing the module only imports argparse and torch; work happens in the main guard.
import argparse

import torch


def dump(filename):
    # [FUNCTION DOC] HELPER: Collects all JIT operator schemas plus custom class schemas into
    # one list, then opens the given file path for writing (truncating it) and writes each
    # schema as text followed by a newline. Argument: filename, the output path. Returns
    # nothing; raises OSError if the file cannot be opened.
    schemas = torch._C._jit_get_all_schemas()
    schemas += torch._C._jit_get_custom_class_schemas()
    with open(filename, "w") as f:
        for s in schemas:
            f.write(str(s))
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "-f",
        "--filename",
        help="filename to dump the schemas",
        type=str,
        default="schemas.txt",
    )
    args = parser.parse_args()
    dump(args.filename)
