# [MODULE DOC] Command-line script (argument: path to a build log file) that prints the log
# lines worth reading. It reads the whole file and prints every line except those containing
# known noise markers (CPU instruction set probes, symbol-existence checks, compilation error
# formatting tests). Runs entirely at import time; no functions. Raises an error if the
# argument is missing or the file cannot be read.
# KNOWN ISSUE: lines are printed with their own trailing newline retained, so output has a
# blank line after each line.
import sys


log_file_path = sys.argv[1]

with open(log_file_path) as f:
    lines = f.readlines()

for line in lines:
    # Ignore errors from CPU instruction set, symbol existing testing,
    # or compilation error formatting
    ignored_keywords = [
        "src.c",
        "CheckSymbolExists.c",
        "test_compilation_error_formatting",
    ]
    if all(keyword not in line for keyword in ignored_keywords):
        print(line)
