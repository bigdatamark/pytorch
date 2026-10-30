# [MODULE DOC] Command line script that scans converted torch_np test files for leftovers of the
# pytest to PyTorch test conversion. It reports module-level test functions, test classes not
# deriving from TestCase, stray pytest marks and xfail/skip/param usage, and parametrize
# decorators that are outside a class or whose class lacks the instantiate_parametrized_tests
# decorator. Prints findings to stdout; takes one path argument (file or directory).
import sys
import textwrap
from pathlib import Path


def check(path):
    # [FUNCTION DOC] Scan one test file and print a report of conversion problems. Prints
    # the file name as a header, then walks every line (skipping comment lines) and calls
    # report_violation for each issue found. For parametrize decorators it walks backwards
    # to find the nearest class line with indent below 8 and checks that the line above that
    # class is the instantiate_parametrized_tests decorator. Returns nothing; only reads the
    # file.
    # KNOWN ISSUE: the backward search for the class may not find one, and the check of the
    # line before the class is then done on stale loop variables; behavior is heuristic.
    """Check a test file for common issues with pytest->pytorch conversion."""
    print(path.name)
    print("=" * len(path.name), "\n")

    src = path.read_text().split("\n")
    for num, line in enumerate(src):
        if is_comment(line):
            continue

        # module level test functions
        if line.startswith("def test"):
            report_violation(line, num, header="Module-level test function")

        # test classes must inherit from TestCase
        if line.startswith("class Test") and "TestCase" not in line:
            report_violation(
                line, num, header="Test class does not inherit from TestCase"
            )

        # last vestiges of pytest-specific stuff
        if "pytest.mark" in line:
            report_violation(line, num, header="pytest.mark.something")

        for part in ["pytest.xfail", "pytest.skip", "pytest.param"]:
            if part in line:
                report_violation(line, num, header=f"stray {part}")

        if textwrap.dedent(line).startswith("@parametrize"):
            # backtrack to check
            nn = num
            for nn in range(num, -1, -1):
                ln = src[nn]
                if "class Test" in ln:
                    # hack: large indent => likely an inner class
                    if len(ln) - len(ln.lstrip()) < 8:
                        break
            else:
                report_violation(line, num, "off-class parametrize")
            if not src[nn - 1].startswith("@instantiate_parametrized_tests"):
                report_violation(
                    line, num, f"missing instantiation of parametrized tests in {ln}?"
                )


def is_comment(line):
    # [FUNCTION DOC] Return True if the line, ignoring leading whitespace, starts with a
    # hash.
    return textwrap.dedent(line).startswith("#")


def report_violation(line, lineno, header):
    # [FUNCTION DOC] Print one violation: the zero-based line number, the header text and
    # the line.
    print(f">>>> line {lineno} : {header}\n {line}\n")


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 2:
        raise ValueError("Usage : python check_tests_conform path/to/file/or/dir")

    path = Path(argv[1])

    if path.is_dir():
        # run for all files in the directory (no subdirs)
        for this_path in path.glob("test*.py"):
            #   breakpoint()
            check(this_path)
    else:
        check(path)
