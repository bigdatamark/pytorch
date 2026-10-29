# [MODULE DOC] Script that generates a GNU ld linker script which places a prioritized list of
# text symbols first in the text section, for link-time optimization. Uses the output of the
# linker verbose mode. Run as a command with optional --filein and --fout options; no
# import-time side effects.
import argparse
import os
import subprocess
from pathlib import Path


def gen_linker_script(
    filein: str = "cmake/prioritized_text.txt", fout: str = "cmake/linker_script.ld"
) -> None:
    # [FUNCTION DOC] Build a linker script from the linker's default script and a priority
    # list. Args: filein, path to a text file of symbol names (one per line, blank lines
    # ignored); fout, output path (parent directories are created). Runs the linker named by
    # the LD environment variable (default ld) in verbose mode, takes the default script
    # found between the two long separator lines, and writes it to fout with a block of text
    # sections for each prioritized symbol inserted two lines after the start of the text
    # section definition. Raises AssertionError unless exactly one text section start is
    # found; a failing linker call raises a subprocess error; a missing separator pair
    # raises an IndexError.
    with open(filein) as f:
        prioritized_text = f.readlines()
        prioritized_text = [
            line.replace("\n", "") for line in prioritized_text if line != "\n"
        ]
    ld = os.environ.get("LD", "ld")
    linker_script_lines = subprocess.check_output([ld, "-verbose"], text=True).split(
        "\n"
    )

    indices = [
        i
        for i, x in enumerate(linker_script_lines)
        if x == "=================================================="
    ]
    linker_script_lines = linker_script_lines[indices[0] + 1 : indices[1]]

    text_line_start = [
        i for i, line in enumerate(linker_script_lines) if ".text           :" in line
    ]
    if len(text_line_start) != 1:
        raise AssertionError("The linker script has multiple text sections!")
    text_line_start = text_line_start[0]

    # ensure that parent directory exists before writing
    # pyrefly: ignore [bad-assignment]
    fout = Path(fout)
    # pyrefly: ignore [missing-attribute]
    fout.parent.mkdir(parents=True, exist_ok=True)

    with open(fout, "w") as f:
        for lineid, line in enumerate(linker_script_lines):
            if lineid == text_line_start + 2:
                f.write("    *(\n")
                f.writelines(f"      .text.{plines}\n" for plines in prioritized_text)
                f.write("    )\n")
            f.write(f"{line}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate linker file based on prioritized symbols. Used for link-time optimization.",
    )
    parser.add_argument(
        "--filein",
        help="Path to prioritized_text.txt input file",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fout", help="Output path for linker ld file", default=argparse.SUPPRESS
    )
    # convert args to a dict to pass to gen_linker_script
    kwargs = vars(parser.parse_args())
    gen_linker_script(**kwargs)
