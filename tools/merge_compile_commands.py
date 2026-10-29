"""Merge compile_commands.json from CMake build subdirectories.

Collects compile_commands.json files produced by Ninja and CMake sub-builds,
fixes gcc -> g++ for cquery compatibility, and writes a merged file to the
project root.

Called by cmake/PostBuildSteps.cmake as a build-time custom command.
"""

# [MODULE DOC]
# Build-time script that merges the compile_commands.json files from a CMake build directory
# into a single compile_commands.json in the project source directory. Main content: main.
# No import-time side effects.
import argparse
import itertools
import json
import pathlib


def main() -> None:
    # [FUNCTION DOC] Command line entry point taking the build directory and source directory.
    # Reads compile command files matching the compile commands pattern in the build directory
    # and in subdirectories of the torch lib build folder, concatenates all entries, rewrites
    # commands starting with gcc to use g++, and writes the merged JSON (indent 2) to the source
    # directory only if the file is missing or its content differs. Raises OSError or a JSON
    # error on bad input.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=pathlib.Path, help="CMake binary directory")
    parser.add_argument(
        "source_dir", type=pathlib.Path, help="Project source directory"
    )
    args = parser.parse_args()

    build = args.build_dir
    ninja = list(build.glob("*compile_commands.json"))
    cmake_sub_dir = build / "torch" / "lib" / "build"
    cmake_sub = (
        list(cmake_sub_dir.glob("*/compile_commands.json"))
        if cmake_sub_dir.exists()
        else []
    )

    cmds = [
        entry
        for f in itertools.chain(ninja, cmake_sub)
        for entry in json.loads(f.read_text(encoding="utf-8"))
    ]

    # cquery does not like C++ compiles that start with gcc — it forgets to
    # include the C++ header directories.  Replace with g++ as a workaround.
    for c in cmds:
        if c.get("command", "").startswith("gcc "):
            c["command"] = "g++ " + c["command"][4:]

    out = args.source_dir / "compile_commands.json"
    new = json.dumps(cmds, indent=2)
    if not out.exists() or out.read_text(encoding="utf-8") != new:
        out.write_text(new, encoding="utf-8")


if __name__ == "__main__":
    main()
