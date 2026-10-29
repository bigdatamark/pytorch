# mypy: allow-untyped-defs
# [MODULE DOC] Command line script that zips Python sources into an archive, skipping a deny
# list of standard library names. Defines the global strip_file_dir used for relative archive
# paths. Runs main only when executed as a script.
import argparse
import glob
import os
from pathlib import Path
from zipfile import ZipFile


# Exclude some standard library modules to:
# 1. Slim down the final zipped file size
# 2. Remove functionality we don't want to support.
DENY_LIST = [
    # Interface to unix databases
    "dbm",
    # ncurses bindings (terminal interfaces)
    "curses",
    # Tcl/Tk GUI
    "tkinter",
    "tkinter",
    # Tests for the standard library
    "test",
    "tests",
    "idle_test",
    "__phello__.foo.py",
    # importlib frozen modules. These are already baked into CPython.
    "_bootstrap.py",
    "_bootstrap_external.py",
]

strip_file_dir = ""


def remove_prefix(text, prefix):
    # [FUNCTION DOC] HELPER: Return text without the leading prefix if it starts with it, else
    # unchanged.
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def write_to_zip(file_path, strip_file_path, zf, prepend_str="") -> None:
    # [FUNCTION DOC] HELPER: Add one file to the open zip zf under its path with the global
    # strip directory removed and prepend_str added in front. Skips files whose base name is
    # in DENY_LIST.
    # KNOWN ISSUE: the strip_file_path parameter is unused (the global is used), and DENY_LIST
    #   entries for directories such as test are never matched because only the file name is
    #   checked; tkinter is listed twice.
    stripped_file_path = prepend_str + remove_prefix(file_path, strip_file_dir + "/")
    path = Path(stripped_file_path)
    if path.name in DENY_LIST:
        return
    zf.write(file_path, stripped_file_path)


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses command line arguments (paths, install dir, strip
    # dir, prepend string, zip name), sets the global strip directory, and writes a new zip at
    # install dir plus zip name containing every given file and every Python file found
    # recursively under given directories, in sorted order. Overwrites the output file.
    global strip_file_dir
    parser = argparse.ArgumentParser(description="Zip py source")
    parser.add_argument("paths", nargs="*", help="Paths to zip.")
    parser.add_argument(
        "--install-dir", "--install_dir", help="Root directory for all output files"
    )
    parser.add_argument(
        "--strip-dir",
        "--strip_dir",
        help="The absolute directory we want to remove from zip",
    )
    parser.add_argument(
        "--prepend-str",
        "--prepend_str",
        help="A string to prepend onto all paths of a file in the zip",
        default="",
    )
    parser.add_argument("--zip-name", "--zip_name", help="Output zip name")

    args = parser.parse_args()

    zip_file_name = args.install_dir + "/" + args.zip_name
    strip_file_dir = args.strip_dir
    prepend_str = args.prepend_str
    with ZipFile(zip_file_name, mode="w") as zf:
        for p in sorted(args.paths):
            if os.path.isdir(p):
                files = glob.glob(p + "/**/*.py", recursive=True)
                for file_path in sorted(files):
                    # strip the absolute path
                    write_to_zip(
                        file_path, strip_file_dir + "/", zf, prepend_str=prepend_str
                    )
            else:
                write_to_zip(p, strip_file_dir + "/", zf, prepend_str=prepend_str)


if __name__ == "__main__":
    main()  # pragma: no cover
