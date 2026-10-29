# [MODULE DOC]
# Command line script that reads a template file, applies literal string
# replacements and writes the result into an install directory. Run as a script
# via main; no import-time side effects.

import argparse
import os
import os.path


def main() -> None:
    # [FUNCTION DOC]
    # Parses --input-file, --output-file, --install-dir and repeated --replace
    # OLD NEW pairs. Reads the input file, applies each replacement in order, creates
    # the output's parent directories under the install directory and writes the
    # result there. Raises if required arguments are missing.
    # KNOWN ISSUE: omitting --replace leaves the replacement list unset, so the
    # loop over it fails with a type error instead of doing no replacements.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file")
    parser.add_argument("--output-file")
    parser.add_argument("--install-dir", "--install_dir")
    parser.add_argument("--replace", action="append", nargs=2)
    options = parser.parse_args()

    with open(options.input_file) as f:
        contents = f.read()

    output_file = os.path.join(options.install_dir, options.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for old, new in options.replace:
        contents = contents.replace(old, new)

    with open(output_file, "w") as f:
        f.write(contents)


if __name__ == "__main__":
    main()  # pragma: no cover
