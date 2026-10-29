# [MODULE DOC]
# HELPER: Small utilities for the AutoHeuristic tests: reading a file into a string and running
# a bash script. No import-time side effects.
import subprocess


def read_file_to_string(file_path: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the full text content of the file at file_path. Raises
    # OSError if it cannot be opened.
    with open(file_path) as file:
        return file.read()


def run_bash(bash_script_path: str) -> None:
    # [FUNCTION DOC] HELPER: Runs the given script with bash, capturing output, and prints its
    # standard output. If the script exits non-zero, prints the error and its stderr instead of
    # raising, so callers cannot detect the failure. Side effect: runs a subprocess.
    try:
        print("Executing: ", bash_script_path)
        result = subprocess.run(
            ["bash", bash_script_path], capture_output=True, text=True, check=True
        )
        # Print the output
        print(f"Output of {bash_script_path}: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred executing {bash_script_path}: {e}")
        print("Error output:", e.stderr)
