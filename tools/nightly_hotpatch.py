#!/usr/bin/env python3
# [MODULE DOC] Command line tool that downloads the diff of a PyTorch GitHub pull request
# and applies it with the system patch utility to the installed PyTorch package (or a
# user-given directory). Intended for hot-patching disposable environments. No import-time
# side effects; the work happens in main when run as a script.

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from typing import cast, NoReturn


def parse_arguments() -> argparse.Namespace:
    # [FUNCTION DOC] Parses the command line: a required integer PR number, optional
    # target directory and strip count (default 1). Returns the namespace; argparse exits
    # on invalid input.
    """
    Parses command-line arguments using argparse.

    Returns:
        argparse.Namespace: The parsed arguments containing the PR number, optional target directory, and strip count.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download and apply a Pull Request (PR) patch from the PyTorch GitHub repository "
            "to your local PyTorch installation.\n\n"
            "Best Practice: Since this script involves hot-patching PyTorch, it's recommended to use "
            "a disposable environment like a Docker container or a dedicated Python virtual environment (venv). "
            "This ensures that if the patching fails, you can easily recover by resetting the environment."
        ),
        epilog=(
            "Example:\n"
            "  python nightly_hotpatch.py 12345\n"
            "  python nightly_hotpatch.py 12345 --directory /path/to/pytorch --strip 1\n\n"
            "These commands will download the patch for PR #12345 and apply it to your local "
            "PyTorch installation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "PR_NUMBER",
        type=int,
        help="The number of the Pull Request (PR) from the PyTorch GitHub repository to download and apply as a patch.",
    )

    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=None,
        help="Optional. Specify the target directory to apply the patch. "
        "If not provided, the script will use the PyTorch installation path.",
    )

    parser.add_argument(
        "--strip",
        "-p",
        type=int,
        default=1,
        help="Optional. Specify the strip count to remove leading directories from file paths in the patch. Default is 1.",
    )

    return parser.parse_args()


def get_pytorch_path() -> str:
    # [FUNCTION DOC] Imports torch and returns the parent directory of the installed torch
    # package (where patch paths are relative to), printing both paths. If torch cannot be
    # imported, exits the program with status 1 through the import error handler.
    """
    Retrieves the installation path of PyTorch in the current environment.

    Returns:
        str: The directory of the PyTorch installation.

    Exits:
        If PyTorch is not installed in the current Python environment, the script will exit.
    """
    try:
        import torch

        torch_paths: list[str] = cast(list[str], torch.__path__)
        torch_path: str = torch_paths[0]
        parent_path: str = os.path.dirname(torch_path)
        print(f"PyTorch is installed at: {torch_path}")
        print(f"Parent directory for patching: {parent_path}")
        return parent_path
    except ImportError:
        handle_import_error()


def handle_import_error() -> NoReturn:
    # [FUNCTION DOC] HELPER: Prints a message that PyTorch is not installed and exits the
    # process with status 1.
    """
    Handle the case where PyTorch is not installed and exit the program.

    Exits:
        NoReturn: This function will terminate the program.
    """
    print("Error: PyTorch is not installed in the current Python environment.")
    sys.exit(1)


def download_patch(pr_number: int, repo_url: str, download_dir: str) -> str:
    # [FUNCTION DOC] Downloads the diff of the pull request from the repository URL into
    # a file in the download directory and returns that file path. Network and file I/O.
    # Exits with status 1 on an HTTP error or any other failure.
    # KNOWN ISSUE: the error module is referenced through urllib without being imported
    # explicitly; it works only because the request module imports it.
    """
    Downloads the patch file for a given PR from the specified GitHub repository.

    Args:
        pr_number (int): The pull request number.
        repo_url (str): The URL of the repository where the PR is hosted.
        download_dir (str): The directory to store the downloaded patch.

    Returns:
        str: The path to the downloaded patch file.

    Exits:
        If the download fails, the script will exit.
    """
    patch_url = f"{repo_url}/pull/{pr_number}.diff"
    patch_file = os.path.join(download_dir, f"pr-{pr_number}.patch")
    print(f"Downloading PR #{pr_number} patch from {patch_url}...")
    try:
        with (
            urllib.request.urlopen(patch_url) as response,
            open(patch_file, "wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)
        if not os.path.isfile(patch_file):
            print(f"Failed to download patch for PR #{pr_number}")
            sys.exit(1)
        print(f"Patch downloaded to {patch_file}")
        return patch_file
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} when downloading patch for PR #{pr_number}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while downloading the patch: {e}")
        sys.exit(1)


def apply_patch(patch_file: str, target_dir: str | None, strip_count: int) -> None:
    # [FUNCTION DOC] Runs the patch utility with the given strip count on the patch file,
    # inside the target directory, printing progress and any output on failure. Exits
    # with status 1 if patch fails, is missing, or any error occurs.
    # KNOWN ISSUE: when no target directory is given, the else branch still adds a
    # directory option containing the text None, so patch would target a directory named
    # None; the branches are otherwise identical.
    """
    Applies the downloaded patch to the specified directory using the given strip count.

    Args:
        patch_file (str): The path to the patch file.
        target_dir (Optional[str]): The directory to apply the patch to. If None, uses PyTorch installation path.
        strip_count (int): The number of leading directories to strip from file paths in the patch.

    Exits:
        If the patch command fails or the 'patch' utility is not available, the script will exit.
    """
    if target_dir:
        print(f"Applying patch in directory: {target_dir}")
    else:
        print("No target directory specified. Using PyTorch installation path.")

    print(f"Applying patch with strip count: {strip_count}")
    try:
        # Construct the patch command with -d and -p options
        patch_command = ["patch", f"-p{strip_count}", "-i", patch_file]

        if target_dir:
            patch_command.insert(
                1, f"-d{target_dir}"
            )  # Insert -d option right after 'patch'
            print(f"Running command: {' '.join(patch_command)}")
            result = subprocess.run(patch_command, capture_output=True, text=True)
        else:
            patch_command.insert(1, f"-d{target_dir}")
            print(f"Running command: {' '.join(patch_command)}")
            result = subprocess.run(patch_command, capture_output=True, text=True)

        # Check if the patch was applied successfully
        if result.returncode != 0:
            print("Failed to apply patch.")
            print("Patch output:")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)
        else:
            print("Patch applied successfully.")
    except FileNotFoundError:
        print("Error: The 'patch' utility is not installed or not found in PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while applying the patch: {e}")
        sys.exit(1)


def main() -> None:
    # [FUNCTION DOC] Orchestrates the tool: parses arguments, picks the target directory
    # (the custom one, which must exist or the program exits with status 1, else the
    # installed torch parent), downloads the PR patch into a temporary directory and
    # applies it. The temporary directory is removed afterwards.
    """
    Main function to orchestrate the patch download and application process.

    Steps:
        1. Parse command-line arguments to get the PR number, optional target directory, and strip count.
        2. Retrieve the local PyTorch installation path or use the provided target directory.
        3. Download the patch for the provided PR number.
        4. Apply the patch to the specified directory with the given strip count.
    """
    args = parse_arguments()
    pr_number = args.PR_NUMBER
    custom_target_dir = args.directory
    strip_count = args.strip

    if custom_target_dir:
        if not os.path.isdir(custom_target_dir):
            print(
                f"Error: The specified target directory '{custom_target_dir}' does not exist."
            )
            sys.exit(1)
        target_dir = custom_target_dir
        print(f"Using custom target directory: {target_dir}")
    else:
        target_dir = get_pytorch_path()

    repo_url = "https://github.com/pytorch/pytorch"

    with tempfile.TemporaryDirectory() as tmpdirname:
        patch_file = download_patch(pr_number, repo_url, tmpdirname)
        apply_patch(patch_file, target_dir, strip_count)


if __name__ == "__main__":
    main()
