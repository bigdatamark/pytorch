# [MODULE DOC] Script that downloads the MNIST dataset archives from an S3 mirror (with retries
# and a progress bar) and unzips them into a destination directory. Run as a script it calls
# main. Defines the mirror list, retry settings and resource file names.
import argparse
import gzip
import os
import socket
import sys
import time
from http.client import HTTPException
from urllib.request import urlretrieve


MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",  # @lint-ignore
]

# Number of attempts per mirror before giving up on it. The download hits a
# single S3 mirror with no built-in retry, so a transient network blip fails
# the whole job; retry a few times with backoff first.
RETRIES = 3
RETRY_BACKOFF_S = 5

RESOURCES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def report_download_progress(
    chunk_number: int,
    chunk_size: int,
    file_size: int,
) -> None:
    # [FUNCTION DOC] Reporthook for urlretrieve: when the total size is known, writes a
    # carriage-return progress bar of up to 64 characters and a percentage to stdout (no
    # flush, no newline). Does nothing when the size is unknown (-1).
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write(f"\r0% |{bar:<64}| {int(percent * 100)}%")


def download(destination_path: str, resource: str, quiet: bool) -> None:
    # [FUNCTION DOC] Downloads one resource from the mirrors to destination_path unless the
    # file already exists. Each mirror is tried up to RETRIES times with growing sleep between
    # attempts; a partial file left by a failed attempt is deleted. Prints progress unless
    # quiet. Returns on first success; raises RuntimeError if every attempt failed.
    if os.path.exists(destination_path):
        if not quiet:
            print(f"{destination_path} already exists, skipping ...")
    else:
        for mirror in MIRRORS:
            url = mirror + resource
            for attempt in range(1, RETRIES + 1):
                print(f"Downloading {url} ...")
                try:
                    hook = None if quiet else report_download_progress
                    urlretrieve(url, destination_path, reporthook=hook)
                    return
                except (OSError, HTTPException) as e:
                    # A failed urlretrieve leaves a partial file behind; remove
                    # it so the exists check above doesn't skip it on a rerun.
                    if os.path.exists(destination_path):
                        os.remove(destination_path)
                    print(f"Failed to download (attempt {attempt}/{RETRIES}):\n{e}")
                    if attempt < RETRIES:
                        time.sleep(RETRY_BACKOFF_S * attempt)
                finally:
                    if not quiet:
                        # Just a newline.
                        print()
        raise RuntimeError("Error downloading resource!")


def unzip(zipped_path: str, quiet: bool) -> None:
    # [FUNCTION DOC] Decompresses a gzip file next to itself, naming the output by dropping the
    # last extension. Skips if the output already exists. Reads the whole file into memory.
    # Prints messages unless quiet.
    unzipped_path = os.path.splitext(zipped_path)[0]
    if os.path.exists(unzipped_path):
        if not quiet:
            print(f"{unzipped_path} already exists, skipping ... ")
        return
    with gzip.open(zipped_path, "rb") as zipped_file:
        with open(unzipped_path, "wb") as unzipped_file:
            unzipped_file.write(zipped_file.read())
            if not quiet:
                print(f"Unzipped {zipped_path} ...")


def main() -> None:
    # [FUNCTION DOC] Entry point: parses the destination directory and quiet options, sets a
    # global 60 second socket timeout, creates the destination directory if needed, then
    # downloads and unzips each MNIST resource. A keyboard interrupt prints a message and
    # ends normally; download errors propagate as RuntimeError.
    parser = argparse.ArgumentParser(
        description="Download the MNIST dataset from the internet"
    )
    parser.add_argument(
        "-d", "--destination", default=".", help="Destination directory"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Don't report about progress"
    )
    options = parser.parse_args()

    # urlretrieve has no timeout parameter and defaults to blocking forever on
    # a stalled connection, which would defeat the retry logic in download().
    socket.setdefaulttimeout(60)

    if not os.path.exists(options.destination):
        os.makedirs(options.destination)

    try:
        for resource in RESOURCES:
            path = os.path.join(options.destination, resource)
            download(path, resource, options.quiet)
            unzip(path, options.quiet)
    except KeyboardInterrupt:
        print("Interrupted")


if __name__ == "__main__":
    main()
