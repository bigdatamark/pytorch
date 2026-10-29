# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Loading of flight recorder dump files: read_dump loads one rank's pickle, read_dir loads every dump in a directory tree (inferring the filename prefix if needed). Creates the shared FlightRecorderLogger and a regular expression for trailing digits at import.
import argparse
import gc
import os
import pickle
import re
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from torch.distributed.flight_recorder.components.fr_logger import FlightRecorderLogger


__all__ = [
    "read_dump",
    "read_dir",
]


logger: FlightRecorderLogger = FlightRecorderLogger()


def read_dump(prefix: str, filename: str) -> dict[str, str | int | list[Any]]:
    # [FUNCTION DOC] Load one rank's dump. The rank is the integer after the prefix in the file base name. Unpickles the file (executes pickle loading, so only use trusted files) and returns a mapping with host_name (synthetic from rank), rank, entries, version and pg_config. Raises ValueError for a non-numeric rank suffix and KeyError for missing dump fields.
    basename = os.path.basename(filename)

    rank = int(basename[len(prefix) :])
    host_name = f"host_rank{rank}"

    with open(filename, "rb") as infile:
        dump = pickle.load(infile)

    entries = dump["entries"]
    version = dump["version"]
    pg_config = dump["pg_config"]

    return {
        "host_name": host_name,
        "rank": rank,
        "entries": entries,
        "version": version,
        "pg_config": pg_config,
    }


exp = re.compile(r"([\w\-\_]*?)(\d+)$")


def _determine_prefix(files: list[str]) -> str:
    # [FUNCTION DOC] HELPER: Infer the common filename prefix by matching a trailing number in each file name; returns the prefix if exactly one distinct prefix is found (logging it at debug level), otherwise raises ValueError asking the user to pass the prefix.
    """If the user doesn't specify a prefix, but does pass a dir full of similarly-prefixed files, we should be able to
    infer the common prefix most of the time.  But if we can't confidently infer, just fall back to requiring the user
    to specify it
    """
    possible_prefixes: defaultdict[str, set[int]] = defaultdict(set)
    for f in files:
        m = exp.search(f)
        if m:
            p, r = m.groups()
            possible_prefixes[p].add(int(r))
    if len(possible_prefixes) == 1:
        prefix = next(iter(possible_prefixes))
        logger.debug("Inferred common prefix %s", prefix)
        return prefix
    else:
        raise ValueError(
            "Unable to automatically determine the common prefix for the trace file names. "
            "Please specify --prefix argument manually"
        )


@contextmanager
def _disable_gc() -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Context manager that disables garbage collection for the block (to speed up loading large pickles) and re-enables it afterwards only if it was enabled before.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def read_dir(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], str]:
    # [FUNCTION DOC] Load all dumps under args.trace_dir with garbage collection disabled. Returns a tuple of (mapping from file name to the dump mapping from read_dump, version string taken from the first loaded file).
    # Uses args.prefix, inferring it from the first directory that has files when None. Files not containing the prefix are skipped. Raises AssertionError if the directory does not exist or nothing was loaded.
    with _disable_gc():
        prefix = args.prefix
        details = {}
        t0 = time.time()
        version = ""
        filecount = 0
        if not os.path.isdir(args.trace_dir):
            raise AssertionError(f"folder {args.trace_dir} does not exist")
        for root, _, files in os.walk(args.trace_dir):
            if prefix is None:
                prefix = _determine_prefix(files)
            for f in files:
                if (offset := f.find(prefix)) == -1:
                    continue
                details[f] = read_dump(f[:offset] + prefix, os.path.join(root, f))
                filecount += 1
                if not version:
                    version = str(details[f]["version"])
        tb = time.time()
        if len(details) <= 0:
            raise AssertionError(
                f"no files loaded from {args.trace_dir} with prefix {prefix}"
            )
        logger.debug("loaded %s files in %ss", filecount, tb - t0)
        return details, version
