"""Collect instruction counts for continuous integration."""

# mypy: ignore-errors
# [MODULE DOC] Entry point logic for collecting instruction counts in continuous integration. Holds
# run constants (repeats, per-job timeout in seconds, retries, expected version and an expected md5
# of the work order list) and one function that runs every benchmark and returns nothing. Not
# runnable by itself; it is invoked from the package main script. Imports sibling packages core,
# definitions and execution, so it needs the benchmark directory on the path.

import argparse
import hashlib
import json
import time

from core.expand import materialize
from definitions.standard import BENCHMARKS
from execution.runner import Runner
from execution.work import WorkOrder


REPEATS = 5
TIMEOUT = 600  # Seconds
RETRIES = 2

VERSION = 0
MD5 = "4d55e8abf881ad38bb617a96714c1296"


def main(argv: list[str]) -> None:
    # [FUNCTION DOC] Runs the benchmark suite and optionally writes results. Parses a destination
    # file option and a subset flag from argv. Expands the standard benchmark definitions, repeats
    # them several times into work orders (with timeout and retries), and hashes the unique work
    # order keys. If the hash differs from the expected one (and not in subset mode) the version
    # becomes minus one and a warning is printed. Subset mode also uses version minus one and only
    # the first ten benchmarks. Runs the work through the Runner and groups wall times and
    # instruction counts per work order key. Writes a JSON document (version, md5, start and end
    # time, values) to the destination if given. In debug mode (subset, or no destination) prints a
    # truncated copy and stops in the pdb debugger. Returns None.
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, default=None)
    parser.add_argument("--subset", action="store_true")
    args = parser.parse_args(argv)

    t0 = int(time.time())
    version = VERSION
    benchmarks = materialize(BENCHMARKS)

    # Useful for local development, since e2e time for the full suite is O(1 hour)
    in_debug_mode = args.subset or args.destination is None
    if args.subset:
        version = -1
        benchmarks = benchmarks[:10]

    work_orders = tuple(
        WorkOrder(label, autolabels, timer_args, timeout=TIMEOUT, retries=RETRIES)
        for label, autolabels, timer_args in benchmarks * REPEATS
    )

    keys = tuple({str(work_order): None for work_order in work_orders}.keys())
    md5 = hashlib.md5(usedforsecurity=False)
    for key in keys:
        md5.update(key.encode("utf-8"))

    # Warn early, since collection takes a long time.
    if md5.hexdigest() != MD5 and not args.subset:
        version = -1
        print(f"WARNING: Expected {MD5}, got {md5.hexdigest()} instead")

    results = Runner(work_orders, cadence=30.0).run()

    # TODO: Annotate with TypedDict when 3.8 is the minimum supported version.
    grouped_results: dict[str, dict[str, list[float | int]]] = {
        key: {"times": [], "counts": []} for key in keys
    }

    for work_order, r in results.items():
        key = str(work_order)
        grouped_results[key]["times"].extend(r.wall_times)
        grouped_results[key]["counts"].extend(r.instructions)

    final_results = {
        "version": version,
        "md5": md5.hexdigest(),
        "start_time": t0,
        "end_time": int(time.time()),
        "values": grouped_results,
    }

    if args.destination:
        with open(args.destination, "w") as f:
            json.dump(final_results, f)

    if in_debug_mode:
        result_str = json.dumps(final_results)
        print(f"{result_str[:30]} ... {result_str[-30:]}\n")
        import pdb

        pdb.set_trace()
