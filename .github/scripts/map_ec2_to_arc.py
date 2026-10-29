#!/usr/bin/env python3
# [MODULE DOC]
# Command line tool that rewrites the runner labels of a GitHub Actions test matrix from EC2
# runner names to ARC equivalents, using runner_mapping and meta_only_runners from arc.yaml, and
# publishes the result as the test-matrix step output. Import time side effects: none besides
# imports (the PyYAML package is required).
"""Map EC2 runner labels to ARC equivalents using .github/arc.yaml.

Takes a GitHub Actions test matrix, replaces each runner with its ARC
equivalent, and prints the updated matrix as JSON.

Usage:
    python map_ec2_to_arc.py --prefix mt- '{ include: [
      { config: "default", shard: 1, num_shards: 5, runner: "mt-linux.4xlarge" },
    ]}'
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    # [FUNCTION DOC] Parses command line arguments: positional matrix string and optional prefix
    # (default empty) naming a runner prefix to strip. Returns the argparse namespace; exits on
    # bad arguments.
    parser = argparse.ArgumentParser(
        description="Map EC2 runner labels to ARC runner labels in a test matrix"
    )
    parser.add_argument(
        "matrix",
        help="GitHub Actions test matrix string to transform",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Runner prefix to strip from labels (e.g. 'mt-')",
    )
    return parser.parse_args()


def strip_prefix(label: str, prefix: str) -> str:
    # [FUNCTION DOC] Returns the label with the prefix removed if the prefix is non-empty and
    # the label starts with it, otherwise the label unchanged.
    if prefix and label.startswith(prefix):
        return label[len(prefix) :]
    return label


def load_mapping(arc_yaml: Path) -> dict[str, str]:
    # [FUNCTION DOC] Loads arc.yaml and returns the runner_mapping dictionary (EC2 label to ARC
    # label). Raises KeyError if the key is missing.
    with open(arc_yaml) as f:
        data = yaml.safe_load(f)
    return data["runner_mapping"]


def load_meta_only(arc_yaml: Path) -> set[str]:
    # [FUNCTION DOC] Loads arc.yaml and returns the set of runner labels listed under
    # meta_only_runners (empty set if absent or empty).
    with open(arc_yaml) as f:
        data = yaml.safe_load(f)
    return set(data.get("meta_only_runners") or [])


def set_output(name: str, val: str) -> None:
    # [FUNCTION DOC] Prints the name and value and, if env var GITHUB_OUTPUT is set, appends
    # name=value to that file.
    print(f"Setting {name}={val}")
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            print(f"{name}={val}", file=f)


def main() -> None:
    # [FUNCTION DOC] Entry point. Loads the mapping from arc.yaml next to the scripts directory,
    # parses the matrix argument as yaml, and outputs it unchanged if empty or without include
    # entries. Otherwise drops entries whose config is onnx, then maps each entry runner
    # (whitespace and prefix stripped): exits with status 1 and an error on stderr if no mapping
    # exists; meta only runners get the mt- prefix; passthrough runners (mapped equal to
    # original) keep the label; others get the prefix added. Prints the result as json in the
    # test-matrix output.
    args = parse_args()
    arc_yaml = Path(__file__).resolve().parent.parent / "arc.yaml"
    mapping = load_mapping(arc_yaml)
    meta_only = load_meta_only(arc_yaml)

    matrix = yaml.safe_load(args.matrix)
    if not matrix:
        set_output("test-matrix", args.matrix)
        return

    entries = matrix.get("include", [])
    if not entries:
        set_output("test-matrix", json.dumps(matrix))
        return

    # TODO(huydo): onnxruntime uses hardware_concurrency() to size its thread
    # pool, which sees all host CPUs (e.g., 192) on ARC k8s instead of the
    # container's cpuset (e.g., 16). This causes pthread_setaffinity_np errors.
    # Skip onnx tests on ARC until the onnxruntime session options are fixed to
    # use cgroup-aware CPU counts.
    excluded_configs = {"onnx"}
    filtered = []
    for entry in entries:
        if entry.get("config") in excluded_configs:
            print(f"Excluding config '{entry['config']}' from ARC test matrix")
            continue
        filtered.append(entry)
    matrix["include"] = filtered

    for entry in filtered:
        if "runner" not in entry:
            continue
        clean = strip_prefix(entry["runner"].strip(), args.prefix)
        if clean not in mapping:
            print(f"error: no ARC runner found for '{clean}'", file=sys.stderr)
            sys.exit(1)
        mapped = mapping[clean]
        # Some hardware (H100, B200) exists only on the Meta fleet, so force the
        # 'mt-' prefix regardless of the build job's fleet assignment.
        if clean in meta_only:
            entry["runner"] = "mt-" + mapped
            continue
        # Passthrough runners (e.g. linux.rocm.gpu.2, linux.idc.xpu) are not
        # OSDC-managed so they keep their original label without the prefix.
        entry["runner"] = mapped if mapped == clean else args.prefix + mapped

    set_output("test-matrix", json.dumps(matrix))


if __name__ == "__main__":
    main()
