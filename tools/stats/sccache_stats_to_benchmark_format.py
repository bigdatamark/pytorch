# [MODULE DOC]
# Script converting sccache stats json files found at the repo root into a
# benchmark-format record list written under test/test-reports. Defines REPO_ROOT;
# no import-time side effects.

import glob
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def flatten_data(d: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Recursively flattens nested mappings into a single-level mapping whose keys
    # join the nesting path with underscores. Non-mapping values are kept as is.
    # Returns a new mapping; the input is not modified.
    # Flatten the sccache stats data from a possibly nested dictionary to a flat
    # dictionary.  For example, the input:
    # {
    #     "cache": {
    #         "hit": 1,
    #         "miss": 2,
    #     },
    # }
    # will be transformed to:
    # {
    #     "cache_hit": 1,
    #     "cache_miss": 2,
    # }
    flat_data = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for k, v in flatten_data(value).items():
                flat_data[f"{key}_{k}"] = v
        else:
            flat_data[key] = value
    return flat_data


def main() -> None:
    # [FUNCTION DOC]
    # For each sccache-stats json file in the repo root, builds one benchmark record
    # (name sccache_stats) whose value is the BUILD_TIME environment variable as an
    # integer (0 if unset or empty) and whose extra info is the flattened stats.
    # Writes all records as json to test/test-reports/sccache-stats.json, creating
    # the directory if needed. Raises on unreadable or invalid json.
    records = []
    for file in glob.glob(str(REPO_ROOT / "sccache-stats-*.json")):
        with open(file) as f:
            data = json.load(f)

            # I don't know what sccache info will be most useful yet, and the
            # sccache json has a decent number of keys, so just flatten the data
            # and store all of it
            records.append(
                {
                    "benchmark": {
                        "name": "sccache_stats",
                    },
                    "metric": {
                        "name": "sccache_stats",
                        "benchmark_values": [int(os.environ.get("BUILD_TIME") or 0)],
                        "extra_info": flatten_data(data),
                    },
                }
            )
    output_file = REPO_ROOT / "test" / "test-reports" / "sccache-stats.json"
    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(records, f)


if __name__ == "__main__":
    main()
