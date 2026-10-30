# [MODULE DOC] Script that reports a benchmarking duration to the open source signpost scribe
# logger under the pr_time_benchmarks subsystem.
import json
import sys

import torch._logging.scribe as scribe


def main():
    # [FUNCTION DOC] Entry point. Parses the first command line argument as an integer duration
    # and sends it as a JSON-encoded parameter of a signpost named duration. Raises an error if
    # the argument is missing or not an integer.
    duration = int(sys.argv[1])
    scribe.open_source_signpost(
        subsystem="pr_time_benchmarks",
        name="duration",
        parameters=json.dumps(duration),
    )


if __name__ == "__main__":
    main()
