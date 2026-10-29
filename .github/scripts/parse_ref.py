#!/usr/bin/env python3
# [MODULE DOC]
# Small GitHub Actions helper that reads the GITHUB_REF environment variable and publishes a
# branch or tag name as a step output. Run as a script; no import time side effects.


import os
import re


def set_output(name: str, val: str) -> None:
    # [FUNCTION DOC] HELPER: Prints the output being set, then appends name=value to the file
    # named by env var GITHUB_OUTPUT if set, otherwise prints the deprecated set-output
    # workflow command.
    print(f"Setting output {name}={val}")
    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print(f"{name}={val}", file=env)
    else:
        print(f"::set-output name={name}::{val}")


def main() -> None:
    # [FUNCTION DOC] Parses the GITHUB_REF env var (raises KeyError if unset). For refs of heads
    # sets output branch to the branch name; for pull refs sets branch to pull/ plus the PR
    # number; for tags sets output tag. Other refs set nothing.
    # KNOWN ISSUE: the pattern uses word characters for the category so a ref outside
    #     refs/<word>/ is silently ignored.
    ref = os.environ["GITHUB_REF"]
    m = re.match(r"^refs/(\w+)/(.*)$", ref)
    if m:
        category, stripped = m.groups()
        if category == "heads":
            set_output("branch", stripped)
        elif category == "pull":
            set_output("branch", "pull/" + stripped.split("/")[0])
        elif category == "tags":
            set_output("tag", stripped)


if __name__ == "__main__":
    main()
