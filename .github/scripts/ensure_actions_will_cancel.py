#!/usr/bin/env python3

# [MODULE DOC] CI lint script that checks every pull-request-triggered workflow in
# .github/workflows has a unique name, a concurrency group starting with the expected
# pattern (workflow name plus PR number or sha plus a dispatch-dependent suffix), and
# cancel-in-progress enabled. Problems are printed to standard error and the script exits with
# status 1 if any group or name problem was found. Constants hold the expected group strings.
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
EXPECTED_GROUP_PREFIX = (
    "${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}"
)
# Standard pattern - dispatches in same concurrency group will cancel each other
EXPECTED_GROUP_STANDARD = (
    EXPECTED_GROUP_PREFIX + "-${{ github.event_name == 'workflow_dispatch' }}"
)
# Concurrent dispatch pattern - uses run_id to allow concurrent dispatches (e.g., from autorevert bot)
EXPECTED_GROUP_CONCURRENT = (
    EXPECTED_GROUP_PREFIX
    + "-${{ github.event_name == 'workflow_dispatch' && github.run_id }}"
)


def should_check(filename: Path) -> bool:
    # [FUNCTION DOC] HELPER: Loads a workflow yml file and returns true if its triggers contain
    # a pull request trigger. The trigger key is looked up as on, falling back to the boolean
    # key that YAML produces for the bare word on. Raises if the file cannot be read or parsed.
    with open(filename) as f:
        content = f.read()

    data = yaml.safe_load(content)
    on = data.get("on", data.get(True, {}))
    return "pull_request" in on


if __name__ == "__main__":
    # KNOWN ISSUE: a missing or false cancel-in-progress setting prints an error below but does
    # not set the error flag, so it never causes a non-zero exit.
    errors_found = False
    files = [f for f in WORKFLOWS.glob("*.yml") if should_check(f)]
    names = set()
    for filename in files:
        with open(filename) as f:
            data = yaml.safe_load(f)

        name = data.get("name")
        if name is not None and name in names:
            print("ERROR: duplicate workflow name:", name, file=sys.stderr)
            errors_found = True
        names.add(name)
        actual = data.get("concurrency", {})
        if filename.name == "create_release.yml":
            if not actual.get("group", "").startswith(EXPECTED_GROUP_PREFIX):
                print(
                    f"'concurrency' incorrect or not found in '{filename.relative_to(REPO_ROOT)}'",
                    file=sys.stderr,
                )
                print(
                    f"concurrency group should start with {EXPECTED_GROUP_PREFIX} but found {actual.get('group', None)}",
                    file=sys.stderr,
                )
                errors_found = True
        elif not (
            actual.get("group", "").startswith(EXPECTED_GROUP_STANDARD)
            or actual.get("group", "").startswith(EXPECTED_GROUP_CONCURRENT)
        ):
            print(
                f"'concurrency' incorrect or not found in '{filename.relative_to(REPO_ROOT)}'",
                file=sys.stderr,
            )
            print(
                f"concurrency group should start with {EXPECTED_GROUP_STANDARD} "
                f"or {EXPECTED_GROUP_CONCURRENT} but found {actual.get('group', None)}",
                file=sys.stderr,
            )
            errors_found = True
        if not actual.get("cancel-in-progress", False):
            print(
                f"'concurrency' incorrect or not found in '{filename.relative_to(REPO_ROOT)}'",
                file=sys.stderr,
            )
            print(
                f"concurrency cancel-in-progress should be True but found {actual.get('cancel-in-progress', None)}",
                file=sys.stderr,
            )

    if errors_found:
        sys.exit(1)
