#!/usr/bin/env python3
# [MODULE DOC] Claude Code PreToolUse hook for the issue-triage skill. Reads hook JSON from
# standard input, filters the requested labels (forbidden, unknown, redundant), merges the
# rest with the labels already on the issue and replies on standard output with an allow
# decision carrying rewritten tool input, or exits with status 2 to block. Import-time:
# reads the debug log path and defines label lists and file locations.
# KNOWN ISSUE: the docstring names the update_issue tool, but the code never checks the tool
# name, so it applies to whatever tool the hook is configured for.
"""
PreToolUse hook to validate and sanitize labels during triage.

This hook intercepts mcp__github__update_issue calls and:
1. Strips forbidden labels (CI/infrastructure/severity)
2. Strips non-existent labels (not in labels.json)
3. Strips redundant labels (e.g. module: nn when module: rnn is present)
4. Fetches existing labels on the issue and merges with valid new labels
5. Rewrites the tool input via updatedInput so the MCP SET preserves existing labels

Exit codes:
  0 - Allow the tool call (with possible input rewrite via stdout JSON)
  2 - Block the tool call (feedback sent to Claude)
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEBUG_LOG = os.environ.get("TRIAGE_HOOK_DEBUG_LOG", "/tmp/triage_hooks.log")
SCRIPT_DIR = Path(__file__).parent
LABELS_FILE = SCRIPT_DIR.parent / "labels.json"
DISTRIBUTED_LABELS_FILE = (
    SCRIPT_DIR.parent.parent / "distributed-triage" / "distributed-labels.json"
)

FORBIDDEN_PATTERNS = [
    r"^ciflow/",
    r"^test-config/",
    r"^release notes:",
    r"^ci-",
    r"^ci:",
    r"^sev",
    r"deprecated",
]

FORBIDDEN_EXACT = [
    "actionable",
    "merge blocking",
    "needs design",
    "needs reproduction",
    "needs research",
    "oncall: releng",  # Not a triage redirect target; use module: ci instead
]

REDUNDANT_PAIRS = [
    ("module: rnn", "module: nn"),
]


def debug_log(msg: str, to_stderr: bool = False):
    # [FUNCTION DOC] HELPER: Appends a timestamped line to the debug log file (write errors
    # ignored) and echoes it to stderr when asked to or when the verbose variable is set.
    timestamp = datetime.now().isoformat()
    formatted = f"[{timestamp}] [PreToolUse] {msg}"
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(formatted + "\n")
    except Exception:
        pass
    if to_stderr or os.environ.get("TRIAGE_HOOK_VERBOSE"):
        print(f"[DEBUG] {formatted}", file=sys.stderr)


def is_forbidden(label: str) -> bool:
    # [FUNCTION DOC] HELPER: Returns true if the label, compared case-insensitively, matches
    # any forbidden regular expression (searched anywhere in the label) or equals a
    # forbidden exact name. Pure function.
    label_lower = label.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, label_lower):
            return True
    return label_lower in [f.lower() for f in FORBIDDEN_EXACT]


def _load_labels_from_file(path: Path) -> set[str]:
    # [FUNCTION DOC] HELPER: Reads a JSON file and returns the set of label names from its
    # labels list. Raises RuntimeError if the file is missing, unreadable, not valid JSON,
    # or has entries without a name.
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Labels file not found at {path}") from None
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Labels file contains invalid JSON: {e}") from None
    except PermissionError:
        raise RuntimeError(
            f"Cannot read labels file: permission denied: {path}"
        ) from None

    labels_list = data.get("labels", [])
    try:
        return {label["name"] for label in labels_list}
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"Labels file has malformed entries: {e}: {path}") from None


def load_valid_labels() -> set[str]:
    # [FUNCTION DOC] Returns the set of all label names allowed for triage: those in the main
    # labels file plus, if that file exists, those in the distributed labels file. Raises
    # RuntimeError (from the loader) if a file cannot be read or parsed.
    labels = _load_labels_from_file(LABELS_FILE)
    if DISTRIBUTED_LABELS_FILE.exists():
        labels |= _load_labels_from_file(DISTRIBUTED_LABELS_FILE)
    return labels


def strip_redundant(labels: list[str]) -> tuple[list[str], list[str]]:
    # [FUNCTION DOC] Drops the general label of each specific/general pair when both appear
    # in the list. Returns (labels kept in original order, sorted list of removed labels).
    labels_set = set(labels)
    to_remove = set()
    for specific, general in REDUNDANT_PAIRS:
        if specific in labels_set and general in labels_set:
            to_remove.add(general)
    return [l for l in labels if l not in to_remove], sorted(to_remove)


def fetch_existing_labels(owner: str, repo: str, issue_number: int) -> list[str]:
    # [FUNCTION DOC] Runs the gh CLI (15 second timeout) to view the issue and returns the
    # names of the labels currently on it. Raises RuntimeError on a nonzero exit, and
    # TimeoutExpired or JSON errors propagate to the caller.
    result = subprocess.run(
        [
            "gh",
            "issue",
            "view",
            str(issue_number),
            "--repo",
            f"{owner}/{repo}",
            "--json",
            "labels",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Cannot fetch existing labels (gh exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    data = json.loads(result.stdout)
    return [label["name"] for label in data.get("labels", [])]


def allow_with_updated_input(tool_input: dict, merged_labels: list[str]) -> None:
    # [FUNCTION DOC] Prints to standard output a hook response that allows the tool call
    # with a copy of the input whose labels are replaced by merged_labels, then exits the
    # process with status 0. Never returns.
    updated = dict(tool_input)
    updated["labels"] = merged_labels
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": updated,
            }
        },
        sys.stdout,
    )
    sys.exit(0)


def main():
    # [FUNCTION DOC] Hook entry point. Reads hook JSON from standard input. With no labels
    # requested, exits 0 (allow unchanged). Otherwise requires an issue number, strips
    # forbidden labels (adding the triage review label for a human), unknown labels and
    # redundant ones, exits 2 if nothing is left, else merges with the issue's existing
    # labels and emits the rewritten input. Any error exits 2 after printing to stderr.
    try:
        data = json.load(sys.stdin)
        debug_log(f"Hook invoked with data: {json.dumps(data, indent=2)}")
        tool_input = data.get("tool_input", {})

        requested_labels = tool_input.get("labels", []) or []
        debug_log(f"Labels requested: {requested_labels}")

        if not requested_labels:
            debug_log("No labels provided, allowing")
            sys.exit(0)

        owner = tool_input.get("owner", "pytorch")
        repo = tool_input.get("repo", "pytorch")
        issue_number = tool_input.get("issue_number")
        if not issue_number:
            raise RuntimeError("tool_input missing issue_number")

        forbidden = [l for l in requested_labels if is_forbidden(l)]
        clean_labels = [l for l in requested_labels if not is_forbidden(l)]

        if forbidden:
            debug_log(f"Stripped forbidden labels: {forbidden}")
            if not clean_labels:
                clean_labels = ["triage review"]
            elif "triage review" not in clean_labels:
                clean_labels.append("triage review")
            print(
                f"Stripped forbidden labels (require human decision): {forbidden}. "
                f"Added 'triage review' for human attention.",
                file=sys.stderr,
            )

        valid_labels = load_valid_labels()
        nonexistent = [l for l in clean_labels if l not in valid_labels]
        clean_labels = [l for l in clean_labels if l in valid_labels]

        if nonexistent:
            debug_log(f"Stripped non-existent labels: {nonexistent}")
            print(
                f"Stripped non-existent labels: {nonexistent}",
                file=sys.stderr,
            )

        clean_labels, removed_redundant = strip_redundant(clean_labels)
        if removed_redundant:
            debug_log(f"Stripped redundant labels: {removed_redundant}")
            print(
                f"Stripped redundant labels: {removed_redundant}",
                file=sys.stderr,
            )

        if not clean_labels:
            debug_log("No valid labels remain after filtering, blocking")
            print(
                "All requested labels were invalid. No labels to apply.",
                file=sys.stderr,
            )
            sys.exit(2)

        existing_labels = fetch_existing_labels(owner, repo, issue_number)
        debug_log(f"Existing labels on issue: {existing_labels}")

        merged = sorted(set(existing_labels) | set(clean_labels))
        debug_log(f"Merged labels (existing + new): {merged}")

        allow_with_updated_input(tool_input, merged)

    except json.JSONDecodeError as e:
        debug_log(f"JSON decode error: {e}")
        print(f"Hook error: Invalid JSON input: {e}", file=sys.stderr)
        print(
            "Hook was unable to validate labels; stopping triage.",
            file=sys.stderr,
        )
        sys.exit(2)
    except Exception as e:
        debug_log(f"Unexpected error: {type(e).__name__}: {e}")
        print(f"Hook error: {e}", file=sys.stderr)
        print(
            "Hook was unable to validate labels; stopping triage.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
