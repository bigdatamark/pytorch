#!/usr/bin/env python3
# [MODULE DOC] Claude Code PostToolUse hook for the issue-triage skill. Reads the hook JSON
# from standard input and adds the bot-triaged label to the issue named in it via the gh
# CLI. Always exits 0 so it never blocks the agent. Import-time: reads two environment
# variables to set the debug log path.
"""
PostToolUse hook to automatically add the bot-triaged label after any issue mutation.

This hook runs after successful GitHub issue mutations (label/comment/close/transfer)
and directly applies the `bot-triaged` label via the gh CLI.

Exit codes:
  0 - Success
"""

import json
import os
import subprocess
import sys
from datetime import datetime


DEBUG_LOG = os.environ.get("TRIAGE_HOOK_DEBUG_LOG", "/tmp/triage_hooks.log")
BOT_TRIAGED_LABEL = "bot-triaged"


def debug_log(msg: str):
    # [FUNCTION DOC] HELPER: Appends a timestamped line to the debug log file (errors writing
    # it are ignored) and, if the verbose environment variable is set, echoes it to stderr.
    """Append a debug message to the log file and stderr (for CI visibility)."""
    timestamp = datetime.now().isoformat()
    formatted = f"[{timestamp}] [PostToolUse] {msg}"
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(formatted + "\n")
    except Exception:
        pass
    if os.environ.get("TRIAGE_HOOK_VERBOSE"):
        print(f"[DEBUG] {formatted}", file=sys.stderr)


def main():
    # [FUNCTION DOC] Hook entry point. Parses JSON from standard input, takes owner, repo
    # and issue_number from its tool input, and if all are present runs the gh CLI to add
    # the label (its result is only logged). Every path, including JSON and unexpected
    # errors, ends by exiting with status 0.
    try:
        data = json.load(sys.stdin)
        debug_log(f"Hook invoked with data: {json.dumps(data, indent=2)}")
        tool_input = data.get("tool_input", {})

        owner = tool_input.get("owner")
        repo = tool_input.get("repo")
        issue_number = tool_input.get("issue_number")

        if not all([owner, repo, issue_number]):
            debug_log(
                f"Missing required fields - owner={owner}, repo={repo}, issue_number={issue_number}"
            )
            sys.exit(0)

        cmd = [
            "gh",
            "issue",
            "edit",
            str(issue_number),
            "--repo",
            f"{owner}/{repo}",
            "--add-label",
            BOT_TRIAGED_LABEL,
        ]
        debug_log(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, check=False)
        debug_log(
            f"gh exit code: {result.returncode}, stderr: {result.stderr.decode()}"
        )
        sys.exit(0)

    except json.JSONDecodeError as e:
        debug_log(f"JSON decode error: {e}")
        sys.exit(0)
    except Exception as e:
        debug_log(f"Unexpected error: {e}")
        sys.exit(0)


if __name__ == "__main__":
    main()
