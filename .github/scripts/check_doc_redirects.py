#!/usr/bin/env python3
# [MODULE DOC] CI script that finds documentation source files moved or deleted relative to
# a base git ref and checks that docs/source/redirects.py has redirect entries for them.
# Reports missing ones, and with the auto-fix option appends entries for moved files to that
# file; exits nonzero if redirects are missing or deleted files need manual handling.
# No import-time side effects.
"""
Check for moved/deleted doc files and optionally auto-update redirects.py.

This script detects when documentation files in docs/source/ are moved or deleted
and verifies that corresponding redirects exist in docs/source/redirects.py.

Usage:
    # Check only (CI mode) - reports missing redirects
    python check_doc_redirects.py --base-ref origin/main

    # Auto-update redirects.py with missing entries
    python check_doc_redirects.py --base-ref origin/main --auto-fix
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path


def run_git(args: list[str]) -> str:
    # [FUNCTION DOC] HELPER: Runs git with the given arguments and returns stripped standard
    # output. The exit status is ignored, so failures yield empty or partial output.
    """Run a git command and return stdout."""
    result = subprocess.run(["git"] + args, capture_output=True, text=True)
    return result.stdout.strip()


def get_doc_changes(base_ref: str) -> list[tuple[str, str, str | None]]:
    # [FUNCTION DOC] Runs a git name-status diff with rename detection between the base ref
    # and HEAD limited to rst and md files directly under docs/source, and returns the
    # renames and deletions as (status, old path, new path or None) tuples.
    """
    Get moved/deleted doc files between base_ref and HEAD.

    Returns:
        List of (status, old_path, new_path) tuples.
        For deletions, new_path is None.
    """
    diff = run_git(
        [
            "diff",
            "--name-status",
            "-M",  # Enable rename detection
            f"{base_ref}...HEAD",
            "--",
            "docs/source/*.rst",
            "docs/source/*.md",
        ]
    )

    changes = []
    for line in diff.split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]

        # Renames: R100 (100% similar), R095 (95% similar), etc.
        if status.startswith("R") and len(parts) >= 3:
            changes.append((status, parts[1], parts[2]))
        # Deletions
        elif status == "D" and len(parts) >= 2:
            changes.append((status, parts[1], None))

    return changes


def path_to_key(path: str) -> str:
    # [FUNCTION DOC] HELPER: Converts a docs source path to its redirect key by removing the
    # docs/source prefix and the rst or md extension. Pure function.
    """
    Convert a file path to a redirect key.

    Example: docs/source/torch.compiler.rst -> torch.compiler
    Example: docs/source/user_guide/foo.rst -> user_guide/foo
    """
    return re.sub(r"\.(rst|md)$", "", path.replace("docs/source/", ""))


def path_to_url(path: str) -> str:
    # [FUNCTION DOC] HELPER: Converts a docs source path to the relative HTML page address by
    # removing the docs/source prefix and replacing the rst or md extension with html.
    """
    Convert a file path to an HTML URL for redirects.

    Example: docs/source/user_guide/foo.rst -> user_guide/foo.html
    """
    return re.sub(r"\.(rst|md)$", ".html", path.replace("docs/source/", ""))


def parse_existing_redirects(redirects_file: Path) -> dict[str, str]:
    # [FUNCTION DOC] Parses the redirects file as syntax tree (without executing it), finds
    # the assignment to the name redirects whose value is a dictionary literal and returns
    # the entries whose key and value are both constants. Returns an empty mapping if no such
    # assignment exists. Raises OSError or SyntaxError if the file cannot be read or parsed.
    """
    Parse redirects.py and return the existing redirects dictionary.

    Uses AST parsing for robustness.
    """
    content = redirects_file.read_text()
    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "redirects":
                    if isinstance(node.value, ast.Dict):
                        return {
                            k.value: v.value
                            for k, v in zip(node.value.keys, node.value.values)
                            if isinstance(k, ast.Constant)
                            and isinstance(v, ast.Constant)
                        }
    return {}


def find_missing_redirects(
    changes: list[tuple[str, str, str | None]],
    existing: dict[str, str],
) -> list[tuple[str, str | None]]:
    # [FUNCTION DOC] For each change, computes the old path's redirect key and, if it is not
    # already in the existing mapping, records (key, new page address or None for deletions).
    # Returns the list of missing entries in input order. Pure function.
    """
    Find file changes that don't have corresponding redirects.

    Returns:
        List of (old_key, new_url) tuples. new_url is None for deletions.
    """
    missing = []
    for status, old_path, new_path in changes:
        old_key = path_to_key(old_path)
        if old_key not in existing:
            new_url = path_to_url(new_path) if new_path else None
            missing.append((old_key, new_url))
    return missing


def update_redirects_file(
    redirects_file: Path, new_entries: list[tuple[str, str]]
) -> None:
    # [FUNCTION DOC] Rewrites the redirects file in place, inserting a comment line and one
    # line per (key, address) entry just before the last line that consists only of a
    # closing brace. Prints a confirmation. Exits the process with status 1 if no such line
    # exists. Entries are written as double-quoted strings without escaping.
    """
    Add new redirect entries to redirects.py.

    Inserts entries just before the closing brace of the redirects dict.
    """
    content = redirects_file.read_text()
    lines = content.split("\n")

    # Find the line with the closing brace
    insert_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "}":
            insert_idx = i
            break

    if insert_idx is None:
        print("Error: Could not find closing brace in redirects.py", file=sys.stderr)
        sys.exit(1)

    # Generate new entries
    new_lines = ["    # Auto-generated redirects for moved files"]
    for old_key, new_url in new_entries:
        new_lines.append(f'    "{old_key}": "{new_url}",')

    # Insert the new entries before the closing brace
    lines = lines[:insert_idx] + new_lines + lines[insert_idx:]

    redirects_file.write_text("\n".join(lines))
    print(f"✅ Added {len(new_entries)} redirect(s) to {redirects_file}")


def main() -> None:
    # [FUNCTION DOC] Command-line entry point with options for base ref, redirects file path
    # and auto-fix. Exits 1 if the redirects file is missing. Returns normally if there are
    # no moved or deleted docs or all have redirects. In auto-fix mode it appends entries for
    # moves (exiting 1 if deletions still need manual entries); otherwise it prints a report
    # with suggested additions and exits 1.
    parser = argparse.ArgumentParser(
        description="Check for missing doc redirects and optionally auto-fix"
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Base git ref to compare against (default: origin/main)",
    )
    parser.add_argument(
        "--redirects-file",
        default="docs/source/redirects.py",
        help="Path to redirects.py (default: docs/source/redirects.py)",
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically add missing redirects for moved files",
    )
    args = parser.parse_args()

    redirects_file = Path(args.redirects_file)
    if not redirects_file.exists():
        print(f"Error: {redirects_file} not found", file=sys.stderr)
        sys.exit(1)

    # Get file changes
    changes = get_doc_changes(args.base_ref)
    if not changes:
        print("✅ No doc files were moved or deleted")
        return

    print(f"Found {len(changes)} moved/deleted doc file(s)")

    # Parse existing redirects
    existing = parse_existing_redirects(redirects_file)
    print(f"Found {len(existing)} existing redirect(s)")

    # Find missing redirects
    missing = find_missing_redirects(changes, existing)

    if not missing:
        print("✅ All moved/deleted doc files have corresponding redirects")
        return

    # Separate auto-fixable (moves with known destination) from manual (deletes)
    auto_fixable = [(k, v) for k, v in missing if v is not None]
    manual_needed = [(k, v) for k, v in missing if v is None]

    # Auto-fix mode
    if args.auto_fix and auto_fixable:
        update_redirects_file(redirects_file, auto_fixable)

        if manual_needed:
            print(f"\n⚠️  {len(manual_needed)} deleted file(s) need manual redirects:")
            for old_key, _ in manual_needed:
                print(f"  • {old_key}")
            print("\nPlease add redirects for deleted files manually.")
            sys.exit(1)
        return

    # Report mode - show what's missing
    print("\n❌ Missing redirects detected!\n")

    for old_key, new_url in missing:
        if new_url:
            print(f"  • MOVED: {old_key} → {new_url}")
        else:
            print(f"  • DELETED: {old_key} (needs manual redirect target)")

    if auto_fixable:
        print("\n📝 Suggested additions to docs/source/redirects.py:\n")
        for old_key, new_url in auto_fixable:
            print(f'    "{old_key}": "{new_url}",')
        print("\n💡 To auto-fix, run:")
        print(
            f"    python3 .github/scripts/check_doc_redirects.py "
            f"--base-ref {args.base_ref} --auto-fix"
        )

    if manual_needed:
        print(f"\n⚠️  {len(manual_needed)} deleted file(s) need manual redirects.")
        print("Please determine appropriate redirect targets for deleted files.")

    sys.exit(1)


if __name__ == "__main__":
    main()
