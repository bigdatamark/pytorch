#!/usr/bin/env python3
# [MODULE DOC] Lintrunner adapter for torchgen/aoti/fallback_ops.py: parses the file's
# module-level op dictionaries with the ast module, compares them with the copy at the git
# merge-base with main, and reports new ops or variants lacking a correct since version and
# shipped ops whose since value changed. Import-time side effects: puts the repo root first
# on sys.path and imports shared helpers from the stable shim utilities module.
"""
GENERATED_SHIMS_VERSION: Ensures that newly added entries (or vN variants) in
torchgen/aoti/fallback_ops.py carry a "since" key set to the current
TORCH_VERSION, and that "since" values on already-shipped entries are not
modified.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.linter.adapters._stable_shim_utils import (
    get_current_version,
    LintMessage,
    LintSeverity,
    merge_base_with_main,
    read_file_at_revision,
)


LINTER_CODE = "GENERATED_SHIMS_VERSION"

_TORCH_VERSION_PATTERN = re.compile(r"^TORCH_VERSION_\d+_\d+_\d+$")

# {op_name: {op_version: (since_value_or_None, lineno)}}
# op_version == 1 represents the base op (top-level "since" in the per-op dict);
# op_version >= 2 represents a "vN" variant.
ParsedOps = dict[str, dict[int, tuple[str | None, int]]]


def _const_str(node: ast.expr | None) -> str | None:
    # [FUNCTION DOC] HELPER: string value if the node is a string literal constant, else None.
    """Return `node`'s string-literal value, or None if it isn't one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _str_value_for_key(dict_node: ast.Dict, key: str) -> str | None:
    # [FUNCTION DOC] HELPER: scan a dict literal for a string-literal key equal to key and return
    # its value if that is a string literal; None if absent or not a literal.
    """Return the str literal at `dict_node[key]`, or None if absent/non-literal."""
    for k_node, v_node in zip(dict_node.keys, dict_node.values):
        if _const_str(k_node) == key:
            return _const_str(v_node)
    return None


def _ingest_op_dict(dict_node: ast.Dict) -> ParsedOps:
    # [FUNCTION DOC] HELPER: convert a dict literal of op name to per-op dict literal into a
    # mapping op name to {version number: (since string or None, line number)}. Version 1 is the
    # base op (its since key, line of the op name); keys of the form v followed by digits with a
    # number above 1 add variants. Entries with non-literal names or non-dict values are skipped.
    """Walk one of the top-level fallback dicts and return a `ParsedOps` map."""
    out: ParsedOps = {}
    for op_key_node, op_val_node in zip(dict_node.keys, dict_node.values):
        if op_key_node is None or not isinstance(op_val_node, ast.Dict):
            continue
        op_name = _const_str(op_key_node)
        if op_name is None:
            continue
        per_op: dict[int, tuple[str | None, int]] = {
            1: (_str_value_for_key(op_val_node, "since"), op_key_node.lineno),
        }
        for var_key_node, var_val_node in zip(op_val_node.keys, op_val_node.values):
            if var_key_node is None:
                continue
            var_key = _const_str(var_key_node)
            if var_key is None or not (
                var_key.startswith("v") and var_key[1:].isdigit()
            ):
                continue
            ver_id = int(var_key[1:])
            if ver_id <= 1:
                continue
            since = (
                _str_value_for_key(var_val_node, "since")
                if isinstance(var_val_node, ast.Dict)
                else None
            )
            per_op[ver_id] = (since, var_key_node.lineno)
        out[op_name] = per_op
    return out


def parse_fallback_ops(src: str) -> dict[str, ParsedOps]:
    # [FUNCTION DOC] Parse Python source text and, for each module-level assignment (plain or
    # annotated) of a dict literal to a simple name, store the ingested op mapping under that
    # variable name. Raises SyntaxError on invalid source. Pure.
    """
    Parse a `fallback_ops.py` source string. Returns a map keyed by the
    module-level variable name (e.g. `inductor_fallback_ops`,
    `aten_shimified_ops`) so current-vs-base comparison can pair dicts by name
    rather than by position.
    """
    out: dict[str, ParsedOps] = {}
    tree = ast.parse(src)
    for node in tree.body:
        if not (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Dict)
        ):
            continue
        target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
        if isinstance(target, ast.Name):
            out[target.id] = _ingest_op_dict(node.value)
    return out


def _read_at_merge_base(filename: str) -> str | None:
    # [FUNCTION DOC] HELPER: find the merge-base of HEAD with origin/main and return the file's
    # text at that commit, or None if it did not exist there. Runs git; raises RuntimeError
    # on git failures.
    """
    Return `filename` contents at the merge-base of HEAD with origin/main, or
    None if the file did not exist at that point. Raises if git operations fail.
    """
    merge_base = merge_base_with_main()
    return read_file_at_revision(merge_base, filename, cwd=REPO_ROOT)


def _msg(filename: str, line: int, name: str, description: str) -> LintMessage:
    # [FUNCTION DOC] HELPER: build an error-severity LintMessage for the file at the given line
    # (no column) with the given name and description.
    return LintMessage(
        path=filename,
        line=line,
        char=None,
        code=LINTER_CODE,
        severity=LintSeverity.ERROR,
        name=name,
        original=None,
        replacement=None,
        description=description,
    )


def _check_one_dict(
    current: ParsedOps,
    base: ParsedOps,
    expected_since: str,
    filename: str,
) -> list[LintMessage]:
    # [FUNCTION DOC] HELPER: compare current ops with base ops. For ops or variants absent
    # from base: error if since is missing, malformed (not the TORCH_VERSION_X_Y_Z form) or not
    # equal to expected_since. For ones present in base: error if the since value changed.
    # Returns the messages with the current file line numbers.
    messages: list[LintMessage] = []
    for op_name, variants in current.items():
        prev_variants = base.get(op_name, {})
        for ver_id, (since, lineno) in variants.items():
            label = f"'{op_name}'" if ver_id == 1 else f"'{op_name}' v{ver_id}"
            prev = prev_variants.get(ver_id)

            if prev is None:
                # Newly added op (ver_id == 1) or newly added vN variant.
                if since is None:
                    messages.append(
                        _msg(
                            filename,
                            lineno,
                            "missing-version-on-new-op",
                            f"{label} is newly added; it must include a "
                            f"'since' key set to {expected_since}.",
                        )
                    )
                elif not _TORCH_VERSION_PATTERN.match(since):
                    messages.append(
                        _msg(
                            filename,
                            lineno,
                            "malformed-version",
                            f"{label} 'since' value {since!r} does not match "
                            f"the format TORCH_VERSION_X_Y_Z.",
                        )
                    )
                elif since != expected_since:
                    messages.append(
                        _msg(
                            filename,
                            lineno,
                            "wrong-version-on-new-op",
                            f"{label} 'since' should be {expected_since} "
                            f"(the current TORCH_VERSION); got {since}.",
                        )
                    )
            else:
                prev_since, _ = prev
                if since != prev_since:
                    messages.append(
                        _msg(
                            filename,
                            lineno,
                            "op-version-modified",
                            f"{label} 'since' changed from {prev_since!r} "
                            f"to {since!r}; once an op has shipped its "
                            f"'since' version value must not be modified.",
                        )
                    )

    return messages


def check_file(filename: str) -> list[LintMessage]:
    # [FUNCTION DOC] Lint one fallback ops file: parse it, parse its merge-base version (empty
    # if the file is new), derive the expected since label from the current version in
    # version.txt, and check each same-named dictionary pair. Returns all messages; reads
    # the file and runs git, errors propagate.
    current_dicts = parse_fallback_ops(Path(filename).read_text())
    base_src = _read_at_merge_base(filename)
    base_dicts = parse_fallback_ops(base_src) if base_src is not None else {}

    major, minor, patch = get_current_version()
    expected_since = f"TORCH_VERSION_{major}_{minor}_{patch}"

    messages: list[LintMessage] = []
    for name, current in current_dicts.items():
        base = base_dicts.get(name, {})
        messages.extend(_check_one_dict(current, base, expected_since, filename))
    return messages


# Script entry: parses verbose flag and file names, configures logging to stderr, lints each
# file and prints each message as JSON on stdout.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generated shims version linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
    )

    lint_messages: list[LintMessage] = []
    for fn in args.filenames:
        lint_messages.extend(check_file(fn))
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
