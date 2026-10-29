#!/usr/bin/env python3
"""
This lint verifies that every Python test file (file that matches test_*.py or
*_test.py in the test folder) has a cuda hard code in `requires_gpu()` or
`requires_triton()` decorated function or `if HAS_GPU:` guarded main section,
to ensure that the test not fail on other GPU devices.
"""

# [MODULE DOC]
# Lintrunner adapter that parses test files with the ast module and reports hard-coded GPU
# device names (cuda, xpu, mps) in functions decorated with requires_gpu or requires_triton,
# or in every function of a file whose main section is guarded by a GPU flag.
# Main contents: LintSeverity/LintMessage types, the main-guard detector, the AST visitor
# DeviceBiasVisitor, check_file and main (parallel over files with 8 worker processes,
# printing JSON lines). No import-time side effects.
from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "TEST_DEVICE_BIAS"


class LintSeverity(str, Enum):
    # [CLASS DOC] HELPER: String-valued enumeration of lint severities (error, warning,
    # advice, disabled) serialized into the JSON output consumed by lintrunner.
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    # [CLASS DOC] HELPER: Immutable record of one lint finding in the lintrunner JSON format:
    # path, line and char (each optional), linter code, severity, short name, optional
    # original and replacement text (for autofix), and an optional description.
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


DEVICE_BIAS = ["cuda", "xpu", "mps"]
GPU_RELATED_DECORATORS = {"requires_gpu", "requires_triton"}


def is_main_has_gpu(tree: ast.AST) -> bool:
    # [FUNCTION DOC] Returns True if the module has a top-level-style main guard (comparison of
    # the module name with the main string) whose body directly contains a conditional that
    # tests the HAS_GPU or RUN_GPU flag, i.e. the test suite is shared across GPU devices.
    # Walks the whole tree; pure function, no side effects.
    def _contains_has_gpu(node: ast.AST) -> bool:
        # [FUNCTION DOC] HELPER: Returns True if the expression is, or recursively contains
        # through boolean operations, unary operations and comparisons, a name equal to
        # HAS_GPU or RUN_GPU. Conditional expressions and calls count as False.
        if isinstance(node, ast.Name) and node.id in ["HAS_GPU", "RUN_GPU"]:
            return True
        elif isinstance(node, ast.BoolOp):
            return any(_contains_has_gpu(value) for value in node.values)
        elif isinstance(node, ast.UnaryOp):
            return _contains_has_gpu(node.operand)
        elif isinstance(node, ast.Compare):
            return _contains_has_gpu(node.left) or any(
                _contains_has_gpu(comp) for comp in node.comparators
            )
        elif isinstance(node, (ast.IfExp, ast.Call)):
            return False
        return False

    for node in ast.walk(tree):
        # Detect if __name__ == "__main__":
        if isinstance(node, ast.If):
            if (
                isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
            ):
                if any(
                    isinstance(comp, ast.Constant) and comp.value == "__main__"
                    for comp in node.test.comparators
                ):
                    for inner_node in node.body:
                        if isinstance(inner_node, ast.If) and _contains_has_gpu(
                            inner_node.test
                        ):
                            return True
    return False


class DeviceBiasVisitor(ast.NodeVisitor):
    # [CLASS DOC] AST visitor that collects device-bias lint messages for one file.
    # State: filename, the accumulating lint_messages list, and is_gpu_test_suite (whether the
    # main section is GPU guarded, which makes every function checked). Function definitions
    # are checked when decorated with a GPU decorator or when the file is a GPU test suite.
    def __init__(self, filename: str, is_gpu_test_suite: bool) -> None:
        # [FUNCTION DOC] Stores the file name and suite flag and starts with no messages.
        self.filename = filename
        self.lint_messages: list[LintMessage] = []
        self.is_gpu_test_suite = is_gpu_test_suite

    def _has_proper_decorator(self, node: ast.FunctionDef) -> bool:
        # [FUNCTION DOC] HELPER: Returns True if any decorator of the function, either bare or
        # called, is named requires_gpu or requires_triton.
        for d in node.decorator_list:
            if isinstance(d, ast.Name) and d.id in GPU_RELATED_DECORATORS:
                return True
            if (
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id in GPU_RELATED_DECORATORS
            ):
                return True
        return False

    # check device = "cuda" or torch.device("cuda")
    def _check_keyword_device(self, subnode: ast.keyword, msg_prefix: str) -> None:
        # [FUNCTION DOC] HELPER: For a keyword argument named device, records a message if its
        # value is a string constant containing a biased device name, or a call to a device
        # attribute whose first argument is such a string. Other keywords are ignored.
        # KNOWN ISSUE: assumes a constant value supports substring tests; a non-string
        # constant (for example a number) would raise a TypeError.
        if subnode.arg != "device":
            return
        val = subnode.value
        if isinstance(val, ast.Constant) and any(
            # pyrefly: ignore [not-iterable, unsupported-operation]
            bias in val.value
            for bias in DEVICE_BIAS
        ):
            self.record(
                subnode,
                f"{msg_prefix} device='{val.value}', suggest to use device=GPU_TYPE",
            )
        elif isinstance(val, ast.Call):
            if (
                isinstance(val.func, ast.Attribute)
                and val.func.attr == "device"
                and len(val.args) > 0
                and isinstance(val.args[0], ast.Constant)
                # pyrefly: ignore [not-iterable, unsupported-operation]
                and any(bias in val.args[0].value for bias in DEVICE_BIAS)
            ):
                self.record(
                    val,
                    f"{msg_prefix} torch.device('{val.args[0].value}'), suggest to use torch.device(GPU_TYPE)",
                )

    # check .cuda() or .to("cuda")
    def _check_device_methods(self, subnode: ast.Call, msg_prefix: str) -> None:
        # [FUNCTION DOC] HELPER: For a method call, records a message when the method name is
        # one of the biased device names (cuda, xpu, mps) or when it is the to method with a
        # first string-constant argument containing such a name.
        # KNOWN ISSUE: a non-string constant first argument to the to method would raise a
        # TypeError on the substring test.
        func = subnode.func
        if not isinstance(func, ast.Attribute):
            return
        method_name = func.attr
        if method_name in DEVICE_BIAS:
            self.record(
                subnode,
                f"{msg_prefix} .{method_name}(), suggest to use .to(GPU_TYPE)",
            )
        elif method_name == "to" and subnode.args:
            arg = subnode.args[0]
            if isinstance(arg, ast.Constant) and any(
                # pyrefly: ignore [not-iterable, unsupported-operation]
                bias in arg.value
                for bias in DEVICE_BIAS
            ):
                self.record(
                    subnode,
                    f"{msg_prefix} .to('{arg.value}'), suggest to use .to(GPU_TYPE)",
                )

    def _check_with_statement(self, node: ast.With, msg_prefix: str) -> None:
        # [FUNCTION DOC] HELPER: Records a message for each context manager item that is a call
        # to torch device with a first string-constant argument containing a biased name.
        for item in node.items:
            ctx_expr = item.context_expr
            if isinstance(ctx_expr, ast.Call):
                func = ctx_expr.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "device"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "torch"
                    and ctx_expr.args
                    and isinstance(ctx_expr.args[0], ast.Constant)
                    # pyrefly: ignore [not-iterable, unsupported-operation]
                    and any(bias in ctx_expr.args[0].value for bias in DEVICE_BIAS)
                ):
                    self.record(
                        ctx_expr,
                        f"{msg_prefix} `with torch.device('{ctx_expr.args[0].value}')`, suggest to use torch.device(GPU_TYPE)",
                    )

    def _check_node(self, node: ast.AST, msg_prefix: str) -> None:
        # [FUNCTION DOC] HELPER: Walks every descendant of the node and dispatches keyword
        # arguments, attribute calls and with statements to the matching device checks,
        # prefixing messages with msg_prefix.
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.keyword):
                self._check_keyword_device(subnode, msg_prefix)
            elif isinstance(subnode, ast.Call) and isinstance(
                subnode.func, ast.Attribute
            ):
                self._check_device_methods(subnode, msg_prefix)
            elif isinstance(subnode, ast.With):
                self._check_with_statement(subnode, msg_prefix)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # [FUNCTION DOC] Visitor hook for function definitions. Checks the function body with a
        # decorator-specific message prefix if it has a GPU decorator, otherwise with a
        # shared-suite prefix if the file is a GPU test suite. Then continues into nested nodes.
        # KNOWN ISSUE: async function definitions are not visited by this hook.
        if self._has_proper_decorator(node):
            msg_prefix = (
                "`@requires_gpu` or `@requires_triton` function should not hardcode"
            )
            self._check_node(node, msg_prefix)
        elif self.is_gpu_test_suite:
            # If the function is guarded by HAS_GPU in main(), we still need to check for device bias
            msg_prefix = "The test suites is shared amount GPUS, should not hardcode"
            self._check_node(node, msg_prefix)
        self.generic_visit(node)

    def record(self, node: ast.AST, message: str) -> None:
        # [FUNCTION DOC] Appends an error-severity device-bias LintMessage to the list, using the
        # node line number (None if the node has none) and the given description.
        self.lint_messages.append(
            LintMessage(
                path=self.filename,
                line=getattr(node, "lineno", None),
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[device-bias]",
                original=None,
                replacement=None,
                description=message,
            )
        )


def check_file(filename: str) -> list[LintMessage]:
    # [FUNCTION DOC] Reads and parses one Python file, determines whether its main section is
    # GPU guarded, runs the device-bias visitor and returns its messages. Raises OSError if
    # the file cannot be read and SyntaxError if it cannot be parsed.
    with open(filename) as f:
        source = f.read()
        tree = ast.parse(source, filename=filename)
        is_gpu_test_suite = is_main_has_gpu(tree)
        checker = DeviceBiasVisitor(filename, is_gpu_test_suite)
        checker.visit(tree)
    return checker.lint_messages


def main() -> None:
    # [FUNCTION DOC] Command line entry point. Parses the list of file paths, checks them in a
    # pool of 8 processes, flattens the results and prints each message as a JSON line on
    # standard output.
    parser = argparse.ArgumentParser(
        description="Detect Device bias in functions decorated with requires_gpu/requires_triton"
        " or guarded by HAS_GPU block in main() that may break other GPU devices.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    with mp.Pool(8) as pool:
        lint_messages = pool.map(check_file, args.filenames)

    flat_lint_messages = []
    for sublist in lint_messages:
        flat_lint_messages.extend(sublist)

    for lint_message in flat_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
