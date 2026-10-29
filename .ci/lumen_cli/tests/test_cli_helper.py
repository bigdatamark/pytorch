# [MODULE DOC] Unit tests for the argparse helpers in cli_helper: that register_targets
# builds the target choice metavar, applies per-target and common options, constructs and
# runs the runner, falls back to the runner docstring for the description, and requires a
# target. Defines small dummy runners and parser builders as fixtures.
from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from cli.lib.common.cli_helper import BaseRunner, register_targets, RichHelp, TargetSpec


# ---- Dummy runners for unittests----
class FooRunner(BaseRunner):
    # [CLASS DOC] HELPER: Dummy runner with a docstring, used to test description
    # fallback; its run does nothing and is mocked in tests.
    """Foo description from docstring."""

    def run(self) -> None:  # replaced by mock
        # [FUNCTION DOC] HELPER: Intentionally does nothing.
        pass


class BarRunner(BaseRunner):
    # [CLASS DOC] HELPER: Dummy runner without a docstring.
    def run(self) -> None:  # replaced by mock
        # [FUNCTION DOC] HELPER: Intentionally does nothing.
        pass


def add_foo_args(p: argparse.ArgumentParser) -> None:
    # [FUNCTION DOC] HELPER: Add a required integer option named x to the parser.
    p.add_argument("--x", type=int, required=True, help="x value")


def common_args(p: argparse.ArgumentParser) -> None:
    # [FUNCTION DOC] HELPER: Add a boolean verbose flag to the parser.
    p.add_argument("--verbose", action="store_true", help="verbose flag")


def build_parser(specs: dict[str, TargetSpec]) -> argparse.ArgumentParser:
    # [FUNCTION DOC] HELPER: Create a parser named app and register the given target
    # specs on it together with the common verbose option; returns the parser.
    parser = argparse.ArgumentParser(prog="app", formatter_class=RichHelp)
    register_targets(
        parser=parser,
        target_specs=specs,
        common_args=common_args,
    )
    return parser


def get_subparser(
    parser: argparse.ArgumentParser, name: str
) -> argparse.ArgumentParser:
    # [FUNCTION DOC] HELPER: Find the subparsers action of a parser (via a private
    # attribute) and return the child parser registered under name.
    subparsers_action = next(
        a
        for a in parser._subparsers._group_actions  # type: ignore[attr-defined]
        if isinstance(a, argparse._SubParsersAction)
    )
    return subparsers_action.choices[name]


class TestRegisterTargets(unittest.TestCase):
    # [CLASS DOC] Test case for register_targets behavior using the dummy runners.
    def test_metavar_lists_targets(self):
        # [FUNCTION DOC] Checks the target choice metavar lists names in registration
        # order, in braces separated by commas.
        specs: dict[str, TargetSpec] = {
            "foo": {"runner": FooRunner, "add_arguments": add_foo_args},
            "bar": {"runner": BarRunner},
        }
        parser = build_parser(specs)
        subparsers_action = next(
            a
            for a in parser._subparsers._group_actions  # type: ignore[attr-defined]
            if isinstance(a, argparse._SubParsersAction)
        )
        self.assertEqual(subparsers_action.metavar, "{foo,bar}")

    def test_add_arguments_and_common_args_present(self):
        # [FUNCTION DOC] Checks that the target help shows both its own option and the
        # common option.
        specs: dict[str, TargetSpec] = {
            "foo": {"runner": FooRunner, "add_arguments": add_foo_args},
        }
        parser = build_parser(specs)
        foo = get_subparser(parser, "foo")
        help_text = foo.format_help()
        self.assertIn("--x", help_text)
        self.assertIn("--verbose", help_text)

    def test_runner_constructed_with_ns_and_run_called(self):
        # [FUNCTION DOC] With constructor and run mocked, parses arguments and calls the
        # stored handler; expects one construction with the parsed namespace and one
        # argument-free call to run.
        specs: dict[str, TargetSpec] = {
            "foo": {"runner": FooRunner, "add_arguments": add_foo_args},
        }
        parser = build_parser(specs)

        with (
            patch.object(FooRunner, "__init__", return_value=None) as mock_init,
            patch.object(FooRunner, "run", return_value=None) as mock_run,
        ):
            ns = parser.parse_args(["foo", "--x", "3", "--verbose"])
            ns.func(ns)  # set by register_targets
            # __init__ received the Namespace
            self.assertEqual(mock_init.call_count, 1)
            (called_ns,), _ = mock_init.call_args
            self.assertIsInstance(called_ns, argparse.Namespace)
            # run() called with no args
            mock_run.assert_called_once_with()

    def test_runner_docstring_used_as_description_when_missing(self):
        # [FUNCTION DOC] Checks that without an explicit description the runner
        # docstring text appears in the target help.
        specs: dict[str, TargetSpec] = {
            "foo": {"runner": FooRunner, "add_arguments": add_foo_args},
        }
        parser = build_parser(specs)
        foo = get_subparser(parser, "foo")
        help_text = foo.format_help()
        self.assertIn("Foo description from docstring.", help_text)

    def test_missing_target_raises_systemexit_with_usage(self):
        # [FUNCTION DOC] Parsing no arguments must exit with SystemExit and print a
        # usage message to standard error, since a target is required.
        specs: dict[str, TargetSpec] = {"foo": {"runner": FooRunner}}
        parser = build_parser(specs)
        buf = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(buf):
            parser.parse_args([])
        err = buf.getvalue()
        self.assertIn("usage:", err)


if __name__ == "__main__":
    unittest.main()
