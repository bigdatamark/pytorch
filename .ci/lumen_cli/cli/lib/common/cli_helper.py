# [MODULE DOC] Argparse utilities shared by lumen CLI commands: the BaseRunner abstract
# class, a help formatter, the TargetSpec typed mapping and register_targets, which
# turns a mapping of target names to specs into subcommands. Import falls back to
# typing_extensions for Required on older Python versions.
"""
Cli Argparser Utility helpers for CLI tasks.

"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod


try:
    from collections.abc import Callable  # Python 3.11+
    from typing import Any, Required, TypedDict
except ImportError:
    from collections.abc import Callable  # noqa: TC003
    from typing import Any, TypedDict

    from typing_extensions import Required  # Fallback for Python <3.11


class BaseRunner(ABC):
    # [CLASS DOC] Abstract base for command runners. Holds the parsed argument
    # namespace in the args attribute; subclasses must implement run.
    def __init__(self, args: Any) -> None:
        # [FUNCTION DOC] Store the parsed command line namespace as args.
        self.args = args

    @abstractmethod
    def run(self) -> None:
        # [FUNCTION DOC] Abstract: perform the command's work; implemented by subclasses.
        """runs main logics, required"""


# Pretty help: keep newlines + show defaults
class RichHelp(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    # [CLASS DOC] Help formatter combining default value display with preservation of
    # newlines in descriptions. No added state or methods.
    pass


class TargetSpec(TypedDict, total=False):
    # [CLASS DOC] Typed mapping describing one subcommand target. Keys: runner
    # (required, a BaseRunner subclass), help text, description text, and
    # add_arguments (callable that adds target specific options to a parser).
    """CLI subcommand specification with bA."""

    runner: Required[type[BaseRunner]]
    help: str
    description: str
    add_arguments: Callable[[argparse.ArgumentParser], None]


def register_targets(
    parser: argparse.ArgumentParser,
    target_specs: dict[str, TargetSpec],
    common_args: Callable[[argparse.ArgumentParser], None] = lambda _: None,
) -> None:
    # [FUNCTION DOC] Add one required subcommand per entry of target_specs to parser,
    # stored under the destination name target. Each subcommand uses the spec help and
    # a description taken from the spec or else the runner class docstring. It sets
    # defaults func (builds the runner with the args and calls run) and _runner_class,
    # then calls the optional add_arguments hook and the common_args hook.
    """Register target subcommands."""
    targets = parser.add_subparsers(
        dest="target",
        required=True,
        metavar="{" + ",".join(target_specs.keys()) + "}",
    )

    for name, spec in target_specs.items():
        desc = spec.get("description") or spec["runner"].__doc__ or ""

        p = targets.add_parser(
            name,
            help=spec.get("help", ""),
            description=desc.strip(),
            formatter_class=RichHelp,
        )
        p.set_defaults(
            func=lambda args, cls=spec["runner"]: cls(args).run(),
            _runner_class=spec["runner"],
        )
        if "add_arguments" in spec and callable(spec["add_arguments"]):
            spec["add_arguments"](p)
        if common_args:
            common_args(p)
