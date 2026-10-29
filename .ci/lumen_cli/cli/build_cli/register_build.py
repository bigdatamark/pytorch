# [MODULE DOC] Registers the build subcommand of the lumen CLI. Holds a mapping from
# external target names (currently vllm) to runner specs and a function that wires
# them into an argparse subparser tree. Import has no side effects beyond a logger.
from __future__ import annotations

import argparse  # noqa: TC003
import logging

from cli.lib.common.cli_helper import register_targets, RichHelp, TargetSpec
from cli.lib.core.vllm.vllm_build import VllmBuildRunner


logger = logging.getLogger(__name__)

# Maps targets to their argparse configuration and runner
# it adds new target to path python -m cli.run build external {target} with buildrunner
_TARGETS: dict[str, TargetSpec] = {
    "vllm": {
        "runner": VllmBuildRunner,
        "help": "Build vLLM using docker buildx.",
    }
    # add yours ...
}


def register_build_commands(subparsers: argparse._SubParsersAction) -> None:
    # [FUNCTION DOC] Add the build command to the given subparsers object, with a
    # required nested build_command choice. Adds an external subcommand whose help
    # lists the known targets, then registers each target of the mapping under it.
    # Mutates the parser tree only; returns nothing.
    build_parser = subparsers.add_parser(
        "build",
        help="Build related commands",
        formatter_class=RichHelp,
    )
    build_subparsers = build_parser.add_subparsers(dest="build_command", required=True)
    overview = "\n".join(
        f"  {name:12} {spec.get('help', '')}" for name, spec in _TARGETS.items()
    )
    external_parser = build_subparsers.add_parser(
        "external",
        help="Build external targets",
        description="Build third-party targets.\n\nAvailable targets:\n" + overview,
        formatter_class=RichHelp,
    )
    register_targets(external_parser, _TARGETS)
