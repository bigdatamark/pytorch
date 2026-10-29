# main.py

# [MODULE DOC] Entry point of the lumen CLI (run as a module with python -m cli.run).
# Builds the top-level argument parser, lets the build and test packages register their
# subcommands, then dispatches to the selected runner. No import-time side effects.
import argparse
import logging

from cli.build_cli.register_build import register_build_commands
from cli.lib.common.logger import setup_logging
from cli.test_cli.register_test import register_test_commands


logger = logging.getLogger(__name__)


def main():
    # [FUNCTION DOC] Parse the command line (global log level option plus the build and
    # test command trees), configure logging at the requested level (falls back to INFO
    # for unknown names) and call the handler stored on the parsed arguments; prints the
    # help text if no handler exists. Exits via argparse on invalid arguments.
    # Define top-level parser
    parser = argparse.ArgumentParser(description="Lumos CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser.add_argument(
        "--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)"
    )

    # registers second-level subcommands
    register_build_commands(subparsers)
    register_test_commands(subparsers)

    # parse args after all options are registered
    args = parser.parse_args()

    # setup global logging
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    logger.debug("Parsed args: %s", args)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
