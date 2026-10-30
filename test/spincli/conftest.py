# [MODULE DOC] Pytest configuration for the spin CLI tests. Defines the repository root path
# (three directory levels above this file), a generator helper that temporarily changes the
# working directory, and a fixture that runs a test from the repository root.
import os
import pathlib

import pytest


def dir_switcher(path):
    # [FUNCTION DOC] HELPER: Generator used as fixture body. Saves the current working directory,
    # changes into the given path, yields to the test, and in a finally clause restores the saved
    # directory even if the test fails. Raises OSError if the path cannot be entered.
    # Pre-test code
    cwd = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        # Post test code
        os.chdir(cwd)


PYTORCH_ROOT = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture()
def main_pkg():
    # [FUNCTION DOC] Pytest fixture (not shared beyond function scope) that makes the repository
    # root the working directory for the duration of a test, by delegating to the directory
    # switching generator, then restores the previous directory.
    yield from dir_switcher(PYTORCH_ROOT)
