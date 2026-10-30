# Owner(s): ["module: dynamo"]
# [MODULE DOC] pytest configuration for the torch_np tests. Adds the slow marker and the
# --runslow and --nonp command line options, parametrizes tests that take an np fixture over
# torch._numpy and (when importable and not disabled) real NumPy, skips slow tests unless
# requested, and skips the whole suite under dynamo-wrapped CI.

import os
import sys

import pytest

import torch._numpy as tnp


def pytest_configure(config):
    # [FUNCTION DOC] pytest hook: register the slow marker description in the pytest
    # configuration.
    config.addinivalue_line("markers", "slow: very slow tests")


def pytest_addoption(parser):
    # [FUNCTION DOC] pytest hook: add the boolean options --runslow (run slow tests) and
    # --nonp (make any access to NumPy raise an error).
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption("--nonp", action="store_true", help="error when NumPy is accessed")


class Inaccessible:
    # [CLASS DOC] Stand-in object installed in place of the numpy module when --nonp is
    # given. Any attribute access on it raises RuntimeError, so tests that touch NumPy fail
    # loudly.
    def __getattribute__(self, attr):
        # [FUNCTION DOC] Raise RuntimeError naming the attribute that was accessed; never
        # returns.
        raise RuntimeError(f"Using --nonp but accessed np.{attr}")


def pytest_sessionstart(session):
    # [FUNCTION DOC] pytest hook run at session start. If --nonp was passed (default False
    # when the option is not registered), replace the numpy entry in the module table with
    # an Inaccessible instance. Side effect: modifies the global module table.
    # Use default=False to handle case when option isn't registered
    # (e.g., when pytest is run from the root directory)
    if session.config.getoption("--nonp", default=False):
        sys.modules["numpy"] = Inaccessible()


def pytest_generate_tests(metafunc):
    # [FUNCTION DOC] pytest hook that parametrizes tests having a fixture argument named np.
    # Builds a list of numpy-like modules: always torch._numpy, plus real NumPy if it
    # imports and is not the Inaccessible stand-in. Parametrizes the np argument over that
    # list. Tests without such an argument are left unchanged.
    """
    Hook to parametrize test cases
    See https://docs.pytest.org/en/6.2.x/parametrize.html#pytest-generate-tests

    The logic here allows us to test with both NumPy-proper and torch._numpy.
    Normally we'd just test torch._numpy, e.g.

        import torch._numpy as np
        ...
        def test_foo():
            np.array([42])
            ...

    but this hook allows us to test NumPy-proper as well, e.g.

        def test_foo(np):
            np.array([42])
            ...

    np is a pytest parameter, which is either NumPy-proper or torch._numpy. This
    allows us to sanity check our own tests, so that tested behaviour is
    consistent with NumPy-proper.

    pytest will have test names respective to the library being tested, e.g.

        $ pytest --collect-only
        test_foo[torch._numpy]
        test_foo[numpy]

    """
    np_params = [tnp]

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if not isinstance(np, Inaccessible):  # i.e. --nonp was used
            np_params.append(np)

    if "np" in metafunc.fixturenames:
        metafunc.parametrize("np", np_params)


def pytest_collection_modifyitems(config, items):
    # [FUNCTION DOC] pytest hook that marks tests as skipped after collection. Unless
    # --runslow is given (default False if unregistered), adds a skip marker to items with
    # the slow keyword. If the environment variable PYTORCH_TEST_WITH_DYNAMO equals 1, adds
    # a skip marker to every collected item. Mutates the items only.
    # Use default=False to handle case when option isn't registered
    # (e.g., when pytest is run from the root directory)
    # See: https://github.com/pytorch/pytorch/issues/171563
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="slow test, use --runslow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip the entire torch_np suite under dynamo_wrapped CI. These tests
    # produce constant churn between expected_failure and unexpected_success
    # entries; running them under dynamo provides little incremental signal.
    if os.environ.get("PYTORCH_TEST_WITH_DYNAMO") == "1":
        skip_dynamo = pytest.mark.skip(
            reason="torch_np tests are skipped under dynamo_wrapped"
        )
        for item in items:
            item.add_marker(skip_dynamo)
