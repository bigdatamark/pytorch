# [MODULE DOC] Pytest configuration for the fastrnns benchmark tests: defines the default RNN and
# CNN network name lists, a hook that parametrizes the TestBenchNetwork class, and command line
# options selecting the fuser and executor.
import pytest  # noqa: F401


default_rnns = [
    "cudnn",
    "aten",
    "jit",
    "jit_premul",
    "jit_premul_bias",
    "jit_simple",
    "jit_multilayer",
    "py",
]
default_cnns = ["resnet18", "resnet18_jit", "resnet50", "resnet50_jit"]
all_nets = default_rnns + default_cnns


def pytest_generate_tests(metafunc):
    # [FUNCTION DOC] Pytest hook generating test parameters. Only for test classes named
    # TestBenchNetwork: parametrizes net_name over all RNN and CNN names, and executor and fuser
    # with the single value taken from the corresponding command line option, all with class
    # scope. Other classes are left alone. Note it reads the class name, so it assumes
    # metafunc.cls is set.
    # This creates lists of tests to generate, can be customized
    if metafunc.cls.__name__ == "TestBenchNetwork":
        metafunc.parametrize("net_name", all_nets, scope="class")
        metafunc.parametrize(
            "executor", [metafunc.config.getoption("executor")], scope="class"
        )
        metafunc.parametrize(
            "fuser", [metafunc.config.getoption("fuser")], scope="class"
        )


def pytest_addoption(parser):
    # [FUNCTION DOC] Pytest hook registering two command line options: --fuser (default old) and
    # --executor (default legacy), consumed by the parametrization hook.
    parser.addoption("--fuser", default="old", help="fuser to use for benchmarks")
    parser.addoption(
        "--executor", default="legacy", help="executor to use for benchmarks"
    )
