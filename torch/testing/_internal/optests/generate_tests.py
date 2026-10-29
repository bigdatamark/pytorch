# mypy: ignore-errors
# [MODULE DOC] Machinery that auto-generates "opcheck" tests for custom operators from the
# tests of an existing TestCase. Main contents: generate_opcheck_tests (adds one test per
# existing test and per check), the safe_* check wrappers and ALL_TEST_UTILS registry,
# OpCheckMode (a torch function mode that runs the checks on each intercepted operator
# call), the FailuresDict JSON store of expected failures (xfail/skip/xsuccess), the
# opcheck entry point for direct use, and OpCheckError. Reads the environment variable
# PYTORCH_OPCHECK_ACCEPT to rewrite the failures file. Only defines state at import time
# (a thread-local flag marking whether code runs inside an opcheck mode).

import datetime
import difflib
import functools
import inspect
import json
import os
import re
import tempfile
import threading
import unittest
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch._dynamo
import torch.utils._pytree as pytree
from torch._dynamo.utils import clone_input
from torch._library.custom_ops import CustomOpDef
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch._utils_internal import get_file_path_2
from torch.overrides import TorchFunctionMode
from torch.testing._internal.optests import (
    aot_autograd_check,
    autograd_registration_check,
    fake_check,
)


def dontGenerateOpCheckTests(reason: str):
    # [FUNCTION DOC] Decorator factory: the returned decorator marks a test function with an
    # attribute so generate_opcheck_tests skips it, and returns the same function. The
    # reason is documentation only and is not stored.
    def inner(fun):
        # [FUNCTION DOC] HELPER: Sets the skip marker attribute on fun and returns fun.
        fun._torch_dont_generate_opcheck_tests = True
        return fun

    return inner


def is_abstract(tensor: torch.Tensor) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if the tensor is a meta tensor or a fake tensor.
    if tensor.is_meta:
        return True
    if torch._subclasses.fake_tensor.is_fake(tensor):
        return True
    return False


def safe_schema_check(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    copy_inputs: bool = True,
    rtol: float | None = None,
    atol: float | None = None,
) -> Any:
    # [FUNCTION DOC] HELPER: Schema check test util. Optionally deep-copies tensor inputs,
    # returns None without checking if any tensor is meta or fake, otherwise runs the op
    # under SchemaCheckMode (which raises if the op violates its declared schema, e.g.
    # undeclared mutation or aliasing) and returns the op result. rtol and atol are unused.
    if copy_inputs:
        args, kwargs = deepcopy_tensors((args, kwargs))
    if pytree.tree_any_only(torch.Tensor, is_abstract, (args, kwargs)):
        return None
    with SchemaCheckMode():
        result = op(*args, **kwargs)
        return result


def safe_autograd_registration_check(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    copy_inputs: bool = True,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Autograd registration test util. Skips (returns) if any tensor
    # is abstract or if no tensor input requires grad; otherwise runs
    # autograd_registration_check on (optionally copied) inputs. rtol and atol are unused.
    if pytree.tree_any_only(torch.Tensor, is_abstract, (args, kwargs)):
        return
    if copy_inputs:
        args, kwargs = deepcopy_tensors((args, kwargs))
    # Don't perform autograd_registration_check if none of the inputs require grad.
    if not pytree.tree_any_only(
        torch.Tensor, lambda x: x.requires_grad, (args, kwargs)
    ):
        return
    return autograd_registration_check(op, args, kwargs)


def safe_fake_check(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    copy_inputs: bool = True,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Fake tensor test util. Returns None without checking if any
    # tensor is abstract; otherwise runs fake_check on (optionally copied) inputs.
    # rtol and atol are unused.
    if pytree.tree_any_only(torch.Tensor, is_abstract, (args, kwargs)):
        return None
    if copy_inputs:
        args, kwargs = deepcopy_tensors((args, kwargs))
    return fake_check(op, args, kwargs)


def safe_aot_autograd_check(
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    dynamic: bool,
    *,
    copy_inputs: bool = True,
    rtol: float | None = None,
    atol: float | None = None,
) -> Any:
    # [FUNCTION DOC] HELPER: AOT dispatch test util (static or dynamic shapes per dynamic).
    # Returns None if any tensor is abstract; otherwise runs aot_autograd_check on the op
    # with automatic gradient checking and input copying always on (copy_inputs is
    # ignored). Uses assert_close with the given tolerances when both are truthy.
    # KNOWN ISSUE: the tolerances apply only if both rtol and atol are nonzero, so a
    # zero value for either silently falls back to the default tolerances.
    # NB: the copy_inputs argument to this wrapper is ignored: safe AOT checks
    # always need to copy inputs because they run func(*args, **kwargs) multiple
    # times.
    if pytree.tree_any_only(torch.Tensor, is_abstract, (args, kwargs)):
        return None

    def func(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the operator with the given arguments.
        return op(*args, **kwargs)

    # aot_autograd_check runs func(*args, **kwargs) multiple times
    # and assumes `func` does not modify its inputs.
    if rtol and atol:
        assert_equals_fn = functools.partial(
            torch.testing.assert_close, rtol=rtol, atol=atol
        )
    else:
        assert_equals_fn = torch.testing.assert_close
    return aot_autograd_check(
        func,
        args,
        kwargs,
        dynamic,
        check_gradients="auto",
        assert_equals_fn=assert_equals_fn,
        copy_inputs=True,
    )


def deepcopy_tensors(inputs: Any) -> Any:
    # [FUNCTION DOC] HELPER: Returns inputs (a pytree) with every tensor replaced by a
    # dynamo-style clone; other leaves are unchanged.
    return pytree.tree_map_only(torch.Tensor, clone_input, inputs)


# Test util requirements
# - The test util must have signature (op: OpOverload, args, kwargs)
# - The test util must NOT mutate args, kwargs.
# - The test utils in this list must not be prefixes of each other. For example,
#   having both "test_schema" and "test_schema_is_functional" is NOT OK.
# - The order of items in this dict matters (for opcheck), we'll run them
#   in order.
ALL_TEST_UTILS = {
    "test_schema": safe_schema_check,
    "test_autograd_registration": safe_autograd_registration_check,
    "test_faketensor": safe_fake_check,
    "test_aot_dispatch_static": functools.partial(
        safe_aot_autograd_check,
        dynamic=False,
    ),
    "test_aot_dispatch_dynamic": functools.partial(
        safe_aot_autograd_check,
        dynamic=True,
    ),
}

GDOC = "https://docs.google.com/document/d/1Pj5HRZvdOq3xpFpbEjUZp2hBovhy7Wnxw14m6lF2154/edit"

DEFAULT_TEST_UTILS = [
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
]

DEPRECATED_DEFAULT_TEST_UTILS = DEFAULT_TEST_UTILS + [
    "test_aot_dispatch_static",
]


def generate_opcheck_tests(
    testcase: Any,
    namespaces: list[str],
    failures_dict_path: str | None = None,
    additional_decorators: dict[str, Callable] | None = None,
    test_utils: list[str] = DEFAULT_TEST_UTILS,
) -> None:
    # [FUNCTION DOC] Adds new test methods to the testcase class: for every existing method
    # whose name starts with test_ and every selected test util, a method named
    # util-name, two underscores, original-name that runs the original under an
    # OpCheckMode (skipping methods marked by dontGenerateOpCheckTests). Loads and
    # validates the failures dict (default file failures_dict.json next to the caller's
    # file; may create it when PYTORCH_OPCHECK_ACCEPT is 1). Under pytest with the
    # opcheck_only_one mark, trims non-device parametrize marks to one value. Applies
    # additional_decorators by generated name. Also generates tag tests. Raises
    # RuntimeError if a generated name already exists. Mutates the testcase class.
    """Given an existing TestCase, use the existing tests to generate
    additional validation tests for custom operators.

    For {all existing tests in the TestCase} x {all test utils},
    we will generate one new test. The new test runs a TorchFunctionMode
    that intercepts ``op(*args, **kwargs)`` calls and invokes
    ``test_util(op, *args, **kwargs)``, where ``op`` is an operator.

    The test_util that we support are in ALL_TEST_UTILS. They are:
    - test_schema: This runs SchemaCheckMode.
    - test_autograd_registration: This runs autograd_registration_check.
    - test_faketensor: This runs CrossRefFakeMode.
    - test_aot_dispatch_static: This runs aot_autograd_check, which:
        checks that the outputs (and gradients, if they are computable)
        are the same under eager-mode PyTorch and using AOTAutograd.
    - test_aot_dispatch_dynamic: Same as aot_dispatch_static, but
        runs AOTAutograd using dynamic shapes instead of static shapes.

    The generated test will have name ``{test_util}__{original_name}``.
    For example, if there is a method named ``test_cumsum``, then
    we will generate a ``test_schema__test_cumsum``,
    ``test_faketensor__test_cumsum``, etc.

    For more details, see https://docs.google.com/document/d/1Pj5HRZvdOq3xpFpbEjUZp2hBovhy7Wnxw14m6lF2154/edit

    Args:
        testcase: The testcase we will modify and generate additional tests for.
        namespaces: We will only intercept calls to custom operators with these
                    namespaces.
        failures_dict_path: See ``validate_failures_dict_structure`` for more details
        test_utils: a list of test_utils to generate. Example: ["test_schema", "test_faketensor"]
    """
    if additional_decorators is None:
        additional_decorators = {}
    test_methods = [
        m
        for m in dir(testcase)
        if m.startswith("test_") and callable(getattr(testcase, m))
    ]
    if failures_dict_path is None:
        # The default failures_dict_path is failures_dict.json in
        # the same directory as the test file.
        prev_frame = inspect.currentframe().f_back
        filename = inspect.getframeinfo(prev_frame)[0]
        failures_dict_path = get_file_path_2(
            os.path.dirname(filename), "failures_dict.json"
        )
    failures_dict = FailuresDict.load(
        failures_dict_path, create_file=should_update_failures_dict()
    )
    validate_failures_dict_structure(failures_dict, test_utils, testcase)
    validate_failures_dict_formatting(failures_dict_path)

    def construct_method(attr, prefix, tester):
        # [FUNCTION DOC] HELPER: Creates and attaches the generated test for the existing
        # method attr and the check named prefix (tester is the check function). Does
        # nothing if the method is marked to be skipped. Raises RuntimeError on a name clash.
        method = getattr(testcase, attr)
        if getattr(method, "_torch_dont_generate_opcheck_tests", False):
            return
        new_method_name = prefix + "__" + attr

        @functools.wraps(method)
        def new_method(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Runs the original test method inside an OpCheckMode for
            # this check and returns the method's result.
            with OpCheckMode(
                namespaces,
                prefix,
                tester,
                failures_dict,
                f"{testcase.__name__}.{new_method_name}",
                failures_dict_path,
            ):
                result = method(*args, **kwargs)
            return result

        if pytestmark := new_method.__dict__.get("pytestmark"):
            import pytest

            # check if we need to simplify the parametrize marks
            # NB: you need to add this mark to your pytest.ini
            opcheck_only_one = False
            for mark in pytestmark:
                if isinstance(mark, pytest.Mark) and mark.name == "opcheck_only_one":
                    opcheck_only_one = True

            if opcheck_only_one:
                new_pytestmark = []
                for mark in pytestmark:
                    if isinstance(mark, pytest.Mark) and mark.name == "parametrize":
                        argnames, argvalues = mark.args
                        if mark.kwargs:
                            raise AssertionError("NYI: mark.kwargs is not empty")
                        # Special case for device, we want to run on all
                        # devices
                        if argnames != "device":
                            new_pytestmark.append(
                                pytest.mark.parametrize(
                                    argnames, (next(iter(argvalues)),)
                                )
                            )
                            continue
                    new_pytestmark.append(mark)
                new_method.__dict__["pytestmark"] = new_pytestmark

        if new_method_name in additional_decorators:
            for dec in additional_decorators[new_method_name]:
                new_method = dec(new_method)

        if hasattr(testcase, new_method_name):
            raise RuntimeError(
                f"Tried to autogenerate {new_method_name} but {testcase} already "
                f"has method named {new_method_name}. Please rename the original "
                f"method on the TestCase."
            )
        setattr(testcase, new_method_name, new_method)

    test_utils = {name: ALL_TEST_UTILS[name] for name in test_utils}
    for attr in test_methods:
        for prefix, tester in test_utils.items():
            construct_method(attr, prefix, tester)

    generate_tag_tests(testcase, failures_dict, additional_decorators)


def generate_tag_tests(testcase, failures_dict, additional_decorators):
    # [FUNCTION DOC] For each operator in the failures dict adds a test named with the
    # prefix test_pt2_compliant_tag and the mangled operator name. The test fails if the
    # operator carries the pt2 compliant tag yet has an xfail entry (ignoring static AOT
    # entries); it skips if the operator cannot be looked up. Applies matching entries of
    # additional_decorators. Raises RuntimeError if the generated name already exists.
    def generate_test(qualname, definitely_not_pt2_compliant, xfailed_tests):
        # [FUNCTION DOC] HELPER: Returns a test function for one operator qualname.
        def inner(self):
            # [FUNCTION DOC] HELPER: Looks up the op (skips the test if not importable),
            # returns if it lacks the compliant tag or has no xfails, otherwise raises
            # AssertionError naming the xfailed tests.
            try:
                op = torch._library.utils.lookup_op(qualname)
            except AttributeError as e:
                # Operator not importable in this test file
                raise unittest.SkipTest(f"Can't import operator {qualname}") from e
            op_marked_as_compliant = torch.Tag.pt2_compliant_tag in op.tags
            if not op_marked_as_compliant:
                return
            if not definitely_not_pt2_compliant:
                return
            raise AssertionError(
                f"op '{qualname}' was tagged with torch.Tag.pt2_compliant_tag "
                f"but it failed some of the generated opcheck tests "
                f"({xfailed_tests}). This may lead to silent correctness issues, "
                f"please fix this."
            )

        return inner

    for qualname, test_dict in failures_dict.data.items():
        xfailed_tests = [
            test
            for test, status_dict in test_dict.items()
            # We're about to delete the following test after Ed's PR
            # to specialize on C++ .size() calls
            if "test_aot_dispatch_static" not in test
            and status_dict["status"] == "xfail"
        ]
        definitely_not_pt2_compliant = len(xfailed_tests) > 0
        generated = generate_test(qualname, definitely_not_pt2_compliant, xfailed_tests)

        # Could result in collisions, but unlikely. We'll raise if we see one below.
        mangled_qualname = qualname.replace("::", "_").replace(".", "_")
        test_name = "test_pt2_compliant_tag_" + mangled_qualname

        # You can skip this test via the additional_decorators argument
        # in generate_opcheck_tests
        if test_name in additional_decorators:
            for decorator in additional_decorators[test_name]:
                generated = decorator(generated)

        if hasattr(testcase, test_name):
            raise RuntimeError(
                f"Tried to generate a test named {test_name}, but it exists "
                f"already. This could be because of a name collision (where "
                f"we generated two tests with the same name), or where we "
                f"generated a test with the same name as an existing test."
            )
        setattr(testcase, test_name, generated)


TEST_OPTIONS = ("xfail", "skip", "xsuccess")


def validate_failures_dict_formatting(failures_dict_path: str) -> None:
    # [FUNCTION DOC] Checks that the file's text equals the canonical serialization of its
    # loaded content. On mismatch, rewrites the file if PYTORCH_OPCHECK_ACCEPT is 1;
    # otherwise raises RuntimeError containing a unified diff. Reads (and maybe writes)
    # the file.
    with open(failures_dict_path) as fp:
        actual = fp.read()
    failures_dict = FailuresDict.load(failures_dict_path)
    expected = failures_dict._save(to_str=True)
    if actual == expected:
        return
    if should_update_failures_dict():
        failures_dict = FailuresDict.load(failures_dict_path)
        failures_dict.save()
        return
    expected = expected.splitlines(1)
    actual = actual.splitlines(1)
    diff = difflib.unified_diff(actual, expected)
    diff = "".join(diff)
    raise RuntimeError(
        f"\n{diff}\n\nExpected the failures dict to be formatted "
        f"a certain way. Please see the above diff; you can correct "
        f"this either manually or by re-running the test with "
        f"PYTORCH_OPCHECK_ACCEPT=1"
    )


def validate_failures_dict_structure(
    failure_dict: "FailuresDict", test_utils: list[str], testcase: Any
) -> None:
    # [FUNCTION DOC] Validates the failures dict data: every test entry must have exactly
    # the keys comment and status, status must be one of TEST_OPTIONS, and the test name
    # must be class-dot-method where the method begins with a selected test util name. For
    # entries of the given testcase class, the original test name (after the util prefix
    # and two underscores, minus any bracketed pytest suffix) must exist on the testcase.
    # Raises RuntimeError otherwise.
    # KNOWN ISSUE: the last error message has an unbalanced quote/bracket around the
    # testcase name, and the name check silently passes for entries of other classes.
    """Validates the failures dict.

    The failure dict looks something like the following.
    It maps operator name (qualname) to a list of autogenerated tests.
    Each autogenerated test may have a check for the operator (if the operator is
    called by the test); the dictionary specifies if we should skip the check,
    or if we expect some check to fail.

    {
        "fbgemm::split_lengths": {
            "test_schema__test_split_lengths": {
                "comment": "you can put whatever you want into the comment section",
                "status": "xfail",
            }
            "test_schema__test_split_lengths_empty": {
                "comment": "",
                "status": "skip",
            },
        },
        "fbgemm::gather_lengths": {
            "test_schema__test_gather_lengths": {
                "comment": "",
                "status": "skip",
            },
        },
    }

    """
    failure_dict = failure_dict.data
    for test_to_option in failure_dict.values():
        for test_name, test_dict in test_to_option.items():
            if set(test_dict.keys()) != set({"comment", "status"}):
                raise RuntimeError(
                    "in failures_dict, expected sub-dict to have keys 'comment' and 'status'"
                )
            test_option = test_dict["status"]
            if test_option not in TEST_OPTIONS:
                raise RuntimeError(
                    f"In failures_dict, got status={test_option} but it needs to be in {TEST_OPTIONS}"
                )
            test_class, actual_test_name = test_name.split(".")
            if not any(actual_test_name.startswith(test) for test in test_utils):
                raise RuntimeError(
                    f"In failures_dict, test name '{test_name}' should begin with one of {test_utils}"
                )
            for test in test_utils:
                if not actual_test_name.startswith(test):
                    continue
                base_test_name = actual_test_name[len(test) + 2 :]
                # remove potential pytest parametrization suffix
                base_test_name = re.sub(r"\[.*\]", "", base_test_name)
                if testcase.__name__ != test_class:
                    continue
                if hasattr(testcase, base_test_name):
                    continue
                raise RuntimeError(
                    f"In failures dict, got test name '{test_name}'. We parsed this as "
                    f"running test '{test}' on '{base_test_name}', but "
                    f"{base_test_name} does not exist on the TestCase '{testcase.__name__}]. "
                    f"Maybe you need to change the test name?"
                )


def should_update_failures_dict() -> bool:
    # [FUNCTION DOC] Returns True if the environment variable PYTORCH_OPCHECK_ACCEPT is set
    # to the string 1 (meaning failures files may be created or rewritten).
    key = "PYTORCH_OPCHECK_ACCEPT"
    return key in os.environ and os.environ[key] == "1"


_is_inside_opcheck_mode = threading.local()
_is_inside_opcheck_mode.value = False


def is_inside_opcheck_mode():
    # [FUNCTION DOC] Returns the flag saying whether the calling thread is currently inside
    # an OpCheckMode context.
    # KNOWN ISSUE: the flag is only initialized for the importing thread; other threads
    # raise an attribute error when reading it before it is set there.
    return _is_inside_opcheck_mode.value


class OpCheckMode(TorchFunctionMode):
    # [CLASS DOC] Torch function mode that intercepts calls to operators in the given
    # namespaces, executes the call normally, and also runs one check function on deep
    # copies of the inputs. Errors from the check are suppressed during the test and
    # collected per operator name in seen_ops_to_errors, then compared against expected
    # statuses (xfail, skip, xsuccess) from the failures dict when the mode exits.
    # Entering sets the thread-local opcheck flag, disables dynamo through the
    # environment and turns off the functorch custom op aliasing check; exit restores them.
    """
    For a given test, OpCheckMode intercepts calls to operators and runs
    test_util(op, args, kwargs) for each intercepted (op, args, kwargs).
    """

    def __init__(
        self,
        namespaces: list[str],
        test_util_name: str,
        test_util: Callable,
        failures_dict: "FailuresDict",
        test_name: str,
        failures_dict_path: str,
    ):
        # [FUNCTION DOC] Stores the configuration (namespaces to intercept, check function
        # and its name, failures dict, running test name and failures file path) and
        # creates an empty mapping from operator name to collected errors.
        # We will intercept calls to ops with these namespaces
        self.namespaces = namespaces
        # The test utility function. Its signature should be (op, args, kwargs) -> None.
        # Examples of test utilities are: schema_check, make_fx_check
        self.test_util = test_util
        self.test_util_name = test_util_name
        # The name of the test that is running this OpCheckMode.
        self.test_name = test_name
        # Maps qualname -> test_name -> skip/xfail
        # Tells us if we should skip a test or assert that there is a failure.
        self.failures_dict = failures_dict
        # Location of the failures dict. Makes it so that the error message is better.
        self.failures_dict_path = failures_dict_path

        # OpCheckMode suppresses errors, collects them here, and then raises them on exit.
        # Maps qualname -> List[(Exception, func, maybe args, maybe kwargs)]
        self.seen_ops_to_errors = {}

    def maybe_raise_errors_on_exit(self) -> None:
        # [FUNCTION DOC] Reconciles collected errors with expected statuses. For operators
        # with no errors: in accept mode records them as xsuccess, otherwise raises
        # OpCheckError if they were expected to fail (unexpected success). For operators
        # expected to succeed that had errors: in accept mode marks them xfail; otherwise
        # raises OpCheckError (chained from the first stored error) with a repro command.
        # Mutates the failures dict in accept mode. Returns nothing when all is as expected.
        # KNOWN ISSUE: the unexpected success message lacks a separator between the file
        # path and the text that follows it.
        # Check expected failures first
        for qualname in self.seen_ops_to_errors:
            option = self.failures_dict.get_status(qualname, self.test_name)
            if len(self.seen_ops_to_errors[qualname]) == 0:
                if should_update_failures_dict():
                    self.failures_dict.set_status(
                        qualname, self.test_name, "xsuccess", comment=""
                    )
                else:
                    if option == "xfail":
                        raise OpCheckError(
                            f"generate_opcheck_tests: Unexpected success for operator "
                            f"{qualname} on test {self.test_name}. This may mean that "
                            f"you have fixed this test failure. Please rerun the test with "
                            f"PYTORCH_OPCHECK_ACCEPT=1 to automatically update the test runner "
                            f"or manually remove the "
                            f"expected failure in the failure dict at "
                            f"{self.failures_dict_path}"
                            f"For more details, see "
                            f"{GDOC}"
                        )
                continue
        failed_ops = []
        for qualname in self.seen_ops_to_errors:
            option = self.failures_dict.get_status(qualname, self.test_name)
            if option != "xsuccess":
                continue
            if len(self.seen_ops_to_errors[qualname]) == 0:
                continue
            failed_ops.append(qualname)
        if not failed_ops:
            return

        if should_update_failures_dict():
            for op in failed_ops:
                self.failures_dict.set_status(op, self.test_name, "xfail")
            return

        # Raise from the first error but also report about all of them to make
        # recording xfails easier.
        ex, op, args, kwargs = self.seen_ops_to_errors[failed_ops[0]][0]
        repro_command = generate_repro(
            self.test_util_name, op, args, kwargs, save_data=should_print_better_repro()
        )
        raise OpCheckError(
            f"Test generated by `generate_opcheck_tests`, {self.test_name}, "
            f"failed on operators {failed_ops}. This usually means that the "
            f"operators are not implemented correctly and may lead to silently "
            f"incorrect behavior. Set PYTORCH_OPCHECK_PRINT_BETTER_REPRO=1 for a standalone repro, "
            f"or please see "
            f"{GDOC} "
            f"for more recommendations. "
            f"To reproduce this problem locally, try to run the following:\n{repro_command}"
        ) from ex

    def __enter__(self, *args, **kwargs):
        # [FUNCTION DOC] Saves then overrides global state: marks the thread as inside
        # opcheck, sets the environment variable TORCHDYNAMO_DISABLE to 1 and turns off
        # functorch's custom op aliasing check. Then enters the base mode and returns its
        # result.
        self.prev_is_opcheck_mode = _is_inside_opcheck_mode.value
        self.prev_dynamo_disable = os.environ.get("TORCHDYNAMO_DISABLE", "")
        _is_inside_opcheck_mode.value = True
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        # When running this test mode, we want to disable
        # default torch.compile custom op checker
        self.prev_functorch_config_for_checking_custom_op = (
            torch._functorch.config.check_custom_op_aliasing
        )
        torch._functorch.config.check_custom_op_aliasing = False
        return super().__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        # [FUNCTION DOC] Restores the saved global state, then evaluates the collected
        # results (maybe_raise_errors_on_exit, which may raise OpCheckError) and in accept
        # mode saves the failures dict to disk. Always exits the base mode and returns its
        # result.
        _is_inside_opcheck_mode.value = self.prev_is_opcheck_mode
        os.environ["TORCHDYNAMO_DISABLE"] = self.prev_dynamo_disable
        torch._functorch.config.check_custom_op_aliasing = (
            self.prev_functorch_config_for_checking_custom_op
        )
        try:
            self.maybe_raise_errors_on_exit()
            if should_update_failures_dict():
                self.failures_dict.save()
        finally:
            result = super().__exit__(*args, **kwargs)
        return result

    def run_test_util(self, op, args, kwargs):
        # [FUNCTION DOC] Runs the configured check on op without copying inputs (the caller
        # already copied). Silently ignores the unsupported fake tensor exception; other
        # exceptions propagate to the caller.
        try:
            self.test_util(op, args, kwargs, copy_inputs=False)
        except torch._subclasses.fake_tensor.UnsupportedFakeTensorException:
            # We might get here if the input is already a FakeTensor
            # or if we're in a torch.compile block. Just ignore these
            # since we can't handle them and reporting them as failures
            # is too noisy.
            pass

    def __torch_function__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Intercepts a call. Non-operator calls, calls while tracing,
        # scripting or compiling, and operators outside the namespaces run unchanged. An
        # operator packet is resolved to its unique overload (RuntimeError if ambiguous).
        # Otherwise the real call is executed on the original inputs and returned, after
        # running the check on copies of the inputs when the expected status is xfail or
        # xsuccess (not skip); any check error is stored for exit-time evaluation, with
        # the arguments kept only when better repros are enabled.
        kwargs = kwargs if kwargs else {}

        # Only intercept calls to operators
        if not isinstance(func, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
            return func(*args, **kwargs)
        if (
            torch.jit.is_tracing()
            or torch.jit.is_scripting()
            or torch._dynamo.is_compiling()
        ):
            return func(*args, **kwargs)
        # Pre-existing code may not use the .default overload. If we see an
        # OpOverloadPacket and we cannot resolve the overload, then we just throw
        # and ask the user to clarify. Otherwise, we attempt to resolve the overload.
        if isinstance(func, torch._ops.OpOverloadPacket):
            func = resolve_unique_overload_or_throw(func)
        qualname = func.name()
        ns = qualname.split("::")[0]
        if ns not in self.namespaces:
            return func(*args, **kwargs)

        args_c, kwargs_c = deepcopy_tensors((args, kwargs))
        result = func(*args, **kwargs)

        option = self.failures_dict.get_status(qualname, self.test_name)
        if option == "xsuccess" or option == "xfail":
            # Suppress all errors during execution. Raise them during __exit__.
            try:
                if qualname not in self.seen_ops_to_errors:
                    self.seen_ops_to_errors[qualname] = []
                self.run_test_util(func, args_c, kwargs_c)
            except Exception as ex:
                if should_print_better_repro():
                    self.seen_ops_to_errors[qualname].append((ex, func, args, kwargs))
                else:
                    self.seen_ops_to_errors[qualname].append((ex, func, None, None))
        elif option == "skip":
            pass
        return result


def should_print_better_repro() -> None:
    # [FUNCTION DOC] Returns True if the environment variable
    # PYTORCH_OPCHECK_PRINT_BETTER_REPRO is set to 1, else False.
    # KNOWN ISSUE: the return annotation says None but a bool is returned.
    """If set, the tests generated by `generate_opcheck_tests` will print a
    repro command on failure.

    In order to print the repro command, we need to save some tensors to disk.
    These will be saved under the following directory:
    {tempfile.gettempdir()}/pytorch_opcheck_safe_to_delete/.

    Although this is a temp folder, it will usually not automatically get cleaned
    up, so you'll need to manually delete it.
    """
    key = "PYTORCH_OPCHECK_PRINT_BETTER_REPRO"
    if key not in os.environ:
        return False
    value = os.environ[key]
    return value == "1" or value == 1


def opcheck(
    op: torch._ops.OpOverload | torch._ops.OpOverloadPacket | CustomOpDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    test_utils: str | Sequence[str] = DEFAULT_TEST_UTILS,
    raise_exception: bool = True,
    rtol: float | None = None,
    atol: float | None = None,
) -> dict[str, str]:
    # [FUNCTION DOC] Public entry point that runs the selected test utils (a name, a
    # sequence of names, or ALL) on one operator call. Accepts an OpOverload, a packet with
    # a single overload, or a CustomOpDef. rtol and atol must be given together. Returns a
    # mapping from util name to SUCCESS or, when raise_exception is false, the caught
    # exception. If raise_exception is true the first failure is raised as OpCheckError
    # (chained). Raises ValueError for bad op type, bad test_utils or a lone rtol/atol.
    """See torch.library.opcheck for docstring"""

    if (rtol is None) ^ (atol is None):
        raise ValueError(
            "opcheck(op, ...): if you specify one of rtol/atol, you must specify both"
        )

    if kwargs is None:
        kwargs = {}
    if isinstance(op, CustomOpDef):
        op = op._opoverload
    if isinstance(op, torch._ops.OpOverloadPacket):
        op = resolve_unique_overload_or_throw(op)
    if not isinstance(op, torch._ops.OpOverload):
        raise ValueError(
            f"opcheck(op, ...): op must be instance of torch._ops.OpOverload, "
            f"e.g. torch.ops.aten.sin.default, got {type(op)}"
        )
    if test_utils == "ALL":
        test_utils = tuple(ALL_TEST_UTILS.keys())
    if isinstance(test_utils, str):
        test_utils = (test_utils,)
    if not isinstance(test_utils, (tuple, list)) or not set(test_utils).issubset(
        ALL_TEST_UTILS.keys()
    ):
        raise ValueError(
            f"opcheck(op, ..., test_utils={test_utils}), expected test_utils "
            f"to be subset of {tuple(ALL_TEST_UTILS.keys())} but it was not"
        )

    results_dict = {}
    for test_util in test_utils:
        tester = ALL_TEST_UTILS[test_util]
        try:
            tester(op, args, kwargs, rtol=rtol, atol=atol)
            results_dict[test_util] = "SUCCESS"
        except Exception as ex:
            if raise_exception:
                raise OpCheckError(
                    f"opcheck(op, ...): {test_util} failed with {ex} "
                    f"(scroll up for stack trace)"
                ) from ex
            results_dict[test_util] = ex
    return results_dict


class OpCheckError(Exception):
    # [CLASS DOC] Exception raised when an opcheck test util fails or when expected
    # failures do not match reality. Carries no extra state.
    pass


def generate_repro(
    test: str,
    op: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    save_data: bool,
    dry_run: bool = False,
) -> str:
    # [FUNCTION DOC] Builds the text of a standalone script that reruns opcheck for the
    # given test util name on the op. If save_data is true the args and kwargs are saved
    # to a timestamped file in a temporary directory (skipped when dry_run) and the script
    # loads them; otherwise placeholders for empty args and kwargs are emitted. Returns
    # the script text. Side effect: may create a directory and write a file.
    if save_data:
        now = datetime.datetime.now()
        path = os.path.join(tempfile.gettempdir(), "pytorch_opcheck_safe_to_delete")
        unix_timestamp = datetime.datetime.timestamp(now) * 100000
        filepath = os.path.join(path, f"repro_{unix_timestamp}.pt")
        if not dry_run:
            os.makedirs(path, exist_ok=True)
            torch.save((args, kwargs), filepath)
        args_kwargs = f'args, kwargs = torch.load("{filepath}")'
    else:
        args_kwargs = (
            "# If you rerun your test with PYTORCH_OPCHECK_PRINT_BETTER_REPRO=1\n"
            "# we will fill them in same (args, kwargs) as in your test\n"
            "args = ()  # args to the operator\n"
            "kwargs = {}  # kwargs to the operator"
        )

    ns, name = op._schema.name.split("::")
    overload = op._overloadname

    repro_command = (
        f"# =========================================================\n"
        f"# BEGIN REPRO SCRIPT\n"
        f"# =========================================================\n"
        f"import torch\n"
        f"from torch.testing._internal.optests import opcheck\n"
        f"\n"
        f"# Make sure you have loaded the library that contains the op\n"
        f"# via an import or torch.ops.load_library(...)\n"
        f"op = torch.ops.{ns}.{name}.{overload}\n"
        f"\n"
        f"{args_kwargs}\n"
        f'opcheck(op, args, kwargs, test_utils="{test}")\n'
        f"# =========================================================\n"
        f"# END REPRO SCRIPT\n"
        f"# =========================================================\n"
    )
    return repro_command


def resolve_unique_overload_or_throw(
    op: torch._ops.OpOverloadPacket,
) -> torch._ops.OpOverload:
    # [FUNCTION DOC] Returns the single overload of an operator packet (the default one if
    # its overload name is empty). Raises RuntimeError listing the overload names if the
    # operator has more than one (or zero) schema.
    all_schemas = torch._C._jit_get_schemas_for_operator(op._qualified_op_name)
    if len(all_schemas) != 1:
        raise RuntimeError(
            f"opcheck can only test operators without overloads. "
            f"Got the following overloads for {op._qualified_op_name}: "
            f"{[schema.overload_name for schema in all_schemas]}"
        )

    overload_name = all_schemas[0].overload_name
    if overload_name == "":
        return op.default
    return getattr(op, overload_name)


DUMP_OPTIONS = {"indent": 2, "sort_keys": True}


FailuresDictData = dict[str, dict[str, dict[str, str]]]


VERSION = 1
DESCRIPTION = (
    f"This is a dict containing failures for tests autogenerated by "
    f"generate_opcheck_tests. "
    f"For more details, please see {GDOC}"
)


class FailuresDict:
    # [CLASS DOC] In-memory form of the failures JSON file. Attributes: path (file
    # location) and data (mapping operator name -> test name -> entry with status and
    # comment). A missing entry means the default status xsuccess. Key methods: load,
    # save, get_status, set_status.
    def __init__(self, path: str, data: FailuresDictData):
        # [FUNCTION DOC] Stores the file path and the data mapping.
        self.path = path
        self.data = data

    @staticmethod
    def load(path, *, create_file=False) -> "FailuresDict":
        # [FUNCTION DOC] Reads the JSON file at path and returns a FailuresDict of its data
        # section. An empty file gives empty data. Raises AssertionError if the data key or
        # the expected version is missing. If create_file is true and the file is absent,
        # intends to create it and return an empty dict.
        # KNOWN ISSUE: the create path calls save on the class instead of the new
        # instance, so it fails with a missing-argument error and never creates the file.
        if create_file and not os.path.exists(path):
            result = FailuresDict(path, {})
            FailuresDict.save()
            return result
        with open(path) as fp:
            contents = fp.read()
            if contents.strip() == "":
                dct = {
                    "_description": DESCRIPTION,
                    "data": {},
                    "_version": VERSION,
                }
            else:
                dct = json.loads(contents)
                if "data" not in dct:
                    raise AssertionError("Expected 'data' in dct")
                if "_version" not in dct or dct["_version"] != VERSION:
                    raise AssertionError(
                        f"Expected '_version' in dct with value {VERSION}"
                    )
        return FailuresDict(path, dct["data"])

    def _save(self, to_str=False) -> str | None:
        # [FUNCTION DOC] HELPER: Serializes description, data and version to indented,
        # key-sorted JSON with a trailing newline. Returns the text if to_str is true;
        # otherwise writes it to the file path and returns None.
        to_dump = {
            "_description": DESCRIPTION,
            "data": self.data,
            "_version": VERSION,
        }
        # json.dumps doesn't end with a newline. Let's add one because files
        # should end in newlines.
        serialized = json.dumps(to_dump, **DUMP_OPTIONS) + "\n"
        if to_str:
            return serialized
        with open(self.path, "w") as fp:
            fp.write(serialized)
        return None

    def save(self) -> None:
        # [FUNCTION DOC] Writes the current data to the file path (overwriting it).
        return self._save()

    def get_status(self, qualname: str, test_name: str) -> str:
        # [FUNCTION DOC] Returns the status string for the operator and test, or xsuccess
        # when either has no entry.
        if qualname not in self.data:
            return "xsuccess"
        dct = self.data[qualname]
        if test_name not in dct:
            return "xsuccess"
        return dct[test_name]["status"]

    def set_status(
        self,
        qualname: str,
        test_name: str,
        status: str,
        *,
        comment: str | None = None,
    ):
        # [FUNCTION DOC] Sets the status for the operator and test, creating entries as
        # needed. Status xsuccess removes the test entry (it is the default). Otherwise
        # stores the status and, if a comment is given, the comment (new entries start with
        # an empty comment). Mutates in memory only; call save to persist.
        if qualname not in self.data:
            self.data[qualname] = {}
        dct = self.data[qualname]
        if test_name not in dct:
            dct[test_name] = {"status": None, "comment": ""}

        if status == "xsuccess":
            # The default status is "xsuccess".
            del dct[test_name]
        else:
            dct[test_name]["status"] = status
            if comment is not None:
                dct[test_name]["comment"] = comment
