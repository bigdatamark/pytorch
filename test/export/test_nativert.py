# Owner(s): ["oncall: export"]
# [MODULE DOC] Tests that run exported programs through the native C++ runtime (nativert). Provides
# a helper that decomposes a program, packages it as a pt2 archive, runs it with the C++ model
# runner and compares to eager results, mocked export functions that call that helper after every
# export, a factory creating generated variants of the main export test class, and AOTI-delegate
# and scalar-overload tests. Generated variants are only created in fbcode builds, and the main
# block skips running on XPU.


import copy
import tempfile
import unittest

from parameterized import parameterized

import torch
import torch._dynamo as torchdynamo
from torch._C._nativert import PyModelRunner
from torch._dynamo.test_case import TestCase
from torch._environment import is_fbcode
from torch._subclasses.fake_tensor import FakeTensor
from torch.nativert.backends._lower_utils import (
    lower_exported_program,
    package_nativert_with_aoti_delegate,
)
from torch.testing._internal.common_utils import IS_WINDOWS
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.utils import _pytree as pytree


try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing

from torch.export import export


test_classes = {}


def _use_real_inputs(ep):
    # [FUNCTION DOC] HELPER: Returns a shallow copy of an exported program in which fake tensors in
    # the example inputs, state dict and constants are replaced by random real tensors of the same
    # shape, device and requires_grad (parameters are rewrapped as parameters). Example inputs are
    # replaced only when at least one fake tensor was found. Uses the update method to produce the
    # final program.
    ep = copy.copy(ep)

    has_fake_tensor = False

    def _to_real_tensor(t):
        # [FUNCTION DOC] HELPER: Converts one value. A parameter is rebuilt from its converted
        # data; a fake tensor sets a flag in the enclosing function and becomes a random real
        # tensor with same shape, device and gradient flag; anything else is returned unchanged.
        if isinstance(t, torch.nn.Parameter):
            return torch.nn.Parameter(_to_real_tensor(t.data))
        if isinstance(t, FakeTensor):
            nonlocal has_fake_tensor
            has_fake_tensor = True
            return torch.randn(t.shape, device=t.device, requires_grad=t.requires_grad)
        return t

    new_example_inputs = pytree.tree_map_only(
        (torch.Tensor, torch.nn.Parameter), _to_real_tensor, ep.example_inputs
    )
    if has_fake_tensor:
        ep.example_inputs = new_example_inputs

    ep = ep._update(
        ep.graph_module,
        ep.graph_signature,
        state_dict=pytree.tree_map_only(
            (torch.Tensor, torch.nn.Parameter), _to_real_tensor, ep.state_dict
        ),
        constants=pytree.tree_map_only(
            (torch.Tensor, torch.nn.Parameter), _to_real_tensor, ep.constants
        ),
    )
    return ep


def _is_supported_types(arg) -> bool:
    # [FUNCTION DOC] HELPER: Returns whether the argument is made only of types the runner can take
    # directly: tensors, ints, floats, bools, strings and None, nested in lists, tuples and dicts.
    # Lists and dicts must additionally hold values of a single type. Anything else gives false.
    if isinstance(arg, list):
        return (
            all(_is_supported_types(a) for a in arg)
            and len({type(a) for a in arg}) <= 1
        )
    elif isinstance(arg, tuple):
        return all(_is_supported_types(a) for a in arg)
    elif isinstance(arg, dict):
        return (
            all(_is_supported_types(a) for a in arg.values())
            and len({type(a) for a in arg.values()}) <= 1
        )
    elif isinstance(arg, (torch.Tensor, int, float, bool, str)):
        return True
    elif arg is None:
        return True
    else:
        return False


def run_with_nativert(ep):
    # [FUNCTION DOC] HELPER: Runs an exported program through the native runtime and checks it
    # against eager. Deep-copies and decomposes the program, makes inputs real, writes a pt2
    # archive to a temporary file (flushing it), evaluates the module in eager mode (any exception
    # there skips the test) and runs the C++ model runner with the same seeds, using flat inputs
    # when the arguments are not of directly supported types. Raises AssertionError on count, type,
    # shape, dtype or device mismatch, and uses assert_close for tensors (equal_nan) and equality
    # otherwise. A RuntimeError mentioning an unknown pytree node type is swallowed since pytree
    # types cannot be registered in Python tests; other RuntimeErrors propagate. Returns the
    # original program.
    # Downstream tests might mutate the exported program in subtle ways, so
    # we need to make a copy here.
    ep_infer = copy.deepcopy(ep)
    ep_infer = _use_real_inputs(ep_infer.run_decompositions())
    MODEL_NAME = "forward"

    # TODO Does named tempfile have collision?
    with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
        torch.export.pt2_archive._package.package_pt2(
            f, exported_programs={MODEL_NAME: ep_infer}
        )
        # Flush so PyModelRunner, which reopens the file by name below, sees the
        # fully written archive (the zip central directory is written last).
        f.flush()
        filename = f.name

        try:
            ep_args, ep_kwargs = ep_infer.example_inputs
            ep_args_copied, ep_kwargs_copied = (
                copy.deepcopy(ep_args),
                copy.deepcopy(ep_kwargs),
            )
            torch.manual_seed(0)
            try:
                flat_expected = pytree.tree_leaves(
                    ep_infer.module()(*ep_args_copied, **ep_kwargs_copied)
                )
            except Exception as e:
                raise unittest.case.SkipTest(str(e)) from e

            model_runner = PyModelRunner(filename, MODEL_NAME)
            torch.manual_seed(0)
            if _is_supported_types((ep_args, ep_kwargs)):
                results = model_runner.run(*ep_args, **ep_kwargs)
            else:
                results = model_runner.run_with_flat_inputs_and_outputs(
                    *pytree.tree_leaves((ep_args, ep_kwargs))
                )
            flat_results = pytree.tree_leaves(results)
            if len(flat_results) != len(flat_expected):
                raise AssertionError(
                    f"Expected {len(flat_expected)} results, got {len(flat_results)}"
                )
            for result, expected in zip(flat_results, flat_expected):
                if type(result) is not type(expected):
                    raise AssertionError(
                        f"Expected type {type(expected)}, got {type(result)}"
                    )
                if isinstance(result, torch.Tensor) and isinstance(
                    expected, torch.Tensor
                ):
                    if result.shape != expected.shape:
                        raise AssertionError(
                            f"Expected shape {expected.shape}, got {result.shape}"
                        )
                    if result.dtype != expected.dtype:
                        raise AssertionError(
                            f"Expected dtype {expected.dtype}, got {result.dtype}"
                        )
                    if result.device != expected.device:
                        raise AssertionError(
                            f"Expected device {expected.device}, got {result.device}"
                        )
                    torch.testing.assert_close(result, expected, equal_nan=True)
                else:
                    if result != expected:
                        raise AssertionError(f"Expected {expected}, got {result}")
        except RuntimeError as e:
            # User need to register pytree type on the cpp side, which
            # cannot be tested in python unittest.
            if "Unknown pytree node type" in str(e):
                pass
            else:
                raise e
        return ep


def mocked_nativert_export_strict(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement export function: calls the real export in strict mode
    # unless the caller passed a strict keyword, runs the native runtime check on the result and
    # returns the program.
    if "strict" in kwargs:
        ep = export(*args, **kwargs)
    else:
        ep = export(*args, **kwargs, strict=True)

    run_with_nativert(ep)
    return ep


def mocked_nativert_export_nonstrict(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement export function: calls the real export in non-strict mode
    # unless the caller passed a strict keyword, runs the native runtime check on the result and
    # returns the program.
    if "strict" in kwargs:
        ep = export(*args, **kwargs)
    else:
        ep = export(*args, **kwargs, strict=False)

    run_with_nativert(ep)
    return ep


def make_dynamic_cls(cls, strict=False):
    # [FUNCTION DOC] HELPER: Creates a NativeRT-prefixed variant of a test class using the strict
    # or non-strict mocked export, with matching name suffix and expected-failure marker, and tests
    # that are expected failures are skipped unless nothing else fails. Registers it in the module
    # mapping and module globals (required for discovery) and fixes its module attribute. Unlike
    # the sibling modules it returns nothing.
    cls_prefix = "NativeRT"

    if strict:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            cls_prefix,
            test_export.CPP_RUNTIME_STRICT_SUFFIX,
            mocked_nativert_export_strict,
            xfail_prop="_expected_failure_cpp_runtime",
            test_only_if_no_xfail=True,
        )
    else:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            cls_prefix,
            test_export.CPP_RUNTIME_NONSTRICT_SUFFIX,
            mocked_nativert_export_nonstrict,
            xfail_prop="_expected_failure_cpp_runtime_non_strict",
            test_only_if_no_xfail=True,
        )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__


@unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
@unittest.skipIf(not is_fbcode(), "FBcode only for now")
class TestNativeRT(TestCase):
    # [CLASS DOC] Native runtime tests, running only in fbcode on non-Windows with Dynamo support.
    # Builds at class definition time a list of parameter tuples (device, module, sample inputs)
    # for cpu and, when CUDA with Triton exists, cuda, covering a single-output model, a
    # multi-output model and a nested-tuple input model.
    @staticmethod
    def get_module():
        # [FUNCTION DOC] HELPER: Static factory returning a linear plus relu module (four
        # features).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear layer followed by relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer and relu.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer then relu.
                return self.relu(self.linear(x))

        return M()

    @staticmethod
    def get_module_multi_output():
        # [FUNCTION DOC] HELPER: Static factory returning a module that returns the activation and
        # the input.
        class MMultiOutput(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a pair of the relu(linear) output and the
            # unchanged input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer and relu.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple of the relu(linear) result and the input.
                return (self.relu(self.linear(x)), x)

        return MMultiOutput()

    @staticmethod
    def get_model_pytree():
        # [FUNCTION DOC] HELPER: Static factory returning a module that takes a nested tuple input.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two linear layers applied to a nested tuple of three
            # tensors.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two linear layers.
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Unpacks the input into one tensor and a pair, applies the
                # first linear to the first and the second linear to each of the other two, and
                # returns a nested tuple of the results.
                x1, (x2, x3) = x
                y1 = self.linear1(x1)
                y2 = self.linear2(x2)
                y3 = self.linear2(x3)
                return (y1, (y2, y3))

        return M()

    parameters = []
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            continue
        for module, sample_inputs in [
            (get_module.__func__().to(device), (torch.randn(4, 4).to(device),)),
            (
                get_module_multi_output.__func__().to(device),
                (torch.randn(4, 4).to(device),),
            ),
            (
                get_model_pytree.__func__().to(device),
                (
                    (
                        torch.randn(4, 4).to(device),
                        (
                            torch.randn(4, 4).to(device),
                            torch.randn(4, 4).to(device),
                        ),
                    ),
                ),
            ),
        ]:
            parameters.append(
                (
                    device,
                    module,
                    sample_inputs,
                )
            )

    @parameterized.expand(parameters)
    def test_aoti(self, device, m, sample_inputs):
        # [FUNCTION DOC] Parameterized over device, module and inputs. Exports the module, lowers
        # it to an AOTI backend delegate, packages the original and delegate programs into a pt2
        # archive in a temporary file, runs the delegate with the C++ model runner and compares
        # results with eager outputs, as in the shared helper (counts, types, shapes, dtypes,
        # devices, values). Eager failures cause a skip; unknown pytree node RuntimeErrors are
        # ignored.
        MODEL_NAME = "model"
        BACKEND_ID = "aoti"

        # get the original EP
        original_ep = torch.export.export(m, sample_inputs)

        aoti_delegate_ep, aoti_files = lower_exported_program(
            original_ep, MODEL_NAME, BACKEND_ID
        )

        # package everything needed for the NativeRT to execute the AOTI delegate
        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            package_nativert_with_aoti_delegate(
                f,
                MODEL_NAME,
                BACKEND_ID,
                original_ep,
                aoti_delegate_ep,
                aoti_files,
            )
            filename = f.name

            try:
                ep_args, ep_kwargs = aoti_delegate_ep.example_inputs
                ep_args_copied, ep_kwargs_copied = (
                    copy.deepcopy(ep_args),
                    copy.deepcopy(ep_kwargs),
                )
                torch.manual_seed(0)
                try:
                    flat_expected = pytree.tree_leaves(
                        aoti_delegate_ep.module()(*ep_args_copied, **ep_kwargs_copied)
                    )
                except Exception as e:
                    raise unittest.case.SkipTest(str(e)) from e

                model_runner = PyModelRunner(filename, f"{MODEL_NAME}-{BACKEND_ID}")
                torch.manual_seed(0)
                if _is_supported_types((ep_args, ep_kwargs)):
                    results = model_runner.run(*ep_args, **ep_kwargs)
                else:
                    results = model_runner.run_with_flat_inputs_and_outputs(
                        *pytree.tree_leaves((ep_args, ep_kwargs))
                    )
                flat_results = pytree.tree_leaves(results)
                if len(flat_results) != len(flat_expected):
                    raise AssertionError(
                        f"Expected {len(flat_expected)} results, got {len(flat_results)}"
                    )
                for result, expected in zip(flat_results, flat_expected):
                    if type(result) is not type(expected):
                        raise AssertionError(
                            f"Expected type {type(expected)}, got {type(result)}"
                        )
                    if isinstance(result, torch.Tensor) and isinstance(
                        expected, torch.Tensor
                    ):
                        if result.shape != expected.shape:
                            raise AssertionError(
                                f"Expected shape {expected.shape}, got {result.shape}"
                            )
                        if result.dtype != expected.dtype:
                            raise AssertionError(
                                f"Expected dtype {expected.dtype}, got {result.dtype}"
                            )
                        if result.device != expected.device:
                            raise AssertionError(
                                f"Expected device {expected.device}, got {result.device}"
                            )
                        torch.testing.assert_close(result, expected, equal_nan=True)
                    else:
                        if result != expected:
                            raise AssertionError(f"Expected {expected}, got {result}")
            except RuntimeError as e:
                # User need to register pytree type on the cpp side, which
                # cannot be tested in python unittest.
                if "Unknown pytree node type" in str(e):
                    pass
                else:
                    raise e


@unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
@unittest.skipIf(not is_fbcode(), "FBcode only for now")
class TestSelectScalarOverload(TestCase):
    # [CLASS DOC] Test case (fbcode only) checking that the runtime rewrites scalar overload calls
    # on load.
    def test_floor_divide_default_scalar(self) -> None:
        # [FUNCTION DOC] Exports a floor division of an integer tensor by a constant, confirms the
        # graph has the default overload node with a scalar argument, then runs the program through
        # the native runtime helper so the load-time overload selection is exercised end to end
        # against eager.
        # PT2 export lowers `x // 10` to aten.floor_divide.default with a scalar
        # Int argument (the #90923 Scalar->Tensor broadcast).  Its default
        # schema is (Tensor, Tensor), so nativert's boxed C10Kernel dispatch
        # raises "Expected Tensor but got Int" unless selectScalarOverload
        # rewrites the node to floor_divide.Scalar during load.  Running the
        # exported program through nativert exercises that pass end to end.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning floor division of its input by ten.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the floor division of the input by ten.
                return torch.floor_divide(x, 10)

        ep = export(M(), (torch.randint(0, 1000, (999,), dtype=torch.long),))
        # The export must actually produce the .default overload this fixes.
        self.assertTrue(
            any(
                n.op == "call_function"
                and n.target is torch.ops.aten.floor_divide.default
                for n in ep.graph.nodes
            )
        )

        # Run the exported program through nativert -- which applies
        # selectScalarOverload on load -- and compare against eager.  Without
        # the pass, boxed C10Kernel dispatch raises "Expected Tensor but got
        # Int" on the floor_divide.default node.
        run_with_nativert(ep)


if is_fbcode():
    for test in [test_export.TestExport]:
        make_dynamic_cls(test, strict=True)
        make_dynamic_cls(test, strict=False)
    del test


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # nativert has not been supported on XPU yet.
    if not torch.xpu.is_available():
        run_tests()
