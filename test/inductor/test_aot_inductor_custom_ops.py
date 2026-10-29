# Owner(s): ["module: inductor"]
# This test requires libaoti_custom_ops.so to be built, which happens when BUILD_TEST = 1
# [MODULE DOC] AOTInductor tests for custom operators. Requires the test shared library of custom
# operators (built when test building is enabled), which the test case base loads at setup.
# Defines several Python-registered custom ops with fake implementations (optional-tensor schema
# misuse, list and single dynamic-size returns, dtype argument, memory-format and layout
# arguments), a template class of export-compile-compare tests that use both those and the C++
# library ops, a logging test, and CPU and GPU test classes created by copying the template. Exits
# or skips early on Windows CI or missing dependencies.
import logging
import os
import sys
import unittest

import torch
import torch._export
import torch._inductor
import torch._inductor.config
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.export import Dim, export
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_CI,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from torch.utils._python_dispatch import TorchDispatchMode


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    try:
        from .test_aot_inductor_utils import (
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from .test_torchinductor import copy_tests, TestFailure
    except ImportError:
        from test_aot_inductor_utils import (  # @manual=fbcode//caffe2/test/inductor:aot_inductor_utils-library
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            copy_tests,
            TestFailure,
        )
except (unittest.SkipTest, ImportError):
    if __name__ == "__main__":
        sys.exit(0)
    raise


@torch.library.custom_op(
    "aoti_custom_ops::fn_with_incorrect_optional_tensor", mutates_args=()
)
def fn_with_incorrect_optional_tensor(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    # [FUNCTION DOC] Custom op body registered under the aoti_custom_ops namespace. Returns the
    # sum of the first two inputs, adding the third only when it is not None. Its schema declares
    # the third argument as a required tensor, which is the deliberate mismatch exercised by the
    # incorrect-schema test.
    if z is None:
        return x + y
    else:
        return x + y + z


@fn_with_incorrect_optional_tensor.register_fake
def fn_with_incorrect_optional_tensor_fake(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    # [FUNCTION DOC] Fake (shape propagation) implementation matching the real op: sums the first
    # two tensors and also the third when it is not None.
    if z is None:
        return x + y
    else:
        return x + y + z


@torch.library.custom_op(
    "aoti_custom_ops::fn_ret_list_of_single_tensor", mutates_args={}
)
def fn_ret_list_of_single_tensor(x: torch.Tensor) -> list[torch.Tensor]:
    # [FUNCTION DOC] Custom op that sums the input, converts the sum to a 64-bit integer and
    # returns a one-element list holding a random 1-D tensor of that length. Output size is
    # data dependent.
    s = x.sum().to(torch.int64)
    return [torch.randn(s.item())]


@fn_ret_list_of_single_tensor.register_fake
def _(x):
    # [FUNCTION DOC] Fake implementation for the list-returning op: asks the custom-op context
    # for a new dynamic (unbacked) size and returns a one-element list with a random tensor of
    # that size.
    ctx = torch._custom_op.impl.get_ctx()
    i0 = ctx.new_dynamic_size()
    return [torch.randn(i0)]


@torch.library.custom_op("aoti_custom_ops::fn_ret_single_tensor", mutates_args={})
def fn_ret_single_tensor(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Custom op that sums the input, casts the sum to 64-bit integer and returns a
    # random 1-D tensor with that many elements (data-dependent output size).
    s = x.sum().to(torch.int64)
    return torch.randn(s.item())


@fn_ret_single_tensor.register_fake
def _(x):
    # [FUNCTION DOC] Fake implementation for the single-tensor op: allocates a new dynamic size
    # from the custom-op context and returns a random 1-D tensor of that size.
    ctx = torch._custom_op.impl.get_ctx()
    i0 = ctx.new_dynamic_size()
    return torch.randn(i0)


# Custom op to test ScalarType argument serialization
@torch.library.custom_op("aoti_custom_ops::fn_with_dtype_arg", mutates_args={})
def fn_with_dtype_arg(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # [FUNCTION DOC] Custom op taking a tensor and a dtype; returns a zeros tensor shaped like the
    # input with the requested dtype. Used to test serialization of scalar-type arguments.
    return torch.zeros_like(x, dtype=dtype)


@fn_with_dtype_arg.register_fake
def _(x, dtype):
    # [FUNCTION DOC] Fake implementation of the dtype-argument op; returns zeros shaped like the
    # input with the given dtype.
    return torch.zeros_like(x, dtype=dtype)


# Custom ops to test MemoryFormat and Layout argument serialization
# Using lower-level torch.library API since custom_op decorator doesn't support these types
_memory_format_test_lib = torch.library.Library("aoti_custom_ops", "FRAGMENT")  # noqa: SCOPED_LIBRARY
_memory_format_test_lib.define(
    "fn_with_memory_format_arg(Tensor x, MemoryFormat memory_format) -> Tensor"
)


def _fn_with_memory_format_arg_impl(x, memory_format):
    # [FUNCTION DOC] HELPER: Eager kernel registered for the memory-format op on CPU, CUDA and
    # XPU; returns the input made contiguous in the given memory format.
    return x.contiguous(memory_format=memory_format)


_memory_format_test_lib.impl(
    "fn_with_memory_format_arg", _fn_with_memory_format_arg_impl, "CPU"
)
_memory_format_test_lib.impl(
    "fn_with_memory_format_arg", _fn_with_memory_format_arg_impl, "CUDA"
)
_memory_format_test_lib.impl(
    "fn_with_memory_format_arg", _fn_with_memory_format_arg_impl, "XPU"
)


@torch.library.register_fake("aoti_custom_ops::fn_with_memory_format_arg")
def fn_with_memory_format_arg_abstract(x, memory_format):
    # [FUNCTION DOC] Fake (abstract) implementation of the memory-format op; returns the input
    # made contiguous in the requested memory format.
    return x.contiguous(memory_format=memory_format)


_layout_test_lib = torch.library.Library("aoti_custom_ops", "FRAGMENT")  # noqa: SCOPED_LIBRARY
_layout_test_lib.define("fn_with_layout_arg(Tensor x, Layout layout) -> Tensor")


def _fn_with_layout_arg_impl(x, layout):
    # [FUNCTION DOC] HELPER: Eager kernel registered for the layout op on CPU, CUDA and XPU. It
    # ignores the layout value (receiving it without error is the check) and returns a clone of
    # the input.
    # Layout argument is received correctly if we get here without error
    return x.clone()


_layout_test_lib.impl("fn_with_layout_arg", _fn_with_layout_arg_impl, "CPU")
_layout_test_lib.impl("fn_with_layout_arg", _fn_with_layout_arg_impl, "CUDA")
_layout_test_lib.impl("fn_with_layout_arg", _fn_with_layout_arg_impl, "XPU")


@torch.library.register_fake("aoti_custom_ops::fn_with_layout_arg")
def fn_with_layout_arg_abstract(x, layout):
    # [FUNCTION DOC] Fake (abstract) implementation of the layout op; returns a clone of
    # the input.
    return x.clone()


class AOTInductorTestsTemplate:
    # [CLASS DOC] Template of device-agnostic tests (not itself a test case) that export a small
    # module using custom ops, compile it through AOTInductor and compare against eager via the
    # check helper supplied by the target class. Relies on target attributes device and
    # check_model; tests are copied into concrete CPU and GPU classes.
    def test_custom_op_add(self) -> None:
        # [FUNCTION DOC] Runs a model that adds a constant tensor and then a stored weight tensor
        # to the input using the library add op, and checks the compiled result against eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing the device and a random 3 by 3 weight; forward
            # applies the custom add op with a one-element constant and then with the weight.
            def __init__(self, device):
                # [FUNCTION DOC] HELPER: Stores the device and creates a random 3 by 3 weight
                # on it.
                super().__init__()
                self.device = device
                self.w = torch.randn(3, 3, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a one-element constant tensor, applies the
                # custom add op to the input and constant, then to that result and the weight;
                # returns it.
                const = torch.tensor([1], device=self.device)
                x = torch.ops.aoti_custom_ops.custom_add(x, const)
                return torch.ops.aoti_custom_ops.custom_add(x, self.w)

        m = M(self.device).to(device=self.device)
        args = (torch.randn(3, 3, device=self.device),)
        self.check_model(m, args)

    def test_custom_op_add_output_path(self) -> None:
        # [FUNCTION DOC] Sets the AOTInductor output path option to a package file name and
        # expects the model check to raise an exception, since a package path is not
        # accepted there.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the custom add op to its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the custom add op applied to the two inputs.
                return torch.ops.aoti_custom_ops.custom_add(x, y)

        m = M().to(device=self.device)
        args = (
            torch.randn(3, 3, device=self.device),
            torch.randn(3, 3, device=self.device),
        )
        with config.patch("aot_inductor.output_path", "model.pt2"):
            with self.assertRaises(Exception):
                self.check_model(m, args)

    def test_fn_with_optional_tensor_output(self) -> None:
        # [FUNCTION DOC] Checks a model calling the library op that returns a tensor plus two
        # optional tensors against eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the optional-tensor-output custom op.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the result of the optional-tensor-output op on
                # both inputs.
                return torch.ops.aoti_custom_ops.fn_with_optional_tensor_output(x, y)

        m = M().to(device=self.device)
        args = (
            torch.randn(3, 3, device=self.device),
            torch.randn(3, 3, device=self.device),
        )
        self.check_model(m, args)

    def test_fn_with_optional_tensor_output_2(self) -> None:
        # [FUNCTION DOC] Checks a model calling the second variant of the library op that returns
        # a tensor and optional tensors against eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the second optional-tensor-output custom op.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the result of the second optional-tensor-output
                # op on both inputs.
                return torch.ops.aoti_custom_ops.fn_with_optional_tensor_output_2(x, y)

        m = M().to(device=self.device)
        args = (
            torch.randn(3, 3, device=self.device),
            torch.randn(3, 3, device=self.device),
        )
        self.check_model(m, args)

    def test_fn_with_optional_tensor_nullopt_output(self) -> None:
        # [FUNCTION DOC] Checks a model calling the library op whose optional outputs can be empty
        # (null) against eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the nullopt-output custom op.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the result of the op whose optional outputs may
                # be absent.
                return torch.ops.aoti_custom_ops.fn_with_optional_tensor_nullopt_output(
                    x, y
                )

        m = M().to(device=self.device)
        args = (
            torch.randn(3, 3, device=self.device),
            torch.randn(3, 3, device=self.device),
        )
        self.check_model(m, args)

    def test_fn_with_int_output(self) -> None:
        # [FUNCTION DOC] Checks a model that calls the library op returning a tensor, optional
        # tensors and two integers, passing the first dimension size as an integer argument, and
        # combines outputs into a scaled result.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module using the integer-output custom op and scaling the tensor
            # by the sum of the returned and passed integers.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Calls the op with the first dimension size, then returns
                # the tensor and the tensor multiplied by the sum of both returned integers and
                # that size.
                i = x.shape[0]
                z, _, _, i1, i2 = torch.ops.aoti_custom_ops.fn_with_int_output(x, y, i)
                return z, z * (i1 + i2 + i)

        m = M().to(device=self.device)
        args = (
            torch.randn(3, 3, device=self.device),
            torch.randn(3, 3, device=self.device),
        )
        self.check_model(m, args)

    def test_custom_op_all_inputs(self) -> None:
        # [FUNCTION DOC] Calls the library op that takes every supported argument kind (tensors,
        # lists, bools, ints, symbolic ints, floats, scalars, strings, device, and optional
        # variants) with a mix of None and concrete values inside a no-grad region, runs the model
        # once in eager, then checks the compiled result.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that derives symbolic ints from input shapes and calls
            # the all-inputs custom op, returning its result.
            # pyre-fixme[3]: Return type must be annotated.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Only calls the parent constructor.
                super().__init__()

            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Under no-grad, computes sizes from both inputs,
                # concatenates the first input with itself, calls the all-inputs op with keyword
                # arguments built from them and constants, and returns its result.
                with torch.no_grad():
                    x_dim0 = x.shape[0]
                    x_dim1 = x.shape[1]
                    y_dim0 = y.shape[0]
                    y_dim1 = y.shape[1]
                    symint_0 = x_dim0 + x_dim1
                    symint_1 = y_dim0 * y_dim1

                    z = torch.concat((x, x))

                    _2547 = torch.ops.aoti_custom_ops.fn_with_all_inputs(
                        tensor=x,
                        tensors=[x, y],
                        optional_tensors=[None, z],
                        b8=False,
                        b8s=[True, False],
                        i64=42,
                        i64s=[16, 17],
                        symint=symint_0,
                        symints=[symint_0, symint_1],
                        f64=3.14,
                        f64s=[2.2, 3.3],
                        scalar=1.23,
                        scalars=[45, 67],
                        string="hello",
                        strings=["ab", "cde"],
                        # dtype=torch.float16,
                        # memory_format=torch.contiguous_format,
                        # layout=torch.strided,
                        device=torch.device("cpu"),
                        # optional
                        o_tensor=None,
                        o_tensors=[x, y],
                        o_b8=False,
                        o_b8s=[True, False],
                        o_i64=None,
                        o_i64s=[16, 17],
                        o_symint=symint_1,
                        o_symints=[symint_1, symint_0],
                        o_f64=3.14,
                        o_f64s=None,
                        o_scalar=None,
                        o_scalars=[89, 910],
                        o_string="hello",
                        o_strings=["ab", "cde"],
                        # o_dtype=None,
                        # o_memory_format=torch.contiguous_format,
                        # o_layout=torch.strided,
                        o_device=None,
                    )

                return _2547

        m = MyModel().to(device=self.device)
        x = torch.zeros(4, 8, device=self.device)
        y = torch.ones(3, 9, device=self.device)
        args = (x, y)
        m(*args)

        self.check_model(m, args)

    def test_custom_op_with_multiple_outputs(self) -> None:
        # [FUNCTION DOC] Checks a model that chains library ops returning a tuple of tensors, a
        # list of tensors, and a tuple of a tensor and a list; runs eager once then compares the
        # compiled results.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module chaining the tuple, list and mixed multi-output
            # custom ops.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Adds the inputs, applies the tuple-output, list-output
                # and mixed-output ops in turn and returns all seven resulting tensors.
                out = x + y
                # tuple of Tensor output
                out3, out4 = torch.ops.aoti_custom_ops.fn_with_tuple_output(out, 1)
                # TensorList output
                out5, out6 = torch.ops.aoti_custom_ops.fn_with_list_output(
                    [out3, out4], 1
                )
                # tuple of Tensor and TensorList
                out7, [out8, out9] = torch.ops.aoti_custom_ops.fn_with_mix_outputs(
                    out5, [out6, out4]
                )
                return out3, out4, out5, out6, out7, out8, out9

        m = Model().to(device=self.device)
        args = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )
        m(*args)

        self.check_model(m, args)

    def test_custom_op_out_variant_without_return(self) -> None:
        # [FUNCTION DOC] Checks a model calling an out-variant op that mutates its second argument
        # and returns nothing; the model returns the mutated second input.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the in-place out-variant op and returning its
            # second input.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Calls the op for its side effect on the second input and
                # returns that input.
                torch.ops.aoti_custom_ops.fn_out_variant_without_return(x, y)
                return y

        m = Model().to(device=self.device)
        args = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        m(*args)

        self.check_model(m, args)

    def test_custom_op_with_reinterpret_view_inputs(self) -> None:
        # [FUNCTION DOC] Checks the default-input op applied to a permuted (reinterpreted
        # view) input.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module permuting the input dimensions then calling the
            # default-input op.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Transposes the input and returns the default-input op
                # applied with an integer of one.
                out = x.permute([1, 0])
                return torch.ops.aoti_custom_ops.fn_with_default_input(out, 1)

        m = Model().to(device=self.device)
        args = (torch.randn(2, 3, device=self.device),)

        self.check_model(m, args)

    def test_custom_op_with_concat_inputs(self) -> None:
        # [FUNCTION DOC] Checks the default-input op applied to the concatenation of two inputs
        # along dimension zero.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module concatenating two inputs and calling the
            # default-input op.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Concatenates both inputs along the first dimension and
                # returns the default-input op on it with integer one.
                out = torch.concat([x, y], dim=0)
                return torch.ops.aoti_custom_ops.fn_with_default_input(out, 1)

        m = Model().to(device=self.device)
        args = (
            torch.randn(2, 3, device=self.device),
            torch.randn(2, 3, device=self.device),
        )

        self.check_model(m, args)

    def test_custom_op_missing_arg_with_default_value(self) -> None:
        # [FUNCTION DOC] Checks a call to the default-input op that omits its optional second
        # argument, so the default must be filled in by the compiler.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the default-input op without its second argument.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the default-input op applied to the input alone.
                # missing second arg
                return torch.ops.aoti_custom_ops.fn_with_default_input(x)

        m = Model().to(device=self.device)
        args = (torch.randn(2, 3, device=self.device),)

        self.check_model(m, args)

    def test_custom_op_return_list_of_single_tensor(self) -> None:
        # [FUNCTION DOC] Checks a model using the Python-registered op that returns a list with
        # one data-dependent-size tensor, then adds one to the first element.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking the first list element of the op result plus one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the first element of the op output with
                # one added.
                return torch.ops.aoti_custom_ops.fn_ret_list_of_single_tensor(x)[0] + 1

        m = Model().to(device=self.device)
        args = (torch.randn(3, 4),)
        self.check_model(m, args)

    def test_custom_op_return_single_tensor(self) -> None:
        # [FUNCTION DOC] Checks a model using the Python-registered op that returns one
        # data-dependent-size tensor, then adds one.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding one to the single-tensor op result.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the op output plus one.
                return torch.ops.aoti_custom_ops.fn_ret_single_tensor(x) + 1

        m = Model().to(device=self.device)
        args = (torch.randn(3, 4),)
        self.check_model(m, args)

    @unittest.skipIf(IS_FBCODE, "FbProxyExecutor doesn't have these error msgs")
    def test_incorrect_custom_op_schema(self):
        # [FUNCTION DOC] Skipped in FB code. Passes a None for the third argument of an op whose
        # declared schema requires a tensor, and expects the model check to raise a runtime error
        # mentioning an expected extern kernel.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the incorrectly-declared optional tensor op with
            # a None third argument.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the op applied to both inputs and None.
                return torch.ops.aoti_custom_ops.fn_with_incorrect_optional_tensor(
                    x, y, None
                )

        m = M().to(device=self.device)
        args = (
            torch.randn(2, 3, device=self.device),
            torch.randn(2, 3, device=self.device),
        )

        with self.assertRaisesRegex(RuntimeError, "Expected extern kernel"):
            self.check_model(m, args)

    def test_boxed_run_inputs_clearing(self):
        # [FUNCTION DOC] Exports the add model, builds and loads an AOTInductor package, then runs
        # the loader boxed entry point on a list of inputs under a dispatch mode that, when the
        # add op is dispatched, records that it was seen and asserts the input list is already
        # empty (inputs freed before the kernel runs). Afterwards asserts the list is empty and
        # the sentinel was seen.
        # Borrowed from test_torchinductor
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the custom add op to its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the custom add op applied to the two inputs.
                return torch.ops.aoti_custom_ops.custom_add(x, y)

        inps = [
            torch.rand(5, 5, device=self.device),
            torch.rand(5, 5, device=self.device),
        ]
        model = Model().to(device=self.device)
        # NOTE: There are additional references to inps if we use
        # strict=True here, which will cause inps not deallocated
        # in time later in this test.
        ep = torch.export.export(model, tuple(inps), strict=False)
        package = torch._inductor.aoti_compile_and_package(ep)
        fn_compiled = torch._inductor.aoti_load_package(package)

        test_self = self
        sentinel_seen = False

        class TestRefMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that checks the enclosing test's input list has
            # been cleared at the time the add op executes.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Intercepts every dispatched op; for the add op it sets
                # the sentinel flag and asserts the input list is empty. Always forwards to the
                # real op and returns its result.
                kwargs = kwargs if kwargs else {}
                nonlocal inps
                nonlocal test_self
                nonlocal sentinel_seen
                if func is torch.ops.aoti_custom_ops.custom_add.default:
                    # inputs should be deallocated by this point
                    sentinel_seen = True
                    test_self.assertEqual(len(inps), 0)

                return func(*args, **kwargs)

        with TestRefMode():
            fn_compiled.loader.boxed_run(inps)

        self.assertEqual(len(inps), 0)
        self.assertTrue(sentinel_seen)

    def test_custom_op_with_dtype_arg_various_dtypes(self) -> None:
        # [FUNCTION DOC] Checks a model that calls the dtype-argument op with five different
        # dtypes (float32, float64, int32, uint16, bfloat16) and returns all five results.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the dtype-argument op for five dtypes.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple of five outputs of the dtype-argument op,
                # one per dtype.
                a = torch.ops.aoti_custom_ops.fn_with_dtype_arg(x, torch.float32)
                b = torch.ops.aoti_custom_ops.fn_with_dtype_arg(x, torch.float64)
                c = torch.ops.aoti_custom_ops.fn_with_dtype_arg(x, torch.int32)
                d = torch.ops.aoti_custom_ops.fn_with_dtype_arg(x, torch.uint16)
                e = torch.ops.aoti_custom_ops.fn_with_dtype_arg(x, torch.bfloat16)
                return a, b, c, d, e

        m = Model().to(device=self.device)
        args = (torch.randn(2, 2, device=self.device),)
        self.check_model(m, args)

    def test_custom_op_with_memory_format_arg(self) -> None:
        # [FUNCTION DOC] Checks a model that calls the memory-format-argument op with the
        # channels-last format on a 4-D input.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the memory-format op with channels-last.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the memory-format op applied to the input
                # with channels-last.
                return torch.ops.aoti_custom_ops.fn_with_memory_format_arg(
                    x, torch.channels_last
                )

        m = Model().to(device=self.device)
        args = (torch.randn(1, 3, 4, 4, device=self.device),)
        self.check_model(m, args)

    def test_custom_op_with_layout_arg(self) -> None:
        # [FUNCTION DOC] Checks a model that calls the layout-argument op with the strided layout.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the layout op with the strided layout.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the layout op applied to the input with
                # strided layout.
                return torch.ops.aoti_custom_ops.fn_with_layout_arg(x, torch.strided)

        m = Model().to(device=self.device)
        args = (torch.randn(4, 4, device=self.device),)
        self.check_model(m, args)

    @unittest.skipIf(IS_FBCODE, "unable to find library -laoti_custom_ops")
    def test_custom_op_square(self) -> None:
        # [FUNCTION DOC] Skipped in FB code. Compiles a model calling the square op, using config
        # patches that register C shim declarations for the CPU, CUDA and XPU variants and the
        # custom operator library name to link against, then checks against eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the square custom op.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the square op applied to the input.
                return torch.ops.aoti_custom_ops.fn_square(x)

        m = Model().to(device=self.device)
        args = (torch.randn(2, 3, device=self.device),)
        with (
            config.patch(
                "aot_inductor.custom_ops_to_c_shims",
                {
                    torch.ops.aoti_custom_ops.fn_square.default: [
                        """
                AOTITorchError
                aoti_torch_cpu_fn_square(
                    AtenTensorHandle input,
                    AtenTensorHandle* ret)""",
                        """
                AOTITorchError
                aoti_torch_cuda_fn_square(
                    AtenTensorHandle input,
                    AtenTensorHandle* ret)""",
                        """
                AOTITorchError
                aoti_torch_xpu_fn_square(
                    AtenTensorHandle input,
                    AtenTensorHandle* ret)""",
                    ],
                },
            ),
            config.patch(
                "aot_inductor.custom_op_libs",
                ["aoti_custom_ops"],
            ),
        ):
            self.check_model(m, args)


class AOTInductorLoggingTest(LoggingTestCase):
    # [CLASS DOC] Logging test case verifying logging behavior of AOTInductor compilation.
    @make_logging_test(dynamic=logging.DEBUG)
    def test_shape_env_reuse(self, records):
        # [FUNCTION DOC] Runs under a logger test decorator capturing debug records for the
        # dynamic-shapes logger. Exports a trivial model with automatic dynamic dimensions,
        # compiles it with the AOT compile entry point, and asserts that exactly one record with
        # the message for creating a shape environment was logged, meaning the shape environment
        # is reused.
        # make sure ShapeEnv is only created once and reused afterwards
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Trivial module adding two to its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus two.
                return x + 2

        inputs = (torch.randn(4, 4),)
        dynamic_shapes = {
            "x": {0: Dim.AUTO, 1: Dim.AUTO},
        }
        ep = export(Foo(), inputs, dynamic_shapes=dynamic_shapes, strict=False)
        with torch.no_grad():
            torch._inductor.aot_compile(ep.module(), inputs)
        self.assertEqual([r.msg == "create_env" for r in records].count(True), 1)


common_utils.instantiate_parametrized_tests(AOTInductorTestsTemplate)


class AOTICustomOpTestCase(TestCase):
    # [CLASS DOC] Base test case that loads the custom-ops shared library before each test.
    # Library location differs for FB and Sandcastle, macOS (skipped), and other platforms.
    def setUp(self):
        # [FUNCTION DOC] Loads the custom operator library: by target name in FB or Sandcastle,
        # skipped with an error on macOS, otherwise finds the library file (a dll name on Windows)
        # and raises a skip if it does not exist; loads it into the operator registry. Finally
        # calls the parent setup. Side effect: registers C++ ops globally.
        if IS_SANDCASTLE or IS_FBCODE:
            torch.ops.load_library("//caffe2/test/inductor:custom_ops")
        elif IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        else:
            lib_file_path = find_library_location("libaoti_custom_ops.so")
            if IS_WINDOWS:
                lib_file_path = find_library_location("aoti_custom_ops.dll")
            if not os.path.exists(lib_file_path):
                raise unittest.SkipTest("libaoti_custom_ops not built!")
            torch.ops.load_library(str(lib_file_path))
        super().setUp()


def fail_cpu(is_skip=False):
    # [FUNCTION DOC] HELPER: Builds a failure marker for the cpu suffix; skip or expected failure
    # depending on the argument.
    return TestFailure(
        ("cpu",),
        is_skip=is_skip,
    )


def fail_gpu(suffixes: tuple[str, ...], is_skip=False):
    # [FUNCTION DOC] HELPER: Builds a failure marker for the given GPU suffixes; skip or expected
    # failure depending on the flag.
    return TestFailure(
        suffixes,
        is_skip=is_skip,
    )


# test_failures, xfail by default, set is_skip=True to skip
CPU_TEST_FAILURES = {
    # TODO: failed internally
    "test_multiple_output_alias": fail_cpu(is_skip=True),
}

# test_failures, xfail by default, set is_skip=True to skip
GPU_TEST_FAILURES = {
    # quantized unsupported for GPU
    "test_quantized_linear": fail_gpu(("cuda", "xpu")),
    "test_quanatized_int8_linear": fail_gpu(("cuda", "xpu")),
    "test_quantized_linear_bias_none": fail_gpu(("cuda", "xpu")),
}


class AOTInductorTestABICompatibleCpu(AOTICustomOpTestCase):
    # [CLASS DOC] CPU target class: configuration attributes only (cpu device, shared check
    # helpers, stack allocation and minimal arrayref interface disabled). Template tests are
    # copied in under the cpu suffix.
    device = "cpu"
    device_type = "cpu"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpu,
    "cpu",
    CPU_TEST_FAILURES,
)


@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
class AOTInductorTestABICompatibleGpu(AOTICustomOpTestCase):
    # [CLASS DOC] GPU target class (skipped on macOS): configuration attributes only, using
    # the configured accelerator type as device. Template tests are copied in under the
    # accelerator suffix.
    device = GPU_TYPE
    device_type = GPU_TYPE
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleGpu,
    GPU_TYPE,
    GPU_TEST_FAILURES,
)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_GPU_AND_TRITON or sys.platform == "darwin":
        run_tests(needs="filelock")
