# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for dtype propagation and dtype-aware code generation in Inductor's Triton
# backend. At import time it appends the test directory to the module search path, imports
# functorch and filelock, and builds a module-level list of pointwise OpInfo entries: names of
# non-mutating lowered pointwise ops are collected from the lowerings registry, then the op
# database is filtered to those names, excluding variants with reduction in their variant name.
# A single test class runs those ops with runtime dtype/shape asserts enabled, plus targeted
# tests of float32 upcasting in generated Triton code, low precision reductions, symbolic
# (unbacked/dynamic) scalar kernel argument dtypes and 64-bit random generation. The class is
# instantiated for CUDA and XPU only; when run as a script tests run only if a GPU is available
# (requires filelock).
import importlib
import os
import re
import sys

import torch
from torch._dynamo.utils import disable_cache_limit
from torch._inductor import config
from torch._inductor.codegen.triton import OpDtypeSupport
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code, run_and_get_triton_code, triton_type
from torch.fx.operator_schemas import get_signature_for_torch_op
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


importlib.import_module("functorch")
importlib.import_module("filelock")


from torch._inductor.lowering import lowerings
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.inductor_utils import HAS_GPU


unique_pointwise_op_names = set()

for op in lowerings:
    if not isinstance(op, torch._ops.OpOverload):
        continue

    if torch.Tag.pointwise not in op.tags:
        continue

    if op._schema.is_mutable:
        continue

    op_name = (op.name().split("::")[-1]).split(".")[0]
    unique_pointwise_op_names.add(op_name)

pointwise_ops = [
    op
    for op in op_db
    if op.name in unique_pointwise_op_names and "reduction" not in op.variant_test_name
]


class TestCase(InductorTestCase):
    # [CLASS DOC] Test case (instantiated per device type for cuda and xpu) with generic OpInfo-
    # driven dtype propagation test and many specific checks of generated Triton source for
    # casts, static dtype assertions and kernel signature dtypes. It holds no state of its own.
    @ops(
        pointwise_ops,
        allowed_dtypes=(
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.bool,
        ),
    )
    # @config.patch("triton.codegen_upcast_to_fp32", False) # TODO enable
    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    @config.patch("test_configs.static_cpp_dtype_assert", True)
    @disable_cache_limit()
    def test_op_dtype_propagation(self, op, dtype):
        # [FUNCTION DOC] OpInfo test over pointwise ops restricted to float32, float64, int32,
        # int64 and bool. Runs under configs enabling runtime Triton dtype and shape assertions
        # and static C++ dtype assertions, with the dynamo cache limit disabled. For each sample
        # input it runs the op eagerly and via a dynamic-shape compiled wrapper and asserts both
        # results are equal.
        def run(op, args, kwargs):
            # [FUNCTION DOC] HELPER: Calls the given operator with the given positional and
            # keyword arguments and returns its result; compiled as a unit so the operator runs
            # inside the compiled region.
            return op(*args, **kwargs)

        sample_inputs_itr = op.sample_inputs(GPU_TYPE, dtype, requires_grad=False)
        for sample_input in sample_inputs_itr:
            args = (sample_input.input,) + sample_input.args
            kwargs = sample_input.kwargs
            out = run(op.get_op(), args, kwargs)

            out_c = torch.compile(run, dynamic=True)(op.get_op(), args, kwargs)
            self.assertEqual(out, out_c)

    @requires_gpu()
    @parametrize("upcast_to_fp32", [False, True])
    @config.patch("triton.use_block_ptr", True)
    def test_codegen_upcast_to_fp32(self, upcast_to_fp32):
        # [FUNCTION DOC] Parametrized over whether loads are upcast to float32. With block
        # pointers enabled, compiles a product of four float16 tensors and checks that the
        # generated Triton source contains a cast to float32 if and only if upcasting is
        # configured.
        @torch.compile
        def func(a, b, c, d):
            # [FUNCTION DOC] HELPER: Compiled product of four tensors.
            return a * b * c * d

        inps = (torch.rand((32, 32), device=GPU_TYPE, dtype=torch.float16),) * 4
        with config.patch("triton.codegen_upcast_to_fp32", upcast_to_fp32):
            func_opt = torch.compile(func, backend="inductor")
            code = run_and_get_triton_code(func_opt, *inps)
            fp32_cast_in_code = "to(tl.float32)" in code
            self.assertEqual(fp32_cast_in_code, upcast_to_fp32)

    @requires_gpu()
    @parametrize("input_shape", [(32, 32), (32, 128), (256, 32)])
    @parametrize(
        "reduction_func",
        [
            torch.prod,
            torch.sum,
            torch.argmax,
            torch.argmin,
            torch.min,
            torch.max,
        ],
    )
    @parametrize("input_dtype", [torch.float16, torch.bfloat16])
    @config.patch("triton.use_block_ptr", True)
    def test_low_precision_reduction(self, input_shape, reduction_func, input_dtype):
        # [FUNCTION DOC] Parametrized over input shape, reduction function (prod, sum, argmax,
        # argmin, min, max) and low precision dtype (float16, bfloat16). With upcasting of loads
        # disabled, compiles a reduction of a product of four tensors and checks that the Triton
        # source still contains a cast to float32 (the reduction itself upcasts) and that
        # compiled and eager results are equal.
        @torch.compile
        def func(a, b, c, d):
            # [FUNCTION DOC] HELPER: Applies the parametrized reduction function to the product
            # of four tensors.
            return reduction_func(a * b * c * d)

        inps = (torch.rand(input_shape, device=GPU_TYPE, dtype=input_dtype),) * 4
        with config.patch("triton.codegen_upcast_to_fp32", False):
            func_opt = torch._dynamo.optimize("inductor")(func)
            code = run_and_get_triton_code(func_opt, *inps)
            self.assertTrue(".to(tl.float32)" in code)
            self.assertEqual(func(*inps), func_opt(*inps))

    def test_op_dtype_support(self):
        # [FUNCTION DOC] Checks the table of supported dtypes of Triton ops: for a fixed list of
        # math ops that get upcast to float32, float16 and bfloat16 must not be listed as
        # supported while float32 and float64 must be.
        """
        Triton codegen upcasts values to float32 for certain ops.
        Check that those ops have accurate dtype information.
        """

        for op_name in [
            "rsqrt",
            "sqrt",
            "isnan",
            "floor",
            "ceil",
            "tan",
            "atan",
            "atanh",
            "sigmoid",
            "log2",
            "log10",
            "cosh",
            "sinh",
            "acosh",
            "asinh",
            "asin",
            "acos",
            "asinh",
            "erf",
            "lgamma",
            "sin",
            "cos",
            "exp",
            "expm1",
            "exp2",
            "abs",
            "hypot",
            "nextafter",
        ]:
            # These ops do not support float16 and bfloat16.
            supported_dtypes = OpDtypeSupport.supported_dtypes[op_name]
            self.assertNotIn(torch.float16, supported_dtypes)
            self.assertNotIn(torch.bfloat16, supported_dtypes)

            # These ops should support float32 and float64.
            self.assertIn(torch.float32, supported_dtypes)
            self.assertIn(torch.float64, supported_dtypes)

    @requires_gpu()
    @parametrize("op_name", OpDtypeSupport.supported_dtypes)
    @parametrize("load_upcast_to_fp32", [False, True])
    @parametrize("input_dtype", [torch.float16, torch.bfloat16])
    @config.patch("triton.use_block_ptr", True)
    def test_dtype_aware_codegen(self, op_name: str, load_upcast_to_fp32, input_dtype):
        # [FUNCTION DOC] Parametrized over op name (all entries of the supported dtype table),
        # whether loads are upcast, and half precision input dtype (float16, bfloat16). Requires
        # a GPU. Looks up the torch function for the op name (stripping a libdevice prefix),
        # maps it to the name used in Triton source (round becomes nearbyint, sqrt becomes
        # sqrt_rn, with a device-specific override table), and derives the number of required
        # args as the smallest parameter count among its signatures. Compiles the op over random
        # inputs and inspects the generated source: for atan on non-ROCm CUDA it expects the
        # approximate reciprocal instruction, a separate float32 upcast statement iff loads are
        # not upcast, and a downcast when outputs are converted. For other ops a regex checks
        # that the call has an inline float32 cast of its operand (followed by a downcast if
        # outputs convert) exactly when loads are not already upcast.
        """
        Test dtype aware codegen for some tl.math/libdevice calls.
        Operands should be upcast to float32, and the output should be downcast to float16.
        """

        # Check if the op's output should be upcasted/downcasted.
        supported_dtypes = OpDtypeSupport.supported_dtypes[op_name]
        convert_output = OpDtypeSupport.convert_outputs[op_name]
        self.assertNotIn(input_dtype, supported_dtypes)

        # Retrieve the corresponding torch op.
        torch_op_name = op_name.removeprefix("libdevice_")
        op = getattr(torch, torch_op_name)

        # Edge case: torch.round maps to libdevice.nearbyint.
        triton_op_name_overrides = {
            "default": {
                "round": "nearbyint",
                # torch.sqrt lowers to tl.sqrt_rn after switching away from libdevice.sqrt
                "sqrt": "sqrt_rn",
            },
        }
        if GPU_TYPE in triton_op_name_overrides:
            override = triton_op_name_overrides[GPU_TYPE].get(op_name)
        else:
            override = triton_op_name_overrides["default"].get(op_name)
        triton_op_name = override if override is not None else torch_op_name

        # Get the number of args for the op.
        # Take the minimum over all signatures to isolate required args.
        signatures = get_signature_for_torch_op(op)
        num_args = min(len(signature.parameters) for signature in signatures)

        # Test codegen and check for casts.
        inps = (torch.rand((32, 32), device=GPU_TYPE, dtype=input_dtype),) * num_args
        tl_dtype_str = str(input_dtype).replace("torch", "tl")
        with config.patch("triton.codegen_upcast_to_fp32", load_upcast_to_fp32):
            compiled = torch.compile(op, backend="inductor")
            code = run_and_get_triton_code(compiled, *inps)

            if op_name == "atan" and GPU_TYPE == "cuda" and not torch.version.hip:
                self.assertIn("rcp.approx.ftz.f32", code)
                separate_upcast = (
                    re.search(r"tmp\d+ = tmp\d+\.to\(tl\.float32\)", code) is not None
                )
                self.assertNotEqual(separate_upcast, load_upcast_to_fp32)
                if convert_output:
                    self.assertIn(f".to({tl_dtype_str})", code)
                return

            # Search the code with a regex.
            # Example code: libdevice.floor(tmp3.to(tl.float32)).to(tl.float16)
            output_cast = rf"\.to\({tl_dtype_str}\)" if convert_output else ""
            pattern = rf"{triton_op_name}\(.*\.to\(tl\.float32\)\){output_cast}"
            cast_in_code = re.search(pattern, code, re.MULTILINE) is not None
            self.assertNotEqual(cast_in_code, load_upcast_to_fp32)

    @config.patch("triton.codegen_upcast_to_fp32", False)
    def test_binary_math_mixed_precision(self):
        # [FUNCTION DOC] With load upcasting disabled, compiles a hypot of a float16 and a
        # float32 tensor and checks results are close to eager, that the source has exactly one
        # cast to float32 (only the half precision operand) and no cast to float16.
        """
        Test a binary math operator where only one input needs to be upcast.
        """
        # Create inputs of different dtypes.
        inputs = [
            torch.randn(8, device=GPU_TYPE, dtype=dtype)
            for dtype in (torch.float16, torch.float32)
        ]

        func = torch.hypot
        compiled = torch.compile(backend="inductor")(func)
        result, (code,) = run_and_get_code(compiled, *inputs)

        # Check accuracy.
        ref = func(*inputs)
        self.assertTrue(torch.allclose(ref, result))

        # Check for exactly one upcast.
        num_upcasts = code.count(".to(tl.float32)")
        self.assertEqual(num_upcasts, 1)

        # There should be no downcast, since the input is promoted to float32.
        self.assertNotIn(".to(tl.float16)", code)

    @config.patch("test_configs.static_cpp_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    @config.patch("triton.codegen_upcast_to_fp32", False)
    def test_downcast_div_mod(self):
        # [FUNCTION DOC] With dtype/shape assertions on and upcasting disabled, compiles a
        # function returning both remainder and true division of two float16 tensors. Checks the
        # generated source contains a static assert on dtype and the compiled outputs equal
        # eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the remainder and the quotient of two tensors.
            return x % y, x / y

        x, y = (torch.rand([8], dtype=torch.float16, device=GPU_TYPE) for _ in range(2))

        out, code = run_and_get_code(torch.compile(fn), x, y)

        FileCheck().check("static_assert").check_same(".dtype").run(code[0])
        self.assertEqual(fn(x, y), out)

    @config.patch("test_configs.static_cpp_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    def test_constant(self):
        # [FUNCTION DOC] Compiles a function with no inputs that creates a float16 constant-
        # filled tensor and checks the source has a static dtype assert and the output equals
        # eager.
        def fn():
            # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a float16 tensor of
            # shape two by three filled with a constant, on the GPU.
            return (torch.full((2, 3), 3.1416, device=GPU_TYPE, dtype=torch.float16),)

        out, code = run_and_get_code(torch.compile(fn))
        FileCheck().check("static_assert").check_same(".dtype").run(code[0])
        self.assertEqual(fn(), out)

    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    @config.patch("test_configs.static_cpp_dtype_assert", True)
    @config.patch("triton.persistent_reductions", False)
    def test_any(self):
        # [FUNCTION DOC] With persistent reductions disabled and dtype/shape assertions enabled,
        # compiles an any-reduction over a boolean tensor of 40 elements and checks the result
        # equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Reduces the input with logical any.
            return torch.any(x)

        x = torch.rand([40], device=GPU_TYPE).to(torch.bool)
        out, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(fn(x), out)

    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    @config.patch("test_configs.static_cpp_dtype_assert", True)
    def test_assoc_scan(self):
        # [FUNCTION DOC] With dtype/shape assertions enabled, calls the associative scan higher-
        # order op directly (pointwise combine mode, last dimension) with an add-abs combine
        # function on a random tensor. Only checks that tracing/execution with the dtype checks
        # does not raise; no result check.
        from torch._higher_order_ops.associative_scan import associative_scan

        x = torch.randn(10, device=GPU_TYPE)
        # dtype check correctly
        associative_scan(
            lambda acc, curr: acc + torch.abs(curr), x, dim=-1, combine_mode="pointwise"
        )

    @parametrize("upcast_to_fp32", (False, True))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_upcast_rank_0_cpu(self, dtype: torch.dtype, upcast_to_fp32: bool):
        # [FUNCTION DOC] Parametrized over dtype (float16, bfloat16) and upcast setting. Adds a
        # rank-0 CPU tensor to a rank-1 GPU tensor of the same dtype and checks the result is
        # close to eager. Since Inductor upcasts rank-0 CPU args to float32, the source must
        # contain a downcast to the original dtype zero times when upcasting is on and exactly
        # once when it is off.
        """
        Test whether we implicitly upcast CPU tensors of rank 0 to float32.
        """

        # Test broadcasting a rank-0 CPU tensor to rank 1.
        x = torch.randn(1, dtype=dtype, device="cpu")[0]
        y = torch.randn(8, dtype=dtype, device=GPU_TYPE)
        self.assertEqual(len(x.shape), 0)
        self.assertEqual(len(y.shape), 1)
        inps = (x, y)
        func = torch.add

        with config.patch("triton.codegen_upcast_to_fp32", upcast_to_fp32):
            compiled = torch.compile(func)
            result, (code,) = run_and_get_code(compiled, *inps)

        # Check numerics.
        ref = func(*inps)
        self.assertTrue(torch.allclose(result, ref))

        # Inductor upcasts CPU arguments of rank 0 to float32. Check for a downcast to
        # the original dtype.
        num_downcasts = code.count(f".to({triton_type(dtype)})")
        self.assertEqual(num_downcasts, 0 if upcast_to_fp32 else 1)

    @requires_gpu()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._inductor.config.patch("_use_fp64_for_unbacked_floats", True)
    def test_unbacked_float_uses_fp64_signature(self):
        # [FUNCTION DOC] Requires a GPU, scalar output capture and the option using float64 for
        # unbacked floats. Compiles (full graph) a foreach multiply of a float32 list by a
        # scalar read from a float64 tensor, and checks the Triton source declares that scalar
        # kernel argument with a 64-bit float in the signature and casts it down to float32.
        """
        Test that unbacked float scalars from .item() use fp64 in kernel signature
        and are cast down to fp32 when used with fp32 tensors.

        This verifies the fix for precision loss when Python floats or unbacked
        float symbols were hardcoded to fp32 in Triton kernel signatures.
        """

        def fn(inputs, scalar_tensor):
            # [FUNCTION DOC] HELPER: Reads a Python float from the scalar tensor (an unbacked
            # symbol) and multiplies each tensor of the input list by it with the foreach op.
            # .item() creates an unbacked float symbol that becomes a kernel arg
            val = scalar_tensor.item()
            return torch._foreach_mul(inputs, val)

        inputs = [torch.randn(8, device=GPU_TYPE, dtype=torch.float32)]
        scalar = torch.tensor(
            0.333333333333333333, device=GPU_TYPE, dtype=torch.float64
        )

        compiled = torch.compile(fn, fullgraph=True)
        code = run_and_get_triton_code(compiled, inputs, scalar)

        # The unbacked float should be passed as fp64 ('ks0': 'fp64' in signature)
        # and cast down to fp32 for use with the fp32 tensor
        self.assertIn("'ks0': 'fp64'", code)
        self.assertIn(".to(tl.float32)", code)

    @requires_gpu()
    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    def test_index_expr_ks_arg_dtype_int(self):
        # [FUNCTION DOC] Requires a GPU, dtype/shape assertions. With dynamic shapes, adds two
        # int32 tensors with a scalar alpha. Checks the result equals eager, that the symbolic
        # kernel argument is typed as 64-bit integer in the signature, and that the source casts
        # back to int32 (index dtype).
        """
        ks* kernel args are always int64 in the Triton signature (per
        _decide_tl_dtype), even when the kernel's index_dtype is int32.
        Verify that index_expr emits an explicit final cast to index_dtype.
        """

        def fn(a, b, alpha):
            # [FUNCTION DOC] HELPER: Adds the two tensors with the given alpha multiplier.
            return torch.add(a, b, alpha=alpha)

        a = torch.randint(0, 10, (5, 5), device=GPU_TYPE, dtype=torch.int32)
        b = torch.randint(0, 10, (5, 5), device=GPU_TYPE, dtype=torch.int32)

        compiled = torch.compile(fn, dynamic=True)
        result, codes = run_and_get_code(compiled, a, b, 2)
        code = "\n".join(codes)
        self.assertEqual(result, torch.add(a, b, alpha=2))
        self.assertIn("'ks0': 'i64'", code)
        self.assertIn(".to(tl.int32)", code)

    @requires_gpu()
    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    def test_index_expr_ks_arg_dtype_float(self):
        # [FUNCTION DOC] Requires a GPU, dtype/shape assertions. Compiles dynamically a batched
        # matmul divided by a ratio computed from the square root of a symbolic size, and checks
        # the result is close to eager and that the source casts to float32.
        """
        When a symbolic scalar is used in a float context (e.g. math.sqrt of
        a tensor size), index_expr emits an explicit final cast to the
        requested float dtype.
        """
        import math

        @torch.compile(dynamic=True)
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Computes the reciprocal square root-style scalar from the
            # size of the second dimension of the first input, then returns the batched matmul
            # divided by that scalar.
            r = 1 / math.sqrt(a.size(1))
            return torch.bmm(a, b) / r

        a = torch.randn(2, 4, 4, device=GPU_TYPE)
        b = torch.randn(2, 4, 4, device=GPU_TYPE)
        result, codes = run_and_get_code(fn, a, b)
        code = "\n".join(codes)
        expected = torch.bmm(a, b) / (1 / math.sqrt(a.size(1)))
        self.assertTrue(torch.allclose(result, expected))
        self.assertIn(".to(tl.float32)", code)

    @requires_gpu()
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    @config.patch("test_configs.runtime_triton_shape_assert", True)
    def test_randint_symbolic_bounds_use_value_expr(self):
        # [FUNCTION DOC] Requires a GPU, scalar output capture, dtype/shape assertions. Draws
        # random int64 values in a range from two to the 32nd up to a symbolic upper bound read
        # from a tensor holding two to the 40th. Checks dtype is int64, all values lie in the
        # range, and the source uses the 64-bit random integer helper and casts to int64.
        @torch.compile(fullgraph=True)
        def fn(high):
            # [FUNCTION DOC] HELPER: Random integer tensor of shape five by five between a fixed
            # lower bound and the scalar upper bound taken from the input tensor.
            return torch.randint(2**32, high.item(), (5, 5), device=GPU_TYPE)

        high = torch.tensor(2**40, device=GPU_TYPE, dtype=torch.int64)
        result, codes = run_and_get_code(fn, high)
        code = "\n".join(codes)
        self.assertEqual(result.dtype, torch.int64)
        self.assertGreaterEqual(result.min().item(), 2**32)
        self.assertLess(result.max().item(), 2**40)
        self.assertIn("triton_helpers.randint64", code)
        self.assertIn(".to(tl.int64)", code)

    @requires_gpu()
    @parametrize("dynamic", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_int64_symint_not_downcast_to_int32_index_expr(self, dynamic):
        # [FUNCTION DOC] Parametrized over dynamic mode; requires a GPU and scalar output
        # capture. Compiles (full graph) random int64 generation whose upper bound is an
        # unbacked value known to exceed two to the 31st. Checks all samples are within range,
        # the symbolic argument is typed as 64-bit integer, and the source uses the 64-bit
        # random helper with an int64 cast, i.e. the index expression is not downcast to int32.
        def fn(x, hi):
            # [FUNCTION DOC] HELPER: Reads an unbacked size from the tensor, asserts it is a
            # valid size larger than two to the 31st, and returns random int64 values below it
            # with length of the first input dimension.
            h = hi.item()
            torch._check_is_size(h)
            torch._check(h > 2**31)
            return torch.randint(
                0, h, (x.shape[0],), device=x.device, dtype=torch.int64
            )

        x = torch.zeros(4, device=GPU_TYPE)
        hi = torch.tensor(2**31 + 1234, device=GPU_TYPE, dtype=torch.int64)
        cfn = torch.compile(fn, dynamic=dynamic, fullgraph=True)
        got, codes = run_and_get_code(cfn, x, hi)
        code = "\n".join(codes)
        self.assertTrue(bool((got >= 0).all()))
        self.assertTrue(bool((got < hi.item()).all()))
        self.assertIn("'ks1': 'i64'", code)
        self.assertIn("triton_helpers.randint64", code)
        self.assertIn(".to(tl.int64)", code)


instantiate_device_type_tests(
    TestCase, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
