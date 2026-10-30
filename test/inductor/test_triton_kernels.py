# Owner(s): ["module: inductor"]
# ruff: noqa: F841
# flake8: noqa: E731
# Skip do not assign a lambda expression, use a def
# [MODULE DOC] Tests for user-defined Triton kernels under torch.compile: dynamo tracing, the
# Triton kernel higher-order ops, inductor codegen/caching, autotuning, TMA descriptors, mutation
# analysis, and launch options. Kernels are defined at module level inside a GPU/Triton-available
# guard; the file also defines helper kernels, a launch-params dump context manager, and the
# KernelTests class with its device variants.
# Import side effects: when a GPU is available it imports triton and tl, picks the libdevice
# fast_dividef (CUDA or XPU), and defines module-level constexpr constants and shared kernels. It
# star-imports the shared kernels from the internal triton_utils.
import contextlib
import functools
import logging
import os
import subprocess
import sys
import unittest
from unittest import mock

import torch
import torch._dynamo.testing
import torch._inductor.test_case
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch._dynamo import config as dynamo_config
from torch._higher_order_ops.triton_kernel_wrap import (
    generate_ttir,
    triton_kernel_wrapper_functional,
    triton_kernel_wrapper_mutation,
)
from torch._inductor import config as inductor_config, metrics
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._inductor.utils import (
    fresh_cache,
    run_and_get_code,
    triton_version_uses_attrs_dict,
)
from torch._library import capture_triton
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    parametrize,
    skipIfRocm,
    skipIfWindows,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import (
    get_func_call,
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
    HAS_XPU_AND_TRITON,
    TRITON_HAS_CPU,
)
from torch.testing._internal.logging_utils import log_settings, logs_to_string

# Defines all the kernels for tests
from torch.testing._internal.triton_utils import *  # noqa: F403
from torch.utils._triton import (
    has_triton_experimental_host_tma,
    has_triton_package,
    has_triton_tensor_descriptor_host_tma,
)


@contextlib.contextmanager
def _dump_launch_params(value: str):
    # [FUNCTION DOC] HELPER: Context manager that sets the dump-launch-params inductor env var to
    # the given string for its duration. When the value is 1 it removes any stale launch params
    # file (named after the running script) before and after the body. The env var is patched via
    # a mock that restores it on exit; the file cleanup runs in a finally block.
    launch_params_file = f"{os.path.abspath(sys.argv[0])}.launch_params"
    if value == "1" and os.path.exists(launch_params_file):
        os.remove(launch_params_file)
    with mock.patch.dict(os.environ, {"TORCHINDUCTOR_DUMP_LAUNCH_PARAMS": value}):
        try:
            yield
        finally:
            if value == "1" and os.path.exists(launch_params_file):
                os.remove(launch_params_file)


if HAS_GPU:
    import triton
    from triton import language as tl

    if HAS_CUDA_AND_TRITON:
        try:
            from triton.language.extra.libdevice import (  # @manual
                fast_dividef,
                fast_dividef as my_fast_dividef,
            )
        except ImportError:
            from triton.language.extra.cuda.libdevice import (  # @manual
                fast_dividef,
                fast_dividef as my_fast_dividef,
            )
    elif HAS_XPU_AND_TRITON:
        from triton.language.extra.intel.libdevice import (  # @manual
            fast_dividef,
            fast_dividef as my_fast_dividef,
        )

    def _triton_get_ast_equal_to_str(params):
        # [FUNCTION DOC] HELPER: Returns the textual form of the "equal to 1" specialization
        # attribute for a kernel signature. If triton has the AttrsDescriptor class (newer
        # triton) it returns a text using the attrs-dict key with the given params; otherwise
        # (ImportError) it returns the older key spelling with the params.
        try:
            from triton.backends.compiler import AttrsDescriptor  # noqa: F401

            return f"'tt.equal_to': {params}"
        except ImportError:
            return f"equal_to_1={params}"

    # Define shared triton constants here.
    CONSTANT_C: tl.constexpr = tl.constexpr(4)
    STRING_CONSTANT_C: tl.constexpr = tl.constexpr("CONSTANT_C")
    BOOL_CONSTANT_C: tl.constexpr = tl.constexpr(True)
    FLOAT_CONSTANT_C = tl.constexpr(3.14)  # intentionally un-annotated
    # Compute at module level to avoid dynamo tracing issue with
    # torch.backends.cuda.matmul.fp32_precision getter
    USE_TF32 = torch.backends.cuda.matmul.fp32_precision == "tf32"

    if hasattr(triton, "constexpr_function"):
        # Helper functions for triton kernels must be in globals.
        @triton.constexpr_function
        def log2(n):
            # [FUNCTION DOC] HELPER: Triton constexpr function computing floor(log2) of a
            # positive integer constant as the length of its binary string minus three. Defined
            # at module level because helpers called from kernels must live in globals. Only
            # defined if triton provides constexpr_function.
            return len(bin(n)) - 3

        _get_int_dtype_test = triton.constexpr_function(tl.core.get_int_dtype)

        # [FUNCTION DOC] HELPER: Triton jit helper used by tests; reinterprets (bitcasts) a
        # tensor to a signed integer dtype of the same bit width, with the dtype obtained from
        # the constexpr-wrapped int-dtype lookup.
        @triton.jit
        def _dtype_helper_test(x):
            idtype = _get_int_dtype_test(
                bitwidth=x.dtype.primitive_bitwidth, signed=True
            )
            return x.to(idtype, bitcast=True)

    # Kernel with dunder name for name-mangling regression test (issue #170398).
    # [FUNCTION DOC] HELPER: Triton kernel elementwise-adding two input buffers into the output,
    # masked for indices at or beyond n_elements. Its double-underscore name is deliberate: it
    # checks that generated code does not hit class-based name mangling (issue 170398).
    @triton.jit
    def __dunder_add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    def _get_backend_options_kernel(*, with_enable_fp_fusion):
        # [FUNCTION DOC] HELPER: Factory returning a Triton add kernel. With the keyword-only
        # flag true, the kernel takes an extra constexpr named enable_fp_fusion and computes a
        # sum or a product depending on it (only to make the kwarg binding observable); otherwise
        # it takes no such parameter and always adds.
        # These kernels are intentionally illustrative. `enable_fp_fusion` is a
        # real backend option name, but the branch below only makes the kwarg's
        # kernel-argument binding observable; it is not modeling the compiler
        # option's normal optimization behavior.
        if with_enable_fp_fusion:

            # [FUNCTION DOC] HELPER: Variant of the add kernel with an extra constexpr
            # enable_fp_fusion; adds the inputs when it is true and multiplies them when false.
            # Masked by n_elements.
            @triton.jit
            def add_kernel(
                in_ptr0,
                in_ptr1,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
                enable_fp_fusion: "tl.constexpr",
            ):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(in_ptr0 + offsets, mask=mask)
                y = tl.load(in_ptr1 + offsets, mask=mask)
                if enable_fp_fusion:
                    result = x + y
                else:
                    result = x * y
                tl.store(out_ptr + offsets, result, mask=mask)

        else:

            # [FUNCTION DOC] HELPER: Plain masked elementwise add
            # kernel with no enable_fp_fusion parameter.
            @triton.jit
            def add_kernel(
                in_ptr0,
                in_ptr1,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(in_ptr0 + offsets, mask=mask)
                y = tl.load(in_ptr1 + offsets, mask=mask)
                tl.store(out_ptr + offsets, x + y, mask=mask)

        return add_kernel

    def _get_backend_options_fn(add_kernel, **launch_kwargs):
        # [FUNCTION DOC] HELPER: Builds a function that runs a given kernel with extra launch
        # keyword arguments (stored in the closure) and returns the output.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Allocates a zero output like the first input, launches the
            # kernel over a one-dimensional grid sized by cdiv of the element count and the block
            # size meta parameter, forwards the stored launch kwargs, and returns the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](
                x,
                y,
                output,
                n_elements,
                **launch_kwargs,
            )
            return output

        return f


class KernelTests(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Test case for user-defined Triton kernels with inductor and dynamo. Most tests
    # are gated on a GPU with Triton; they compare compiled results to eager, inspect generated
    # code, or exercise the kernel higher-order ops. Contains small helpers to check whether a
    # kernel is launched in generated code and to capture the options passed to triton.compile.
    def _kernel_launched_in_code(self, kernel_name: str, code: str) -> bool:
        # [FUNCTION DOC] HELPER: Returns whether the generated code text contains a launch of the
        # named kernel. For the cpp wrapper it looks for the launchKernel call prefix with the
        # kernel name; otherwise for the kernel name followed by the run call.
        if inductor_config.cpp_wrapper:
            return f"launchKernel({kernel_name}" in code
        return f"{kernel_name}.run(" in code

    def _run_and_get_triton_compile_options(self, fn, *args):
        # [FUNCTION DOC] HELPER: Compiles fn fullgraph with torch.compile, running it with the
        # given args while mocking triton.compile (wrapping the real one) and forcing a single
        # compile thread. Returns the output, the generated code list, and a list of dicts with
        # the options keyword passed to every triton.compile call that had one. Asserts at least
        # one options dict was captured.
        # TestCase already gives each test a fresh compile cache. This patch
        # only keeps compilation in-process while the triton.compile mock is
        # installed, so we can assert on the actual options passed to Triton.
        with (
            inductor_config.patch("compile_threads", 1),
            mock.patch.object(triton, "compile", wraps=triton.compile) as compile_mock,
        ):
            out, codes = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        compile_options = [
            dict(call.kwargs["options"])
            for call in compile_mock.call_args_list
            if "options" in call.kwargs
        ]
        self.assertTrue(compile_options)
        return out, codes, compile_options

    def _assert_triton_compile_option(self, options, name, expected_value):
        # [FUNCTION DOC] HELPER: Asserts that at least one captured compile-options dict has the
        # given option name equal to the expected value, with a message showing all options.
        self.assertTrue(
            any(option.get(name) == expected_value for option in options),
            f"expected triton.compile options to include "
            f"{name}={expected_value!r}, got {options!r}",
        )

    @requires_gpu
    def test_triton_kernel_with_kernel_param(self):
        # [FUNCTION DOC] Checks dynamo tolerates a Triton kernel whose parameter is itself named
        # kernel and is passed a tensor, under the eager backend. The test only requires that
        # compilation and the call do not crash.
        # [FUNCTION DOC] HELPER: Empty Triton kernel with a single parameter named kernel.
        @triton.jit
        def pass_kernel(kernel):
            pass

        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled (eager backend) function launching the empty kernel
            # over a grid with one program per element, passing x as the kernel parameter.
            grid = (x.numel(),)
            pass_kernel[grid](kernel=x)

        t1 = torch.rand(5, device=GPU_TYPE)
        f(t1)
        # No need to assert anything, the goal is to make sure dynamo does
        # not crash

    @requires_gpu
    def test_triton_kernel_dunder_name_no_name_mangling(self):
        # [FUNCTION DOC] Regression test for issue 170398: a Triton kernel whose name starts with
        # double underscores (fetched from module globals to dodge class name mangling) must not
        # be rewritten by Python name mangling in generated code. Runs eager and compiled,
        # compares results, then checks the generated code does not contain the double-underscore
        # name with the call suffix but does contain the single-underscore form.
        # Regression test for https://github.com/pytorch/pytorch/issues/170398
        # Triton kernels whose names start with ``__`` must not trigger
        # Python class-based name mangling in the generated code.
        # Use globals() to reference the dunder-named kernel without triggering
        # name mangling inside this class body.
        kernel = globals()["__dunder_add_kernel"]

        def f(x, y):
            # [FUNCTION DOC] HELPER: Allocates an empty output, launches the dunder-named kernel
            # with one program per element and block size 16, and returns the output.
            out = torch.empty_like(x)
            n_elements = x.numel()
            kernel[(n_elements,)](x, y, out, n_elements, BLOCK_SIZE=16)
            return out

        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        eager_out = f(x, y)
        compiled_out, (code,) = run_and_get_code(torch.compile(f), x, y)
        self.assertEqual(compiled_out, eager_out)
        # Match as a standalone identifier; substrings inside other names
        # (e.g. cpp_wrapper's `call__dunder_add_kernel_0`) are harmless C++
        # identifiers and aren't subject to Python name mangling.
        import re as _re

        self.assertIsNone(_re.search(r"\b__dunder_add_kernel_0\b", code))
        self.assertIsNotNone(_re.search(r"\b_dunder_add_kernel_0\b", code))

    @inductor_config.patch(strict_signed_zero=True)
    @requires_cuda_and_triton
    def test_dim_max_min_reuse_argreduce_value(self):
        # [FUNCTION DOC] Checks that max/min with indices over the last dim reuses the argreduce
        # value instead of a second reduction. Runs on float32, float16 and bfloat16 (when
        # supported) with dynamic shapes and an extreme value planted in a late column; asserts
        # values and indices match eager, that the generated source uses the with-index helper
        # once and not the value-only helper, and that the value-only form also uses one with-
        # index helper and one store. Then verifies signed zeros are preserved (signbit
        # comparison) for max/min and for separate amax/argmax and amin/argmin pairs. Runs with
        # strict signed zero enabled.
        dtypes = [torch.float32, torch.float16]
        if torch.cuda.is_bf16_supported(including_emulation=False):
            dtypes.append(torch.bfloat16)

        for op, indexed_helper, value_helper, selected_value in (
            (torch.max, "max_with_index", "max2", 100.0),
            (torch.min, "min_with_index", "min2", -100.0),
        ):
            torch._dynamo.reset()

            def fn(x):
                # [FUNCTION DOC] HELPER: Applies the current reduction op (max or min from the
                # loop closure) over the last dim and returns its result.
                return op(x, -1)

            for dtype in dtypes:
                x = torch.randn(8, 4097, dtype=dtype, device=GPU_TYPE)
                x[:, 4095] = selected_value
                expected_values, expected_indices = fn(x)
                with fresh_cache():
                    actual, codes = run_and_get_code(torch.compile(fn, dynamic=True), x)
                actual_values, actual_indices = actual

                self.assertEqual(actual_values, expected_values)
                self.assertEqual(actual_indices, expected_indices)

                source = "\n".join(codes)
                self.assertEqual(source.count(f"triton_helpers.{indexed_helper}"), 1)
                self.assertNotIn(f"triton_helpers.{value_helper}", source)

                torch._dynamo.reset()

                def value_only(x):
                    # [FUNCTION DOC] HELPER: Returns only the values of the reduction op over the
                    # last dim.
                    return op(x, -1).values

                with fresh_cache():
                    actual_values, value_codes = run_and_get_code(
                        torch.compile(value_only, dynamic=True), x
                    )

                self.assertEqual(actual_values, expected_values)

                value_source = "\n".join(value_codes)
                self.assertEqual(
                    value_source.count(f"triton_helpers.{indexed_helper}"), 1
                )
                self.assertNotIn(f"triton_helpers.{value_helper}", value_source)
                self.assertEqual(value_source.count("tl.store("), 1)

        x = torch.tensor([[0.0, -0.0], [-0.0, 0.0]], device=GPU_TYPE)
        for op in (torch.max, torch.min):
            torch._dynamo.reset()

            def fn(x):
                # [FUNCTION DOC] HELPER: Applies the loop's reduction op (max or min) over the
                # last dim and returns its result.
                return op(x, -1)

            expected_values, expected_indices = fn(x)
            with fresh_cache():
                actual_values, actual_indices = torch.compile(fn, dynamic=True)(x)

            self.assertEqual(
                torch.signbit(actual_values), torch.signbit(expected_values)
            )
            self.assertEqual(actual_indices, expected_indices)

            torch._dynamo.reset()

            def value_only(x):
                # [FUNCTION DOC] HELPER: Returns only the values of the reduction op over the
                # last dim.
                return op(x, -1).values

            expected_values = value_only(x)
            with fresh_cache():
                actual_values = torch.compile(value_only, dynamic=True)(x)

            self.assertEqual(
                torch.signbit(actual_values), torch.signbit(expected_values)
            )

        for value_op, index_op in (
            (torch.amax, torch.argmax),
            (torch.amin, torch.argmin),
        ):
            torch._dynamo.reset()

            def separate_value_and_index(x):
                # [FUNCTION DOC] HELPER: Returns a pair: the value reduction (amax or amin) and
                # the index reduction (argmax or argmin) of the input over the last dim.
                return value_op(x, -1), index_op(x, -1)

            expected_values, expected_indices = separate_value_and_index(x)
            with fresh_cache():
                actual_values, actual_indices = torch.compile(
                    separate_value_and_index, dynamic=True
                )(x)

            self.assertEqual(
                torch.signbit(actual_values), torch.signbit(expected_values)
            )
            self.assertEqual(actual_indices, expected_indices)

    @requires_gpu
    def test_triton_kernel_ill_formed(self):
        # [FUNCTION DOC] Checks that a compile error inside a user Triton kernel surfaces through
        # torch.compile with the inductor backend. Skipped under the cpp wrapper (errors occur
        # lazily at runtime). Passes if the raised InductorError wraps a Triton CompilationError
        # directly or a subprocess pool exception whose details mention it; a finally block
        # asserts this flag.
        if inductor_config.cpp_wrapper:
            # With lazy compile, kernel errors happen at runtime, not codegen
            self.skipTest("kernel errors happen at runtime instead of codegen time")

        # [FUNCTION DOC] HELPER: Deliberately ill-formed Triton kernel; one branch assigns an
        # int32 value and the other an int64 value to the same variable, and then stores it,
        # which makes compilation fail.
        @triton.jit
        def ill_formed_kernel(kernel):
            off_n = tl.program_id(0)
            seq_len_a = 5
            if off_n < seq_len_a:
                out_row_idx = off_n + seq_len_a
            else:
                out_row_idx = (off_n + seq_len_a).to(tl.int64)
            tl.store(kernel, out_row_idx)

        @torch.compile(backend="inductor")
        def f(x):
            # [FUNCTION DOC] HELPER: Inductor-compiled function launching the ill-formed kernel
            # over one program per element with x as the kernel argument.
            grid = (x.numel(),)
            ill_formed_kernel[grid](kernel=x)

        catch_error = False
        try:
            t1 = torch.rand(5, device=GPU_TYPE)
            f(t1)
        except torch._inductor.exc.InductorError as e:
            if isinstance(e.inner_exception, triton.compiler.errors.CompilationError):
                catch_error = True
            if (
                isinstance(
                    e.inner_exception,
                    torch._inductor.compile_worker.subproc_pool.SubprocException,
                )
                and "triton.compiler.errors.CompilationError"
                in e.inner_exception.details
            ):
                catch_error = True
        finally:
            # we assert user custom Triton kernel compile error can be captured
            # after torch.compile
            self.assertTrue(catch_error)

    @requires_gpu
    def test_triton_kernel_higher_order_func(self):
        # [FUNCTION DOC] Exercises the Triton kernel higher-order ops directly. Registers the add
        # kernel and its constant args in the kernel side table, then calls the mutation wrapper
        # with a grid callable and checks the output equals the sum and was changed. Then calls
        # the functional wrapper with tensors to clone and checks the returned dict holds the sum
        # while the passed output stays zeros.
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        add_kernel_id = kernel_side_table.add_kernel(add_kernel)

        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)

        torch_add = t1 + t2

        # Test higher order function with mutation
        output = torch.zeros_like(t1)
        n_elements = output.numel()
        constant_args_idx = kernel_side_table.add_constant_args(
            {"n_elements": n_elements, "BLOCK_SIZE": 16}
        )
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        triton_kernel_wrapper_mutation(
            kernel_idx=add_kernel_id,
            constant_args_idx=constant_args_idx,
            grid=[grid],
            tma_descriptor_metadata={},
            kwargs={
                "in_ptr0": t1,
                "in_ptr1": t2,
                "out_ptr": output,
            },
        )
        self.assertEqual(output, torch_add)
        # Make sure it is modified
        self.assertNotEqual(output, torch.zeros_like(t1))

        # Test higher order function without mutation
        output = torch.zeros_like(t1)
        out_dict = triton_kernel_wrapper_functional(
            kernel_idx=add_kernel_id,
            constant_args_idx=constant_args_idx,
            grid=[grid],
            tma_descriptor_metadata={},
            kwargs={
                "in_ptr0": t1,
                "in_ptr1": t2,
                "out_ptr": output,
            },
            tensors_to_clone=["in_ptr0", "in_ptr1", "out_ptr"],
        )
        self.assertEqual(out_dict["out_ptr"], torch_add)
        # Make sure it is NOT modified
        self.assertEqual(output, torch.zeros_like(t1))

    @requires_gpu
    def test_triton_kernel_functionalize(self):
        # [FUNCTION DOC] Checks functionalization of the functional Triton kernel wrapper using
        # make_fx with the Python functionalize API under FunctionalTensorMode, the C++ API, and
        # torch.func.functionalize; each time the passed output tensor must not be modified. Then
        # traces in fake mode and compares the printed graph to an expected inline string (which
        # hard-codes kernel and constant-arg side table indices after a table reset).
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch._subclasses.functional_tensor import (
            CppFunctionalizeAPI,
            FunctionalTensorMode,
            PythonFunctionalizeAPI,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        kernel_side_table.reset_table()

        def f(x, output):
            # [FUNCTION DOC] HELPER: Calls the functional wrapper for the mul2 kernel with
            # tensors to clone for the input and output, a grid with one program per element and
            # block size 16, and returns the cloned out_ptr entry of the result.
            out = triton_kernel_wrapper_functional(
                kernel_idx=kernel_side_table.add_kernel(mul2_kernel),
                constant_args_idx=kernel_side_table.add_constant_args(
                    {"n_elements": output.numel(), "BLOCK_SIZE": 16}
                ),
                grid=[(x.numel(),)],
                tma_descriptor_metadata={},
                kwargs={
                    "in_ptr0": x,
                    "out_ptr": output,
                },
                tensors_to_clone=["in_ptr0", "out_ptr"],
            )
            return out["out_ptr"]

        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)
        with FunctionalTensorMode():
            gm = make_fx(PythonFunctionalizeAPI().functionalize(f))(t1, t2)
        # Make sure t2 was not modified
        self.assertNotEqual(gm(t1, t2), t2)

        gm = make_fx(CppFunctionalizeAPI().functionalize(f))(t1, t2)
        # Make sure t2 was not modified
        self.assertNotEqual(gm(t1, t2), t2)

        gm = make_fx(torch.func.functionalize(f))(t1, t2)
        # Make sure t2 was not modified
        self.assertNotEqual(gm(t1, t2), t2)

        gm = make_fx(f, tracing_mode="fake")(t1, t2)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1, output_1):
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 3, grid = [(5,)], tma_descriptor_metadata = {}, kwargs = {'in_ptr0': x_1, 'out_ptr': output_1}, tensors_to_clone = ['in_ptr0', 'out_ptr']);  x_1 = output_1 = None
    getitem = triton_kernel_wrapper_functional_proxy['in_ptr0'];  getitem = None
    getitem_1 = triton_kernel_wrapper_functional_proxy['out_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return getitem_1""",
        )

    @requires_gpu
    def test_triton_kernel_mutation_type(self):
        # [FUNCTION DOC] Checks the autograd-hidden-mutation flag on functional tensors under
        # fake mode. An ordinary in-place multiply yields not-all-hidden; a Triton kernel
        # mutation alone yields all-hidden; an ordinary multiply followed by the kernel mutation
        # yields not-all-hidden.
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        def prep():
            # [FUNCTION DOC] HELPER: Creates a ones tensor requiring grad, converts it to a
            # functional tensor under FunctionalTensorMode, asserts the wrapped tensor is
            # functional, and returns the functional tensor.
            x = torch.ones(4, device=GPU_TYPE, requires_grad=True)
            with FunctionalTensorMode():
                x_func = FunctionalTensor.to_functional(x)
            self.assertTrue(torch._is_functional_tensor(x_func.elem))
            return x_func

        # normal mutation only
        with FakeTensorMode():
            x_func = prep()

            with FunctionalTensorMode():
                x_func.mul_(2)

            self.assertFalse(
                torch._functionalize_are_all_mutations_hidden_from_autograd(x_func.elem)
            )

        # triton kernel mutation only
        with FakeTensorMode():
            x_func = prep()

            with FunctionalTensorMode():
                triton_kernel_wrapper_mutation(
                    kernel_idx=kernel_side_table.add_kernel(mul2_inplace_kernel),
                    constant_args_idx=kernel_side_table.add_constant_args(
                        {"n_elements": x_func.numel(), "BLOCK_SIZE": 16}
                    ),
                    grid=[(x_func.numel(),)],
                    tma_descriptor_metadata={},
                    kwargs={
                        "ptr": x_func,
                    },
                )

            self.assertTrue(
                torch._functionalize_are_all_mutations_hidden_from_autograd(x_func.elem)
            )

        # normal mutation + triton kernel mutation
        with FakeTensorMode():
            x_func = prep()

            with FunctionalTensorMode():
                x_func.mul_(2)
                triton_kernel_wrapper_mutation(
                    kernel_idx=kernel_side_table.add_kernel(mul2_inplace_kernel),
                    constant_args_idx=kernel_side_table.add_constant_args(
                        {"n_elements": x_func.numel(), "BLOCK_SIZE": 16}
                    ),
                    grid=[(x_func.numel(),)],
                    tma_descriptor_metadata={},
                    kwargs={
                        "ptr": x_func,
                    },
                )

            self.assertFalse(
                torch._functionalize_are_all_mutations_hidden_from_autograd(x_func.elem)
            )

    @requires_gpu
    def test_triton_kernel_clone_wekdeps(self):
        # [FUNCTION DOC] Checks that cloning for a functional Triton kernel wrapper does
        # not break buffer lifetimes (weak deps). Traces a function that calls a dual-
        # output kernel seven times on a large tensor, where both outputs alias the same
        # buffer, then compiles with all fusion disabled and captures the output code log.
        # Uses FileCheck to require the deletion of a buffer before the kernel run, and
        # compares the compiled result with eager.
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch._inductor.choices import InductorChoices
        from torch._inductor.compile_fx import compile_fx_inner
        from torch._inductor.virtualized import V
        from torch.fx.experimental.proxy_tensor import make_fx

        TENSOR_SIZE = 30_000_000
        BLOCK_SIZE = 1024

        # [FUNCTION DOC] HELPER: Triton kernel converting the block of inputs to float32 and
        # storing it to the first output and storing that plus the input to the second output,
        # masked by n_elements.
        @triton.jit
        def dual_output_kernel_with_inline_asm(
            in_ptr0,
            out_ptr0,
            out_ptr1,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            result = x.to(tl.float32)
            tl.store(out_ptr0 + offsets, result, mask=mask)
            tl.store(out_ptr1 + offsets, result + x, mask=mask)

        kernel_side_table.reset_table()

        def f(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Builds a zero-filled large tensor and, in a seven-iteration
            # loop, calls the functional kernel wrapper with both outputs set to the same tensor
            # to clone, accumulating the two returned outputs into a running sum. Returns a one-
            # element tuple holding the sum.
            full_default = torch.full(
                (TENSOR_SIZE,), 0.0, device=x.device, dtype=x.dtype
            )
            running_sum = torch.empty_like(x)
            running_sum.fill_(0.0)

            for _ in range(7):
                n_elements = full_default.numel()
                grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)
                out = triton_kernel_wrapper_functional(
                    kernel_idx=kernel_side_table.add_kernel(
                        dual_output_kernel_with_inline_asm
                    ),
                    constant_args_idx=kernel_side_table.add_constant_args(
                        {"n_elements": n_elements, "BLOCK_SIZE": BLOCK_SIZE}
                    ),
                    grid=[grid],
                    tma_descriptor_metadata={},
                    kwargs={
                        "in_ptr0": x,
                        "out_ptr0": full_default,
                        "out_ptr1": full_default,
                    },
                    tensors_to_clone=["out_ptr0", "out_ptr1"],
                )
                dk_output = out["out_ptr0"]
                dv_output = out["out_ptr1"]
                running_sum = running_sum + dk_output + dv_output

            return (running_sum,)

        t = torch.rand(TENSOR_SIZE, device=GPU_TYPE)

        class NoFusionChoices(InductorChoices):
            # [CLASS DOC] HELPER: Inductor choices handler that forbids every fusion.
            def can_fuse(self, scheduler, node1, node2, shared_data_score) -> bool:
                # [FUNCTION DOC] HELPER: Always returns false so no
                # pair of scheduler nodes is fused.
                return False

        gm = make_fx(f, tracing_mode="fake")(t)

        with V.set_choices_handler(NoFusionChoices()):
            log_stream, ctx = logs_to_string("torch._inductor.codecache", "output_code")
            with ctx():
                compiled_gm = compile_fx_inner(gm, [t])
                compiled_result = compiled_gm([t])[0]

            output_code = log_stream.getvalue()

        FileCheck().check("del buf3").check(
            "dual_output_kernel_with_inline_asm_0.run(x_1, buf0,"
        ).run(output_code)

        eager_result = f(t.clone())[0]
        self.assertEqual(eager_result, compiled_result)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_with_views(self, dynamic, backend):
        # [FUNCTION DOC] Parametrized over dynamic shapes and backends (eager, aot_eager,
        # inductor). Checks that Triton kernels work with view inputs and view outputs: one
        # compiled function takes a flattened view of a 4x4 tensor, another returns its output
        # viewed as 4x4; results must equal twice the input.
        def call_triton_take_view(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Runs the mul2 kernel on a zeroed output like x and returns
            # the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            return output

        def call_triton_return_view(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Runs the mul2 kernel on a zeroed output like x and returns
            # the output reshaped as a 4 by 4 view.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            return output.view(4, 4)

        t = torch.rand(4, 4, device=GPU_TYPE)
        t_view = t.view(16)

        compiled_func = torch.compile(
            call_triton_take_view, backend=backend, fullgraph=True, dynamic=dynamic
        )
        self.assertEqual(2 * t_view, compiled_func(t_view))
        self.assertEqual(2 * t, compiled_func(t_view).view(4, 4))

        compiled_func = torch.compile(
            call_triton_return_view, backend=backend, fullgraph=True, dynamic=dynamic
        )
        self.assertEqual(2 * t_view, compiled_func(t).view(16))
        self.assertEqual(2 * t, compiled_func(t))

    @requires_gpu
    def test_no_nan_kernels(self):
        # [FUNCTION DOC] Checks that the generated code for a custom autograd Function using a
        # Triton kernel contains no NaN literal (cpp or Python form). Runs a compiled forward and
        # extracts the codecache output code log, dropping its first three lines, and requires
        # non-empty output.
        # [FUNCTION DOC] HELPER: Triton kernel adding one to each input element within a masked
        # block and storing to the output.
        @triton.jit
        def add_one_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            output = x + 1
            tl.store(out_ptr + offsets, output, mask=mask)

        def add_one(x, out):
            # [FUNCTION DOC] HELPER: Launches the add-one kernel over one program per element
            # with block size 4.
            n_elements = x.numel()
            add_one_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)

        class AddOne(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function whose forward and backward both apply the
            # add-one kernel to a fresh output; forward saves its output for backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Allocates an empty output, applies add-one, saves the
                # output for backward, and returns it.
                out = torch.empty_like(x)
                add_one(x, out)
                ctx.save_for_backward(out)
                return out

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Retrieves the saved tensor, allocates an empty output
                # shaped like the gradient, applies add-one to the saved tensor, and returns that
                # (the incoming gradient is not used).
                (saved,) = ctx.saved_tensors
                out = torch.empty_like(grad)
                add_one(saved, out)
                return out

        @torch.compile
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function applying the AddOne autograd function.
            return AddOne.apply(x)

        log_stream, ctx = logs_to_string("torch._inductor.codecache", "output_code")

        x = torch.randn(3, requires_grad=True, device=GPU_TYPE)
        with ctx():
            y = f(x)

        output_code = "\n".join(log_stream.getvalue().strip().split("\n")[3:]).strip()
        self.assertTrue(len(output_code) > 0, msg="output code is not empty")
        if inductor_config.cpp_wrapper:
            self.assertEqual(
                output_code.count("std::numeric_limits<double>::quiet_NaN()"), 0
            )
        else:
            self.assertEqual(output_code.count('float("nan")'), 0)
            self.assertEqual(output_code.count("float('nan')"), 0)

    @requires_cuda_and_triton
    def test_fp_self_sub_rewrite_preserves_nonfinite_semantics(self):
        # [FUNCTION DOC] Checks that the inductor rewrite of subtracting a value from itself
        # keeps floating point semantics for non-finite values. For a square-then-self-subtract
        # function on random input it asserts an exact zero result, that the generated code uses
        # a conditional select and an infinity test, and does not disable fp fusion. Then runs on
        # a tensor holding NaN and both infinities plus a finite value and compares to eager with
        # NaN equal.
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares the input and returns the square minus itself.
            a = x * x
            return a - a

        x = torch.randn(1024, device=GPU_TYPE)
        out, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(out, torch.zeros_like(out), atol=0, rtol=0)
        self.assertIn("tl.where", code)
        self.assertIn("libdevice.isinf", code)
        self.assertNotIn("'enable_fp_fusion': False", code)

        special = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 2.0],
            device=GPU_TYPE,
        )
        expected = fn(special)
        actual = torch.compile(fn, fullgraph=True)(special)
        torch.testing.assert_close(actual, expected, equal_nan=True)

    @requires_cuda_and_triton
    @common_utils.parametrize("per_subkernel", [False, True])
    def test_combo_fp_self_sub_rewrite(self, per_subkernel):
        # [FUNCTION DOC] Parametrized on per-subkernel block sizes. With combo kernels enabled
        # (and benchmarking and memory limits turned off) it compiles a function returning a
        # square minus itself and an unrelated add-one result, asserts exact equality with eager,
        # exactly one generated kernel, a combo grid meta marker, the conditional select in the
        # code, and no fp fusion disabling.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Squares the first input and returns the square minus itself
            # together with the second input plus one.
            c = a * a
            return c - c, b + 1

        a, b = (torch.rand(1024, device=GPU_TYPE) for _ in range(2))
        with inductor_config.patch(
            {
                "combo_kernels": True,
                "benchmark_combo_kernel": False,
                "combo_kernel_per_subkernel_blocks": per_subkernel,
                "combo_kernel_max_distance": -1,
                "combo_kernel_peak_memory_increase_gb": None,
                "combo_kernel_peak_memory_pct_threshold": None,
            }
        ):
            metrics.reset()
            actual, (code,) = run_and_get_code(torch.compile(fn), a, b)

        self.assertEqual(actual, fn(a, b), atol=0, rtol=0)
        self.assertEqual(metrics.generated_kernel_count, 1)
        self.assertIn("'combo_grid_meta':", code)
        self.assertIn("tl.where", code)
        self.assertNotIn("'enable_fp_fusion': False", code)

    @requires_cuda_and_triton
    def test_duplicate_mse_loss_operands_are_zero(self):
        # [FUNCTION DOC] Checks that for gelu and mish, mean-squared-error loss between two
        # separate evaluations of the same op on the same input, reduced by max with no
        # reduction, is exactly zero after compilation (duplicate operands are recognised). Seeds
        # the RNG and uses fullgraph compilation.
        torch.manual_seed(0)
        x = torch.randn(10000, device=GPU_TYPE)

        for op in (F.gelu, F.mish):

            def fn(x, op=op):
                # [FUNCTION DOC] HELPER: Takes x and the activation op bound via a default
                # argument; returns the maximum of the unreduced MSE loss between two
                # applications of the op.
                return F.mse_loss(op(x), op(x), reduction="none").max()

            actual = torch.compile(fn, fullgraph=True)(x)
            self.assertEqual(actual, torch.zeros_like(actual), atol=0, rtol=0)

    @requires_gpu
    @common_utils.parametrize("grad_fn", [torch.no_grad, torch.enable_grad])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_with_grad_option(self, grad_fn, backend):
        # [FUNCTION DOC] Parametrized over grad mode context manager (no_grad or enable_grad) and
        # backend. Checks that a Triton mul2 kernel launched inside a grad-mode context manager
        # compiles fullgraph and gives twice the input.
        def call_triton(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Inside the parametrized grad-mode context, launches the mul2
            # kernel into a zero output and returns it.
            with grad_fn():
                output = torch.zeros_like(x)
                n_elements = output.numel()
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
                return output

        t = torch.rand(5, device=GPU_TYPE)
        compiled_func = torch.compile(call_triton, backend=backend, fullgraph=True)
        self.assertEqual(2 * t, compiled_func(t))

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_inner_triton_function(self, backend):
        # [FUNCTION DOC] Parametrized over backend. Compiles with fullgraph a function that
        # defines a Triton kernel inside its own body and launches it. The result comparison is
        # disabled (TODO: not yet supported), so only compile construction is exercised, and the
        # compiled function is never called.
        def f(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Defines a squaring kernel inline, launches it on a zero
            # output with block size 16, and returns the output.
            # [FUNCTION DOC] HELPER: Kernel squaring each masked input element and storing it to
            # the output.
            @triton.jit
            def pow2_kernel(
                in_ptr0,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(in_ptr0 + offsets, mask=mask)
                output = x * x
                tl.store(out_ptr + offsets, output, mask=mask)

            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            pow2_kernel[grid](x, output, n_elements, BLOCK_SIZE=16)
            return output

        t = torch.rand(5, device=GPU_TYPE)

        compiled_func = torch.compile(f, backend=backend, fullgraph=True)
        # TODO(oulgen): NYI - Support this
        # self.assertEqual(t * t, compiled_func(t))

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @inductor_config.patch("implicit_fallbacks", False)
    def test_triton_kernel_no_clones(self, grad, dynamic):
        # [FUNCTION DOC] Parametrized over grad and dynamic, with implicit fallbacks off.
        # Compiles a function that runs add_kernel directly via its run method and also
        # computes an add-one value. Checks equality with eager, that without grad exactly
        # one kernel is generated, that no copy or clone ops appear in the code, and that
        # only the expected outputs are returned (the exact return text differs for
        # dynamic with grad and the cpp wrapper).
        from torch._inductor.utils import run_and_get_code

        def call_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Runs the add kernel through its run method without warmup
            # with grid one program per element and block 16, writing into the given output; also
            # computes the input plus one; returns the output and that sum.
            n_elements = output.numel()

            tmp = torch.add(x, 1)
            grid = (x.numel(),)
            add_kernel.run(
                x, y, output, n_elements, warmup=False, grid=grid, BLOCK_SIZE=16
            )

            return output, tmp

        t1 = torch.rand(5, device=GPU_TYPE, requires_grad=grad)
        t2 = torch.rand(5, device=GPU_TYPE, requires_grad=grad)
        o1 = torch.zeros_like(t1, requires_grad=grad)

        torch_add = call_triton(t1, t2, o1)
        metrics.reset()
        o2 = torch.zeros_like(t1, requires_grad=grad)
        test, (code,) = run_and_get_code(
            torch.compile(call_triton, dynamic=dynamic), t1, t2, o2
        )
        if not grad:
            self.assertEqual(metrics.generated_kernel_count, 1)
        self.assertEqual(torch_add, test)
        # These two asserts are not optimal since it requires original aten
        # to be in the metadata, so there might be false negatives
        self.assertNotIn(
            "aoti_torch_copy_" if inductor_config.cpp_wrapper else "aten.copy", code
        )
        self.assertNotIn(
            "aoti_torch_clone" if inductor_config.cpp_wrapper else "aten.clone", code
        )
        # The following checks that there are only the tensor output is in
        # the compiled graph
        if dynamic and grad:
            if inductor_config.cpp_wrapper:
                self.assertIn("output_handles[0] = ", code)
                self.assertIn("output_handles[1] = ", code)
            else:
                self.assertIn("return (buf0, s92, )", code)
        else:
            self.assertIn(
                "output_handles[0] = "
                if inductor_config.cpp_wrapper
                else "return (buf0, )",
                code,
            )

    @requires_gpu
    def test_triton_kernel_caching(self):
        # [FUNCTION DOC] Checks Triton kernel caching with an autotuned add kernel invoked four
        # times in a loop. The compiled result must equal five times ones, and the generated code
        # must not contain a second copy of the kernel (the name with suffix 1), i.e. the same
        # kernel is reused.
        from torch._inductor.utils import run_and_get_code

        def add_in_loop(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Allocates a zero output and runs the autotuned add kernel
            # over it with a callable grid; returns the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_autotuned[grid](x, y, output, n_elements)
            return output

        def call_triton_add(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Applies the add-in-loop helper four times, feeding each
            # output back as the first input.
            for _ in range(4):
                x = add_in_loop(x, y)
            return x

        t1 = torch.ones(5, device=GPU_TYPE)
        t2 = torch.ones(5, device=GPU_TYPE)

        test, (code,) = run_and_get_code(torch.compile(call_triton_add), t1, t2)
        self.assertEqual(test, 5 * torch.ones(5, device=GPU_TYPE))
        self.assertTrue("add_kernel_autotuned_1.run" not in code)

    @requires_gpu
    def test_triton_kernel_caching_duplicate(self):
        # [FUNCTION DOC] Checks that two different kernels sharing the same name but defined in
        # different classes are both emitted. Compiles a function calling both, and asserts the
        # generated code launches kernels with the suffixes 0 and 1.
        from torch._inductor.utils import run_and_get_code

        class C:
            # [CLASS DOC] HELPER: Namespace class holding one Triton kernel named pass_kernel.
            # [FUNCTION DOC] HELPER: Kernel copying a masked block of the input to the output.
            @triton.jit
            def pass_kernel(
                in_ptr0,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(in_ptr0 + offsets, mask=mask)
                tl.store(out_ptr + offsets, x, mask=mask)

        class D:
            # [CLASS DOC] HELPER: Second namespace class with a kernel named pass_kernel whose
            # body is identical to the one in the first class.
            # [FUNCTION DOC] HELPER: Kernel copying a masked block of the input to the output.
            @triton.jit
            def pass_kernel(
                in_ptr0,
                out_ptr,
                n_elements,
                BLOCK_SIZE: "tl.constexpr",
            ):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(in_ptr0 + offsets, mask=mask)
                tl.store(out_ptr + offsets, x, mask=mask)

        def call_triton(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Calls each class's kernel on the input into separate outputs
            # and returns the sum of the outputs.
            output1 = torch.zeros_like(x)
            output2 = torch.zeros_like(x)
            n_elements = output1.numel()
            grid = (n_elements,)
            C.pass_kernel[grid](x, output1, n_elements, BLOCK_SIZE=16)
            D.pass_kernel[grid](x, output2, n_elements, BLOCK_SIZE=16)
            return output1 + output2

        t = torch.ones(5, device=GPU_TYPE)
        test, (code,) = run_and_get_code(torch.compile(call_triton), t)
        # Make sure we emitted two kernels here
        self.assertTrue(self._kernel_launched_in_code("pass_kernel_0", code))
        self.assertTrue(self._kernel_launched_in_code("pass_kernel_1", code))

    @requires_gpu
    def test_triton_kernel_various_args(self):
        # [FUNCTION DOC] Checks that compiling a call to an autotuned Triton kernel with many
        # argument kinds (None, a tensor, a float, a constexpr) does not crash. Nothing is
        # asserted on the result.
        # [FUNCTION DOC] HELPER: Autotuned (one config with block size 128, empty key) kernel
        # with an empty body; takes an output pointer, element count, None, tensor and float
        # dummy args and two constexprs.
        @triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 128})],
            key=[],
        )
        @triton.jit
        def pass_kernel(
            out_ptr,
            n_elements,
            dummy_None,
            dummy_empty,
            dummy_float,
            BLOCK_SIZE: "tl.constexpr",
            RANDOM_SIZE: "tl.constexpr",
        ):
            pass

        @torch.compile
        def call_triton(output):
            # [FUNCTION DOC] HELPER: Compiled function launching the kernel with one program per
            # element, passing None, an empty tensor like the output, a float and the constexpr;
            # returns the output.
            n_elements = output.numel()
            grid = (n_elements,)
            pass_kernel[grid](
                output,
                n_elements,
                None,
                torch.empty_like(output),
                3.1415926,
                RANDOM_SIZE=0,
            )
            return output

        output = torch.randn(5, device=GPU_TYPE)
        # Make sure this does not crash
        call_triton(output)

    @requires_gpu
    def test_triton_kernel_dependancies(self):
        # [FUNCTION DOC] Checks a dependency chain of two autotuned add kernels followed by an
        # add-one op: compiled output must equal eager.
        def call_triton(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Runs the autotuned add kernel into a zero output, then runs
            # it again with that output as input into a second zero output, and returns the
            # second plus one.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_autotuned[grid](x, y, output, n_elements)
            output2 = torch.zeros_like(output)
            add_kernel_autotuned[grid](output, y, output2, n_elements)
            output3 = torch.add(output2, 1)
            return output3

        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)
        torch_result = call_triton(t1, t2)
        compiled_result = torch.compile(call_triton)(t1, t2)
        self.assertEqual(torch_result, compiled_result)

    @requires_gpu
    def test_triton_kernel_reinplace_inplaceable_pass(self):
        # [FUNCTION DOC] Checks that the reinplace pass is correct for Triton
        # kernels: compiled output must equal eager for two launches where
        # the second reads and writes the output buffer.
        def call_triton(
            x: torch.Tensor,
            y: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Runs the autotuned add kernel into a zero output, then runs
            # it again with the output as first input and the same output as destination; returns
            # the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_autotuned[grid](x, y, output, n_elements)
            add_kernel_autotuned[grid](output, x, output, n_elements)
            return output

        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)
        torch_result = call_triton(t1, t2)
        compiled_result = torch.compile(call_triton)(t1, t2)
        self.assertEqual(torch_result, compiled_result)

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    def test_triton_kernel_multi_kernel(self, grad):
        # [FUNCTION DOC] Parametrized on grad. Checks nested Triton kernel calls and constexpr
        # activation selection: a kernel that calls an indirection kernel (which doubles inputs
        # in place) on both inputs and optionally applies zero-negatives. Runs a compiled
        # function launching it for a float case with the zero-negs activation and an integer
        # case without activation, then compares to the expected values: floats double-sum then
        # negatives zeroed; ints double-sum.
        # [FUNCTION DOC] HELPER: Kernel that doubles both inputs in place through the indirection
        # kernel, loads both, adds them, optionally zeroes negatives per the activation
        # constexpr, and stores the result masked.
        @triton.jit
        def mul2_and_add_and_zero_negatives_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
            ACTIVATION: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            indirection_kernel(
                in_ptr0,
                in_ptr0,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                ACTIVATION="mul2_inplace_kernel",
            )
            indirection_kernel(
                in_ptr1,
                in_ptr1,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                ACTIVATION="mul2_inplace_kernel",
            )
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            if ACTIVATION == "zero_negs":
                output = zero_negs(output)
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.compile
        def call_triton(
            x: torch.Tensor,
            y: torch.Tensor,
            xi: torch.Tensor,
            yi: torch.Tensor,
            output: torch.Tensor,
            outputi: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Compiled function launching the combined kernel once with
            # the zero-negs activation on the float tensors and once without activation on the
            # integer tensors; returns both outputs.
            n_elements = output.numel()

            grid = (x.numel(),)
            mul2_and_add_and_zero_negatives_kernel[grid](
                x, y, output, n_elements, BLOCK_SIZE=16, ACTIVATION="zero_negs"
            )
            mul2_and_add_and_zero_negatives_kernel[grid](
                xi, yi, outputi, n_elements, BLOCK_SIZE=16, ACTIVATION=None
            )

            return (output, outputi)

        t1 = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0], device=GPU_TYPE, requires_grad=grad
        )
        t2 = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0], device=GPU_TYPE, requires_grad=grad
        )
        float_result = 2 * t1 + 2 * t2
        float_result = float_result.where(float_result >= 0, 0.0)

        t1i = torch.randint(-2, 2, (5,), device=GPU_TYPE)
        t2i = torch.randint(-2, 2, (5,), device=GPU_TYPE)
        o = torch.zeros_like(t1, requires_grad=grad)
        oi = torch.zeros_like(t1i)
        int_result = 2 * t1i + 2 * t2i

        (result, resulti) = call_triton(t1, t2, t1i, t2i, o, oi)
        self.assertEqual(float_result, result)
        self.assertEqual(int_result, resulti)

    @requires_gpu
    def test_triton_kernel_constants(self):
        # [FUNCTION DOC] Checks how global constexpr constants are handled in kernels:
        # temporarily rebinds the global constant to a different value after the kernel was
        # parsed, then compares eager and compiled results (they must match, since kernels
        # capture constants at parse time). Restores the original global at the end (not in a
        # finally block, so a failure leaves it modified).
        # [FUNCTION DOC] HELPER: Kernel that reads a constexpr name; if it equals the string
        # constant, multiplies the input by the global constant; if the bool constant is true,
        # multiplies again by the constant; stores the result.
        @triton.jit
        def mulC_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
            CONSTANT_NAME: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE

            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            if CONSTANT_NAME == STRING_CONSTANT_C:
                output = CONSTANT_C * x
            if BOOL_CONSTANT_C:
                output *= CONSTANT_C
            tl.store(out_ptr + offsets, output, mask=mask)

        def call_triton(
            x: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Launches the constants kernel on a zero output with one
            # program per element, block 16 and the constant name; returns the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()

            grid = (x.numel(),)
            mulC_kernel[grid](
                x, output, n_elements, BLOCK_SIZE=16, CONSTANT_NAME="CONSTANT_C"
            )
            return output

        # Triton kernels capture global constants by their parse time value
        # not runtime value
        global CONSTANT_C
        prev_c = CONSTANT_C
        # If the behavior of triton kernels change, this test will fail
        CONSTANT_C = tl.constexpr(10)
        if CONSTANT_C == prev_c:
            raise AssertionError

        t = torch.randn(5, device=GPU_TYPE)
        torch_result = call_triton(t)
        compiled_result = torch.compile(call_triton)(t)

        self.assertEqual(torch_result, compiled_result)

        # reset back
        CONSTANT_C = prev_c

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("grid_type", [1, 2, 3])
    def test_triton_kernel_autotune(self, grad, dynamic, backend, grid_type):
        # [FUNCTION DOC] Parametrized over grad, dynamic, backend and grid type (tuple, lambda,
        # or named function). Compiles with fullgraph a function launching the autotuned add
        # kernel and compares to eager.
        def call_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Builds the grid according to grid_type, either a fixed
            # tuple, a lambda of the meta or a named function, then launches the autotuned add
            # kernel and returns the output.
            n_elements = output.numel()

            def grid_fn(meta):
                # [FUNCTION DOC] HELPER: Returns a one-element grid using ceiling division of the
                # element count by the block size meta parameter.
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            if grid_type == 1:
                grid = (n_elements,)
            elif grid_type == 2:
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            elif grid_type == 3:
                grid = grid_fn

            add_kernel_autotuned[grid](x, y, output, n_elements)
            return output

        t1 = torch.rand(256, device=GPU_TYPE, requires_grad=grad)
        t2 = torch.rand(256, device=GPU_TYPE, requires_grad=grad)
        output = torch.zeros_like(t1, requires_grad=grad)

        torch_add = call_triton(t1, t2, output)
        compiled_func = torch.compile(
            call_triton, backend=backend, fullgraph=True, dynamic=dynamic
        )

        output2 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, output2), torch_add)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @inductor_config.patch("unsafe_ignore_unsupported_triton_autotune_args", True)
    def test_triton_kernel_autotune_with_unsupported_args(self, backend):
        # [FUNCTION DOC] Parametrized over backend, with inductor config to ignore unsupported
        # autotune args. Checks that a kernel autotuned with options not supported by tooling
        # still compiles under fullgraph and matches eager.
        def call_triton(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Runs the kernel with unsupported autotune args on a zero
            # output with one program per element and returns it.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            add_kernel_autotuned_with_unsupported_args[(n_elements,)](
                x, y, output, n_elements
            )
            return output

        t1 = torch.rand(256, device=GPU_TYPE)
        t2 = torch.rand(256, device=GPU_TYPE)

        torch_add = call_triton(t1, t2)
        compiled_func = torch.compile(call_triton, backend=backend, fullgraph=True)
        compiled_add = compiled_func(t1, t2)
        self.assertEqual(compiled_add, torch_add)

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("grid_type", [1, 2, 3])
    @common_utils.parametrize("tdlp", ["0", "1"])
    def test_triton_kernel_2d_autotune(self, grad, dynamic, backend, grid_type, tdlp):
        # [FUNCTION DOC] Parametrized over grad, dynamic, backend, grid type and the dump-launch-
        # params setting. Compiles a 2D autotuned kernel launch and compares to eager, with
        # compilation and the call running inside the dump launch params context manager.
        def call_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Builds a grid from the output's two dims
            # (tuple, lambda or named function by grid_type), runs the 2D
            # autotuned add kernel, and returns the output.
            x_elements = output.size()[0]
            y_elements = output.size()[1]

            def grid_fn(meta):
                # [FUNCTION DOC] HELPER: Returns a two-dimensional grid using ceiling division of
                # each dim by the matching block meta parameter.
                return (
                    triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                    triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                )

            if grid_type == 1:
                grid = (x_elements, y_elements)
            elif grid_type == 2:
                grid = lambda meta: (
                    triton.cdiv(x_elements, meta["BLOCK_SIZE_X"]),
                    triton.cdiv(y_elements, meta["BLOCK_SIZE_Y"]),
                )
            elif grid_type == 3:
                grid = grid_fn

            add_kernel_2d_autotuned[grid](x, y, output, x_elements, y_elements)
            return output

        t1 = torch.rand((512, 256), device=GPU_TYPE, requires_grad=grad)
        t2 = torch.rand((512, 256), device=GPU_TYPE, requires_grad=grad)
        output = torch.zeros_like(t1, requires_grad=grad)

        torch_result = call_triton(t1, t2, output)

        with _dump_launch_params(tdlp):
            compiled_func = torch.compile(
                call_triton, backend=backend, fullgraph=True, dynamic=dynamic
            )
            output2 = torch.zeros_like(t1, requires_grad=grad)
            self.assertEqual(compiled_func(t1, t2, output2), torch_result)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_tracing(self, dynamic):
        # [FUNCTION DOC] Parametrized on dynamic. Traces several launch variants with make_fx in
        # symbolic or fake mode using the capture wrapper for kernels, covering tuple, lambda,
        # named function and list grids, positional and keyword block size, and autotuned
        # kernels. Also runs each variant eagerly and checks it equals the plain sum.
        def call_triton_add(
            x: torch.Tensor,
            y: torch.Tensor,
            grid_type: int,
            num=1,
            positional=False,
            autotuned=False,
        ):
            # [FUNCTION DOC] HELPER: Allocates an empty output, selects the grid per grid_type
            # (tuple, lambda, named function, or list), and launches add_kernel or the autotuned
            # add kernel through the capture wrapper (positional or keyword block size); returns
            # the output.
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid_fn(meta):
                # [FUNCTION DOC] HELPER: Returns a one-element grid from the num parameter and
                # the block size meta parameter.
                return (triton.cdiv(num, meta["BLOCK_SIZE"]),)

            if grid_type == 0:
                grid = (x.numel(),)
            elif grid_type == 1:
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            elif grid_type == 2:
                grid = grid_fn
            else:
                grid = [x.numel()]

            if autotuned:
                capture_triton(add_kernel_autotuned)[grid](x, y, output, n_elements)
            else:
                if positional:
                    capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
                else:
                    capture_triton(add_kernel)[grid](
                        x, y, output, n_elements, BLOCK_SIZE=16
                    )

            return output

        t0 = torch.rand(5, device=GPU_TYPE, requires_grad=True)
        t1 = torch.rand(5, device=GPU_TYPE, requires_grad=True)
        t2 = torch.rand(5, device=GPU_TYPE, requires_grad=True)
        t3 = torch.rand(5, device=GPU_TYPE, requires_grad=True)
        torch_add = t2 + t3

        tests = [
            functools.partial(call_triton_add, grid_type=0),
            functools.partial(call_triton_add, grid_type=1),
            functools.partial(call_triton_add, grid_type=1, num=1, positional=True),
            functools.partial(call_triton_add, grid_type=2, num=200),
            functools.partial(call_triton_add, grid_type=3),
            functools.partial(call_triton_add, grid_type=0, autotuned=True),
            functools.partial(call_triton_add, grid_type=1, num=1, autotuned=True),
            functools.partial(call_triton_add, grid_type=2, num=200, autotuned=True),
            functools.partial(call_triton_add, grid_type=3, autotuned=True),
        ]
        from torch.fx.experimental.proxy_tensor import make_fx

        tracing_mode = "symbolic" if dynamic else "fake"

        for test in tests:
            gm = make_fx(test, tracing_mode=tracing_mode)(t0, t1)
            result = test(t2, t3)
            self.assertEqual(result, torch_add)

    @requires_gpu
    @common_utils.parametrize("grad", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @inductor_config.patch("implicit_fallbacks", False)
    def test_triton_kernel_native(self, grad, dynamic, backend):
        # [FUNCTION DOC] Parametrized over grad, dynamic and backend with implicit fallbacks off.
        # Checks that launching add_kernel natively works eagerly (keyword and positional block
        # size) and compiled with fullgraph for a plain tuple grid, a lambda grid (keyword and
        # positional) and a named grid function.
        def call_triton_add(
            x: torch.Tensor,
            y: torch.Tensor,
            output: torch.Tensor,
            grid_type: int,
            num=1,
            positional=False,
        ):
            # [FUNCTION DOC] HELPER: Selects the grid per grid_type (tuple, lambda, named
            # function) and launches add_kernel with a keyword or positional block size of 16;
            # returns the output.
            n_elements = output.numel()

            def grid_fn(meta):
                # [FUNCTION DOC] HELPER: Returns a one-element grid from the num parameter and
                # the block size meta parameter.
                return (triton.cdiv(num, meta["BLOCK_SIZE"]),)

            if grid_type == 0:
                grid = (x.numel(),)
            elif grid_type == 1:
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            else:
                grid = grid_fn

            if positional:
                add_kernel[grid](x, y, output, n_elements, 16)
            else:
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)

            return output

        t1 = torch.rand(5, device=GPU_TYPE, requires_grad=grad)
        t2 = torch.rand(5, device=GPU_TYPE, requires_grad=grad)
        o1 = torch.zeros_like(t1, requires_grad=grad)

        torch_add = t1 + t2

        # No Dynamo -- Make sure triton kernel works
        self.assertEqual(call_triton_add(t1, t2, o1, 1), torch_add)
        # No Dynamo -- Make sure triton kernel works (with positional BLOCK_SIZE)
        o2 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(call_triton_add(t1, t2, o2, 1, True), torch_add)

        # With Dynamo
        compiled_func = torch.compile(
            call_triton_add, backend=backend, fullgraph=True, dynamic=dynamic
        )
        # With simple kernel
        o3 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o3, 0), torch_add)
        # With lambda kernel
        o4 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o4, 1), torch_add)
        # With lambda kernel (with positional BLOCK_SIZE)
        o5 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o5, 1, 1, True), torch_add)
        # With user defined function kernel
        o6 = torch.zeros_like(t1, requires_grad=grad)
        self.assertEqual(compiled_func(t1, t2, o6, 2, 200), torch_add)

    @requires_gpu
    def test_triton_kernel_mutation_not_mark_dirty(self):
        # [FUNCTION DOC] Checks that a compiled in-place Triton add on a clone does not mark the
        # input dirty in a way that breaks autograd. Runs a compiled function launching
        # add_kernel with the same tensor as inputs and output, after taking a sine of the clone,
        # and backpropagates the sum with no exception expected.
        @torch.compile
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function running add_kernel in place with x for
            # both inputs and the output, returning x.
            n_elements = x.numel()
            add_kernel[(n_elements,)](x, x, x, n_elements, 16)
            return x

        x = torch.randn(5, device=GPU_TYPE, requires_grad=True)
        x_cloned = x.clone()
        out = x_cloned.sin()
        f(x_cloned)
        out.sum().backward()

    @requires_gpu
    @inductor_config.patch("allow_buffer_reuse", True)
    def test_triton_kernel_inputs_buffer_reuse(self):
        # [FUNCTION DOC] With buffer reuse allowed, checks buffer reuse across chained Triton
        # kernel calls: four doublings in a loop plus one. Verifies compiled equals eager, that
        # exactly two buffers are allocated in the generated code and three buffers are reused.
        def _mul2(x):
            # [FUNCTION DOC] HELPER: Allocates an empty output like x, launches the mul2 kernel
            # on ten programs of block size 1, and returns the output.
            y = torch.empty_like(x)
            mul2_kernel[(10,)](
                in_ptr0=x,
                out_ptr=y,
                n_elements=x.numel(),
                BLOCK_SIZE=1,
            )
            return y

        @torch.compile
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function applying the helper four times in sequence
            # and adding one.
            for _ in range(4):
                # The output of one kernel is the input to the next kernel, but
                # at some point we should reuse buffers not allocate new ones.
                x = _mul2(x)
            return x + 1

        x = torch.randn(10, device=GPU_TYPE, dtype=torch.float32)
        eager_out = f(x)
        compiled_out, (code,) = run_and_get_code(torch.compile(f), x)
        self.assertEqual(compiled_out, eager_out)

        # Check that we're allocating the minimal # of buffers.
        code_string = (
            "aoti_torch_empty_strided("
            if inductor_config.cpp_wrapper
            else f"empty_strided_{GPU_TYPE}((10, ), (1, ), torch.float32)"
        )
        num_bufs_allocated = code.count(code_string)
        self.assertEqual(num_bufs_allocated, 2)

        # Check we're reusing buffers if not allocating.
        num_bufs_reused = code.count(
            "// reuse" if inductor_config.cpp_wrapper else "# reuse"
        )
        self.assertEqual(num_bufs_reused, 3)

    @requires_gpu
    def test_triton_kernel_matmul_tracking(self):
        # [FUNCTION DOC] Checks that a Triton kernel writing ones into a buffer is tracked before
        # a matmul: the compiled function zeroes a tensor, fills it with the kernel, then
        # multiplies it by the input and adds ten; result must equal the matmul of ones with the
        # input plus ten.
        # [FUNCTION DOC] HELPER: Kernel storing the constant one for every masked element.
        @triton.jit
        def ones_kernel(x_ptr, n_elements, BLOCK_SIZE: "tl.constexpr"):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = 1.0
            tl.store(x_ptr + offsets, x, mask=mask)

        @torch.compile
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function that zeroes a tensor, runs the ones kernel
            # on it with a grid of four programs, and returns its matmul with x plus ten.
            out = torch.zeros_like(x)
            ones_kernel[(4,)](out, 16, BLOCK_SIZE=16)
            return torch.mm(out, x) + 10

        x = torch.randn(4, 4, device=GPU_TYPE)
        torch_out = f(x)
        python_out = torch.mm(torch.ones(4, 4, device=GPU_TYPE), x) + 10
        self.assertEqual(torch_out, python_out)

    @requires_gpu
    def test_triton_kernel_strided_input(self):
        # [FUNCTION DOC] Checks a Triton kernel with a strided input: splits a 64 by 256 tensor
        # into two column halves and runs the double-strided kernel on the first, which has row
        # stride 256. Compares compiled with eager.
        def f(inp):
            # [FUNCTION DOC] HELPER: Splits the input along dim 1 into two
            # halves, runs the double-strided kernel on the left half with
            # its row strides, and returns the output.
            # left has strides [256, 1]
            left, right = torch.split(inp, [128, 128], dim=1)
            out = torch.empty_like(left)
            X_BLOCK_SIZE, Y_BLOCK_SIZE = 32, 16
            grid = (left.size(1) // X_BLOCK_SIZE, left.size(0) // Y_BLOCK_SIZE)
            double_strided_kernel[grid](
                in_ptr=left,
                out_ptr=out,
                in_y_stride=left.stride(0),
                out_y_stride=out.stride(0),
                X_BLOCK_SIZE=X_BLOCK_SIZE,
                Y_BLOCK_SIZE=Y_BLOCK_SIZE,
            )
            return out

        inp = torch.randn(64, 256, device=GPU_TYPE)

        eager_out = f(inp)
        compiled_out = torch.compile(f)(inp)
        self.assertEqual(compiled_out, eager_out)

    @inductor_config.patch(
        triton_kernel_default_layout_constraint="needs_fixed_stride_order"
    )
    @requires_gpu
    def test_layout_constraint_needs_fixed_stride_order(self):
        # [FUNCTION DOC] Under the config choosing the needs-fixed-stride-order default layout
        # constraint, checks that inductor forces a custom op's output to the stride order seen
        # at tracing time before passing it to a Triton kernel that fills values by flat offset.
        # The custom op's lowering would otherwise produce a different layout; compiled output
        # must equal eager.
        # Construct a custom op whose output strides are (1, 2)
        @torch.library.custom_op("mylib::weird_op_with_lowering", mutates_args={})
        def weird_op_with_lowering(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op body returning an uninitialized 2 by 2 tensor with
            # column-major strides.
            return torch.empty_strided((2, 2), (1, 2), dtype=x.dtype, device=x.device)

        @weird_op_with_lowering.register_fake
        def _(x):
            # [FUNCTION DOC] HELPER: Fake implementation of the custom op returning an empty
            # strided tensor with column-major strides.
            return torch.empty_strided((2, 2), (1, 2), dtype=x.dtype, device=x.device)

        # The lowering for the custom op produces output strides (2, 1).
        from torch._inductor.lowering import empty_strided, register_lowering

        @register_lowering(torch.ops.mylib.weird_op_with_lowering)
        def _(x):
            # [FUNCTION DOC] HELPER: Inductor lowering for the custom op
            # producing an empty strided buffer on the GPU with row-major
            # strides (differing from the traced strides).
            return empty_strided(
                x.shape, (2, 1), dtype=x.dtype, device=torch.device(GPU_TYPE, 0)
            )

        # Triton kernel that has different behavior depending on the input strides.
        # [FUNCTION DOC] HELPER: Kernel storing each flat offset index into the output, masked by
        # element count; its result depends on the input strides.
        @triton.jit
        def kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            output = offsets
            tl.store(out_ptr + offsets, output, mask=mask)

        def arange_out(x, out):
            # [FUNCTION DOC] HELPER: Launches the offset-filling kernel with block size 4 over
            # the output.
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            kernel[grid](x, out, n_elements, BLOCK_SIZE=4)

        def f(x):
            # [FUNCTION DOC] HELPER: Calls the custom op, fills its output in place with the
            # kernel, and returns the input plus that output.
            y = weird_op_with_lowering(x)
            # Inductor lowering will decide that y is better having strides (2, 1).
            # This is different from the strides at tracing time (1, 2).
            # Under the "needs_fixed_stride_order" config, inductor will coerce
            # y to have strides (1, 2) before passing it to arange_out.
            # If it doesn't, then the result will be different from eager mode.
            arange_out(x, y)
            return x + y

        x = torch.randn(2, 2, device=GPU_TYPE)
        eager_out = f(x)

        compiled_inductor_f = torch.compile(f, backend="inductor", fullgraph=True)
        compiled_inductor_out = compiled_inductor_f(x)
        self.assertEqual(compiled_inductor_out, eager_out)

    @requires_gpu
    def test_triton_kernel_strided_input_nonzero_offset(self):
        # [FUNCTION DOC] Checks a Triton kernel with a strided input that has a nonzero storage
        # offset (the second column half). Compares compiled with eager.
        def f(inp):
            # [FUNCTION DOC] HELPER: Splits the input along dim 1, runs the double-strided kernel
            # on the right half with its strides, and returns the output.
            # right has strides [256, 1] and storage offset 128
            left, right = torch.split(inp, [128, 128], dim=1)
            out = torch.empty_like(right)
            X_BLOCK_SIZE, Y_BLOCK_SIZE = 32, 16
            grid = (right.size(1) // X_BLOCK_SIZE, right.size(0) // Y_BLOCK_SIZE)
            double_strided_kernel[grid](
                in_ptr=right,
                out_ptr=out,
                in_y_stride=right.stride(0),
                out_y_stride=out.stride(0),
                X_BLOCK_SIZE=X_BLOCK_SIZE,
                Y_BLOCK_SIZE=Y_BLOCK_SIZE,
            )
            return out

        inp = torch.randn(64, 256, device=GPU_TYPE)

        eager_out = f(inp)
        compiled_out = torch.compile(f)(inp)
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    def test_triton_kernel_slice_and_view_input(self):
        # [FUNCTION DOC] Checks a Triton kernel on a sliced and viewed input (first 128 columns
        # reshaped to three dims). Compares compiled with eager.
        def f(inp):
            # [FUNCTION DOC] HELPER: Slices and reshapes the input, runs the double-strided
            # kernel using the leading stride, and returns the output plus the input view.
            # left has strides [256, 1]
            left = inp[:, :128]
            left = left.view(64, 4, 32)
            out = torch.empty_like(left)
            X_BLOCK_SIZE, Y_BLOCK_SIZE = 32, 16
            grid = (
                (left.size(1) * left.size(2)) // X_BLOCK_SIZE,
                left.size(0) // Y_BLOCK_SIZE,
            )
            double_strided_kernel[grid](
                in_ptr=left,
                out_ptr=out,
                in_y_stride=left.stride(0),
                out_y_stride=out.stride(0),
                X_BLOCK_SIZE=X_BLOCK_SIZE,
                Y_BLOCK_SIZE=Y_BLOCK_SIZE,
            )
            return out + left

        inp = torch.randn(64, 256, device=GPU_TYPE)

        eager_out = f(inp)
        compiled_out = torch.compile(f)(inp)
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    def test_triton_kernel_fallback(self):
        # [FUNCTION DOC] Checks Triton kernels fed by extern and fallback kernel outputs: one
        # launch takes a matrix multiply result, another a sort result (a multi-output fallback).
        # Compares compiled with eager.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Launches add_kernel with a matmul result as
            # the second input and then with the values of a sort; returns the
            # destination and an untouched zeros tensor.
            out = torch.zeros_like(x)
            out2 = torch.zeros_like(x)
            # torch.mm is ExternKernelOut
            add_kernel[(4,)](x, torch.mm(x, y), out, 4, 16)
            # torch.sort creates fallback kernel and hence MultiOutput
            add_kernel[(4,)](x, torch.sort(y).values, out, 4, 16)
            return out, out2

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)
        eager_out = f(x, y)
        compiled_out = torch.compile(f)(x, y)
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    def test_triton_kernel_to_cpu(self):
        # [FUNCTION DOC] Checks a Triton kernel followed by a device-to-host copy in the same
        # compiled function. Compares compiled with eager.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Runs add_kernel on one program into a zero output, copies
            # the output to the CPU, adds one, and returns it.
            out = torch.zeros_like(x)
            add_kernel[(1,)](x, y, out, 16, 16)
            out_cpu = out.cpu() + 1
            return out_cpu

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)
        eager_out = f(x, y)
        compiled_out = torch.compile(f)(x, y)
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    def test_triton_kernel_out_of_order(self):
        # [FUNCTION DOC] Checks a Triton kernel whose constexpr parameter sits in the middle of
        # its argument list. The compiled launch with positional arguments must match eager.
        # [FUNCTION DOC] HELPER: Local add kernel whose parameter order is two inputs, then
        # block-size constexpr, then output pointer and element count.
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            BLOCK_SIZE: "tl.constexpr",
            out_ptr,
            n_elements,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Zero output; launches the out-of-order add kernel
            # positionally with block size 4; returns the output.
            out = torch.zeros_like(x)
            n_elements = x.numel()
            add_kernel[(n_elements,)](x, y, 4, out, n_elements)
            return out

        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        eager_out = f(x, y)
        compiled_out = torch.compile(f)(x, y)
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @dynamo_config.patch(capture_dynamic_output_shape_ops=True)
    @dynamo_config.patch(capture_scalar_outputs=True)
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_unbacked_shape_tensor(self, backend):
        # [FUNCTION DOC] Parametrized over backend, with dynamo configs capturing dynamic-output-
        # shape ops and scalar outputs. Checks a Triton kernel launched on a tensor whose size is
        # data dependent (boolean mask selection), compiled with fullgraph, equals eager.
        # [FUNCTION DOC] HELPER: Kernel squaring each masked element.
        @triton.jit
        def square(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            output = x * x
            tl.store(out_ptr + offsets, output, mask=mask)

        def f(x):
            # [FUNCTION DOC] HELPER: Selects elements greater than two, then launches the
            # squaring kernel on the selection with a grid from the data-dependent element count;
            # returns the output.
            x = x[x > 2]
            n_elements = x.numel()
            output = torch.zeros_like(x)
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            square[grid](x, output, n_elements, BLOCK_SIZE=16)
            return output

        x = torch.randn(4, device=GPU_TYPE)
        eager_out = f(x)
        compiled_out = torch.compile(f, fullgraph=True, backend=backend)(x)
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @common_utils.parametrize("dump_launch_params", ["0", "1"])
    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_equal_to_1_arg(self, dynamic, dump_launch_params):
        # [FUNCTION DOC] Parametrized over dynamic and dump-launch-params. Checks the equal-to-
        # one argument specialization in generated Triton source for a kernel receiving half the
        # element count (equal to one). With attrs-dict style triton the source must not mention
        # it; otherwise the dynamic case must show an empty specialization tuple and the static
        # case one containing the argument position three. Output must equal eager.
        # [FUNCTION DOC] HELPER: Kernel adding two inputs for indices below twice its
        # half_n_elements argument.
        @triton.jit
        def add_kernel_half_n_elements(
            in_ptr0,
            in_ptr1,
            out_ptr,
            half_n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < half_n_elements * 2
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Computes half of the input size and launches the half-count
            # kernel over that many programs with block 16; returns the output.
            out = torch.empty_like(x)
            half_n_elements = x.numel() // 2
            add_kernel_half_n_elements[(half_n_elements,)](
                x, y, out, half_n_elements, BLOCK_SIZE=16
            )
            return out

        x = torch.randn(2, device=GPU_TYPE)
        y = torch.randn(2, device=GPU_TYPE)
        eager_out = f(x, y)

        with _dump_launch_params(dump_launch_params):
            compiled_out, sources = run_and_get_code(
                torch.compile(f, dynamic=dynamic), x, y
            )

            if triton_version_uses_attrs_dict():
                self.assertFalse("equal_to" in sources[0])
            else:
                if dynamic:
                    # when half_n_elements passed to the Triton kernel is
                    # dynamic, equal_to_1 specialization can't be enforced

                    # also, equal_to_1 specialization doesn't occur (or appear in the signature)
                    # for newer versions of triton (i.e. the ones where triton_version_uses_attrs_dict() == True)
                    self.assertTrue(_triton_get_ast_equal_to_str(()) in sources[0])
                else:
                    self.assertTrue(_triton_get_ast_equal_to_str((3,)) in sources[0])
            self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_equal_to_1_float_arg(self, dynamic):
        # [FUNCTION DOC] Parametrized on dynamic. Checks that a float argument equal to 1.0
        # (derived from a size expression to the power zero divided by one) is not added to the
        # equal-to-1 specialization: for non-attrs-dict triton the source shows an empty tuple.
        # Output must equal eager.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Computes a float scale of one from the element count and
            # launches the scaling add kernel, passing it; returns the output.
            out = torch.empty_like(x)
            n_elements = x.numel()
            scaling_factor = (n_elements**0) / 1.0
            add_kernel_with_scaling[(n_elements,)](
                x,
                y,
                out,
                n_elements,
                scaling_factor,
                BLOCK_SIZE=16,
            )
            return out

        x = torch.randn(2, device=GPU_TYPE)
        y = torch.randn(2, device=GPU_TYPE)
        eager_out = f(x, y)
        compiled_out, sources = run_and_get_code(
            torch.compile(f, dynamic=dynamic), x, y
        )

        # float 1.0 (both literal or symbolic)
        # should not be added to equal_to_1
        if not triton_version_uses_attrs_dict():
            self.assertTrue(_triton_get_ast_equal_to_str(()) in sources[0])
        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    def test_triton_kernel_with_imported_symbol(self):
        # [FUNCTION DOC] Checks a kernel that calls an imported function (the libdevice fast
        # divide) works compiled. Compares with eager.
        # [FUNCTION DOC] HELPER: Kernel storing the fast division of each masked input element by
        # the constant 3.14.
        @triton.jit
        def add_kernel_with_imported_symbol(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            output = fast_dividef(x, 3.14)
            tl.store(out_ptr + offsets, output, mask=mask)

        def f(x):
            # [FUNCTION DOC] HELPER: Launches the kernel with one program per element, block 16,
            # and returns the output.
            out = torch.empty_like(x)
            n_elements = x.numel()
            add_kernel_with_imported_symbol[(n_elements,)](
                x, out, n_elements, BLOCK_SIZE=16
            )
            return out

        x = torch.randn(4, device=GPU_TYPE)
        eager_out = f(x)
        compiled_out = torch.compile(f)(x)

        self.assertEqual(compiled_out, eager_out)

    @unittest.skipIf(
        not HAS_GPU or not hasattr(triton, "constexpr_function"),
        "newer triton version required",
    )
    def test_triton_kernel_with_constexpr_function(self):
        # [FUNCTION DOC] Skipped unless triton provides constexpr_function. Checks that a kernel
        # calling a module-level constexpr function (log2) is compiled with fullgraph, and that
        # the generated source contains the constexpr function decorator. Compares with eager.
        # [FUNCTION DOC] HELPER: Kernel adding the log2 of the loaded block's first dimension (a
        # constexpr) to each element.
        @triton.jit
        def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)

            FIRST_DIM: tl.constexpr = x.shape[0]
            output = x + log2(FIRST_DIM)
            tl.store(output_ptr + offsets, output, mask=mask)

        def f(x):
            # [FUNCTION DOC] HELPER: Zero output; launches the kernel with callable grid and
            # block 16; returns the output.
            out = torch.zeros_like(x)
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            kernel[grid](x, out, n_elements, BLOCK_SIZE=16)
            return out

        x = torch.randn(16, device=GPU_TYPE)
        eager_out = f(x)
        compiled_out, (triton_code,) = run_and_get_code(
            torch.compile(f, fullgraph=True), x
        )

        self.assertIn("@triton.constexpr_function", triton_code)
        self.assertEqual(compiled_out, eager_out)

    @unittest.skipIf(
        not HAS_GPU or not hasattr(triton, "constexpr_function"),
        "newer triton version required",
    )
    def test_triton_kernel_with_constexpr_dtype_annotations(self):
        # [FUNCTION DOC] Skipped unless triton provides constexpr_function. Checks codegen of a
        # jit helper that takes a dtype produced by a constexpr function: generated source must
        # import the dtype class, include the constexpr decorator, alias the int dtype lookup,
        # emit the int8 dtype instance assignment, the jit decorator and helper definition. The
        # result must equal eager and be all ones.
        """
        Test that constexpr functions with dtype type annotations work correctly.
        This tests the fix for proper handling of:
        1. Type annotations using triton.language.core.dtype
        2. Dtype instances (int8, uint8, etc.) that lack __name__ attribute
        3. Function name aliasing
        """

        # [FUNCTION DOC] HELPER: Kernel building a float tensor of ones, passing it through the
        # dtype helper (bitcast to an int type and back), and storing it to the output.
        @triton.jit
        def kernel_with_dtype_annotation(out_ptr, n: tl.constexpr):
            offs = tl.arange(0, n)
            x = tl.full([n], 1.0, dtype=tl.float32)
            y = _dtype_helper_test(x)
            tl.store(out_ptr + offs, y.to(tl.float32, bitcast=True))

        def f(n):
            # [FUNCTION DOC] HELPER: Allocates an empty output of size n and launches the kernel
            # with one program; returns the output.
            out = torch.empty(n, device=GPU_TYPE)
            kernel_with_dtype_annotation[(1,)](out, n)
            return out

        n = 8
        eager_out = f(n)
        compiled_out, (triton_code,) = run_and_get_code(
            torch.compile(f, fullgraph=True), n
        )

        # Verify the generated code has proper imports
        self.assertIn("from triton.language.core import dtype as dtype", triton_code)
        self.assertIn("@triton.constexpr_function", triton_code)
        self.assertIn("_get_int_dtype_test = get_int_dtype", triton_code)
        # Verify dtype instances are emitted correctly
        self.assertIn("int8 = tl.int8", triton_code)
        self.assertIn("@triton.jit", triton_code)
        self.assertIn("def _dtype_helper_test", triton_code)

        # Verify correctness
        self.assertEqual(compiled_out, eager_out)
        self.assertTrue(torch.all(compiled_out == 1.0).item())

    @requires_gpu
    def test_triton_kernel_with_imported_symbol_with_custom_name(self):
        # [FUNCTION DOC] Same as the imported-symbol test but the imported function is referenced
        # through an alias name defined at module level. Checks compiled equals eager.
        # [FUNCTION DOC] HELPER: Kernel storing the fast division by 3.14 using the aliased
        # function name.
        @triton.jit
        def add_kernel_with_imported_symbol(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            output = my_fast_dividef(x, 3.14)
            tl.store(out_ptr + offsets, output, mask=mask)

        def f(x):
            # [FUNCTION DOC] HELPER: Launches the kernel with one program per element, block 16;
            # returns the output.
            out = torch.empty_like(x)
            n_elements = x.numel()
            add_kernel_with_imported_symbol[(n_elements,)](
                x, out, n_elements, BLOCK_SIZE=16
            )
            return out

        x = torch.randn(4, device=GPU_TYPE)
        eager_out = f(x)
        compiled_out = torch.compile(f)(x)

        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @common_utils.parametrize("size", [4, 16])
    @common_utils.parametrize("dynamic", [False, True])
    def test_triton_kernel_different_shapes(self, size, dynamic):
        # [FUNCTION DOC] Parametrized over size and dynamic. Calls add_kernel with a one-
        # dimensional and a two-dimensional input; asserts that, for size four and static shapes,
        # two separately specialized kernels are launched (divisibility), otherwise only one.
        # Output must match eager.
        from torch._inductor.utils import run_and_get_code

        def f(x, y, xx, yy):
            # [FUNCTION DOC] HELPER: Launches add_kernel (block 4) on the first pair into one
            # output and on the second pair into another; returns both outputs.
            n_elements = x.numel()
            output_1 = torch.zeros_like(x)
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, output_1, n_elements, BLOCK_SIZE=4)

            n_elements = xx.numel()
            output_2 = torch.zeros_like(xx)
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](xx, yy, output_2, n_elements, BLOCK_SIZE=4)

            return output_1, output_2

        x = torch.rand(size, device=GPU_TYPE)
        y = torch.rand(size, device=GPU_TYPE)
        xx = torch.rand(size, size, device=GPU_TYPE)
        yy = torch.rand(size, size, device=GPU_TYPE)
        args = [x, y, xx, yy]

        eager_out = f(*args)
        compiled_out, (code,) = run_and_get_code(
            torch.compile(f, fullgraph=True, dynamic=dynamic, backend="inductor"), *args
        )
        if size == 4 and not dynamic:
            # Produce 2 kernels due to divisibility
            self.assertTrue(self._kernel_launched_in_code("add_kernel_0", code))
            self.assertTrue(self._kernel_launched_in_code("add_kernel_1", code))
        else:
            # size == 16 or dynamic
            # Only one kernel
            self.assertTrue(self._kernel_launched_in_code("add_kernel_0", code))
            self.assertFalse(self._kernel_launched_in_code("add_kernel_1", code))

        self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_triton_dtype(self, dynamic, backend):
        # [FUNCTION DOC] Parametrized over dynamic and backend. Checks passing triton dtype
        # objects as constexpr kernel args: float32 always, and bfloat16 when supported. Compiled
        # fullgraph results equal eager.
        # [FUNCTION DOC] HELPER: Kernel loading both inputs, casting them to the constexpr dtype,
        # adding them and storing.
        @triton.jit
        def add_kernel_with_dtype(
            in_ptr0,
            in_ptr1,
            out_ptr,
            dtype: "tl.constexpr",
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask).to(dtype)
            y = tl.load(in_ptr1 + offsets, mask=mask).to(dtype)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        def f(x, y, dtype_torch, dtype_triton):
            # [FUNCTION DOC] HELPER: Zero output cast to the given torch dtype; launches the
            # dtype kernel with the triton dtype; returns the output.
            output = torch.zeros_like(x).to(dtype=dtype_torch)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_with_dtype[grid](
                x, y, output, dtype_triton, n_elements, BLOCK_SIZE=4
            )
            return output

        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        args_list = [(x, y, torch.float32, tl.float32)]
        if torch.cuda.is_bf16_supported(including_emulation=False):
            args_list.append((x, y, torch.bfloat16, tl.bfloat16))

        for args in args_list:
            eager_out = f(*args)
            compiled_out = torch.compile(
                f, fullgraph=True, backend=backend, dynamic=dynamic
            )(*args)
            self.assertEqual(compiled_out, eager_out)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_special_kwargs_with_autotune(self, backend):
        # [FUNCTION DOC] Parametrized on backend. Checks that the special launch kwargs for warps
        # and stages are accepted together with an autotuned kernel (two block-size configs,
        # keyed on element count). The test only requires the call not to fail.
        # [FUNCTION DOC] HELPER: Autotuned (two configs) elementwise add kernel with a mask.
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 128}),
                triton.Config({"BLOCK_SIZE": 64}),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.compile(fullgraph=True, backend=backend)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Compiled fullgraph function launching the autotuned kernel
            # with the warps and stages kwargs set; returns the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](
                x,
                y,
                output,
                n_elements,
                num_warps=8,
                num_stages=3,
            )
            return output

        x = torch.randn(4, device=GPU_TYPE)
        f(x, x)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_empty_autotune_config_dict(self, backend):
        # [FUNCTION DOC] Parametrized on backend. Checks an autotuner whose configs have empty
        # dicts and differ only in stage count, while block size is passed at launch. The test
        # only requires the call not to fail.
        # [FUNCTION DOC] HELPER: Autotuned (two empty-dict
        # configs) elementwise add kernel with a mask.
        @triton.autotune(
            configs=[
                triton.Config({}, num_stages=2),
                triton.Config({}, num_stages=3),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.compile(fullgraph=True, backend=backend)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Compiled fullgraph function launching the kernel with block
            # size 128; returns the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](
                x,
                y,
                output,
                n_elements,
                BLOCK_SIZE=128,
            )
            return output

        x = torch.randn(4, device=GPU_TYPE)
        f(x, x)

    @requires_gpu
    @common_utils.parametrize("autotune", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_special_params(self, autotune, backend):
        # [FUNCTION DOC] Parametrized over autotune and backend. Skipped for inductor with the
        # cpp wrapper. Checks a kernel that declares warps and stages as constexpr parameters and
        # uses them in arithmetic: passed as launch kwargs, or through an autotuner with matching
        # configs. The eager and compiled outputs must equal x times stages plus warps.
        if backend == "inductor" and inductor_config.cpp_wrapper:
            raise unittest.SkipTest(
                "cpp-wrapper doesn't support num_warps/num_stages as constexpr"
            )

        # [FUNCTION DOC] HELPER: Kernel storing the input times the stages constexpr plus the
        # warps constexpr.
        @triton.jit
        def special_params_kernel(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
            num_warps: "tl.constexpr",
            num_stages: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            output = x * num_stages + num_warps
            tl.store(out_ptr + offsets, output, mask=mask)

        NUM_WARPS = 4
        NUM_STAGES = 3

        if autotune:
            special_params_kernel = triton.autotune(
                configs=[
                    triton.Config(
                        {"BLOCK_SIZE": 128},
                        num_stages=NUM_STAGES,
                        num_warps=NUM_WARPS,
                    ),
                    triton.Config(
                        {"BLOCK_SIZE": 64},
                        num_stages=NUM_STAGES,
                        num_warps=NUM_WARPS,
                    ),
                ],
                key=["n_elements"],
            )(special_params_kernel)
            kwargs = {}
        else:
            kwargs = {
                "BLOCK_SIZE": 128,
                "num_stages": NUM_STAGES,
                "num_warps": NUM_WARPS,
            }

        def f(x):
            # [FUNCTION DOC] HELPER: Zero output; launches the kernel using the extra kwargs
            # chosen by the autotune flag; returns the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            special_params_kernel[grid](
                x,
                output,
                n_elements,
                **kwargs,
            )
            return output

        x = torch.randn(4, device=GPU_TYPE)
        eager_out = f(x)
        compiled_out = torch.compile(f, fullgraph=True, backend=backend)(x)
        expected_out = x * NUM_STAGES + NUM_WARPS
        self.assertEqual(eager_out, expected_out)
        self.assertEqual(compiled_out, expected_out)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_on_device_tma(self, dynamic, tma_version):
        # [FUNCTION DOC] Parametrized over dynamic and TMA API version. Skips when the host lacks
        # support for that API. Checks launching an add kernel that builds TMA descriptors on the
        # device: allocates a zeroed 3 by 128 byte workspace for the old API, passes none for the
        # new API, and sets an allocator for triton. Eager and compiled outputs equal the sum.
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")
        kernel = (
            add_kernel_on_device_tma_new_api
            if tma_version == "new"
            else add_kernel_on_device_tma_old_api
        )

        def f(a, b):
            # [FUNCTION DOC] HELPER: Zero output; for the old API allocates a workspace of bytes;
            # launches the on-device TMA kernel over a 2D grid with block 32; returns the output.
            BLOCK_SIZE = 32
            out = torch.zeros_like(a)
            m, n = out.size()

            # Allocate workspace for on-device TMA descriptors
            # Need 128 bytes per descriptor, 3 descriptors total
            if tma_version == "old":
                workspace = torch.zeros(3 * 128, dtype=torch.uint8, device=a.device)
            else:
                workspace = None

            grid = lambda meta: (
                triton.cdiv(m, meta["BLOCK_SIZE"]),
                triton.cdiv(n, meta["BLOCK_SIZE"]),
            )

            kernel[grid](
                a,
                b,
                out,
                m,
                n,
                workspace,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out

        a = torch.randn((32, 32), device=GPU_TYPE)
        b = torch.randn((32, 32), device=GPU_TYPE)

        expected_out = a + b
        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device=GPU_TYPE
            )
        )
        eager_out = f(a, b)
        compiled_out = torch.compile(f, fullgraph=True, dynamic=dynamic)(a, b)

        self.assertEqual(eager_out, expected_out)
        self.assertEqual(compiled_out, expected_out)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_multiple_outputs(self, dynamic, backend):
        # [FUNCTION DOC] Parametrized over dynamic and backend. Checks a kernel with two output
        # buffers (the sum and the sum plus one), compiled fullgraph with inputs requiring grad;
        # the third value returned is the square of another input to exercise training. Compares
        # results with expected values.
        # [FUNCTION DOC] HELPER: Kernel storing the sum of two inputs to the first output and
        # that sum plus one to the second.
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            out_ptr2,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)
            tl.store(out_ptr2 + offsets, output + 1, mask=mask)

        @torch.compile(fullgraph=True, backend=backend, dynamic=dynamic)
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Compiled function launching the two-output kernel and
            # returning both outputs and the square of the third argument.
            output = torch.empty_like(x)
            output2 = torch.empty_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, output, output2, n_elements, BLOCK_SIZE=16)
            # The z return is intentional: we're testing training
            return output, output2, z**2

        x = torch.randn(3, requires_grad=True, device=GPU_TYPE)
        y = torch.randn(3, requires_grad=True, device=GPU_TYPE)
        z = torch.randn(3, requires_grad=True, device=GPU_TYPE)
        out, out2, out3 = f(x, y, z)
        self.assertEqual(out, x + y)
        self.assertEqual(out2, x + y + 1)
        self.assertEqual(out3, z**2)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_tma_capture_and_functionalize(self, dynamic, tma_version):
        # [FUNCTION DOC] Parametrized over dynamic and TMA version. Skips if unsupported; resets
        # the kernel side table. Compiles a function creating host TMA descriptors for three
        # tensors and launching a 1D TMA add kernel, using a backend recording the ahead-of-time
        # forward graphs. Checks the forward graph text matches the expected string for the four
        # combinations of dynamic and API version; the expected strings show the functional
        # triton wrapper with TMA descriptor metadata.
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        kernel_side_table.reset_table()
        kernel = (
            add_kernel_with_tma_1d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_1d_old_api
        )

        def f(a, b):
            # [FUNCTION DOC] HELPER: Zero output; builds three TMA descriptors with block 256;
            # launches the 1D TMA kernel on the descriptors; returns the output.
            BLOCK_SIZE = 256
            out = torch.zeros_like(a)
            n_elements = out.numel()

            desc_a, desc_b, desc_out = (
                create_tensor_descriptor_shim(
                    t, [BLOCK_SIZE], new_api=(tma_version == "new")
                )
                for t in (a, b, out)
            )

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            kernel[grid](
                desc_a,
                desc_b,
                desc_out,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out

        a = torch.randn(301, device=GPU_TYPE)
        b = torch.randn(301, device=GPU_TYPE)

        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()
        _ = f(a, b)
        torch.compile(
            f,
            fullgraph=True,
            backend=backend,
            dynamic=dynamic,
        )(a, b)

        if dynamic:
            if tma_version == "new":
                self.assertExpectedInline(
                    backend.fw_graphs[0].code.strip(),
                    """\
def forward(self, arg0_1, arg1_1, arg2_1):
    zeros_like = torch.ops.aten.zeros_like.default(arg1_1, pin_memory = False)
    add_2 = arg0_1 + 256;  arg0_1 = None
    sub_1 = add_2 - 1;  add_2 = None
    floordiv = sub_1 // 256;  sub_1 = None
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 0, grid = [(floordiv, 1, 1)], tma_descriptor_metadata = {'in_desc_ptr0': ('stable', ([256],)), 'in_desc_ptr1': ('stable', ([256],)), 'out_desc_ptr': ('stable', ([256],))}, kwargs = {'in_desc_ptr0': arg1_1, 'in_desc_ptr1': arg2_1, 'out_desc_ptr': zeros_like}, tensors_to_clone = ['out_desc_ptr']);  floordiv = arg1_1 = arg2_1 = zeros_like = None
    getitem = triton_kernel_wrapper_functional_proxy['out_desc_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return (getitem,)""",
                )
            elif tma_version == "old":
                self.assertExpectedInline(
                    backend.fw_graphs[0].code.strip(),
                    """\
def forward(self, arg0_1, arg1_1, arg2_1):
    zeros_like = torch.ops.aten.zeros_like.default(arg1_1, pin_memory = False)
    add_2 = arg0_1 + 256
    sub_1 = add_2 - 1;  add_2 = None
    floordiv = sub_1 // 256;  sub_1 = None
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 0, grid = [(floordiv, 1, 1)], tma_descriptor_metadata = {'in_desc_ptr0': ('experimental', ([arg0_1], [256], 4)), 'in_desc_ptr1': ('experimental', ([arg0_1], [256], 4)), 'out_desc_ptr': ('experimental', ([arg0_1], [256], 4))}, kwargs = {'in_desc_ptr0': arg1_1, 'in_desc_ptr1': arg2_1, 'out_desc_ptr': zeros_like}, tensors_to_clone = ['out_desc_ptr']);  floordiv = arg0_1 = arg1_1 = arg2_1 = zeros_like = None
    getitem = triton_kernel_wrapper_functional_proxy['out_desc_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return (getitem,)""",
                )
        else:
            if tma_version == "new":
                self.assertExpectedInline(
                    backend.fw_graphs[0].code.strip(),
                    """\
def forward(self, arg0_1, arg1_1):
    zeros_like = torch.ops.aten.zeros_like.default(arg0_1, pin_memory = False)
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 0, grid = [(2, 1, 1)], tma_descriptor_metadata = {'in_desc_ptr0': ('stable', ([256],)), 'in_desc_ptr1': ('stable', ([256],)), 'out_desc_ptr': ('stable', ([256],))}, kwargs = {'in_desc_ptr0': arg0_1, 'in_desc_ptr1': arg1_1, 'out_desc_ptr': zeros_like}, tensors_to_clone = ['out_desc_ptr']);  arg0_1 = arg1_1 = zeros_like = None
    getitem = triton_kernel_wrapper_functional_proxy['out_desc_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return (getitem,)""",
                )
            elif tma_version == "old":
                self.assertExpectedInline(
                    backend.fw_graphs[0].code.strip(),
                    """\
def forward(self, arg0_1, arg1_1):
    zeros_like = torch.ops.aten.zeros_like.default(arg0_1, pin_memory = False)
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 0, grid = [(2, 1, 1)], tma_descriptor_metadata = {'in_desc_ptr0': ('experimental', ([301], [256], 4)), 'in_desc_ptr1': ('experimental', ([301], [256], 4)), 'out_desc_ptr': ('experimental', ([301], [256], 4))}, kwargs = {'in_desc_ptr0': arg0_1, 'in_desc_ptr1': arg1_1, 'out_desc_ptr': zeros_like}, tensors_to_clone = ['out_desc_ptr']);  arg0_1 = arg1_1 = zeros_like = None
    getitem = triton_kernel_wrapper_functional_proxy['out_desc_ptr'];  triton_kernel_wrapper_functional_proxy = None
    return (getitem,)""",
                )

    @requires_gpu
    @common_utils.parametrize("after_data_ptr", [False, True])
    @common_utils.parametrize("after_create_desc", [False, True])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_tma_graph_breaks(self, after_data_ptr, after_create_desc, tma_version):
        # [FUNCTION DOC] Parametrized over graph-break placement and TMA version. Skips when
        # unsupported. Checks that graph breaks placed before descriptor creation, after it, or
        # both do not break correctness: compiled (eager backend, not fullgraph, static) equals
        # eager and the sum.
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_with_tma_1d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_1d_old_api
        )

        def f(a, b):
            # [FUNCTION DOC] HELPER: Zero output; optionally forces a graph break, creates three
            # descriptors, optionally forces another break, then launches the TMA kernel with the
            # descriptors; returns the output.
            BLOCK_SIZE = 256
            out = torch.zeros_like(a)
            n_elements = out.numel()

            if after_data_ptr:
                torch._dynamo.graph_break()

            descs = [
                create_tensor_descriptor_shim(
                    t, [BLOCK_SIZE], new_api=(tma_version == "new")
                )
                for t in (a, b, out)
            ]

            if after_create_desc:
                torch._dynamo.graph_break()

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            kernel[grid](
                *descs,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out

        a = torch.randn(301, device=GPU_TYPE)
        b = torch.randn(301, device=GPU_TYPE)

        expected_out = a + b
        eager_out = f(a, b)
        compiled_out = torch.compile(
            f,
            fullgraph=False,
            backend="eager",
            dynamic=False,
        )(a, b)

        self.assertEqual(eager_out, expected_out)
        self.assertEqual(compiled_out, expected_out)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_tma_descriptor_1d(self, dynamic, backend, tma_version):
        # [FUNCTION DOC] Parametrized over dynamic, backend and TMA version. Skips when
        # unsupported. Checks a kernel using host TMA descriptors for 1D tensors of 301 elements.
        # For the inductor backend it also captures the post-fusion IR debug log and uses
        # FileCheck to require a weak dependency between buffers three and four in the fourth
        # op's unmet dependencies.
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_with_tma_1d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_1d_old_api
        )

        def f(a, b):
            # [FUNCTION DOC] HELPER: Zero output; builds descriptors for the two inputs and the
            # output with block 256; launches the 1D TMA kernel; returns the output.
            BLOCK_SIZE = 256
            out = torch.zeros_like(a)
            n_elements = out.numel()

            desc_a, desc_b, desc_out = (
                create_tensor_descriptor_shim(
                    t, [BLOCK_SIZE], new_api=(tma_version == "new")
                )
                for t in (a, b, out)
            )

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            kernel[grid](
                desc_a,
                desc_b,
                desc_out,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out

        a = torch.randn(301, device=GPU_TYPE)
        b = torch.randn(301, device=GPU_TYPE)

        expected_out = a + b
        eager_out = f(a, b)

        if backend == "inductor":
            log_stream, ctx = logs_to_string("torch._inductor.debug", "ir_post_fusion")
            with ctx():
                compiled_out = torch.compile(
                    f,
                    fullgraph=True,
                    backend=backend,
                    dynamic=dynamic,
                )(a, b)

            FileCheck().check("op4.unmet_dependencies").check(
                "WeakDep(name='buf3', mutating_buf='buf4', is_fake=False)"
            ).run(log_stream.getvalue())
        else:
            compiled_out = torch.compile(
                f,
                fullgraph=True,
                backend=backend,
                dynamic=dynamic,
            )(a, b)

        self.assertEqual(eager_out, expected_out)
        self.assertEqual(compiled_out, expected_out)

    @requires_gpu
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_tma_descriptor_dedup(self, tma_version):
        # [FUNCTION DOC] Parametrized on TMA version; compiles with inductor and dynamic shapes.
        # Passes the same descriptor twice for both inputs; the sum must equal twice the input.
        # Checks the generated code creates exactly two descriptors (one deduplicated for the two
        # inputs, one for the output), counting the new-API from-tensor calls (only when not the
        # cpp wrapper or with compile-time autotuning) or the old-API 1D creation calls.
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_with_tma_1d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_1d_old_api
        )

        def f(a):
            # [FUNCTION DOC] HELPER: Zero output; creates descriptors for the input and the
            # output with block 256; launches the 1D TMA kernel with the input descriptor twice;
            # returns the output.
            BLOCK_SIZE = 256
            out = torch.zeros_like(a)
            n_elements = out.numel()

            desc_a, desc_out = (
                create_tensor_descriptor_shim(
                    t, [BLOCK_SIZE], new_api=(tma_version == "new")
                )
                for t in (a, out)
            )

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            kernel[grid](
                desc_a,
                desc_a,
                desc_out,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return out

        a = torch.randn(301, device=GPU_TYPE)

        expected_out = a + a
        eager_out = f(a)
        compiled_out, (code,) = run_and_get_code(
            torch.compile(
                f,
                fullgraph=True,
                backend="inductor",
                dynamic=True,
            ),
            a,
        )

        self.assertEqual(eager_out, expected_out)
        self.assertEqual(compiled_out, expected_out)

        # 2 calls: one for two inputs (dedupped), one for the output
        if tma_version == "new":
            if (
                not inductor_config.cpp_wrapper
                or inductor_config.triton.autotune_at_compile_time
            ):
                # Lazy kernel compilation implicitly calls TensorDescriptor.from_tensor
                # when calling run_triton_kernel_with_autotune
                self.assertEqual(code.count("TensorDescriptor.from_tensor("), 2)
        else:
            self.assertEqual(code.count("create_1d_tma_descriptor("), 2)

    @requires_gpu
    def test_wrap_tma_args_skips_constexpr(self):
        # [FUNCTION DOC] Skips unless the new tensor-descriptor host TMA API is available. Checks
        # the helper wrapping runtime arguments into TMA descriptors when the signature has
        # constexpr entries before and after the descriptor params: it uses a mock kernel whose
        # signature lists a constexpr then three tensor descriptor entries then a constexpr,
        # passes three tensors, and requires three descriptor objects in the result, so the index
        # offset for skipped constexprs is correct.
        if not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")

        from unittest.mock import Mock

        from triton.tools.tensor_descriptor import TensorDescriptor

        from torch._inductor.runtime.triton_lazy_compile import _wrap_tma_args

        # Simulate a signature where constexpr entries appear before TMA
        # descriptor params. The signature includes constexpr entries but
        # runtime args exclude them, so _wrap_tma_args must account for the
        # index offset.
        kernel_fn = Mock()
        kernel_fn.triton_meta = {
            "signature": {
                "ADD_ALPHA": "constexpr",
                "in_desc_ptr0": "tensordesc<fp32[256]>",
                "in_desc_ptr1": "tensordesc<fp32[256]>",
                "out_desc_ptr": "tensordesc<fp32[256]>",
                "BLOCK_SIZE": "constexpr",
            }
        }

        t0 = torch.randn(256, device=GPU_TYPE)
        t1 = torch.randn(256, device=GPU_TYPE)
        t2 = torch.zeros(256, device=GPU_TYPE)
        args = [t0, t1, t2]

        wrapped = _wrap_tma_args(args, kernel_fn)

        self.assertEqual(len(wrapped), 3)
        self.assertIsInstance(wrapped[0], TensorDescriptor)
        self.assertIsInstance(wrapped[1], TensorDescriptor)
        self.assertIsInstance(wrapped[2], TensorDescriptor)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("backend", ["eager", "aot_eager"])
    @common_utils.parametrize("tma_version", ["new", "old"])
    def test_tma_descriptor_2d(self, dynamic, backend, tma_version):
        # [FUNCTION DOC] Parametrized over dynamic, backend (eager, aot_eager) and TMA version;
        # skips when unsupported. Launches a 2D TMA add kernel on 25 by 16 tensors with
        # descriptors of block sizes 16 and 32, over a 2D grid; eager and compiled results equal
        # the sum.
        if tma_version == "new" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor TMA support")
        if tma_version == "old" and not has_triton_experimental_host_tma():
            self.skipTest("requires triton.tools.experimental_descriptor TMA support")

        kernel = (
            add_kernel_with_tma_2d_new_api
            if tma_version == "new"
            else add_kernel_with_tma_2d_old_api
        )

        def f(a, b):
            # [FUNCTION DOC] HELPER: Zero output; creates three 2D descriptors; launches the 2D
            # TMA kernel with a grid from the output size; returns the output.
            BLOCK_SIZE_X = 16
            BLOCK_SIZE_Y = 32
            out = torch.zeros_like(a)
            x_size, y_size = out.size()

            desc_a, desc_b, desc_out = (
                create_tensor_descriptor_shim(
                    t, [BLOCK_SIZE_X, BLOCK_SIZE_Y], new_api=(tma_version == "new")
                )
                for t in (a, b, out)
            )

            grid = lambda meta: (
                triton.cdiv(x_size, meta["BLOCK_SIZE_X"]),
                triton.cdiv(y_size, meta["BLOCK_SIZE_Y"]),
            )
            kernel[grid](
                desc_a,
                desc_b,
                desc_out,
                BLOCK_SIZE_X=BLOCK_SIZE_X,
                BLOCK_SIZE_Y=BLOCK_SIZE_Y,
            )

            return out

        a = torch.randn((25, 16), device=GPU_TYPE)
        b = torch.randn((25, 16), device=GPU_TYPE)

        expected_out = a + b
        eager_out = f(a, b)
        compiled_out = torch.compile(
            f,
            fullgraph=True,
            backend=backend,
            dynamic=dynamic,
        )(a, b)

        self.assertEqual(eager_out, expected_out)
        self.assertEqual(compiled_out, expected_out)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_num_ctas(self, backend):
        # [FUNCTION DOC] Parametrized on backend. Checks that passing num_ctas directly to a
        # Triton kernel launch, either by index call or the run method, raises the dynamo
        # Unsupported error with the message to use an autotune Config instead.
        # [FUNCTION DOC] HELPER: Triton kernel with one parameter and an empty body.
        @triton.jit
        def kernel(X):
            return

        @torch.compile(fullgraph=True, backend=backend)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled fullgraph function that launches the kernel with
            # num_ctas set via index call and via the run method, then returns x.
            kernel[(1,)](x, num_ctas=1)
            kernel.run(x, num_ctas=1, grid=(1,), warmup=False)
            return x

        msg = "Passing num_ctas directly to the Triton kernel is not supported. Please use a Config in @triton.autotune instead."
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            x = torch.randn(4, device=GPU_TYPE)
            f(x)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_special_kwargs_without_autotune(self, backend):
        # [FUNCTION DOC] Parametrized on backend. Checks a plain (not
        # autotuned) kernel accepts the warps and stages launch kwargs with
        # fullgraph compile. Only requires the call to succeed.
        # [FUNCTION DOC] HELPER: Masked elementwise add kernel.
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.compile(fullgraph=True, backend=backend)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Compiled fullgraph function launching the add kernel with
            # block size 128, warps 8 and stages 3; returns the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](
                x,
                y,
                output,
                n_elements,
                BLOCK_SIZE=128,
                num_warps=8,
                num_stages=3,
            )
            return output

        x = torch.randn(4, device=GPU_TYPE)
        f(x, x)

    @requires_gpu
    def test_triton_kernel_backend_options_without_signature(self):
        # [FUNCTION DOC] Checks that a launch kwarg that is not in the kernel signature is
        # preserved as a backend compile option. Builds an add kernel without such a parameter,
        # runs it launching with a false fp fusion kwarg, and asserts the sum, and that a compile
        # call captured the option false.
        # A launch kwarg that is not in the kernel signature should be kept as
        # a backend option and emitted in the generated Triton metadata.
        add_kernel = _get_backend_options_kernel(with_enable_fp_fusion=False)
        f = _get_backend_options_fn(add_kernel, BLOCK_SIZE=128, enable_fp_fusion=False)

        x = torch.randn(4, device=GPU_TYPE)
        out, _, options = self._run_and_get_triton_compile_options(f, x, x)
        self.assertEqual(out, x + x)
        self._assert_triton_compile_option(options, "enable_fp_fusion", False)

    @requires_gpu
    def test_triton_kernel_invalid_backend_options_error(self):
        # [FUNCTION DOC] Checks that launch kwargs which are neither kernel parameters nor valid
        # backend options produce a RuntimeError at compile time with the message about launch
        # kwargs; dynamo accepts them but inductor validates once the backend is known.
        # Dynamo leaves the determination of the target triton backend to
        # inductor, so it will accept constant non-kernel arguments provided
        # to a user defined triton kernels launch. Inductor will then validate
        # backend compiler options once the target is known:
        # every launch kwarg must be either a kernel parameter or a
        # backend option accepted by backend.parse_options(), matching direct
        # Triton's _pack_args check.
        add_kernel = _get_backend_options_kernel(with_enable_fp_fusion=False)
        f = _get_backend_options_fn(
            add_kernel, BLOCK_SIZE=128, not_a_backend_option=False
        )

        x = torch.randn(4, device=GPU_TYPE)
        msg = "Triton launch kwargs must be kernel parameters or valid backend options"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.compile(f, fullgraph=True)(x, x)

    @requires_gpu
    def test_triton_kernel_backend_options_also_used_as_kernel_kwarg(self):
        # [FUNCTION DOC] Checks the dual-meaning launch kwarg: a kernel with a constexpr
        # parameter named like a backend option, launched with that name false. The eager-backend
        # result must be the product (the kernel branch), and the inductor result must be the
        # product too and the compile options must also include the option as false.
        # For `def kernel(..., enable_fp_fusion: tl.constexpr)`, the call
        # `kernel[grid](..., enable_fp_fusion=False)` has two meanings in
        # direct Triton: the binder uses it as a kernel parameter, and
        # _pack_args still exposes it to backend.parse_options(), so Triton
        # also parses it as a compiler option. Preserve both meanings without
        # replaying it as the illegal form
        # `kernel[grid](..., False, enable_fp_fusion=False)`, which passes the
        # same kernel parameter both positionally and by keyword.
        add_kernel = _get_backend_options_kernel(with_enable_fp_fusion=True)
        f = _get_backend_options_fn(add_kernel, BLOCK_SIZE=128, enable_fp_fusion=False)

        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        eager_backend_out = torch.compile(f, backend="eager", fullgraph=True)(x, y)
        self.assertEqual(eager_backend_out, x * y)

        out, _, options = self._run_and_get_triton_compile_options(f, x, y)
        self.assertEqual(out, x * y)
        self._assert_triton_compile_option(options, "enable_fp_fusion", False)

    @requires_gpu
    def test_triton_kernel_backend_options_also_kernel_kwarg_with_autotune(self):
        # [FUNCTION DOC] Same dual-meaning kwarg as the previous test but through a user
        # autotuner with one config. Compares eager-backend and inductor results to the product,
        # requires the generated code to contain the user autotune decorator text, and the
        # compile options to include the option as false.
        # Same dual-meaning kwarg case as above, but through a user-provided
        # triton.Config. The launch kwarg should remain both a kernel
        # parameter and a backend option after the autotune wrapper is applied.
        add_kernel = triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 128}, num_warps=4)],
            key=[],
        )(_get_backend_options_kernel(with_enable_fp_fusion=True))
        f = _get_backend_options_fn(add_kernel, enable_fp_fusion=False)

        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        eager_backend_out = torch.compile(f, backend="eager", fullgraph=True)(x, y)
        self.assertEqual(eager_backend_out, x * y)

        out, (code,), options = self._run_and_get_triton_compile_options(f, x, y)
        self.assertEqual(out, x * y)
        self.assertIn("@triton_heuristics.user_autotune", code)
        self._assert_triton_compile_option(options, "enable_fp_fusion", False)

    @requires_gpu
    def test_triton_kernel_autotune_config_conflicts_with_launch_kwarg(self):
        # [FUNCTION DOC] Checks that a launch kwarg also present in an autotune config raises
        # Unsupported with the conflict message, both with the eager backend and with the default
        # inductor backend. The autotune has two configs, one of which sets the same key.
        # Direct Triton errors when a generic autotune config kwarg is also
        # passed as a launch kwarg. The config does not override the launch
        # kwarg; it would create duplicate keyword arguments at launch.
        add_kernel = triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
                triton.Config(
                    {"BLOCK_SIZE": 128, "enable_fp_fusion": True}, num_warps=4
                ),
            ],
            key=[],
        )(_get_backend_options_kernel(with_enable_fp_fusion=True))
        f = _get_backend_options_fn(add_kernel, enable_fp_fusion=False)

        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        msg = "Triton launch kwargs conflict with autotune config kwargs"
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            torch.compile(f, backend="eager", fullgraph=True)(x, y)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            torch.compile(f, fullgraph=True)(x, y)

    @requires_gpu
    def test_triton_kernel_backend_options_with_autotune(self):
        # [FUNCTION DOC] Checks that backend options passed at launch survive the user autotune
        # wrapper: the result equals the sum, the generated code contains the user autotune
        # decorator, and the compile options include fp fusion false.
        # Backend options passed at launch should survive through the
        # user-defined Triton autotune wrapper as compile metadata.
        add_kernel = triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 128}, num_warps=4)],
            key=[],
        )(_get_backend_options_kernel(with_enable_fp_fusion=False))
        f = _get_backend_options_fn(add_kernel, enable_fp_fusion=False)

        x = torch.randn(4, device=GPU_TYPE)
        out, (code,), options = self._run_and_get_triton_compile_options(f, x, x)
        self.assertEqual(out, x + x)
        self.assertIn("@triton_heuristics.user_autotune", code)
        self._assert_triton_compile_option(options, "enable_fp_fusion", False)

    @requires_gpu
    def test_triton_kernel_backend_options_preserved_in_fx_wrapper_hop(self):
        # [FUNCTION DOC] Checks that with the FX wrapper codegen enabled (single compile thread)
        # the recorded Triton kernel mutation higher-order-op node keeps the backend option name
        # in its launch kwargs and the option value in the combined kwargs and constant args, so
        # a direct run compiles the same kernel. Patches the FX converter generate method to
        # record each generated graph module.
        add_kernel = _get_backend_options_kernel(with_enable_fp_fusion=False)
        f = _get_backend_options_fn(add_kernel, BLOCK_SIZE=128, enable_fp_fusion=False)

        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch._inductor.codegen.wrapper_fxir import FxConverter

        captured_gms = []
        orig_generate = FxConverter.generate

        def record_generate(self):
            # [FUNCTION DOC] HELPER: Replacement for the converter generate method; calls the
            # original, records the resulting graph module in a list and returns it.
            gm = orig_generate(self)
            captured_gms.append(gm)
            return gm

        x = torch.randn(4, device=GPU_TYPE)
        with (
            inductor_config.patch({"fx_wrapper": True, "compile_threads": 1}),
            mock.patch.object(FxConverter, "generate", record_generate),
        ):
            out = torch.compile(f, fullgraph=True)(x, x)

        self.assertEqual(out, x + x)
        self.assertTrue(captured_gms)
        hop_node = next(
            node
            for node in captured_gms[0].graph.nodes
            if node.op == "call_function"
            and node.target is triton_kernel_wrapper_mutation
            and "enable_fp_fusion" in node.kwargs.get("launch_kwargs", ())
        )
        constant_args = kernel_side_table.get_constant_args(
            hop_node.kwargs["constant_args_idx"]
        )
        hop_kwargs = {**hop_node.kwargs["kwargs"], **constant_args}

        # The FX wrapper runs the HOP directly, so names alone are not enough:
        # non-signature backend options also need their values in the HOP
        # payload. Otherwise the direct FXIR run re-enters Triton JIT with
        # default compiler options and may compile a different kernel.
        self.assertEqual(hop_node.kwargs["launch_kwargs"], ("enable_fp_fusion",))
        self.assertEqual(hop_kwargs["enable_fp_fusion"], False)

    @requires_gpu
    def test_triton_kernel_special_kwargs_with_autotune_use_configs(self):
        # [FUNCTION DOC] Checks that the special launch names for warps and stages update the
        # autotune configs rather than being kept as backend options: with launch warps 8 the
        # generated code contains the warps entry 8 and no backend options text; result equals
        # the sum.
        # SPECIAL_CONFIG_NAMES keep the existing behavior: they update Triton
        # configs instead of being preserved as generic backend options.
        add_kernel = triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 128}, num_warps=4)],
            key=[],
        )(_get_backend_options_kernel(with_enable_fp_fusion=False))
        f = _get_backend_options_fn(add_kernel, num_warps=8)

        x = torch.randn(4, device=GPU_TYPE)
        out, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x, x)
        self.assertEqual(out, x + x)
        self.assertIn("'num_warps': 8", code)
        self.assertNotIn("backend_options", code)

    @requires_gpu
    def test_capture_triton_backend_options_preserve_launch_kwarg_names(self):
        # [FUNCTION DOC] Traces with make_fx in symbolic mode a capture_triton launch with
        # keyword args including a non-signature backend option, and checks the higher-order-op
        # node launch kwargs equals the set of the three keyword names (element count, block
        # size, option).
        # [FUNCTION DOC] HELPER: Add kernel with an empty body used only for tracing.
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pass

        def f(x, y):
            # [FUNCTION DOC] HELPER: Empty output; launches the captured
            # kernel with positional tensors and keyword element count, block
            # size and fp fusion false; returns the output.
            output = torch.empty_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            capture_triton(add_kernel)[grid](
                x,
                y,
                output,
                n_elements=n_elements,
                BLOCK_SIZE=16,
                enable_fp_fusion=False,
            )
            return output

        from torch.fx.experimental.proxy_tensor import make_fx

        x = torch.randn(4, device=GPU_TYPE)
        y = torch.randn(4, device=GPU_TYPE)
        gm = make_fx(f, tracing_mode="symbolic")(x, y)
        hop_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is triton_kernel_wrapper_mutation
        )

        # capture_triton preserves the original launch kwarg names separately
        # from the normalized argument values. Kernel args are present here too
        # because names such as enable_fp_fusion may have direct Triton dual
        # meaning as both a kernel parameter and a compiler option. Inductor
        # later recovers values by name and filters them with the target
        # backend before serializing triton_meta.
        self.assertEqual(
            set(hop_node.kwargs["launch_kwargs"]),
            {
                "n_elements",
                "BLOCK_SIZE",
                "enable_fp_fusion",
            },
        )

    @requires_gpu
    def test_capture_triton_backend_options_also_kernel_kwarg(self):
        # [FUNCTION DOC] Traces a capture_triton launch where the kernel has a constexpr
        # parameter with the same name as a backend option, and checks the node's launch kwargs
        # keep both launch-site names (block size and the option).
        # [FUNCTION DOC] HELPER: Kernel with an empty body and constexpr parameters including one
        # named like a backend option.
        @triton.jit
        def add_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
            enable_fp_fusion: "tl.constexpr",
        ):
            pass

        def f(x):
            # [FUNCTION DOC] HELPER: Empty output; launches the captured kernel with block size
            # 16 and the option false; returns the output.
            output = torch.empty_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            capture_triton(add_kernel)[grid](
                x,
                output,
                n_elements,
                BLOCK_SIZE=16,
                enable_fp_fusion=False,
            )
            return output

        from torch.fx.experimental.proxy_tensor import make_fx

        gm = make_fx(f)(torch.randn(4, device=GPU_TYPE))
        hop_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is triton_kernel_wrapper_mutation
        )

        # Direct Triton gives a kwarg such as enable_fp_fusion two meanings
        # when the kernel also has a parameter with that name: it binds the
        # tl.constexpr parameter and backend.parse_options() still sees it as
        # a backend option. The tracing path should preserve the launch-site
        # name instead of dropping it just because it is also a kernel arg.
        self.assertEqual(
            hop_node.kwargs["launch_kwargs"],
            ("BLOCK_SIZE", "enable_fp_fusion"),
        )

    @requires_gpu
    def test_capture_triton_autotune_config_does_not_materialize_default_args(self):
        # [FUNCTION DOC] Checks that tracing a capture_triton launch of an autotuned kernel with
        # a default-valued constexpr parameter keeps the omitted default out of the node kwargs
        # and constant args; the grid is the single program computed from the config's block
        # size, and no launch kwargs are recorded.
        # [FUNCTION DOC] HELPER: Autotuned (one config, block 16) kernel with an empty body and a
        # constexpr block size defaulting to two.
        @triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 16}, num_warps=4)],
            key=[],
        )
        @triton.jit
        def add_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr" = 2,
        ):
            pass

        def f(x):
            # [FUNCTION DOC] HELPER: Empty output; launches the captured kernel without block
            # size; returns the output.
            output = torch.empty_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            capture_triton(add_kernel)[grid](
                x,
                output,
                n_elements,
            )
            return output

        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch.fx.experimental.proxy_tensor import make_fx

        gm = make_fx(f)(torch.randn(4, device=GPU_TYPE))
        hop_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is triton_kernel_wrapper_mutation
        )

        # Omitted Python defaults should stay omitted from the HOP payload.
        # The autotune config still provides BLOCK_SIZE to grid(meta), giving a
        # single-program grid for four elements. This prevents the default
        # BLOCK_SIZE=2 from being recorded as an explicit constant argument and
        # competing with the config's BLOCK_SIZE=16 downstream.
        constant_args = kernel_side_table.get_constant_args(
            hop_node.kwargs["constant_args_idx"]
        )
        self.assertEqual(
            set(hop_node.kwargs["kwargs"]), {"in_ptr0", "out_ptr", "n_elements"}
        )
        self.assertNotIn("BLOCK_SIZE", hop_node.kwargs["kwargs"])
        self.assertNotIn("BLOCK_SIZE", constant_args)
        self.assertEqual(hop_node.kwargs["grid"], [(1, 1, 1)])
        self.assertNotIn("launch_kwargs", hop_node.kwargs)

    @requires_gpu
    def test_capture_triton_dynamic_backend_options_error(self):
        # [FUNCTION DOC] Checks that make_fx in symbolic mode raises a RuntimeError saying Triton
        # backend options must be concrete values when a non-signature launch kwarg (a register
        # limit) is a symbolic size.
        # [FUNCTION DOC] HELPER: Kernel with an empty body.
        @triton.jit
        def add_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pass

        def f(x):
            # [FUNCTION DOC] HELPER: Empty output; launches the captured kernel with a register-
            # limit kwarg equal to the element count; returns the output.
            output = torch.empty_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            capture_triton(add_kernel)[grid](
                in_ptr0=x,
                out_ptr=output,
                n_elements=n_elements,
                BLOCK_SIZE=16,
                maxnreg=n_elements,
            )
            return output

        from torch.fx.experimental.proxy_tensor import make_fx

        # Unlike n_elements above, maxnreg is not a kernel parameter. If it
        # contains a symbolic value, it cannot be represented as a concrete
        # compile-time backend option.
        with self.assertRaisesRegex(
            RuntimeError, "Triton backend options must be concrete values"
        ):
            make_fx(f, tracing_mode="symbolic")(torch.randn(4, device=GPU_TYPE))

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("autotune_at_compile_time", [True, False])
    def test_triton_kernel_restore_value(self, backend, autotune_at_compile_time):
        # [FUNCTION DOC] Parametrized over backend and compile-time autotuning (only meaningful
        # for inductor, else skipped). Checks restore_value on an autotuned in-place increment
        # kernel: after one compiled call the input must equal the original plus exactly one,
        # i.e. the buffer was restored after benchmarking configs.
        if autotune_at_compile_time and backend != "inductor":
            raise unittest.SkipTest("compile-time autotuning only exists in inductor")

        # [FUNCTION DOC] HELPER: Autotuned (two configs, restore_value for the input pointer)
        # kernel that adds one to each masked element in place.
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 16}, num_stages=3, num_warps=8),
                triton.Config({"BLOCK_SIZE": 32}, num_stages=3, num_warps=8),
            ],
            key=[],
            restore_value=["in_ptr0"],
        )
        @triton.jit
        def increment_kernel(
            in_ptr0,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            output = x + 1
            tl.store(in_ptr0 + offsets, output, mask=mask)

        @torch.compile(fullgraph=True, backend=backend)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled fullgraph function launching the increment kernel
            # on x; returns x.
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            increment_kernel[grid](x, n_elements=n_elements)
            return x

        x = torch.rand(4, device=GPU_TYPE)
        prev = x.clone()

        with inductor_config.patch(
            {"triton.autotune_at_compile_time": autotune_at_compile_time}
        ):
            f(x)

        # make sure x was restored after autotuning
        torch.testing.assert_close(x, prev + 1)

    @requires_gpu
    @parametrize("dtype", (torch.float16, torch.float32, torch.float64))
    def test_triton_kernel_float64_constant(self, dtype):
        # [FUNCTION DOC] Parametrized on float dtype. Checks that a Python float constant
        # combined with a dynamic size multiplies correctly in half, single and double precision:
        # compiled dynamic result equals eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Multiplies x by 0.12 times its first dimension size.
            return x * (0.12 * x.shape[0])

        x = torch.ones(200, device=GPU_TYPE, dtype=dtype)

        eager_out = f(x)
        compiled_out = torch.compile(f, dynamic=True)(x)
        self.assertEqual(compiled_out, eager_out)

    # TODO enable this test case on XPU.
    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180126")
    @requires_gpu_and_triton
    @parametrize("cfg", ["normal", "cpp_wrapper"])
    def test_triton_kernel_dtype_view(self, cfg):
        # [FUNCTION DOC] Parametrized on normal or cpp wrapper config (skipped on ROCm for issue
        # 180126). Regression for issue 136159: a Triton kernel writes ones into a half-precision
        # buffer through its bfloat16 view. Both eager and compiled outputs must have the two
        # tensors sharing a data pointer and equal values.
        # https://github.com/pytorch/pytorch/issues/136159
        if cfg == "normal":
            config_kwargs = {"cpp_wrapper": False}
        elif cfg == "cpp_wrapper":
            config_kwargs = {"cpp_wrapper": True}

        with inductor_config.patch(**config_kwargs):

            # [FUNCTION DOC] HELPER: Kernel storing the half-
            # precision constant one for masked offsets.
            @triton.jit
            def _triton_kernel(out_ptr, numel, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(0)
                offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
                mask = offsets < numel
                ones = tl.full((BLOCK_SIZE,), 1, tl.float16)
                tl.store(out_ptr + offsets, ones, mask)

            def fn(x):
                # [FUNCTION DOC] HELPER: Allocates an empty half buffer, makes a bfloat16 view of
                # it, launches the kernel on the view, returns both tensors.
                buf = torch.empty(x.shape, device=x.device, dtype=torch.float16)
                # the buf.view() should be a view sharing the same storage as buf.
                bfloat_buf = buf.view(dtype=torch.bfloat16)
                BLOCK_SIZE = 256
                numel = buf.numel()
                grid = (triton.cdiv(numel, BLOCK_SIZE),)
                _triton_kernel[grid](bfloat_buf, numel, BLOCK_SIZE)
                return buf, bfloat_buf

            fn_c = torch.compile(fn)

            x = torch.randn(8, device=GPU_TYPE)
            out_c = fn_c(x)
            out_e = fn(x)

            # expect view() to be an actual view, sharing the same data as the original buffer
            # verify first that this is true in the eager output
            self.assertEqual(out_e[0].data_ptr(), out_e[1].data_ptr())
            # .. and also in the compiled output
            self.assertEqual(out_c[0].data_ptr(), out_c[1].data_ptr())

            self.assertEqual(out_e[0], out_c[0])
            self.assertEqual(out_e[1], out_c[1])

    # TODO enable this test case on XPU.
    @requires_gpu
    def test_i64_input(self):
        # [FUNCTION DOC] Checks a 64-bit integer scalar argument (a seed from a random int item,
        # captured as a scalar output with a dynamic input) is typed as 64 bit in the signature,
        # with fullgraph compile. Result values (add random noise to a squared uniform input)
        # must lie in the unit-to-two range.
        # The i64 "seed" input needs to be marked as "i64", not "i32".
        # [FUNCTION DOC] HELPER: Kernel adding uniform noise generated from a seed and offsets to
        # each masked input and storing the result.
        @triton.jit
        def triton_add_noise_(x_ptr, y_ptr, seed, numel, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

            x = tl.load(x_ptr + offsets, mask=(offsets < numel))
            rnd = tl.rand(seed, offsets)
            res = x + rnd
            tl.store(y_ptr + offsets, res, mask=(offsets < numel))

        def add_noise(x, seed):
            # [FUNCTION DOC] HELPER: Empty output; launches the noise kernel with a grid by cdiv
            # and block 256; returns the output.
            y = torch.empty_like(x)
            numel = x.numel()
            BLOCK_SIZE = 256

            def grid(meta):
                # [FUNCTION DOC] HELPER: Returns a one-element grid by dividing the element count
                # by the block size meta parameter.
                return (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

            triton_add_noise_[grid](x, y, seed, numel, BLOCK_SIZE)
            return y

        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x, draws a 64 bit integer seed in a large range as a
            # Python scalar, and returns the noise-added result.
            x = x * x
            seed = torch.randint(
                low=2**32, high=2**62, size=(1,), dtype=torch.int64
            ).item()
            return add_noise(x, seed)

        inp = torch.rand(400, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(inp, 0)

        fn_c = torch.compile(fn, fullgraph=True)
        with dynamo_config.patch(capture_scalar_outputs=True):
            res = fn_c(inp)

        self.assertTrue(((res < 2) & (res >= 0)).all().item())

    @requires_gpu
    @parametrize("wrapped", [False, True])
    @parametrize("autotune", [False, True])
    def test_constexpr_dynamic_shapes(self, wrapped, autotune):
        # [FUNCTION DOC] Parametrized over wrapped (capture and triton op) and autotune.
        # Regression for issue 136504: constexpr arguments derived from dynamic sizes (the
        # element count and its parity). Runs the compiled function on sizes 517 and 1029 and
        # requires the result to equal x squared.
        # https://github.com/pytorch/pytorch/issues/136504
        # [FUNCTION DOC] HELPER: Kernel squaring each masked element and adding one when the odd
        # constexpr is true.
        @triton.jit
        def triton_(
            x_ptr,
            y_ptr,
            NUMEL: tl.constexpr,
            IS_ODD: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
            mask = offsets < NUMEL

            data = tl.load(x_ptr + offsets, mask)
            result = data * data
            if IS_ODD:
                result = result + 1

            tl.store(y_ptr + offsets, result, mask)

        if autotune:
            triton_ = triton.autotune(
                [
                    triton.Config(kwargs={"BLOCK_SIZE": 128}),
                    triton.Config(kwargs={"BLOCK_SIZE": 256}),
                ],
                key=[],
            )(triton_)

        def triton_kernel_impl(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Builds args from the input, output, element count and odd
            # flag (plus block 256 when not autotuned), and launches the kernel directly or
            # through capture; returns the output.
            y = torch.empty_like(x)
            numel = x.numel()

            args = [x, y, numel, numel % 2 == 0]
            if not autotune:
                args.append(256)  # BLOCK_SIZE

            def grid(meta):
                # [FUNCTION DOC] HELPER: Returns a one-element grid by dividing the element count
                # by the block size meta parameter.
                return (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

            if wrapped:
                capture_triton(triton_)[grid](*args)
            else:
                triton_[grid](*args)
            return y

        if wrapped:
            triton_kernel = torch.library.triton_op(
                "constexpr_test::square", triton_kernel_impl, mutates_args={}
            )
        else:
            triton_kernel = triton_kernel_impl

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the kernel implementation (possibly a registered
            # triton op) on x.
            return triton_kernel(x)

        fn_c = torch.compile(fn, dynamic=True)

        x = torch.randn(512 + 5, device=GPU_TYPE)
        res = fn_c(x)
        self.assertEqual(x * x, res)

        x2 = torch.randn(1024 + 5, device=GPU_TYPE)
        res2 = fn_c(x2)
        self.assertEqual(x2 * x2, res2)

    @requires_gpu
    def test_triton_kernel_none_args(self):
        # [FUNCTION DOC] Regression for issue 115344: passing None for a pointer argument to an
        # autotuned kernel works with fullgraph compile. Runs the kernel with a real tensor and
        # then with none and requires compiled outputs to equal eager.
        # https://github.com/pytorch/pytorch/issues/115344
        # [FUNCTION DOC] HELPER: Autotuned (two configs) kernel that loads the input if given,
        # otherwise uses zero, then stores the sine.
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 32}, num_stages=5, num_warps=2),
                triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def sin_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            if in_ptr0 is not None:
                x = tl.load(in_ptr0 + offsets, mask=mask)
            else:
                x = 0.0
            output = tl.sin(x)
            tl.store(out_ptr + offsets, output, mask=mask)

        def sin_triton(x, out):
            # [FUNCTION DOC] HELPER: Launches the sine kernel with one program per element.
            n_elements = out.numel()
            sin_kernel[(n_elements,)](x, out, n_elements)

        x = torch.randn(65, device=GPU_TYPE)
        out = torch.empty_like(x)
        out_compiled = torch.empty_like(x)
        sin_triton_compiled = torch.compile(fullgraph=True)(sin_triton)

        sin_triton(x, out)
        sin_triton_compiled(x, out_compiled)
        self.assertEqual(out, out_compiled)

        sin_triton(None, out)
        sin_triton_compiled(None, out_compiled)
        self.assertEqual(out, out_compiled)

    @requires_gpu
    def test_triton_kernel_global_constexpr(self):
        # [FUNCTION DOC] Checks a kernel using a global constexpr float that is not type-
        # annotated at module level: asserts it is absent from the module annotations while the
        # string constant is present, then compares compiled output to x plus 3.14.
        # [FUNCTION DOC] HELPER: Kernel adding the global float constexpr to each element.
        @triton.jit
        def triton_(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            x = tl.load(in_ptr + offsets)
            output = x + FLOAT_CONSTANT_C
            tl.store(out_ptr + offsets, output)

        def fn(x):
            # [FUNCTION DOC] HELPER: Empty output; launches the kernel with block 256; returns
            # the output.
            y = torch.empty_like(x)
            BLOCK_SIZE = 256
            grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
            triton_[grid](x, y, BLOCK_SIZE)
            return y

        # make sure FLOAT_CONSTANT_C is NOT annotated
        self.assertFalse("FLOAT_CONSTANT_C" in sys.modules[__name__].__annotations__)
        # sanity check: STRING_CONSTANT_C _should_ be annotated
        self.assertTrue("STRING_CONSTANT_C" in sys.modules[__name__].__annotations__)

        x = torch.randn(512, device=GPU_TYPE)
        expected = x + 3.14
        actual = torch.compile(fn)(x)
        self.assertEqual(expected, actual)

    @requires_gpu
    def test_triton_kernel_constexpr_arg(self):
        # [FUNCTION DOC] Checks that a global constexpr integer can be passed as a constexpr
        # kernel argument that selects a branch (absolute value or square). Compiled fullgraph
        # output must equal eager.
        # [FUNCTION DOC] HELPER: Kernel storing either the absolute value or the square of each
        # masked element depending on the mode constexpr.
        @triton.jit
        def kernel(in_ptr, out_ptr, n, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n
            x = tl.load(in_ptr + offsets, mask=mask)
            if MODE == 0:
                out = tl.abs(x)
            else:
                out = x * x
            tl.store(out_ptr + offsets, out, mask=mask)

        def fn(x):
            # [FUNCTION DOC] HELPER: Empty output; launches the kernel with the mode set to the
            # global constexpr and block 1024; returns the output.
            out = torch.empty_like(x)
            n = x.numel()
            grid = (triton.cdiv(n, 1024),)
            kernel[grid](x, out, n, MODE=CONSTANT_C, BLOCK_SIZE=1024)
            return out

        x = torch.randn(2048, device=GPU_TYPE)
        expected = fn(x)
        actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(expected, actual)

    @requires_gpu
    @unittest.skipIf(
        not triton_version_uses_attrs_dict(),
        "Test is only valid for new triton versions where attrs is represented by a raw dict",
    )
    def test_triton_attrs_dict_equal_1_None_format(self):
        # [FUNCTION DOC] Only runs when triton represents attributes as a raw dict. Checks that
        # for a kernel called with a None argument and a one-element input, the generated triton
        # metadata lists the element count and the None value among constants, and the signature
        # marks the count, the None argument and block size as constexpr. Uses FileCheck on the
        # generated code; the compiled output must equal eager.
        # [FUNCTION DOC] HELPER: Kernel squaring the input and optionally adding an amount when
        # it is not None.
        @triton.jit
        def triton_(in_ptr, out_ptr, numel, add_amount, BLOCK_SIZE: tl.constexpr):
            offsets = tl.arange(0, BLOCK_SIZE)
            x = tl.load(in_ptr + offsets, mask=(offsets < numel))
            output = x * x
            if add_amount is not None:
                output = output + add_amount
            tl.store(out_ptr + offsets, output, mask=(offsets < numel))

        def fn(x):
            # [FUNCTION DOC] HELPER: Empty output; launches the kernel with a single program, the
            # element count, None as the amount and block 256; returns the output.
            y = torch.empty_like(x)
            BLOCK_SIZE = 256
            grid = (1,)
            triton_[grid](x, y, x.numel(), None, BLOCK_SIZE)
            return y

        x = torch.full((1,), 2.5, device=GPU_TYPE)
        expected = fn(x)

        fn_c = torch.compile(fn)
        res, code = run_and_get_code(fn_c, x)
        self.assertEqual(expected, res)

        FileCheck().check("triton_meta=").check("'constants':").check("'numel': 1").run(
            code[0]
        )
        FileCheck().check("triton_meta=").check("'constants':").check(
            "'add_amount': None"
        ).run(code[0])
        FileCheck().check("triton_meta=").check("'constants':").check(
            "'BLOCK_SIZE': 256"
        ).run(code[0])

        FileCheck().check("triton_meta=").check("'signature':").check(
            "'numel': 'constexpr'"
        ).run(code[0])
        FileCheck().check("triton_meta=").check("'signature':").check(
            "'add_amount': 'constexpr'"
        ).run(code[0])
        FileCheck().check("triton_meta=").check("'signature':").check(
            "'BLOCK_SIZE': 'constexpr'"
        ).run(code[0])

    @requires_gpu
    @inductor_config.patch({"triton.autotune_at_compile_time": True})
    @parametrize("quotes", ["single", "double"])
    def test_kernel_with_docstring(self, quotes):
        # [FUNCTION DOC] Parametrized on single or double quotes in the kernel's docstring; runs
        # with compile-time autotuning on. Regression for issue 155006: kernels with docstrings
        # must compile. Compares eager and compiled outputs for size 345.
        kernel = (
            kernel_with_docstring_single_quotes
            if quotes == "single"
            else kernel_with_docstring_double_quotes
        )

        # https://github.com/pytorch/pytorch/issues/155006
        def fn(sz):
            # [FUNCTION DOC] HELPER: Allocates an empty tensor of the given size and launches the
            # docstring kernel over a grid by cdiv with block 32; returns the tensor.
            x = torch.empty(sz, device=GPU_TYPE)
            BLOCK_SIZE = 32
            grid = (triton.cdiv(sz, BLOCK_SIZE),)
            kernel[grid](x, sz, BLOCK_SIZE)
            return x

        actual = fn(345)
        expected = torch.compile(fn, fullgraph=True)(345)
        self.assertEqual(actual, expected)

    @requires_gpu
    @inductor_config.patch({"triton.autotune_at_compile_time": True})
    def test_kernel_with_backslash_in_docstring(self):
        # [FUNCTION DOC] Regression for issue 183420: a kernel with backslash escapes in its
        # docstring must be spliced into the generated wrapper without turning escape sequences
        # into real newlines (a syntax error). Runs with compile-time autotuning; compares eager
        # and compiled outputs for size 345.
        # https://github.com/pytorch/pytorch/issues/183420
        # Backslash escapes in the kernel source text must be doubled before
        # the source is spliced into the generated triple-quoted wrapper,
        # otherwise '\n' becomes a real newline in the second-stage .py file
        # and parsing fails with SyntaxError.
        def fn(sz):
            # [FUNCTION DOC] HELPER: Allocates an empty tensor of the given size and launches the
            # backslash-docstring kernel with block 32; returns the tensor.
            x = torch.empty(sz, device=GPU_TYPE)
            BLOCK_SIZE = 32
            grid = (triton.cdiv(sz, BLOCK_SIZE),)
            kernel_with_backslash_in_docstring[grid](x, sz, BLOCK_SIZE)
            return x

        actual = fn(345)
        expected = torch.compile(fn, fullgraph=True)(345)
        self.assertEqual(actual, expected)

    @requires_gpu
    @skipIfXpu(
        msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs, "
        "https://github.com/pytorch/pytorch/pull/167786"
    )
    @inductor_config.patch({"triton.autotune_at_compile_time": True})
    @parametrize("quotes", ["single", "double"])
    def test_kernel_inline_asm(self, quotes):
        # [FUNCTION DOC] Parametrized on quote style; skipped on XPU. Chooses an inline-assembly
        # kernel variant for CUDA or ROCm and single or double quotes, runs with compile-time
        # autotuning, and compares eager and compiled outputs on 345 random elements (regression
        # for issue 155006).
        if torch.version.cuda:
            kernel = (
                kernel_inline_asm_single_quotes
                if quotes == "single"
                else kernel_inline_asm_double_quotes
            )
        else:
            kernel = (
                kernel_inline_asm_rocm_single_quotes
                if quotes == "single"
                else kernel_inline_asm_rocm_double_quotes
            )

        # https://github.com/pytorch/pytorch/issues/155006
        def fn(inp):
            # [FUNCTION DOC] HELPER: Allocates an empty output sized like the input and launches
            # the inline-asm kernel over cdiv grid with block 32; returns the output.
            sz = inp.size(0)
            x = torch.empty(sz, device=GPU_TYPE)
            BLOCK_SIZE = 32
            grid = (triton.cdiv(sz, BLOCK_SIZE),)
            kernel[grid](inp, x, sz, BLOCK_SIZE)
            return x

        inp = torch.randn(345, device=GPU_TYPE)
        actual = fn(inp)
        expected = torch.compile(fn, fullgraph=True)(inp)
        self.assertEqual(actual, expected)

    @requires_gpu
    @inductor_config.patch("emulate_precision_casts", True)
    def test_triton_kernel_emulate_precision_unaffected(self):
        # [FUNCTION DOC] With the emulate-precision-casts config on, checks that a user Triton
        # kernel's generated code is not given the fp fusion disabling option.
        # [FUNCTION DOC] HELPER: Kernel squaring the input and optionally adding an amount when
        # it is not None.
        @triton.jit
        def triton_(in_ptr, out_ptr, numel, add_amount, BLOCK_SIZE: tl.constexpr):
            offsets = tl.arange(0, BLOCK_SIZE)
            x = tl.load(in_ptr + offsets, mask=(offsets < numel))
            output = x * x
            if add_amount is not None:
                output = output + add_amount
            tl.store(out_ptr + offsets, output, mask=(offsets < numel))

        def fn(x):
            # [FUNCTION DOC] HELPER: Empty output; launches the kernel with one program, None
            # amount and block 256; returns the output.
            y = torch.empty_like(x)
            BLOCK_SIZE = 256
            grid = (1,)
            triton_[grid](x, y, x.numel(), None, BLOCK_SIZE)
            return y

        t1 = torch.rand(5, device=GPU_TYPE)
        fn = torch.compile(fn)
        _, (code,) = run_and_get_code(fn, t1)
        self.assertTrue("enable_fp_fusion" not in code)

    @requires_gpu
    @inductor_config.patch("emulate_precision_casts", True)
    @inductor_config.patch("max_autotune_gemm_backends", "TRITON")
    def test_triton_kernel_emulate_precision_mm_kernels_do_not_change(self):
        # [FUNCTION DOC] With emulate-precision-casts on and the Triton gemm backend only,
        # compiles a matrix multiply with max-autotune mode and checks the generated code lacks
        # the fp fusion disabling option. If inductor reports no valid choices the test is
        # skipped; other errors are re-raised.
        from torch._inductor.utils import run_and_get_code

        @torch.compile(mode="max-autotune")
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the matrix product of the two inputs.
            return a @ b

        t1 = torch.rand(512, 512, device=GPU_TYPE)
        t2 = torch.rand(512, 512, device=GPU_TYPE)
        try:
            _, (code,) = run_and_get_code(fn, t1, t2)
            self.assertTrue("enable_fp_fusion" not in code)
        except Exception as e:
            if "NoValidChoicesError" in str(e):
                raise unittest.SkipTest(
                    "where inductor has no triton mm kernels available, this test is meaningless"
                ) from e
            raise

    @requires_gpu
    def test_constexpr_handling(self):
        # [FUNCTION DOC] Uses the ttir generation helper directly on a copy kernel whose
        # arguments include a None value and constexpr block size. Checks that the textual module
        # contains the pointer, element count and stride parameters but not the constexpr block
        # size or None-valued parameter, which are inlined.
        # [FUNCTION DOC] HELPER: Kernel loading with a stride, scaling by one half or one
        # depending on a parameter, and storing back.
        @triton.jit
        def copy_kernel(
            src_ptr,
            dst_ptr,
            n_elements,
            stride,
            maybe_param,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(src_ptr + offs * stride, mask=mask)
            scale = tl.where(maybe_param != 0, 0.5, 1.0)
            x = x * scale

            tl.store(dst_ptr + offs * stride, x, mask=mask)

        t = torch.randn(1024, device=GPU_TYPE)
        out = torch.empty(1024, device=GPU_TYPE)

        kwargs = {
            "src_ptr": t,
            "dst_ptr": out,
            "n_elements": 1024,
            "stride": 1,
            "maybe_param": None,  # semantically wrong, but testing frontend specialization
            "BLOCK_SIZE": 256,
        }

        ttir_module, _ = generate_ttir(copy_kernel, kwargs, tma_descriptor_metadata={})
        ttir_str = str(ttir_module)

        # `constexpr` and None values get inlined, and do not appear as function parameters.
        self.assertIn("src_ptr", ttir_str)
        self.assertIn("dst_ptr", ttir_str)
        self.assertIn("n_elements", ttir_str)
        self.assertIn("stride", ttir_str)
        self.assertNotIn("BLOCK_SIZE", ttir_str)
        self.assertNotIn("maybe_param", ttir_str)


def make_mutation_test(fn):
    # [FUNCTION DOC] HELPER: Test factory. Takes a function returning a kernel, its inputs, TMA
    # metadata and expected mutated names, and returns a GPU-gated test method that runs the
    # access analysis and asserts the names of written tensors equal the expected list in order.
    @requires_gpu
    def test_fn(self):
        # [FUNCTION DOC] HELPER: Calls the factory's function to obtain the case, runs the
        # accessed-tensor analysis on it, collects the names of written dependencies, and asserts
        # they equal the expected list.
        from torch._higher_order_ops.triton_kernel_wrap import identify_accessed_tensors

        kernel, inputs, tma_descriptor_metadata, outputs = fn()
        tensor_accesses = identify_accessed_tensors(
            kernel, inputs, tma_descriptor_metadata
        )
        mutated_tensor_names = [dep.name for dep in tensor_accesses.read_writes.writes]
        self.assertListEqual(
            mutated_tensor_names,
            outputs,
        )

    return test_fn


# Triton codegen suffers from scoping issues.
# Define helpers here
if HAS_GPU:

    # [FUNCTION DOC] HELPER: Triton helper returning its argument; defined at module level
    # because kernel scoping requires it.
    @triton.jit
    def helper_id(p):
        return p

    # [FUNCTION DOC] HELPER: Triton helper returning the sum of
    # two values and the given output pointer.
    @triton.jit
    def helper_add_and_out(x, y, out_ptr):
        return x + y, out_ptr


class MutationTests(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Test case for the mutation analysis of Triton kernels. Most tests are generated
    # by wrapping a function in the test factory: each function defines a kernel and returns the
    # kernel, inputs, TMA metadata and expected list of mutated argument names, and the generated
    # test asserts the analysis finds exactly those writes. The class also has one ordinary test
    # for inference mode.
    # Tests injected below

    # Test that a scalar args are not flagged as mutated when passed
    # to a tt.call op.
    @make_mutation_test
    def test_scalar_via_nested_write():
        # [FUNCTION DOC] Checks that a scalar argument passed on to a nested kernel call is not
        # reported as mutated; only the output pointer is.
        # [FUNCTION DOC] HELPER: Kernel storing one at the pointer plus a scalar offset.
        @triton.jit
        def inner(ptr, scalar_offset):
            tl.store(ptr + scalar_offset, 1.0)

        # [FUNCTION DOC] HELPER: Kernel calling the inner kernel with its output pointer and
        # element count.
        @triton.jit
        def outer(out_ptr, n_elements):
            inner(out_ptr, n_elements)

        t = torch.randn(4)
        return (
            outer,
            {"out_ptr": t, "n_elements": 4},
            {},
            ["out_ptr"],
        )

    # Regression test for #169782
    @make_mutation_test
    def test_with_none_arg():
        # [FUNCTION DOC] Regression test for issue 169782: a kernel with an optional pointer
        # argument given as None; only the output pointer is expected as mutated.
        # [FUNCTION DOC] HELPER: Kernel adding an optional second input if it is not None, then
        # storing to the output.
        @triton.jit
        def add_kernel_with_none_arg(
            in_ptr0,
            optional_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            if optional_ptr is not None:
                y = tl.load(optional_ptr + offsets, mask=mask)
                x = x + y
            tl.store(out_ptr + offsets, x, mask=mask)

        t = torch.randn(4)
        return (
            add_kernel_with_none_arg,
            {
                "in_ptr0": t,
                "optional_ptr": None,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        )

    @make_mutation_test
    def test_out_of_order_kernel():
        # [FUNCTION DOC] Checks mutation analysis on a kernel whose
        # parameters are not in the usual order (element count before the
        # second input); only the output pointer is mutated.
        # [FUNCTION DOC] HELPER: Elementwise add kernel with
        # element count in the middle of its parameters.
        @triton.jit
        def add_kernel_out_of_order(
            in_ptr0,
            n_elements,
            in_ptr1,
            out_ptr,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_kernel_out_of_order,
            {
                "in_ptr0": t,
                "n_elements": 4,
                "in_ptr1": t,
                "out_ptr": t,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        )

    @make_mutation_test
    def test_out_of_order_kernel_call():
        # [FUNCTION DOC] Checks that a kernel calling another kernel with arguments in a
        # different order is analysed correctly; the output pointer is expected as mutated.
        # [FUNCTION DOC] HELPER: Kernel that calls the reordered add kernel with the inputs,
        # element count and output pointer.
        @triton.jit
        def add_kernel_out_of_order_fn1(
            in_ptr0,
            n_elements,
            in_ptr1,
            out_ptr,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            add_kernel_out_of_order_fn2(
                in_ptr0, in_ptr1, n_elements, out_ptr, BLOCK_SIZE=BLOCK_SIZE
            )

        t = torch.randn(4)
        return (
            add_kernel_out_of_order_fn1,
            {
                "in_ptr0": t,
                "n_elements": 4,
                "in_ptr1": t,
                "out_ptr": t,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        )

    @make_mutation_test
    def test_reduce_sum():
        # [FUNCTION DOC] Checks a kernel using a sum reduction. With the newer MLIR-based
        # analysis the expected mutated list is just the output pointer; with the older text-
        # parsing analysis (when the module has no walk method) both pointers are expected.
        # [FUNCTION DOC] HELPER: Kernel loading a 4 by 4 block, summing along axis one and
        # storing the result at the output pointer.
        @triton.jit
        def reduce_sum_kernel(a_ptr, c_ptr, stride_am, stride_an):
            offs_am = tl.arange(0, 4)
            offs_an = tl.arange(0, 4)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_an[None, :] * stride_an
            )
            a = tl.load(a_ptrs)
            m = tl.sum(a, axis=1)
            tl.store(c_ptr + tl.arange(0, 4), m)

        t = torch.randn(4)
        kernel = reduce_sum_kernel
        kwargs = {
            "a_ptr": t,
            "c_ptr": t,
            "stride_am": 4,
            "stride_an": 4,
        }

        # TODO(aakhundov): tt.reduce is now supported, but only
        # in the new MLIR-based Triton analysis pass (not in the
        # old TTIR string parsing-based one). remove this gating
        # and use ["c_ptr"] as `expected` after the new Triton
        # pin lands both in OSS and internally.
        ttir_module, _ = generate_ttir(kernel, kwargs, tma_descriptor_metadata={})
        if hasattr(ttir_module, "walk"):
            # with MLIR-based Triton analysis pass
            expected = ["c_ptr"]
        else:
            # with TTIR string parsing-based Triton analysis pass
            expected = ["a_ptr", "c_ptr"]

        return (
            kernel,
            kwargs,
            {},
            expected,
        )

    @make_mutation_test
    def test_argmax():
        # [FUNCTION DOC] Same as the reduction test but with an argmax reduction; the expected
        # mutated list depends on whether the MLIR-based analysis is used.
        # [FUNCTION DOC] HELPER: Kernel loading a 4 by 4 block, taking argmax along axis one and
        # storing the result.
        @triton.jit
        def argmax_kernel(a_ptr, c_ptr, stride_am, stride_an):
            offs_am = tl.arange(0, 4)
            offs_an = tl.arange(0, 4)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_an[None, :] * stride_an
            )
            a = tl.load(a_ptrs)
            m = tl.argmax(a, axis=1)
            tl.store(c_ptr + tl.arange(0, 4), m)

        t = torch.randn(4)
        kernel = argmax_kernel
        kwargs = {
            "a_ptr": t,
            "c_ptr": t,
            "stride_am": 4,
            "stride_an": 4,
        }

        # TODO(aakhundov): tt.reduce is now supported, but only
        # in the new MLIR-based Triton analysis pass (not in the
        # old TTIR string parsing-based one). remove this gating
        # and use ["c_ptr"] as `expected` after the new Triton
        # pin lands both in OSS and internally.
        ttir_module, _ = generate_ttir(kernel, kwargs, tma_descriptor_metadata={})
        if hasattr(ttir_module, "walk"):
            # with MLIR-based Triton analysis pass
            expected = ["c_ptr"]
        else:
            # with TTIR string parsing-based Triton analysis pass
            expected = ["a_ptr", "c_ptr"]

        return (
            kernel,
            kwargs,
            {},
            expected,
        )

    @requires_gpu
    def test_triton_kernel_inference_mode(self):
        # [FUNCTION DOC] Checks a Triton kernel launch under inference mode: compiled output
        # equals the eager output.
        def f(x, y, out):
            # [FUNCTION DOC] HELPER: Launches add_kernel with block 4 on the given tensors,
            # writing into out.
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=4)

        with torch.inference_mode():
            x = torch.ones(32, device=GPU_TYPE)
            y = torch.ones(32, device=GPU_TYPE)
            out_ref = torch.zeros_like(x)
            out_test = torch.zeros_like(x)
            f(x, y, out_ref)
            torch.compile(f)(x, y, out_test)
            self.assertEqual(out_ref, out_test)

    @make_mutation_test
    def test_cumsum():
        # [FUNCTION DOC] Checks mutation analysis for a kernel with a cumulative sum and device
        # assertion; the expected list is the output pointer with the MLIR analysis and both
        # pointers with the older text-parsing analysis.
        # [FUNCTION DOC] HELPER: Kernel loading a row, computing its cumulative sum, asserting it
        # does not exceed the sum, and storing it to the output.
        @triton.jit
        def cumsum_kernel(in_ptr, out_ptr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
            rindex = tl.arange(0, RBLOCK)[None, :]
            xindex = tl.arange(0, XBLOCK)[:, None]
            data = tl.load(in_ptr + rindex)
            scan = tl.cumsum(data, 1)
            expected_max = tl.sum(data, 1)
            tl.device_assert(scan <= expected_max)
            tl.store(out_ptr + xindex * RBLOCK + rindex, scan)

        t = torch.randn(4)
        kernel = cumsum_kernel
        kwargs = {
            "in_ptr": t,
            "out_ptr": t,
            "XBLOCK": 4,
            "RBLOCK": 16,
        }

        # TODO(aakhundov): tt.scan is now supported, but only
        # in the new MLIR-based Triton analysis pass (not in the
        # old TTIR string parsing-based one). remove this gating
        # and use ["out_ptr"] as `expected` after the new Triton
        # pin lands both in OSS and internally.
        ttir_module, _ = generate_ttir(kernel, kwargs, tma_descriptor_metadata={})
        if hasattr(ttir_module, "walk"):
            # with MLIR-based Triton analysis pass
            expected = ["out_ptr"]
        else:
            # with TTIR string parsing-based Triton analysis pass
            expected = ["in_ptr", "out_ptr"]

        return (
            kernel,
            kwargs,
            {},
            expected,
        )

    @make_mutation_test
    def test_fn_call_one_return():
        # [FUNCTION DOC] Checks that a helper kernel returning the output pointer used for a
        # store is tracked; the output pointer must be reported mutated.
        # [FUNCTION DOC] HELPER: Add kernel that stores through the pointer returned by the
        # identity helper.
        @triton.jit
        def add_kernel_with_fn_call(
            in_ptr0,
            in_ptr1,
            n_elements,
            out_ptr,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            out = helper_id(out_ptr)
            tl.store(out + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_kernel_with_fn_call,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "n_elements": 4,
                "out_ptr": t,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        )

    @make_mutation_test
    def test_fn_call_multi_return():
        # [FUNCTION DOC] Checks tracking of the output pointer through a helper returning
        # multiple values (the sum and the pointer); the output pointer must be reported mutated.
        # [FUNCTION DOC] HELPER: Add kernel that gets its output value and pointer from the add-
        # and-out helper and stores.
        @triton.jit
        def add_kernel_with_fn_call(
            in_ptr0,
            in_ptr1,
            n_elements,
            out_ptr,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output, out = helper_add_and_out(x, y, out_ptr)
            tl.store(out + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_kernel_with_fn_call,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "n_elements": 4,
                "out_ptr": t,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        )

    @make_mutation_test
    def test_nested_cond_op_kernel():
        # [FUNCTION DOC] Checks that a store inside nested conditionals on program ids is still
        # reported as a write to the output pointer.
        # [FUNCTION DOC] HELPER: Kernel that stores the sum only inside two nested conditions on
        # program ids.
        @triton.jit
        def nested_cond_op_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            if tl.program_id(0) == 0:
                if tl.program_id(1) == 0:
                    output = x + y
                    tl.store(out_ptr + offsets, output, mask=mask)
            else:
                pass

        t = torch.randn(4)
        return (
            nested_cond_op_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        )

    @make_mutation_test
    def test_add_for_loop():
        # [FUNCTION DOC] Checks mutation analysis of a store following a loop that accumulates
        # four times.
        # [FUNCTION DOC] HELPER: Kernel accumulating the sum of two inputs four times in a loop
        # then storing it.
        @triton.jit
        def add_4_times_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements: "tl.constexpr",
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            for _ in range(4):
                output += x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_4_times_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        )

    @make_mutation_test
    def test_add_for_loop2():
        # [FUNCTION DOC] Checks mutation analysis for a loop over the block size with a multiple-
        # of hint inside; output pointer expected as mutated.
        # [FUNCTION DOC] HELPER: Kernel with a loop applying a
        # multiple-of hint, then storing the sum.
        @triton.jit
        def add_1_time_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            for i in range(BLOCK_SIZE):
                i = tl.multiple_of(i, 1)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_1_time_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        )

    @make_mutation_test
    def test_add_nested_for_loop():
        # [FUNCTION DOC] Checks mutation analysis of a kernel with a doubly nested loop
        # accumulating the sum.
        # [FUNCTION DOC] HELPER: Kernel accumulating the sum in two nested loops of two
        # iterations and storing it.
        @triton.jit
        def add_4_times_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements: "tl.constexpr",
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            for _ in range(2):
                for _ in range(2):
                    output += x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_4_times_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        )

    @make_mutation_test
    def test_add_nested_for_loop_multi_return():
        # [FUNCTION DOC] Checks mutation analysis of a kernel with two accumulators updated
        # inside nested loops.
        # [FUNCTION DOC] HELPER: Kernel accumulating one input in one buffer and the other in
        # another buffer inside nested loops, then storing their sum.
        @triton.jit
        def add_4_times_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements: "tl.constexpr",
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output1 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            output2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            for _ in range(2):
                for _ in range(2):
                    output1 += y
                    output2 += x
            output = output1 + output2
            tl.store(out_ptr + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            add_4_times_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        )

    @make_mutation_test
    def test_labels():
        # [FUNCTION DOC] Checks mutation analysis of a kernel with an early return (a label in
        # the lowered code) for program ids above one.
        # [FUNCTION DOC] HELPER: Add kernel that returns early when the program id exceeds one.
        @triton.jit
        def kernel_with_label(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            if pid > 1:
                return
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        t = torch.randn(4)
        return (
            kernel_with_label,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        )

    @make_mutation_test
    def test_for_loop_arg():
        # [FUNCTION DOC] Checks mutation analysis of a kernel with a loop-carried pointer
        # argument and matrix products; only the output pointer is expected.
        # [FUNCTION DOC] HELPER: Kernel looping over column blocks, computing a matrix product of
        # the input block with a weight block plus bias, and storing it to the output.
        @triton.jit
        def fwd_kernel(
            X_ptr,
            W1_ptr,
            b1_ptr,
            O_ptr,
            M: tl.constexpr,
            C1: tl.constexpr,
            C2: tl.constexpr,
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_C2: tl.constexpr,
        ):
            # Get program ids
            pid_m = tl.program_id(0)

            # Compute offsets
            offs_c1 = tl.arange(0, C1)
            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

            # Load input data
            x_block_ptr = X_ptr + offs_m[:, None] * C1 + offs_c1[None, :]
            x = tl.load(x_block_ptr)

            # Compute gating
            for c2 in range(tl.cdiv(C2, BLOCK_SIZE_C2)):
                # Compute block pointers
                offs_c2 = c2 * BLOCK_SIZE_C2 + tl.arange(0, BLOCK_SIZE_C2)
                o_block_ptr = O_ptr + offs_m[:, None] * C2 + offs_c2[None, :]
                w1_block_ptr = W1_ptr + offs_c1[:, None] * C2 + offs_c2[None, :]
                b1_block_ptr = b1_ptr + offs_c2

                # Compute output
                w = tl.load(w1_block_ptr)
                b = tl.load(b1_block_ptr)
                o = tl.dot(x, w, allow_tf32=False)
                o += b[None, :]

                # Store output
                tl.store(o_block_ptr, o)

        t = torch.randn(64)
        return (
            fwd_kernel,
            {
                "X_ptr": t,
                "W1_ptr": t,
                "b1_ptr": t,
                "O_ptr": t,
                "M": 64,
                "C1": 64,
                "C2": 64,
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_C2": 64,
            },
            {},
            ["O_ptr"],
        )

    @make_mutation_test
    def test_for_loop_arg_2():
        # [FUNCTION DOC] Checks mutation analysis of a kernel that copies batches through block
        # pointers in a counted loop, advancing both pointers each iteration; only the output
        # pointer is expected.
        # [FUNCTION DOC] HELPER: Kernel creating input and output block pointers and, for each
        # batch iteration, loading a block, storing it and advancing both pointers.
        @triton.jit
        def fwd_kernel(
            x_ptr,
            o_ptr,
            M,
            N,
            stride_m,
            stride_n,
            BLOCK_B: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            # Get program ids
            pid_m = tl.program_id(0)
            X_block_ptr = tl.make_block_ptr(
                base=x_ptr,
                shape=(M, N),
                strides=(stride_m, stride_n),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            O_block_ptr = tl.make_block_ptr(
                base=o_ptr,
                shape=(M, N),
                strides=(stride_m, stride_n),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )

            for _ in range(BLOCK_B):
                x = tl.load(X_block_ptr)
                tl.store(O_block_ptr, x)

                X_block_ptr = tl.advance(X_block_ptr, (BLOCK_M, 0))
                O_block_ptr = tl.advance(O_block_ptr, (BLOCK_M, 0))

        t = torch.randn((32, 64, 128))
        o = torch.empty_like(t)
        B, M, N = t.shape
        return (
            fwd_kernel,
            {
                "x_ptr": t,
                "o_ptr": o,
                "M": M,
                "N": N,
                "stride_m": N,
                "stride_n": 1,
                "BLOCK_B": B,
                "BLOCK_M": M,
                "BLOCK_N": N,
            },
            {},
            ["o_ptr"],
        )

    @make_mutation_test
    def test_while_loop():
        # [FUNCTION DOC] Checks mutation analysis of a kernel that copies blocks in a while loop
        # with block pointers; only the output pointer is expected.
        # [FUNCTION DOC] HELPER: Kernel creating block pointers and, in a while loop up to the
        # batch count, loading, storing and advancing both pointers.
        @triton.jit
        def fwd_kernel(
            x_ptr,
            o_ptr,
            M,
            N,
            stride_m,
            stride_n,
            BLOCK_B: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            # Get program ids
            pid_m = tl.program_id(0)
            X_block_ptr = tl.make_block_ptr(
                base=x_ptr,
                shape=(M, N),
                strides=(stride_m, stride_n),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            O_block_ptr = tl.make_block_ptr(
                base=o_ptr,
                shape=(M, N),
                strides=(stride_m, stride_n),
                offsets=(0, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )

            i = 0
            while i < BLOCK_B:
                x = tl.load(X_block_ptr)
                tl.store(O_block_ptr, x)

                X_block_ptr = tl.advance(X_block_ptr, (BLOCK_M, 0))
                O_block_ptr = tl.advance(O_block_ptr, (BLOCK_M, 0))
                i += 1

        t = torch.randn((32, 64, 128))
        o = torch.empty_like(t)
        B, M, N = t.shape
        return (
            fwd_kernel,
            {
                "x_ptr": t,
                "o_ptr": o,
                "M": M,
                "N": N,
                "stride_m": N,
                "stride_n": 1,
                "BLOCK_B": B,
                "BLOCK_M": M,
                "BLOCK_N": N,
            },
            {},
            ["o_ptr"],
        )

    @make_mutation_test
    def test_branch_with_multiple_yield_args():
        # [FUNCTION DOC] Checks that a conditional branch yielding several
        # pointers (inputs and output shifted by one, from a loaded flag)
        # still reports only the output pointer as mutated.
        # [FUNCTION DOC] HELPER: Kernel that, depending on a loaded flag, offsets the three
        # pointers by one, then adds the two inputs and stores the sum to the output.
        @triton.jit
        def branch_with_multiple_yield_args(
            in_ptr0,
            in_ptr1,
            out_ptr,
            conditional_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            conditional = tl.load(conditional_ptr)
            if conditional:
                in0 = in_ptr0 + 1
                in1 = in_ptr1 + 1
                out = out_ptr + 1
            else:
                in0 = in_ptr0
                in1 = in_ptr1
                out = out_ptr
            x = tl.load(in0 + offsets, mask=mask)
            y = tl.load(in1 + offsets, mask=mask)
            tl.store(out + offsets, x + y, mask=mask)

        x = torch.randn(15)
        y = torch.randn(15)
        out = torch.zeros(15)
        conditional = torch.tensor(True)
        return (
            branch_with_multiple_yield_args,
            {
                "in_ptr0": x,
                "in_ptr1": y,
                "out_ptr": out,
                "conditional_ptr": conditional,
                "n_elements": 14,
                "BLOCK_SIZE": 16,
            },
            {},
            ["out_ptr"],
        )

    def test_get_tma_stores(self):
        # [FUNCTION DOC] Unit test of the TMA-store discovery helper on hand-built function
        # graphs. A helper function that only reinterprets a descriptor, called from main, yields
        # empty sets for both. After adding an experimental descriptor store op in the helper and
        # resetting the cache, the helper reports its first parameter and intermediate, and main
        # reports its first parameter.
        from torch._higher_order_ops.triton_kernel_wrap import (
            get_tma_stores,
            Intermediate,
            Op,
            Param,
        )

        functions = {
            "helper": {
                Intermediate(idx=0): [
                    Op(
                        "tt.reinterpret_tensor_descriptor",
                        None,
                        [Param(idx=0)],
                        Intermediate(idx=0),
                    )
                ],
            },
            "main": {
                Intermediate(idx=-1): [
                    Op(
                        "tt.call",
                        "helper",
                        [Param(idx=0), Param(idx=1)],
                        Intermediate(idx=-1),
                    )
                ],
            },
        }

        self.assertEqual(get_tma_stores(functions, "helper"), set())
        self.assertEqual(get_tma_stores(functions, "main"), set())

        functions["helper"][Intermediate(idx=-1)] = [
            Op(
                "tt.experimental_descriptor_store",
                None,
                [Intermediate(idx=0), Param(idx=1)],
                Intermediate(idx=-1),
            )
        ]
        get_tma_stores.reset()

        self.assertEqual(
            get_tma_stores(functions, "helper"), {Param(idx=0), Intermediate(idx=0)}
        )
        self.assertEqual(get_tma_stores(functions, "main"), {Param(idx=0)})

    def test_tensormap_create_marks_descriptor_write(self):
        # [FUNCTION DOC] Checks that the tensormap-create operation marks its first
        # argument (the workspace) as written, not the second (a global pointer).
        # Builds a one-function graph, resets the cache, runs the access analysis
        # with two argument names and compares write names.
        from torch._higher_order_ops.triton_kernel_wrap import (
            analyze_kernel_access,
            Intermediate,
            Op,
            Param,
        )

        functions = {
            "main": {
                Intermediate(idx=-1): [
                    Op(
                        "tt.experimental_tensormap_create",
                        None,
                        [Param(idx=0), Param(idx=1)],
                        Intermediate(idx=-1),
                    )
                ],
            },
        }

        analyze_kernel_access.reset()
        tensor_accesses = analyze_kernel_access(
            functions,
            "main",
            2,
            ("workspace", "global_ptr"),
            frozenset({0, 1}),
        )

        write_names = [dep.name for dep in tensor_accesses.read_writes.writes]
        self.assertListEqual(write_names, ["workspace"])

    def test_register_kernel_access_op_updates_direct_analysis_after_reset(self):
        # [FUNCTION DOC] Checks that registering a custom kernel access op changes
        # later analysis. Before registration the custom op writes nothing; after
        # registering it with write index zero and resetting the analysis cache, the
        # first argument is reported written. The finally block unregisters the op
        # and resets the analysis and TMA store caches.
        from torch._higher_order_ops import triton_kernel_wrap as tkw
        from torch._higher_order_ops.triton_kernel_wrap import (
            analyze_kernel_access,
            get_tma_stores,
            Intermediate,
            Op,
            Param,
        )

        custom_store = "custom_tma.cache_test_store"
        functions = {
            "main": {
                Intermediate(idx=-1): [
                    Op(custom_store, None, [Param(idx=0)], Intermediate(idx=-1))
                ],
            },
        }

        try:
            analyze_kernel_access.reset()
            tensor_accesses = analyze_kernel_access(
                functions,
                "main",
                1,
                ("out_ptr",),
                frozenset({0}),
            )
            self.assertEqual(len(tensor_accesses.read_writes.writes), 0)

            tkw.register_kernel_access_op(custom_store, write_indexes=[0])
            # identify_accessed_tensors resets production caches per invocation.
            # Direct analyze_kernel_access callers must reset before re-analysis.
            analyze_kernel_access.reset()
            tensor_accesses = analyze_kernel_access(
                functions,
                "main",
                1,
                ("out_ptr",),
                frozenset({0}),
            )
            write_names = [dep.name for dep in tensor_accesses.read_writes.writes]
            self.assertListEqual(write_names, ["out_ptr"])
        finally:
            tkw.unregister_kernel_access_op(custom_store)
            analyze_kernel_access.reset()
            get_tma_stores.reset()

    def test_register_kernel_access_op_rejects_ignore_with_indexes(self):
        # [FUNCTION DOC] Checks that registering a kernel access op with an ignore predicate
        # together with write indexes raises an AssertionError with the message that ignore_if
        # cannot be combined with read/write indexes.
        from torch._higher_order_ops import triton_kernel_wrap as tkw

        msg = "custom_tma.invalid: ignore_if cannot be combined with read/write indexes"
        with self.assertRaisesRegex(AssertionError, msg):
            tkw.register_kernel_access_op(
                "custom_tma.invalid", write_indexes=[0], ignore_if=lambda op: True
            )

    def test_kernel_access_op_can_read_and_write(self):
        # [FUNCTION DOC] Checks that the atomic read-modify-write op is reported as both a read
        # and a write of its single pointer argument.
        from torch._higher_order_ops.triton_kernel_wrap import (
            analyze_kernel_access,
            Intermediate,
            Op,
            Param,
        )

        functions = {
            "main": {
                Intermediate(idx=-1): [
                    Op("tt.atomic_rmw", None, [Param(idx=0)], Intermediate(idx=-1))
                ],
            },
        }

        analyze_kernel_access.reset()
        tensor_accesses = analyze_kernel_access(
            functions,
            "main",
            1,
            ("in_out_ptr",),
            frozenset({0}),
        )
        read_names = [dep.name for dep in tensor_accesses.read_writes.reads]
        write_names = [dep.name for dep in tensor_accesses.read_writes.writes]
        self.assertListEqual(read_names, ["in_out_ptr"])
        self.assertListEqual(write_names, ["in_out_ptr"])

    @unittest.skipIf(
        not has_triton_experimental_host_tma(),
        "requires experimental TMA descriptor API",
    )
    @make_mutation_test
    def test_add_kernel_on_device_tma_old_api():
        # [FUNCTION DOC] Skipped when the experimental host TMA API is missing. Mutation test for
        # the old-API on-device TMA add kernel on large tensors and a workspace: the output and
        # the workspace are expected as written.
        a = torch.randn(1024, 1024)
        b = torch.randn(1024, 1024)
        c = torch.empty(1024, 1024)
        workspace = torch.empty(128 * 3, dtype=torch.int8)
        return (
            add_kernel_on_device_tma_old_api,
            {
                "a_ptr": a,
                "b_ptr": b,
                "c_ptr": c,
                "m": 1024,
                "n": 1024,
                "workspace": workspace,
                "BLOCK_SIZE": 32,
            },
            {},
            ["c_ptr", "workspace"],
        )

    @unittest.skipIf(
        not has_triton_tensor_descriptor_host_tma(),
        "requires TensorDescriptor API in Triton",
    )
    @make_mutation_test
    def test_add_kernel_on_device_tma_new_api():
        # [FUNCTION DOC] Skipped when the tensor-descriptor TMA API is missing. Mutation test for
        # the new-API on-device TMA add kernel; only the output is expected as written (the
        # workspace is unused).
        a = torch.randn(1024, 1024)
        b = torch.randn(1024, 1024)
        c = torch.empty(1024, 1024)
        workspace = torch.empty(
            128 * 3, dtype=torch.int8
        )  # Not used by the new API but kept for consistency
        return (
            add_kernel_on_device_tma_new_api,
            {
                "a_ptr": a,
                "b_ptr": b,
                "c_ptr": c,
                "m": 1024,
                "n": 1024,
                "workspace": workspace,
                "BLOCK_SIZE": 32,
            },
            {},
            ["c_ptr"],
        )

    @unittest.skipIf(
        not has_triton_tensor_descriptor_host_tma(),
        "requires TensorDescriptor API in Triton",
    )
    @requires_gpu
    def test_descriptor_load_store_read_write_detection(self):
        # [FUNCTION DOC] Skipped without the tensor-descriptor API; GPU only. Regression test
        # that descriptor loads and stores created by device-side make-tensor-descriptor are
        # recognised by the access analysis: for a kernel loading from two descriptors and
        # storing to a third, the reads are the two inputs and the write is the output.
        """
        Regression test: tl.make_tensor_descriptor + desc.load()/desc.store()
        generates tt.descriptor_load/tt.descriptor_store ops in TTIR. These
        must be recognized by identify_accessed_tensors so that Inductor
        correctly tracks read/write dependencies.
        """
        from torch._higher_order_ops.triton_kernel_wrap import identify_accessed_tensors

        # [FUNCTION DOC] HELPER: Kernel building three tensor descriptors and storing the sum of
        # two loaded tiles to the output descriptor.
        @triton.jit
        def add_kernel_descriptor_method(
            in_ptr0,
            in_ptr1,
            out_ptr,
            M,
            N,
            BLOCK_M: "tl.constexpr",
            BLOCK_N: "tl.constexpr",
        ):
            in0_desc = tl.make_tensor_descriptor(
                in_ptr0,
                [M, N],
                [N, 1],
                [BLOCK_M, BLOCK_N],
            )
            in1_desc = tl.make_tensor_descriptor(
                in_ptr1,
                [M, N],
                [N, 1],
                [BLOCK_M, BLOCK_N],
            )
            out_desc = tl.make_tensor_descriptor(
                out_ptr,
                [M, N],
                [N, 1],
                [BLOCK_M, BLOCK_N],
            )
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            off_m = pid_m * BLOCK_M
            off_n = pid_n * BLOCK_N
            a = in0_desc.load([off_m, off_n])
            b = in1_desc.load([off_m, off_n])
            out_desc.store([off_m, off_n], a + b)

        t = torch.randn(64, 64)
        tensor_accesses = identify_accessed_tensors(
            add_kernel_descriptor_method,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "M": 64,
                "N": 64,
                "BLOCK_M": 32,
                "BLOCK_N": 32,
            },
            {},
        )
        read_names = [dep.name for dep in tensor_accesses.read_writes.reads]
        write_names = [dep.name for dep in tensor_accesses.read_writes.writes]
        self.assertListEqual(read_names, ["in_ptr0", "in_ptr1"])
        self.assertListEqual(write_names, ["out_ptr"])

    @unittest.skipUnless(
        HAS_GPU or (HAS_CPU and TRITON_HAS_CPU),
        "requires gpu or triton cpu",
    )
    def test_custom_tma_descriptor_ops_keep_triton_launcher(self):
        # [FUNCTION DOC] Runs on GPU or Triton CPU. Registers custom descriptor store (write
        # index zero, marked as a TMA store) and load (read index zero) ops, patches TTIR
        # generation and function parsing with canned values, and checks the analysis reports
        # in_ptr0 as read and out_ptr as written. Then compiles a function using mul2_kernel (CPU
        # Triton backend if on CPU) and checks the output is doubled and the code still uses the
        # normal Triton launcher (cpp wrapper or run form). The finally block unregisters both
        # ops and resets the caches.
        import triton

        from torch._higher_order_ops import triton_kernel_wrap as tkw
        from torch._higher_order_ops.triton_kernel_wrap import (
            analyze_kernel_access,
            get_tma_stores,
            identify_accessed_tensors,
            Intermediate,
            Op,
            Param,
        )

        custom_store = "custom_tma.descriptor_store"
        custom_load = "custom_tma.descriptor_load"
        device = GPU_TYPE if HAS_GPU else "cpu"

        functions = {
            "mul2_kernel": {
                Intermediate(idx=0): [
                    Op(
                        "tt.make_tensor_descriptor",
                        None,
                        [Param(idx=0)],
                        Intermediate(idx=0),
                    )
                ],
                Intermediate(idx=1): [
                    Op(
                        "tt.make_tensor_descriptor",
                        None,
                        [Param(idx=1)],
                        Intermediate(idx=1),
                    )
                ],
                Intermediate(idx=2): [
                    Op(custom_load, None, [Intermediate(idx=0)], Intermediate(idx=2))
                ],
                Intermediate(idx=-1): [
                    Op(
                        custom_store,
                        None,
                        [Intermediate(idx=1), Intermediate(idx=2)],
                        Intermediate(idx=-1),
                    )
                ],
            }
        }

        try:
            tkw.register_kernel_access_op(
                custom_store, write_indexes=[0], is_tma_store=True
            )
            tkw.register_kernel_access_op(custom_load, read_indexes=[0])
            with (
                mock.patch.object(
                    tkw,
                    "generate_ttir",
                    return_value=(
                        object(),
                        ["in_ptr0", "out_ptr", "n_elements", "BLOCK_SIZE"],
                    ),
                ),
                mock.patch.object(tkw, "ttir_to_functions", return_value=functions),
            ):
                x = torch.rand(16, device=device)
                tensor_accesses = identify_accessed_tensors(
                    mul2_kernel,
                    {
                        "in_ptr0": x,
                        "out_ptr": x,
                        "n_elements": x.numel(),
                        "BLOCK_SIZE": 16,
                    },
                    {},
                )
                read_names = [dep.name for dep in tensor_accesses.read_writes.reads]
                write_names = [dep.name for dep in tensor_accesses.read_writes.writes]
                self.assertListEqual(read_names, ["in_ptr0"])
                self.assertListEqual(write_names, ["out_ptr"])

                def f(x):
                    # [FUNCTION DOC] HELPER: Zero output; launches mul2_kernel with grid by cdiv
                    # and block 16; returns the output.
                    output = torch.zeros_like(x)
                    grid = (triton.cdiv(x.numel(), 16),)
                    mul2_kernel[grid](x, output, x.numel(), BLOCK_SIZE=16)
                    return output

                config_patch = (
                    inductor_config.patch("cpu_backend", "triton")
                    if device == "cpu"
                    else contextlib.nullcontext()
                )
                with config_patch:
                    actual, (code,) = run_and_get_code(torch.compile(f), x)
                self.assertEqual(actual, 2 * x)
                if inductor_config.cpp_wrapper:
                    self.assertIn("launchKernel(mul2_kernel_0", code)
                else:
                    self.assertIn("mul2_kernel_0.run(", code)
        finally:
            tkw.unregister_kernel_access_op(custom_store)
            tkw.unregister_kernel_access_op(custom_load)
            analyze_kernel_access.reset()
            get_tma_stores.reset()


if HAS_GPU:
    t = torch.randn(4)
    tt = torch.randn(4, 1)
    tests = [
        [
            add_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        ],
        [
            add_kernel_2d_autotuned,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "x_elements": 4,
                "y_elements": 4,
            },
            {},
            ["out_ptr"],
        ],
        [
            indirection_kernel,
            {
                "in_ptr0": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
                "ACTIVATION": "mul2_inplace_kernel",
            },
            {},
            ["in_ptr0", "out_ptr"],
        ],
        [
            indirection_kernel,
            {
                "in_ptr0": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
                "ACTIVATION": "add_kernel",
            },
            {},
            ["out_ptr"],
        ],
        [
            mul2_inplace_kernel,
            {"ptr": t, "n_elements": 4, "BLOCK_SIZE": 4},
            {},
            ["ptr"],
        ],
        [
            inline_asm_kernel_is_pure_true,
            {"X": t, "Y": t, "Z": t, "n": 4, "BLOCK": 4},
            {},
            ["Z"],
        ],
        [
            inline_asm_kernel_is_pure_false,
            {"X": t, "Y": t, "Z": t, "n": 4, "BLOCK": 4},
            {},
            ["X", "Y", "Z"],
        ],
        [
            add_kernel_with_block_ptr,
            {
                "x_ptr": t,
                "y_ptr": t,
                "output_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["output_ptr"],
        ],
        [
            kernel_with_block_ptr_2d,
            {
                "x_ptr": tt,
                "output_ptr": tt,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["output_ptr"],
        ],
        [
            add_kernel_with_import,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        ],
        [
            atomic_add_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        ],
        [
            add_4_times_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        ],
        [
            cond_op_kernel,
            {
                "in_ptr0": t,
                "in_ptr1": t,
                "out_ptr": t,
                "n_elements": 4,
                "BLOCK_SIZE": 4,
            },
            {},
            ["out_ptr"],
        ],
    ]
    for kernel, inputs, tma_descriptor_metadata, outputs in tests:
        fn = make_mutation_test(
            # Add default arguments to avoid Python lambda capture pitfall
            # This forces the capture at lambda creation
            lambda kernel=kernel,
            inputs=inputs,
            tma_descriptor_metadata=tma_descriptor_metadata,
            outputs=outputs: (
                kernel,
                inputs,
                tma_descriptor_metadata,
                outputs,
            )
        )
        name = f"test_mutations_{kernel.fn.__name__}"
        # Poor way to make test names be unique
        while name in MutationTests.__dict__:
            name += "1"

        if kernel.fn.__name__ == "add_kernel_2d_autotuned":
            fn = unittest.skip("Fails with Triton update")(fn)

        setattr(MutationTests, name, fn)


class CustomOpTests(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Tests of custom ops that wrap Triton kernels via the triton op decorator and
    # the capture wrapper.
    """Tests for custom ops wrapping triton kernels"""

    @requires_gpu
    @common_utils.parametrize("autotuned", [False, True])
    @common_utils.parametrize("dynamic", [False, True])
    def test_add_kernel(self, autotuned, dynamic):
        # [FUNCTION DOC] Parametrized on autotune and dynamic. Registers a triton op
        # that adds two tensors using the capture wrapper (autotuned or fixed block 16
        # kernel). Eager result equals the sum; compiled result equals it with exactly
        # one generated code object, and the generated code contains neither the
        # library nor the op name (the op is decomposed).
        from torch._inductor.utils import run_and_get_code

        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        @torch.library.triton_op(f"{libname}::{opname}", mutates_args={})
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Triton op body; empty output; launches
            # the (autotuned or fixed) add kernel through the capture wrapper
            # with a callable grid; returns the output.
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                # [FUNCTION DOC] HELPER: Returns a one-element grid by dividing the element count
                # by the block size meta parameter.
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            if autotuned:
                capture_triton(add_kernel_autotuned)[grid](x, y, output, n_elements)
            else:
                capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output

        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls the registered op on the two tensors.
            return add(x, y)

        x = torch.randn(3, device=GPU_TYPE)
        y = torch.randn(3, device=GPU_TYPE)

        out = f(x, y)
        expected = x + y
        self.assertEqual(out, expected)
        out_compiled, codes = run_and_get_code(torch.compile(f, dynamic=dynamic), x, y)
        self.assertEqual(out_compiled, expected)
        self.assertEqual(len(codes), 1)

        # Check that we decomposed the operator away
        code = "\n".join(codes[0])
        self.assertNotIn(libname, code)
        self.assertNotIn(opname, code)

    @requires_gpu
    def test_subclass(self):
        # [FUNCTION DOC] Checks the triton op with a tensor subclass holding two inner tensors:
        # the fullgraph compiled result's inner tensors must equal those of the eager result.
        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        @torch.library.triton_op(f"{libname}::{opname}", mutates_args={})
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Triton op body that launches add_kernel through the capture
            # wrapper with block 16 and returns the output.
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                # [FUNCTION DOC] HELPER: Returns a one-element grid by dividing the element count
                # by the block size meta parameter.
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output

        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls the registered op on the two tensors.
            return add(x, y)

        x0 = torch.randn(3, device=GPU_TYPE)
        y0 = torch.randn(3, device=GPU_TYPE)
        x1 = torch.randn(3, device=GPU_TYPE)
        y1 = torch.randn(3, device=GPU_TYPE)

        from torch.testing._internal.two_tensor import TwoTensor

        x = TwoTensor(x0, x1)
        y = TwoTensor(y0, y1)

        out = torch.compile(f, fullgraph=True)(x, y)
        expected = f(x, y)
        self.assertEqual(out.a, expected.a)
        self.assertEqual(out.b, expected.b)

    @requires_gpu
    @dynamo_config.patch("recompile_limit", 1)
    def test_triton_dynamic_grid_no_recompile(self):
        # [FUNCTION DOC] With a recompile limit of one, checks that a dynamic compiled function
        # calling a triton op whose grid depends on the element count does not recompile when
        # called with sizes 8 and 16.
        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        @torch.library.triton_op(f"{libname}::{opname}", mutates_args={})
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Triton op body launching add_kernel with a grid of one
            # program per element through the capture wrapper; returns the output.
            output = torch.empty_like(x)
            n_elements = output.numel()
            capture_triton(add_kernel)[(n_elements,)](x, y, output, n_elements, 16)
            return output

        @torch.compile(fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled (fullgraph, dynamic) function calling the op with
            # the same tensor twice.
            return add(x, x)

        f(torch.randn(8, device=GPU_TYPE))
        f(torch.randn(16, device=GPU_TYPE))

    @unittest.skipIf(not has_triton_package(), "requires triton")
    def test_capture_triton_meta(self):
        # [FUNCTION DOC] Skipped without triton. Checks the meta device implementation of a
        # triton op (derived from the captured kernel) returns an empty tensor shaped like the
        # input when called on meta tensors.
        import triton
        import triton.language as tl

        # [FUNCTION DOC] HELPER: Local masked elementwise add kernel.
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.library.triton_op("mylib::add", mutates_args=())
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Triton op body that launches the local kernel through the
            # capture wrapper with block 16; returns the output.
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                # [FUNCTION DOC] HELPER: Returns a one-element grid by dividing the element count
                # by the block size meta parameter.
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output

        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls the op on the two tensors.
            return add(x, y)

        x = torch.randn(3, device="meta")
        y = torch.randn(3, device="meta")

        out = f(x, y)
        expected = torch.empty_like(x)
        self.assertEqual(out, expected)

    @requires_gpu
    def test_wrap_triton_disabled_in_triton_op(self):
        # [FUNCTION DOC] Checks that wrap_triton is disabled while a triton op body runs. A
        # registered op records whether wrapping is enabled into a status list, asserts
        # wrap_triton returns the kernel itself, launches a pre-wrapped kernel both by index and
        # by the run method into two outputs, and returns their sum. The test requires the last
        # status false and the result equal to twice the sum.
        import triton  # @manual
        import triton.language as tl  # @manual

        # [FUNCTION DOC] HELPER: Local masked elementwise add kernel.
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        add_kernel_decorated = torch.library.wrap_triton(add_kernel)

        status = []

        @torch.library.triton_op("mylib::add", mutates_args=())
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Triton op body; records whether wrap_triton is enabled,
            # asserts wrap_triton returns the same kernel object, runs the pre-wrapped kernel by
            # index and by the run method into two outputs, and returns the sum of the outputs.
            import torch._higher_order_ops.triton_kernel_wrap

            status.append(torch._library.triton.is_wrap_triton_enabled())

            # capture_triton should return the kernel directly if disabled
            result = torch.library.wrap_triton(add_kernel)
            self.assertIs(result, add_kernel)

            # Smoke test: check that with capture_triton disabled this still does something
            output = torch.empty_like(x)
            output2 = torch.empty_like(x)

            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel_decorated[grid](x, y, output, n_elements, BLOCK_SIZE=16)

            add_kernel_decorated.run(
                x, y, output2, n_elements, BLOCK_SIZE=16, grid=grid, warmup=False
            )

            return output + output2

        x = torch.randn(3, device=GPU_TYPE)
        y = torch.randn(3, device=GPU_TYPE)
        z = add(x, y)
        self.assertEqual(status[-1], False)
        self.assertEqual(z, (x + y) * 2)

    @unittest.skipIf(not HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
    def test_wrap_triton_triton_interpret_eager(self):
        # [FUNCTION DOC] Needs CUDA and Triton. Runs a script in a subprocess (with the Triton
        # interpreter env var set) that defines plain and autotuned add kernels, launches both
        # through the wrap_triton wrapper in eager mode comparing with the sum, and checks that
        # wrapping an arbitrary object with a fn attribute raises a RuntimeError mentioning that
        # wrap_triton only works for kernels. The subprocess must exit successfully.
        script = """
import os
os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton


@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.autotune(configs=[triton.Config({"BLOCK_SIZE": 16})], key=[])
@triton.jit
def autotuned_add_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    wrap_triton(add_kernel)[grid](x, y, output, n_elements, 16)
    return output


def autotuned_add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    wrap_triton(autotuned_add_kernel)[grid](x, y, output, n_elements)
    return output


x = torch.randn(1024, device="cuda")
torch.testing.assert_close(add(x, x), x + x)
torch.testing.assert_close(autotuned_add(x, x), x + x)


class Wrapper:
    fn = add_kernel


try:
    wrap_triton(Wrapper())
except RuntimeError as exc:
    assert "wrap_triton only works" in str(exc)
else:
    raise AssertionError("wrap_triton accepted an arbitrary .fn wrapper")
"""

        subprocess.run([sys.executable, "-c", script], check=True)

    @requires_gpu
    @common_utils.parametrize(
        "variant", ["triton_kernel", "custom_op", "mutable_custom_op"]
    )
    def test_preserves_strides(self, variant):
        # [FUNCTION DOC] Parametrized over variant (direct kernel, custom op, mutable custom op).
        # Registers a graph pattern replacing permute with a version followed by a forced
        # channels-last custom op and registers custom ops in a scoped library, one returning a
        # freshly allocated contiguous result and one writing into an out argument. Checks the
        # compiled function (fullgraph, using the custom post-grad pass) equals eager, that the
        # pattern was applied, i.e. strides are preserved around the Triton kernel.
        import triton
        import triton.language as tl

        # [FUNCTION DOC] HELPER: Local masked elementwise add kernel.
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        x = torch.randn(4, 4, 2, 2, device=GPU_TYPE)
        other = torch.randn(4, 4, 2, 2, device=GPU_TYPE)

        def add_triton(y, z):
            # [FUNCTION DOC] HELPER: Allocates a contiguous empty tensor like the second input
            # and runs the add kernel with block 16; returns the output.
            grid = (z.numel(),)
            out = torch.empty_like(z, memory_format=torch.contiguous_format)
            add_kernel[grid](y, z, out, z.numel(), BLOCK_SIZE=16)
            return out

        class _CustomPass(PatternMatcherPass):
            # [CLASS DOC] HELPER: Pattern matcher pass subclass whose call applies its registered
            # patterns to the graph.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent constructor with no extra state.
                super().__init__()

            def __call__(self, g: torch.fx.Graph):
                # [FUNCTION DOC] HELPER: Applies the pass's patterns to the given graph.
                self.apply(g)

        g = _CustomPass()
        called = False

        @register_graph_pattern(
            CallFunctionVarArgs(torch.ops.aten.permute),
            pass_dict=g,
        )
        def _(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Pattern handler for permute nodes: flattens the matched
            # args, defines a replacement that applies permute then the force-channels-last
            # custom op, marks the outer flag as called, and replaces the match by that example.
            flat_args, spec = pytree.tree_flatten((args, kwargs))

            def decomp(*flat_args):
                # [FUNCTION DOC] HELPER: Rebuilds the original args and kwargs from the flat list
                # and returns the permute result passed through the force-channels-last op.
                args, kwargs = pytree.tree_unflatten(flat_args, spec)
                return torch.ops.mylib.force_channels_last(
                    torch.ops.aten.permute(*args, **kwargs)
                )

            nonlocal called
            called = True
            match.replace_by_example(decomp, flat_args)

        from torch._inductor import config

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "force_channels_last(Tensor x) -> Tensor",
                tags=[torch._C.Tag.flexible_layout],
            )

            def impl2(x):
                # [FUNCTION DOC] HELPER: Implementation of the custom op returning a clone in
                # channels-last memory format.
                return x.clone(memory_format=torch.channels_last)

            lib.impl("force_channels_last", impl2, "CompositeExplicitAutograd")

            lib.define(
                "add_op(Tensor x, Tensor y) -> Tensor",
            )

            def impl(x, y):
                # [FUNCTION DOC] HELPER: Implementation of the add
                # op that runs the Triton add wrapper.
                return add_triton(x, y)

            def meta(x, y):
                # [FUNCTION DOC] HELPER: Meta implementation returning an empty contiguous tensor
                # like the second input.
                return torch.empty_like(y, memory_format=torch.contiguous_format)

            lib.impl("add_op", impl, "CompositeExplicitAutograd")
            lib.impl("add_op", meta, "Meta")

            lib.define(
                "add_out_op(Tensor x, Tensor y, Tensor(a!) out) -> ()",
            )

            def impl_out(x, y, out):
                # [FUNCTION DOC] HELPER: Implementation of the out-argument op; launches the add
                # kernel writing into out with block 16.
                grid = (y.numel(),)
                add_kernel[grid](x, y, out, y.numel(), BLOCK_SIZE=16)

            lib.impl("add_out_op", impl_out, "CompositeExplicitAutograd")
            lib.impl("add_out_op", lambda x, y, out: None, "Meta")

            def f(x, other):
                # [FUNCTION DOC] HELPER: Builds a transposed, contiguous, transposed-back tensor
                # and a sine of it transposed; depending on the variant, returns the Triton add,
                # the custom op result, or the output filled by the mutable op; raises an
                # AssertionError for unknown variants.
                y = x.transpose(2, 3).contiguous().transpose(2, 3)
                z = y.sin().transpose(2, 3)
                if variant == "triton_kernel":
                    return add_triton(y, z)
                elif variant == "custom_op":
                    return torch.ops.mylib.add_op.default(y, z)
                elif variant == "mutable_custom_op":
                    out = torch.empty_like(y, memory_format=torch.contiguous_format)
                    torch.ops.mylib.add_out_op(y, z, out)
                    return out
                else:
                    raise AssertionError("should not be hit")

            with config.patch(
                post_grad_custom_post_pass=g,
            ):
                f_compile = torch.compile(f, fullgraph=True)
                self.assertEqual(f(x, other), f_compile(x, other))
                self.assertTrue(called)

    @requires_gpu
    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("autotune", [False, True])
    def test_capture_triton_special_kwargs(self, dynamic, autotune):
        # [FUNCTION DOC] Parametrized over dynamic and autotune. Checks that the capture wrapper
        # accepts warps and stages launch kwargs (with block size given directly or via an
        # autotuner). Eager result equals the sum and the graph traced by make_fx (symbolic or
        # fake mode) also computes the sum.
        # [FUNCTION DOC] HELPER: Local masked elementwise add kernel.
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        if autotune:
            add_kernel = triton.autotune(
                configs=[
                    triton.Config({"BLOCK_SIZE": 128}),
                    triton.Config({"BLOCK_SIZE": 64}),
                ],
                key=["n_elements"],
            )(add_kernel)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Zero output; launches the captured kernel with warps 8 and
            # stages 3 and, when not autotuned, block 128; returns the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            if autotune:
                kwargs = {}
            else:
                kwargs = {"BLOCK_SIZE": 128}
            capture_triton(add_kernel)[grid](
                x,
                y,
                output,
                n_elements,
                num_warps=8,
                num_stages=3,
                **kwargs,
            )
            return output

        x = torch.randn(4, device=GPU_TYPE)
        tracing_mode = "symbolic" if dynamic else "fake"

        result = f(x, x)
        self.assertEqual(result, x + x)

        from torch.fx.experimental.proxy_tensor import make_fx

        gm = make_fx(f, tracing_mode=tracing_mode)(x, x)
        self.assertEqual(gm(x, x), x + x)

    @skipIfWindows(msg="AOTI/Cpp_Wrapper have not enabled on Windows")
    @requires_gpu
    @inductor_config.patch("cpp_wrapper", True)
    @inductor_config.patch("triton.autotune_at_compile_time", True)
    def test_autotune_unbacked(self):
        # [FUNCTION DOC] Skipped on Windows; GPU only, with the cpp wrapper and compile-time
        # autotuning on. Checks the grid expressions emitted during compile-time autotuning for
        # an unbacked first dimension: the log must contain the compile-time auto-tuning marker,
        # a precomputed grid, and ceiling divisions by 32 and 128 of the launcher symbol (one per
        # config block size).
        import triton
        import triton.language as tl

        def get_op_configs():
            # [FUNCTION DOC] HELPER: Returns two autotune configs with different block sizes,
            # stage counts and warp counts.
            return [
                triton.Config(
                    {
                        "BLOCK_M": 32,
                        "BLOCK_N": 64,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=5,
                    num_warps=2,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 256,
                        "BLOCK_K": 64,
                        "GROUP_M": 8,
                    },
                    num_stages=3,
                    num_warps=8,
                ),
            ]

        # [FUNCTION DOC] HELPER: Autotuned (keyed on the matrix sizes) kernel that writes zeros
        # into a tile of the output with grouped program ordering; reads nothing.
        @triton.autotune(
            configs=get_op_configs(),
            key=["N", "K"],
        )
        @triton.jit
        def op_zeros(
            x_ptr,
            w_ptr,
            z_ptr,
            M,
            N,
            K,
            stride_xm,
            stride_xk,
            stride_wk,
            stride_wn,
            stride_zm,
            stride_zn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr,
            ALLOW_TF32: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_m = tl.arange(0, BLOCK_M)
            offs_n = tl.arange(0, BLOCK_N)
            mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
            mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N

            z_mask = mask_m & mask_n
            z = 0.0
            z_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_zm
            z_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_zn
            z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
            tl.store(z_ptrs, z, mask=z_mask)

        @torch.compile()
        def foo(x, w):
            # [FUNCTION DOC] HELPER: Compiled function that checks inner
            # dimensions agree (raising an AssertionError otherwise), allocates
            # an uninitialized output and launches the zeros kernel with a grid
            # by cdiv of the block sizes; returns the output.
            M, K = x.shape
            KB, N = w.shape
            assert K == KB, f"incompatible dimensions {K}, {KB}"  # noqa: S101

            z = torch.empty((M, N), device=x.device, dtype=x.dtype)

            def grid(META):
                # [FUNCTION DOC] HELPER: Returns a one-element grid with the product of the
                # ceiling divisions of the sizes by the block meta parameters.
                return (
                    triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
                )

            op_zeros[grid](
                x,
                w,
                z,
                M,
                N,
                K,
                x.stride(0),
                x.stride(1),
                w.stride(0),
                w.stride(1),
                z.stride(0),
                z.stride(1),
                ALLOW_TF32=USE_TF32,
            )
            return z

        M, K, N = 128, 64, 32
        x = torch.randn(M, K, device=GPU_TYPE)
        w = torch.randn(K, N, device=GPU_TYPE)

        torch._dynamo.decorators.mark_unbacked(x, 0)

        with (
            log_settings("+output_code"),
            self.assertLogs(logger="torch._inductor", level=logging.DEBUG) as log,
        ):
            foo(x, w)

        output = "\n".join(record.getMessage() for record in log.records)
        # correct grid example values updated per block size
        FileCheck().check("Compile-time auto-tuning block:").check(
            "PrecomputedGrid"
        ).check("(31 + _launcher_s0) // 32").check("(127 + _launcher_s0) // 128").run(
            output
        )

    # Triton 3.2.0 adds the required flags to the Autotuner object for this test
    # PR: https://github.com/triton-lang/triton/pull/5092
    @requires_gpu
    def test_autotune_no_pre_or_post_hook_user_defined(self):
        # [FUNCTION DOC] Checks that a kernel autotuned with pre and post hooks (and a per-config
        # pre hook) is flagged on the Autotuner object (skipped for old triton lacking the flag
        # attributes) and that compiling it raises dynamo Unsupported with the message that hooks
        # are not supported. The eager call must work first and give an output of all twos.
        from triton.runtime.autotuner import Autotuner

        def init_to_zero(name):
            # [FUNCTION DOC] HELPER: Returns a hook that zeroes the named tensor from the hook's
            # argument mapping in place.
            return lambda nargs: nargs[name].zero_()

        # [FUNCTION DOC] HELPER: Autotuned kernel (hooks reset the output) that atomically adds
        # the sum of two inputs into the output.
        @triton.autotune(
            configs=[
                triton.Config(
                    {"BLOCK_SIZE": 1024},
                    num_warps=4,
                    num_stages=2,
                    pre_hook=init_to_zero("output_ptr"),
                )
            ],
            pre_hook=init_to_zero("output_ptr"),
            post_hook=init_to_zero("output_ptr"),
            key=["n_elements"],
        )
        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)

            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.atomic_add(output_ptr + offsets, output, mask=mask)

        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Output of ones; launches the kernel with callable grid;
            # returns the output.
            output = torch.ones(x.shape, device=x.device, dtype=x.dtype)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, output, n_elements)
            return output

        x = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)
        y = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)

        # should always pass
        if add(x, y).mean() != 2:
            raise AssertionError("Problem with add kernel")

        # assert that the user_defined_* flags are properly set on the kernel before compilation
        self.assertEqual(isinstance(add_kernel, Autotuner), True)
        if not hasattr(add_kernel, "user_defined_pre_hook") or not hasattr(
            add_kernel, "user_defined_post_hook"
        ):
            raise unittest.SkipTest(
                "test requires Triton version >= 3.2.0 for Autotuner.user_defined* hooks"
            )

        self.assertEqual(add_kernel.user_defined_pre_hook, True)
        self.assertEqual(add_kernel.user_defined_post_hook, True)

        # this should cause an exception, since pre_hook is not allowed
        msg = "pre_hook and post_hook are not supported in triton.Autotune or triton.Config"
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            add_compiled = torch.compile(add, mode="reduce-overhead", fullgraph=True)
            add_compiled(x, y).mean()

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("autotune_at_compile_time", [True, False])
    def test_triton_kernel_reset_to_zero(self, backend, autotune_at_compile_time):
        # [FUNCTION DOC] Parametrized over backend and compile-time autotuning (skipped for non-
        # inductor with it). Checks reset_to_zero on an autotuned kernel: the increment pointer
        # is zeroed before each autotune benchmark so that the input is only incremented once by
        # the real run; after the call the input equals the original plus the increment.
        if autotune_at_compile_time and backend != "inductor":
            raise unittest.SkipTest("compile-time autotuning only exists in inductor")

        # [FUNCTION DOC] HELPER: Autotuned (three configs, reset the increment pointer to zero)
        # kernel adding the loaded increment values to the input in place.
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=8),
                triton.Config({"BLOCK_SIZE": 32}, num_stages=3, num_warps=8),
                triton.Config({"BLOCK_SIZE": 16}, num_stages=3, num_warps=8),
            ],
            key=[],
            reset_to_zero=["increment_ptr"],
        )
        @triton.jit
        def increment_kernel(
            in_ptr0,
            increment_ptr,  # reset this to zero every time
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            in_ptr_vals = tl.load(in_ptr0 + offsets, mask=mask)
            increment_val = tl.load(increment_ptr + offsets, mask=mask)
            # increment_val should always be zero
            tl.store(in_ptr0 + offsets, in_ptr_vals + increment_val, mask=mask)

        @torch.compile(fullgraph=True, backend=backend)
        def f(x, increment):
            # [FUNCTION DOC] HELPER: Compiled fullgraph function launching the increment kernel
            # on x; returns x.
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            increment_kernel[grid](x, increment, n_elements=n_elements)
            return x

        x = torch.rand(4, device=GPU_TYPE)
        y = torch.clone(x)
        increment = torch.rand(4, device=GPU_TYPE)

        # during autotuning, x should not change in value
        with inductor_config.patch(
            {"triton.autotune_at_compile_time": autotune_at_compile_time}
        ):
            # we will add rand a single time to x
            f(x, increment)

        self.assertEqual(y + increment, x)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_single_autotune(self, backend):
        # [FUNCTION DOC] Parametrized on backend. Checks that stacking two autotune decorators on
        # a kernel raises dynamo Unsupported with the message that multiple autotune decorators
        # are not supported.
        # [FUNCTION DOC] HELPER: Add kernel decorated by two autotune decorators; per the comment
        # only one is supported.
        # KNOWN ISSUE: the comment before the expected exception says pre_hook is not allowed,
        # copied from another test; the actual message concerns multiple autotune decorators.
        @triton.autotune(
            configs=[
                triton.Config(
                    {"BLOCK_SIZE": 4096},
                )
            ],
            key=["n_elements"],
        )
        # Currently, this autotuning decorator will never run!
        # We only support having a single autotuning decorator on each Triton kernel
        @triton.autotune(
            configs=[
                triton.Config(
                    {"BLOCK_SIZE": 1024},
                )
            ],
            key=["n_elements"],
        )
        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)

            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Output of ones; launches the add kernel with callable grid;
            # returns the output.
            output = torch.ones(x.shape, device=x.device, dtype=x.dtype)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, output, n_elements)
            return output

        x = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)
        y = torch.ones((4096,), device=GPU_TYPE, dtype=torch.float16)

        # this should cause an exception, since pre_hook is not allowed
        msg = "Passing multiple @triton.autotune decorators is not supported. Please use a single @triton.autotune decorator instead."
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            add_compiled = torch.compile(
                add, mode="reduce-overhead", fullgraph=True, backend=backend
            )
            add_compiled(x, y).mean()

    @requires_gpu
    @common_utils.parametrize("non_strict", [True, False])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("with_perf_model", [True, False])
    def test_triton_kernel_prune_configs_by(self, backend, with_perf_model, non_strict):
        # [FUNCTION DOC] Parametrized over backend, perf-model use and non-strict (triton op with
        # wrap_triton) mode. Builds an autotuned kernel pruned either by a perf model keeping the
        # top one or by an early config prune hook that keeps the first config, which records
        # calls in a dict. With the perf model the largest block is chosen, one record exists and
        # destination equals source plus 1.5. Without it block 32 is chosen, destination equals
        # source, and three records show the hook ran with the size kwarg and the named args.
        # for non-strict mode
        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        records = {}

        def early_config_prune(configs, named_args, **kwargs):
            # [FUNCTION DOC] HELPER: Early prune hook that records that it ran, records if the
            # size kwarg is 1024 and if named args hold destination and source with three
            # entries, and returns only the first config.
            # we need to save the records to the returned config
            records["run_early_config_prune"] = True
            if "N" in kwargs and kwargs["N"] == 1024:
                records["capture_kwargs"] = True
            # named args are: dst, src, add_float
            if "dst" in named_args and "src" in named_args and len(named_args) == 3:
                records["capture_named_args"] = True
            return [configs[0]]

        def perf_model(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Perf model hook that records it ran and returns the negative
            # of the block size, making larger blocks better.
            records["run_perf_model"] = True
            return kwargs["BLOCK_SIZE"] * -1

        if with_perf_model:
            prune_configs_by = {"perf_model": perf_model, "top_k": 1}
        else:
            prune_configs_by = {"early_config_prune": early_config_prune}

        # [FUNCTION DOC] HELPER: Autotuned (two block sizes, pruned) kernel that copies the
        # source to the destination, adding a float when block size is 128.
        @triton.autotune(
            configs=[
                triton.Config(kwargs={"BLOCK_SIZE": 32}),
                triton.Config(kwargs={"BLOCK_SIZE": 128}),
            ],
            key=["N"],
            prune_configs_by=prune_configs_by,
        )
        @triton.jit
        def prune_by_kernel(
            dst,
            src,
            add_float,
            N,
            BLOCK_SIZE: tl.constexpr,
        ):
            offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            x = tl.load(src + offsets, mask=offsets < N)
            # we only modify dst if our perf_model is applied (and a BLOCK_SIZE of 128 is selected)
            if BLOCK_SIZE == 128:
                x = x + add_float
            tl.store(dst + offsets, x, mask=offsets < N)

        def f(
            dst: torch.Tensor,
            src: torch.Tensor,
            add_float: float,
            N: int,
        ) -> None:
            # [FUNCTION DOC] HELPER: Launches the pruned kernel through wrap_triton (non-strict)
            # or directly, with a callable grid by cdiv of the size.
            grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
            if non_strict:
                torch.library.wrap_triton(prune_by_kernel)[grid](
                    dst, src, add_float, N=N
                )
            else:
                prune_by_kernel[grid](dst, src, add_float, N=N)

        if non_strict:
            decorator = torch.library.triton_op(
                f"{libname}::{opname}", mutates_args={"dst"}
            )(f)
        else:
            # we can just pass the function 'f' for dynamo
            decorator = f

        compiled_f = torch.compile(decorator, backend=backend)
        N = 1024
        src = torch.randn(N, device=GPU_TYPE)
        dst = torch.empty(N, device=GPU_TYPE)
        compiled_f(dst, src, 1.5, N)

        if with_perf_model:
            # when applying the perf_model: kwargs["BLOCK_SIZE"] * -1, the largest config (BLOCK_SIZE==128) is selected
            self.assertEqual(len(records), 1)
            self.assertEqual(src + 1.5, dst)
        else:
            # without the perf_model, the BLOCK_SIZE==32, and as a result dst is not modified and remains equal to src
            self.assertEqual(src, dst)
            self.assertEqual(len(records), 3)
            self.assertTrue(records["run_early_config_prune"])
            self.assertTrue(records["capture_kwargs"])
            self.assertTrue(records["capture_named_args"])

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("with_perf_model", [True, False])
    def test_triton_kernel_prune_configs_by_recompile(self, backend, with_perf_model):
        # [FUNCTION DOC] Parametrized over backend and perf-model use. Checks guards: the compile
        # counter's op count stays one on repeated calls, becomes two after the autotuner's
        # configs are replaced with a single block-64 config, and stays two on a further call,
        # i.e. changing autotuner configs triggers a recompile. Resets dynamo first.
        """
        We want to recompile if anyone changes configs in the autotuner object
        In short if for example the following sequence of events happens:
        1. foo = torch.compile(bar)
        1. call foo
        2. autotuner.configs = [new configs list]
        3. call foo

        A recompile event should occur, which we check with Dynamo counters
        This tests that we are installing guards on input objects properly
        """

        # We don't modify records here because we are testing whether or not
        # recompiles occur/guards are installed
        # If we modified the non-local records dict here, this would trigger
        # recompile events.
        def early_config_prune(configs, named_args, **kwargs):
            # [FUNCTION DOC] HELPER: Early prune hook returning only the first config;
            # deliberately records nothing to avoid triggering recompiles.
            return [configs[0]]

        def perf_model(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Perf model returning the
            # negative block size; records nothing.
            return kwargs["BLOCK_SIZE"] * -1

        if with_perf_model:
            prune_configs_by = {"perf_model": perf_model, "top_k": 1}
        else:
            prune_configs_by = {"early_config_prune": early_config_prune}

        # [FUNCTION DOC] HELPER: Autotuned (two block sizes, pruned) kernel that copies the
        # source to the destination, adding a float when block size is 128.
        @triton.autotune(
            configs=[
                triton.Config(kwargs={"BLOCK_SIZE": 32}),
                triton.Config(kwargs={"BLOCK_SIZE": 128}),
            ],
            key=["N"],
            prune_configs_by=prune_configs_by,
        )
        @triton.jit
        def prune_by_kernel(
            dst,
            src,
            add_float,
            N,
            BLOCK_SIZE: tl.constexpr,
        ):
            offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            x = tl.load(src + offsets, mask=offsets < N)
            # Let's make sure we always select a block size of 128 based on our perf_model
            if BLOCK_SIZE == 128:
                x = x + add_float
            tl.store(dst + offsets, x, mask=offsets < N)

        torch._dynamo.reset()
        counter = torch._dynamo.testing.CompileCounterWithBackend(backend=backend)

        @torch.compile(fullgraph=True, backend=counter)
        def f(dst, src, add_float, N):
            # [FUNCTION DOC] HELPER: Compiled (fullgraph, counting backend) function launching
            # the pruned kernel with a callable grid.
            grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
            prune_by_kernel[grid](dst, src, add_float, N=N)

        N = 1024
        src = torch.randn(N, device=GPU_TYPE)
        dst = torch.empty(N, device=GPU_TYPE)

        # first compilation, this prunes the configs
        f(dst, src, 1.5, N)

        self.assertEqual(counter.op_count, 1)

        f(dst, src, 1.5, N)

        # this should not trigger a recompilation
        # this is because we modified the test to not touch the records dict
        # as we do in test_triton_kernel_prune_configs_by. If we kept it, it would trigger a recompile here.
        self.assertEqual(counter.op_count, 1)

        # Modify the autotuner object
        prune_by_kernel.configs = [triton.Config(kwargs={"BLOCK_SIZE": 64})]

        # Calling the kernel after modifying the autotuner should
        # trigger a recompile
        f(dst, src, 1.5, N)

        self.assertEqual(counter.op_count, 2)

        # there should be no recompile here
        f(dst, src, 1.5, N)

        self.assertEqual(counter.op_count, 2)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_prune_configs_by_called_twice(self, backend):
        # [FUNCTION DOC] Parametrized on backend. Checks that an autotuned kernel with an early
        # prune hook can be launched twice in one compiled function (the second call using the
        # first output) and that the result equals the input plus the second input twice.
        def early_config_prune(configs, named_args, **kwargs):
            # [FUNCTION DOC] HELPER: Early prune hook returning only the first config.
            return [configs[0]]

        # [FUNCTION DOC] HELPER: Autotuned (two block sizes, pruned) elementwise add kernel with
        # a mask.
        @triton.autotune(
            configs=[
                triton.Config(kwargs={"BLOCK_SIZE": 128}),
                triton.Config(kwargs={"BLOCK_SIZE": 256}),
            ],
            key=["N"],
            prune_configs_by={"early_config_prune": early_config_prune},
        )
        @triton.jit
        def add_kernel(
            x_ptr,
            y_ptr,
            out_ptr,
            N,
            BLOCK_SIZE: tl.constexpr,
        ):
            offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        def call_kernel(x, y):
            # [FUNCTION DOC] HELPER: Empty output; launches the add kernel with a callable grid;
            # returns the output.
            out = torch.empty_like(x)
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_kernel[grid](x, y, out, N=n_elements)
            return out

        @torch.compile(fullgraph=True, backend=backend)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls the kernel helper on x and y, then again on that
            # output and y; returns the result.
            out = call_kernel(x, y)
            return call_kernel(out, y)

        x = torch.randn(1024, device=GPU_TYPE)
        y = torch.randn(1024, device=GPU_TYPE)

        self.assertEqual(f(x, y), x + y + y)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_heuristics_and_prune_configs_by(self, backend):
        # [FUNCTION DOC] Parametrized on backend. Checks that an autotuner combined with a
        # heuristics decorator and a no-op prune hook works with fullgraph compile: the kernel
        # doubles its input, and the compiled result equals twice the input.
        def noop_prune(configs, named_args, **kwargs):
            # [FUNCTION DOC] HELPER: Prune hook that returns all configs unchanged.
            return configs

        # [FUNCTION DOC] HELPER: Kernel decorated by autotune (two block sizes, no-op pruning)
        # and a heuristics decorator for an even-size flag; stores twice the masked input.
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK": 128}),
                triton.Config({"BLOCK": 256}),
            ],
            key=["N"],
            prune_configs_by={"early_config_prune": noop_prune},
        )
        @triton.heuristics({"EVEN": lambda args: args["N"] % 128 == 0})
        @triton.jit
        def kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2, mask=mask)

        def f(x):
            # [FUNCTION DOC] HELPER: Empty output; launches the kernel with a grid of cdiv of the
            # size by 128; returns the output.
            out = torch.empty_like(x)
            kernel[(triton.cdiv(x.numel(), 128),)](x, out, x.numel())
            return out

        compiled_f = torch.compile(f, backend=backend, fullgraph=True)
        x = torch.randn(1024, device=GPU_TYPE)
        self.assertEqual(compiled_f(x), x * 2)

    @requires_gpu
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_triton_kernel_run_with_prune_configs_by(self, backend):
        # [FUNCTION DOC] Parametrized on backend. Checks the run method of an autotuned kernel
        # with a no-op prune hook under fullgraph compile: the output equals the input plus one.
        def noop_prune(configs, named_args, **kwargs):
            # [FUNCTION DOC] HELPER: Prune hook that returns all configs unchanged.
            return configs

        # [FUNCTION DOC] HELPER: Autotuned (two block sizes, no-op pruning) kernel storing the
        # masked input plus one.
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK": 64}),
                triton.Config({"BLOCK": 128}),
            ],
            key=["N"],
            prune_configs_by={"early_config_prune": noop_prune},
        )
        @triton.jit
        def kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

        def f(x):
            # [FUNCTION DOC] HELPER: Empty output; calls the kernel run method with a grid of
            # cdiv by 64 and no warmup; returns the output.
            out = torch.empty_like(x)
            kernel.run(
                x,
                out,
                x.numel(),
                grid=(triton.cdiv(x.numel(), 64),),
                warmup=False,
            )
            return out

        compiled_f = torch.compile(f, backend=backend, fullgraph=True)
        x = torch.randn(1024, device=GPU_TYPE)
        self.assertEqual(compiled_f(x), x + 1)

    # see: https://github.com/triton-lang/triton/blob/67ea999935f4511a535a25bdecb27e79e3c3af41/python/test/unit/language/test_decorator.py#L31
    @requires_gpu
    @common_utils.parametrize("non_strict", [True, False])
    @common_utils.parametrize("backend", ["eager", "aot_eager", "inductor"])
    @common_utils.parametrize("autotune_at_compile_time", [True, False])
    def test_triton_kernel_heuristic(
        self, backend, autotune_at_compile_time, non_strict
    ):
        # [FUNCTION DOC] Parametrized over backend, compile-time autotuning and non-strict
        # mode (triton op). Checks that heuristics decorators are applied in order
        # (including modifying an existing key, chained updates, and a flag from a None
        # check) in dynamo: the compiled results must match direct Triton (first element
        # 2120 and second zero). For direct launches it also redefines the kernel with a
        # heuristic returning a non-constant tensor and expects dynamo Unsupported with the
        # message that heuristics must return constants.
        # for non-strict mode
        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        # [FUNCTION DOC] HELPER: Kernel with autotune (block 32) plus several heuristics
        # decorators; stores the sum of the even-N and block size constexprs and the None-flag.
        @triton.autotune(
            configs=[
                triton.Config(kwargs={"BLOCK_SIZE": 32}),
            ],
            key=["N"],
        )
        # we should be able to modify existing keys in kwargs
        @triton.heuristics({"BLOCK_SIZE": lambda nargs: nargs["BLOCK_SIZE"] * 2})
        # test kwargs
        @triton.heuristics({"EVEN_N": lambda nargs: nargs["N"] + 10})
        @triton.heuristics({"EVEN_N": lambda nargs: nargs["EVEN_N"] * 2})
        # test args
        # There are differences here from OSS Triton because we run these functions in Dynamo
        # We don't have access to the .data_ptr() of TensorVariables
        @triton.heuristics({"NDIM_src": lambda nargs: nargs["src"] is None})
        # test that heuristics are applied in the correct order
        @triton.heuristics({"EVEN_N": lambda nargs: nargs["EVEN_N"] - 10})
        @triton.jit
        def heuristics_kernel(
            dst,
            src,
            N,
            BLOCK_SIZE: tl.constexpr,
            EVEN_N: tl.constexpr,
            NDIM_src: tl.constexpr,
        ):
            tl.store(dst, EVEN_N + BLOCK_SIZE)
            tl.store(dst + 1, NDIM_src)

        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)

        def f(
            dst: torch.Tensor,
            src: torch.Tensor,
            N: int,
        ) -> None:
            # [FUNCTION DOC] HELPER: Launches the heuristics kernel through wrap_triton (non-
            # strict) or directly with a grid by cdiv of N.
            grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
            if non_strict:
                torch.library.wrap_triton(heuristics_kernel)[grid](dst, src, N=N)
            else:
                heuristics_kernel[grid](dst, src, N=N)

        if non_strict:
            decorator = torch.library.triton_op(
                f"{libname}::{opname}", mutates_args={"dst"}
            )(f)
        else:
            # we can just pass the function 'f' for dynamo
            decorator = f

        compiled_f = torch.compile(decorator, backend=backend)

        N = 1023
        src = torch.empty(N, device=GPU_TYPE)
        dst = torch.zeros(N, device=GPU_TYPE)

        with inductor_config.patch(
            {"triton.autotune_at_compile_time": autotune_at_compile_time}
        ):
            compiled_f(dst, src, N=N)

        # now let's run without torch.compile to compare
        triton_src = torch.empty(N, device=GPU_TYPE)
        triton_dst = torch.zeros(N, device=GPU_TYPE)
        heuristics_kernel[grid](triton_dst, triton_src, N=N)

        # triton_dst[0].item() is 2120
        # (1023 + 10) * 2 - 10 + BLOCK_SIZE = 2056 + 64 = 2120
        # this is to test that we apply the heuristics in the correct order
        self.assertEqual(triton_dst[0].item(), 2120)
        self.assertEqual(triton_dst[1].item(), 0.0)

        # Results should match
        self.assertEqual(dst[0].item(), triton_dst[0].item())
        self.assertEqual(dst[1].item(), triton_dst[1].item())

        # @triton.heuristics cannot return non-constant values
        # check for the exception
        if not non_strict:

            # [FUNCTION DOC] HELPER: Redefined kernel whose heuristic returns a non-constant
            # tensor value; stores N.
            @triton.autotune(
                configs=[
                    triton.Config(kwargs={"BLOCK_SIZE": 32}),
                ],
                key=["N"],
            )
            # torch.randint(...)[0] will produce a non-constant value
            @triton.heuristics({"EVEN_N": lambda nargs: torch.randint(1, (1, 1))[0]})
            @triton.jit
            def heuristics_kernel(
                dst,
                src,
                N,
                BLOCK_SIZE: tl.constexpr,
                EVEN_N: tl.constexpr,
            ):
                tl.store(dst, N)

            grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)

            def f(
                dst: torch.Tensor,
                src: torch.Tensor,
                N: int,
            ) -> None:
                # [FUNCTION DOC] HELPER: Launches the redefined kernel with a callable grid by
                # cdiv of N.
                grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
                heuristics_kernel[grid](dst, src, N=N)

            compiled_f = torch.compile(f, backend=backend, fullgraph=True)
            N = 1023
            src = torch.empty(N, device=GPU_TYPE)
            dst = torch.zeros(N, device=GPU_TYPE)
            msg = "@triton.heuristics must return constant values because configs can only contain constant values."
            with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
                compiled_f(dst, src, N=N)


class TestUserKernelEpilogueFusion(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Tests of epilogue fusion into user-defined Triton kernels (config flag enabled
    # for the class). Each test compiles a function calling a user kernel followed by a pointwise
    # op, and checks numerics plus kernel, allocation and deletion counts in the generated code.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: calls the parent setup, creates an exit stack stored on the
        # class, and enables the config for epilogue fusion of user-defined kernels.
        # KNOWN ISSUE: the exit stack is never used or closed, and teardown sets the flag to
        # false instead of restoring its previous value.
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        torch._inductor.config.epilogue_fusion_user_defined_triton_kernel = True

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: disables the epilogue fusion for user-defined kernels
        # config and calls the parent teardown.
        torch._inductor.config.epilogue_fusion_user_defined_triton_kernel = False
        super().tearDownClass()

    def check_code(self, code_str, num_kernels, num_allocs, num_deallocs):
        # [FUNCTION DOC] HELPER: Asserts on generated code text, using FileCheck after
        # the call function marker: exactly the given number of kernel launches (call
        # prefix for cpp wrapper, run otherwise); if given, the exact count of
        # empty_strided allocations; and, unless using the cpp wrapper or when none is
        # given, the exact count of delete statements.
        from torch._inductor import config

        kernel_launch = "call_" if config.cpp_wrapper else ".run("
        FileCheck().check(get_func_call()).check_count(
            kernel_launch,
            num_kernels,
            exactly=True,
        ).run(code_str)

        if num_allocs is not None:
            FileCheck().check(get_func_call()).check_count(
                "empty_strided", num_allocs, exactly=True
            ).run(code_str)

        # skip the deallocation check when using cpp_wrapper; most deallocations happen
        # outside of our control via RAIIAtenTensorHandle
        if num_deallocs is not None and not config.cpp_wrapper:
            FileCheck().check(get_func_call()).check_count(
                "del", num_deallocs, exactly=True
            ).run(code_str)

    @requires_cuda_and_triton
    def test_fusion_relu_epilogue(self):
        # [FUNCTION DOC] CUDA only. A user add kernel followed by relu should fuse into one
        # kernel (one allocation, two deletions); output equals eager within tolerance.
        # [FUNCTION DOC] HELPER: Masked elementwise add kernel.
        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Empty output; launches the add kernel; returns the relu of
            # the output.
            out = torch.empty_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
            return out.relu()

        a = torch.tensor([-0.3] * 8, dtype=torch.float32, device="cuda")
        b = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32, device="cuda"
        )

        out, code = run_and_get_code(torch.compile(fn), a, b)
        self.assertEqual(out, fn(a, b), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

    @requires_cuda_and_triton
    def test_fusion_sigmoid_epilogue(self):
        # [FUNCTION DOC] CUDA only. A user kernel doubling its input followed by sigmoid should
        # fuse into one kernel (one allocation, one deletion).
        # [FUNCTION DOC] HELPER: Kernel storing the input added to itself, masked.
        @triton.jit
        def add_kernel(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            tl.store(out_ptr + offs, x + x, mask=mask)

        def fn(a):
            # [FUNCTION DOC] HELPER: Empty output; launches the kernel; returns the sigmoid of
            # the output.
            out = torch.empty_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, out, a.numel(), BLOCK_SIZE=1024)
            return out.sigmoid()

        a = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32, device="cuda"
        )

        out, code = run_and_get_code(torch.compile(fn), a)
        self.assertEqual(out, fn(a), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=1)

    @requires_cuda_and_triton
    def test_fusion_with_unused_buffer(self):
        # [FUNCTION DOC] CUDA only. A kernel with an extra unused buffer argument followed by
        # relu fuses to one kernel with two allocations and two deletions; output equals eager.
        # [FUNCTION DOC] HELPER: Kernel storing twice the masked input and ignoring its third
        # pointer argument.
        @triton.jit
        def kernel_unused_tensor(
            X,
            Y,
            unused,
            n: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = idx < n
            x = tl.load(X + idx, mask=mask)
            tl.store(Y + idx, x * 2, mask=mask)

        def fn(a):
            # [FUNCTION DOC] HELPER: Empty output and unused buffer; launches the kernel with
            # block 16; returns the relu of the output.
            out = torch.empty_like(a)
            unused = torch.empty(a.numel(), device="cuda")
            kernel_unused_tensor[(1,)](a, out, unused, a.numel(), BLOCK_SIZE=16)
            return out.relu()

        a = torch.randn(10, device="cuda")

        out, code = run_and_get_code(torch.compile(fn), a)
        self.assertEqual(out, fn(a))
        self.check_code(code[0], num_kernels=1, num_allocs=2, num_deallocs=2)

    @requires_cuda_and_triton
    def test_fusion_with_ordering_constraints(self):
        # [FUNCTION DOC] CUDA only. Two chained user kernel adds followed by relu: expects two
        # kernel launches (epilogue fused into the second), two allocations and three deletions.
        # [FUNCTION DOC] HELPER: Masked elementwise add kernel.
        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        def add(a, b):
            # [FUNCTION DOC] HELPER: Empty output; launches the add kernel with one program per
            # element and block 1; returns the output.
            out = torch.empty_like(a)
            add_kernel[(a.numel(),)](a, b, out, a.numel(), BLOCK_SIZE=1)
            return out

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds a and b, adds b to the result, and returns the relu.
            c = add(a, b)
            c = add(c, b)
            return c.relu()

        a = torch.randn(10, device="cuda")
        b = torch.randn(10, device="cuda")

        out, code = run_and_get_code(torch.compile(fn), a, b)
        self.assertEqual(out, fn(a, b))
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=3)

    @requires_cuda_and_triton
    def test_fusion_cache(self):
        # [FUNCTION DOC] CUDA only. Two chained user adds, each followed by a pointwise op
        # (sigmoid then relu); expects two kernel launches, two allocations and three deletions;
        # output equals eager.
        # [FUNCTION DOC] HELPER: Masked elementwise add kernel.
        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        def add(a, b):
            # [FUNCTION DOC] HELPER: Empty output; launches the add kernel with one program per
            # element and block 1; returns the output.
            out = torch.empty_like(a)
            add_kernel[(a.numel(),)](a, b, out, a.numel(), BLOCK_SIZE=1)
            return out

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds a and b and applies sigmoid, adds b to that and applies
            # relu, and returns the result.
            c = add(a, b).sigmoid()
            c = add(c, b).relu()
            return c

        a = torch.randn(10, device="cuda")
        b = torch.randn(10, device="cuda")

        out, code = run_and_get_code(torch.compile(fn), a, b)
        self.assertEqual(out, fn(a, b))
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=3)

    @requires_cuda_and_triton
    def test_fusion_custom_kernel_with_linebreaks(self):
        # [FUNCTION DOC] CUDA only. Checks that fusion handles a user kernel source with awkward
        # line breaks (blank line before the store and a multi-line value expression with
        # comments), which the source rewriting must handle. Expects one kernel, one allocation
        # and one deletion.
        # we do AST manipulation / string manipulation of the kernel source code
        # so we wanna make sure to correctly handle edge cases with tricky line breaks
        # which `ast.parse`-`ast.unparse` roundtrip does not recover
        # [FUNCTION DOC] HELPER: Kernel doubling its input with a store call split across lines
        # with comments.
        @triton.jit
        def add_kernel(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            # forcing newlines before `tl.store`

            tl.store(
                out_ptr + offs,
                # forcing the `value` arg to be written in multiple lines
                (
                    # forcing the `value` arg to be written in multiple lines
                    x + x
                ),
                mask=mask,
            )

        def fn(a):
            # [FUNCTION DOC] HELPER: Empty output; launches the kernel; returns the sigmoid of
            # the output.
            out = torch.empty_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, out, a.numel(), BLOCK_SIZE=1024)
            return out.sigmoid()

        a = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32, device="cuda"
        )

        out, code = run_and_get_code(torch.compile(fn), a)
        self.assertEqual(out, fn(a), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=1)

    @requires_cuda_and_triton
    def test_no_fusion_for_read_write_tensors(self):
        # [FUNCTION DOC] CUDA only. No fusion when the kernel reads the output before writing it:
        # expects two kernels (user kernel and relu), two allocations and three deletions.
        # cannot epilogue-fuse this one because the kernel reads from `out_ptr` before writing to it
        # [FUNCTION DOC] HELPER: Kernel that loads the two inputs and the original output and
        # stores their sum.
        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            original_out = tl.load(out_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y + original_out, mask=mask)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Empty output; launches the
            # kernel; returns the relu of the output.
            out = torch.empty_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
            return out.relu()

        a = torch.tensor([-0.3] * 8, dtype=torch.float32, device="cuda")
        b = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32, device="cuda"
        )

        _, code = run_and_get_code(torch.compile(fn), a, b)
        # no fusion, so 2 kernels: add_kernel, relu
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=3)

    @requires_cuda_and_triton
    def test_no_fusion_for_multiple_stores(self):
        # [FUNCTION DOC] CUDA only. No fusion when the kernel stores twice, once directly and
        # once through another jit function (defined at module level after the class because
        # local jit functions referencing each other are problematic). Expects two kernels, two
        # allocations, three deletions.
        # cannot epilogue-fuse this one because the kernel stores twice (once directly and once indirectly via another triton func)
        """
        We defined these globally after this class, because triton doesn't like local jit functions that reference each other..

        @triton.jit
        def custom_store(ptr, val, mask):
            tl.store(ptr, val, mask)

        @triton.jit
        def add_kernel_with_custom_store(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)
            custom_store(out_ptr + offs, x + y, mask=mask)
        """

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Empty output; launches the module-level double-store kernel;
            # returns the relu of the output.
            out = torch.empty_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel_with_custom_store[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
            return out.relu()

        a = torch.tensor([-0.3] * 8, dtype=torch.float32, device="cuda")
        b = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32, device="cuda"
        )

        _, code = run_and_get_code(torch.compile(fn), a, b)
        # no fusion, so 2 kernels: add_kernel, relu
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=3)

    @requires_cuda_and_triton
    def test_no_fusion_for_non_empty_outputs(self):
        # [FUNCTION DOC] CUDA only. No fusion when the output starts as ones and the kernel
        # writes only half of its indices (not an uninitialized buffer). Expects three kernels
        # (ones, user kernel, relu), two allocations, three deletions; output equals eager.
        # cannot epilogue-fuse this one because the kernel writes to buffer initialized with non-UB values
        # [FUNCTION DOC] HELPER: Kernel storing the sum at half the offsets.
        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.store(out_ptr + offs // 2, x + y, mask=mask)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Output of ones; launches the
            # kernel; returns the relu of the output.
            out = torch.ones_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
            return out.relu()

        a = torch.tensor([-0.3] * 8, dtype=torch.float32, device="cuda")
        b = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32, device="cuda"
        )

        out, code = run_and_get_code(torch.compile(fn), a, b)
        self.assertEqual(out, fn(a, b), atol=0.05, rtol=0.05)
        # no fusion, so 3 kernels: ones, add_kernel, relu
        self.check_code(code[0], num_kernels=3, num_allocs=2, num_deallocs=3)

    @requires_cuda_and_triton
    def test_no_fusion_for_concat_outputs(self):
        # [FUNCTION DOC] CUDA only. No fusion when the output is a concatenation of two buffers:
        # expects three kernels (concat, user kernel, relu), two allocations, three deletions.
        # [FUNCTION DOC] HELPER: Masked elementwise add kernel.
        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Allocates two half-size buffers, concatenates them, launches
            # the kernel into the result and returns its relu.
            out_first_half = torch.empty(int(len(a) / 2), dtype=a.dtype, device="cuda")
            out_second_half = torch.empty(int(len(a) / 2), dtype=a.dtype, device="cuda")
            out = torch.concat((out_first_half, out_second_half))
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
            return out.relu()

        a = torch.tensor([-0.3] * 8, dtype=torch.float32, device="cuda")
        b = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32, device="cuda"
        )

        out, code = run_and_get_code(torch.compile(fn), a, b)
        self.assertEqual(out, fn(a, b), atol=0.05, rtol=0.05)
        # no fusion, so 3 kernels: concat, add_kernel, relu
        self.check_code(code[0], num_kernels=3, num_allocs=2, num_deallocs=3)

    @requires_cuda_and_triton
    def test_no_fusion_for_multiple_reads_on_mutated_tensor(self):
        # [FUNCTION DOC] CUDA only. No fusion when the kernel output feeds two consumers (relu
        # and sigmoid): expects two kernels (user kernel and one generated for both), three
        # allocations, three deletions.
        # [FUNCTION DOC] HELPER: Masked elementwise add kernel.
        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Empty output; launches the kernel; returns both the relu and
            # sigmoid of the output.
            out = torch.empty_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
            return out.relu(), out.sigmoid()

        a = torch.tensor([-0.3] * 8, dtype=torch.float32, device="cuda")
        b = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32, device="cuda"
        )

        out, code = run_and_get_code(torch.compile(fn), a, b)
        self.assertEqual(out, fn(a, b), atol=0.05, rtol=0.05)
        # two kernels: user defined `add_kernel`, and the generated one for both sigmoid and relu
        self.check_code(code[0], num_kernels=2, num_allocs=3, num_deallocs=3)

    @requires_cuda_and_triton
    def test_no_fusion_for_atomic_store(self):
        # [FUNCTION DOC] CUDA only; returns immediately (passing) when not built
        # with CUDA because the atomic exchange fails to compile on ROCm. No
        # fusion for a kernel using an atomic exchange store: two kernels, two
        # allocations, three deletions; output equals eager.
        # on ROCm we skip this because `tl.atomic_xchg` fails to compile in ROCm
        if torch.version.cuda is None:
            return

        # [FUNCTION DOC] HELPER: Kernel storing the sum using an atomic exchange.
        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.atomic_xchg(out_ptr + offs, x + y, mask=mask)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Empty output; launches the
            # kernel; returns the relu of the output.
            out = torch.empty_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
            return out.relu()

        a = torch.tensor([-0.3] * 8, dtype=torch.float32, device="cuda")
        b = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32, device="cuda"
        )

        out, code = run_and_get_code(torch.compile(fn), a, b)
        self.assertEqual(out, fn(a, b), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=3)

    @requires_cuda_and_triton
    def test_no_fusion_indexing_var_usage(self):
        # [FUNCTION DOC] CUDA only. No fusion when the epilogue needs an
        # index variable (adding an arange) not available in the user kernel:
        # two kernels, two allocations, three deletions.
        # [FUNCTION DOC] HELPER: Masked elementwise add kernel.
        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Empty output; launches the kernel; returns relu of the
            # output plus an arange over its length.
            out = torch.empty_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
            # this arange causes the final kernel to require indexing expressions not available in the user kernel
            return out.relu() + torch.arange(out.shape[0], device="cuda")

        a = torch.tensor([-0.3] * 8, dtype=torch.float32, device="cuda")
        b = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32, device="cuda"
        )

        out, code = run_and_get_code(torch.compile(fn), a, b)
        self.assertEqual(out, fn(a, b), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=3)

    @requires_cuda_and_triton
    def test_no_fusion_non_unary_epilogue(self):
        # [FUNCTION DOC] CUDA only. No fusion when the epilogue adds a third tensor (non-unary):
        # two kernels, two allocations, four deletions; output equals eager.
        # [FUNCTION DOC] HELPER: Masked elementwise add kernel with one program per element.
        @triton.jit
        def add_kernel(a_ptr, b_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < numel
            a = tl.load(a_ptr + offs, mask=mask)
            b = tl.load(b_ptr + offs, mask=mask)
            out = a + b
            tl.store(out_ptr + offs, out, mask=mask)

        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Empty output; launches the add kernel with block size 1;
            # returns the output plus c.
            out = torch.empty_like(a)
            GRID = (a.numel(),)
            add_kernel[GRID](a, b, out, a.numel(), 1)
            return out + c

        a = torch.randn(10, dtype=torch.float32, device="cuda")
        b = torch.randn(10, dtype=torch.float32, device="cuda")
        c = torch.randn(10, dtype=torch.float32, device="cuda")

        out, code = run_and_get_code(torch.compile(fn), a, b, c)
        self.assertEqual(out, fn(a, b, c))
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=4)

    @requires_cuda_and_triton
    def test_fusion_cast_epilogue(self):
        # [FUNCTION DOC] CUDA only. A cast of the kernel output to half precision fuses into one
        # kernel (one allocation, two deletions); output equals eager.
        # [FUNCTION DOC] HELPER: Masked elementwise add kernel.
        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Empty output; launches the kernel; returns the output cast
            # to float16.
            out = torch.empty_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
            return out.to(torch.float16)

        a = torch.randn(8, 16, dtype=torch.float32, device="cuda")
        b = torch.randn(8, 16, dtype=torch.float32, device="cuda")

        out, code = run_and_get_code(torch.compile(fn), a, b)
        self.assertEqual(out, fn(a, b))
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

    @requires_cuda_and_triton
    def test_no_fusion_for_out_of_place_epilogue(self):
        # [FUNCTION DOC] CUDA only. No fusion for an epilogue that transposes the output to
        # contiguous then relu (out-of-place): two kernels, two allocations, three deletions.
        # [FUNCTION DOC] HELPER: Masked elementwise add kernel.
        @triton.jit
        def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(in_ptr0 + offs, mask=mask)
            y = tl.load(in_ptr1 + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Empty output; launches the kernel; returns the transposed,
            # made contiguous, relu of the output.
            out = torch.empty_like(a)
            grid = (triton.cdiv(a.numel(), 1024),)
            add_kernel[grid](a, b, out, a.numel(), BLOCK_SIZE=1024)
            return out.T.contiguous().relu()

        a = torch.randn(32, 32, device="cuda")
        b = torch.randn(32, 32, device="cuda")
        out, code = run_and_get_code(torch.compile(fn), a, b)
        self.assertEqual(out, fn(a, b))
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=3)


if HAS_CUDA_AND_TRITON:

    # [FUNCTION DOC] HELPER: Module-level jit helper storing a value at a pointer with a mask;
    # used by the double-store kernel.
    @triton.jit
    def custom_store(ptr, val, mask):
        tl.store(ptr, val, mask)

    # [FUNCTION DOC] HELPER: Kernel storing the sum of two masked inputs and then storing it
    # again via the custom store helper.
    @triton.jit
    def add_kernel_with_custom_store(
        in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(in_ptr0 + offs, mask=mask)
        y = tl.load(in_ptr1 + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)
        custom_store(out_ptr + offs, x + y, mask=mask)


common_utils.instantiate_parametrized_tests(KernelTests)
common_utils.instantiate_parametrized_tests(CustomOpTests)
common_utils.instantiate_parametrized_tests(TestUserKernelEpilogueFusion)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
