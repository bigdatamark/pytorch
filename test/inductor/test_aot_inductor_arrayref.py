# Owner(s): ["module: inductor"]
# [MODULE DOC] AOTInductor tests for the CPU stack-allocation and minimal array-reference wrapper
# variants. Imports the shared AOTInductor test template and helpers (falling back to non-package
# imports), skips or exits early on Windows CI or missing dependencies, defines a table of tests
# expected to fail or be skipped per variant, and copies every template test into two CPU test
# classes under variant-specific name suffixes. The minimal-arrayref variant is only populated
# when running inside FB code. Also holds an extra template with arrayref-specific code-generation
# checks and a regression test for CPU wrapper class selection.
import sys
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_cpp_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_CI, IS_FBCODE, IS_WINDOWS


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    try:
        from .test_aot_inductor import (
            AOTInductorTestsTemplate,
            AOTIRunnerUtil,
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from .test_torchinductor import copy_tests, TestFailure
    except ImportError:
        from test_aot_inductor import (  # @manual
            AOTInductorTestsTemplate,
            AOTIRunnerUtil,
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


def fail_stack_allocation(is_skip=False):
    # [FUNCTION DOC] HELPER: Builds a failure marker for tests that do not work with stack
    # allocation. It applies to both the stack-allocation suffix and the
    # stack-allocation-plus-minimal-arrayref suffix; when the argument asks for it the test is
    # skipped, otherwise it is an expected failure.
    return TestFailure(
        (
            "cpu_with_stack_allocation",
            "cpu_with_stack_allocation_and_minimal_arrayref_interface",
        ),
        is_skip=is_skip,
    )


def fail_minimal_arrayref_interface(is_skip=False):
    # [FUNCTION DOC] HELPER: Builds a failure marker that applies only to the
    # stack-allocation-plus-minimal-arrayref suffix. The skip flag selects skip versus
    # expected failure.
    return TestFailure(
        ("cpu_with_stack_allocation_and_minimal_arrayref_interface",),
        is_skip=is_skip,
    )


class AOTInductorArrayRefTestsTemplate(AOTInductorTestsTemplate):
    # [CLASS DOC] Mixin template of extra tests, derived from the main AOTInductor template, that
    # check generated C++ for the arrayref wrapper. It relies on the target class to supply
    # device, the stack-allocation flag, the minimal-arrayref flag and the code counting helper.
    # Only copied into the minimal-arrayref class.
    def test_proxy_executor_runtime_lookup_handles_arrayref_tensor_inputs(self):
        # [FUNCTION DOC] Registers a custom op with a fake implementation that returns two
        # tensors, wraps it in a model that sums them, and compiles and runs it under the
        # configured stack-allocation and minimal-arrayref options while capturing generated C++.
        # Checks that the C++ compiles (running builds it), the result is close to eager, and
        # that the generated code contains the proxy executor call and the
        # borrow-arrayref-as-tensor conversion.
        @torch.library.custom_op(
            "aoti_arrayref_test::proxy_executor_tensor_arg", mutates_args=()
        )
        def proxy_executor_tensor_arg(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Custom op implementation returning the input plus one and the
            # input plus two as a pair.
            return x + 1, x + 2

        @proxy_executor_tensor_arg.register_fake
        def _(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Fake (meta) implementation for the custom op; returns two
            # empty tensors shaped like the input.
            return torch.empty_like(x), torch.empty_like(x)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that calls the custom op and adds its two outputs.
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Calls the registered custom op and returns the sum of its
                # two results.
                y, z = torch.ops.aoti_arrayref_test.proxy_executor_tensor_arg(x)
                return y + z

        example_inputs = (torch.randn(4, device=self.device),)
        model = Model()
        with config.patch(
            {
                "aot_inductor.allow_stack_allocation": self.allow_stack_allocation,
                "aot_inductor.use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,
            }
        ):
            # run() compiles the generated wrapper.cpp into an AOTI package before
            # loading it, so this catches ArrayRefTensor -> AtenTensorHandle C++
            # compiler failures in the proxy-executor path.
            actual, code = run_and_get_cpp_code(
                AOTIRunnerUtil.run, model, example_inputs
            )

        self.assertTrue(torch.allclose(actual, model(*example_inputs)))
        FileCheck().check("aoti_torch_proxy_executor_call_function").check(
            "borrow_arrayref_tensor_as_tensor("
        ).run(code)

    def test_thread_local_cached_output_uses_brace_init(self):
        # [FUNCTION DOC] Compiles (without running) a model whose output is a reinterpreted view
        # of a matmul, under the configured options, and uses a pattern check on the generated C++
        # to verify that the thread-local cached output is initialized with brace initialization
        # of an owning tensor handle.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a one-tuple holding a bit-reinterpretation to a
            # 32-bit integer view of a matmul.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Multiplies the two inputs as matrices and returns a tuple
                # with the int32 view of the product.
                z = torch.matmul(x, y)
                return (z.view(torch.int32),)

        example_inputs = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )
        model = Model()
        with config.patch(
            {
                "aot_inductor.allow_stack_allocation": self.allow_stack_allocation,
                "aot_inductor.use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,
            }
        ):
            _, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, model, example_inputs
            )

        FileCheck().check("ThreadLocalCachedOutput").check_regex(
            r"cached_output_\d+\{RAIIAtenTensorHandle\("
        ).run(code)

    def test_thread_local_cached_output_uses_brace_init_multi_output(self):
        # [FUNCTION DOC] Same as the single-output brace-initialization test, but the model
        # returns two outputs, so the generated C++ must contain two thread-local cached outputs,
        # each brace-initialized with an owning tensor handle.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two differently typed views of one
            # matmul result.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the matmul product viewed as int32 and
                # as float16.
                z = torch.matmul(x, y)
                return z.view(torch.int32), z.view(torch.float16)

        example_inputs = (
            torch.randn(4, 4, device=self.device),
            torch.randn(4, 4, device=self.device),
        )
        model = Model()
        with config.patch(
            {
                "aot_inductor.allow_stack_allocation": self.allow_stack_allocation,
                "aot_inductor.use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,
            }
        ):
            _, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, model, example_inputs
            )

        FileCheck().check("ThreadLocalCachedOutput").check_regex(
            r"cached_output_\d+\{RAIIAtenTensorHandle\("
        ).check("ThreadLocalCachedOutput").check_regex(
            r"cached_output_\d+\{RAIIAtenTensorHandle\("
        ).run(code)

    def test_simple_v2_interface(self):
        # [FUNCTION DOC] Compiles a linear-plus-add model and verifies the generated C++ contains
        # the second-version minimal arrayref interface with the expected input count of two and
        # output count of one and the matching size checks. Also asserts, via the code counting
        # helper, that the first-version minimal arrayref entry point appears exactly once.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one 10 by 10 linear layer; forward adds the first
            # input to the linear layer applied to the second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the first input plus the linear transform of
                # the second.
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        model = Model()
        with config.patch(
            {
                "aot_inductor.allow_stack_allocation": self.allow_stack_allocation,
                "aot_inductor.use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,
            }
        ):
            _, code = run_and_get_cpp_code(
                AOTIRunnerUtil.compile, model, example_inputs
            )

        FileCheck().check("AOTInductorModelRunMinimalArrayrefInterfaceV2(").check(
            "constexpr int32_t expected_num_inputs = 2;"
        ).check("constexpr int32_t expected_num_outputs = 1;").check(
            "if (num_inputs != expected_num_inputs)"
        ).check("if (num_outputs != expected_num_outputs)").run(code)
        self.code_check_count(
            model, example_inputs, "AOTInductorModelRunMinimalArrayrefInterface(", 1
        )


# test_failures, xfail by default, set is_skip=True to skip
CPU_TEST_FAILURES = {
    # TODO: error: ‘complex64’ was not declared in this scope
    "test_add_complex": fail_minimal_arrayref_interface(is_skip=True),
    "test_conv_freezing": fail_minimal_arrayref_interface(is_skip=True),
    "test_deconv_freezing": fail_minimal_arrayref_interface(is_skip=True),
    "test_cond_share_predicate": fail_stack_allocation(is_skip=True),
    "test_cond_predicate_on_cpu": fail_stack_allocation(is_skip=True),
    "test_while_loop_with_mixed_device_dynamic_True": fail_stack_allocation(),
    "test_while_loop_with_mixed_device_dynamic_False": fail_stack_allocation(),
    "test_while_loop_with_pytree_inputs": fail_stack_allocation(),
    # ArrayRefTensor outputs do not expose AtenTensorHandle, so this wrapper
    # variant intentionally skips fallback output metadata assertions.
    "test_aoti_custom_op_bad_fake_dtype_fails_fast": fail_stack_allocation(
        is_skip=True
    ),
    # FIXME: failed with Segfault while exiting the Python runtime
    "test_duplicate_constant_folding": fail_stack_allocation(is_skip=True),
    "test_aot_inductor_consts_cpp_build": fail_stack_allocation(is_skip=True),
    "test_stride_with_unbacked_expr": fail_minimal_arrayref_interface(is_skip=True),
    # TODO: use of deleted function RAIIAtenTensorHandle
    "test_dup_unbacked_sym_decl": fail_minimal_arrayref_interface(is_skip=True),
    # TODO: use of deleted function RAIIAtenTensorHandle
    "test_dup_unbacked_sym_decl_with_refinement": fail_minimal_arrayref_interface(
        is_skip=True
    ),
    # https://github.com/pytorch/pytorch/issues/129550
    # https://github.com/pytorch/pytorch/issues/123691
    "test_dynamic_scalar": fail_stack_allocation(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/122980
    "test_fft_c2c": fail_stack_allocation(is_skip=True),
    "test_freezing": fail_minimal_arrayref_interface(is_skip=True),
    "test_linear_freezing": fail_minimal_arrayref_interface(is_skip=True),
    # FIXME: failed with Segfault while exiting the Python runtime
    "test_missing_cubin": fail_stack_allocation(is_skip=True),
    # minimal arrayref interface only works with CPU; test crashes.
    # https://github.com/pytorch/pytorch/issues/122983
    "test_multi_device": fail_minimal_arrayref_interface(is_skip=True),
    # the test segfaults
    "test_repeat_output": fail_stack_allocation(is_skip=True),
    # segfault
    "test_buffer_mutation_1": fail_stack_allocation(is_skip=True),
    # segfault
    "test_buffer_mutation_2": fail_stack_allocation(is_skip=True),
    # segfault
    "test_bool_input": fail_stack_allocation(is_skip=True),
    # segfault
    "test_int_list_input": fail_stack_allocation(is_skip=True),
    # segfault
    # 'AOTInductorTestABICompatibleCpuWithStackAllocation' object has no attribute 'code_check_count'
    "test_buffer_mutation_3": fail_stack_allocation(is_skip=True),
    "test_zero_size_buffer": fail_stack_allocation(is_skip=True),
    # FIXME: failed with Segfault while exiting the Python runtime
    "test_scatter_fallback": fail_stack_allocation(is_skip=True),
    # Looks like the same issue as https://github.com/pytorch/pytorch/issues/122978
    "test_scatter_reduce_fallback": fail_minimal_arrayref_interface(is_skip=True),
    # Looks like the same issue as https://github.com/pytorch/pytorch/issues/122978
    "test_index_put_fallback": fail_minimal_arrayref_interface(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/122984
    "test_index_put_with_none_index": fail_minimal_arrayref_interface(is_skip=True),
    # FIXME: failed with Segfault while exiting the Python runtime
    "test_constant": fail_stack_allocation(is_skip=True),
    # Looks like the same issue as https://github.com/pytorch/pytorch/issues/122978
    "test_shifted_constraint_ranges": fail_stack_allocation(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/123691
    "test_amp_fallback_random": fail_minimal_arrayref_interface(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/123691
    "test_zero_grid_with_unbacked_symbols": fail_minimal_arrayref_interface(
        is_skip=True
    ),
    # failed on MacOS
    "test_zero_grid_with_backed_symbols": fail_stack_allocation(is_skip=True),
    # https://github.com/pytorch/pytorch/issues/122990
    "test_cond_non_tensor_predicates_dynamic_False": fail_stack_allocation(
        is_skip=True
    ),
    # same issue as https://github.com/pytorch/pytorch/issues/122990
    "test_cond_non_tensor_predicates_dynamic_True": fail_stack_allocation(is_skip=True),
    "test_cond_mismatched_branch_output_dynamic_True": fail_stack_allocation(
        is_skip=True
    ),
    "test_cond_mismatched_branch_output_dynamic_False": fail_stack_allocation(
        is_skip=True
    ),
    # https://github.com/pytorch/pytorch/issues/122991
    "test_runtime_checks_complex": fail_stack_allocation(is_skip=True),
    "test_runtime_checks_fp8": fail_stack_allocation(is_skip=True),
    "test_while_loop_simple": fail_stack_allocation(is_skip=True),
    "test_while_loop_nested": fail_stack_allocation(is_skip=True),
    "test_while_loop_with_outer_code": fail_stack_allocation(is_skip=True),
    # TODO: error: cannot convert ArrayRefTensor<float> to AtenTensorHandle
    "test_while_loop_with_outer_buffers": fail_stack_allocation(is_skip=True),
    # TODO: use of undeclared identifier 'float8_e4m3fn' and 'half'
    "test_fp8": fail_minimal_arrayref_interface(is_skip=True),
    "test_size_from_multi_output": fail_stack_allocation(is_skip=True),
    "test_torchvision_transforms_functional_tensor_resize": fail_minimal_arrayref_interface(),
    # TODO: AttributeError: 'ShapeAsConstantBuffer' object has no attribute 'dtype'
    "test_symint_item": fail_minimal_arrayref_interface(is_skip=True),
    # TODO: AttributeError: 'ShapeAsConstantBuffer' object has no attribute 'dtype'
    "test_symbool_item": fail_minimal_arrayref_interface(is_skip=True),
    # TODO: AttributeError: 'ShapeAsConstantBuffer' object has no attribute 'dtype'
    "test_symfloat_item": fail_minimal_arrayref_interface(is_skip=True),
    # Causes a segfault when the process exits
    "test_view_outputs": fail_stack_allocation(is_skip=True),
    "test_pytree_inputs": fail_stack_allocation(is_skip=True),
    "test_duplicated_params": fail_stack_allocation(is_skip=True),
    "test_output_misaligned": fail_stack_allocation(is_skip=True),
    "test_no_args": fail_stack_allocation(is_skip=True),
    "test_fqn": fail_stack_allocation(is_skip=True),
    "test_assert_tensor_meta": fail_stack_allocation(is_skip=True),
    "test_clamp_decomposition": fail_stack_allocation(is_skip=True),
    "test_aoti_constant_tensor_name_collision": fail_stack_allocation(is_skip=True),
    "test_cond_unbacked_symint_closure_dynamic_False": fail_stack_allocation(
        is_skip=True
    ),
    "test_empty_cat_dtype_promotion": fail_stack_allocation(is_skip=True),
    "test_pad_fallback": fail_stack_allocation(is_skip=True),
    "test_simple_embed_kernel_binary_False_max_autotune_True": fail_stack_allocation(
        is_skip=True
    ),
    "test_simple_embed_kernel_binary_True_max_autotune_True": fail_stack_allocation(
        is_skip=True
    ),
    # When running test_seq with test_issue_140766, the process segfaults
    "test_seq": fail_stack_allocation(is_skip=True),
}


class AOTInductorTestABICompatibleCpuWithStackAllocation(TestCase):
    # [CLASS DOC] Empty target class (only configuration attributes) that receives the copied
    # template tests for the CPU stack-allocation variant. Attributes set the cpu device,
    # reuse the shared check helpers, enable stack allocation and disable the minimal
    # arrayref interface.
    device = "cpu"
    device_type = "cpu"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = True
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpuWithStackAllocation,
    "cpu_with_stack_allocation",
    CPU_TEST_FAILURES,
)


class AOTInductorTestABICompatibleCpuWithStackAllocationAndMinimalArrayRefInterface(
    TestCase
):
    # [CLASS DOC] Target class for the CPU variant with both stack allocation and the minimal
    # arrayref interface enabled. Carries only configuration attributes and shared check helpers;
    # test methods are attached via the copy helper when running in FB code.
    device = "cpu"
    device_type = "cpu"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = True
    use_minimal_arrayref_interface = True


if IS_FBCODE:
    # The following tests look like they pass in both pytest and unittest (xml
    # and terminal output say pass), but the process will segfault.  This only
    # happens in OSS CI and is fine internally.
    # See https://github.com/pytorch/pytorch/issues/123691
    copy_tests(
        AOTInductorTestsTemplate,
        AOTInductorTestABICompatibleCpuWithStackAllocationAndMinimalArrayRefInterface,
        "cpu_with_stack_allocation_and_minimal_arrayref_interface",
        CPU_TEST_FAILURES,
    )
    copy_tests(
        AOTInductorArrayRefTestsTemplate,
        AOTInductorTestABICompatibleCpuWithStackAllocationAndMinimalArrayRefInterface,
        "cpu_with_stack_allocation_and_minimal_arrayref_interface",
        CPU_TEST_FAILURES,
    )


class TestCppWrapperCpuSelection(TestCase):
    # [CLASS DOC] Test case holding a regression test for how the CPU C++ wrapper class is
    # selected from configuration.
    def test_cpu_cpp_wrapper_follows_current_stack_allocation_config(self):
        # [FUNCTION DOC] After initializing backend registration, asserts that the wrapper code
        # generator returned for the cpu device with the C++ wrapper enabled is the
        # array-reference wrapper when stack allocation is allowed and the plain CPU wrapper when
        # it is disallowed, using config patches for each case. Guards against the choice being
        # frozen at first registration.
        # Regression test: the CPU cpp wrapper class (CppWrapperCpu vs
        # CppWrapperCpuArrayRef) must track the current allow_stack_allocation
        # config at each compile. It used to be frozen at the process's first
        # backend registration, so whichever config was active for the first
        # compile decided the wrapper for the whole process -- making tests that
        # toggle the config order-dependent and flaky.
        from torch._inductor.codegen.common import (
            get_wrapper_codegen_for_device,
            init_backend_registration,
        )
        from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
        from torch._inductor.codegen.cpp_wrapper_cpu_array_ref import (
            CppWrapperCpuArrayRef,
        )

        init_backend_registration()
        with config.patch({"aot_inductor.allow_stack_allocation": True}):
            self.assertIs(
                get_wrapper_codegen_for_device("cpu", cpp_wrapper=True),
                CppWrapperCpuArrayRef,
            )
        with config.patch({"aot_inductor.allow_stack_allocation": False}):
            self.assertIs(
                get_wrapper_codegen_for_device("cpu", cpp_wrapper=True),
                CppWrapperCpu,
            )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests(needs="filelock")
