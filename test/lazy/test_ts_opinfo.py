# Owner(s): ["oncall: jit"]
# [MODULE DOC] OpInfo-driven tests for the lazy TorchScript backend. Import-time side effects:
# initializes the TS lazy backend, reads the TS native functions yaml file (via init_lists) to
# build op lists, binds them to module globals, and seeds the global random generator with 42.
# Defines hand-written lazy tensor tests, an OpInfo class checking that each listed op is
# dispatched to a lazy kernel and that lazy results match eager ones (also with IR reuse), and
# tests for dynamic shape mode. Device-generic tests are instantiated for CPU only.

import functools
import itertools
import os
from collections.abc import Sequence
from pathlib import Path
from unittest import skip

import yaml

import torch
import torch._lazy
import torch._lazy.config
import torch._lazy.ir_cache
import torch._lazy.metrics
import torch._lazy.ts_backend
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.jit_utils import JitTestCase


torch._lazy.ts_backend.init()


def get_test_device():
    # [FUNCTION DOC] HELPER: Returns the string cuda when the LTC_TS_CUDA environment variable is
    # present, otherwise cpu; this is the eager reference device.
    return "cuda" if "LTC_TS_CUDA" in os.environ else "cpu"


def remove_suffixes(l):
    # [FUNCTION DOC] HELPER: Given a list of op names, returns a new list in which each name is cut
    # at its first dot, dropping overload suffixes.
    return [x.split(".")[0] for x in l]


def init_lists():
    # [FUNCTION DOC] HELPER: Loads the ts_native_functions yaml file located relative to this test
    # file and builds the op lists used by the tests. Returns a 7-tuple in this order: the set of
    # lazy op names (full codegen, supported and autograd entries, suffixes removed), the fallback
    # set (just clamp), ops skipped for runtime errors, ops skipped for incorrect results, ops that
    # run decomposed functional kernels and so do not show in metrics, the list of ops carrying a
    # symint suffix from the yaml, and variant names to skip. Reads a file; no other side effects.
    path_to_script = Path(os.path.abspath(os.path.dirname(__file__)))
    TS_NATIVE_FUNCTIONS_PATH = (
        path_to_script.parent.parent / "aten/src/ATen/native/ts_native_functions.yaml"
    )
    with open(TS_NATIVE_FUNCTIONS_PATH) as f:
        yaml_ts = yaml.load(f, yaml.SafeLoader)
    LAZY_OPS_LIST = set(
        remove_suffixes(
            itertools.chain(
                yaml_ts["full_codegen"], yaml_ts["supported"], yaml_ts["autograd"]
            )
        )
    )
    HAS_SYMINT_SUFFIX = yaml_ts["symint"]
    FALLBACK_LIST = {"clamp"}
    SKIP_RUNTIME_ERROR_LIST = {
        "index_select",  # Empty output_sizes is not supported
        "clone",  # is clone decomposed?
        # General ASAN Failure due to related to generating bool values.
        # https://github.com/pytorch/pytorch/issues/74519
        # https://github.com/pytorch/pytorch/issues/63034
        "nonzero",  # ASAN failure (paste: P501906539)
        "all",  # ASAN failure
        "any",  # ASAN failure
        "logdet",  # ASAN failure
    }
    SKIP_INCORRECT_RESULTS_LIST = {
        "squeeze",  # Value out of range
        "t",  # Value out of range
        "transpose",  # Value out of range
        "bernoulli",  # incorrect results
        "pow",  # incorrect results
        "addcdiv",  # incorrect results (on CI not locally?)
    }
    # The following ops all show up directly in ts_native_functions.yaml,
    # but run functionalized versions of the composite kernels in core.
    # This means that we don't expect the ops to show directly in the LTC metrics.
    FUNCTIONAL_DECOMPOSE_LIST = {
        "diag_embed",
        "block_diag",
        "new_empty_strided",
        "narrow_copy",
        "pixel_shuffle",
        "pixel_unshuffle",
        "select_backward",
        "_trilinear",
        "linalg_inv_ex",
        "linalg_pinv.atol_rtol_tensor",
        "logsumexp",
        "svd",
    }
    # For some ops, we don't support all variants. Here we use formatted_name
    # to uniquely identify the variant.
    SKIP_VARIANT_LIST = {"norm_nuc", "min_reduction_with_dim"}

    return (
        LAZY_OPS_LIST,
        FALLBACK_LIST,
        SKIP_RUNTIME_ERROR_LIST,
        SKIP_INCORRECT_RESULTS_LIST,
        FUNCTIONAL_DECOMPOSE_LIST,
        HAS_SYMINT_SUFFIX,
        SKIP_VARIANT_LIST,
    )


(
    LAZY_OPS_LIST,
    FALLBACK_LIST,
    SKIP_RUNTIME_ERROR_LIST,
    SKIP_INCORRECT_RESULTS_LIST,
    FUNCTIONAL_DECOMPOSE_LIST,
    HAS_SYMINT_SUFFIX,
    SKIP_VARIANT_LIST,
) = init_lists()

torch.manual_seed(42)


def clone_move(t):
    # [FUNCTION DOC] HELPER: Returns a detached copy of the tensor that requires grad and lives on
    # the lazy device.
    dev = "lazy"
    copy_t = t.detach().clone().requires_grad_(True).to(device=dev)
    return copy_t


class TestLazyTensor(JitTestCase):
    # [CLASS DOC] Test case (JIT test base class) with hand-written checks comparing lazy and
    # eager execution.
    @skip("Disable until autograd supports symints")
    def testConvolutionBackward(self):
        # [FUNCTION DOC] Compares gradients of a convolution with input, weight and bias computed
        # eagerly and on lazy copies, with the expected values on the eager side. Currently skipped
        # by the decorator until autograd supports symbolic ints.
        test_device = get_test_device()
        inp = torch.rand(1, 3, 128, 128, device=test_device, requires_grad=True)
        inp_copy = clone_move(inp)
        grad = torch.rand(1, 32, 121, 121, device=test_device)  # no requires_grad
        grad_copy = clone_move(grad)
        weight = torch.rand(32, 3, 8, 8, device=test_device, requires_grad=True)
        weight_copy = clone_move(weight)
        bias = torch.rand(32, device=test_device, requires_grad=True)
        bias_copy = clone_move(bias)

        # run eager
        conv_out = torch.nn.functional.conv2d(inp, weight, bias)
        (inp_grad, weight_grad, bias_grad) = torch.autograd.grad(
            [conv_out], [inp, weight, bias], [grad]
        )

        # run lazy
        conv_copy_out = torch.nn.functional.conv2d(inp_copy, weight_copy, bias_copy)
        (inp_copy_grad, weight_copy_grad, bias_copy_grad) = torch.autograd.grad(
            [conv_copy_out], [inp_copy, weight_copy, bias_copy], [grad_copy]
        )

        # check numerics
        torch.testing.assert_close(bias_copy_grad.cpu(), bias_grad.cpu())

        torch.testing.assert_close(weight_copy_grad.cpu(), weight_grad.cpu())
        torch.testing.assert_close(inp_copy_grad.cpu(), inp_grad.cpu())

    def test_view_mark_step_preserved(self):
        # [FUNCTION DOC] Runs a function that makes a view of a tensor, mutates the view in place,
        # optionally marks a step, then mutates the view again, on eager and on lazy (with a step),
        # and asserts the final values match, i.e. the alias survives the step marker.
        test_device = get_test_device()
        inp = torch.rand(4, device=test_device)
        inp_lazy = clone_move(inp)

        def foo(x, *, mark_step):
            # [FUNCTION DOC] HELPER: Views the input as 2 by 2, adds 1 in place through the view,
            # computes an unused sum, optionally marks a step, adds 1 again through the view, and
            # returns the input.
            y = x.view(2, 2)
            y.add_(1)
            z = x + x  # noqa: F841

            if mark_step:
                torch._lazy.mark_step()

            # y and x should continue to be aliased after the mark_step call.
            y.add_(1)
            return x

        out_ref = foo(inp, mark_step=False)
        out = foo(inp_lazy, mark_step=True)
        # out will have some pending mutations, which will be synced by the .cpu() call.
        torch.testing.assert_close(out_ref.cpu(), out.cpu())

    def test_tensor_ctr(self):
        # [FUNCTION DOC] Builds a tensor with the constructor on the eager device and on lazy,
        # applies a view-all helper to both, and asserts equal results.
        test_device = get_test_device()
        inp = torch.tensor([[1, 2, 3, 4, 5]], device=test_device)
        inp_lazy = torch.tensor([[1, 2, 3, 4, 5]], device="lazy")

        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the input flattened with a view op, which forces the
            # functionalization wrapper to be exercised.
            # Calling a view op to ensure that functionalization wrapping occurs.
            return x.view(-1)

        out_ref = foo(inp)
        out = foo(inp_lazy)
        torch.testing.assert_close(out_ref.cpu(), out.cpu())


class TestLazyOpInfo(TestCase):
    # [CLASS DOC] Device-type test case (instantiated for CPU only) running one sample of each
    # selected OpInfo through the lazy backend.
    @ops(
        [
            op
            for op in op_db
            if op.name in LAZY_OPS_LIST
            and op.name not in SKIP_RUNTIME_ERROR_LIST
            and op.name not in FUNCTIONAL_DECOMPOSE_LIST
            and op.formatted_name not in SKIP_VARIANT_LIST
        ],
        allowed_dtypes=(torch.float,),
    )
    def test_dispatched_to_lazy(self, device, dtype, op):
        # [FUNCTION DOC] For one OpInfo (float dtype; ops in the lazy list minus runtime-error,
        # functional-decompose and skipped-variant entries), takes the first lazy sample input,
        # flushes the lazy queue and metrics, runs the op, flushes again, and asserts that a
        # counter named for the op or one of its aliases (prefix aten for fallback ops else lazy,
        # plus a symint suffix when listed) exists among the counter names.
        def get_name(op):
            # [FUNCTION DOC] HELPER: Builds a dotted name from the op name and its variant name
            # when present.
            # KNOWN ISSUE: this helper is never called.
            l = [op.name]
            if op.variant_test_name != "":
                l.append(op.variant_test_name)
            return ".".join(l)

        global HAS_SYMINT_SUFFIX, FALLBACK_LIST
        samples = op.sample_inputs("lazy", dtype, requires_grad=False)
        sample = next(iter(samples))
        args = [sample.input] + list(sample.args)
        kwargs = sample.kwargs
        torch._lazy.mark_step()
        torch._lazy.wait_device_ops()
        torch._lazy.metrics.reset()

        op(*args, **kwargs)
        torch._lazy.mark_step()
        torch._lazy.wait_device_ops()
        prefix = "aten" if op.name in FALLBACK_LIST else "lazy"
        symint_suffix = "_symint" if op.name in HAS_SYMINT_SUFFIX else ""
        metrics = remove_suffixes(torch._lazy.metrics.counter_names())
        cands = [f"{prefix}::{op.name}{symint_suffix}"]
        # check aliases
        for alias in op.aliases:
            cands.append(f"{prefix}::{alias.name}{symint_suffix}")

        self.assertTrue(
            any(c in metrics for c in cands),
            lambda msg: f"{msg}\nnone of {cands} not found in {metrics}",
        )

    @ops(
        [
            op
            for op in op_db
            if op.name in LAZY_OPS_LIST
            and op.name not in SKIP_RUNTIME_ERROR_LIST | SKIP_INCORRECT_RESULTS_LIST
        ],
        allowed_dtypes=(torch.float,),
    )
    def test_correctness(self, device, dtype, op):
        # [FUNCTION DOC] For each sample of the OpInfo (float dtype, ops in the lazy list minus
        # those with runtime errors or incorrect results), runs the op on a copy of the arguments
        # on the eager device and on lazy arguments (step marker before and after), then compares
        # results with the recursive helper.
        # KNOWN ISSUE: the helper only calls map on the sequence elements without consuming it, so
        #     nested sequence elements are never actually compared.
        test_device = get_test_device()

        def clone_to_device(input, dev):
            # [FUNCTION DOC] HELPER: Returns a copy of a tensor moved to the given device,
            # recursing into non-string sequences (returned as tuples); other values are
            # returned unchanged.
            if isinstance(input, torch.Tensor):
                return input.detach().clone().to(device=dev)
            if isinstance(input, Sequence) and not isinstance(input, str):
                return tuple(map(functools.partial(clone_to_device, dev=dev), input))
            return input

        def assert_allclose_rec(t):
            # [FUNCTION DOC] HELPER: Takes a pair of actual and expected values, asserts they have
            # the same type, and for tensors checks closeness (absolute tolerance 1e-4, NaNs equal)
            # after moving the first to the eager device.
            # KNOWN ISSUE: for sequences it only builds a lazy map object that is never iterated,
            #     so no element is compared.
            a, b = t
            self.assertEqual(type(a), type(b))
            if isinstance(a, torch.Tensor):
                self.assertTrue(
                    torch.allclose(
                        clone_to_device(a, test_device), b, atol=1e-4, equal_nan=True
                    )
                )

            if isinstance(a, Sequence):
                map(assert_allclose_rec, zip(a, b))

        samples = op.sample_inputs("lazy", dtype, requires_grad=False)
        for sample in samples:
            # Need to run mark step so that all random ops are computed in the right order
            torch._lazy.mark_step()

            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            copy_args = clone_to_device(args, test_device)

            r_exp = op(*copy_args, **kwargs)
            r_actual = op(*args, **kwargs)

            torch._lazy.mark_step()
            assert_allclose_rec((r_actual, r_exp))

    @ops(
        [
            op
            for op in op_db
            if op.name in LAZY_OPS_LIST
            and op.name not in SKIP_RUNTIME_ERROR_LIST | SKIP_INCORRECT_RESULTS_LIST
        ],
        allowed_dtypes=(torch.float,),
    )
    def test_correctness_with_reusing_ir(self, device, dtype, op):
        # [FUNCTION DOC] Same comparison as the correctness test but with IR reuse enabled in the
        # lazy config for the duration of the test; at the end it resets the IR cache and turns
        # reuse off.
        # KNOWN ISSUE: reuse is only switched off at the end, so a failure earlier leaves it on.
        torch._lazy.config.set_reuse_ir(True)
        test_device = get_test_device()

        def clone_to_device(input, dev):
            # [FUNCTION DOC] HELPER: Returns a copy of a tensor moved to the given device,
            # recursing into non-string sequences (returned as tuples); other values are
            # returned unchanged.
            if isinstance(input, torch.Tensor):
                return input.detach().clone().to(device=dev)
            if isinstance(input, Sequence) and not isinstance(input, str):
                return tuple(map(functools.partial(clone_to_device, dev=dev), input))
            return input

        def assert_allclose_rec(t):
            # [FUNCTION DOC] HELPER: Takes a pair of actual and expected values, asserts they have
            # the same type, and for tensors checks closeness (absolute tolerance 1e-4, NaNs equal)
            # after moving the first to the eager device.
            # KNOWN ISSUE: for sequences it only builds a lazy map object that is never iterated,
            #     so no element is compared.
            a, b = t
            self.assertEqual(type(a), type(b))
            if isinstance(a, torch.Tensor):
                self.assertTrue(
                    torch.allclose(
                        clone_to_device(a, test_device), b, atol=1e-4, equal_nan=True
                    )
                )

            if isinstance(a, Sequence):
                map(assert_allclose_rec, zip(a, b))

        samples = op.sample_inputs("lazy", dtype, requires_grad=False)
        for sample in samples:
            # Need to run mark step so that all random ops are computed in the right order
            torch._lazy.mark_step()

            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            copy_args = clone_to_device(args, test_device)

            r_exp = op(*copy_args, **kwargs)
            r_actual = op(*args, **kwargs)

            torch._lazy.mark_step()
            assert_allclose_rec((r_actual, r_exp))

        torch._lazy.ir_cache.reset()
        torch._lazy.config.set_reuse_ir(False)


# TODO: after we move to master, add Lazy as a new Device here:
# https://github.com/pytorch/pytorch/blob/master/torch/testing/_internal/common_device_type.py#L532
instantiate_device_type_tests(TestLazyOpInfo, globals(), only_for="cpu")


class TestLazyDynamicOps(TestCase):
    # [CLASS DOC] Test case for lazy tensors in symbolic (dynamic) shape mode, which is enabled for
    # the whole class in setUpClass and restored in tearDownClass.
    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] Saves the current symbolic shape mode in a class attribute, enables the
        # mode, and delegates to the parent setup.
        # Setup the dynamic shape mode
        cls.old_ssa_mode = torch._C._lazy._get_symbolic_shape_mode()
        torch._C._lazy._set_symbolic_shape_mode(True)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        # [FUNCTION DOC] Restores the symbolic shape mode saved in setUpClass and delegates to the
        # parent teardown.
        torch._C._lazy._set_symbolic_shape_mode(cls.old_ssa_mode)
        return super().tearDownClass()

    def test_nonzero_dynamic(self):
        # [FUNCTION DOC] Applies nonzero to a lazy tensor in dynamic shape mode and checks the
        # materialized eager result has shape 3 by 2. The upper-bound size check is commented out
        # until bindings exist.
        # Test that nonzero gives upper bounds sizes when symbolic shape mode is enabled
        test_device = get_test_device()
        x1 = torch.tensor(
            [[0, 1.0, 2.0], [3.0, 0, 0]], device=test_device, requires_grad=True
        )
        x1_lazy = clone_move(x1)
        x2_lazy = torch.nonzero(x1_lazy)

        # FIXME: Add bindings to get upper bounds
        # self.assertEqual(tuple(x2_lazy.size()), (6, 2))

        # We should still be able to instantiate it and get the actual result
        x2_eager = x2_lazy.cpu()
        self.assertEqual(tuple(x2_eager.size()), (3, 2))

    def test_adaptiveavgpool3d_dynamic(self):
        # [FUNCTION DOC] Applies 3D adaptive average pooling with output size 2 to a zero tensor on
        # CPU and again on the eager test device (cpu or cuda), and asserts the output shapes
        # are equal.
        # KNOWN ISSUE: despite the lazy-focused class and comment, neither run uses the lazy
        #     device, so the lazy backend is not exercised.
        # Test that adaptive_avg_pool3d gives correct shapes with lazy backend
        img_cpu = torch.zeros([2, 3, 4, 5, 6], device="cpu")
        out_cpu = torch.nn.AdaptiveAvgPool3d(2).to(device="cpu")(img_cpu)

        test_device = get_test_device()
        img_lazy = torch.zeros([2, 3, 4, 5, 6], device=test_device)
        out_lazy = torch.nn.AdaptiveAvgPool3d(2).to(test_device)(img_lazy)

        self.assertEqual(out_cpu.shape, out_lazy.shape)


if __name__ == "__main__":
    run_tests()
