# Owner(s): ["module: unknown"]
# [MODULE DOC] Gradient correctness tests for every operator in the OpInfo database (plus higher
# order op and custom op databases). Defines one device-generic test class that runs gradcheck
# and gradgradcheck on functional and in-place variants, over double and complex double dtypes
# only. Per-operator skips and expected failures are given by shared sets of skip decorators.
# Importing instantiates the device-specific classes (XPU allowed); running as main enables the
# default dtype check and calls the test runner.

from functools import partial

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
    skip,
    skipOps,
    xfail,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
    TestGradients,
    unMarkDynamoStrictTest,
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.hop_db import hop_db


# gradcheck requires double precision
_gradcheck_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=[torch.double, torch.cdouble]
)


# Device-agnostic skips migrated from OpInfo definitions (see #177259).
_bwd_grad_all = {
    skip("as_strided"),
    skip("as_strided_copy"),
    skip("round", variant_name="decimals_3"),
    skip("round", variant_name="decimals_neg_3"),
    skip("__rpow__"),
    skip("polygamma", variant_name="polygamma_n_1"),
    skip("polygamma", variant_name="polygamma_n_2"),
    skip("polygamma", variant_name="polygamma_n_3"),
    skip("polygamma", variant_name="polygamma_n_4"),
    xfail("bfloat16"),
    xfail("float"),
    xfail("half"),
    xfail("cfloat"),
    xfail("chalf"),
    skip("normal"),
    skip("normal", variant_name="number_mean"),
    skip("linalg.lstsq"),
}


@unMarkDynamoStrictTest
class TestBwdGradients(TestGradients):
    # [CLASS DOC] Device-generic gradient test class, derived from the shared gradients base
    # class and not marked strict for Dynamo. Each test is parametrized over operators and
    # double precision dtypes via the module level decorator, with skip and expected-failure
    # sets applied per test. Method variant tests are intentionally disabled (commented out) as
    # redundant.
    # Tests that gradients are computed correctly
    @skipOps(
        _bwd_grad_all
        | {
            xfail("cov"),
            xfail("istft"),
            skip("sparse.sampled_addmm"),
            skip("sparse.mm", variant_name="reduce"),
            xfail("as_strided_scatter"),
            skip("nn.functional.max_unpool1d"),
            skip("nn.functional.max_unpool2d"),
            skip("nn.functional.max_unpool3d"),
            xfail("linalg.norm", variant_name="subgradients_at_zero"),
        }
    )
    @_gradcheck_ops(op_db + hop_db + custom_op_db)
    def test_fn_grad(self, device, dtype, op):
        # [FUNCTION DOC] Check first order gradients of the functional variant of an operator
        # with gradcheck. Skips if the dtype is not among the operator's supported backward
        # dtypes for the device type; otherwise delegates to the base class gradient helper
        # using the operator's function. Covers the OpInfo, higher order op and custom op
        # databases.
        # This is verified by test_dtypes in test_ops.py
        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest("Skipped! Dtype is not in supported backward dtypes!")
        else:
            self._grad_test_helper(device, dtype, op, op.get_op())

    # Method grad (and gradgrad, see below) tests are disabled since they're
    #   costly and redundant with function grad (and gradgad) tests
    # @_gradcheck_ops(op_db)
    # def test_method_grad(self, device, dtype, op):
    #     self._skip_helper(op, device, dtype)
    #     self._grad_test_helper(device, dtype, op, op.get_method())

    @skipOps(
        _bwd_grad_all
        | {
            skip("abs", dtypes=(torch.cdouble,)),
            xfail("as_strided", variant_name="partial_views"),
        }
    )
    @_gradcheck_ops(op_db + custom_op_db)
    def test_inplace_grad(self, device, dtype, op):
        # [FUNCTION DOC] Check first order gradients of the in-place variant. Skips for
        # unsupported dtypes/autograd and when the operator has no in-place variant. If the
        # operator claims no in-place autograd support, verifies that running the in-place
        # variant (on cloned inputs) then backward raises some exception for each non-
        # broadcasting sample; otherwise runs the gradient helper on the safe in-place wrapper.
        self._skip_helper(op, device, dtype)
        if not op.inplace_variant:
            self.skipTest("Op has no inplace variant!")

        # Verifies an operation doesn't support inplace autograd if it claims not to
        if not op.supports_inplace_autograd:
            inplace = self._get_safe_inplace(op.get_inplace())
            for sample in op.sample_inputs(device, dtype, requires_grad=True):
                if sample.broadcasts_input:
                    continue
                with self.assertRaises(Exception):
                    result = inplace(sample)
                    result.sum().backward()
        else:
            self._grad_test_helper(
                device, dtype, op, self._get_safe_inplace(op.get_inplace())
            )

    # Test that gradients of gradients are computed correctly
    @skipOps(
        _bwd_grad_all
        | {
            xfail("cov"),
            skip("sparse.sampled_addmm"),
            skip("sparse.mm", variant_name="reduce"),
            xfail("native_layer_norm"),
            skip("nn.functional.max_unpool1d"),
            skip("nn.functional.max_unpool2d"),
            skip("nn.functional.max_unpool3d"),
            xfail("cat"),
            xfail("nn.functional.ctc_loss", dtypes=(torch.float64,)),
            xfail("nn.functional.linear_cross_entropy", variant_name="chunked_none"),
            xfail("nn.functional.linear_cross_entropy", variant_name="chunked"),
            xfail("linalg.norm"),
            xfail("linalg.norm", variant_name="subgradients_at_zero"),
            skip("masked.logaddexp"),
        }
    )
    @_gradcheck_ops(op_db + hop_db + custom_op_db)
    def test_fn_gradgrad(self, device, dtype, op):
        # [FUNCTION DOC] Check second order gradients of the functional variant with the
        # backward-over-backward gradgradcheck. Skips for unsupported dtype/autograd, and when
        # the operator declares it lacks gradgrad support (this claim is not verified here);
        # otherwise runs the shared check helper.
        self._skip_helper(op, device, dtype)
        if not op.supports_gradgrad:
            self.skipTest(
                "Op claims it doesn't support gradgrad. This is not verified."
            )
        else:
            self._check_helper(device, dtype, op, op.get_op(), "bwgrad_bwgrad")

    # Test that gradients of gradients are properly raising
    @skipOps(_bwd_grad_all | {skip("sparse.mm", variant_name="reduce")})
    @_gradcheck_ops(op_db + custom_op_db)
    def test_fn_fail_gradgrad(self, device, dtype, op):
        # [FUNCTION DOC] For operators that declare no gradgrad support, verify that a second
        # order check raises a RuntimeError whose message says the derivative is not
        # implemented. Skips for unsupported dtype/autograd or when the operator does support
        # gradgrad.
        self._skip_helper(op, device, dtype)
        if op.supports_gradgrad:
            self.skipTest("Skipped! Operation does support gradgrad")

        err_msg = r"derivative for .* is not implemented"
        with self.assertRaisesRegex(RuntimeError, err_msg):
            self._check_helper(device, dtype, op, op.get_op(), "bwgrad_bwgrad")

    # Method gradgrad (and grad, see above) tests are disabled since they're
    #   costly and redundant with function gradgrad (and grad) tests
    # @_gradcheck_ops(op_db)
    # def test_method_gradgrad(self, device, dtype, op):
    #     self._skip_helper(op, device, dtype)
    #     self._gradgrad_test_helper(device, dtype, op, op.get_method())

    @skipOps(
        _bwd_grad_all
        | {
            skip("abs", dtypes=(torch.cdouble,)),
            xfail("as_strided", variant_name="partial_views"),
            xfail("nn.functional.hardsigmoid"),
        }
    )
    @_gradcheck_ops(op_db)
    def test_inplace_gradgrad(self, device, dtype, op):
        # [FUNCTION DOC] Check second order gradients of the in-place variant (on cloned inputs)
        # using the backward-over-backward check. Skips for unsupported dtype/autograd, when
        # there is no in-place variant, or when it does not support in-place autograd.
        self._skip_helper(op, device, dtype)
        if not op.inplace_variant or not op.supports_inplace_autograd:
            self.skipTest("Skipped! Operation does not support inplace autograd.")
        self._check_helper(
            device, dtype, op, self._get_safe_inplace(op.get_inplace()), "bwgrad_bwgrad"
        )


instantiate_device_type_tests(TestBwdGradients, globals(), allow_xpu=True)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
