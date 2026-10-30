# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for TorchScript autocast support. TestAutocast scripts functions that use
# autocast context managers and checks result dtypes (half precision for matmul-like ops, float32
# for ops with a float32 policy, promotion policy for mixed ops), nesting, callees, tracing
# interaction, modules, freezing and the ignore flag. TestJitTraceAutocast traces and freezes CPU
# models under autocast. Most tests need CUDA. Import-time side effects: when run as a script it
# parses command line args first; it computes a module flag for bfloat16 support on CUDA and
# imports a model fixture and the base test case from sibling test modules.

import torch
from torch.cuda.amp import autocast

import sys
import unittest
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_LINUX,
    IS_CPU_EXT_SVE_SUPPORTED,
    parse_cmd_line_args,
    run_tests,
    skipIfTorchDynamo,
    xfailIf,
)
from torch.testing import FileCheck
from jit.test_models import MnistNet

if __name__ == '__main__':
    # The value of GRAPH_EXECUTOR depends on command line arguments so make sure they're parsed
    # before instantiating tests.
    parse_cmd_line_args()

from test_jit import JitTestCase
TEST_BFLOAT16 = TEST_CUDA and torch.cuda.is_bf16_supported()

@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestAutocast(JitTestCase):
    # [CLASS DOC] JIT test case for autocast scripting. setUp creates random 2 by 2 half and
    # float32 CUDA tensors named a through d with suffixes fp16 and fp32 (only when CUDA exists)
    # and turns on the JIT autocast mode, remembering the old setting; tearDown restores it.
    # Skipped under TorchDynamo.
    def setUp(self):
        # [FUNCTION DOC] Creates the shared CUDA input tensors (when CUDA is available), enables
        # JIT autocast mode saving the previous value in old_value, then calls the base setUp.
        # common input tensors
        if TEST_CUDA:
            self.a_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
            self.b_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
            self.c_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
            self.d_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
            self.a_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
            self.b_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
            self.c_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
            self.d_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
        self.old_value = torch._C._jit_set_autocast_mode(True)
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] Restores the JIT autocast mode saved by setUp, then calls the
        # base tearDown.
        torch._C._jit_set_autocast_mode(self.old_value)
        super().tearDown()

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_generic_autocast(self):
        # [FUNCTION DOC] Checks that scripted functions using the CUDA autocast class and the
        # generic torch.amp autocast with device type cuda produce equal results on float32 inputs.
        # Needs CUDA.
        @torch.jit.script
        def fn_cuda_autocast(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture doing a matrix multiply and sum under
            # autocast, returning both.
            with autocast():
                x = torch.mm(a, b)
                y = torch.sum(x)
                return x, y

        @torch.jit.script
        def fn_generic_autocast(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture like the other one but using the generic amp
            # autocast with device type cuda.
            with torch.amp.autocast(device_type='cuda'):
                x = torch.mm(a, b)
                y = torch.sum(x)
                return x, y
        self.assertEqual(fn_cuda_autocast(self.a_fp32, self.b_fp32), fn_generic_autocast(self.a_fp32, self.b_fp32))

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_minimal(self):
        # [FUNCTION DOC] Checks that under autocast a scripted matmul yields float16 while a
        # following sum yields float32. Needs CUDA.
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture returning a matrix product and its sum
            # computed inside an autocast block.
            with autocast():
                x = torch.mm(a, b)
                y = torch.sum(x)
                return x, y
        x, y = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(x.dtype, torch.float16)
        self.assertEqual(y.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA or not TEST_BFLOAT16, "No cuda bfloat16 support")
    def test_linear_bf16(self):
        # [FUNCTION DOC] Same as the minimal test but autocast uses bfloat16: matmul result is
        # bfloat16 and sum float32. Needs CUDA with bfloat16 support.
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture returning a matrix product and its sum inside
            # autocast with bfloat16.
            with autocast(dtype=torch.bfloat16):
                x = torch.mm(a, b)
                y = torch.sum(x)
                return x, y
        x, y = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(x.dtype, torch.bfloat16)
        self.assertEqual(y.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_minimal_cpu(self):
        # [FUNCTION DOC] Checks that a scripted autocast matmul on CPU copies of float32 inputs
        # (default CUDA autocast does not cast CPU tensors) stays float32. Test requires CUDA only
        # to build inputs.
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture returning a matrix product
            # computed inside autocast.
            with autocast():
                return torch.mm(a, b)
        result = fn(self.a_fp32.to('cpu'), self.b_fp32.to('cpu'))
        self.assertEqual(result.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_minimal_off(self):
        # [FUNCTION DOC] Checks that autocast with enabled false leaves a scripted matmul in
        # float32. Needs CUDA.
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture returning a matrix product inside autocast
            # with enabled false.
            with autocast(enabled=False):
                return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_runtime_autocast_state(self):
        # [FUNCTION DOC] Checks that passing a runtime boolean as the autocast enabled argument is
        # rejected: calling the scripted function raises a runtime error. Needs CUDA.
        @torch.jit.script
        def fn(a, b, use_amp: bool):
            # [FUNCTION DOC] HELPER: Scripted fixture taking a boolean argument that is passed as
            # the enabled flag of autocast.
            with autocast(enabled=use_amp):
                return torch.mm(a, b)
        # runtime values for autocast enable argument are not supported
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32, True)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_runtime_autocast_state_expr(self):
        # [FUNCTION DOC] Checks that an enabled flag computed from tensor data at runtime is
        # rejected: the call of the scripted function raises a runtime error. Needs CUDA.
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture whose autocast enabled flag comes from a
            # tensor element converted to bool.
            with autocast(enabled=bool((a[0][0] > 0.5).item())):
                return torch.mm(a, b)
        # runtime values for autocast enable argument are not supported
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_explicit_casts(self):
        # [FUNCTION DOC] Checks explicit casts inside autocast: double matmul cast back to float
        # gives float32, a matmul of float inputs cast to double gives float64, and a later double
        # matmul outside autocast gives float64. Needs CUDA.
        @torch.jit.script
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Scripted fixture doing double matmul cast to float and a float
            # matmul cast to double inside autocast, then a double matmul outside.
            with autocast():
                e = torch.mm(a.double(), b.double()).float()
                f = torch.mm(c, d).double()
            g = torch.mm(c.double(), f)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float32)
        self.assertEqual(f.dtype, torch.float64)
        self.assertEqual(g.dtype, torch.float64)

    # multiple uses of the same input value
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_duplicate_inputs(self):
        # [FUNCTION DOC] Checks that using the same input value several times (matmul of a with
        # itself then of the result with itself) works under autocast and yields float16 for both
        # results. Needs CUDA.
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture squaring a matrix and squaring the result
            # inside autocast.
            with autocast():
                e = torch.mm(a, a)
                f = torch.mm(e, e)
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_policy(self):
        # [FUNCTION DOC] Checks that a log (float32 policy op) on float16 input under autocast
        # returns float32. Needs CUDA.
        @torch.jit.script
        def fn(a):
            # [FUNCTION DOC] HELPER: Scripted fixture applying log to its input inside autocast.
            with autocast(enabled=True):
                return torch.log(a)
        result = fn(self.a_fp16)
        self.assertEqual(result.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_policy_with_fp64(self):
        # [FUNCTION DOC] Checks that the float32 policy does not narrow double inputs: log of a
        # double tensor stays float64. Needs CUDA.
        @torch.jit.script
        def fn(a):
            # [FUNCTION DOC] HELPER: Scripted fixture applying log to its input inside autocast.
            with autocast(enabled=True):
                return torch.log(a)
        # fp32 policy should not narrow fp64 to fp32!
        result = fn(self.a_fp32.double())
        self.assertEqual(result.dtype, torch.float64)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_promote_policy(self):
        # [FUNCTION DOC] Checks the promote policy: a matmul gives float16 and a following addcmul
        # mixing it with float32 inputs gives float32. Needs CUDA.
        @torch.jit.script
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Scripted fixture doing a matmul and then addcmul inside
            # autocast, returning both.
            with autocast():
                e = torch.mm(a, b)
                f = torch.addcmul(e, c, d, value=0.1)
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_promote_policy_fp64(self):
        # [FUNCTION DOC] Checks that addcmul on double inputs under autocast stays
        # float64. Needs CUDA.
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture doing addcmul on its inputs inside autocast.
            with autocast(enabled=True):
                return torch.addcmul(a, a, b, value=0.1)
        result = fn(self.a_fp32.double(), self.b_fp32.double())
        self.assertEqual(result.dtype, torch.float64)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_set_opt_dtype_policy(self):
        # [FUNCTION DOC] Checks softmax dtype handling under autocast with half inputs: no dtype or
        # a None dtype gives float32, an explicit float64 gives float64, and a runtime optional
        # dtype argument left None keeps float16. Needs CUDA.
        @torch.jit.script
        def fn(a, b, c, d, dtype: int | None):
            # [FUNCTION DOC] HELPER: Scripted fixture calling softmax four ways (default, None
            # dtype, float64 dtype, optional argument dtype) inside autocast.
            with autocast(enabled=True):
                x = torch.softmax(a, 0)
                y = torch.softmax(b, 0, None)
                z = torch.softmax(c, 0, torch.float64)
                w = torch.softmax(d, 0, dtype)
            return x, y, z, w
        x, y, z, w = fn(self.a_fp16, self.b_fp16, self.c_fp16, self.d_fp16, None)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(z.dtype, torch.float64)
        self.assertEqual(w.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_set_opt_dtype_policy_fp64(self):
        # [FUNCTION DOC] Same as the opt-dtype policy test but with double inputs: all four softmax
        # results must stay float64. Needs CUDA.
        @torch.jit.script
        def fn(a, b, c, d, dtype: int | None):
            # [FUNCTION DOC] HELPER: Scripted fixture calling softmax four ways inside autocast;
            # same shape as in the half variant.
            with autocast(enabled=True):
                x = torch.softmax(a, 0)
                y = torch.softmax(b, 0, None)
                z = torch.softmax(c, 0, torch.float64)
                w = torch.softmax(d, 0, dtype)
            return x, y, z, w
        x, y, z, w = fn(self.a_fp32.double(), self.b_fp32.double(), self.c_fp32.double(), self.d_fp32.double(), None)
        self.assertEqual(x.dtype, torch.float64)
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual(z.dtype, torch.float64)
        self.assertEqual(w.dtype, torch.float64)

    @unittest.skipIf(True, "broken due to lack of type propagation")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_control_flow(self):
        # [FUNCTION DOC] Skipped (lack of type propagation). Intended to check autocast across if
        # branches with half results. Needs CUDA.
        @torch.jit.script
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Scripted fixture with a data-dependent branch computing a
            # matmul in each side inside autocast, then another matmul scaled by an integer.
            with autocast():
                if a[0][0] > 0.5:
                    e = torch.mm(a, b)
                    x = 1
                else:
                    e = torch.mm(c, d)
                    x = 2
                f = torch.mm(d, e) * x
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)

    # this works find in regular Python, but it creates a delicate
    # situation in TorchScript where the types are not consistent across
    # the then/else branches
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_divergent_types(self):
        # [FUNCTION DOC] Checks that branches producing different dtypes (one half and one float)
        # combine correctly: a scripted function casts both to float before a final matmul, giving
        # float32. Needs CUDA.
        @torch.jit.script
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Scripted fixture whose two branches produce half or float
            # values, then multiplies the float casts.
            with autocast():
                if a[0][0] > 0.5:
                    e = torch.mm(a, b)
                    f = torch.mm(a, b).float()
                else:
                    e = torch.mm(c, d).float()
                    f = torch.mm(a, b)
            return torch.mm(e.float(), f.float())
        result = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(result.dtype, torch.float32)

    # another, more complex case of divergent types
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_divergent_autocast(self):
        # [FUNCTION DOC] Checks a script with autocast context objects created outside and entered
        # in different branches runs without error. No result assertions. Needs CUDA.
        @torch.jit.script
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Scripted fixture creating an enabled and a disabled autocast
            # object and entering one per branch.
            autocast_on = autocast(enabled=True)
            autocast_off = autocast(enabled=False)
            if a[0][0] > 0.5:
                with autocast_on:
                    e = torch.mm(a, b)
            else:
                with autocast_off:
                    e = torch.mm(c, d)
            return torch.mm(e, e)
        fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_conditional_autocast(self):
        # [FUNCTION DOC] Checks that choosing the autocast context with a conditional expression is
        # not supported: calling the scripted function raises a runtime error. Needs CUDA.
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture using a conditional expression to pick
            # between two autocast objects.
            autocast_on = autocast(enabled=True)
            autocast_off = autocast(enabled=False)
            with autocast_on if a[0][0] > 0.5 else autocast_off:
                return torch.mm(a, b)
        # conditional autocast expressions are not supported
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_nested_autocast(self):
        # [FUNCTION DOC] Checks nested autocast blocks off, on, off: outer matmul float32, middle
        # float16, innermost float32. Needs CUDA.
        @torch.jit.script
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Scripted fixture nesting disabled, enabled and disabled
            # autocast blocks with a matmul in each level.
            with autocast(enabled=False):
                e = torch.mm(a, b)
                with autocast(enabled=True):
                    f = torch.mm(e, c)
                    with autocast(enabled=False):
                        g = torch.mm(e, d)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float32)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_implicitly_nested_autocast(self):
        # [FUNCTION DOC] Checks a single with statement entering two autocast managers (disabled
        # then enabled): the inner enabled one wins and the result is float16. Needs CUDA.
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture entering a disabled and an enabled autocast
            # in one statement.
            with autocast(enabled=False), autocast(enabled=True):
                return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_reused_autocast(self):
        # [FUNCTION DOC] Checks reusing one autocast object in nested blocks: all results are
        # float16. Needs CUDA.
        @torch.jit.script
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Scripted fixture that enters the same autocast object twice
            # nested, then does a matmul after the blocks.
            autocast_instance = autocast(enabled=True)
            with autocast_instance:
                e = torch.mm(a, b)
                with autocast_instance:
                    e = torch.mm(c, d)
                    f = torch.mm(d, e)
            g = torch.mm(e, f)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float16)

    # TODO: fix and enable this test?
    #   (we could technically fix this, but is it really worth it?)
    @unittest.skipIf(True, "unsupported autocast syntax")
    def test_reused_autocast_expr(self):
        # [FUNCTION DOC] Skipped (unsupported syntax). Intended to check reusing an autocast object
        # bound by the as clause; the TODO suggests it may be fixable but might not be worth it.
        @torch.jit.script
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Scripted fixture binding the autocast object with as and
            # entering it again inside.
            with autocast(enabled=True) as autocast_instance:
                e = torch.mm(a, b)
                with autocast_instance:
                    e = torch.mm(c, d)
                    f = torch.mm(d, e)
            g = torch.mm(e, f)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_callees(self):
        # [FUNCTION DOC] Checks that a plain Python helper called from a scripted function inside
        # an autocast block is autocast too: results are float16. Needs CUDA.
        def helper(a, b):
            # [FUNCTION DOC] HELPER: Plain function returning a matrix product of its arguments.
            return torch.mm(a, b)

        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture calling the helper five times inside an
            # enabled autocast block.
            with autocast(enabled=True):
                tmp = helper(a, b)
                tmp = helper(tmp, tmp)
                tmp = helper(tmp, tmp)
                tmp = helper(tmp, tmp)
                return helper(tmp, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_callees_with_autocast_on(self):
        # [FUNCTION DOC] Checks that a callee that enables autocast itself keeps it even when
        # called from a disabled region: result is float16. Needs CUDA.
        def helper(a, b):
            # [FUNCTION DOC] HELPER: Callee computing a matmul inside an enabled autocast block.
            with autocast(enabled=True):
                return torch.mm(a, b)

        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture calling the helper inside a
            # disabled autocast block.
            with autocast(enabled=False):
                return helper(a, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_callees_with_autocast_off(self):
        # [FUNCTION DOC] Checks that a callee disabling autocast stays float32 when called from an
        # enabled region. Needs CUDA.
        def helper(a, b):
            # [FUNCTION DOC] HELPER: Callee computing a matmul inside a disabled autocast block.
            with autocast(enabled=False):
                return torch.mm(a, b)

        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture calling the helper inside an
            # enabled autocast block.
            with autocast(enabled=True):
                return helper(a, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float32)

    # scripting inside eager autocast
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_eager_and_script(self):
        # [FUNCTION DOC] Checks scripting inside eager autocast: eight iterations alternate
        # autocast enabled and disabled around a call of a scripted matmul and compare the result
        # dtype with half or float32 accordingly. Needs CUDA.
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture returning a matrix product, with no autocast
            # of its own.
            return torch.mm(a, b)
        for i in range(8):
            use_autocast = (i % 2 == 0)
            expected_dtype = torch.float16 if use_autocast else torch.float32
            with autocast(enabled=use_autocast):
                result = fn(self.a_fp32, self.b_fp32)
            self.assertEqual(result.dtype, expected_dtype)

    # traced inside scripting
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_and_tracing(self):
        # [FUNCTION DOC] Checks a traced function (built with the first input twice) called from a
        # scripted function inside autocast gives float16. Needs CUDA.
        def helper(a, b):
            # [FUNCTION DOC] HELPER: Function returning a matrix product, traced by the test.
            return torch.mm(a, b)

        traced = torch.jit.trace(helper, (self.a_fp32, self.a_fp32))

        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture calling the traced function inside an enabled
            # autocast block.
            with autocast(enabled=True):
                return traced(a, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    # traced with autocast inside scripting
    @unittest.skipIf(True, "autocast(False) is ignored inside traced functions")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_and_tracing_with_autocast(self):
        # [FUNCTION DOC] Skipped (autocast disabling is ignored in traced functions).
        # Intended to check a traced function that disables autocast inside a scripted
        # autocast region yields float32.
        def helper(a, b):
            # [FUNCTION DOC] HELPER: Function with a disabled autocast block returning a matrix
            # product times two.
            with autocast(enabled=False):
                return torch.mm(a, b) * 2.0

        traced = torch.jit.trace(helper, (self.a_fp32, self.a_fp32))

        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture calling the traced function
            # inside enabled autocast.
            with autocast(enabled=True):
                return traced(a, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float32)

    # scripted called from traced
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_tracing_and_script(self):
        # [FUNCTION DOC] Checks a scripted function with autocast called from a traced function:
        # traced result is float16. Needs CUDA.
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture returning a matrix product
            # computed in autocast.
            with autocast():
                return torch.mm(a, b)

        def traced(a, b):
            # [FUNCTION DOC] HELPER: Plain function delegating to the scripted function; later
            # replaced by its traced version.
            return fn(a, b)

        traced = torch.jit.trace(traced, (self.a_fp32, self.b_fp32))
        result = traced(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    # scripted called from traced with autocast
    @unittest.skipIf(True, "scripted called from traced TorchScript is not yet working")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_tracing_with_autocast_and_script(self):
        # [FUNCTION DOC] Skipped (scripted called from traced TorchScript not working). Intended to
        # check a traced function that enables autocast around a scripted call yields float16.
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture returning a matrix product.
            return torch.mm(a, b)

        def traced(a, b):
            # [FUNCTION DOC] HELPER: Function calling the scripted function inside enabled
            # autocast; to be traced.
            with autocast(enabled=True):
                return fn(a, b)

        traced = torch.jit.trace(traced, (self.a_fp32, self.b_fp32))
        result = traced(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_module(self):
        # [FUNCTION DOC] Scripts a module holding a float parameter and a linear layer whose
        # forward applies them under enabled autocast, moves it to CUDA, and checks the output is
        # float16. Needs CUDA.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter and a linear layer, computing a
            # matrix-vector product then the linear layer under autocast.
            def __init__(self, N, M):
                # [FUNCTION DOC] HELPER: Creates a random float32 weight parameter of the given
                # shape and a float linear layer.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand((N, M), dtype=torch.float32))
                self.linear = torch.nn.Linear(N, M).float()

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Under enabled autocast, multiplies the weight by the input
                # vector and applies the linear layer.
                with autocast(enabled=True):
                    output = self.weight.mv(input)
                    output = self.linear(output)
                    return output

        scripted_module = torch.jit.script(TestModule(2, 3)).cuda()
        input = torch.rand(3, dtype=torch.float32, device='cuda')
        result = scripted_module(input)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(True, "autocast decorators not supported")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_decorator(self):
        # [FUNCTION DOC] Skipped (decorators not supported). Intended to check scripting a function
        # decorated with autocast gives float16.
        @torch.jit.script
        @autocast(enabled=True)
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture decorated with enabled autocast, returning a
            # matrix product.
            return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    # this is equivalent to running scripted functions inside autocast)
    # (see also test_eager_and_script)
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_decorator_outside_jit(self):
        # [FUNCTION DOC] Checks that applying the autocast decorator on top of a scripted function
        # (autocast outside the JIT) yields float16. Needs CUDA.
        @autocast(enabled=True)
        @torch.jit.script
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Scripted fixture returning a matrix product, wrapped by the
            # eager autocast decorator.
            return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_inplace(self):
        # [FUNCTION DOC] Checks in-place and out-variant ops under autocast: addmm gives float16,
        # while the out variant and the in-place variant writing into the float32 tensor stay
        # float32. Needs CUDA.
        @torch.jit.script
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Scripted fixture calling addmm, addmm with an out argument,
            # and in-place addmm inside autocast.
            with autocast(enabled=True):
                x = torch.addmm(a, b, c)
                y = torch.addmm(a, b, c, out=a)
                z = a.addmm_(b, c)
                return x, y, z
        x, y, z = fn(self.a_fp32, self.b_fp32, self.c_fp32)
        self.assertEqual(x.dtype, torch.float16)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(z.dtype, torch.float32)

    def _test_autocast(self, func, cast_op, *args):
        # [FUNCTION DOC] HELPER: Scripts func, runs both eager and scripted on args, and when
        # cast_op is not None checks the scripted graph for that op with FileCheck. Then asserts
        # every output pair has the same dtype.
        jit_func = torch.jit.script(func)
        o = func(*args)
        jit_o = jit_func(*args)
        if cast_op is not None:
            FileCheck().check(cast_op).run(jit_func.graph_for(*args))
        for o0, o1 in zip(o, jit_o):
            self.assertEqual(o0.dtype, o1.dtype)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_api(self):
        # [FUNCTION DOC] Checks several autocast API spellings (CPU with bfloat16, CUDA with half,
        # device-type keyword forms) are scriptable, insert the reduced-precision cast op, and
        # match eager dtypes. Needs CUDA.

        def t_autocast_cpu(x, y):
            # [FUNCTION DOC] HELPER: Matmul under CPU autocast with bfloat16.
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return torch.mm(x, y)

        def t_autocast_cuda(x, y):
            # [FUNCTION DOC] HELPER: Matmul under CUDA autocast with half.
            with torch.autocast("cuda", dtype=torch.half):
                return torch.mm(x, y)

        def t_cuda_amp_autocast(x, y):
            # [FUNCTION DOC] HELPER: Matmul under autocast by device type keyword cuda.
            with torch.autocast(device_type="cuda"):
                return torch.mm(x, y)

        def t_cpu_amp_autocast(x, y):
            # [FUNCTION DOC] HELPER: Matmul under autocast by device type keyword cpu.
            with torch.autocast(device_type="cpu"):
                return torch.mm(x, y)

        x = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        y = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        self._test_autocast(t_autocast_cpu, "aten::_autocast_to_reduced_precision", x, y)
        self._test_autocast(t_autocast_cuda, "aten::_autocast_to_reduced_precision", x, y)
        self._test_autocast(t_cuda_amp_autocast, "aten::_autocast_to_reduced_precision", x, y)
        self._test_autocast(t_cpu_amp_autocast, "aten::_autocast_to_reduced_precision", x, y)

    @unittest.skipIf(True, "we need to provide dtype argument at this moment")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_api_not_supported(self):
        # [FUNCTION DOC] Skipped (a dtype argument is required for now). Intended to check CPU and
        # CUDA autocast without dtype are scriptable.

        def t_autocast_cpu(x, y):
            # [FUNCTION DOC] HELPER: Matmul under CPU autocast with no dtype.
            # no dtype provided is not currently supported
            with torch.autocast("cpu"):
                return torch.mm(x, y)

        def t_autocast_cuda(x, y):
            # [FUNCTION DOC] HELPER: Matmul under CUDA autocast with no dtype.
            # no dtype provided is not currently supported
            with torch.autocast("cuda"):
                return torch.mm(x, y)

        x = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        y = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        self._test_autocast(t_autocast_cpu, "aten::_autocast_to_reduced_precision", x, y)
        self._test_autocast(t_autocast_cuda, "aten::_autocast_to_reduced_precision", x, y)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_mixed_dtypes(self):
        # [FUNCTION DOC] Checks nested CPU bfloat16 and CUDA float16 autocast in one function:
        # scripting works and the cast op appears with dtypes agreeing with eager. Needs CUDA.

        def t(cpu0, cpu1, cuda0, cuda1):
            # [FUNCTION DOC] HELPER: Under nested CPU and CUDA autocast, multiplies the CPU pair
            # and the CUDA pair, returning both.
            with torch.autocast("cpu", torch.bfloat16):
                with torch.autocast("cuda", torch.float16):
                    cpu_o = torch.mm(cpu0, cpu1)
                    cuda_o = torch.mm(cuda0, cuda1)
                    return cpu_o, cuda_o

        torch.jit.script(t)
        cpu0 = torch.randn(5, 5, device="cpu", dtype=torch.float32)
        cpu1 = torch.randn(5, 5, device="cpu", dtype=torch.float32)
        cuda0 = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        cuda1 = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        self._test_autocast(t, "aten::_autocast_to_reduced_precision", cpu0, cpu1, cuda0, cuda1)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_executor_under_autocast(self):
        # [FUNCTION DOC] Checks the JIT executor honors an outer autocast context: a function with
        # no autocast of its own is scripted, then run under CPU only, CUDA only, and both contexts
        # expecting the cast op, and outside any context expecting no cast op. Needs CUDA.

        def t(cpu0, cpu1, cuda0, cuda1):
            # [FUNCTION DOC] HELPER: Function multiplying the CPU pair and the CUDA
            # pair, returning both.
            cpu_o = torch.mm(cpu0, cpu1)
            cuda_o = torch.mm(cuda0, cuda1)
            return cpu_o, cuda_o

        torch.jit.script(t)
        cpu0 = torch.randn(5, 5, device="cpu", dtype=torch.float32)
        cpu1 = torch.randn(5, 5, device="cpu", dtype=torch.float32)
        cuda0 = torch.randn(5, 5, device="cuda", dtype=torch.float32)
        cuda1 = torch.randn(5, 5, device="cuda", dtype=torch.float32)

        with torch.autocast("cpu", torch.bfloat16):
            with torch.autocast("cuda", torch.float16):
                self._test_autocast(t, "aten::_autocast_to_reduced_precision", cpu0, cpu1, cuda0, cuda1)

        with torch.autocast("cpu", torch.bfloat16):
            self._test_autocast(t, "aten::_autocast_to_reduced_precision", cpu0, cpu1, cuda0, cuda1)

        with torch.autocast("cuda", torch.float16):
            self._test_autocast(t, "aten::_autocast_to_reduced_precision", cpu0, cpu1, cuda0, cuda1)

        # no cast op should be observed when executing outside autocast context
        self._test_autocast(t, None, cpu0, cpu1, cuda0, cuda1)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_autodiff(self):
        # [FUNCTION DOC] Checks autodiff under autocast: scripts a matmul with relu, runs five
        # warm-up forward/backward passes under CUDA half autocast, then compares output, gradients
        # and their dtypes against eager. Needs CUDA.
        def t(t0, t1):
            # [FUNCTION DOC] HELPER: Function returning the relu of a matrix product.
            o = torch.mm(t0, t1)
            return o.relu()

        jit_t = torch.jit.script(t)
        t0 = torch.randn(5, 5, device="cuda", dtype=torch.float32).requires_grad_()
        t1 = torch.randn(5, 5, device="cuda", dtype=torch.float32).requires_grad_()

        # run optimization
        for _ in range(5):
            with torch.autocast("cuda", torch.float16):
                jit_o = jit_t(t0, t1)
            jit_o.sum().backward()

        t0.grad = None
        t1.grad = None
        ref_t0 = t0.detach().requires_grad_()
        ref_t1 = t1.detach().requires_grad_()

        with torch.autocast("cuda", torch.float16):
            o = t(ref_t0, ref_t1)
            jit_o = jit_t(t0, t1)
        jit_o.sum().backward()
        o.sum().backward()
        self.assertEqual(o, jit_o)
        self.assertEqual(t0.grad, ref_t0.grad)
        self.assertEqual(t1.grad, ref_t1.grad)
        self.assertEqual(o.dtype, jit_o.dtype)
        self.assertEqual(t0.grad.dtype, ref_t0.grad.dtype)
        self.assertEqual(t1.grad.dtype, ref_t1.grad.dtype)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_call_method_under_autocast(self):
        # [FUNCTION DOC] Checks calling a method through an interface under autocast: a module
        # under CUDA autocast calls an interface-typed scripted implementation. With a global eager
        # autocast context the result equals the expected double matmul on CPU tensors and nothing
        # raises; outside such a context the call raises a runtime error. Needs CUDA.
        @torch.jit.interface
        class Iface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface class declaring a forward method.
            def forward(self, x, y) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method stub with no body.
                pass

        class Impl(Iface):
            # [CLASS DOC] HELPER: Implementation of the interface whose forward
            # returns a matrix product.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the matrix product of x and y.
                return torch.mm(x, y)

        class Thing1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed attribute impl whose forward
            # does a matmul and then calls impl forward, inside CUDA autocast.
            impl: Iface

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Under CUDA autocast, multiplies x and y then calls the
                # implementation forward with that result and x.
                with torch.autocast(device_type="cuda"):
                    a = torch.mm(x, y)
                    b = self.impl.forward(a, x)
                    return b

        scripted_impl = torch.jit.script(Impl())
        thing1 = Thing1()
        thing1.impl = scripted_impl
        scripted_thing1 = torch.jit.script(thing1)
        x = torch.rand([2, 2])
        y = torch.rand([2, 2])

        # make sure this doesn't throw an error
        with torch.autocast(device_type="cuda"):
            ans = scripted_thing1.forward(x, y)
        self.assertEqual(torch.mm(torch.mm(x, y), x), ans)

        # sanity check: this isn't supported currently when global autocasting
        # isn't enabled
        self.assertRaises(RuntimeError, lambda: scripted_thing1.forward(x, y))

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_freeze_autocast_basic(self):
        # [FUNCTION DOC] Checks freezing a scripted module with autocast: the reduced-precision
        # cast op appears exactly twice in the frozen graph and again after running, so the runtime
        # pass does not duplicate them. Also checks eager/script dtype parity. Needs CUDA.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward does a matmul under CUDA autocast.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the matrix product of x and y under CUDA autocast.
                with torch.autocast(device_type="cuda"):
                    return torch.mm(x, y)

        x = torch.rand((3, 4), dtype=torch.float).cuda()
        y = torch.rand((4, 5), dtype=torch.float).cuda()

        mod = TestModule().eval()

        # sanity check
        self._test_autocast(mod, "aten::_autocast_to_reduced_precision", x, y)

        frozen_mod = torch.jit.freeze(torch.jit.script(mod).eval())
        FileCheck().check_count("aten::_autocast_to_reduced_precision", 2, True).run(frozen_mod.graph)

        # make sure that the runtime pass doesn't duplicate autocast nodes
        frozen_mod(x, y)
        optimized_graph = frozen_mod.graph_for(x, y)
        FileCheck().check_count("aten::_autocast_to_reduced_precision", 2, True).run(optimized_graph)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_freeze_autocast_constants(self):
        # [FUNCTION DOC] Checks freezing a module with a constant tensor attribute: after freezing
        # only one cast op remains since the constant is pre-cast, and after a run the runtime pass
        # keeps one via constant propagation. Needs CUDA.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a CUDA float tensor attribute x and multiplying
            # it with the input under CUDA autocast.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the module and stores a random float CUDA tensor
                # of shape 3 by 4 as x.
                super().__init__()
                self.x = torch.rand((3, 4), dtype=torch.float).cuda()

            def forward(self, y):
                # [FUNCTION DOC] HELPER: Under CUDA autocast, returns the matrix product of the
                # stored tensor and y.
                with torch.autocast(device_type="cuda"):
                    return torch.mm(self.x, y)

        y = torch.rand((4, 5), dtype=torch.float).cuda()
        mod = TestModule().eval()

        frozen_mod = torch.jit.freeze(torch.jit.script(mod).eval())
        # freezing should pre-cast the constant self.x to remove one autocast call
        FileCheck().check_count("aten::_autocast_to_reduced_precision", 1, True).run(frozen_mod.graph)

        # the runtime autocasting pass will re-insert the second autocast call,
        # but constant propagation will merge it with the constant that it's casting.
        frozen_mod(y)
        optimized_graph = frozen_mod.graph_for(y)
        FileCheck().check_count("aten::_autocast_to_reduced_precision", 1, True).run(optimized_graph)

    @unittest.skipIf(TEST_CUDA, "CPU-only test")
    def test_jit_autocast_softmax_cpu(self):
        # [FUNCTION DOC] CPU-only test: scripts a function doing softmax under CPU autocast on a
        # bfloat16 tensor, calls it twice and asserts the result is bfloat16.
        def fn(x):
            # [FUNCTION DOC] HELPER: Softmax along dim 0 under CPU autocast.
            with torch.autocast(device_type="cpu"):
                return torch.nn.functional.softmax(x, dim=0)

        fn_s = torch.jit.script(fn)
        x = torch.rand((2, 2), dtype=torch.bfloat16)
        fn_s(x)
        y = fn_s(x)

        self.assertTrue(y.dtype == torch.bfloat16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_jit_autocast_softmax_gpu(self):
        # [FUNCTION DOC] Scripts a softmax under CUDA autocast on a half CUDA tensor, calls it
        # twice and asserts the result is float32. Needs CUDA.
        def fn(x):
            # [FUNCTION DOC] HELPER: Softmax along dim 0 under CUDA autocast.
            with torch.autocast(device_type="cuda"):
                return torch.nn.functional.softmax(x, dim=0)

        fn_s = torch.jit.script(fn)
        x = torch.rand((2, 2), dtype=torch.half).cuda()
        fn_s(x)
        y = fn_s(x)

        self.assertTrue(y.dtype == torch.float)

    def test_ignore_amp(self):
        # [FUNCTION DOC] Checks the ignore-amp switch on a scripted function: after setting ignore
        # amp true and running twice under CPU autocast, the last executed optimized graph contains
        # no reduced-precision autocast cast.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted fixture returning a matrix product of its argument
            # with itself.
            return torch.mm(x, x)

        inp = torch.rand([10, 10], dtype=torch.float)
        foo._set_ignore_amp(True)
        with torch.autocast(device_type="cpu"):
            foo(inp)
            foo(inp)

        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check_not("_autocast_to_reduced").run(g)

class convbn(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a 3 to 64 channel 7 by 7 stride-2 convolution (optional bias)
    # followed by batch norm; used as a model for trace tests.
    def __init__(self, bias_enabled=True):
        # [FUNCTION DOC] HELPER: Builds the convolution (bias controlled by the flag) and batch
        # norm layers.
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 7, stride=2, bias=bias_enabled)
        self.bn = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the convolution then batch norm.
        return self.bn(self.conv(x))

@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestJitTraceAutocast(JitTestCase):
    # [CLASS DOC] Test case for tracing and freezing CPU models under autocast with the JIT
    # autocast pass turned off. setUp saves and sets default dtype float32, builds three models
    # (MnistNet and convbn with and without bias) and inputs, and disables the JIT autocast mode;
    # tearDown restores both settings.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp, saves the default dtype and sets it to float32,
        # creates the models and CPU random inputs, and disables the JIT autocast mode saving the
        # old value.
        super().setUp()
        self.previous_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.models = [MnistNet(),
                       convbn(bias_enabled=True),
                       convbn(bias_enabled=False)]
        self.inputs = [torch.randn(5, 1, 28, 28, device='cpu'),
                       torch.randn(32, 3, 224, 224, device='cpu'),
                       torch.randn(32, 3, 224, 224, device='cpu')]
        self.previous_jit_autocast_pass = torch._C._jit_set_autocast_mode(False)

    def tearDown(self):
        # [FUNCTION DOC] Restores the JIT autocast mode and default dtype, then calls
        # the base tearDown.
        torch._C._jit_set_autocast_mode(self.previous_jit_autocast_pass)
        torch.set_default_dtype(self.previous_default_dtype)
        super().tearDown()

    def test_generate_autocast_jit_trace_model(self):
        # [FUNCTION DOC] Checks that each model can be traced under CPU autocast (cache disabled)
        # in eval and no-grad mode and then frozen without error.
        def test_generate_autocast_jit_trace_model(model, x):
            # [FUNCTION DOC] HELPER: Puts the model in eval mode, traces it with the input under
            # CPU autocast and no grad, and freezes the trace.
            model.eval()
            with torch.autocast(device_type="cpu", cache_enabled=False), torch.no_grad():
                traced_model = torch.jit.trace(model, x)
            traced_model = torch.jit.freeze(traced_model)
        for i in range(self.models.__len__()):
            test_generate_autocast_jit_trace_model(self.models[i], self.inputs[i])

    @xfailIf(IS_ARM64 and IS_LINUX and not IS_CPU_EXT_SVE_SUPPORTED)
    # see https://github.com/pytorch/pytorch/issues/177247
    def test_nchw_autocast_jit_trace_model(self):
        # [FUNCTION DOC] Traces and freezes each model, then compares frozen output
        # with eager output under CPU autocast within 1e-3 tolerances. Expected failure
        # on ARM64 Linux without SVE.
        def test_nchw_autocast_jit_trace_model(model, x):
            # [FUNCTION DOC] HELPER: Traces and freezes the model under CPU autocast, runs frozen
            # and eager copies on cloned input and asserts outputs are close as doubles.
            model.eval()
            with torch.autocast(device_type="cpu", cache_enabled=False), torch.no_grad():
                traced_model = torch.jit.trace(model, x)
            traced_model = torch.jit.freeze(traced_model)
            with torch.no_grad():
                y = traced_model(x.clone())
            with torch.autocast(device_type="cpu"), torch.no_grad():
                y2 = model(x.clone())
            torch.testing.assert_close(y.double(), y2.double(), rtol=1e-03, atol=1e-03)
        for i in range(self.models.__len__()):
            test_nchw_autocast_jit_trace_model(self.models[i], self.inputs[i])

    @xfailIf(IS_ARM64 and IS_LINUX and not IS_CPU_EXT_SVE_SUPPORTED)
    # see https://github.com/pytorch/pytorch/issues/177247
    def test_nhwc_autocast_jit_trace_model(self):
        # [FUNCTION DOC] Same as the nchw test but with models and inputs in
        # channels-last memory format. Inputs with five dimensions are skipped.
        # Expected failure on ARM64 Linux without SVE.
        def test_nhwc_autocast_jit_trace_model(model, x):
            # [FUNCTION DOC] HELPER: Converts the model to channels-last, traces and freezes under
            # CPU autocast, and asserts frozen output is close to eager output as doubles.
            model = model.to(memory_format=torch.channels_last)
            model.eval()
            with torch.autocast(device_type="cpu", cache_enabled=False), torch.no_grad():
                traced_model = torch.jit.trace(model, x.to(memory_format=torch.channels_last))
            traced_model = torch.jit.freeze(traced_model)
            with torch.no_grad():
                y = traced_model(x.clone().to(memory_format=torch.channels_last))
            with torch.autocast(device_type="cpu"), torch.no_grad():
                y2 = model(x.clone().to(memory_format=torch.channels_last))
            torch.testing.assert_close(y.double(), y2.double(), rtol=1e-03, atol=1e-03)
        for i in range(self.models.__len__()):
            if self.inputs[i].size().__len__() == 5:
                # NHWC 3D case not support yet
                continue
            test_nhwc_autocast_jit_trace_model(self.models[i], self.inputs[i])

    def test_cat_promote(self):
        # [FUNCTION DOC] Checks cat type promotion under CPU bfloat16 autocast with fusers off,
        # with and without freezing: traces a module concatenating a float and a bfloat16 tensor,
        # runs it three times, and asserts the graph has a type conversion node.
        # KNOWN ISSUE: the dtype assertions are assertTrue calls with the dtype as the second
        # argument (the message), so they pass whenever the first is truthy and check nothing.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module concatenating two inputs along dim 0.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the concatenation of a and b along dim 0.
                return torch.cat([a, b], 0)

        with torch.jit.fuser("none"):
            # In this testcase, we will check whether cat has done the promotion in AMP with mixed dtype inputs.
            # To avoid the fusion group from TE, we will disable the fuser here.
            for jit_freeze_or_not in [False, True]:
                test_model = TestModel().eval()
                with torch.autocast(device_type="cpu", cache_enabled=False, dtype=torch.bfloat16), torch.no_grad():
                    a = torch.rand(24, 128, 128)
                    b = torch.rand(24, 128, 128, dtype=torch.bfloat16)
                    c = test_model(a, b)
                    traced = torch.jit.trace(test_model, (a, b))
                if jit_freeze_or_not:
                    traced = torch.jit.freeze(traced)
                for _ in range(3):
                    c2 = traced(a, b)
                self.assertTrue(c.dtype, torch.float32)
                self.assertTrue(c2.dtype, torch.float32)
                traced_graph = traced.graph_for(a, b)
                self.assertTrue(any(n.kind() == "aten::to" for n in traced_graph.nodes()))

    def test_script_autocast_cpu(self):
        # [FUNCTION DOC] Checks scripting a function that branches on whether CPU autocast is
        # enabled: scripted and eager results agree inside CPU autocast contexts, and the graph
        # still holds the enabled-query node.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns relu of x if CPU autocast is enabled, else sine of x.
            if torch.is_autocast_cpu_enabled():
                return x.relu()
            else:
                return x.sin()

        fn_s = torch.jit.script(fn)

        x = torch.rand((4, 4)) - 0.5
        with torch.autocast(device_type="cpu"):
            self.assertEqual(fn_s(x), fn(x))

        with torch.autocast(device_type="cpu", enabled=True):
            self.assertEqual(fn_s(x), fn(x))

        self.assertTrue(any("is_autocast_cpu_enabled" in x.kind() for x in fn_s.graph.nodes()))

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_autocast_cuda(self):
        # [FUNCTION DOC] Checks scripting a function that branches on the CUDA autocast query:
        # results agree under CPU autocast and CUDA autocast contexts and the graph keeps the query
        # node. Needs CUDA.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns relu of x if CUDA autocast is enabled, else sine of x.
            if torch.is_autocast_enabled():
                return x.relu()
            else:
                return x.sin()

        fn_s = torch.jit.script(fn)

        x = torch.rand((4, 4)) - 0.5
        with torch.autocast(device_type="cpu"):
            self.assertEqual(fn_s(x), fn(x))

        with torch.autocast(device_type="cuda", enabled=True):
            self.assertEqual(fn_s(x), fn(x))

        self.assertTrue(any("is_autocast_enabled" in x.kind() for x in fn_s.graph.nodes()))


    def test_scripted_aliasing(self):
        # [FUNCTION DOC] Checks alias analysis: the autocast enabled query node must
        # not be movable after the autocast enter node. Asserts one node of each kind
        # exists and the move check is false.
        # torch.is_autocast_enabled should not be able to move inside of the autocast context.
        def fn(x):
            # [FUNCTION DOC] HELPER: Records the CUDA autocast flag, then does a relu inside
            # enabled CUDA autocast, returning both.
            if torch.is_autocast_enabled():
                y = True
            else:
                y = False
            with torch.autocast(device_type="cuda", enabled=True):
                z = x.relu()
            return y, z

        fn_s = torch.jit.script(fn)
        graph = fn_s.graph

        aliasdb = graph.alias_db()

        is_enabled_nodes = graph.findAllNodes("aten::is_autocast_enabled")
        enter_nodes = graph.findAllNodes("prim::Enter")

        self.assertEqual(len(is_enabled_nodes), 1)
        self.assertEqual(len(enter_nodes), 1)

        self.assertFalse(aliasdb.move_after_topologically_valid(is_enabled_nodes[0], enter_nodes[0]))


    def test_script_autocast_enable_and_check(self):
        # [FUNCTION DOC] Checks that scripted and eager functions agree on the CPU
        # autocast flag and result dtype in off, on and nested-off regions, under outer
        # autocast disabled and enabled.
        def fn(x, y) -> tuple[torch.Tensor, bool, torch.Tensor, bool, torch.Tensor, bool]:
            # [FUNCTION DOC] HELPER: Returns matmul results and CPU-autocast-enabled flags for
            # three regions: no autocast, enabled autocast, and nested disabled autocast.
            b1 = torch.is_autocast_cpu_enabled()
            v1 = torch.mm(x, y)
            with torch.autocast(device_type="cpu", enabled=True):
                b2 = torch.is_autocast_cpu_enabled()
                v2 = torch.mm(x, y)
                with torch.autocast(device_type="cpu", enabled=False):
                    b3 = torch.is_autocast_cpu_enabled()
                    v3 = torch.mm(x, y)
            return (v1, b1, v2, b2, v3, b3)

        # bx = is_autocast_cpu_enabled() result should be False iff (vx = mm(x, y)).dtype is float
        def check_fn_results(arr):
            # [FUNCTION DOC] HELPER: Unpacks the six results and asserts each flag is the opposite
            # of whether its matmul is float32.
            [v1, b1, v2, b2, v3, b3] = arr
            self.assertTrue((v1.dtype == torch.float) != b1)
            self.assertTrue((v2.dtype == torch.float) != b2)
            self.assertTrue((v3.dtype == torch.float) != b3)

        x = torch.rand((2, 2), dtype=torch.float)
        y = torch.rand((2, 2), dtype=torch.float)

        fn_s = torch.jit.script(fn)

        with torch.autocast(device_type="cpu", enabled=False):
            check_fn_results(fn(x, y))
            check_fn_results(fn_s(x, y))

        with torch.autocast(device_type="cpu", enabled=True):
            check_fn_results(fn(x, y))
            check_fn_results(fn_s(x, y))


if __name__ == "__main__":
    if sys.version_info < (3, 14):
        run_tests()
