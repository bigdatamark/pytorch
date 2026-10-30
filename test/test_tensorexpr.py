# Owner(s): ["NNC"]
# ruff: noqa: F841

# [MODULE DOC] Tests for the TensorExpr (NNC) fuser in TorchScript. Each test traces or scripts
# a small function, runs it enough times for the profiling executor to optimize, then checks
# that the last executed optimized graph was fully fused and that results match numpy or eager.
# Import time: records whether LLVM is built in (LLVM_ENABLED). Tests run on CPU and, when CUDA
# is available, also CUDA. The suite class is skipped under Dynamo.
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import unittest
import itertools

from torch.testing._internal.common_utils import suppress_warnings, num_profiled_runs, run_tests, skipIfTorchDynamo

from torch.testing._internal.jit_utils import JitTestCase, TensorExprTestOptions

LLVM_ENABLED = torch._C._llvm_enabled()

class BaseTestClass(JitTestCase):
    # [CLASS DOC] Base class for fuser tests. setUp installs TensorExprTestOptions (fuser settings,
    # restored in tearDown) and defines the device list (cpu, plus cuda if available) and dtype
    # list (float32, plus bfloat16 when LLVM is enabled). Offers a full-fusion check.
    def setUp(self):
        # [FUNCTION DOC] Runs the parent setUp, creates the fuser test options object that changes
        # global fuser flags, and sets the devices and dtypes attributes used by tests.
        super().setUp()
        self.tensorexpr_options = TensorExprTestOptions()
        self.devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        self.dtypes = [torch.float32, torch.bfloat16] if LLVM_ENABLED else [torch.float32]

    def tearDown(self):
        # [FUNCTION DOC] Restores the global fuser flags saved by setUp and then runs the parent
        # tearDown.
        self.tensorexpr_options.restore()
        super().tearDown()

    def assertLastGraphAllFused(self):
        # [FUNCTION DOC] Asserts that the most recently executed optimized graph consists entirely
        # of fused groups, via the JitTestCase full-fusion assertion.
        self.assertAllFused(torch.jit.last_executed_optimized_graph())


def warmup_and_run_forward(f, *args):
    # [FUNCTION DOC] HELPER: Calls the function on the given arguments one more time than the
    # configured number of profiled runs (so the profiling executor has optimized it) and returns
    # the result of the last call.
    for _ in range(torch._C._jit_get_num_profiled_runs() + 1):
        results = f(*args)
    return results


@skipIfTorchDynamo()
class TestTensorExprFuser(BaseTestClass):
    # [CLASS DOC] Main fuser test suite; many tests trace a function, warm it up and assert full
    # fusion plus numerical agreement with numpy or eager results.
    def test_easy(self):
        # [FUNCTION DOC] Traces a two-input add of 1024 elements, runs it, asserts full fusion and
        # compares against numpy addition.
        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of its two inputs.
            aaa = torch.add(x, y)
            return aaa

        traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024)))

        a = torch.rand(1024)
        b = torch.rand(1024)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(a.numpy() + b.numpy(), x.numpy())

    def test_three_arg(self):
        # [FUNCTION DOC] Traces a chained three-input add, asserts full fusion and compares with
        # numpy.
        def easy(x, y, z):
            # [FUNCTION DOC] HELPER: Returns the sum of three inputs computed as two adds.
            aaa = torch.add(x, y)
            bbb = torch.add(aaa, z)
            return bbb

        traced = torch.jit.trace(
            easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024))
        )

        a = torch.rand(1024)
        b = torch.rand(1024)
        c = torch.rand(1024)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = a.numpy() + b.numpy() + c.numpy()
        np.testing.assert_allclose(npr, x.numpy())

    def test_four_arg(self):
        # [FUNCTION DOC] For each device, traces an addcmul of a sum with two more tensors, asserts
        # full fusion and compares the traced result with the eager result.
        def run_addcmul(x, y, z, w):
            # [FUNCTION DOC] HELPER: Returns addcmul of the sum of x and y with z and w.
            c = torch.addcmul(torch.add(x, y), z, w)
            return c

        for dev in self.devices:
            rand_a = torch.rand(1024, dtype=torch.float, device=dev)
            rand_b = torch.rand(1024, dtype=torch.float, device=dev)
            rand_c = torch.rand(1024, dtype=torch.float, device=dev)
            rand_d = torch.rand(1024, dtype=torch.float, device=dev)

            traced = torch.jit.trace(
                run_addcmul,
                (
                    torch.zeros(1024, dtype=torch.float, device=dev),
                    torch.zeros(1024, dtype=torch.float, device=dev),
                    torch.zeros(1024, dtype=torch.float, device=dev),
                    torch.zeros(1024, dtype=torch.float, device=dev),
                ),
            )

            x = warmup_and_run_forward(traced, rand_a, rand_b, rand_c, rand_d)
            self.assertLastGraphAllFused()
            y = run_addcmul(rand_a, rand_b, rand_c, rand_d)
            np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy(), atol=1e-6)

    def test_three_arg2(self):
        # [FUNCTION DOC] For each device, traces a three-input 32x32 add, calls the traced function
        # once, then warms up and checks full fusion and numpy equality.
        for device in self.devices:
            def test(x, y, z):
                # [FUNCTION DOC] HELPER: Returns the sum of three inputs computed as two adds.
                aaa = torch.add(x, y)
                bbb = torch.add(aaa, z)
                return bbb

            M = 32
            N = 32
            traced = torch.jit.trace(
                test,
                (
                    torch.rand(M, N, device=device),
                    torch.rand(M, N, device=device),
                    torch.rand(M, N, device=device),
                ),
            )

            a = torch.rand(M, N, device=device)
            b = torch.rand(M, N, device=device)
            c = torch.rand(M, N, device=device)
            x = traced(a, b, c)
            x = warmup_and_run_forward(traced, a, b, c)
            self.assertLastGraphAllFused()
            npr = a.cpu().numpy() + b.cpu().numpy() + c.cpu().numpy()
            np.testing.assert_allclose(npr, x.cpu().numpy())

    def test_broadcast3(self):
        # [FUNCTION DOC] For each device, runs a nested body over two shape configurations to check
        # fusion of adds with broadcasting among three differently ranked inputs.
        for device in self.devices:
            def test_body(M, N, L, K):
                # [FUNCTION DOC] HELPER: Traces a three-input add with shapes derived from the four
                # size arguments (2D, 3D and 4D with size-one dims), asserts full fusion and
                # compares with numpy broadcasting.
                def test(x, y, z):
                    # [FUNCTION DOC] HELPER: Returns the sum of three inputs computed as two adds.
                    v1 = torch.add(x, y)
                    v2 = torch.add(v1, z)
                    return v2

                a_shape = [M, N]
                b_shape = [L, M, 1]
                c_shape = [K, L, 1, 1]
                traced = torch.jit.trace(
                    test,
                    (
                        torch.rand(*a_shape, device=device),
                        torch.rand(*b_shape, device=device),
                        torch.rand(*c_shape, device=device),
                    ),
                )

                a = torch.rand(*a_shape, device=device)
                b = torch.rand(*b_shape, device=device)
                c = torch.rand(*c_shape, device=device)
                x = warmup_and_run_forward(traced, a, b, c)
                self.assertLastGraphAllFused()
                npr = a.cpu().numpy() + b.cpu().numpy() + c.cpu().numpy()
                np.testing.assert_allclose(npr, x.cpu().numpy())

            test_configs = [[5, 2, 7, 3], [8, 8, 8, 8]]
            for test_config in test_configs:
                test_body(*test_config)

    def test_all_combos(self):
        # [FUNCTION DOC] Traces a function with several interdependent adds of three inputs
        # (reusing intermediate values), asserts full fusion and compares with the numpy version.
        def easy(x, y, z):
            # [FUNCTION DOC] HELPER: Torch version of the multi-add dataflow returning the last
            # sum.
            a = torch.add(x, y)
            b = torch.add(a, z)
            c = torch.add(x, b)
            d = torch.add(c, a)
            return d

        def np_easy(x, y, z):
            # [FUNCTION DOC] HELPER: Numpy version of the same dataflow, used as the reference.
            a = x + y
            b = a + z
            c = x + b
            d = c + a
            return d

        traced = torch.jit.trace(
            easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024))
        )

        a = torch.rand(1024)
        b = torch.rand(1024)
        c = torch.rand(1024)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = np_easy(a.numpy(), b.numpy(), c.numpy())
        np.testing.assert_allclose(npr, x.numpy())

    def test_rank_two(self):
        # [FUNCTION DOC] Same as the all-combos test but with 32x32 inputs.
        def easy(x, y, z):
            # [FUNCTION DOC] HELPER: Torch version of the multi-add dataflow returning the last
            # sum.
            a = torch.add(x, y)
            b = torch.add(a, z)
            c = torch.add(x, b)
            d = torch.add(c, a)
            return d

        def np_easy(x, y, z):
            # [FUNCTION DOC] HELPER: Numpy version of the same dataflow, used as the reference.
            a = x + y
            b = a + z
            c = x + b
            d = c + a
            return d

        shape = 32, 32
        traced = torch.jit.trace(
            easy, (torch.rand(shape), torch.rand(shape), torch.rand(shape))
        )

        a = torch.rand(shape)
        b = torch.rand(shape)
        c = torch.rand(shape)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = np_easy(a.numpy(), b.numpy(), c.numpy())
        np.testing.assert_allclose(npr, x.numpy())

    def test_broadcast(self):
        # [FUNCTION DOC] Traces an add of a matrix, a vector and a matrix (vector broadcast),
        # asserts full fusion and compares with numpy.
        def easy(x, y, z):
            # [FUNCTION DOC] HELPER: Returns the sum of three inputs computed as two adds.
            a = torch.add(x, y)
            b = torch.add(a, z)
            return b

        def np_easy(x, y, z):
            # [FUNCTION DOC] HELPER: Numpy reference for the sum of three inputs.
            a = x + y
            b = a + z
            return b

        N = 32
        traced = torch.jit.trace(easy, (torch.rand(N, N), torch.rand(N), torch.rand(N, N)))

        a = torch.rand(N, N)
        b = torch.rand(N)
        c = torch.rand(N, N)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = np_easy(a.numpy(), b.numpy(), c.numpy())
        np.testing.assert_allclose(npr, x.numpy())

    def test_broadcast_2(self):
        # [FUNCTION DOC] Traces a function that adds a captured one-element zero tensor to the sum
        # of a 3x4, a 3x1 and a 4-element input; asserts full fusion and compares with numpy.
        zero = torch.tensor([0.0], dtype=torch.float)

        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Adds x and y, then the captured zero tensor, then z.
            aaa = torch.add(x, y)
            bbb = torch.add(zero, aaa)
            return torch.add(bbb, z)

        def foo_np(x, y, z):
            # [FUNCTION DOC] HELPER: Numpy reference doing the same additions.
            a = x + y
            b = zero.numpy() + a
            return b + z

        x = torch.rand(3, 4)
        y = torch.ones(3, 1)
        z = torch.rand(4)
        traced = torch.jit.trace(foo, (x, y, z))

        r = warmup_and_run_forward(traced, x, y, z)
        self.assertLastGraphAllFused()

        rnp = foo_np(x.numpy(), y.numpy(), z.numpy())
        np.testing.assert_allclose(r, rnp)

    def test_broadcast_big2(self):
        # [FUNCTION DOC] Larger variant of the previous broadcast test using 32x1024, 32x1 and 1024
        # element inputs; asserts full fusion and numpy equality.
        zero = torch.tensor([0.0], dtype=torch.float)

        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Adds x and y, then the captured zero tensor, then z.
            aaa = torch.add(x, y)
            bbb = torch.add(zero, aaa)
            return torch.add(bbb, z)

        def foo_np(x, y, z):
            # [FUNCTION DOC] HELPER: Numpy reference doing the same additions.
            a = x + y
            b = zero.numpy() + a
            return b + z

        x = torch.rand(32, 1024)
        y = torch.ones(32, 1)
        z = torch.rand(1024)
        traced = torch.jit.trace(foo, (x, y, z))

        r = warmup_and_run_forward(traced, x, y, z)
        self.assertLastGraphAllFused()
        rnp = foo_np(x.numpy(), y.numpy(), z.numpy())
        np.testing.assert_allclose(r, rnp)

    def test_alpha(self):
        # [FUNCTION DOC] Traces an add with alpha 2 of a one-element tensor with itself and checks
        # the result equals the input plus twice the input. Does not assert fusion.
        def alpha(x):
            # [FUNCTION DOC] HELPER: Returns x plus x scaled by alpha two.
            aaa = torch.add(x, x, alpha=2.0)
            return aaa

        traced = torch.jit.trace(alpha, (torch.tensor([1.0])))

        a = torch.tensor([1.0])
        x = traced(a)
        np.testing.assert_allclose(a.numpy() + 2.0 * a.numpy(), x.numpy())

    @suppress_warnings
    def test_constant(self):
        # [FUNCTION DOC] Warnings are suppressed. Traces a function that creates a constant tensor
        # inside and adds it to the input; asserts full fusion and result equal input plus one.
        def constant(x):
            # [FUNCTION DOC] HELPER: Creates a one-element constant tensor and returns its sum with
            # x.
            bbb = torch.tensor([1.0])
            aaa = torch.add(x, bbb)
            return aaa

        traced = torch.jit.trace(constant, (torch.tensor([1.0])))

        a = torch.tensor([1.0])
        x = warmup_and_run_forward(traced, a)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(a.numpy() + 1.0, x.numpy())

    def test_add_sub(self):
        # [FUNCTION DOC] Traces an add followed by a subtract of three inputs, asserts full fusion
        # and compares with numpy.
        def easy(x, y, z):
            # [FUNCTION DOC] HELPER: Returns x plus y minus z using separate add and sub calls.
            aaa = torch.add(x, y)
            bbb = torch.sub(aaa, z)
            return bbb

        traced = torch.jit.trace(
            easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024))
        )

        a = torch.rand(1024)
        b = torch.rand(1024)
        c = torch.rand(1024)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(a.numpy() + b.numpy() - c.numpy(), x.numpy())

    def test_promotion(self):
        # [FUNCTION DOC] Traces an add of an int32 and a float32 tensor (type promotion), asserts
        # full fusion and compares with numpy.
        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of two inputs.
            aaa = torch.add(x, y)
            return aaa

        traced = torch.jit.trace(
            easy,
            (torch.zeros(1024, dtype=torch.int32), torch.rand(1024, dtype=torch.float32)),
        )

        a = torch.zeros(1024, dtype=torch.int32)
        b = torch.rand(1024, dtype=torch.float32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(a.numpy() + b.numpy(), x.numpy())

    def test_double(self):
        # [FUNCTION DOC] Traces add then multiply on float64 tensors of length 8, asserts full
        # fusion and compares with numpy.
        TENSOR_LEN = 8

        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y multiplied by y.
            aaa = torch.add(x, y)
            bbb = torch.mul(aaa, y)
            return bbb

        traced = torch.jit.trace(
            easy,
            (torch.rand(TENSOR_LEN, dtype=torch.float64), torch.full((TENSOR_LEN,), 0.5, dtype=torch.float64)),
        )

        a = torch.rand(TENSOR_LEN, dtype=torch.double)
        b = torch.full((TENSOR_LEN,), 0.5, dtype=torch.double)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_short(self):
        # [FUNCTION DOC] Same as the double test but with int16 inputs.
        TENSOR_LEN = 8

        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y multiplied by y.
            aaa = torch.add(x, y)
            bbb = torch.mul(aaa, y)
            return bbb

        traced = torch.jit.trace(
            easy,
            (torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int16),
             torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int16)),
        )

        a = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int16)
        b = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int16)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_char(self):
        # [FUNCTION DOC] Same as the double test but with int8 inputs.
        TENSOR_LEN = 8

        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y multiplied by y.
            aaa = torch.add(x, y)
            bbb = torch.mul(aaa, y)
            return bbb

        traced = torch.jit.trace(
            easy,
            (torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8),
             torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8)),
        )

        a = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8)
        b = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_int64_promotion(self):
        # [FUNCTION DOC] Same arithmetic with an int8 and an int64 input (promotion to int64);
        # asserts full fusion and numpy equality.
        TENSOR_LEN = 8

        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y multiplied by y.
            aaa = torch.add(x, y)
            bbb = torch.mul(aaa, y)
            return bbb

        traced = torch.jit.trace(
            easy,
            (torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8),
             torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int64)),
        )

        a = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8)
        b = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int64)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_eq(self):
        # [FUNCTION DOC] Traces an elementwise equality of two inputs, runs it on equal int32 zeros
        # and checks full fusion and that all results are true (ones).
        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise equality of x and y.
            c = torch.eq(x, y)
            return c

        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        a = torch.zeros(1024, dtype=torch.int32)
        b = torch.zeros(1024, dtype=torch.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_ne(self):
        # [FUNCTION DOC] Traces an elementwise not-equal; int32 zeros against ones; checks full
        # fusion and all results true.
        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise not-equal of x and y.
            c = torch.ne(x, y)
            return c

        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        a = torch.zeros(1024, dtype=torch.int32)
        b = torch.ones(1024, dtype=torch.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_ge(self):
        # [FUNCTION DOC] Traces an elementwise greater-or-equal; int32 fives against zeros; checks
        # full fusion and all results true.
        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise greater-or-equal of x and y.
            c = torch.ge(x, y)
            return c

        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        aa = np.empty([1024], dtype=np.int32)
        aa.fill(5)
        a = torch.from_numpy(aa)
        b = torch.zeros(1024, dtype=torch.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_gt(self):
        # [FUNCTION DOC] Traces an elementwise greater-than; int32 ones against zeros; checks full
        # fusion and all results true.
        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise greater-than of x and y.
            c = torch.gt(x, y)
            return c

        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        a = torch.ones(1024, dtype=torch.int32)
        b = torch.zeros(1024, dtype=torch.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_le(self):
        # [FUNCTION DOC] Traces an elementwise less-or-equal; int32 fives against zeros; checks
        # full fusion and all results false (zeros).
        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise less-or-equal of x and y.
            c = torch.le(x, y)
            return c

        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        aa = np.empty([1024], dtype=np.int32)
        aa.fill(5)
        a = torch.from_numpy(aa)
        b = torch.zeros(1024, dtype=torch.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.zeros(1024), x.numpy())

    def test_lt(self):
        # [FUNCTION DOC] For each device, traces an elementwise less-than; int32 ones against
        # zeros; checks full fusion and all results false.
        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise less-than of x and y.
            c = torch.lt(x, y)
            return c

        for dev in self.devices:
            traced = torch.jit.trace(easy, (torch.zeros(1024, device=dev), torch.zeros(1024, device=dev)))
            a = torch.ones(1024, dtype=torch.int32, device=dev)
            b = torch.zeros(1024, dtype=torch.int32, device=dev)
            x = warmup_and_run_forward(traced, a, b)
            self.assertLastGraphAllFused()
            np.testing.assert_allclose(np.zeros(1024), x.cpu().numpy())

    @suppress_warnings
    def test_min_max(self):
        # [FUNCTION DOC] Warnings are suppressed. Traces a maximum of a minimum of two inputs with
        # a constant 4, compares with the numpy equivalent on random values and checks full fusion.
        def test(x, y):
            # [FUNCTION DOC] HELPER: Returns the max of the elementwise min of x and y with a
            # constant 4.
            return torch.max(torch.min(x, y), torch.tensor([4.0]))

        traced = torch.jit.trace(test, (torch.zeros(1024), torch.zeros(1024)))
        a = 8.0 * torch.rand(1024)
        b = 8.0 * torch.rand(1024)
        np.testing.assert_allclose(
            warmup_and_run_forward(traced, a, b), np.maximum(np.minimum(a.numpy(), b.numpy()), [4.0])
        )
        self.assertLastGraphAllFused()

    def test_min_max_reduction(self):
        # [FUNCTION DOC] Traces a function summing the full-tensor min and max reductions, compares
        # with numpy and checks full fusion.
        def test(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the global minimum and maximum of x.
            return torch.min(x) + torch.max(x)

        traced = torch.jit.trace(test, (torch.zeros(1024)))
        a = 8.0 * torch.rand(1024)
        np.testing.assert_allclose(warmup_and_run_forward(traced, a), np.amin(a.numpy()) + np.amax(a.numpy()))
        self.assertLastGraphAllFused()

    def test_min_max_reduction2(self):
        # [FUNCTION DOC] Same as the previous reduction test but uses the method form of min and
        # max.
        def test(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the method-form minimum and maximum of x.
            return x.min() + x.max()

        traced = torch.jit.trace(test, (torch.zeros(1024)))
        a = 8.0 * torch.rand(1024)
        np.testing.assert_allclose(warmup_and_run_forward(traced, a), np.amin(a.numpy()) + np.amax(a.numpy()))
        self.assertLastGraphAllFused()

    def test_min_max_reduction_dim1(self):
        # [FUNCTION DOC] Traces min plus max reductions along dimension 1 of a 16x16 input, takes
        # only values (not indices), compares with numpy and checks full fusion.
        def test(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the values of the min and max along
            # dimension one.
            return torch.min(x, 1)[0] + torch.max(x, 1)[0]

        traced = torch.jit.trace(test, (torch.zeros(16, 16)))
        a = 8.0 * torch.rand(16, 16)
        np.testing.assert_allclose(warmup_and_run_forward(traced, a), np.amin(
            a.numpy(), axis=1) + np.amax(a.numpy(), axis=1))
        self.assertLastGraphAllFused()

    def test_min_max_reduction_dim1_2(self):
        # [FUNCTION DOC] Traces a min reduction along dimension 1 of the squared input; compares
        # the values output with numpy and checks full fusion.
        def test(x):
            # [FUNCTION DOC] HELPER: Returns the min along dimension one (values and indices) of x
            # times x.
            return torch.min(x * x, 1)

        traced = torch.jit.trace(test, (torch.zeros(16, 16)))
        a = 8.0 * torch.rand(16, 16)
        np.testing.assert_allclose(warmup_and_run_forward(traced, a)[0], np.amin((a * a).numpy(), axis=1))
        self.assertLastGraphAllFused()

    def test_clamp(self):
        # [FUNCTION DOC] For each device, traces a clamp of x plus 3 to the range 0 to 6, compares
        # with numpy clip and checks full fusion.
        def test(x):
            # [FUNCTION DOC] HELPER: Returns x plus 3 clamped to the range zero to six.
            return torch.clamp(x + 3.0, 0.0, 6.0)

        for dev in self.devices:
            traced = torch.jit.trace(test, (torch.zeros(1024, device=dev)))
            a = 20.0 * torch.rand(1024, device=dev) - 10.0
            an = a.cpu().numpy()
            np.testing.assert_allclose(warmup_and_run_forward(traced, a).cpu(), np.clip(an + 3.0, 0.0, 6.0))
            self.assertLastGraphAllFused()

    def test_relu(self):
        # [FUNCTION DOC] For each device, traces relu followed by a clamp to at most 0.5, compares
        # with the numpy equivalent and checks full fusion.
        def test(x):
            # [FUNCTION DOC] HELPER: Returns relu of x clamped to the range zero to one half.
            return torch.clamp(F.relu(x), 0, 0.5)

        for dev in self.devices:
            traced = torch.jit.trace(test, (torch.zeros(1024, device=dev)))
            a = 20.0 * torch.rand(1024, device=dev) - 10.0
            an = a.cpu().numpy()
            np.testing.assert_allclose(warmup_and_run_forward(traced, a).cpu(), np.clip((np.maximum(0, an)), 0, 0.5))
            self.assertLastGraphAllFused()

    def test_reps(self):
        # [FUNCTION DOC] Traces an add and reruns it 32 times with ones and zeros, checking the
        # result each time is all ones. Does not assert fusion.
        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y.
            c = torch.add(x, y)
            return c

        traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024)))

        for _ in range(32):
            a = torch.ones(1024)
            b = torch.zeros(1024)
            x = warmup_and_run_forward(traced, a, b)
            np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_add_const_rhs(self):
        # [FUNCTION DOC] Traces adding a constant 3 on the right side of a 4-element tensor, checks
        # full fusion and the value.
        def test(x):
            # [FUNCTION DOC] HELPER: Returns x plus the constant three.
            return x + 3.0

        traced = torch.jit.trace(test, torch.rand(4))
        x = torch.rand(4)
        y = warmup_and_run_forward(traced, x)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(x.numpy() + 3.0, y.numpy())

    def test_int_output(self):
        # [FUNCTION DOC] Traces a product of three random int32 tensors, checks full fusion and
        # equality with the numpy product.
        def test(x, y, z):
            # [FUNCTION DOC] HELPER: Returns the product of the three inputs.
            return x * y * z

        xs = [(torch.rand(4) * 3 + 1).to(torch.int32) for i in range(3)]
        x, y, z = xs
        xn, yn, zn = (t.numpy() for t in xs)
        traced = torch.jit.trace(test, (x, y, z))
        res = warmup_and_run_forward(traced, x, y, z)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(xn * yn * zn, res.numpy())

    def test_binary_ops(self):
        # [FUNCTION DOC] Table-driven test over many binary ops (comparisons and arithmetic) times
        # devices times dtypes (float32 and bfloat16 when LLVM is on), skipping lerp for bfloat16.
        # Traces each op on random data and checks full fusion. For bfloat16 the reference is the
        # traced function fed float32 inputs then cast back (not cast for comparisons), with looser
        # tolerance; otherwise the eager result is the reference. Compares via assertEqual.
        def test_atan2(x, y):
            # [FUNCTION DOC] HELPER: Returns atan2 of the sum of x and y with y.
            c = torch.atan2(torch.add(x, y), y)
            return c

        def test_gt(x, y):
            # [FUNCTION DOC] HELPER: Returns whether the sum of x and y is greater than y.
            c = torch.gt(torch.add(x, y), y)
            return c

        def test_ge(x, y):
            # [FUNCTION DOC] HELPER: Returns whether the sum of x and y is at least y.
            c = torch.ge(torch.add(x, y), y)
            return c

        def test_lt(x, y):
            # [FUNCTION DOC] HELPER: Returns whether the sum of x and y is less than y.
            c = torch.lt(torch.add(x, y), y)
            return c

        def test_le(x, y):
            # [FUNCTION DOC] HELPER: Returns whether the sum of x and y is at most y.
            c = torch.le(torch.add(x, y), y)
            return c

        def test_lerp(x, y):
            # [FUNCTION DOC] HELPER: Returns the interpolation from x plus one toward x with weight
            # two.
            c = torch.lerp(torch.add(x, 1), x, 2.0)
            return c

        def test_mul(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y multiplied by y.
            c = torch.mul(torch.add(x, y), y)
            return c

        def test_ne(x, y):
            # [FUNCTION DOC] HELPER: Returns whether the sum of x and y differs from y.
            c = torch.ne(torch.add(x, y), y)
            return c

        def test_div(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y divided by two.
            c = torch.div(torch.add(x, y), 2)
            return c

        def test_eq(x, y):
            # [FUNCTION DOC] HELPER: Returns whether the sum of x and y equals y.
            c = torch.eq(torch.add(x, y), y)
            return c

        def test_fmod(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y modulo two using the C-style fmod.
            c = torch.fmod(torch.add(x, y), 2)
            return c

        def test_sub(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y minus x.
            c = torch.sub(torch.add(x, y), x)
            return c

        def test_remainder(x, y):
            # [FUNCTION DOC] HELPER: Returns the remainder of the sum of x and y divided by 3.
            c = torch.remainder(torch.add(x, y), 3.0)
            return c

        def test_pow(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y raised to the power two.
            c = torch.pow(torch.add(x, y), 2.0)
            return c

        def test_type_as(x, y):
            # [FUNCTION DOC] HELPER: Returns x converted to the dtype of the sum of x and y.
            return x.type_as(torch.add(x, y))

        cmp_fns = {
            test_gt,
            test_ge,
            test_lt,
            test_le,
            test_ne,
            test_eq
        }

        non_cmp_fns = {
            test_atan2,
            test_lerp,
            test_mul,
            test_div,
            test_fmod,
            test_sub,
            test_remainder,
            test_pow,
            test_type_as,
        }

        all_test_fns = cmp_fns.union(non_cmp_fns)
        fn_dev_dtype = itertools.product(all_test_fns, self.devices, self.dtypes)
        for torch_fn, dev, data_type in fn_dev_dtype:
            if torch_fn is test_lerp and data_type is torch.bfloat16:
                continue
            rand_a = torch.rand(1024, dtype=data_type, device=dev)
            rand_b = torch.rand(1024, dtype=data_type, device=dev)
            in1 = 20 * torch.rand(1024, dtype=data_type, device=dev)
            in2 = 20 * torch.rand(1024, dtype=data_type, device=dev)
            traced = torch.jit.trace(torch_fn, (in1, in2))
            x = warmup_and_run_forward(traced, rand_a, rand_b)
            self.assertLastGraphAllFused()

            _atol = 2e-3
            _rtol = 1e-5
            if data_type is torch.bfloat16:
                # Compared to aten logic, NNC could save additional BF16/Fp32 conversion.
                # Take d = a + b - c as an example, the aten logic is as follows at
                # operator level:
                #    tmp = to_bf16(to_fp32(a) + to_fp32(b))
                #    d = to_bf16(to_fp32(tmp) + to_fp32(c))
                # But NNC could fuse the compression and remove the redundant conversions.
                # The final statement is as follows
                #    d = to_bf16(to_fp32(a) + to_fp32(b) + to_fp32(c))
                # Hence, we simulate NNC computation by feeding fp32 tensors and converting
                # the result tensor back to bf16. The simulation could avoid the numeric
                # deviation to simplify the result comparison
                y = warmup_and_run_forward(traced, rand_a.float(), rand_b.float())
                if torch_fn not in cmp_fns:
                    y = y.bfloat16()
                _atol = 2e-2
            else:
                y = torch_fn(rand_a, rand_b)
            self.assertEqual(x.cpu(), y.cpu(), atol=_atol, rtol=_rtol)

    def test_unary_ops(self):
        # [FUNCTION DOC] Table-driven test over many unary math ops (sin, log, rounding, relu,
        # sigmoid, hardtanh, threshold and others) times devices times dtypes (float32 and bfloat16
        # when LLVM is on), with a fixed random seed. Each op is traced on random data, checks full
        # fusion and compares with a reference: eager for float32, or the traced function on
        # float32 inputs cast to bfloat16 otherwise (except ops in the gpu-only set). Skips lgamma
        # on CUDA. The NaN comparison part is commented out as disabled.
        # KNOWN ISSUE: the gpu-only set (erf, erfc) is unioned into the run set, so those ops run
        # on every device despite the name; the nested helper with the misspelled rqrt name is
        # never used.
        def test_cast_float(x, y):
            # [FUNCTION DOC] HELPER: Returns the cast to float of the sum of x and y.
            c = torch.ops.aten._cast_Float(torch.add(x, y))
            return c

        def test_round(x, y):
            # [FUNCTION DOC] HELPER: Returns the round of the sum of x and y.
            c = torch.round(torch.add(x, y))
            return c

        def test_sin(x, y):
            # [FUNCTION DOC] HELPER: Returns the sine of the sum of x and y.
            c = torch.sin(torch.add(x, y))
            return c

        def test_asin(x, y):
            # [FUNCTION DOC] HELPER: Returns the arcsine of the sum of x and y.
            c = torch.asin(torch.add(x, y))
            return c

        def test_sinh(x, y):
            # [FUNCTION DOC] HELPER: Returns the hyperbolic sine of the sum of x and y.
            c = torch.sinh(torch.add(x, y))
            return c

        def test_cos(x, y):
            # [FUNCTION DOC] HELPER: Returns the cosine of the sum of x and y.
            c = torch.cos(torch.add(x, y))
            return c

        def test_acos(x, y):
            # [FUNCTION DOC] HELPER: Returns the arccosine of the sum of x and y.
            c = torch.acos(torch.add(x, y))
            return c

        def test_cosh(x, y):
            # [FUNCTION DOC] HELPER: Returns the hyperbolic cosine of the sum of x and y.
            c = torch.cosh(torch.add(x, y))
            return c

        def test_tan(x, y):
            # [FUNCTION DOC] HELPER: Returns the tangent of the sum of x and y.
            c = torch.tan(torch.add(x, y))
            return c

        def test_atan(x, y):
            # [FUNCTION DOC] HELPER: Returns the arctangent of the sum of x and y.
            c = torch.atan(torch.add(x, y))
            return c

        def test_tanh(x, y):
            # [FUNCTION DOC] HELPER: Returns the hyperbolic tangent of the sum of x and y.
            c = torch.tanh(torch.add(x, y))
            return c

        def test_sqrt(x, y):
            # [FUNCTION DOC] HELPER: Returns the square root of the sum of x and y.
            c = torch.sqrt(torch.add(x, y))
            return c

        def test_rsqrt(x, y):
            # [FUNCTION DOC] HELPER: Returns the reciprocal square root of the sum of x and y.
            c = torch.rsqrt(torch.add(x, y))
            return c

        def test_floor(x, y):
            # [FUNCTION DOC] HELPER: Returns the floor of the sum of x and y.
            c = torch.floor(torch.add(x, y))
            return c

        def test_ceil(x, y):
            # [FUNCTION DOC] HELPER: Returns the ceiling of the sum of x and y.
            c = torch.ceil(torch.add(x, y))
            return c

        def test_trunc(x, y):
            # [FUNCTION DOC] HELPER: Returns the truncation of the sum of x and y.
            c = torch.trunc(torch.add(x, y))
            return c

        def test_abs(x, y):
            # [FUNCTION DOC] HELPER: Returns the absolute value of the sum of x and y.
            c = torch.abs(torch.add(x, y))
            return c

        def test_log(x, y):
            # [FUNCTION DOC] HELPER: Returns the natural log of the sum of x and y.
            c = torch.log(torch.add(x, y))
            return c

        def test_log2(x, y):
            # [FUNCTION DOC] HELPER: Returns the base 2 log of the sum of x and y.
            c = torch.log2(torch.add(x, y))
            return c

        def test_log10(x, y):
            # [FUNCTION DOC] HELPER: Returns the base 10 log of the sum of x and y.
            c = torch.log10(torch.add(x, y))
            return c

        def test_log1p(x, y):
            # [FUNCTION DOC] HELPER: Returns the log of one plus the sum of x and y.
            c = torch.log1p(torch.add(x, y))
            return c

        def test_rqrt(x, y):
            # [FUNCTION DOC] HELPER: Returns the reciprocal square root of the sum of x and y.
            c = torch.rsqrt(torch.add(x, y))
            return c

        def test_erf(x, y):
            # [FUNCTION DOC] HELPER: Returns the error function of the sum of x and y.
            c = torch.erf(torch.add(x, y))
            return c

        def test_exp(x, y):
            # [FUNCTION DOC] HELPER: Returns the exponential of the sum of x and y.
            c = torch.exp(torch.add(x, y))
            return c

        def test_expm1(x, y):
            # [FUNCTION DOC] HELPER: Returns the exponential minus one of the sum of x and y.
            c = torch.expm1(torch.add(x, y))
            return c

        def test_erfc(x, y):
            # [FUNCTION DOC] HELPER: Returns the complementary error function of the sum of x and
            # y.
            c = torch.erfc(torch.add(x, y))
            return c

        def test_frac(x, y):
            # [FUNCTION DOC] HELPER: Returns the fractional part of the sum of x and y.
            c = torch.frac(torch.add(x, y))
            return c

        def test_lgamma(x, y):
            # [FUNCTION DOC] HELPER: Returns the log-gamma of the sum of x and y.
            c = torch.lgamma(torch.add(x, y))
            return c

        def test_sigmoid(x, y):
            # [FUNCTION DOC] HELPER: Returns the sigmoid of the sum of x and y.
            c = torch.sigmoid(torch.add(x, y))
            return c

        def test_reciprocal(x, y):
            # [FUNCTION DOC] HELPER: Returns the reciprocal of the sum of x and y.
            c = torch.reciprocal(torch.add(x, y))
            return c

        def test_neg(x, y):
            # [FUNCTION DOC] HELPER: Returns the negation of the sum of x and y.
            c = torch.neg(torch.add(x, y))
            return c

        def test_relu(x, y):
            # [FUNCTION DOC] HELPER: Returns the relu of the sum of x and y.
            c = torch.relu(torch.add(x, y))
            return c

        def test_hardtanh(x, y):
            # [FUNCTION DOC] HELPER: Returns the hardtanh with bounds -1 and 1 of the sum of x and
            # y.
            c = F.hardtanh(torch.add(x, y), -1.0, 1.0)
            return c

        def test_threshold(x, y):
            # [FUNCTION DOC] HELPER: Returns the threshold (cutoff 0.5, replacement 10) of the sum
            # of x and y.
            c = F.threshold(torch.add(x, y), 0.5, 10)
            return c

        gpu_only_fns = {
            test_erf,
            test_erfc
        }
        fns = {
            test_round,
            test_sin,
            test_asin,
            test_sinh,
            test_cos,
            test_acos,
            test_cosh,
            test_tan,
            test_atan,
            test_sqrt,
            test_floor,
            test_ceil,
            test_trunc,
            test_abs,
            test_log,
            test_log2,
            test_log10,
            test_log1p,
            test_rsqrt,
            test_exp,
            test_expm1,
            test_frac,
            test_lgamma,
            test_reciprocal,
            test_neg,
            test_threshold,
            test_relu,
            test_tanh,
            test_hardtanh,
            test_sigmoid,
        }
        fn_dev_dtype = itertools.product(gpu_only_fns.union(fns), self.devices, self.dtypes)

        torch.manual_seed(0)
        for torch_fn, dev, data_type in fn_dev_dtype:
            if torch_fn is test_lgamma and dev == "cuda":
                # lgamma_cuda does not support BF16
                continue
            rand_a = torch.rand(1024, dtype=data_type, device=dev)
            rand_b = torch.rand(1024, dtype=data_type, device=dev)

            ins = 20 * torch.rand(1024, dtype=data_type, device=dev)
            cc = np.empty([1024], dtype=np.float32)
            cc.fill(np.nan)
            nans = torch.from_numpy(cc).to(dev)
            traced = torch.jit.trace(torch_fn, (ins, ins))
            x = warmup_and_run_forward(traced, rand_a, rand_b)
            self.assertLastGraphAllFused()

            _atol = 5e-3 if data_type is torch.bfloat16 else 2e-3
            _rtol = 1e-5
            if data_type is torch.bfloat16 and torch_fn not in gpu_only_fns:
                y = warmup_and_run_forward(traced, rand_a.float(), rand_b.float())
                y = y.bfloat16()
            else:
                y = torch_fn(rand_a, rand_b)

            self.assertEqual(x.cpu(), y.cpu(), atol=_atol, rtol=_rtol)
            # nans
            # TODO: reenable. Currently all of the tests fail
            # traced = torch.jit.trace(torch_fn, (ins, ins))
            # x = warmup_and_run_forward(traced, rand_a, rand_b)
            # y = torch_fn(nans, rand_b)
            # try:
            #     np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy())
            #     print("Succeeded on dev=", dev, "function=", torch_fn)
            # except AssertionError:
            #     # Print extra info before exiting:
            #     print("Failed on dev=", dev, "function=", torch_fn)
            #     # np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy())


    def test_round_2(self):
        # [FUNCTION DOC] For float32 and double, traces a round op on a small tensor, checks full
        # fusion, then compares the traced output with the eager round applied to that output.
        # KNOWN ISSUE: the reference rounds the already rounded result instead of the original
        # input, so the check cannot detect a wrong rounding mode.
        def round(x):
            # [FUNCTION DOC] HELPER: Returns the round of x.
            return torch.round(x)

        for data_type in [torch.float32, torch.double]:
            a = torch.tensor([0.2, 1.6, 2.5, 3.5]).to(data_type)
            traced = torch.jit.trace(round, (a))
            x = warmup_and_run_forward(traced, a)
            self.assertLastGraphAllFused()
            y = round(x)
            self.assertEqual(x, y)

    def test_rand_like(self):
        # [FUNCTION DOC] For each device, traces random-like generation shaped as the sum of two
        # inputs (no trace check) and runs it for each dtype asserting full fusion. Then checks
        # that the first three moments of the input sample of 65536 values are close to the moments
        # of a uniform distribution (one half, one third, one quarter) within a 2 percent relative
        # tolerance.
        # KNOWN ISSUE: the moments are computed from the input tensor, not from the generated
        # output, so the statistics check does not test the random op.
        N = 1 << 16

        def run_rand_like(x, y):
            # [FUNCTION DOC] HELPER: Returns a random tensor shaped like the sum of x and y.
            return torch.rand_like(torch.add(x, y))

        for device in self.devices:
            x = torch.rand(N, device=device)
            traced = torch.jit.trace(run_rand_like, (x, x), check_trace=False)

            for data_type in self.dtypes:
                _x = x.to(dtype=data_type)
                x_v = warmup_and_run_forward(traced, _x, _x)
                self.assertLastGraphAllFused()

            x_np = x.cpu().numpy()
            x1_mean = np.mean(x_np)
            x2_mean = np.mean(x_np ** 2)
            x3_mean = np.mean(x_np ** 3)
            np.testing.assert_allclose(x1_mean, 1. / 2, rtol=2e-2)
            np.testing.assert_allclose(x2_mean, 1. / 3, rtol=2e-2)
            np.testing.assert_allclose(x3_mean, 1. / 4, rtol=2e-2)

    def test_nans(self):
        # [FUNCTION DOC] Traces elementwise max and min of doubled inputs and checks that NaN
        # propagates when either argument is NaN, with full fusion asserted after the min and max
        # checks.
        # KNOWN ISSUE: the dtype loop only assigns the inputs; the checks run after the loop and
        # therefore only use the last dtype in the list.
        def test_max(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise maximum of twice x and twice y.
            return torch.max(2 * x, 2 * y)

        def test_min(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise minimum of twice x and twice y.
            return torch.min(2 * x, 2 * y)

        tmax = torch.jit.trace(test_max, (torch.rand(1), torch.rand(1)))
        tmin = torch.jit.trace(test_min, (torch.rand(1), torch.rand(1)))

        for data_type in self.dtypes:
            x = torch.tensor([np.nan]).to(dtype=data_type)
            y = torch.tensor([1.0]).to(dtype=data_type)

        if not np.isnan(warmup_and_run_forward(tmin, x, y).float().item()):
            raise AssertionError("expected nan for tmin(x, y)")
        if not np.isnan(warmup_and_run_forward(tmin, y, x).float().item()):
            raise AssertionError("expected nan for tmin(y, x)")
        self.assertLastGraphAllFused()
        if not np.isnan(warmup_and_run_forward(tmax, x, y).float().item()):
            raise AssertionError("expected nan for tmax(x, y)")
        if not np.isnan(warmup_and_run_forward(tmax, y, x).float().item()):
            raise AssertionError("expected nan for tmax(y, x)")
        self.assertLastGraphAllFused()

    def test_double_intrinsics(self):
        # [FUNCTION DOC] For each device, traces raising a double tensor to the 7th power and
        # asserts full fusion; the numeric result is not checked.
        def do_pow(x):
            # [FUNCTION DOC] HELPER: Returns x raised to the power seven.
            return torch.pow(x, 7)

        for device in self.devices:
            x = torch.rand(10, dtype=torch.double, device=device)
            traced = torch.jit.trace(do_pow, (x))
            x = warmup_and_run_forward(traced, x)
            self.assertLastGraphAllFused()

    def test_remainder(self):
        # [FUNCTION DOC] For each dtype, traces the remainder of the sum of two inputs by the first
        # input and compares with eager for random data, a zero numerator, and a NaN numerator
        # (bfloat16 gets looser tolerance for the random case). Asserts full fusion each time.
        # KNOWN ISSUE: the NaN source is built as a zero-dimension array from the integer 1024, not
        # a 1024-element array, so the NaN case and the zero-valued numerator may not match the
        # intent.
        def run_remainder(x, y):
            # [FUNCTION DOC] HELPER: Returns the remainder of the sum of x and y divided by x.
            c = torch.remainder(torch.add(x, y), x)
            return c

        for data_type in self.dtypes:
            a = torch.rand(1024, dtype=data_type)
            b = torch.rand(1024, dtype=data_type)
            zeros = torch.zeros(1024, dtype=data_type)
            cc = np.array(1024, dtype=float)
            cc.fill(np.nan)
            nans = torch.from_numpy(cc).to(dtype=data_type)

            # random floats
            zeros1 = torch.zeros(1024, dtype=data_type)
            zeros2 = torch.zeros(1024, dtype=data_type)

            traced = torch.jit.trace(run_remainder, (zeros1, zeros2))
            x = warmup_and_run_forward(traced, a, b)
            self.assertLastGraphAllFused()
            y = run_remainder(a, b)
            if data_type is torch.bfloat16:
                self.assertEqual(x, y, atol=4e-3, rtol=2e-3)
            else:
                self.assertEqual(x, y)

            # div by 0
            traced = torch.jit.trace(run_remainder, (zeros1, zeros2))
            x = warmup_and_run_forward(traced, zeros, a)
            self.assertLastGraphAllFused()
            y = run_remainder(zeros, a)
            self.assertEqual(x, y)

            # numerators and denominatos are nan
            traced = torch.jit.trace(run_remainder, (zeros1, zeros2))
            x = warmup_and_run_forward(traced, nans, a)
            self.assertLastGraphAllFused()
            y = run_remainder(nans, a)
            self.assertEqual(x, y)

    def test_multioutput(self):
        # [FUNCTION DOC] Traces a function with two outputs (x plus one and its doubling), checks
        # full fusion and compares both outputs with numpy.
        def easy(x):
            # [FUNCTION DOC] HELPER: Returns a pair holding x plus one and that value added to
            # itself.
            b = x + 1
            c = b + b
            return (b, c)

        traced = torch.jit.trace(easy, (torch.zeros(1024)))

        a = torch.zeros(1024)
        b, c = warmup_and_run_forward(traced, a)
        self.assertLastGraphAllFused()
        bp = a.numpy() + 1
        cp = bp + bp
        np.testing.assert_allclose(b.numpy(), bp)
        np.testing.assert_allclose(c.numpy(), cp)

    def test_chunk(self):
        # [FUNCTION DOC] For each dtype, traces adding one to a tensor then splitting it into two
        # chunks and adding them. Traced with a 1024x1024 input and run on a 32x32 input; asserts
        # full fusion and compares with numpy (cast to float for the comparison).
        def easy(x):
            # [FUNCTION DOC] HELPER: Adds one to x, splits it into two chunks along the first
            # dimension and returns the sum of the chunks.
            y = x + 1
            aaa, bbb = torch.chunk(y, 2)
            return aaa + bbb

        for data_type in self.dtypes:
            trace_input = torch.zeros(1024, 1024, dtype=data_type)
            traced = torch.jit.trace(easy, (trace_input))

            a = torch.zeros(32, 32, dtype=data_type)
            x = warmup_and_run_forward(traced, a)
            self.assertLastGraphAllFused()
            npr = a.float().numpy()
            npr2 = npr + 1
            npr_a, npr_b = np.array_split(npr2, 2)
            np.testing.assert_allclose(npr_a + npr_b, x.float().numpy())

    def test_cat(self):
        # [FUNCTION DOC] For each device, traces a concatenation of three tensors with offsets
        # added (widths 128, 16 and 1) along dimension 1 for each dtype and checks full fusion plus
        # agreement with eager. Then concatenates ten channels-last 4D tensors along each of the
        # four dimensions and checks full fusion and equality.
        for device in self.devices:
            _dim = 1

            def foo(*args):
                # [FUNCTION DOC] HELPER: Adds each argument's position to it, concatenates the
                # results along the dimension held in the enclosing variable, which the test
                # rebinds between runs, and returns the square of the result.
                args_2 = [v + i for i, v in enumerate(args)]
                v = torch.cat(args_2, dim=_dim)
                return v * v

            for data_type in self.dtypes:
                M = 16
                Ns = [128, 16, 1]
                values = [torch.zeros(M, N, dtype=data_type, device=device) for N in Ns]
                traced = torch.jit.trace(foo, values)

                x = warmup_and_run_forward(traced, *values)
                self.assertLastGraphAllFused()
                ref = foo(*values)
                np.testing.assert_allclose(ref.cpu().float().numpy(), x.cpu().float().numpy())

            # Test channels-last
            for _cur_dim in range(4):
                _dim = _cur_dim
                values = [torch.randn((2, 3, 4, 5), device=device).to(memory_format=torch.channels_last) for _ in range(10)]
                traced = torch.jit.trace(foo, values)

                x = warmup_and_run_forward(traced, *values)
                self.assertLastGraphAllFused()
                ref = foo(*values)
                self.assertEqual(ref, x)

    # This test checks that we correctly handle fusion group with just aten::cat in it.
    # Note that the test only makes sense with min_fusion_group=1, otherwise no
    # fusion groups would be formed at all.
    # TODO: Fix and re-enable the test.
    @unittest.skip("cat is broken with fusion group inlining disabled")
    def test_cat_only(self):
        # [FUNCTION DOC] Skipped test. Would trace a concatenation of three tensors that forms a
        # fusion group with only the concatenation, asserting full fusion and eager agreement.
        # Needs a minimum fusion group size of one to be meaningful (see the comment above it).
        for device in self.devices:
            def foo(*args):
                # [FUNCTION DOC] HELPER: Adds each argument's position to it and concatenates the
                # results along dimension 1.
                args_2 = [v + i for i, v in enumerate(args)]
                v = torch.cat(args_2, dim=1)
                return v

            M = 16
            Ns = [128, 16, 1]
            values = [torch.zeros(M, N, device=device) for N in Ns]
            traced = torch.jit.trace(foo, values)

            x = warmup_and_run_forward(traced, *values)
            self.assertLastGraphAllFused()
            ref = foo(*values)
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_cat_negative_dim(self):
        # [FUNCTION DOC] For each device, traces concatenation of three tensors along the last
        # dimension (negative index) followed by squaring; asserts full fusion and eager agreement.
        for device in self.devices:
            def foo(*args):
                # [FUNCTION DOC] HELPER: Concatenates its arguments along the last dimension and
                # returns the square of the result.
                v = torch.cat(args, dim=-1)
                return v * v

            M = 16
            Ns = [128, 16, 1]
            values = [torch.randn(M, N, device=device) for N in Ns]
            traced = torch.jit.trace(foo, values)

            x = warmup_and_run_forward(traced, *values)
            self.assertLastGraphAllFused()
            ref = foo(*values)
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_cat_promote_inputs(self):
        # [FUNCTION DOC] For each device, traces concatenation of half, float and double
        # tensors along dimension 1 and squaring; asserts full fusion and eager agreement (type
        # promotion).
        for device in self.devices:
            def foo(*args):
                # [FUNCTION DOC] HELPER: Concatenates its arguments along dimension 1 and
                # returns the square of the result.
                v = torch.cat(args, dim=1)
                return v * v

            M = 16
            Ns = [128, 16, 1]
            dtypes = [torch.half, torch.float32, torch.double]
            values = [torch.randn(M, N, device=device, dtype=dt) for N, dt in zip(Ns, dtypes)]
            traced = torch.jit.trace(foo, values)

            x = warmup_and_run_forward(traced, *values)
            self.assertLastGraphAllFused()
            ref = foo(*values)
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_cat_empty_tensors(self):
        # [FUNCTION DOC] For each device, traces concatenation including an empty double tensor
        # with three real inputs, then concatenation of only empty tensors; asserts full fusion
        # and eager agreement in both cases.
        for device in self.devices:
            def foo(*args):
                # [FUNCTION DOC] HELPER: Concatenates its arguments along dimension 1 and
                # returns the square of the result.
                v = torch.cat(args, dim=1)
                return v * v

            M = 16
            Ns = [128, 16, 1]
            empty = torch.tensor([], device=device, dtype=torch.double)
            values = [empty] + [torch.randn(M, N, device=device) for N in Ns]
            traced = torch.jit.trace(foo, values)

            x = warmup_and_run_forward(traced, *values)
            self.assertLastGraphAllFused()
            ref = foo(*values)
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

            # now test with only empty tensors
            values = [empty for i in range(3)]
            traced = torch.jit.trace(foo, values)
            x = warmup_and_run_forward(traced, *values)
            self.assertLastGraphAllFused()
            ref = foo(*values)
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_cat_with_constant_dim(self):
        # [FUNCTION DOC] For each device, traces a concatenation of an empty tensor and two 1x64
        # tensors followed by a second single-input concatenation and squaring; asserts full fusion
        # and eager agreement.
        for device in self.devices:
            def foo(*args):
                # [FUNCTION DOC] HELPER: Concatenates its arguments along dimension 1, concatenates
                # that single result again, and returns its square.
                v1 = torch.cat(args, dim=1)
                v2 = torch.cat([v1], dim=1)
                return v2 * v2

            empty = torch.tensor([], device=device, dtype=torch.float32)
            inputs = [empty] + [torch.randn(1, 64, device=device), torch.randn(1, 64, device=device)]
            traced = torch.jit.trace(foo, inputs)

            x = warmup_and_run_forward(traced, *inputs)
            self.assertLastGraphAllFused()
            ref = foo(*inputs)
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_scalar(self):
        # [FUNCTION DOC] Scripts two functions chaining two adds with scalar alpha values, one with
        # float scalars and one with int scalars. For each and each dtype, runs the function twice
        # and checks the second result equals x plus y times a plus z times b with a one and b two.
        @torch.jit.script
        def test_float(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, a: float, b: float) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Scripted function adding y with float alpha a to x, then z
            # with float alpha b.
            return torch.add(torch.add(x, y, alpha=a), z, alpha=b)

        @torch.jit.script
        def test_int(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, a: int, b: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Scripted function adding y with int alpha a to x, then z with
            # int alpha b.
            return torch.add(torch.add(x, y, alpha=a), z, alpha=b)

        for test in (test_float, test_int):
            for data_type in self.dtypes:
                x, y, z = (torch.rand(4, dtype=data_type) for i in range(3))
                a, b = 1, 2
                test(x, y, z, a, b)
                r = test(x, y, z, a, b)
                self.assertEqual(r, x + y * a + z * b)

    def test_loop(self):
        # [FUNCTION DOC] Scripts a loop function and runs it twice on 32x32 inputs with four
        # iterations. Only checks that it executes; nothing is asserted.
        @torch.jit.script
        def test(x: torch.Tensor, y: torch.Tensor, z: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Scripted function that loops z times, each time computing an
            # unused sum of x and y and accumulating y into a running value starting at y, and
            # returns that value.
            b = y
            for _ in range(z):
                a = x + y
                b = b + y
            return b

        x, y, z = (torch.zeros(32, 32), torch.ones(32, 32), 4)
        test(x, y, z)
        r = test(x, y, z)

    def test_slice(self):
        # [FUNCTION DOC] Traces a function that slices both inputs with step 2 over the first 512
        # rows and adds them; runs it without warmup and compares with the eager sliced sum.
        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of strided slices of x and y (first 512 rows,
            # step 2).
            a = x[0:512:2]
            b = y[0:512:2]
            return a + b

        traced = torch.jit.trace(easy, (torch.ones(1024, 1024), torch.zeros(1024, 1024)))

        a = torch.ones(1024, 1024)
        x = traced(a, a)
        npr = a[0:512:2]
        npr = npr + npr
        np.testing.assert_allclose(npr.numpy(), x.numpy())

    def test_unsqueeze(self, N=256):
        # [FUNCTION DOC] Traces adding two tensors each unsqueezed at dimension 0, run on an N by N
        # random input (N defaults to 256) and compared with numpy.
        def easy(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y after inserting a leading dimension
            # in each.
            a = torch.unsqueeze(x, 0)
            b = torch.unsqueeze(y, 0)
            return a + b

        traced = torch.jit.trace(easy, (torch.ones(N, N), torch.zeros(N, N)))

        a = torch.rand(N, N)
        x = traced(a, a)
        npr = np.expand_dims(a, 0)
        npr = npr + npr
        np.testing.assert_allclose(npr, x.numpy())

    def _test_softmax(self, device):
        # [FUNCTION DOC] HELPER: Shared body of the softmax tests for one device. For softmax,
        # softmax with negative dims and log-softmax helpers, and each dtype, enables reductions in
        # the fuser (saving the old setting), traces on a 2x3 input, runs on another random input
        # and compares with eager within 1e-6, then restores the saved setting. No fusion
        # assertion.
        def test_softmax(x, y):
            # [FUNCTION DOC] HELPER: Sums softmax of x and y along dimensions 0 and 1 in float32.
            a = F.softmax(x, dim=0, dtype=torch.float32)
            b = F.softmax(y, dim=0, dtype=torch.float32)
            c = F.softmax(x, dim=1, dtype=torch.float32)
            d = F.softmax(y, dim=1, dtype=torch.float32)
            return a + b + c + d

        def test_softmax_neg_index(x, y):
            # [FUNCTION DOC] HELPER: Sums softmax of x and y along dimensions -2 and -1 in float32.
            a = F.softmax(x, dim=-2, dtype=torch.float32)
            b = F.softmax(y, dim=-2, dtype=torch.float32)
            c = F.softmax(x, dim=-1, dtype=torch.float32)
            d = F.softmax(y, dim=-1, dtype=torch.float32)
            return a + b + c + d

        def test_log_softmax(x, y):
            # [FUNCTION DOC] HELPER: Sums log-softmax of x and y along dimensions 0 and 1 in
            # float32.
            a = F.log_softmax(x, dim=0, dtype=torch.float32)
            b = F.log_softmax(y, dim=0, dtype=torch.float32)
            c = F.log_softmax(x, dim=1, dtype=torch.float32)
            d = F.log_softmax(y, dim=1, dtype=torch.float32)
            return a + b + c + d

        for test in (test_softmax, test_log_softmax, test_softmax_neg_index):
            for data_type in self.dtypes:
                old = torch._C._jit_set_texpr_reductions_enabled(True)
                traced_input = torch.randn(2, 3, dtype=data_type, device=device)
                traced = torch.jit.trace(test, (traced_input, traced_input))
                inp = torch.randn(2, 3, dtype=data_type, device=device)
                res = traced(inp, inp)
                # Use eager mode as reference.
                ref = test(inp, inp)
                np.testing.assert_allclose(ref, res.cpu().numpy(), rtol=1e-06, atol=1e-06)
                torch._C._jit_set_texpr_reductions_enabled(old)

    def test_softmax_cpu(self):
        # [FUNCTION DOC] Runs the softmax tests on the CPU.
        self._test_softmax('cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @unittest.skip("global allocs are not supported yet.")
    def test_softmax_cuda(self):
        # [FUNCTION DOC] Skipped (and needs CUDA): would run the softmax tests on CUDA, but global
        # allocations are not yet supported.
        self._test_softmax('cuda')

    def test_half_gelu(self):
        # [FUNCTION DOC] Only does work if CUDA is available. Traces a scripted bias-gelu on half
        # tensors and asserts full fusion; nothing else is checked.
        devices = ["cuda"] if torch.cuda.is_available() else []

        @torch.jit.script
        def bias_gelu(bias, y):
            # [FUNCTION DOC] HELPER: Scripted function adding bias and y, then applying an erf
            # based gelu approximation.
            x = bias + y
            return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

        for device in devices:
            a = torch.rand(1024, dtype=torch.half, device=device)
            b = torch.rand(1024, dtype=torch.half, device=device)
            traced = torch.jit.trace(bias_gelu, (a, b))
            x = warmup_and_run_forward(traced, a, b)
            self.assertLastGraphAllFused()

    def test_half_bn_relu(self):
        # [FUNCTION DOC] Only does work if CUDA is available. Traces batch norm followed by relu on
        # half tensors, prints the traced graph, and asserts full fusion.
        devices = ["cuda"] if torch.cuda.is_available() else []

        def foo(a, b, c):
            # [FUNCTION DOC] HELPER: Applies batch norm with a, b, c as input, running mean and
            # running variance, then relu.
            y = torch.nn.functional.batch_norm(a, b, c)
            z = y.relu()
            return z

        for device in devices:
            a = torch.rand(16, 16, dtype=torch.half, device=device)
            b = torch.rand(16, dtype=torch.half, device=device)
            c = torch.rand(16, dtype=torch.half, device=device)
            traced = torch.jit.trace(foo, (a, b, c))
            print(traced.graph)
            x = warmup_and_run_forward(traced, a, b, c)
            self.assertLastGraphAllFused()

    def test_exp_pow(self):
        # [FUNCTION DOC] For each device, traces a scripted function mixing products and a squared
        # operand on double tensors and asserts full fusion; the values are not checked.
        @torch.jit.script
        def do_exp(x, y, z):
            # [FUNCTION DOC] HELPER: Scripted function returning x times y times two times z
            # squared.
            return ((x * y) * 2) * torch.pow(z, 2)

        for device in self.devices:
            x = torch.rand(10, dtype=torch.double, device=device)
            y = torch.rand(10, dtype=torch.double, device=device)
            z = torch.rand(10, dtype=torch.double, device=device)
            traced = torch.jit.trace(do_exp, (x, y, z))
            x = warmup_and_run_forward(traced, x, y, z)
            self.assertLastGraphAllFused()

    def test_sin_pow(self):
        # [FUNCTION DOC] For each dtype and shape (3, 5 or 10 elements), scripts the helper, warms
        # it up, asserts full fusion and compares with eager.
        def test(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x raised to the power zero.
            return torch.sin(torch.pow(x, 0))

        for data_type, shape in itertools.product(self.dtypes, [[3], [5], [10]]):
            x = torch.rand(shape, dtype=data_type)
            scripted = torch.jit.script(test)
            out = warmup_and_run_forward(scripted, x)
            self.assertLastGraphAllFused()
            self.assertEqual(out, test(x))

    def test_transpose(self):
        # [FUNCTION DOC] Scripts a function adding a transposed 4D tensor to two others and checks
        # two consecutive runs give the same result.
        @torch.jit.script
        def test(x, y, z):
            # [FUNCTION DOC] HELPER: Scripted function returning x with dimensions 0 and 1 swapped
            # plus y plus z.
            return x.transpose(0, 1) + y + z
        x = torch.rand(4, 5, 2, 3)
        y = torch.rand(5, 4, 2, 3)
        z = torch.rand(5, 4, 2, 3)
        ref = test(x, y, z)
        res = test(x, y, z)
        np.testing.assert_allclose(ref.numpy(), res.numpy())

    def test_sliced_stride(self):
        # [FUNCTION DOC] Scripts a three-input add where the first input is a strided slice (every
        # second row) of a larger tensor, and checks two consecutive runs give the same result.
        @torch.jit.script
        def test(x, y, z):
            # [FUNCTION DOC] HELPER: Scripted function returning the sum of three inputs.
            return x + y + z
        x = torch.rand(16, 4, 2, 3)[::2]
        y = torch.rand(8, 4, 2, 3)
        z = torch.rand(8, 4, 2, 3)
        ref = test(x, y, z)
        res = test(x, y, z)
        np.testing.assert_allclose(ref.numpy(), res.numpy())

    @unittest.skip("dynamic shapes are not quite there yet")
    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_dynamic_shape(self):
        # [FUNCTION DOC] Skipped test (needs CUDA). With two profiled runs, would script a product
        # of three tensors and exercise changed sizes, broadcasting, and mismatched shapes that
        # must raise a size-mismatch error before reaching codegen.
        with num_profiled_runs(2):
            @torch.jit.script
            def test(x, y, z):
                # [FUNCTION DOC] HELPER: Scripted function returning the product of three inputs.
                return x * y * z
            x, y, z = (torch.rand(4, 8).cuda() for _ in range(3))
            ref = test(x, y, z)
            _ = test(*[torch.rand(6, 8).cuda() for _ in range(3)])
            res = test(x, y, z)
            np.testing.assert_allclose(ref.cpu().numpy(), res.cpu().numpy())

            # A wild broadcast appears.
            x = torch.rand(4, 8).cuda()
            y = torch.rand(1, 8).cuda()
            z = torch.rand(4, 1).cuda()
            res = test(x, y, z)
            xn, yn, zn = (t.cpu().numpy() for t in (x, y, z))
            np.testing.assert_allclose(res.cpu().numpy(), xn * yn * zn)

            # Mismatched shapes shouldn't reach codegen.
            x = torch.rand(4, 8).cuda()
            y = torch.rand(4, 8).cuda()
            z = torch.rand(5, 8).cuda()
            try:
                res = test(x, y, z)
            except RuntimeError as e:
                if "The size of tensor a (4) must match" not in e.args[0]:
                    raise AssertionError(f"unexpected error message: {e.args[0]}") from None

            # Changing a static dimension fails guards.
            # x, y, z = [torch.rand(4, 7).cuda() for _ in range(3)]
            # xn, yn, zn = [t.cpu().numpy() for t in (x, y, z)]
            # res = test(x, y, z)
            # print(test.graph_for(x, y, z))
            # np.testing.assert_allclose(res.cpu().numpy(), xn * yn * zn)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_guard_fails(self):
        # [FUNCTION DOC] Requires CUDA. Scripts a three-input product and calls it three times on
        # length-4 CUDA tensors and then once on length-7 tensors so the shape guard fails. Only
        # checks that none of the calls raises; nothing is asserted on results.
        @torch.jit.script
        def test(x, y, z):
            # [FUNCTION DOC] HELPER: Scripted function returning the product of three inputs.
            return x * y * z
        r1 = test(*[torch.rand(4).cuda() for _ in range(3)])
        r2 = test(*[torch.rand(4).cuda() for _ in range(3)])
        r3 = test(*[torch.rand(4).cuda() for _ in range(3)])
        r4 = test(*[torch.rand(7).cuda() for _ in range(3)])

    def test_bitwise_ops(self):
        # [FUNCTION DOC] For each device and each of five bitwise helpers (and, or, xor, left
        # shift, right shift), traces on int32 ones, runs on ones and zeros, asserts full fusion
        # and compares with the eager result.
        def run_and(x, y):
            # [FUNCTION DOC] HELPER: Returns x bitwise-and the bitwise-and of x and y.
            return x & (x & y)

        def run_or(x, y):
            # [FUNCTION DOC] HELPER: Returns x bitwise-and the bitwise-or of x and y.
            return x & (x | y)

        def run_xor(x, y):
            # [FUNCTION DOC] HELPER: Returns x bitwise-xor the bitwise-xor of x and y.
            return x ^ (x ^ y)

        def run_lshift(x, y):
            # [FUNCTION DOC] HELPER: Returns x bitwise-and the result of shifting x left by y.
            return x & (x << y)

        def run_rshift(x, y):
            # [FUNCTION DOC] HELPER: Returns x bitwise-and the result of shifting x right by y.
            return x & (x >> y)

        fns = {run_and, run_or, run_xor, run_lshift, run_rshift}

        for device in self.devices:
            for fn in fns:
                a = torch.ones(128, dtype=torch.int32, device=device)
                b = torch.zeros(128, dtype=torch.int32, device=device)
                inp = torch.ones(128, dtype=torch.int32, device=device)
                traced = torch.jit.trace(fn, (inp, inp))
                x = warmup_and_run_forward(traced, a, b)
                self.assertLastGraphAllFused()
                y = fn(a, b)
                np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy())

    def test_where(self):
        # [FUNCTION DOC] For each dtype, traces a where selecting the larger of two inputs, asserts
        # full fusion and compares with the eager result after casting to float.
        def run_where(x, y):
            # [FUNCTION DOC] HELPER: Returns x where x is greater than y, else y.
            return torch.where(torch.gt(x, y), x, y)

        for data_type in self.dtypes:
            a = torch.rand(1024, dtype=data_type)
            b = torch.rand(1024, dtype=data_type)
            zeros = torch.zeros(1024, dtype=data_type)
            traced = torch.jit.trace(run_where, (zeros, zeros))
            x = warmup_and_run_forward(traced, a, b)
            self.assertLastGraphAllFused()
            y = run_where(a, b)
            np.testing.assert_allclose(x.float().numpy(), y.float().numpy())

    def test_multi_rand(self):
        # [FUNCTION DOC] For each device and dtype, scripts a function that draws a random
        # tensor and combines it so the random part cancels; asserts full fusion and that the
        # output is close to twice the input (looser tolerance once bfloat16 is reached).
        # Raises AssertionError on mismatch.
        for device in self.devices:
            def test(x):
                # [FUNCTION DOC] HELPER: Draws a random tensor y like x and returns the sum of
                # x and y minus the difference of y and x, which equals twice x.
                y = torch.rand_like(x)
                return (x + y) - (y - x)

            _atol = 2e-3
            _rtol = 1e-5
            for data_type in self.dtypes:
                if data_type is torch.bfloat16:
                    _atol = 2e-2
                a = torch.rand(4, dtype=data_type, device=device)
                scripted = torch.jit.script(test)
                out = warmup_and_run_forward(scripted, a)
                self.assertLastGraphAllFused()
                if not torch.allclose(out, 2 * a, atol=_atol, rtol=_rtol):
                    raise AssertionError("output does not match expected")

    def test_mask(self):
        # [FUNCTION DOC] For each device and dtype, scripts a function comparing an unsqueezed
        # boolean mask to zero, asserts full fusion and exact equality with eager. Raises
        # AssertionError on mismatch.
        def test(x):
            # [FUNCTION DOC] HELPER: Returns whether x with a new dimension inserted at position 1
            # is equal to zero.
            return x.unsqueeze(1) == 0

        for d in self.devices:
            for data_type in self.dtypes:
                x = torch.rand(4, dtype=data_type, device=d) > 0.5
                scripted = torch.jit.script(test)
                out = warmup_and_run_forward(scripted, x)
                self.assertLastGraphAllFused()
                if not torch.equal(out, test(x)):
                    raise AssertionError("output does not match expected")

    def test_simple_add(self):
        # [FUNCTION DOC] Temporarily enables generating block code and allowing fallback in the
        # fuser (saving the old values), traces an add of two 256x256 ones tensors, checks the
        # result is all twos, then restores the two global settings. The restore is skipped if an
        # earlier step raises.
        val = torch._C._jit_get_te_generate_block_code()
        torch._C._jit_set_te_generate_block_code(True)
        fall_bk = torch._C._jit_texpr_fallback_allowed()
        torch._C._jit_texpr_set_fallback_allowed(True)

        def simple(a, b):
            # [FUNCTION DOC] HELPER: Returns the sum of a and b.
            return torch.add(a, b)

        a = torch.ones(256, 256)
        b = torch.ones(256, 256)
        traced = torch.jit.trace(simple,
                                 (torch.ones(256, 256), torch.ones(256, 256)))
        f = traced(a, b)
        f_test = np.full((256, 256), 2, dtype=float)
        np.testing.assert_allclose(f.numpy(), f_test)
        torch._C._jit_set_te_generate_block_code(val)
        torch._C._jit_texpr_set_fallback_allowed(fall_bk)

    def test_strided_output_preserved(self):
        # [FUNCTION DOC] Scripts a function and checks that the output keeps the stride of eager
        # execution. First uses a custom-strided 2x3 integer tensor, then a channels-last 4D float
        # tensor; each time runs the script three times, compares values and strides with eager and
        # asserts full fusion.
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Returns a plus b minus a.
            return a + b - a

        # smaller, easier to debug example
        x = torch.arange(6)
        x = torch.as_strided(x, (2, 3), (1, 2))
        total = 0
        for i in range(2):
            for j in range(3):
                x[i, j] = total
                total += 1
        foo_script = torch.jit.script(foo)
        foo_script(x, x)
        foo_script(x, x)
        out_s = foo_script(x, x)
        out_eager = foo(x, x)
        self.assertEqual(out_s, out_eager)
        self.assertEqual(out_s.stride(), out_eager.stride())
        self.assertLastGraphAllFused()

        # more dims
        N, C, H, W, = 2, 3, 4, 5
        x = torch.rand(N, C, H, W).to(memory_format=torch.channels_last)
        foo_script = torch.jit.script(foo)
        foo_script(x, x)
        foo_script(x, x)
        out_s = foo_script(x, x)
        out_eager = foo(x, x)
        self.assertEqual(out_s, out_eager)
        self.assertEqual(out_s.stride(), out_eager.stride())
        self.assertLastGraphAllFused()

    def test_alias_analysis_module(self):
        # [FUNCTION DOC] Scripts and runs a module that mutates one of its attributes in place and
        # reads another. Checks scripted equals eager first, then makes the read attribute the same
        # object as the mutated one in both modules and checks equality again (alias analysis for
        # module state).
        class AliasModule(nn.Module):
            # [CLASS DOC] HELPER: Module with three random 128x128 attributes (seeded) whose
            # forward adds the first attribute to z, adds y in place into the second attribute,
            # adds the first again and then x, returning the result.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Seeds the generator and creates three random 128x128
                # tensor attributes a, b and c.
                super().__init__()
                torch.manual_seed(1337)
                self.a = torch.randn(128, 128)
                self.b = torch.randn(128, 128)
                self.c = torch.randn(128, 128)

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Computes z plus a, mutates b in place with y, then returns
                # that sum plus a plus x.
                z = z + self.a
                self.b.add_(y)
                w = z + self.a
                z = w + x
                return z
        x = torch.randn(128, 128)

        def getModule(script):
            # [FUNCTION DOC] HELPER: Creates the alias module and returns it scripted if the flag
            # is true, else eager.
            am = AliasModule()
            if script:
                return torch.jit.script(am)
            return am

        am = getModule(False)
        am_s = getModule(True)
        ref = am(x, x, x)
        test = am_s(x, x, x)
        torch.testing.assert_close(ref, test)

        # Now do the aliasing
        am.a = am.b
        ref = am(x, x, x)

        am_s.a = am_s.b
        test = am_s(x, x, x)

        torch.testing.assert_close(ref, test)

    def test_alias_analysis_inputs(self):
        # [FUNCTION DOC] Checks a scripted module against eager when forward mutates an input in
        # place and the same tensor is passed for all three arguments. Inputs are regenerated from
        # the same seed for both runs and results compared.
        class AliasModule(nn.Module):
            # [CLASS DOC] HELPER: Module with three seeded random attributes whose forward adds y
            # into x in place, then returns z plus attribute a plus x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Seeds the generator and creates three random 128x128
                # tensor attributes a, b and c.
                super().__init__()
                torch.manual_seed(1337)
                self.a = torch.randn(128, 128)
                self.b = torch.randn(128, 128)
                self.c = torch.randn(128, 128)

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Adds y into x in place, then returns z plus a plus x.
                x.add_(y)
                w = z + self.a
                z = w + x
                return z

        def getModule(script):
            # [FUNCTION DOC] HELPER: Creates the alias module and returns it scripted if the flag
            # is true, else eager.
            am = AliasModule()
            if script:
                return torch.jit.script(am)
            return am
        am = getModule(False)
        am_s = getModule(True)

        torch.manual_seed(1337)
        x = torch.randn(128, 128)
        ref = am(x, x, x)

        torch.manual_seed(1337)
        x = torch.randn(128, 128)
        test = am_s(x, x, x)

        torch.testing.assert_close(ref, test)

    def test_alias_analysis_input_and_module(self):
        # [FUNCTION DOC] Like the input aliasing test, but additionally replaces the module
        # attribute b with the input tensor itself in both the eager and scripted module before
        # running, so the input and module state alias. Compares results.
        class AliasModule(nn.Module):
            # [CLASS DOC] HELPER: Module with three seeded random attributes whose forward adds y
            # into x in place, then returns z plus attribute b plus x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Seeds the generator and creates three random 128x128
                # tensor attributes a, b and c.
                super().__init__()
                torch.manual_seed(1337)
                self.a = torch.randn(128, 128)
                self.b = torch.randn(128, 128)
                self.c = torch.randn(128, 128)

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Adds y into x in place, then returns z plus b plus x.
                x.add_(y)
                w = z + self.b
                z = w + x
                return z

        def getModule(script):
            # [FUNCTION DOC] HELPER: Creates the alias module and returns it scripted if the flag
            # is true, else eager.
            am = AliasModule()
            if script:
                return torch.jit.script(am)
            return am
        am = getModule(False)
        am_s = getModule(True)

        torch.manual_seed(1337)
        x = torch.randn(128, 128)
        am.b = x
        ref = am(x, x, x)

        torch.manual_seed(1337)
        x = torch.randn(128, 128)
        am_s.b = x
        test = am_s(x, x, x)

        torch.testing.assert_close(ref, test)

    def test_multiple_outputs(self):
        # [FUNCTION DOC] For each device and dtype, traces a three-output function whose second
        # input is a strided view (stride 29 over a longer buffer) and whose third input is int64,
        # runs the trace twice and compares all outputs with eager. Regression for a bug cited in
        # the comment.
        for device in self.devices:
            # A bug reported internally similar to the one reported in #48533
            def foo(a, b, c):
                # [FUNCTION DOC] HELPER: Computes c plus one, its product with b, and a times that
                # value unsqueezed at dimension 1; returns the three results.
                t_next = c + 1
                t5 = t_next * b
                t6 = torch.unsqueeze(t_next, 1)
                t7 = a * t6
                return (t7, t5, t_next)

            for data_type in self.dtypes:
                a = torch.rand(20, 20, dtype=data_type, device=device)
                b = torch.rand(20 * 29, dtype=data_type, device=device).as_strided([20], [29])
                c = torch.ones(20, dtype=torch.int64, device=device)
                traced = torch.jit.trace(foo, (a, b, c))
                ref = foo(a, b, c)
                exp = traced(a, b, c)
                exp = traced(a, b, c)
                self.assertEqual(ref, exp)

    def test_propagated_mem_layout(self):
        # [FUNCTION DOC] Checks that memory layout (contiguous or channels-last, optionally
        # permuted) of inputs is preserved through traced functions with one or several outputs,
        # for several shapes and for the STATIC and DYNAMIC fusion strategies. Restores the old
        # fusion strategy after each strategy.
        # KNOWN ISSUE: the configuration product is a one-shot iterator created before the strategy
        # loop, so it is exhausted during the STATIC pass and the DYNAMIC pass runs no cases.
        def foo(a, b, c):
            # [FUNCTION DOC] HELPER: Returns a times (b times (c plus one)).
            t_next = c + 1
            t5 = t_next * b
            t7 = a * t5
            return t7

        def foo_multi_outputs(a, b, c):
            # [FUNCTION DOC] HELPER: Returns a tuple of the product a times (b times (c plus one)),
            # the inner product b times (c plus one), and c plus one.
            t_next = c + 1
            t5 = b * t_next
            t7 = a * t5
            return (t7, t5, t_next)

        def foo_multi_outputs_i_nhwc_o_nchw(a, b, c):
            # [FUNCTION DOC] HELPER: Like the multi output helper but first returns a
            # contiguous-format copy of the final product, followed by the three values of that
            # helper.
            t_next = c + 1
            t5 = b * t_next
            t7 = a * t5
            t8 = t7.to(memory_format=torch.contiguous_format)
            return (t8, t7, t5, t_next)

        def run_foo_case(foo, a, b, c):
            # [FUNCTION DOC] HELPER: Traces the given function on a, b, c, runs the trace twice and
            # asserts the second output equals the eager output.
            traced_contiguous = torch.jit.trace(foo, (a, b, c))
            ref = foo(a, b, c)
            exp = traced_contiguous(a, b, c)
            exp = traced_contiguous(a, b, c)
            self.assertEqual(ref, exp)

        mem_layouts = list(itertools.product([torch.contiguous_format, torch.channels_last], repeat=3))
        shapes = [(2, 3, 4, 5), (2, 1, 1, 5), (1, 1, 1, 1)]
        permutes = [(0, 3, 2, 1), (0, 3, 1, 2)]
        funcs = [foo, foo_multi_outputs, foo_multi_outputs_i_nhwc_o_nchw]
        configs = itertools.product(funcs, shapes, mem_layouts, permutes)
        for strategy in ["STATIC", "DYNAMIC"]:
            old_strategy = torch.jit.set_fusion_strategy([(strategy, 10)])
            for _func, _shape, _mem_layouts, _permute in configs:
                a = torch.rand(_shape, dtype=torch.float32).to(memory_format=_mem_layouts[0])
                b = torch.rand(_shape, dtype=torch.float32).to(memory_format=_mem_layouts[1])
                c = torch.rand(_shape, dtype=torch.float32).to(memory_format=_mem_layouts[2])
                run_foo_case(_func, a, b, c)

                a = a.permute(dims=_permute)
                b = b.permute(dims=_permute)
                c = c.permute(dims=_permute)
                run_foo_case(_func, a, b, c)

            torch.jit.set_fusion_strategy(old_strategy)

if __name__ == '__main__':
    run_tests()
