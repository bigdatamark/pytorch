# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for inductor's native triton matmul lowering (the triton.native_matmul
# config), where matrix multiplies are emitted as generated dot-reduction kernels rather than
# template or library calls. Each test compiles a small function, checks numerical agreement with
# eager, and inspects the generated triton code for the number of kernels and dot operations. At
# import time, when a GPU is available, sets the default device to the GPU type; the main block
# runs the tests only with a GPU.


from collections.abc import Callable

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


aten = torch.ops.aten


@inductor_config.patch({"triton.native_matmul": True})
class TestTritonDotReduction(TestCase):
    # [CLASS DOC] Test case run with the native matmul config enabled via the class-level patch.
    # Holds two checking helpers plus tests for mm, bmm, expanded operands, fusion with pointwise
    # ops and with index-add, and grid ordering of batched kernels.
    def _check_equal(
        self, f: Callable, example_inputs: tuple[torch.Tensor], tol: float = 1e-4
    ):
        # [FUNCTION DOC] HELPER: Compiles the function, runs it and the eager version on the
        # inputs and asserts the results agree under the dynamo same comparison with the given
        # tolerance (used as both absolute and relative tolerance).
        compiled = torch.compile(f)
        actual = compiled(*example_inputs)
        expect = f(*example_inputs)
        self.assertTrue(same(expect, actual, tol=tol))

    def _check_code(
        self,
        f: Callable,
        example_inputs: tuple[torch.Tensor],
        kernel_count: int,
        dot_count: int,
    ):
        # [FUNCTION DOC] HELPER: Compiles the function and fetches the generated triton code.
        # Asserts the code launches a kernel with mm in its name, then asserts the number of jit
        # kernels equals kernel_count and the number of dot operations equals dot_count (both
        # exact counts).
        f = torch.compile(f)
        code = run_and_get_triton_code(f, *example_inputs)
        FileCheck().check_regex(r"triton.*mm.*\.run\(").run(code)

        FileCheck().check_count("@triton.jit", kernel_count, exactly=True).check_count(
            "tl.dot", dot_count, exactly=True
        ).run(code)

    def test_matmul(self):
        # [FUNCTION DOC] Plain 128 cube matmul of two row-major GPU tensors: matches eager and
        # generates one kernel with one dot.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Matrix product of the two inputs.
            z = x @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    def test_mm_1d_expand(self):
        # [FUNCTION DOC] The left operand is a vector broadcast along the inner dimension (view
        # with a new axis then expanded to M by K) before the matmul; checks equality and one
        # kernel with one dot.
        def f(x, y, M, K):
            # [FUNCTION DOC] HELPER: Expands the vector x to shape M by K and multiplies by y.
            z = x[:, None].expand(M, K) @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M,), (1,), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, M, K))
        self._check_code(f, (x, y, M, K), 1, 1)

    def test_mm_2_expand(self):
        # [FUNCTION DOC] Like the previous test but the left operand is a single element with
        # stride zero, expanded to M by K; checks equality and one kernel with one dot.
        def f(x, y, M, K):
            # [FUNCTION DOC] HELPER: Expands the one-element x to shape M by K and
            # multiplies by y.
            z = x[:, None].expand(M, K) @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((1,), (0,), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, M, K))
        self._check_code(f, (x, y, M, K), 1, 1)

    def test_matmul_fp16(self):
        # [FUNCTION DOC] Half precision left operand times a float32 right operand cast to half;
        # compares with a looser tolerance of 1e-3 since the same helper uses one value for both
        # absolute and relative tolerance, then checks one kernel and one dot.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Casts y to the dtype of x and multiplies.
            z = x @ y.to(x.dtype)
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), dtype=torch.float16, device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), dtype=torch.float32, device=GPU_TYPE)

        # _check_equal calls torch._dynamo.utils.same with kwarg tol=1e-4.
        # For fp16 dtype, torch.allclose() defaults to atol=1e-3 rtol=1e-5,
        # but same() uses the single value to assign both, resulting in
        # Accuracy failed: allclose not within tol=0.0001.
        self._check_equal(f, (x, y), tol=1e-3)
        self._check_code(f, (x, y), 1, 1)

    def test_reduction_mask_zeroout(self):
        # [FUNCTION DOC] Matmul of size 62 operands (not a block multiple) with pointwise ops on
        # both inputs; verifies the masked reduction gives correct results and one kernel with
        # one dot.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Multiplies the input x plus one by the input y minus two.
            return (x + 1) @ (y - 2)

        M, K, N = 62, 62, 62
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    def test_3mm_add(self):
        # [FUNCTION DOC] Sum of three independent matmuls; checks equality and that everything
        # fuses into one kernel with three dots.
        def f(x, y, z, w, r, t):
            # [FUNCTION DOC] HELPER: Returns the sum of the products x@y, z@w and r@t.
            return x @ y + z @ w + r @ t

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)
        w = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        z = rand_strided((K, N), (N, 1), device=GPU_TYPE)
        r = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        t = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, z, w, r, t))
        self._check_code(f, (x, y, z, w, r, t), 1, 3)

    def test_mm_complex(self):
        # [FUNCTION DOC] Matmul whose left operand is gathered with an integer index tensor,
        # followed by an add and a constant add; checks equality and one kernel with one dot.
        def f(x, y, z, w):
            # [FUNCTION DOC] HELPER: Gathers rows of x by z, multiplies by y, then adds w and the
            # constant 3.
            return x[z] @ y + w + 3

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        z = torch.randint(M, (M, K), dtype=torch.long, device=GPU_TYPE)
        w = rand_strided((M, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, z, w))
        self._check_code(f, (x, y, z, w), 1, 1)

    def test_batchmatmul(self):
        # [FUNCTION DOC] Batched matmul with batch 256; checks equality and one kernel
        # with one dot.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Batched matrix product of the two inputs.
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 256, 128, 128, 128
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    def test_bmm_vertical_fusion(self):
        # [FUNCTION DOC] Batched matmul followed by relu and a pointwise combination of the
        # result; checks that the epilogue fuses into one kernel with one dot.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Computes bmm, then relu of it plus the square of the
            # bmm result.
            z = torch.bmm(x, y)
            w = torch.nn.functional.relu(z)
            v = w + z * z
            return v

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    def test_bmm_horizontal_fusion(self):
        # [FUNCTION DOC] Two independent batched matmuls combined pointwise; checks equality and
        # one kernel containing two dots.
        def f(x, y, z, w):
            # [FUNCTION DOC] HELPER: Combines two bmm results by subtraction and product-add.
            bmm1 = torch.bmm(x, y)
            bmm2 = torch.bmm(z, w)
            return bmm1 - bmm2 + bmm1 * bmm2

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)
        z = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        w = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, z, w))
        self._check_code(f, (x, y, z, w), 1, 2)

    def test_bmm_fusion_complex1(self):
        # [FUNCTION DOC] Batched matmul using gathered inputs and a weighted einsum, accumulated
        # into an output with index add (comment in code describes the target indexing); with
        # random int64 index tensors, checks equality and a single kernel with one dot.
        # Out[O[b,i],j] += Input[I[b,i],r] * Weight[W[b],r,j] * Val[b,i]
        def f(inp, weight, val, I_idx, W_idx, O_idx, out):
            # [FUNCTION DOC] HELPER: Gathers input rows and per-batch weights, contracts them with
            # a value tensor via einsum, index-adds the flattened result into out along dim 0 and
            # returns out. Uses the outer variable J for the reshape width.
            x = inp[I_idx]  # [B,I,R]
            y = weight[W_idx]  # [B,R,J]
            t = torch.einsum("bir,brj,bi->bij", x, y, val)
            out.index_add_(0, O_idx.reshape(-1), t.reshape(-1, J))
            return out

        B, I, R, J = 32, 16, 128, 16
        N_in = 128
        N_w = 64
        N_out = 128

        inp = rand_strided((N_in, R), (R, 1), device=GPU_TYPE)
        weight = rand_strided((N_w, R, J), (R * J, J, 1), device=GPU_TYPE)
        val = rand_strided((B, I), (I, 1), device=GPU_TYPE)

        I_idx = torch.randint(0, N_in, (B, I), device=GPU_TYPE, dtype=torch.int64)
        W_idx = torch.randint(0, N_w, (B,), device=GPU_TYPE, dtype=torch.int64)
        O_idx = torch.randint(0, N_out, (B, I), device=GPU_TYPE, dtype=torch.int64)

        out = torch.zeros((N_out, J), device=GPU_TYPE)

        self._check_equal(f, (inp, weight, val, I_idx, W_idx, O_idx, out))
        self._check_code(f, (inp, weight, val, I_idx, W_idx, O_idx, out), 1, 1)

    def test_bmm_fusion_complex2(self):
        # [FUNCTION DOC] Batched matmul with a four-dimensional left operand and gathered right
        # operand, contracted via einsum and index-added into out; checks equality and one kernel
        # with one dot.
        # out[Ai[g],m,n] += Av[g,m,k,p] * B[Ak[g,p],k,n]
        def f(Av, B, Ai, Ak, out):
            # [FUNCTION DOC] HELPER: Gathers B by Ak, contracts with Av via einsum, index-adds the
            # result into out using Ai and returns out.
            Bg = B[Ak]  # [G,P,K,N]
            Cg = torch.einsum("gmkp,gpkn->gmn", Av, Bg)  # [G,M,N]
            out.index_add_(0, Ai, Cg)
            return out

        G, M, K, P, N = 128, 16, 64, 8, 16
        NB = 128
        NA = 128

        Av = rand_strided((G, M, K, P), (M * K * P, K * P, P, 1), device=GPU_TYPE)
        B = rand_strided((NB, K, N), (K * N, N, 1), device=GPU_TYPE)

        Ai = torch.randint(0, NA, (G,), device=GPU_TYPE, dtype=torch.int64)
        Ak = torch.randint(0, NB, (G, P), device=GPU_TYPE, dtype=torch.int64)

        out = torch.zeros((NA, M, N), device=GPU_TYPE)

        self._check_equal(f, (Av, B, Ai, Ak, out))
        self._check_code(f, (Av, B, Ai, Ak, out), 1, 1)

    def test_bmm_large_batch_reversed_pid(self):
        # [FUNCTION DOC] Batched matmul with batch 65537 (past the grid limit). Checks the
        # generated code assigns the batch offset to the first program id, then the second, then
        # the third in order.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Batched matrix product of the two inputs.
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 65537, 16, 16, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        f = torch.compile(f)
        code = run_and_get_triton_code(f, x, y)

        FileCheck().check("zoffset = tl.program_id(0)").check(
            "yoffset = tl.program_id(1)"
        ).check("xoffset = tl.program_id(2)").run(code)

    def test_bmm_no_z_broadcast(self):
        # [FUNCTION DOC] Checks the generated bmm triton code does not index the first block
        # dimension with a broadcast across the other three axes, and that the block aranges
        # appear with the expected broadcast positions for each of the four dimensions.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Batched matrix product of the two inputs.
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        f = torch.compile(f)
        code = run_and_get_triton_code(f, x, y)

        FileCheck().check_not("tl.arange(0, ZBLOCK)[:, None, None, None]").check(
            "tl.arange(0, ZBLOCK)"
        ).check("tl.arange(0, YBLOCK)[None, :, None, None]").check(
            "tl.arange(0, XBLOCK)[None, None, :, None]"
        ).check("tl.arange(0, R0_BLOCK)[None, None, None, :]").run(code)


if HAS_GPU:
    torch.set_default_device(GPU_TYPE)

if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
