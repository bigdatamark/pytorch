# Owner(s): ["module: inductor"]
# [MODULE DOC] GPU tests for Inductor online softmax: a fused single-pass computation of the row
# maximum and the sum of shifted exponentials, replacing the three-pass softmax. Tests check
# numerical accuracy against eager and float64 references, generated code shape (counts of
# reduction loops, presence of online_softmax_reduce and online_softmax_combine_with_sum
# markers), persistent reductions, split reductions, signed zero handling, NaN and minus-
# infinity inputs, SDPA pattern coexistence, CPU fallback, and partitioner interaction. Module
# level flags DO_PERF_TEST and USE_LARGE_INPUT are read from environment variables and enlarge
# inputs / enable benchmarking. Tests run only on Linux with a GPU and Triton when executed as a
# script.

import math
import os

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._dynamo.utils import rmse, same
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, HAS_TRITON


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"
USE_LARGE_INPUT = os.environ.get("USE_LARGE_INPUT") == "1" or DO_PERF_TEST


def _prepare_softmax(x, dim):
    # [FUNCTION DOC] HELPER: Reference for the first stage of softmax. Computes the maximum of the
    # input along the given dimension (kept as size one), then the sum along that dimension of the
    # exponential of the input minus that maximum. Returns the pair of maximum and sum tensors.
    xmax = x.amax(dim=dim, keepdim=True)
    xsum = (x - xmax).exp().sum(dim=dim, keepdim=True)
    return xmax, xsum


class TestOnlineSoftmax(TestCase):
    # [CLASS DOC] Test case for online softmax lowering on GPU. Helpers run softmax-like
    # functions eagerly and compiled, and inspect generated wrapper source; individual tests
    # cover accuracy, codegen structure and edge cases. Parametrized tests are expanded at
    # module end.
    def do_test_acc_and_perf(self, op):
        # [FUNCTION DOC] HELPER: Compares a compiled and eager version of the given softmax-like
        # operator applied along the last dimension to a random bfloat16 GPU matrix. Sizes are
        # large (gpt2 padded vocab) when perf testing is on, otherwise small to avoid CI memory
        # trouble. Asserts the results match within a tolerance of 1e-2 and, with perf testing,
        # benchmarks both and prints the timings.
        if DO_PERF_TEST:
            N = 32 * 1024
            V = 50304  # padded version for gpt2
        else:
            N, V = 1024, 2048  # small value to avoid OOM in CI

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the captured operator along the last dimension of
            # its input.
            return op(x, dim=-1)

        x = torch.randn(N, V, dtype=torch.bfloat16, device=GPU_TYPE)
        opt_f = torch.compile(f)
        expected = f(x)
        actual = opt_f(x)

        self.assertTrue(same(expected, actual, tol=1e-2))

        if DO_PERF_TEST:
            from triton.testing import do_bench

            eager_ms = do_bench(lambda: f(x))
            opt_ms = do_bench(lambda: opt_f(x))
            print(f"{eager_ms=}")
            print(f"{opt_ms=}")

    def test_softmax(self):
        # [FUNCTION DOC] Accuracy check of softmax compiled with online softmax versus eager.
        self.do_test_acc_and_perf(torch.softmax)

    def test_log_softmax(self):
        # [FUNCTION DOC] Accuracy check of log_softmax compiled with online softmax versus
        # eager.
        self.do_test_acc_and_perf(torch.log_softmax)

    @inductor_config.patch(use_fast_math=True)
    def test_prepare_softmax_perf(self):
        # [FUNCTION DOC] Accuracy check of the prepare-softmax helper (max and sum of
        # exponentials) with fast math enabled in the Inductor config.
        self.do_test_acc_and_perf(_prepare_softmax)

    def get_softmax_wrapper(self, V=50304, use_log_softmax=False, device=GPU_TYPE):
        # [FUNCTION DOC] HELPER: Compiles a softmax (or log_softmax if requested) over the last
        # dimension of a random bfloat16 matrix with 32768 rows and V columns on the given
        # device, runs it while capturing generated code, and returns the first captured source
        # string (the wrapper code).
        N = 32 * 1024

        @torch.compile
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function applying log_softmax or softmax on the
            # last dimension depending on the enclosing flag.
            if use_log_softmax:
                return torch.log_softmax(x, dim=-1)
            else:
                return torch.softmax(x, dim=-1)

        x = torch.randn(N, V, dtype=torch.bfloat16, device=device)
        out, source_codes = run_and_get_code(f, x)
        return source_codes[0]

    def test_codegen_3pass_softmax_due_to_disable(self):
        # [FUNCTION DOC] With online softmax disabled in config, the generated code must contain
        # exactly three reduction loops, i.e. the classic three-pass softmax.
        with inductor_config.patch(online_softmax=False):
            wrapper_code = self.get_softmax_wrapper()

        self.assertEqual(wrapper_code.count("for r0_offset in"), 3)

    @parametrize("V", [2048, 50304])
    @parametrize("use_log_softmax", [False, True])
    def test_codegen_online_softmax(self, use_log_softmax, V):
        # [FUNCTION DOC] Parametrized over vocabulary size and softmax kind. With online softmax
        # enabled the generated code must contain exactly two reduction loops (one fused max/sum
        # pass, one output pass).
        wrapper_code = self.get_softmax_wrapper(use_log_softmax=use_log_softmax, V=V)

        self.assertEqual(wrapper_code.count("for r0_offset in"), 2)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @parametrize("use_log_softmax", [False, True])
    def test_codegen_online_softmax_unbacked_non_reduction_dim(self, use_log_softmax):
        # [FUNCTION DOC] Same two-loop expectation when the number of rows is data dependent
        # (unbacked, obtained from a scalar tensor with scalar output capture enabled) and the
        # reduction runs over the last dimension. Requires full graph compilation and checks
        # that the online reduce marker appears in the generated code. Parametrized over softmax
        # or log_softmax.
        def f(x, n):
            # [FUNCTION DOC] HELPER: Slices the leading rows by a value read from a tensor, then
            # applies softmax or log_softmax on the last dimension.
            x = x[: n.item(), :]
            if use_log_softmax:
                return torch.log_softmax(x, dim=-1)
            else:
                return torch.softmax(x, dim=-1)

        x = torch.randn(1024, 2048, dtype=torch.bfloat16, device=GPU_TYPE)
        n = torch.tensor(1024)
        _, source_codes = run_and_get_code(torch.compile(f, fullgraph=True), x, n)
        wrapper_code = "\n".join(source_codes)

        self.assertEqual(wrapper_code.count("for r0_offset in"), 2)
        self.assertTrue("online_softmax_reduce" in wrapper_code)

    def test_no_online_softmax_for_cpu(self):
        # [FUNCTION DOC] On CPU, online softmax is not used; the generated C++ must contain
        # exactly four explicit row-level loops (the GPU parallelizes rows in hardware).
        code = self.get_softmax_wrapper(V=2048, device="cpu")

        # CPU need an explicit loop across different rows.
        # For GPU, this is parallelized by the hardware.
        self.assertEqual(code.count("for(int64_t"), 4)

    def test_codegen_softmax_persistent_reduction(self):
        # [FUNCTION DOC] For a small row length (1024) the reduction is persistent, so the
        # generated code has no reduction loop at all.
        """
        Persistent reduction has no for loops.
        """
        wrapper_code = self.get_softmax_wrapper(1024)
        self.assertEqual(wrapper_code.count("for r0_offset in"), 0)

    @inductor_config.patch("triton.persistent_reductions", False)
    def test_sdpa(self):
        # [FUNCTION DOC] With persistent reductions disabled, compiles the attention-like
        # pattern (scaled matmul, softmax, matmul) and checks the output matches eager within
        # 1e-2 and that the fused scaled dot product attention op is still pattern matched, i.e.
        # online softmax does not interfere.
        """
        Make sure online softmax here does not conflict with the sdpa
        patterns.
        """
        q, k, v = (
            torch.randn((4, 2, 16, 32), device=GPU_TYPE, dtype=torch.bfloat16)
            for _ in range(3)
        )

        def f(q, k, v):
            # [FUNCTION DOC] HELPER: Reference attention computation: query times transposed
            # key, divided by the square root of the last key dimension, softmax, times value.
            return (
                torch.matmul(q, k.transpose(-2, -1))
                .div(math.sqrt(k.shape[-1]))
                .softmax(dim=-1)
                .matmul(v)
            )

        opt_f = torch.compile(f)
        ref = f(q, k, v)
        act, (code,) = run_and_get_code(opt_f, q, k, v)
        self.assertTrue(torch.allclose(ref, act, atol=1e-2, rtol=1e-2))
        self.assertTrue("aten._scaled_dot_product_" in code)

    @parametrize("nrow", [2, 2048])
    @parametrize("dim", [-1, 0, 1])
    def test_prepare_softmax(self, dim, nrow):
        # [FUNCTION DOC] Parametrized over reduction dimension and row count. Compiles the
        # prepare-softmax helper and compares with eager (tolerance 1e-2). For the tall case
        # reducing over dimension zero the code must contain at least one (or two when a split
        # reduction is expected, which depends on the device multiprocessor count) generated
        # kernels. Otherwise the number of reduction loops must be zero for the two-row case
        # reducing over dimension zero (persistent reduction) and one elsewhere.
        x = torch.randn(nrow, 2048, dtype=torch.bfloat16, device=GPU_TYPE)
        act, (code,) = run_and_get_code(torch.compile(_prepare_softmax), x, dim)
        ref = _prepare_softmax(x, dim)
        self.assertTrue(same(ref, act, tol=1e-2))

        if nrow == 2048 and dim == 0:
            num_kernels = 2
            # split reduction may be triggered depending on the device's SM/CU count.
            # The heuristic in num_splits() in ir.py returns split=1 (no split) when:
            #   numel_hint >= num_sm * 2 * 32
            # When dim=0, numel_hint (output size) = ncol (2048)
            # split is expected only when num_sm > 32
            props = DeviceProperties.create(torch.device(GPU_TYPE))
            num_sm = props.multi_processor_count
            split_not_expected = 2048 >= num_sm * 2 * 32
            if split_not_expected:
                num_kernels = 1

            self.assertTrue(code.count("def triton") >= num_kernels)
        else:
            if nrow == 2 and dim == 0:
                # persistent reduction triggered
                expected_num_loop = 0
            else:
                # A single loop due to online softmax
                expected_num_loop = 1
            self.assertEqual(code.count("for r0_offset in"), expected_num_loop)

    @inductor_config.patch(strict_signed_zero=True)
        # [FUNCTION DOC] With strict signed zero enabled, checks that the online reduce path is
        # used and that the maximum of rows made of mixed positive and negative zeros is bit-
        # identical to a plain compiled maximum (compared as 32-bit integer views), and that the
        # sums equal eager.
    def test_prepare_softmax_signed_zero(self):
        def reduce_max(x):
            # [FUNCTION DOC] HELPER: Maximum over the last dimension, keeping the dimension.
            return x.amax(dim=-1, keepdim=True)

        x = torch.zeros(2, 2048, device=GPU_TYPE)
        x[0, 1::2] = -0.0
        x[1, ::2] = -0.0

        ref = _prepare_softmax(x, -1)
        ref_max = torch.compile(reduce_max, fullgraph=True)(x)
        act, (code,) = run_and_get_code(torch.compile(_prepare_softmax), x, -1)

        self.assertIn("online_softmax_reduce", code)
        self.assertEqual(ref_max.view(torch.int32), act[0].view(torch.int32))
        self.assertEqual(ref[1], act[1])

    def test_prepare_softmax_after_partitioning(self):
        # [FUNCTION DOC] Verifies that applying the prepare-softmax replacement on the joint
        # forward and backward graph, before min-cut partitioning, rewrites the softmax into the
        # online form and does not increase the bytes of saved activations versus not applying
        # it. Builds a custom autograd backend, compiles a small attention-like function with
        # gradients, and inspects generated joint graph code and saved tensors. Asserts the
        # pattern is absent without the pass and matched exactly once with it.
        from torch._dynamo.backends.common import aot_autograd
        from torch._functorch.aot_autograd import make_boxed_func
        from torch._functorch.partitioners import min_cut_rematerialization_partition
        from torch._inductor.decomposition import select_decomp_table
        from torch._inductor.fx_passes import post_grad
        from torch._inductor.fx_passes.joint_graph import joint_graph_passes
        from torch._inductor.pattern_matcher import (
            fwd_only,
            PatternMatcherPass,
            register_replacement,
        )

        def apply_online_softmax_in_joint_graph(gm):
            # [FUNCTION DOC] HELPER: Registers the prepare-softmax pattern and replacement into
            # a fresh pattern matcher pass (using the example input and a fixed last dimension)
            # and applies it to the graph of the given module. Returns the pattern matcher
            # result (the match count).
            patterns = PatternMatcherPass(pass_name="test_joint_online_softmax")
            register_replacement(
                post_grad.prepare_softmax_pattern,
                post_grad.prepare_softmax_replacement,
                [torch.empty(4, 8)],
                fwd_only,
                patterns,
                scalar_workaround={"dim": -1},
                extra_check=post_grad.prepare_softmax_extra_check,
            )
            return patterns.apply(gm.graph)

        def saved_activation_bytes(fw_module, num_fwd_outputs):
            # [FUNCTION DOC] HELPER: Sums the byte sizes of tensor values recorded in node
            # metadata for the saved (non-forward) outputs of a partitioned forward module. Non-
            # node outputs and nodes without a tensor value are ignored. Returns the total as an
            # integer.
            (output_node,) = fw_module.graph.find_nodes(op="output")
            saved_values = output_node.args[0][num_fwd_outputs:]
            total = 0
            for node in saved_values:
                if not isinstance(node, torch.fx.Node):
                    continue
                value = node.meta.get("val")
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
            return total

        def compile_and_capture_saved_bytes(apply_joint_online_softmax):
            # [FUNCTION DOC] HELPER: Compiles the attention-like function with a capturing
            # backend and runs forward and backward. Resets dynamo first. A custom partition
            # function runs joint graph passes, optionally the online softmax pattern pass (then
            # lints and recompiles the graph), records the joint graph code, partitions with
            # min-cut rematerialization and records the saved activation bytes. Returns a
            # mapping with the captured entries (joint graph code, saved bytes, and the match
            # count when the pass was applied).
            captured = {}

            def partition_fn(gm, joint_inputs, *, num_fwd_outputs, **kwargs):
                # [FUNCTION DOC] HELPER: Partition function used by the custom backend; as
                # described by the enclosing helper, it records data into the shared capture
                # mapping and returns the forward and backward modules.
                joint_graph_passes(gm)
                if apply_joint_online_softmax:
                    captured["joint_online_softmax_count"] = (
                        apply_online_softmax_in_joint_graph(gm)
                    )
                    gm.graph.lint()
                    gm.recompile()
                captured["joint_graph_code"] = gm.code
                fw_module, bw_module = min_cut_rematerialization_partition(
                    gm,
                    joint_inputs,
                    num_fwd_outputs=num_fwd_outputs,
                    **kwargs,
                )
                captured["saved_activation_bytes"] = saved_activation_bytes(
                    fw_module, num_fwd_outputs
                )
                return fw_module, bw_module

            def compiler(gm, example_inputs):
                # [FUNCTION DOC] HELPER: Trivial compiler that returns the module forward
                # wrapped as a boxed function, with no optimization.
                return make_boxed_func(gm.forward)

            backend = aot_autograd(
                fw_compiler=compiler,
                bw_compiler=compiler,
                partition_fn=partition_fn,
                decompositions=select_decomp_table(),
            )

            def fn(q, k, target):
                # [FUNCTION DOC] HELPER: Function under test: attention-like scores from query
                # and transposed key, softmax probabilities weighted by the target and summed,
                # plus the sum of sine of the scores.
                scores = q @ k.transpose(-2, -1)
                probs = scores.softmax(dim=-1)
                return (probs * target).sum() + scores.sin().sum()

            B, S, D = 2, 128, 16
            args = (
                torch.randn(
                    B, S, D, device=GPU_TYPE, dtype=torch.float16, requires_grad=True
                ),
                torch.randn(
                    B, S, D, device=GPU_TYPE, dtype=torch.float16, requires_grad=True
                ),
                torch.randn(
                    B, S, S, device=GPU_TYPE, dtype=torch.float16, requires_grad=True
                ),
            )

            torch._dynamo.reset()
            loss = torch.compile(fn, backend=backend, fullgraph=True)(*args)
            torch.autograd.grad(loss, args)
            return captured

        current = compile_and_capture_saved_bytes(apply_joint_online_softmax=False)
        joint_online = compile_and_capture_saved_bytes(apply_joint_online_softmax=True)

        self.assertNotIn("prepare_softmax_online", current["joint_graph_code"])
        self.assertEqual(joint_online["joint_online_softmax_count"], 1)
        self.assertIn("prepare_softmax_online", joint_online["joint_graph_code"])
        self.assertLessEqual(
            joint_online["saved_activation_bytes"],
            current["saved_activation_bytes"],
        )

    @parametrize("strict_signed_zero", [False, True])
    def test_split_reduction(self, strict_signed_zero):
        # [FUNCTION DOC] Parametrized over strict signed zero. For a one-row, very long (just
        # over a million) bfloat16 input, the reduction is split. Checks the compiled softmax
        # matches eager within 1e-3, that multiple kernels are emitted, and that both the online
        # reduce and the combine-with-sum markers appear (partial results are merged by another
        # online reduce).
        """
        Split online_softmax_reduce into partial max/sum tuples and combine
        the partials with another online_softmax_reduce.
        """
        # tensor shape to trigger split reduction
        x = torch.randn(1, 2**20 + 13, dtype=torch.bfloat16, device=GPU_TYPE)
        ref = torch.softmax(x, dim=-1)
        with inductor_config.patch(strict_signed_zero=strict_signed_zero):
            act, (code,) = run_and_get_code(torch.compile(torch.softmax), x, dim=-1)
        self.assertTrue(torch.allclose(ref, act, atol=1e-3, rtol=1e-3))
        self.assertTrue(code.count("def triton") >= 2)
        self.assertTrue("online_softmax_reduce" in code)
        self.assertTrue("online_softmax_combine_with_sum" in code)

    def test_kl_div_log_softmax_backward_split_reduction(self):
        # [FUNCTION DOC] Compiles a KL divergence of a log_softmax against softmax targets over
        # a single very long float32 row and checks that forward value and input gradient match
        # eager, and that both split-reduction markers (online reduce and combine with sum) show
        # up in the generated code of forward and backward.
        logits = torch.randn(
            1, 2**20, dtype=torch.float32, device=GPU_TYPE, requires_grad=True
        )
        targets = F.softmax(torch.randn_like(logits), dim=-1)
        ref_logits = logits.detach().clone().requires_grad_()
        ref_targets = targets.detach().clone()

        def f(logits, targets):
            # [FUNCTION DOC] HELPER: Batch-mean KL divergence between log_softmax of the logits
            # and the targets.
            return F.kl_div(
                F.log_softmax(logits, dim=-1), targets, reduction="batchmean"
            )

        ref = f(ref_logits, ref_targets)
        ref.backward()

        opt_f = torch.compile(f)
        act, codes = run_and_get_code(opt_f, logits, targets)
        act.backward()
        code = "\n".join(codes)

        self.assertEqual(ref, act)
        self.assertEqual(ref_logits.grad, logits.grad)
        self.assertTrue("online_softmax_reduce" in code)
        self.assertTrue("online_softmax_combine_with_sum" in code)

    @parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
    def test_prepare_softmax_acc_with_fp64(self, dtype):
        # [FUNCTION DOC] Parametrized over dtype (bfloat16, half, float32). Checks the compiled
        # prepare-softmax helper uses the online reduce, gives exactly the same maximum as
        # eager, and that the root mean squared error of the sum against a float64 reference is
        # not worse than eager by more than 0.1. Prints both errors. Input size depends on the
        # large input flag.
        if USE_LARGE_INPUT:
            M, N = 32768, 50257
        else:
            M, N = 1024, 2048

        x = torch.randn(M, N, device=GPU_TYPE, dtype=dtype)

        ref_fp64 = _prepare_softmax(x.to(dtype=torch.float64), dim=-1)
        ref = _prepare_softmax(x, dim=-1)
        res, (code,) = run_and_get_code(torch.compile(_prepare_softmax), x, dim=-1)
        self.assertTrue("online_softmax_reduce" in code)

        # Max should be exactly equal
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[0].to(dtype=torch.float64), ref_fp64[0])

        ref_error = rmse(ref_fp64[1], ref[1]).item()
        res_error = rmse(ref_fp64[1], res[1]).item()

        # My local tests even shows a smaller res_error:
        #   ref_error=2.1065, res_error=2.1028
        # for bf16
        #   ref_error=0.2611, res_error=0.2609
        # for fp16
        #   ref_error=0.0001, res_error=0.0001
        # for fp32
        print(f"{ref_error=:.4f}, {res_error=:.4f}")

        self.assertTrue(
            res_error < ref_error + 0.1
        )  # Is this good enough to make CI stable

    @parametrize("fn", [torch.log_softmax, torch.softmax])
    @parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
    def test_softmax_acc_with_fp64(self, dtype, fn):
        # [FUNCTION DOC] Parametrized over function (softmax or log_softmax) and dtype. Checks
        # online reduce is used and that the compiled result's root mean squared error against a
        # float64 reference is within 0.1 of eager's error. Prints both errors. Input size
        # depends on the large input flag.
        if USE_LARGE_INPUT:
            M, N = 32768, 50257
        else:
            M, N = 1024, 2048

        x = torch.randn(M, N, device=GPU_TYPE, dtype=dtype)

        ref_fp64 = fn(x.to(dtype=torch.float64), dim=-1)
        ref = fn(x, dim=-1)
        res, (code,) = run_and_get_code(torch.compile(fn), x, dim=-1)
        self.assertTrue("online_softmax_reduce" in code)

        ref_error = rmse(ref_fp64, ref).item()
        res_error = rmse(ref_fp64, res).item()

        # For torch.softmax,
        # I get almost 0 for ref_error/res_error for all 3 dtypes. It's because
        # each value is very small since each row add up to 1.0
        #
        # For torch.log_softmax
        #   ref_error=0.0180399032, res_error=0.0180399031
        # for bf16
        #   ref_error=0.0022548872, res_error=0.0022548872
        # for fp16
        #   ref_error=0.0000003744, res_error=0.0000003748
        # for fp32
        print(f"{ref_error=:.10f}, {res_error=:.10f}")

        self.assertTrue(
            res_error < ref_error + 0.1
        )  # Is this good enough to make CI stable

    def test_softmin(self):
        # [FUNCTION DOC] A softmin over a dimension of size one has a trivial reduction that
        # must be unrolled: the result matches eager and no online reduce appears in the
        # generated code.
        """
        The rnumel==1 kind of reduction should be unrolled.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Softmin along the first dimension.
            return F.softmin(x, dim=0)

        x = torch.randn(1, device=GPU_TYPE)
        ref = f(x)
        act, (code,) = run_and_get_code(torch.compile(f), x)
        self.assertTrue(torch.allclose(ref, act))
        self.assertTrue("online_softmax_reduce" not in code)

    def test_causal_mask(self):
        # [FUNCTION DOC] Softmax of a causal-masked matrix (upper triangle set to minus
        # infinity) must not produce NaN in either eager or compiled output, and the two must be
        # close.
        def f(x):
            # [FUNCTION DOC] HELPER: Softmax along the last dimension.
            return x.softmax(dim=-1)

        x = torch.randn(2048, 2048, device=GPU_TYPE)
        mask = torch.tril(torch.ones(2048, 2048, device=GPU_TYPE))
        x.masked_fill_(mask == 0, float("-inf"))

        ref = f(x)
        act = torch.compile(f)(x)
        self.assertTrue(not ref.isnan().any())
        self.assertTrue(not act.isnan().any())
        self.assertTrue(torch.allclose(ref, act))

    def test_tb_speech_transformer_attn(self):
        # [FUNCTION DOC] Regression test from a speech transformer attention. Because the online
        # algorithm uses a running maximum over partial elements, an all minus-infinity prefix
        # would give NaN from subtracting infinity from itself; the implementation treats that
        # case as zero. Seeds the RNG, masks random positions with minus infinity, and checks
        # neither eager nor compiled sum contains NaN and they are close.
        """
        This is an example extracted from speech_transformer.
        Since online softmax use the max from partial elements of an entire
        row, if the input contains '-inf', it's possible that the
        max of those partial elements is '-inf' even if the entire row
        has non '-inf' value. In this cause, online softmax will need
        do things like 'float(-inf) - float(-inf)' which becomes 'nan'.
        We fixed this by interpreting 'float(-inf) - float(-inf)' as 0
        if we found both operands are 'float(-inf)'.
        """
        torch.manual_seed(1337)

        def f(x, mask):
            # [FUNCTION DOC] HELPER: Replaces masked positions with minus infinity, then returns
            # the sum along the last dimension of the exponential of the input minus its
            # maximum.
            x = torch.where(mask, float("-inf"), x)
            xmax = x.amax(dim=-1, keepdim=True)
            xsum = (x - xmax).exp().sum(dim=-1, keepdim=True)
            return xsum

        x = torch.randn(8, 10, 22, 204, device=GPU_TYPE)
        mask = torch.randint(0, 2, (10, 204), device=GPU_TYPE) == 0
        mask = mask.view(1, 10, 1, 204)

        ref = f(x, mask)
        act = torch.compile(f)(x, mask)
        self.assertTrue(not ref.isnan().any())
        self.assertTrue(not act.isnan().any())
        self.assertTrue(torch.allclose(ref, act))

    @inductor_config.patch(split_reductions=False)
    def test_3d_tiled_online_softmax(self):
        # [FUNCTION DOC] With split reductions disabled, checks a softmax of the product of two
        # permuted (non-contiguous, 3D tiled) inputs matches eager within 1e-3.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Softmax along the last dimension of the elementwise
            # product.
            return (x * y).softmax(dim=-1)

        M, N, K = 32, 8, 1024

        x = torch.randn(K, N, M, device=GPU_TYPE).permute(2, 1, 0)
        y = torch.randn(K, M, N, device=GPU_TYPE).permute(1, 2, 0)

        opt_f = torch.compile(f)
        torch.testing.assert_close(f(x, y), opt_f(x, y), atol=1e-3, rtol=1e-3)

    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_nan_propagation(self, dtype):
        # [FUNCTION DOC] Parametrized over dtype (bfloat16, float32). Skipped without Triton.
        # Puts a NaN at the start, middle and end of three different rows of a random matrix.
        # Checks the generated code uses the NaN-ignoring maximum helper, that those three
        # output rows are entirely NaN in both eager and compiled results, and that the clean
        # fourth row matches eager.
        """
        The softmax-internal max uses fmax (non-NaN-propagating) for
        performance, but NaN must still propagate to the final output
        because the original input flows through exp(x - xmax).
        Place NaN at the beginning, middle, and end of separate rows.

        This is Triton-only because fmax is only implemented in the
        Triton persistent reduction path.
        """
        if not HAS_TRITON:
            self.skipTest("requires triton")

        M, N = 4, 1024
        x = torch.randn(M, N, device=GPU_TYPE, dtype=dtype)

        x[0, 0] = float("nan")
        x[1, N // 2] = float("nan")
        x[2, N - 1] = float("nan")
        # row 3 has no NaN

        ref = torch.softmax(x, dim=-1)
        act, (code,) = run_and_get_code(torch.compile(torch.softmax), x, dim=-1)

        self.assertIn("fmax2", code)

        # Rows with NaN input must produce all-NaN output
        for row in range(3):
            self.assertTrue(
                ref[row].isnan().all(),
                f"eager row {row} should be all NaN",
            )
            self.assertTrue(
                act[row].isnan().all(),
                f"compiled row {row} should be all NaN",
            )

        # Row without NaN must match exactly
        self.assertFalse(act[3].isnan().any())
        torch.testing.assert_close(ref[3], act[3])


instantiate_parametrized_tests(TestOnlineSoftmax)

if __name__ == "__main__":
    if IS_LINUX and HAS_GPU and HAS_TRITON:
        run_tests()
