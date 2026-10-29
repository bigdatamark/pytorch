# Owner(s): ["module: inductor"]
# ruff: noqa: F841
# [MODULE DOC] Inductor regression tests that run on a GPU (CUDA, or XPU where noted): minified
# repros of past bugs plus checks on generated Triton code.
# The only test class is CudaReproTests; the file ends with a main block that runs the tests when
# GPU and Triton are available and ASAN is off.
# Import-time side effects: on ROCm it sets the layout optimization config and the MIOpen NHWC
# environment variable; it reads DO_PERF_TEST from the environment; it raises a skip (or exits
# quietly when run as a script) if triton or test_torchinductor cannot be imported; it picks
# device_type from the current accelerator, falling back to cpu.

import copy
import functools
import gc
import math
import os
import sys
import unittest

import torch
import torch._dynamo.config as dynamo_config
import torch.backends.cuda
import torch.nn.functional as F
from torch import nn
from torch._dynamo.debug_utils import same_two_models
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.utils import (
    run_and_get_code,
    run_and_get_graph_lowering,
    run_fw_bw_and_get_code,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    SM80OrLater,
    SM90OrLater,
    TEST_MULTIGPU,
    tf32_on_and_off,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    freeze_rng_state,
    instantiate_parametrized_tests,
    IS_FBCODE,
    MI350_ARCH,
    parametrize,
    skipIfCachingAllocatorDisabled,
    skipIfRocm,
    skipIfRocmArch,
    skipIfXpu,
    TEST_CUDA,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TEST_XPU,
    xfailIfROCm,
)
from torch.testing._internal.inductor_utils import HAS_GPU, IS_BIG_GPU


if TEST_WITH_ROCM:
    config.force_layout_optimization = 1
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


requires_multigpu = functools.partial(
    unittest.skipIf, not TEST_MULTIGPU, "requires multiple cuda devices"
)
from torch._dynamo.utils import counters
from torch.testing._internal.inductor_utils import skipCUDAIf


def _strip_triton_static_asserts(code):
    # [FUNCTION DOC] HELPER: Return the given generated source text with every line that mentions
    # a Triton static assert removed, so FileCheck patterns are not confused by those lines. Lines
    # are re-joined with newlines.
    return "\n".join(
        line for line in code.splitlines() if "tl.static_assert" not in line
    )


try:
    try:
        import triton  # @manual
        from triton import language as tl  # @manual
    except ImportError:
        raise unittest.SkipTest("requires triton")  # noqa: B904

    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


TestCase = test_torchinductor.TestCase
ToTuple = test_torchinductor.ToTuple
check_model_cuda = test_torchinductor.check_model_cuda
aten = torch.ops.aten


device_type = (
    acc.type
    if (acc := torch.accelerator.current_accelerator(check_available=True))
    else "cpu"
)


@instantiate_parametrized_tests
class CudaReproTests(TestCase):
    # [CLASS DOC] Main collection of GPU inductor repro tests. The class attribute device is the
    # detected accelerator type and common is the model checking helper from test_torchinductor
    # (compares compiled and eager results). Each test builds a small model or function, compiles
    # it with inductor, and compares against eager or inspects generated code. Decorated so
    # parametrized tests are instantiated.
    device = device_type
    common = check_model_cuda

    def test_mm_out_dtype_compile(self):
        # [FUNCTION DOC] Compile a matrix multiply that requests a float32 output from float16
        # inputs and check the compiled result has the same dtype and values as eager.
        a = torch.randn(1, 3, device=device_type, dtype=torch.float16)
        b = torch.randn(3, 2, device=device_type, dtype=torch.float16)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Matrix multiply of the two arguments with an explicit float32
            # output dtype.
            return torch.mm(x, y, out_dtype=torch.float32)

        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result, expected)

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_frexp_non_finite(self):
        # [FUNCTION DOC] Check that frexp on infinity, negative infinity and NaN gives the same
        # mantissa and exponent under the inductor backend as in eager (NaN treated as equal).
        # Skipped without CUDA.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the frexp decomposition (mantissa, exponent) of the
            # input.
            return torch.frexp(x)

        x = torch.tensor(
            [float("inf"), float("-inf"), float("nan")], device=device_type
        )
        expected_mantissa, expected_exponent = fn(x)
        actual_mantissa, actual_exponent = torch.compile(
            fn, backend="inductor", fullgraph=True
        )(x)

        self.assertEqual(actual_mantissa, expected_mantissa, equal_nan=True)
        self.assertEqual(actual_exponent, expected_exponent)

    def test_index_put_issue(self):
        # [FUNCTION DOC] Regression test for an index_put accumulation graph: traces a hand-
        # written aten forward with make_fx, compiles it with compile_fx_inner on ones-filled GPU
        # inputs (plus a leading dummy scalar for the self argument) and runs it. Passes if
        # compilation and execution do not raise.
        def forward(
            self,
            arg76_1,
            expand_default,
            full_like_default,
            _to_copy_default_67,
            zeros,
        ):
            # [FUNCTION DOC] HELPER: Traced graph body. Sums one input over dim 0, reshapes it,
            # selects between it and a filler tensor using a mask, then does an accumulating
            # index_put into a clone of the zeros input at the given indices. Returns a one-
            # element tuple. The self argument is unused.
            sum_sym_int_19 = torch.ops.aten.sum(_to_copy_default_67, [0], True)
            view_default_57 = torch.ops.aten.view.default(sum_sym_int_19, [512, 768])
            where_self = torch.ops.aten.where.self(
                expand_default, view_default_57, full_like_default
            )
            clone_default_12 = torch.ops.aten.clone.default(zeros)
            index_put__default = torch.ops.aten.index_put_.default(
                clone_default_12, [arg76_1], where_self, True
            )
            return (index_put__default,)

        inps = [
            (torch.Size([512]), torch.int64),
            (torch.Size([512, 768]), torch.bool),
            (torch.Size([512, 768]), torch.float16),
            (torch.Size([4, 512, 768]), torch.float16),
            (torch.Size([512, 768]), torch.float16),
        ]
        inps = [torch.zeros(())] + [
            torch.ones(shape, dtype=dtype, device=device_type)
            for (shape, dtype) in inps
        ]
        mod = make_fx(forward)(*inps)
        compiled = compile_fx_inner(mod, inps)
        compiled(inps)

    def test_view_replay_padding_issue_163328(self):
        # [FUNCTION DOC] Compile a ReproModule copy with fullgraph and reduce-overhead mode and
        # assert that both output tensors match eager with the same weights. Seeds the RNG first.
        class ReproModule(nn.Module):
            # [CLASS DOC] HELPER: Module with two linear layers (16 inputs to 240 outputs each)
            # used to produce a base tensor reshaped in two different ways.
            # [FUNCTION DOC] Regression test (issue 163328) for view replay with padded strides
            # under cudagraph mode: compares compiled and eager outputs of a module whose two
            # reshaped views come from one concatenated tensor.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the two linear layers and store the sizes: 120
                # output points and 2 trailing positions that go to the second linear.
                super().__init__()
                self.num_points_out = 120
                self.lc_num = 2
                input_channels = 16
                self.linear_main = nn.Linear(input_channels, self.num_points_out * 2)
                self.linear_lc = nn.Linear(input_channels, self.num_points_out * 2)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Split the input along dim 1 into a main part and the last
                # two entries, flatten each to 2D, run each through its own linear, concatenate
                # the results along dim 1 and return a mapping with two different reshapes of the
                # result (5D and 3D) under the keys ten0 and ten1.
                bs, num_lat, num_lon, channels = x.shape
                index = num_lat - self.lc_num

                main_x = x[:, :index].reshape(bs * index * num_lon, channels)
                lc_x = x[:, index:].reshape(bs * self.lc_num * num_lon, channels)

                refline = self.linear_main(main_x).reshape(bs, index, num_lon, -1)
                lc_refline = self.linear_lc(lc_x).reshape(bs, self.lc_num, num_lon, -1)

                base = torch.cat([refline, lc_refline], dim=1).contiguous()
                out0 = base.reshape(bs, num_lat, num_lon, self.num_points_out, 2)
                out1 = base.reshape(bs, num_lat * num_lon, self.num_points_out * 2)
                return {"ten0": out0, "ten1": out1}

        torch.manual_seed(0)
        model = ReproModule().to(device_type)
        inputs = torch.randn(36, 9, 7, 16, device=device_type, requires_grad=True)

        eager_out = model(inputs)
        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=True,
        )
        compiled_out = compiled_model(inputs)

        self.assertEqual(compiled_out["ten0"], eager_out["ten0"])
        self.assertEqual(compiled_out["ten1"], eager_out["ten1"])

    def test_effn_attn_bias_padding(self):
        # [FUNCTION DOC] Check that an expanded attention bias passed to the efficient SDPA op is
        # padded so its last dim stride is 1 (needed for alignment) while the broadcast head dim
        # keeps stride 0. Runs the compiled op, uses FileCheck on the generated code for the
        # expected stride pattern, and compares the first two outputs against eager using a fully
        # materialized bias (the remaining outputs are RNG related and skipped).
        batch_size, num_heads, seq_len, head_dim = 2, 32, 512, 128

        def fn(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            input_tensor: torch.Tensor,  # This will be our starting point
        ):
            # [FUNCTION DOC] HELPER: Expand the bias to the full head and sequence shape and call
            # the efficient scaled dot product attention op with log-sumexp, no dropout and no
            # causal mask. Returns that op's result tuple.
            # Input tensor should be [2, 1, 8192, 1] with appropriate strides
            bias = torch.ops.aten.expand(
                input_tensor, [2, 32, seq_len, seq_len]
            )  # Expands with stride pattern [65536, 0, 8, 0]

            return torch.ops.aten._scaled_dot_product_efficient_attention(
                query,
                key,
                value,
                bias,
                compute_log_sumexp=True,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
            )

        query = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device_type
        )
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device_type)
        value = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device_type
        )

        input_tensor = torch.rand([2, 1, seq_len, 1], device=device_type)

        out, code = run_and_get_code(torch.compile(fn), query, key, value, input_tensor)

        input_tensor2 = torch.rand([2, 32, seq_len, seq_len], device=device_type).copy_(
            input_tensor
        )
        # even though the last dim is broadcasted, needs stride 1 for alignment
        # but dim 1 stride can be 0
        FileCheck().check("buf0").check("(262144, 0, 512, 1").run(code[0])

        # dont check rng state
        self.assertEqual(out[:2], fn(query, key, value, input_tensor2)[:2])

    # Fails on ROCm MI350
    # Mismatched elements: 23 / 33062912 (0.0%)
    # Greatest absolute difference: 0.07861328125 at index (14, 13, 1008, 36) (up to 1e-05 allowed)
    # Greatest relative difference: 2.90625 at index (14, 13, 1008, 36) (up to 0.016 allowed)
    @skipIfXpu(msg="RuntimeError, not target, torch-xpu-ops: 2697")
    @skipIfRocmArch(MI350_ARCH)
    def test_effn_attn_bias_padding_misaligned(self):
        # [FUNCTION DOC] For three sequence lengths around 1008 (misaligned versus 8), compile
        # bf16 efficient-attention SDPA with a boolean mask and check the padded bias has an
        # expanded dim, only a single strided buffer allocation appears in the generated code, and
        # the output matches eager. Resets dynamo each iteration. Skipped on XPU and on ROCm MI350
        # because of known mismatches.
        seqlen_start = 1008

        for offset in range(-1, 2):
            seqlen = seqlen_start + offset
            torch._dynamo.reset()

            bsz = 32
            q = torch.randn(
                bsz, 16, seqlen, 64, dtype=torch.bfloat16, device=device_type
            )
            k = torch.randn(
                bsz, 16, seqlen, 64, dtype=torch.bfloat16, device=device_type
            )
            v = torch.randn(
                bsz, 16, seqlen, 64, dtype=torch.bfloat16, device=device_type
            )
            mask = torch.ones(
                [bsz, 1, seqlen, seqlen], dtype=torch.bool, device=device_type
            )
            inputs = [q, k, v, mask]

            def f(q, k, v, mask):
                # [FUNCTION DOC] HELPER: Scaled dot product attention restricted to the efficient
                # attention backend with the given boolean mask and no dropout.
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    return F.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask, dropout_p=0.0
                    )

            f_compiled = torch.compile(f)

            out, code = run_and_get_code(f_compiled, *inputs)
            # padded bias should have an expanded dim
            FileCheck().check("buf0 =").check_same(", 0, ").run(code[0])
            # single fused padded kernel
            FileCheck().check_count("empty_strided_cuda(", 1, exactly=True).check(
                "return"
            ).run(code[0])

            self.assertEqual(out, f(*inputs))

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Does not support mem_eff_attention",
    )
    def test_mha_mem_eff_attention_backward_large_batch_compile(self):
        # [FUNCTION DOC] Regression test for compiled MultiheadAttention backward with a very
        # large batch (65536) and a key padding mask using fp16 on GPU. Compares outputs and the
        # input gradients of the compiled and eager modules, with looser tolerances on XPU.
        # Skipped when memory-efficient attention is not supported.
        torch.manual_seed(0)

        eager = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=1,
            batch_first=True,
            device=device_type,
            dtype=torch.float16,
        )
        compiled = copy.deepcopy(eager)
        compiled = torch.compile(compiled)

        q = torch.rand(2**16, 1, 8, device=device_type, dtype=torch.float16)
        kv = torch.rand(2**16, 2, 8, device=device_type, dtype=torch.float16)
        mask = torch.randint(0, 2, (2**16, 2), device=device_type, dtype=torch.bool)

        eager_q = q.detach().clone().requires_grad_(True)
        eager_kv = kv.detach().clone().requires_grad_(True)
        compiled_q = q.detach().clone().requires_grad_(True)
        compiled_kv = kv.detach().clone().requires_grad_(True)

        eager_out = eager(
            query=eager_q,
            key=eager_kv,
            value=eager_kv,
            key_padding_mask=mask,
            attn_mask=None,
            need_weights=False,
        )[0]
        eager_out.sum().backward()

        compiled_out = compiled(
            query=compiled_q,
            key=compiled_kv,
            value=compiled_kv,
            key_padding_mask=mask,
            attn_mask=None,
            need_weights=False,
        )[0]
        compiled_out.sum().backward()

        tol_kwargs = {}
        if device_type == "xpu":
            # XPU fp16 SDPA compile-vs-eager can drift ~1% relative.
            tol_kwargs = {"atol": 1e-4, "rtol": 1e-2}

        self.assertEqual(compiled_out, eager_out, **tol_kwargs)
        self.assertEqual(compiled_q.grad, eager_q.grad, **tol_kwargs)
        self.assertEqual(compiled_kv.grad, eager_kv.grad, **tol_kwargs)

    def test_input_channels_last(self):
        # [FUNCTION DOC] Check a conv model with a channels_last input: first via the common
        # checker, then verify that a compiled call returns an output that is still channels_last
        # contiguous.
        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1, 1),
            ToTuple(),
        ).to(device_type)
        inp = (
            torch.randn([2, 3, 16, 16])
            .to(memory_format=torch.channels_last)
            .to(device_type)
        )

        self.common(
            m,
            (inp,),
            check_lowp=False,
        )

        @torch.compile()
        def foo(m, inp):
            # [FUNCTION DOC] HELPER: Compiled wrapper that just calls the model on the input.
            return m(inp)

        self.assertTrue(foo(m, inp)[0].is_contiguous(memory_format=torch.channels_last))

    # https://github.com/pytorch/torchdynamo/issues/1681#issuecomment-1283433527
    def test_unspec_inputs_interop(self):
        # [FUNCTION DOC] Regression test (torchdynamo issue 1681) for mixing a GPU tensor with odd
        # strides and a CPU zero-dim int64 input: traces a small aten module with make_fx,
        # compiles it with compile_fx_inner and runs it. Passes if no exception is raised.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a permuted unsqueezed view of the first input
            # and the second input plus one.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Unsqueeze x at dim 4 and permute to move the last two
                # dims, add 1 to y, and return both in a list.
                unsqueeze = torch.ops.aten.unsqueeze.default(x, 4)
                permute = torch.ops.aten.permute.default(unsqueeze, [0, 1, 2, 4, 3])
                add = torch.ops.aten.add.Tensor(y, 1)
                return [permute, add]

        inps = [
            rand_strided(
                (12, 3, 512, 64), (64, 196608, 768, 1), torch.float32, device_type
            ),
            rand_strided((), (), torch.int64, "cpu"),
        ]
        mod = make_fx(Repro().to(device=device_type))(*inps)
        compiled = compile_fx_inner(mod, inps)
        compiled(inps)

    @unittest.skipIf(
        IS_FBCODE, "RuntimeError: Triton Error [CUDA]: invalid device context"
    )
    def test_backward_context(self):
        # [FUNCTION DOC] Compile a trivial multiply and run backward through it, checking that the
        # backward pass works with the device context set up by inductor. Skipped in fbcode
        # because of an invalid device context error.
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiply the input by 3.
            return x * 3

        x = torch.randn(4, device=device_type, requires_grad=True)
        gO = torch.rand_like(x)
        opt_fn = torch.compile(fn)
        out = opt_fn(x)
        out.backward(gO)

    @config.patch(fallback_random=True)
    def test_dtype_factory_issue(self):
        # [FUNCTION DOC] Regression test with fallback_random enabled: traces a graph that creates
        # a random normal tensor with explicit device and compiles it with compile_fx_inner.
        # Raises AssertionError if the output is not on the expected device type.
        def forward():
            # [FUNCTION DOC] HELPER: Create a normal random tensor of fixed shape on the device
            # and return it unsqueezed on the last dim in a tuple.
            randn = torch.ops.aten.randn.default(
                [12, 64, 1, 64],
                dtype=torch.float32,
                device=torch.device(type=device_type, index=0),
                pin_memory=False,
            )
            unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(randn, -1)
            return (unsqueeze_default_2,)

        mod = make_fx(forward)()
        compiled = compile_fx_inner(mod, ())
        if compiled([])[0].device.type != device_type:
            raise AssertionError(
                f"Expected device type {device_type}, got {compiled([])[0].device.type!r}"
            )

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_no_device_idx_repro_cudagraphs(self):
        # [FUNCTION DOC] Regression test for cudagraphs when output creation ops carry an explicit
        # device index: runs the Repro module through the common checker with triton cudagraphs
        # and automatic dynamic shapes enabled.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no inputs that returns two constant filled tensors,
            # an int64 zeros and a float32 ones, of shape 8 by 512 on the device.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Plain parent initialization, no state.
                super().__init__()

            def forward(self):
                # [FUNCTION DOC] HELPER: Return a tuple of the int64 zeros tensor followed by the
                # float32 ones tensor, both created with explicit device index 0.
                full = torch.ops.aten.full.default(
                    [8, 512],
                    1,
                    dtype=torch.float32,
                    layout=torch.strided,
                    device=torch.device(type=device_type, index=0),
                    pin_memory=False,
                )
                full_1 = torch.ops.aten.full.default(
                    [8, 512],
                    0,
                    dtype=torch.int64,
                    layout=torch.strided,
                    device=torch.device(type=device_type, index=0),
                    pin_memory=False,
                )
                return (full_1, full)

        self.common(Repro(), ())

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_expanded_inputs_cudagraphs(self):
        # [FUNCTION DOC] Check that compiled addition of two expanded (stride 0) inputs under
        # triton cudagraphs gives the same result as eager.
        @torch.compile(backend="inductor")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Add the two inputs, compiled with the inductor backend.
            return x + y

        inputs = (
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device=device_type),
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device=device_type),
        )
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(
        automatic_dynamic_shapes=True,
        assume_static_by_default=False,
    )
    def test_dynamic_to_static_cudagraphs(self):
        # [FUNCTION DOC] With cudagraphs enabled and dynamic shapes, loop over cudagraph_trees off
        # and on, compile a function returning a sum and its first dim size, and call it with 5 by
        # 5 then 6 by 6 inputs; checks both results match eager, including the integer size.
        for b in [False, True]:
            with config.patch({"triton.cudagraph_trees": b}):

                @torch.compile(backend="inductor")
                def fn(x, y):
                    # [FUNCTION DOC] HELPER: Return the sum of x and y together with the size of
                    # its first dimension.
                    r = x + y
                    return r, r.size(0)

                inputs = (
                    torch.randn((5, 5), device=device_type),
                    torch.randn((5, 5), device=device_type),
                )
                self.assertTrue(same(fn(*inputs), (inputs[0] + inputs[1], 5)))

                inputs = (
                    torch.randn((6, 6), device=device_type),
                    torch.randn((6, 6), device=device_type),
                )
                self.assertTrue(same(fn(*inputs), (inputs[0] + inputs[1], 6)))

    def _test_split_reduction_impl(self, x):
        # [FUNCTION DOC] HELPER: Shared body for the split reduction tests: compile a global max
        # reduction, run it on the input collecting the generated code, and check against eager.
        # If DO_PERF_TEST is set, also benchmark compiled versus eager on GPU and print the
        # timings.
        def max(x):
            # [FUNCTION DOC] HELPER: Global maximum of the input.
            return torch.max(x)

        max_c = torch.compile(max)

        out, code = run_and_get_code(max_c, x)
        self.assertEqual(out, max(x))

        if DO_PERF_TEST:
            ms_c = benchmarker.benchmark_gpu(lambda: max_c(x))
            ms_eager = benchmarker.benchmark_gpu(lambda: max(x))
            print(f"compile {ms_c=:.03f}, eager {ms_eager=:.03f}")

    def test_split_reduction_transposed(self):
        # [FUNCTION DOC] Run the split reduction check on a large bf16 tensor whose layout is
        # transposed (a contiguous transpose of a contiguous transpose).
        x = torch.randn(4096, 8192, dtype=torch.bfloat16, device=device_type)
        x = x.t().contiguous().t()

        self._test_split_reduction_impl(x)

    def test_split_reduction_channels_last(self):
        # [FUNCTION DOC] Run the split reduction check on a large bf16 tensor reshaped to 4 dims
        # and converted to channels_last memory format.
        x = torch.randn(4096, 8192, dtype=torch.bfloat16, device=device_type)
        x = x.reshape([256, 256, 256, 2]).to(memory_format=torch.channels_last)

        self._test_split_reduction_impl(x)

    @tf32_on_and_off(0.001)
    def test_split_with_sizes_reshape_cat_cantsplit_regression(self):
        # [FUNCTION DOC] Regression test for a split_with_sizes followed by reshapes and a cat,
        # where the split could not be done as views. Compiles a module with embedding, linear and
        # conv under no_grad and compares against eager. Run with tf32 both on and off using a
        # tolerance of 0.001.
        class Repro(nn.Module):
            # [CLASS DOC] HELPER: Module combining an embedding lookup, a linear, and a conv. The
            # last time step of the linear output is concatenated with the flattened conv output,
            # split into three chunks of widths 32, 64 and the rest, each reshaped to 3D, and
            # concatenated along dim 1.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the embedding (128 by 32), linear (32 to 96) and
                # conv (3 to 16 channels, 3 by 3, padding 1) layers.
                super().__init__()
                self.embedding = nn.Embedding(num_embeddings=128, embedding_dim=32)
                self.linear = nn.Linear(32, 96)
                self.conv = nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x, indices):
                # [FUNCTION DOC] HELPER: Compute the combined tensor, split it with
                # split_with_sizes along dim 1, reshape the three parts to batch by k by 8, and
                # return their concatenation along dim 1.
                embedded = self.embedding(indices)
                linear_out = self.linear(embedded)
                conv_out = self.conv(x)
                batch_size = conv_out.shape[0]
                conv_flat = conv_out.view(batch_size, -1)
                seq_out = linear_out[:, -1, :]
                combined = torch.cat([seq_out, conv_flat], dim=1)
                chunks = torch.ops.aten.split_with_sizes.default(
                    combined, split_sizes=[32, 64, combined.size(1) - 96], dim=1
                )
                chunk0_reshaped = torch.ops.aten.reshape.default(
                    chunks[0], (batch_size, 4, 8)
                )
                chunk1_reshaped = torch.ops.aten.reshape.default(
                    chunks[1], (batch_size, 8, 8)
                )
                chunk2_reshaped = torch.ops.aten.reshape.default(
                    chunks[2], (batch_size, -1, 8)
                )
                return torch.ops.aten.cat.default(
                    [chunk0_reshaped, chunk1_reshaped, chunk2_reshaped], dim=1
                )

        model = Repro().to(device_type)
        x = torch.randn(2, 3, 32, 32, dtype=torch.float32, device=device_type)
        indices = torch.randint(0, 128, (2, 10), dtype=torch.long, device=device_type)

        with torch.no_grad():
            eager_out = model(x, indices)
            compiled_out = torch.compile(model)(x, indices)

        self.assertEqual(compiled_out, eager_out)

    @config.patch({"emulate_precision_casts": True})
    def test_bool_emulate_low_precision(self):
        # [FUNCTION DOC] Regression test with emulate_precision_casts: builds a boolean mask on
        # the GPU from a CPU float tensor, expands it to a large square and uses it in a where
        # between float16 zero and negative infinity; asserts compiled output equals eager. Note
        # it also assigns the config option directly on the inductor config module (not restored
        # afterwards) in addition to the decorator patch.
        # KNOWN ISSUE: the direct assignment to the config module leaks into later tests because
        #   it is not undone, only the decorator patch is restored.
        from torch import device

        inf = float("inf")

        def forward():
            # [FUNCTION DOC] HELPER: Graph body without inputs: make a ones float32 CPU tensor,
            # move it to the device, cast to bool, unsqueeze and expand to 1536 by 1536, then use
            # where to choose 0.0 where true and negative infinity elsewhere in float16.
            full_1 = torch.ops.aten.full.default(
                [6, 6],
                1,
                dtype=torch.float32,
                layout=torch.strided,
                device=device(type="cpu"),
                pin_memory=False,
            )
            device_put_3 = torch.ops.prims.device_put.default(
                full_1, device(type=device_type, index=0)
            )
            full_1 = None

            convert_element_type_40 = torch.ops.prims.convert_element_type.default(
                device_put_3, torch.bool
            )
            device_put_3 = None
            unsqueeze_4 = torch.ops.aten.unsqueeze.default(convert_element_type_40, 1)
            convert_element_type_40 = None
            unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3)
            unsqueeze_4 = None
            expand = torch.ops.aten.expand.default(unsqueeze_5, [-1, 256, -1, 256])
            unsqueeze_5 = None
            clone = torch.ops.aten.clone.default(
                expand, memory_format=torch.contiguous_format
            )
            expand = None
            view_15 = torch.ops.aten.reshape.default(clone, [1536, 1536])
            clone = None
            scalar_tensor = torch.ops.aten.scalar_tensor.default(
                -inf, dtype=torch.float16, device=device(type=device_type, index=0)
            )
            scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(
                0.0,
                dtype=torch.float16,
                layout=torch.strided,
                device=device(type=device_type, index=0),
            )
            where = torch.ops.aten.where.self(view_15, scalar_tensor_1, scalar_tensor)
            view_15 = scalar_tensor_1 = scalar_tensor = None
            return where

        from torch._inductor import config

        config.emulate_precision_casts = True
        self.assertEqual(torch.compile(forward)(), forward())

    @config.patch({"emulate_precision_casts": True})
    def test_emulate_low_precision(self):
        # [FUNCTION DOC] With emulate_precision_casts enabled, compile gelu times ten in bf16
        # forward and backward, and check each generated kernel (after stripping static asserts)
        # shows two down-cast then up-cast pairs matching eager's rounding, and the result equals
        # eager.
        def foo(x):
            # [FUNCTION DOC] HELPER: GELU of the input multiplied by 10.0.
            return torch.nn.functional.gelu(x) * 10.0

        inp = torch.rand(
            [32], device=device_type, requires_grad=True, dtype=torch.bfloat16
        )
        out, codes = run_fw_bw_and_get_code(lambda: torch.compile(foo)(inp))

        # fwd, backward
        for code in codes:
            code = _strip_triton_static_asserts(code)
            f = FileCheck()
            # in eager, there are two down casts
            for _ in range(2):
                f.check(".to(tl.bfloat16)").check_next(".to(tl.float32)")
            f.run(code)

        self.assertEqual(foo(inp), out)

    # TODO: Abstract this out, test more extensively
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic_shapes(self):
        # [FUNCTION DOC] Compile a cos, view, sin function with dynamic shapes on and check that
        # calling with sizes 32 then 16 compiles only once (frame count 1) and the output matches
        # eager. Resets dynamo before and after.
        torch._dynamo.reset()  # Needed since everywhere else uses "inductor"

        def f(x):
            # [FUNCTION DOC] HELPER: Cosine, reshape to the same shape, then sine.
            return x.cos().view(x.shape).sin()

        cnts = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        f2 = torch.compile(f, backend=cnts)

        f2(torch.randn(32))

        inp = torch.randn(16)
        real_out = f(inp)
        compiled_out = f2(inp)

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(real_out, compiled_out)
        torch._dynamo.reset()

    @config.patch({"triton.cudagraphs": True, "size_asserts": False})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_expanded_inputs_cudagraphs_no_size_asserts(self):
        # [FUNCTION DOC] Same as the expanded inputs cudagraph test but with size_asserts disabled
        # in the inductor config.
        @torch.compile(backend="inductor")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Add the two inputs, compiled with the inductor backend.
            return x + y

        inputs = (
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device=device_type),
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device=device_type),
        )
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    @config.patch({"triton.cudagraph_trees": False})
    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_inplace_updates_cudagraphs(self):
        # [FUNCTION DOC] With cudagraph trees disabled and plain cudagraphs enabled, train a
        # linear-weight module for two iterations comparing the gradients of an eager copy and a
        # compiled one, and after each step add 1 to every parameter of both so parameter updates
        # are visible to the graph.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one parameter of shape 10 by 20 that matrix-
            # multiplies the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the weight parameter with random values.
                super().__init__()
                self.weight1 = torch.nn.Parameter(
                    torch.randn(10, 20, requires_grad=True)
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Matrix multiply the input by the weight and return it.
                x = torch.matmul(x, self.weight1)
                return x

        from copy import deepcopy

        model = Repro().to(device_type)
        model_ref = deepcopy(model)
        model_opt = torch.compile(model, backend="inductor")

        input = torch.randn(10, 10, device=device_type, requires_grad=True)

        for _ in range(2):
            output_ref = model_ref(input)
            output_res = model_opt(input)
            output_ref.sum().backward()
            output_res.sum().backward()
            for p_ref, p_res in zip(model_ref.parameters(), model_opt.parameters()):
                self.assertEqual(p_ref.grad, p_res.grad)
            with torch.no_grad():
                for param in model_ref.parameters():
                    param.add_(1.0)
                for param in model_opt.parameters():
                    param.add_(1.0)

    # https://github.com/pytorch/torchdynamo/issues/1850
    def test_inductor_output_aliases_intermediate(self):
        # [FUNCTION DOC] Placeholder for torchdynamo issue 1850 (output aliasing an intermediate).
        # The compiled path is commented out as broken; the test only runs eager and mutates its
        # output, so it effectively tests nothing about inductor.
        # KNOWN ISSUE: the compiled call and comparison are commented out, so foo_opt is created
        #   but never called.
        def foo(x):
            # [FUNCTION DOC] HELPER: Return the transpose of the input added to itself.
            out = x + x
            return out.t()

        foo_opt = torch.compile(foo, backend="inductor")

        inpt = torch.randn(10, 10, device=device_type, requires_grad=True)
        # TODO: this is broken, fix later
        # out = foo_opt(inpt)
        # out.add_(2)

        out_ref = foo(inpt)
        out_ref.add_(2)
        # self.assertEqual(out_ref, out)

    def test_accuracy_issue1(self):
        # [FUNCTION DOC] Accuracy repro: a linear plus split plus cross entropy model with an
        # ignore index of 128 is compared between eager and compiled using same_two_models under
        # autocast disabled. Raises AssertionError if the models differ.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear (768 to 2, with bias) whose first output
            # column is used as logits for a cross entropy loss against clamped start positions.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_features=768, out_features=2, bias=True
                )

            def forward(self, start_positions: torch.Tensor, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Apply the linear to x, take the first column, clamp the
                # positions to the range 0 to 128, and return the mean cross entropy with ignore
                # index 128.
                linear = self.linear(x)
                split = linear.split(1, dim=-1)
                getitem = split[0]
                squeeze = getitem.squeeze(-1)
                clamp = start_positions.clamp(0, 128)
                cross_entropy = torch.nn.functional.cross_entropy(
                    squeeze, clamp, None, None, 128, None, "mean", 0.0
                )
                return cross_entropy

        mod = Repro().to(device_type)
        opt_mod = torch.compile(mod, backend="inductor")
        mod.eval()
        opt_mod.eval()

        args = [
            ((1,), (1,), torch.int64, device_type, False),
            ((1, 128, 768), (98304, 768, 1), torch.float32, device_type, True),
        ]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        with torch.cuda.amp.autocast(enabled=False):
            if not same_two_models(mod, opt_mod, args):
                raise AssertionError("Dynamo failed")

    @config.patch(allow_buffer_reuse=False)
    def test_issue103461(self):
        # [FUNCTION DOC] Regression test (issue 103461) with buffer reuse disabled: variance and
        # mean over dim 2 with keepdim, returning only the mean, compiled with fullgraph must
        # equal eager.
        def forward(add_1):
            # [FUNCTION DOC] HELPER: Return the mean from var_mean with correction 0 over dim 2,
            # keeping the dim.
            var_mean = torch.ops.aten.var_mean.correction(
                add_1, [2], correction=0, keepdim=True
            )
            getitem_1 = var_mean[1]
            return getitem_1

        x = torch.randn(1, 8, 768, device=device_type)
        correct = forward(x)
        actual = torch.compile(forward, fullgraph=True)(x)
        self.assertEqual(actual, correct)

    def test_full_copy(self):
        # [FUNCTION DOC] Compile a function that adds a CPU float64 tensor to a GPU-created zeros
        # tensor copied to the CPU, and check it matches eager (tests a full node being copied
        # across devices).
        def forward(x):
            # [FUNCTION DOC] HELPER: Create a zeros float64 tensor on the device, move it to the
            # CPU and add it to x.
            full_10 = torch.ops.aten.full.default(
                [204, 204, 28],
                0,
                dtype=torch.float64,
                layout=torch.strided,
                device=device_type,
                pin_memory=False,
            )
            return x + full_10.to("cpu")

        o = torch.randn([204, 204, 28], dtype=torch.float64)
        correct = forward(o)
        actual = torch.compile(forward, fullgraph=True)(o)
        self.assertEqual(actual, correct)

    def test_autotune_inplace_kernel(self):
        # [FUNCTION DOC] Verify that autotuning an in-place Triton kernel does not corrupt its
        # in/out buffer when several configs are benchmarked. A CachingAutotuner is built manually
        # with two block-size configs and saving the cache forced off, the kernel is run on two
        # copies of the data, and the results are compared; raises AssertionError if they differ.
        # Picks the XPU or CUDA raw stream getter by availability.
        """
        This UT tests autotune on an inplace kernel. The autotune should not contaminate
        the input buffers when tuning with multiple configs. For more details, refer to
        https://github.com/triton-lang/triton/issues/781
        https://github.com/pytorch/torchdynamo/issues/1670
        """
        if TEST_XPU:
            from torch._C import _xpu_getCurrentRawStream as get_gpu_stream
        else:
            from torch._C import _cuda_getCurrentRawStream as get_gpu_stream

        from torch._inductor.runtime.hints import AttrsDescriptorWrapper, HeuristicType
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from torch._inductor.utils import triton_version_uses_attrs_dict

        def autotune(configs, meta):
            # [FUNCTION DOC] HELPER: Build a decorator factory that, given the configs and
            # metadata, wraps a Triton function in a CachingAutotuner with pointwise heuristic
            # type, mutated arg in_out_ptr0 and memory optimization on. Adds XBLOCK as a constexpr
            # to the signature when this Triton version puts constexprs there (mutates the meta
            # mapping).
            def decorator(fn):
                # [FUNCTION DOC] HELPER: Wrap the given Triton function in a CachingAutotuner
                # built from the enclosing configs and metadata and return it.
                if triton_version_uses_attrs_dict():
                    # Newer versions of Triton puts constexpr in signature
                    # Ref: https://github.com/pytorch/pytorch/pull/145051
                    meta["signature"]["XBLOCK"] = "constexpr"

                return CachingAutotuner(
                    # force autotune by setting save_cache_hook to False
                    fn,
                    triton_meta=meta,
                    configs=configs,
                    save_cache_hook=False,
                    mutated_arg_names=["in_out_ptr0"],
                    reset_to_zero_arg_names=[],
                    optimize_mem=True,
                    heuristic_type=HeuristicType.POINTWISE,
                    inductor_meta={"grid_type": "Grid1D"},
                )

            return decorator

        @autotune(
            configs=[
                triton.Config({"XBLOCK": 1}),
                triton.Config({"XBLOCK": 2}),
            ],
            meta={
                "signature": {
                    "in_out_ptr0": "*fp32",
                    "in_ptr0": "*fp32",
                    "xnumel": "i32",
                },
                "device": DeviceProperties.create(torch.device(device_type)),
                "configs": [
                    AttrsDescriptorWrapper(divisible_by_16=(0, 1), equal_to_1=())
                ],
                "constants": {},
            },
        )
        # [FUNCTION DOC] HELPER: Triton kernel that adds the second input to the first input in
        # place, element by element, with a mask for the tail; each program handles XBLOCK
        # elements.
        @triton.jit
        def kernel(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
            pid = tl.program_id(0)
            block_start = pid * XBLOCK
            offsets = block_start + tl.arange(0, XBLOCK)
            mask = offsets < xnumel
            x = tl.load(in_out_ptr0 + offsets, mask=mask, other=0.0)
            y = tl.load(in_ptr0 + offsets, mask=mask, other=0.0)
            output = x + y
            tl.store(in_out_ptr0 + offsets, output, mask=mask)

        xnumel = 384
        in0 = rand_strided((xnumel,), (1,), device=device_type, dtype=torch.float32)
        inout1 = rand_strided((xnumel,), (1,), device=device_type, dtype=torch.float32)
        inout2 = inout1.clone()

        stream0 = get_gpu_stream(0)
        kernel.run(inout1, in0, xnumel, stream=stream0)
        kernel.run(inout2, in0, xnumel, stream=stream0)

        if not same(inout1, inout2, tol=0.001, equal_nan=True):
            raise AssertionError("failed autotune with inplace kernel")

    def test_sort_stride_issue(self):
        # [FUNCTION DOC] Regression test (from detectron2 maskrcnn) for a false error in size
        # asserts when sorting a tensor with a zero stride; compiles a descending sort with
        # fullgraph and runs it on a rand_strided input.
        # This minified testcase comes from detectron2_maskrcnn_r_50_fpn
        # There was a false error from our size_assert code
        @torch.compile(fullgraph=True)
        def forward(pred_objectness_logits_3_: torch.Tensor):
            # [FUNCTION DOC] HELPER: Sort the input descending along dim 1 and return the sorted
            # values.
            sort_3 = pred_objectness_logits_3_.sort(descending=True, dim=1)
            getitem_12 = sort_3[0]
            return getitem_12

        args = [((1, 100), (0, 1), torch.float16, device_type, False)]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        result = forward(*args)
        if not same(result, torch.sort(args[0], descending=True, dim=1)[0]):
            raise AssertionError

    def test_scalar_triton_index(self):
        # [FUNCTION DOC] Regression test (torchdynamo issue 1515): indexing a float tensor with an
        # int64 index tensor of zeros used to produce bad Triton code that crashed Triton. Checks
        # the compiled result matches eager; raises AssertionError otherwise.
        # The indirect indexing via a scalar like below used to lead to
        # bad triton code that made triton segfault when compiling.
        # See https://github.com/pytorch/torchdynamo/issues/1515
        def fn(a):
            # [FUNCTION DOC] HELPER: Build 16 int64 zeros and use them to gather from the input,
            # returning a one-element tuple.
            zero = torch.zeros((16,), device=a.device, dtype=torch.int64)
            return (a[zero],)

        a = torch.randn((8,), dtype=torch.float32, device=device_type)

        fn_optimized = torch.compile(fn, backend="inductor")
        if not same(fn(a), fn_optimized(a)):
            raise AssertionError

    def test_indirect_indexing_dense_mask(self):
        # [FUNCTION DOC] Compile a graph using not-equal, sum, subtract, unsqueeze, gather,
        # squeeze and multiply on int64 zero inputs (indirect indexing with a dense mask) and
        # raise AssertionError if the result differs from eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Count non-one entries per row of x, subtract one, use that as
            # a gather index into x, and multiply the squeezed gathered value with y. Returns a
            # tuple.
            ne = torch.ops.aten.ne.Scalar(x, 1)
            sum_1 = torch.ops.aten.sum.dim_IntList(ne, [1])
            sub = torch.ops.aten.sub.Tensor(sum_1, 1)
            unsqueeze = torch.ops.aten.unsqueeze.default(sub, -1)
            gather = torch.ops.aten.gather.default(x, 1, unsqueeze)
            squeeze = torch.ops.aten.squeeze.default(gather)
            out = torch.ops.aten.multiply(y, squeeze)
            return (out,)

        a = torch.zeros((1, 128), dtype=torch.int64, device=device_type)
        b = torch.zeros((1, 128), dtype=torch.int64, device=device_type)

        fn_optimized = torch.compile(fn, backend="inductor")
        if not same(fn(a, b), fn_optimized(a, b)):
            raise AssertionError

    def test_simplify_dims(self):
        # [FUNCTION DOC] Run the common checker on a pointwise add of one to a strided slice of a
        # 5D tensor, exercising dimension simplification of non-contiguous inputs.
        def fn(a):
            # [FUNCTION DOC] HELPER: Add one to the input and return it in a tuple.
            return (a + 1,)

        self.common(
            fn, (torch.randn(2, 3, 10, 5, 6, device=device_type)[:, :, 2::2, :, :],)
        )

    @config.patch(permute_fusion=True)
    def test_permute_fusion(self):
        # [FUNCTION DOC] With permute_fusion enabled, compare eager and compiled outputs of a
        # module doing a permute, a reshape (unused) and a batched matmul on two large strided
        # float32 GPU tensors that require grad. Asserts the results are the same.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that permutes the first input and batch-matmuls it with
            # the second input; it also builds an unused reshape. Returns a one-element tuple.
            def forward(self, view, reshape_2):
                # [FUNCTION DOC] HELPER: Permute view to swap dims 1 and 2, build a reshape to
                # shape with 642 columns (not used in the result), and return the batched matmul
                # of the permuted tensor with reshape_2 in a tuple.
                permute = view.permute(0, 2, 1)
                view = None
                reshape = torch.reshape(permute, (-1, 642))
                bmm = torch.bmm(permute, reshape_2)
                return (bmm,)

        args = [
            ((1024, 642, 160), (102720, 160, 1), torch.float32, device_type, True),
            ((1024, 642, 20), (12840, 20, 1), torch.float32, device_type, True),
        ]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        mod = Repro()
        opt_mod = torch.compile(mod, backend="inductor")

        ref = mod(*args)
        res = opt_mod(*args)
        self.assertTrue(same(ref, res))

    @config.patch({"triton.autotune_pointwise": True})
    def test_inplace_add_alpha_autotune(self):
        # [FUNCTION DOC] With pointwise autotuning on, compile a traced in-place add with alpha
        # 0.55 into a zero tensor, where the other operand is channels_last, and check the in-
        # place result matches eager (raises AssertionError otherwise).
        def fn(x, y):
            # [FUNCTION DOC] HELPER: In-place add of y scaled by alpha 0.55 to x using the aten
            # overload; returns x in a tuple.
            aten.add_.Tensor(x, y, alpha=0.55)
            return (x,)

        x1 = torch.zeros(2, 3, 4, 10, device=device_type)
        x2 = torch.zeros(2, 3, 4, 10, device=device_type)
        x3 = torch.zeros(2, 3, 4, 10, device=device_type)
        y = torch.randn(2, 3, 4, 10, device=device_type).to(
            memory_format=torch.channels_last
        )
        fn_fx = make_fx(fn)(x1, y)
        fn_compiled = compile_fx_inner(fn_fx, [x1, y])
        fn(x2, y)
        fn_compiled([x3, y])
        if not same(x2, x3):
            raise AssertionError

    @config.patch({"triton.autotune_pointwise": True})
    def test_inplace_buffer_autotune(self):
        # [FUNCTION DOC] With pointwise autotuning on, run the common checker (low precision check
        # off) on a matmul whose result is unsqueezed twice and added to a channels_last tensor,
        # which exercises in-place buffer reuse under autotune.
        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Matmul x and y, add two leading unit dims, and add z.
            a = x @ y
            return a.unsqueeze(0).unsqueeze(0) + z

        x = torch.zeros(5, 5, device=device_type)
        y = torch.zeros(5, 5, device=device_type)
        z = torch.zeros(1, 1, 5, 5, device=device_type).to(
            memory_format=torch.channels_last
        )
        self.common(
            foo,
            (x, y, z),
            check_lowp=False,
        )

    @skipIfXpu(msg="TypeError, torch-xpu-ops: 3004")
    def test_memory_history_inductor(self):
        # [FUNCTION DOC] Check that with memory history recording on, the allocator snapshot taken
        # after running a compiled function contains the name of a Python function called inside
        # the compiled region (so stack traces are recorded). Recording is always turned off at
        # the end, and the cache is emptied first. Skipped on XPU.
        def called_inside_compile(x, w, b):
            # [FUNCTION DOC] HELPER: Affine map of x by w and b followed by sigmoid; used to
            # appear in the recorded stack trace.
            a = x @ w + b
            return torch.sigmoid(a)

        @torch.compile
        def fn(x, w, b):
            # [FUNCTION DOC] HELPER: Apply the inner helper twice in a row, compiled.
            x = called_inside_compile(x, w, b)
            return called_inside_compile(x, w, b)

        w = torch.rand(3, 3, device=device_type)
        b = torch.rand(3, device=device_type)
        x = torch.rand(3, device=device_type)

        def record_memory_history(value: bool):
            # [FUNCTION DOC] HELPER: Turn allocator memory history recording on or off using the
            # XPU module when XPU is available, otherwise the CUDA module.
            if torch.xpu.is_available():
                torch.xpu.memory._record_memory_history(value)
            else:
                torch.cuda.memory._record_memory_history(value)

        try:
            torch.accelerator.memory.empty_cache()
            record_memory_history(True)
            r = fn(x, w, b)
        finally:
            record_memory_history(False)
        if torch.xpu.is_available():
            snapshot = str(torch.xpu.memory._snapshot())
        else:
            snapshot = str(torch.cuda.memory._snapshot())
        self.assertTrue("called_inside_compile" in snapshot)

    def test_negative_arange_dynamic_shapes(self):
        # [FUNCTION DOC] Regression test from alibi relative position encodings: compiles with
        # dynamic shapes a model that builds an arange with a negative step from float bounds and
        # feeds a transformer decoder layer. Compares forward outputs for eight growing decoder
        # input lengths with same_two_models and raises AssertionError on any mismatch.
        # Repro from alibi relative encodings
        def sign(x):
            # [FUNCTION DOC] HELPER: Elementwise sign of a number or tensor, computed as the
            # difference of the greater-than-zero and less-than-zero booleans.
            return (x > 0) - (x < 0)

        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Decoder-style module with 16 heads. It holds a registered buffer
            # of per-head scales (a geometric progression built from an arange with a negative
            # step), an embedding of size 1024 by 256, a pre-norm transformer decoder layer and a
            # linear head. The forward builds a causal mask from padding and adds a relative-
            # position bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Compute the scales buffer from log2 bounds using a float
                # arange of step (end minus start) over (heads minus one) with a sign-adjusted
                # epsilon, shaped for 16 heads, and create the embedding, decoder layer and head.
                super().__init__()
                nheads = 16
                start = math.log2(0.5)
                end = math.log2(1 / (2**8))

                self.scales = nn.Buffer(
                    2
                    ** torch.arange(
                        start,
                        end + 1e-6 * sign(end - start),
                        (end - start) / (nheads - 1),
                    ).view(1, nheads, 1, 1),
                )
                self.emb = nn.Embedding(1024, 256)
                self.dec_layer = nn.TransformerDecoderLayer(
                    256, 16, 512, batch_first=True, norm_first=True
                )
                self.head = nn.Linear(256, 1024)

            def forward(self, enc_out: torch.Tensor, dec_in: torch.Tensor):
                # [FUNCTION DOC] HELPER: Build a lower-triangular padding mask from the token ids,
                # a relative position matrix from two aranges, scale it by the per-head scales and
                # make it lower triangular in place, add head 0 of the bias to the mask, then
                # embed, run the decoder layer with enc_out as memory and the mask as target mask,
                # and return the head's logits.
                padmask = dec_in == 0
                dec_mask = padmask.unsqueeze(-1) == padmask.unsqueeze(-2)
                dec_mask = dec_mask.to(dtype=torch.float32)
                dec_mask = dec_mask.tril(diagonal=0).to(device_type)

                q_pos = torch.arange(
                    dec_in.size(1), dtype=torch.long, device=device_type
                )
                k_pos = torch.arange(
                    dec_in.size(1), dtype=torch.long, device=device_type
                )
                rel_pos = k_pos[None, :] - q_pos[:, None]
                values = rel_pos.abs().neg().unsqueeze(0).unsqueeze(0)
                dec_bias = values * self.scales
                dec_bias.tril_(diagonal=0)

                dec_mask = dec_mask + dec_bias[0]
                out = self.emb(dec_in)
                out = self.dec_layer(out, enc_out, tgt_mask=dec_mask)
                return self.head(out)

        mod = Repro().to(device_type)
        opt_mod = torch.compile(mod, backend="inductor", dynamic=True)
        mod.eval()
        opt_mod.eval()

        enc_out = torch.rand(1, 512, 256).to(device_type)
        dec_inputs = [
            torch.randint(0, 512, (1, i + 1), dtype=torch.long).to(device_type)
            for i in range(8)
        ]

        for dec_inp in dec_inputs:
            if not same_two_models(mod, opt_mod, [enc_out, dec_inp], only_fwd=True):
                raise AssertionError("Inductor with dynamic shapes failed")

    def test_issue97695_1input(self):
        # [FUNCTION DOC] Regression test (issue 97695) for a concat of a single addmm result:
        # checks the traced and compiled graph via compile_fx_inner, and a fullgraph compile of
        # the original function, each match the eager result; raises AssertionError otherwise.
        def fn(arg3_1, relu, permute_1):
            # [FUNCTION DOC] HELPER: Compute an addmm and concatenate that single result along dim
            # 1; returns a one-element tuple.
            addmm_1 = torch.ops.aten.addmm.default(arg3_1, relu, permute_1)
            cat_2 = torch.ops.aten.cat.default([addmm_1], 1)
            return (cat_2,)

        args = [
            ((96,), (1,), torch.float32, device_type),
            ((10, 256), (256, 1), torch.float32, device_type),
            ((256, 96), (1, 256), torch.float32, device_type),
        ]
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
        correct = fn(*args)

        mod = make_fx(fn, tracing_mode="real")(*args)
        compiled = compile_fx_inner(mod, args)
        ref = compiled(list(args))
        if not same(ref, correct):
            raise AssertionError

        ref = torch.compile(fn, fullgraph=True)(*args)
        if not same(ref, correct):
            raise AssertionError

    def test_issue_103924(self):
        # [FUNCTION DOC] Regression test (issue 103924): a module that divides by a ones vector
        # with a trailing dim added and applies softmax along dim 1 must match eager when
        # compiled.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a softmax over dim 1 that divides its input rows by
            # a ones vector (one per sample) before the softmax. The temperature attribute is
            # stored but unused.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Store temperature 1 (unused) and create the softmax
                # layer.
                super().__init__()
                self.temperature = 1
                self.layer = torch.nn.Softmax(dim=1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Create a ones vector of float length n_samples with the
                # input's dtype and device, divide each row of x by it and return the softmax
                # along dim 1.
                n_samples, _ = x.shape
                y = 1.0 * torch.ones(n_samples, dtype=x.dtype, device=x.device)
                inp = x / y[..., None]
                return self.layer(inp)

        x = torch.rand([4, 4], device=device_type)
        m = MyModule()
        opt_m = torch.compile(backend="inductor")(m)
        self.assertEqual(opt_m(x), m(x))

    def test_issue97695_2input(self):
        # [FUNCTION DOC] Same as the single-input case but concatenating two addmm results; only a
        # fullgraph compile versus eager comparison is made, raising AssertionError on mismatch.
        def fn(arg3_1, arg3_2, relu, permute_1):
            # [FUNCTION DOC] HELPER: Compute two addmm results sharing the same matrix operands
            # and concatenate them along dim 1; returns a tuple.
            addmm_1 = torch.ops.aten.addmm.default(arg3_1, relu, permute_1)
            addmm_2 = torch.ops.aten.addmm.default(arg3_2, relu, permute_1)
            cat_2 = torch.ops.aten.cat.default([addmm_1, addmm_2], 1)
            return (cat_2,)

        args = [
            ((96,), (1,), torch.float32, device_type),
            ((96,), (1,), torch.float32, device_type),
            ((10, 256), (256, 1), torch.float32, device_type),
            ((256, 96), (1, 256), torch.float32, device_type),
        ]
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
        correct = fn(*args)

        ref = torch.compile(fn, fullgraph=True)(*args)
        if not same(ref, correct):
            raise AssertionError

    def test_scatter_index_not_wrapped(self):
        # [FUNCTION DOC] Compile scatter_reduce with sum reduction and check the generated code
        # has a device assertion that the index is within bounds (without wrapping negative
        # numbers) immediately followed by an atomic add, and that the result equals eager.
        src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=self.device)
        index = torch.tensor([0, 1, 0, 1, 2, 0], device=self.device)
        input = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
        compiled_sr = torch.compile(torch.scatter_reduce)

        input_orig = input.clone()
        out, code = run_and_get_code(compiled_sr, input, 0, index, src, "sum")
        # tmp0 - not wrapping of negative numbers
        FileCheck().check("tl.device_assert(((0 <= tmp0) & (tmp0 < 4))").check_next(
            "atomic_add"
        ).run(_strip_triton_static_asserts(code[0]))
        self.assertEqual(
            out, torch.scatter_reduce(input_orig.clone(), 0, index, src, "sum")
        )

    def test_normalize_norm_leq_one(self):
        # [FUNCTION DOC] Check that normalizing a vector whose norm is just below 4 gives an
        # output with norm no greater than one, when compiled.
        def fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Normalize the input along the last dim.
            return torch.nn.functional.normalize(x, dim=-1)

        inp = torch.tensor(
            [[3.799999, 0.0, 0.0]], device=device_type, dtype=torch.float32
        )
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        out = compiled(inp)
        norm = out.norm(dim=-1)
        self.assertTrue(
            torch.all(norm <= 1.0),
            lambda msg: f"{msg}\nexpected norm <= 1.0 but got {norm.item()}",
        )

    def test_libdevice_routing(self):
        # [FUNCTION DOC] Check which math functions inductor routes through libdevice in generated
        # code: exp for float64 and float32 (not the Triton math wrapper), log when eager-numerics
        # libdevice use is on, and sigmoid for float64 which uses exp. Each result is also
        # compared to eager.
        def foo(x):
            # [FUNCTION DOC] HELPER: Exponential of the input.
            return x.exp()

        inp = torch.ones(64, device=device_type).to(torch.float64)

        out, code = run_and_get_code(torch.compile(foo), inp)
        FileCheck().check("libdevice.exp").run(code[0])
        self.assertEqual(foo(inp), out)

        inp = inp.to(torch.float)
        out, code = run_and_get_code(torch.compile(foo), inp)
        FileCheck().check_not("tl_math.exp").check("libdevice.exp").run(code[0])
        self.assertEqual(foo(inp), out)

        def foo(x):
            # [FUNCTION DOC] HELPER: Natural logarithm of the input. This replaces the earlier
            # local helper of the same name.
            return x.log()

        inp = torch.ones(64, device=device_type, dtype=torch.float32)
        with config.patch({"eager_numerics.use_pytorch_libdevice": True}):
            out, code = run_and_get_code(torch.compile(foo), inp)
        FileCheck().check_not("tl_math.log").check("libdevice.log").run(code[0])
        self.assertEqual(foo(inp), out)

        def foo(x):
            # [FUNCTION DOC] HELPER: Sigmoid of the input. This replaces the earlier local helper
            # of the same name.
            return x.sigmoid()

        inp = torch.ones(64, device=device_type).to(torch.float64)
        out, code = run_and_get_code(torch.compile(foo), inp)
        FileCheck().check("libdevice.exp").run(code[0])
        self.assertEqual(foo(inp), out)

    def test_uint_view_copy(self):
        # [FUNCTION DOC] Check that a compiled in-place copy of a uint16 tensor into a bf16 tensor
        # viewed as uint16 matches the eager result.
        @torch.compile
        def view_copy(target, source):
            # [FUNCTION DOC] HELPER: Check the target is bf16 and source is uint16, then copy
            # source into the uint16 view of target (mutates target). The checks use plain assert
            # statements.
            assert target.dtype == torch.bfloat16  # noqa: S101
            assert source.dtype == torch.uint16  # noqa: S101
            target.view(torch.uint16).copy_(source)

        target = torch.ones(1024, dtype=torch.bfloat16, device=device_type)
        source = torch.full_like(target, 4, dtype=torch.uint16)

        out = target.view(torch.uint16).copy_(source).clone()
        view_copy(target, source)
        self.assertEqual(out, target.view(torch.uint16))

    def test_embedding_var_mean(self):
        # [FUNCTION DOC] Regression test: traces a graph with cumsum position ids, an embedding
        # lookup and variance and mean over the last dim, compiles it with compile_fx_inner, runs
        # it once and synchronizes the accelerator. Passes if nothing raises.
        def forward(arg0_1):
            # [FUNCTION DOC] HELPER: Build position ids from ones via cumsum and multiplication,
            # add 2, look up rows in the embedding table arg0_1, then return the variance, the
            # mean and the ids as a list.
            full = torch.ops.aten.full.default(
                [1, 2048],
                1,
                dtype=torch.float32,
                layout=torch.strided,
                device=torch.device(type=device_type, index=0),
                pin_memory=False,
            )
            convert_element_type_1 = torch.ops.prims.convert_element_type.default(
                full, torch.int64
            )
            cumsum = torch.ops.aten.cumsum.default(convert_element_type_1, 1)
            mul = torch.ops.aten.mul.Tensor(cumsum, convert_element_type_1)
            sub_1 = torch.ops.aten.sub.Tensor(mul, 1)
            slice_5 = torch.ops.aten.slice.Tensor(sub_1, 0, 0, 9223372036854775807)
            slice_6 = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807)
            add_2 = torch.ops.aten.add.Tensor(slice_6, 2)
            embedding_1 = torch.ops.aten.embedding.default(arg0_1, add_2)
            var_mean = torch.ops.aten.var_mean.correction(
                embedding_1, [2], correction=0, keepdim=True
            )
            return [var_mean[0], var_mean[1], add_2]

        emb = torch.randn([2050, 768], device=device_type)
        gm = make_fx(forward)(emb)
        opt = torch._inductor.compile_fx.compile_fx_inner(gm, [emb])
        opt([emb])
        torch.accelerator.synchronize()

    def test_deterministic_algorithms(self):
        # [FUNCTION DOC] Check that, under deterministic algorithms mode, an indexed in-place
        # accumulation of 10000 values into a single slot gives bitwise identical results over 11
        # repeated runs of the compiled function.
        N = 10000

        @torch.compile
        def fn(idx, values):
            # [FUNCTION DOC] HELPER: Make a one-element zeros tensor and add the values into it at
            # the given indices with an accumulating indexed assignment; returns it.
            x = torch.zeros(1, device=device_type)
            x[idx] += values
            return x

        idx = torch.zeros(N, dtype=torch.int64, device=device_type)
        values = torch.randn(N, device=device_type)

        r0 = fn(idx, values)
        with DeterministicGuard(True):
            r1 = fn(idx, values)
            for _ in range(10):
                rn = fn(idx, values)
                self.assertEqual(r1, rn, atol=0, rtol=0)

    # https://github.com/pytorch/pytorch/issues/96406
    def test_linear_cpu_input(self):
        # [FUNCTION DOC] Regression test (issue 96406): a linear model fed a CPU input that moves
        # it to the device inside forward passes the common checker under no_grad in eval mode.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Module with a single 4 by 4 linear layer that first moves its
            # input to the device.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the linear layer.
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, data):
                # [FUNCTION DOC] HELPER: Move the data to the device and apply the linear layer.
                data = data.to(device_type)
                return self.linear(data)

        mod = Model().to(device_type).eval()
        with torch.no_grad():
            self.common(mod, (torch.randn(4, 4),))

    @config.patch({"fallback_random": True, "triton.cudagraphs": True})
    def test_xlnet_lm_stride_repro(self):
        # [FUNCTION DOC] Regression test with fallback_random and cudagraphs: compile a gelu then
        # dropout module on a large 3D input and run backward. The comment notes inductor's
        # predicted layout for the saved activation differs from the real strides, which is
        # harmless. Passes if no exception is raised.
        class Repro(nn.Module):
            # [CLASS DOC] HELPER: Module applying a gelu followed by dropout with probability 0.1.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the dropout layer.
                super().__init__()
                self.dropout = nn.Dropout(p=0.1, inplace=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply gelu to x and then dropout.
                y = torch._C._nn.gelu(x)
                return self.dropout(y)

        mod = Repro()
        x = torch.randn((512, 1, 4096), requires_grad=True, device=device_type)
        y = torch.compile(mod)(x)
        # Inductor claims the output layout of gelu's saved variable for
        # backwards will be (4096, 4096, 1) but in actuality it is (4096,
        # 2097152, 1).  Fortunately this doesn't actually matter in practice.
        y.sum().backward()

    def test_lookup_seed_backward(self):
        # [FUNCTION DOC] Run a compiled dropout-like graph that uses the inductor seed lookup and
        # random primitives, followed by layer-norm-like mean subtraction, on zero inputs. Passes
        # if it compiles and runs.
        @torch.compile(fullgraph=True)
        def forward(inductor_seeds, mul_4, view_15):
            # [FUNCTION DOC] HELPER: Look up seed number 2, generate a random tensor of shape 2 by
            # 512 by 768, mask by greater than 0.1, scale by about 1.111, add the third argument
            # and subtract the per-row mean over the last dim; returns a tuple.
            inductor_lookup_seed_2 = torch.ops.prims.inductor_lookup_seed.default(
                inductor_seeds, 2
            )
            inductor_random_2 = torch.ops.prims.inductor_random.default(
                [2, 512, 768], inductor_lookup_seed_2, "rand"
            )
            gt_2 = torch.ops.aten.gt.Scalar(inductor_random_2, 0.1)
            mul_7 = torch.ops.aten.mul.Tensor(gt_2, view_15)
            mul_8 = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112)
            add_5 = torch.ops.aten.add.Tensor(mul_8, mul_4)
            var_mean_1 = torch.ops.aten.var_mean.correction(
                add_5, [2], correction=0, keepdim=True
            )
            getitem_3 = var_mean_1[1]
            sub_3 = torch.ops.aten.sub.Tensor(add_5, getitem_3)
            return (sub_3,)

        buf0 = torch.zeros((37,), dtype=torch.int64, device=device_type)
        buf1 = torch.zeros((2, 512, 768), device=device_type)
        buf2 = torch.zeros((2, 512, 768), device=device_type)
        forward(buf0, buf1, buf2)

    def test_issue100806(self):
        # [FUNCTION DOC] Regression test (issue 100806): a model of two linears, a concat along
        # dim 1, a view, a slice and a relu is compiled under no_grad in eval mode and compared to
        # eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with linear layers 10 to 20 and 20 to 30 followed by a
            # duplicate concat, a view to groups of two rows, selection of the second group and
            # relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create two linear layers and a relu.
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 20)
                self.linear2 = torch.nn.Linear(20, 30)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply both linears, concatenate x with itself along dim
                # 1, view as rows of 2 by 30, take index 1 along the middle dim, and apply relu.
                x = self.linear1(x)
                x = self.linear2(x)
                x = torch.cat((x, x), dim=1)
                x = x.view(-1, 2, 30)
                x = x[:, 1, :]
                x = self.relu(x)
                return x

        device = device_type
        batch_size = 2
        x = torch.randn(batch_size, 10).to(device)
        func = Model().to(device)

        with torch.no_grad():
            func.train(False)
            jit_func = torch.compile(func)

            res1 = func(x)
            res2 = jit_func(x)
            self.assertEqual(res1, res2)

    def test_issue103481(self):
        # [FUNCTION DOC] Regression test (issue 103481) for a mean over the last four dims of a 6D
        # tensor with keepdim, added to a zero-dim tensor; compiled result must equal eager. The
        # note says it fails only with six dims.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Mean over dims 2 to 5 keeping dims, plus y.
            # NOTE: 6 dimensions is important! does not fail for 5 dimensions
            mean = torch.mean(x, [2, 3, 4, 5], keepdim=True)
            add = mean + y
            return add

        x = torch.rand(4, 4, 4, 4, 4, 4, device=device_type)
        y = torch.rand((), device=device_type)
        expect = fn(x, y)

        opt_fn = torch.compile(fn)
        actual = opt_fn(x, y)

        self.assertEqual(expect, actual)

    @config.patch({"triton.dense_indexing": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_bucketize_dynamic_dense(self):
        # [FUNCTION DOC] Check bucketize works with dense indexing enabled and dynamic shapes
        # (formerly the offsets size was handled wrongly) by comparing to eager.
        """
        Make sure that ops.bucketize() can handle dense_indexing, which previously
        caused issues due to incorrect handling of the size of offsets.
        """

        def fn(values, offsets):
            # [FUNCTION DOC] HELPER: Bucketize values by the sorted boundaries.
            return torch.bucketize(values, offsets)

        values = torch.rand((64, 64), device=device_type)
        offsets = torch.tensor([0.05, 0.1, 0.5, 0.8, 0.85, 0.95], device=device_type)

        expect = fn(values, offsets)

        opt_fn = torch.compile(fn, dynamic=True)
        actual = opt_fn(values, offsets)

        self.assertEqual(expect, actual)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @config.patch(
        {
            "max_autotune_gemm_backends": "TRITON",
            "triton.disallow_failing_autotune_kernels_TESTING_ONLY": True,
            "compile_threads": 1,
        }
    )
    def test_bucketize_epilogue(self):
        # [FUNCTION DOC] Regression test (issue 148764): bucketize used as an epilogue of a matmul
        # with max-autotune on the Triton backend must generate valid code. Autotune failures are
        # made fatal and compile threads set to one so failures are not swallowed. Compares to
        # eager. Skipped without a big GPU.
        """
        See https://github.com/pytorch/pytorch/issues/148764.
        Make sure that when torch.bucketize appears as an epilogue, the codegen is valid.

        Note: during autotuning, there's also the option to _not_ do the fusion.
        So if you run the test with standard configs, the fused kernel would fail during
        autotuning, and another non-fused kernel would be selected (and Inductor would
        throw some errors, but the test would pass)

        So we set disallow_failing_autotune_kernels_TESTING_ONLY=True to prevent the
        autotuner from catching failures. And set compile_threads=1 so that compile
        failures aren't caught by the asyn runner infra.
        """

        def fn(x: torch.Tensor, y: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Matrix multiply x and y, then bucketize the result using the
            # buckets.
            z = torch.mm(x, y)
            return torch.bucketize(z, buckets)

        buckets = torch.arange(-100, 100, 10, device=device_type)
        x = torch.randn(64, 64, device=device_type).clamp(-99, 99)
        y = torch.randn(64, 64, device=device_type).clamp(-99, 99)

        opt_fn = torch.compile(fn, mode="max-autotune")

        expected = fn(x, y, buckets)
        actual = opt_fn(x, y, buckets)

        self.assertEqual(expected, actual)

    def test_float64_constants(self):
        # [FUNCTION DOC] Check that float64 constants (values 0.1 and 0.2 multiplied by a huge
        # constant) are emitted exactly: compiled result equals eager with zero tolerance.
        def fn():
            # [FUNCTION DOC] HELPER: Create a float64 tensor of two distinct values (distinct so
            # they are not constant folded) and multiply it by 2e50.
            # NOTE: tensors of all the same value are constant folded, so we
            # need a tensor with two distinct values
            a = torch.tensor([1 / 10, 2 / 10], dtype=torch.float64, device=device_type)
            return a * 2e50

        cfn = torch.compile(fn)
        expect = fn()
        actual = cfn()
        self.assertEqual(expect, actual, atol=0, rtol=0)

    def test_issue104759(self):
        # [FUNCTION DOC] Regression test (issue 104759): a traced graph of slice scatters,
        # permutes, views and a batched matmul on strided fp16 inputs is compiled with
        # compile_fx_inner and compared with eager; raises AssertionError if different.
        def fn(arg7_1, add_1, permute_2, select_scatter, slice_8):
            # [FUNCTION DOC] HELPER: Large aten graph body doing two slice_scatter operations,
            # several view and permute steps, scaling by about 0.7071 and a final batched matmul;
            # returns a one-element tuple.
            slice_scatter_4 = torch.ops.aten.slice_scatter.default(
                permute_2, select_scatter, 0, 1, 9223372036854775807
            )
            permute_3 = torch.ops.aten.permute.default(slice_scatter_4, [1, 3, 0, 2, 4])
            view_6 = torch.ops.aten.view.default(permute_3, [1, 1000, 48])
            view_7 = torch.ops.aten.view.default(view_6, [1000, 48])
            view_8 = torch.ops.aten.view.default(view_7, [1, 1000, 48])
            view_9 = torch.ops.aten.view.default(view_8, [1, 1000, 3, 4, 4])
            permute_4 = torch.ops.aten.permute.default(view_9, [2, 0, 3, 1, 4])
            slice_7 = torch.ops.aten.slice.Tensor(permute_4, 0, 1, 9223372036854775807)
            slice_scatter_5 = torch.ops.aten.slice_scatter.default(
                slice_8, slice_7, 4, 0, 9223372036854775807
            )
            slice_scatter_6 = torch.ops.aten.slice_scatter.default(
                arg7_1, slice_scatter_5, 3, 0, 1000
            )
            mul_8 = torch.ops.aten.mul.Scalar(add_1, 0.7071067811865476)
            slice_9 = torch.ops.aten.slice.Tensor(slice_scatter_6, 3, 0, 1000)
            slice_10 = torch.ops.aten.slice.Tensor(slice_9, 4, 0, 9223372036854775807)
            select_2 = torch.ops.aten.select.int(slice_10, 0, 0)
            permute_5 = torch.ops.aten.permute.default(select_2, [0, 1, 3, 2])
            mul_9 = torch.ops.aten.mul.Scalar(permute_5, 0.7071067811865476)
            expand = torch.ops.aten.expand.default(mul_8, [1, 4, 1000, 4])
            view_10 = torch.ops.aten.view.default(expand, [4, 1000, 4])
            expand_1 = torch.ops.aten.expand.default(mul_9, [1, 4, 4, 1000])
            view_11 = torch.ops.aten.view.default(expand_1, [4, 4, 1000])
            bmm = torch.ops.aten.bmm.default(view_10, view_11)
            return (bmm,)

        args = []
        args.append(
            torch.randn((2, 1, 4, 1200, 4), dtype=torch.float16, device=device_type)
        )
        args.append(
            rand_strided(
                (1, 4, 1000, 4),
                (16000, 4, 16, 1),
                dtype=torch.float16,
                device=device_type,
            )
        )
        args.append(
            rand_strided(
                (3, 1, 4, 1000, 4),
                (16, 48000, 4, 48, 1),
                dtype=torch.float16,
                device=device_type,
            )
        )
        args.append(
            rand_strided(
                (2, 1, 4, 1000, 4),
                (16, 48000, 4, 48, 1),
                dtype=torch.float16,
                device=device_type,
            )
        )
        args.append(
            rand_strided(
                (2, 1, 4, 1000, 4),
                (19200, 19200, 4800, 4, 1),
                dtype=torch.float16,
                device=device_type,
            )
        )

        correct = fn(*args)
        mod = make_fx(fn, tracing_mode="real")(*args)
        compiled = compile_fx_inner(mod, args)
        ref = compiled(list(args))
        if not same(ref, correct):
            raise AssertionError

    @config.patch({"triton.cudagraphs": True})
    def test_index_put_inplace_cudagraph(self):
        # [FUNCTION DOC] With triton cudagraphs enabled, compile an in-place index_put on a
        # boolean tensor and call it twice to catch cudagraph replay problems, then compare with
        # eager.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Replace x with zeros of the same shape, then do an in-place
            # accumulating index_put at the given indices with values z and return it.
            x = torch.zeros_like(x)
            return x.index_put_([y], z, True)

        x = torch.zeros((512, 512), device=device_type, dtype=torch.bool)
        y = torch.zeros((512,), device=device_type, dtype=torch.int64)
        z = torch.ones((512, 512), device=device_type, dtype=torch.bool)

        opt_fn = torch.compile(fn, backend="inductor")

        ref = fn(x, y, z)

        # run it twice to test cuda graph issue
        res = opt_fn(x, y, z)
        res = opt_fn(x, y, z)

        self.assertEqual(ref, res)

    @config.patch({"triton.cudagraphs": True})
    @config.patch({"fx_graph_cache": True})
    def test_index_put_cudagraph(self):
        # [FUNCTION DOC] Same as the in-place variant but with the out-of-place index_put, with fx
        # graph caching on. It repeats twice, resetting dynamo and collecting garbage in between
        # so the second iteration tests the cache path.
        for _ in range(2):

            def fn(x, y, z):
                # [FUNCTION DOC] HELPER: Replace x with zeros of the same shape and return an
                # accumulating out-of-place index_put at indices y with values z.
                x = torch.zeros_like(x)
                return x.index_put([y], z, True)

            x = torch.zeros((512, 512), device=device_type, dtype=torch.bool)
            y = torch.zeros((512,), device=device_type, dtype=torch.int64)
            z = torch.ones((512, 512), device=device_type, dtype=torch.bool)

            opt_fn = torch.compile(fn, backend="inductor")

            ref = fn(x, y, z)

            # run it twice to test cuda graph issue
            res = opt_fn(x, y, z)
            res = opt_fn(x, y, z)

            self.assertEqual(ref, res)
            torch._dynamo.reset()
            gc.collect()

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "flash attention not supported"
    )
    def test_flash_attention_dynamic(self):
        # [FUNCTION DOC] Check that compiling an attention model with dynamic shapes and the flash
        # backend yields only two frames across three different sequence lengths (512, 513, 514),
        # i.e. a recompile once to become dynamic and no more. Skipped if flash attention is
        # unsupported.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Module with three 1024 by 1024 linear layers producing queries,
            # keys and values reshaped to 8 heads of 128, then computes scaled dot product
            # attention.
            def __init__(self, *args, **kwargs) -> None:
                # [FUNCTION DOC] HELPER: Pass extra arguments to the parent initializer and create
                # the three projection layers.
                super().__init__(*args, **kwargs)

                self.q = nn.Linear(1024, 1024)
                self.k = nn.Linear(1024, 1024)
                self.v = nn.Linear(1024, 1024)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Project x to queries, keys and values, split to 8 heads,
                # move heads before the sequence dim and return their scaled dot product
                # attention.
                batch_size, seq_len, _ = x.size()

                queries = self.q(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
                keys = self.k(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
                values = self.v(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)

                attn = F.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                )

                return attn

        cnts = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        model = Model().to(device_type).half()
        model = torch.compile(model, backend=cnts, dynamic=True)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            input1 = torch.rand(5, 512, 1024, device=device_type, dtype=torch.float16)
            input2 = torch.rand(5, 513, 1024, device=device_type, dtype=torch.float16)
            input3 = torch.rand(5, 514, 1024, device=device_type, dtype=torch.float16)

            out1 = model(input1)
            out2 = model(input2)
            out3 = model(input3)

        self.assertEqual(cnts.frame_count, 2)

    @config.patch({"triton.cudagraphs": True})
    def test_index_put_no_fallback_cudagraph(self):
        # [FUNCTION DOC] Like the other cudagraph index_put tests but with int32 data: compile,
        # call twice, and compare to eager (checks there is no fallback to a non-Triton kernel).
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Replace x with zeros of the same shape and return an
            # accumulating out-of-place index_put.
            x = torch.zeros_like(x)
            return x.index_put([y], z, True)

        x = torch.zeros((512, 512), device=device_type, dtype=torch.int32)
        y = torch.zeros((512,), device=device_type, dtype=torch.int64)
        z = torch.ones((512, 512), device=device_type, dtype=torch.int32)

        opt_fn = torch.compile(fn, backend="inductor")

        ref = fn(x, y, z)

        # run it twice to test cuda graph issue
        res = opt_fn(x, y, z)
        res = opt_fn(x, y, z)

        self.assertEqual(ref, res)

    @parametrize("lowp_dtype", [torch.bfloat16, torch.float16])
    @unittest.skipIf(not (TEST_CUDA or TEST_XPU), "requires CUDA or XPU")
    @config.patch(
        emulate_precision_casts=False,
        emulate_precision_casts_on_saved_tensors=True,
    )
    def test_saved_lowp_checkpoint_truncates_subsequent_uses(self, lowp_dtype):
        # [FUNCTION DOC] Parametrized over bf16 and fp16. With emulation of casts on saved tensors
        # on and general emulation off, a selective-checkpoint function that must save the sigmoid
        # output must be bitwise equal to eager and the generated code must contain a down cast to
        # the low precision dtype (so later uses of the saved value use the truncated value).
        # Needs CUDA or XPU.
        from torch.utils.checkpoint import (
            checkpoint,
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        torch.manual_seed(0)
        lowp_name = str(lowp_dtype).removeprefix("torch.")

        def policy(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Checkpoint policy that always saves the sigmoid result and
            # prefers recomputing everything else.
            if op == torch.ops.aten.sigmoid.default:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        context_fn = functools.partial(create_selective_checkpoint_contexts, policy)

        def g(x):
            # [FUNCTION DOC] HELPER: Sigmoid of x multiplied by itself.
            y = torch.sigmoid(x)
            return y * y

        def fn(x):
            # [FUNCTION DOC] HELPER: Run the helper g under non-reentrant checkpointing with the
            # selective checkpoint context.
            return checkpoint(g, x, use_reentrant=False, context_fn=context_fn)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        x = (torch.randn(64, device=device_type, dtype=lowp_dtype) * 3).requires_grad_(
            True
        )

        expected = fn(x)
        actual, (code,) = run_and_get_code(
            opt_fn, x.detach().clone().requires_grad_(True)
        )

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertIn(f".to(tl.{lowp_name})", code)

    @parametrize("lowp_dtype", [torch.bfloat16, torch.float16])
    @torch._inductor.config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_preserves_explicit_precision_cast(
        self, lowp_dtype
    ):
        # [FUNCTION DOC] Parametrized over bf16 and fp16. With emulate_precision_casts on, an
        # explicit cast of a matmul result to low precision and back to float32 must be kept:
        # compiled output equals eager, the code contains the explicit convert node marker and a
        # Triton down cast to the low precision type. Seeds CPU and device RNGs. Raises SkipTest
        # for fp16 on XPU (upstream Triton XPU folds the cast).
        if TEST_XPU and lowp_dtype is torch.float16:
            # To be enabled once triton-xpu emits a constrained fptrunc for
            # `arith.truncf f32 -> f16`, so the fp32 -> fp16 -> fp32 barrier
            # survives SPIR-V/IGC lowering (currently folded to identity).
            # Upstream triton bug: intel/intel-xpu-backend-for-triton#7491.
            # Tracker: intel/torch-xpu-ops#4358.
            raise unittest.SkipTest(
                "XPU: to be enabled after triton-xpu fix "
                "(intel/intel-xpu-backend-for-triton#7491)"
            )
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0) if TEST_CUDA else torch.xpu.manual_seed_all(0)
        lowp_name = str(lowp_dtype).removeprefix("torch.")
        x = torch.randn(4, 32, 32, device=device_type, dtype=torch.float32)
        w = torch.randn(32, 32, device=device_type, dtype=torch.float32)

        def fn(x, w):
            # [FUNCTION DOC] HELPER: Matmul x and w, round the result to the low precision dtype
            # and back to float32, multiply by its sigmoid, and sum over dim 1.
            x = torch.matmul(x, w)
            x = x.to(lowp_dtype).float()
            x = x * torch.sigmoid(x)
            return x.sum(dim=1)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        expected = fn(x, w)
        actual, (code,) = run_and_get_code(opt_fn, x.clone(), w.clone())
        self.assertEqual(expected, actual)
        self.assertIn("args = (%convert_element_type, torch.float32)", code)
        self.assertIn(f".to(tl.{lowp_name})", code)

    @torch._inductor.config.patch(emulate_precision_casts=True)
    @torch._inductor.config.patch(pattern_matcher=False)
    def test_emulate_precision_casts_convert_element_type(self):
        # [FUNCTION DOC] With emulate_precision_casts on and the pattern matcher off, a float32 to
        # bf16 to float32 round trip compiled with dynamic shapes must equal eager (the rounding
        # must not be optimized away).
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0) if TEST_CUDA else torch.xpu.manual_seed_all(0)

        x = torch.rand(1000, device=device_type, dtype=torch.float32)

        def fn(x):
            # [FUNCTION DOC] HELPER: Cast x to bf16 and back to float32 and return it.
            x_bf16 = x.to(torch.bfloat16)
            x_fp32_casted = x_bf16.to(torch.float32)
            return x_fp32_casted

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)

        expected = fn(x)
        actual = opt_fn(x)
        self.assertEqual(expected, actual)

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    @torch._inductor.config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_promoted_lowp_cuda(self):
        # [FUNCTION DOC] With emulate_precision_casts on (CUDA only), adding a float16 copy of
        # 70000.0 to the float32 original must overflow to infinity as in eager, and the generated
        # code must contain a Triton cast to float16.
        def fn(x):
            # [FUNCTION DOC] HELPER: Cast x to float16 and add the float32 x.
            y = x.to(torch.float16)
            return y + x

        x = torch.tensor([70000.0], device=device_type)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        expected = fn(x)
        actual, (code,) = run_and_get_code(opt_fn, x)

        self.assertEqual(expected, actual)
        self.assertIn(".to(tl.float16)", code)

    @torch._inductor.config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_norm_rounding(self):
        # [FUNCTION DOC] With emulate_precision_casts on, compare eager and compiled (dynamic)
        # results for the norm of a bf16 vector and that norm plus a float32 scalar, to verify
        # rounding of the intermediate norm.
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        x = torch.rand(1000, device=device_type, dtype=torch.bfloat16)
        scalar = torch.rand([], device=device_type, dtype=torch.float32)

        def fn(inp, scale):
            # [FUNCTION DOC] HELPER: Return the norm of inp and the norm plus scale as a tuple.
            y = inp.norm()
            return y, y + scale

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)

        expected = fn(x, scalar)
        actual = opt_fn(x, scalar)

        self.assertEqual(expected, actual)

    @torch._inductor.config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_min_pow_chain(self):
        # [FUNCTION DOC] With emulate_precision_casts on and scalar output capture enabled,
        # compile a graph mixing fp16, bf16 and fp32 inputs (min, sum, repeated subtraction,
        # chained pow) with dynamic shapes and compare both outputs against eager with a loose
        # tolerance (larger on XPU). Inputs require grad but no backward is run.
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0) if TEST_CUDA else torch.xpu.manual_seed_all(0)

        with dynamo_config.patch(
            capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
        ):
            arg0 = torch.rand(
                [383, 55, 2, 3],
                dtype=torch.float16,
                device=device_type,
                requires_grad=True,
            )
            arg1 = torch.rand(
                [383, 55], dtype=torch.bfloat16, device=device_type, requires_grad=True
            )
            arg2 = torch.rand(
                [383, 55], dtype=torch.float32, device=device_type, requires_grad=True
            )
            arg3 = torch.rand(
                [383, 55], dtype=torch.float32, device=device_type, requires_grad=True
            )

            def fn(a0, a1, a2, a3):
                # [FUNCTION DOC] HELPER: Take the min over dim 2 and sum over the new dim 2 of a0,
                # subtract a2 once and a3 three times from a1, add the two, and apply four nested
                # pow operations to produce a second result; returns both results as a tuple.
                t1 = a0.min(dim=2).values
                t2 = t1.sum(dim=2)
                t6 = ((((a1) - a2) - a3) - a3) - a3
                t7 = t6 + t2
                t8 = torch.pow(torch.pow(torch.pow(torch.pow(t2, t7), t7), t7), t7)
                return t7, t8

            opt_fn = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)

            eager_out = fn(arg0, arg1, arg2, arg3)
            compiled_args = [
                arg0.clone().detach().requires_grad_(True),
                arg1.clone().detach().requires_grad_(True),
                arg2.clone().detach().requires_grad_(True),
                arg3.clone().detach().requires_grad_(True),
            ]
            compiled_out = opt_fn(*compiled_args)

            # On XPU, the Triton backend's kernel fusion produces slightly
            # different mixed-precision (fp16/bf16/f32) intermediate values,
            # which the chained pow ops amplify exponentially.
            rtol = 5e-2 if TEST_XPU else 1e-3
            for eager_tensor, compiled_tensor in zip(eager_out, compiled_out):
                torch.testing.assert_close(
                    eager_tensor,
                    compiled_tensor,
                    rtol=rtol,
                    atol=1e-3,
                )

    @torch._inductor.config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_mean_ratio_chain(self):
        # [FUNCTION DOC] With emulate_precision_casts on, compile a graph of bf16, fp16 and fp32
        # inputs combining a mean over pairs, a view, relu and mean, and chained divisions, and
        # compare to eager with loose tolerances (rtol 0.05, atol 0.1). Seeds the RNG with 12345.
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)

        with dynamo_config.patch(
            capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
        ):
            arg0 = torch.rand(
                [125070], dtype=torch.bfloat16, device=device_type, requires_grad=True
            )
            arg1 = torch.rand(
                [1895, 3, 11],
                dtype=torch.float16,
                device=device_type,
                requires_grad=True,
            )
            arg2 = torch.rand(
                [1895, 3, 11],
                dtype=torch.float32,
                device=device_type,
                requires_grad=True,
            )
            arg3 = torch.rand(
                [1895, 3, 11],
                dtype=torch.float32,
                device=device_type,
                requires_grad=True,
            )
            arg4 = torch.rand(
                [1895, 3, 11],
                dtype=torch.float32,
                device=device_type,
                requires_grad=True,
            )
            arg5 = torch.rand(
                [5, 379, 165],
                dtype=torch.float32,
                device=device_type,
                requires_grad=True,
            )

            def fn(a0, a1, a2, a3, a4, a5):
                # [FUNCTION DOC] HELPER: Average a0 in pairs to a 379 by 165 tensor, subtract a2
                # twice, a3 and a4 from a1 and view it as 379 by 165, average the relu of a5 over
                # dim 0, then divide the pair average by those terms in sequence and return the
                # result.
                t2 = a0.view(379, 165, 2).mean(dim=2)
                t7 = ((((a1) - a2) - a3) - a2) - a4
                t8 = t7.view(379, 165)
                t11 = torch.nn.functional.relu(a5).mean(dim=0)
                t12 = t2 - t11
                t13 = (((t2) / t8) / t11) / t12
                return t13

            opt_fn = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)

            eager_out = fn(arg0, arg1, arg2, arg3, arg4, arg5)
            compiled_args = [
                tensor.clone().detach().requires_grad_(True)
                for tensor in (arg0, arg1, arg2, arg3, arg4, arg5)
            ]
            compiled_out = opt_fn(*compiled_args)

            torch.testing.assert_close(
                eager_out,
                compiled_out,
                rtol=5e-2,
                atol=1e-1,
            )

    @torch._inductor.config.patch(emulate_precision_casts=True)
    def test_dont_inplace_disjoint_accesses(self):
        # [FUNCTION DOC] With emulate_precision_casts on, compile a traced rms-norm plus rotary-
        # embedding style graph with three matmuls sharing one input; the generated code must not
        # reuse an input buffer in place (no in-out pointer arguments), since earlier a disjoint-
        # access inplace reuse caused large errors. Inputs are built by parsing the annotations,
        # and outputs are compared to eager with atol 0.01 and rtol 2.
        # TODO - would not need mms if we could annotate donated buffer..
        def forward(
            arg0_1: f"bf16[2048, 2048][2048, 1]{device_type}:0",
            arg1_1: f"bf16[8, 4096, 2048][8388608, 2048, 1]{device_type}:0",
            arg2_1: f"bf16[2048, 2048][2048, 1]{device_type}:0",
            arg3_1: f"bf16[2048, 2048][2048, 1]{device_type}:0",
            arg4_1: f"bf16[2048][1]{device_type}:0",
            arg5_1: f"bf16[2048][1]{device_type}:0",
            arg6_1: f"f32[4096, 128][128, 1]{device_type}:0",
            arg7_1: f"f32[4096, 128][128, 1]{device_type}:0",
        ):
            # [FUNCTION DOC] HELPER: Printed fx graph body. Three linear projections of arg1_1 by
            # arg0_1, arg2_1 and arg3_1 give three tensors; the first two are rms-normalized
            # (float32) with weights arg4_1 and arg5_1, then rotary embedded using the cos and sin
            # tables arg6_1 and arg7_1; all three are viewed into heads of 128 and permuted.
            # Returns the three results as a tuple.
            permute = torch.ops.aten.permute.default(arg0_1, [1, 0])
            arg0_1 = None
            view = torch.ops.aten.view.default(arg1_1, [32768, 2048])
            mm = torch.ops.aten.mm.default(view, permute)
            view = permute = None
            view_1 = torch.ops.aten.view.default(mm, [8, 4096, 2048])
            mm = None
            permute_1 = torch.ops.aten.permute.default(arg2_1, [1, 0])
            arg2_1 = None
            view_2 = torch.ops.aten.view.default(arg1_1, [32768, 2048])
            mm_1 = torch.ops.aten.mm.default(view_2, permute_1)
            view_2 = permute_1 = None
            view_3 = torch.ops.aten.view.default(mm_1, [8, 4096, 2048])
            mm_1 = None
            permute_2 = torch.ops.aten.permute.default(arg3_1, [1, 0])
            arg3_1 = None
            view_4 = torch.ops.aten.view.default(arg1_1, [32768, 2048])
            arg1_1 = None
            mm_2 = torch.ops.aten.mm.default(view_4, permute_2)
            view_4 = permute_2 = None
            view_5 = torch.ops.aten.view.default(mm_2, [8, 4096, 2048])
            mm_2 = None
            convert_element_type_6 = torch.ops.prims.convert_element_type.default(
                view_1, torch.float32
            )
            view_1 = None
            pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2)
            mean = torch.ops.aten.mean.dim(pow_1, [-1], True)
            pow_1 = None
            add = torch.ops.aten.add.Tensor(mean, 1e-06)
            mean = None
            rsqrt = torch.ops.aten.rsqrt.default(add)
            add = None
            mul = torch.ops.aten.mul.Tensor(convert_element_type_6, rsqrt)
            convert_element_type_6 = rsqrt = None
            convert_element_type_7 = torch.ops.prims.convert_element_type.default(
                arg4_1, torch.float32
            )
            arg4_1 = None
            mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_7, mul)
            convert_element_type_7 = mul = None
            convert_element_type_8 = torch.ops.prims.convert_element_type.default(
                mul_1, torch.bfloat16
            )
            mul_1 = None
            convert_element_type_9 = torch.ops.prims.convert_element_type.default(
                view_3, torch.float32
            )
            view_3 = None
            pow_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_9, 2)
            mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True)
            pow_2 = None
            add_1 = torch.ops.aten.add.Tensor(mean_1, 1e-06)
            mean_1 = None
            rsqrt_1 = torch.ops.aten.rsqrt.default(add_1)
            add_1 = None
            mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_9, rsqrt_1)
            convert_element_type_9 = rsqrt_1 = None
            convert_element_type_10 = torch.ops.prims.convert_element_type.default(
                arg5_1, torch.float32
            )
            arg5_1 = None
            mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_10, mul_2)
            convert_element_type_10 = mul_2 = None
            convert_element_type_11 = torch.ops.prims.convert_element_type.default(
                mul_3, torch.bfloat16
            )
            mul_3 = None
            view_6 = torch.ops.aten.view.default(
                convert_element_type_8, [8, 4096, -1, 128]
            )
            convert_element_type_8 = None
            view_7 = torch.ops.aten.view.default(
                convert_element_type_11, [8, 4096, -1, 128]
            )
            convert_element_type_11 = None
            view_8 = torch.ops.aten.view.default(view_5, [8, 4096, -1, 128])
            view_5 = None
            convert_element_type_12 = torch.ops.prims.convert_element_type.default(
                view_6, torch.float32
            )
            view_6 = None
            convert_element_type_13 = torch.ops.prims.convert_element_type.default(
                view_7, torch.float32
            )
            view_7 = None
            unsqueeze = torch.ops.aten.unsqueeze.default(arg6_1, 0)
            unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
            unsqueeze = None
            unsqueeze_2 = torch.ops.aten.unsqueeze.default(arg7_1, 0)
            unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2)
            unsqueeze_2 = None
            mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_12, unsqueeze_3)
            unsqueeze_3 = None
            view_9 = torch.ops.aten.view.default(
                convert_element_type_12, [8, 4096, 16, 2, 64]
            )
            convert_element_type_12 = None
            unbind = torch.ops.aten.unbind.int(view_9, -2)
            view_9 = None
            getitem = unbind[0]
            getitem_1 = unbind[1]
            unbind = None
            neg = torch.ops.aten.neg.default(getitem_1)
            getitem_1 = None
            cat = torch.ops.aten.cat.default([neg, getitem], -1)
            neg = getitem = None
            mul_5 = torch.ops.aten.mul.Tensor(cat, unsqueeze_1)
            cat = unsqueeze_1 = None
            add_2 = torch.ops.aten.add.Tensor(mul_4, mul_5)
            mul_4 = mul_5 = None
            unsqueeze_4 = torch.ops.aten.unsqueeze.default(arg6_1, 0)
            arg6_1 = None
            unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2)
            unsqueeze_4 = None
            unsqueeze_6 = torch.ops.aten.unsqueeze.default(arg7_1, 0)
            arg7_1 = None
            unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2)
            unsqueeze_6 = None
            mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_13, unsqueeze_7)
            unsqueeze_7 = None
            view_10 = torch.ops.aten.view.default(
                convert_element_type_13, [8, 4096, 16, 2, 64]
            )
            convert_element_type_13 = None
            unbind_1 = torch.ops.aten.unbind.int(view_10, -2)
            view_10 = None
            getitem_2 = unbind_1[0]
            getitem_3 = unbind_1[1]
            unbind_1 = None
            neg_1 = torch.ops.aten.neg.default(getitem_3)
            getitem_3 = None
            cat_1 = torch.ops.aten.cat.default([neg_1, getitem_2], -1)
            neg_1 = getitem_2 = None
            mul_7 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_5)
            cat_1 = unsqueeze_5 = None
            add_3 = torch.ops.aten.add.Tensor(mul_6, mul_7)
            mul_6 = mul_7 = None
            convert_element_type_14 = torch.ops.prims.convert_element_type.default(
                add_2, torch.bfloat16
            )
            add_2 = None
            convert_element_type_15 = torch.ops.prims.convert_element_type.default(
                add_3, torch.bfloat16
            )
            add_3 = None
            permute_3 = torch.ops.aten.permute.default(
                convert_element_type_14, [0, 2, 1, 3]
            )
            convert_element_type_14 = None
            permute_4 = torch.ops.aten.permute.default(
                convert_element_type_15, [0, 2, 1, 3]
            )
            convert_element_type_15 = None
            permute_5 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3])
            view_8 = None
            return (permute_3, permute_4, permute_5)

        from torch._dynamo.debug_utils import aot_graph_input_parser

        kwargs = aot_graph_input_parser(forward, device=device_type)
        out, code = run_and_get_code(torch.compile(forward), **kwargs)
        # ignore tiny values.. prior to this fix absolute error was ~28
        self.assertEqual(forward(**kwargs), out, atol=0.01, rtol=2)
        FileCheck().check_not("in_out").run(code[0])

    # https://github.com/pytorch/pytorch/issues/104937
    def test_linear_with_zero_infeature_size(self):
        # [FUNCTION DOC] Regression test (issue 104937): a bias linear layer with zero input and
        # output features applied to a zero-width input must give the same result compiled and in
        # eager.
        m = nn.Linear(in_features=0, out_features=0, bias=True).to(device_type)
        x = torch.rand(1, 1, 0, device=device_type)
        expect = m(x)
        opt_fn = torch.compile(m)
        actual = opt_fn(x)
        self.assertEqual(expect, actual)

    @config.patch(fallback_random=True)
    def test_multi_output_layout_fallback(self):
        # [FUNCTION DOC] With fallback_random on, check an in-place randomized leaky relu module
        # compiled gives the same output as eager when both start from the same frozen RNG state
        # (the compiled one runs under a freeze_rng_state block).
        mod = nn.RReLU(lower=3.2350976, upper=8.4220314, inplace=True)
        inp = torch.rand([4, 4]).to(device_type)
        m = torch.compile(mod)

        with freeze_rng_state():
            o1 = m(inp.clone())

        o2 = mod(inp.clone())

        self.assertEqual(o1, o2)

    def test_sorted_masks(self):
        # [FUNCTION DOC] Check that the generated code for a pointwise add followed by a row sum
        # loads with both the reduction mask and the x mask, in that order.
        @torch.compile()
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Sum of x plus y over dim 1.
            return (x + y).sum(dim=1)

        x = torch.rand([255, 255], device=device_type)
        y = torch.rand([255, 255], device=device_type)

        _, code = run_and_get_code(foo, x, y)
        FileCheck().check("tl.load").check_same("r0_mask").check_same("xmask").run(
            code[0]
        )

    def test_cat_int8_one_kernel(self):
        # [FUNCTION DOC] For uint8 and int8, compile a concatenation of four 256 by 256 tensors
        # plus one, and check the output equals eager, there is no leftover cat call in the code,
        # and exactly one kernel launch.
        @torch.compile()
        def cat(inps):
            # [FUNCTION DOC] HELPER: Concatenate the list along dim 0 and add one.
            return torch.cat(inps) + 1

        for dtype in [torch.uint8, torch.int8]:
            inps = [
                torch.empty([256, 256], dtype=dtype, device=device_type)
                for _ in range(4)
            ]

            out, code = run_and_get_code(cat, inps)
            self.assertEqual(torch.cat(inps) + 1, out)
            FileCheck().check_not("aten.cat.default(").check_count(
                ".run(", 1, exactly=True
            ).run(code[0])

    @config.patch("triton.use_block_ptr", True)
    def test_selecsls42b_misaligned_address(self):
        # [FUNCTION DOC] Regression test (Triton issue 2836) for a misaligned-address crash with
        # block pointers enabled: runs a compiled batch-norm-backward style graph with a bool mask
        # and expanded fp16 input, then synchronizes the device so any kernel error surfaces.
        # https://github.com/triton-lang/triton/issues/2836

        @torch.compile(fullgraph=True)
        def fn(arg207_1, arg208_1, convert_element_type_40, expand, full, mul_3):
            # [FUNCTION DOC] HELPER: Graph body of batch norm backward style math: divide the
            # expanded tensor by 16, select with the boolean mask, convert to float32, compute
            # per-channel sums over dims 0, 2, 3 and use them to produce the final subtraction
            # result. Returns a one-element tuple.
            div = torch.ops.aten.div.Scalar(expand, 16)
            where = torch.ops.aten.where.self(arg207_1, full, div)
            convert_element_type_43 = torch.ops.prims.convert_element_type.default(
                where, torch.float32
            )
            sum_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_43, [0, 2, 3])
            sub = torch.ops.aten.sub.Tensor(convert_element_type_40, arg208_1)
            mul = torch.ops.aten.mul.Tensor(convert_element_type_43, sub)
            sum_3 = torch.ops.aten.sum.dim_IntList(mul, [0, 2, 3])
            mul_1 = torch.ops.aten.mul.Tensor(sum_2, 0.0078125)
            unsqueeze = torch.ops.aten.unsqueeze.default(mul_1, 0)
            unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
            unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3)
            mul_2 = torch.ops.aten.mul.Tensor(sum_3, 0.0078125)
            mul_4 = torch.ops.aten.mul.Tensor(mul_2, mul_3)
            unsqueeze_3 = torch.ops.aten.unsqueeze.default(mul_4, 0)
            unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2)
            unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3)
            mul_6 = torch.ops.aten.mul.Tensor(sub, unsqueeze_5)
            sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_43, mul_6)
            sub_2 = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_2)
            return (sub_2,)

        args = [
            torch.randn((8, 1024, 4, 4), device=device_type) > 0,  # torch.bool tensor
            torch.randn((1, 1024, 1, 1), device=device_type),
            torch.randn((8, 1024, 4, 4), device=device_type),
            torch.randn(
                (8, 1024, 1, 1), dtype=torch.float16, device=device_type
            ).expand((8, 1024, 4, 4)),
            torch.randn((), device=device_type),
            torch.randn((1024,), device=device_type),
        ]
        fn(*args)
        torch.accelerator.synchronize()  # shake out Triton Error [CUDA]: misaligned address

    def test_mutated_aligned_tensor(self):
        # [FUNCTION DOC] Check in-place mutation through a compiled function when the input is a
        # slice that may be misaligned: with dynamic shapes off, call on slices dropping the last
        # and then the first element, and check the compiled function's return and the original
        # tensor's contents stay consistent with the eager run on a cloned copy.
        t = torch.rand(4096, device=device_type, dtype=torch.float16)

        def foo(x):
            # [FUNCTION DOC] HELPER: Add one in place to x and return it.
            return x.add_(1)

        foo_c = torch.compile(dynamic=False)(foo)

        t_orig = t.clone()

        # First invocation, assume alignment, second invocation,
        # copy to alignment and then mutate after fn invocation
        self.assertEqual(foo_c(t[:-1]), foo(t_orig[:-1]))
        self.assertEqual(t, t_orig)

        self.assertEqual(foo_c(t[1:]), foo(t_orig[1:]))
        self.assertEqual(t, t_orig)

    def test_non_commutative_scan_op(self):
        # [FUNCTION DOC] Check that associative_scan with a non-commutative combine function (a
        # linear recurrence over two sequences, float64) matches a sequential baseline loop;
        # compares the second output.
        from torch._higher_order_ops.associative_scan import associative_scan

        a = torch.randn(1024, 8192, dtype=torch.float64, device=device_type)
        b = torch.randn(1024, 8192, dtype=torch.float64, device=device_type)

        def baseline(v, u):
            # [FUNCTION DOC] HELPER: Sequential reference for the recurrence: start with the first
            # column of b, then for each next column compute the previous result times a plus b,
            # and stack along dim 1. Uses the outer a and b tensors, not its arguments beyond the
            # width of v.
            A = []
            A.append(b[:, 0])
            for i in range(1, v.shape[1]):
                A.append(a[:, i] * A[i - 1] + b[:, i])
            return torch.stack(A, dim=1)

        def combine_fn(i, j):
            # [FUNCTION DOC] HELPER: Combine two pairs of the recurrence: returns the product of
            # the first parts and the second part of the left pair times the first part of the
            # right pair plus the second part of the right pair.
            ia, ib = i
            ja, jb = j
            return ia * ja, ib * ja + jb

        @torch.compile
        def compiled_scan(a, b):
            # [FUNCTION DOC] HELPER: Scan over the last dim with the combine function and return
            # only the second tensor.
            return associative_scan(combine_fn, (a, b), dim=-1)[1]

        out1 = baseline(a, b)
        out2 = compiled_scan(a, b)
        self.assertEqual(out1, out2)

    def test_associative_scan_dynamic_ndim(self):
        # [FUNCTION DOC] Check associative_scan lowering with an add combine on 1D dynamic, 2D
        # static and 2D dynamic inputs against cumsum, then with a multiply combine along dim 1 of
        # a clamped 3D dynamic input against eager scan. Compiler state is reset before each
        # compile. Skipped unless the device is CUDA.
        if device_type != "cuda":
            raise unittest.SkipTest("associative_scan pointwise lowering requires CUDA")

        from torch._higher_order_ops.associative_scan import associative_scan

        def add_combine(a, b):
            # [FUNCTION DOC] HELPER: Add combine function.
            return a + b

        def scan_dim0(x):
            # [FUNCTION DOC] HELPER: Associative scan with add along dim 0.
            return associative_scan(add_combine, x, dim=0)

        x1 = torch.randn(7, device=device_type)
        torch.compiler.reset()
        compiled_1d = torch.compile(scan_dim0, dynamic=True)
        self.assertEqual(torch.cumsum(x1, 0), compiled_1d(x1))

        x2 = torch.randn(4, 8, device=device_type)
        torch.compiler.reset()
        compiled_static = torch.compile(scan_dim0)
        self.assertEqual(torch.cumsum(x2, 0), compiled_static(x2))

        torch.compiler.reset()
        compiled_dynamic = torch.compile(scan_dim0, dynamic=True)
        self.assertEqual(torch.cumsum(x2, 0), compiled_dynamic(x2))

        def mul_combine(a, b):
            # [FUNCTION DOC] HELPER: Multiply combine function.
            return a * b

        def scan_dim1(x):
            # [FUNCTION DOC] HELPER: Associative scan with multiply along dim 1.
            return associative_scan(mul_combine, x, dim=1)

        x3 = torch.randn(2, 9, 5, device=device_type).clamp(-2, 2)
        torch.compiler.reset()
        compiled_3d = torch.compile(scan_dim1, dynamic=True)
        self.assertEqual(scan_dim1(x3), compiled_3d(x3))

    def test_dynamic_persistent_reductions(self):
        # [FUNCTION DOC] Check that dynamic-shape reductions whose size is bounded by an assert
        # (inner dim at most 1024, outer dim at most 64) use a persistent reduction, i.e. the
        # generated code contains no looped reduction offset. Checks results against eager too.
        @torch.compile(dynamic=True)
        def inner_reduce(x):
            # [FUNCTION DOC] HELPER: Assert the inner size is at most 1024 and sum over dim 1.
            assert x.shape[1] <= 1024  # noqa: S101
            return x.sum(1)

        a = torch.randn(50, 600, device=device_type)
        out, code = run_and_get_code(inner_reduce, a)
        self.assertEqual(inner_reduce(a), out)
        self.assertTrue("for roffset" not in code)

        @torch.compile(dynamic=True)
        def outer_reduce(x):
            # [FUNCTION DOC] HELPER: Assert the outer size is at most 64 and sum over dim 0.
            assert x.shape[0] <= 64  # noqa: S101
            return x.sum(0)

        out, code = run_and_get_code(outer_reduce, a)
        self.assertEqual(outer_reduce(a), out)
        self.assertTrue("for roffset" not in code)

    @config.patch(
        {
            "triton.multi_kernel": 0,
            "triton.tile_reductions": True,
            "triton.prefer_nd_tiling": False,
            "triton.max_tiles": 3,
            "split_reductions": False,
        }
    )
    def test_persistent_reduction_selection_uses_tiling_scores(self):
        # [FUNCTION DOC] Check that with tiled reductions and nd tiling off, a silu-times-up
        # followed by grouped absolute max (group size 128) over a bf16 input compiles into
        # exactly one persistent reduction kernel with the inner reduction hint and no looped
        # reduction. The compiled value also equals eager. Skipped unless CUDA.
        if device_type != "cuda":
            raise unittest.SkipTest("requires CUDA")

        def fn(x):
            # [FUNCTION DOC] HELPER: Split the columns of x in two halves, multiply silu of the
            # first by the second in float32, group into chunks of 128 and return the max absolute
            # value per group.
            hidden = x.shape[1] // 2
            group_size = 128
            gate = x[:, :hidden].to(torch.float32)
            up = x[:, hidden:].to(torch.float32)
            prod = F.silu(gate) * up
            grouped = prod.view(x.shape[0], hidden // group_size, group_size)
            return torch.amax(torch.abs(grouped), dim=-1)

        x = torch.randn(4, 512, device=device_type, dtype=torch.bfloat16)
        expected = fn(x)
        out, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(expected, out)

        code = "\n".join(code)
        self.assertIn("@triton_heuristics.persistent_reduction", code)
        self.assertEqual(1, code.count("@triton_heuristics.persistent_reduction"))
        persistent_code = code.split("@triton_heuristics.persistent_reduction", 1)[1]
        self.assertIn("reduction_hint=ReductionHint.INNER", persistent_code)
        self.assertNotIn("for roffset", persistent_code)

    @parametrize("use_block_ptr", [False, True])
    @parametrize("dynamic_batch", [False, True])
    @config.patch(
        {
            "triton.multi_kernel": 0,
            "triton.tile_reductions": True,
            "triton.prefer_nd_tiling": False,
            "triton.max_tiles": 3,
            "split_reductions": False,
        }
    )
    def test_persistent_reduction_cse_reindexed_epilogue(
        self, use_block_ptr, dynamic_batch
    ):
        # [FUNCTION DOC] Parametrized over block pointer usage and dynamic batch. Compiles an
        # fp8-style quantization epilogue (silu-gate product, per-group max scale, clamp, cast,
        # in-place copies into outputs) and compares both outputs to eager, including a second
        # batch size when dynamic. Checks the code has a single persistent reduction kernel, that
        # two loads from the first input use block pointers or plain pointer loads depending on
        # the parameter, and that at most one libdevice exp appears (common subexpression reuse).
        # Skipped unless CUDA.
        if device_type != "cuda":
            raise unittest.SkipTest("requires CUDA")

        out_dtype = torch.float16
        fp8_min = -448.0
        fp8_max = 448.0
        min_fp8_scale = 1.0 / (fp8_max * 512.0)

        def fn(out, x, scales_out, scale_ub):
            # [FUNCTION DOC] HELPER: Compute silu-gated group max scales, clamp them to the upper
            # bound and a minimum, quantize the grouped values by dividing by scales, clamp to the
            # fp8 range and cast to the output dtype, then copy the scales and result into the
            # given output buffers (mutates scales_out and out).
            hidden = x.shape[1] // 2
            group_size = 128
            gate = x[:, :hidden].to(torch.float32)
            up = x[:, hidden:].to(torch.float32)
            prod = F.silu(gate) * up
            grouped = prod.view(x.shape[0], hidden // group_size, group_size)
            scales = torch.amax(torch.abs(grouped), dim=-1)
            scales = torch.clamp(scales, max=scale_ub)
            scales = torch.clamp(scales * (1.0 / fp8_max), min=min_fp8_scale)
            y = torch.clamp(grouped / scales[:, :, None], fp8_min, fp8_max).to(
                out.dtype
            )
            scales_out.copy_(scales)
            out.copy_(y.view_as(out))

        x = torch.randn(4, 512, device=device_type, dtype=torch.bfloat16)
        out = torch.empty(4, 256, device=device_type, dtype=out_dtype)
        scales_out = torch.empty(4, 2, device=device_type)
        expected_out = torch.empty_like(out)
        expected_scales = torch.empty_like(scales_out)
        scale_ub = torch.tensor(1e6, device=device_type)

        fn(expected_out, x, expected_scales, scale_ub)
        if dynamic_batch:
            for tensor in (x, out, scales_out):
                torch._dynamo.mark_dynamic(tensor, 0)
        with config.patch("triton.use_block_ptr", use_block_ptr):
            compiled_fn = torch.compile(fn, fullgraph=True)
            _, code = run_and_get_code(
                compiled_fn,
                out,
                x,
                scales_out,
                scale_ub,
            )
            if dynamic_batch:
                x2 = torch.randn(7, 512, device=device_type, dtype=torch.bfloat16)
                out2 = torch.empty(7, 256, device=device_type, dtype=out_dtype)
                scales_out2 = torch.empty(7, 2, device=device_type)
                expected_out2 = torch.empty_like(out2)
                expected_scales2 = torch.empty_like(scales_out2)
                fn(expected_out2, x2, expected_scales2, scale_ub)
                compiled_fn(out2, x2, scales_out2, scale_ub)
                self.assertEqual(expected_out2, out2)
                self.assertEqual(expected_scales2, scales_out2)
        self.assertEqual(expected_out, out)
        self.assertEqual(expected_scales, scales_out)

        code = "\n".join(code)
        self.assertIn("@triton_heuristics.persistent_reduction", code)
        self.assertEqual(1, code.count("@triton_heuristics.persistent_reduction"))
        persistent_code = code.split("@triton_heuristics.persistent_reduction", 1)[1]
        if use_block_ptr:
            self.assertIn("tl.make_block_ptr(in_ptr0", persistent_code)
            self.assertEqual(
                2, persistent_code.count("tl.load(tl.make_block_ptr(in_ptr0")
            )
        else:
            self.assertEqual(2, persistent_code.count("tl.load(in_ptr0 +"))
        self.assertLessEqual(persistent_code.count("libdevice.exp"), 1)

    def test_scaled_dot_product_efficient_attention_backward(self):
        # [FUNCTION DOC] Regression test: compile a small attention module with a mask on length-1
        # inputs and run backward through the efficient SDPA path. Passes if no exception is
        # raised.
        from torch import nn, Tensor

        class SelfAttention(nn.Module):
            # [CLASS DOC] HELPER: Simplified BERT-style self attention: three linear projections
            # to query, key and value, reshaped into heads, then scaled dot product attention with
            # an attention mask and dropout only in training mode. Stores head count, head size
            # and dropout probability.
            def __init__(
                self,
                num_attention_heads: int = 12,
                hidden_size: int = 768,
                attention_probs_dropout_prob: float = 0.1,
            ):
                # [FUNCTION DOC] HELPER: Store the head count and head size (hidden size divided
                # by heads), create the three linear layers and keep the dropout probability.
                super().__init__()

                self.num_attention_heads = num_attention_heads
                self.attention_head_size = hidden_size // num_attention_heads

                self.query = nn.Linear(hidden_size, hidden_size)
                self.key = nn.Linear(hidden_size, hidden_size)
                self.value = nn.Linear(hidden_size, hidden_size)

                self.dropout_prob = attention_probs_dropout_prob

            def transpose_for_scores(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Reshape x so the last dim is split into heads and head
                # size, then permute to put heads before the sequence dim.
                new_x_shape = x.size()[:-1] + (
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                return x.view(new_x_shape).permute(0, 2, 1, 3)

            def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Project hidden_states to query, key and value layers,
                # split into heads and return scaled dot product attention using attention_mask,
                # with dropout only when training.
                query_layer = self.transpose_for_scores(self.query(hidden_states))
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout_prob if self.training else 0.0,
                    is_causal=False,
                )
                return attn_output

        device = torch.device(device_type)
        num_attention_heads = 8
        hidden_size = 512
        attention_probs_dropout_prob = 0.0
        model = SelfAttention(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        ).to(device)

        model = torch.compile(model)

        # runs without failure
        batch_size = 8
        length = 1
        inputs_embeds = torch.randn(batch_size, length, hidden_size, device=device)
        attention_mask = torch.ones(batch_size, 1, length, length, device=device)
        attn_output = model(hidden_states=inputs_embeds, attention_mask=attention_mask)[
            0
        ]
        loss = attn_output.mean()
        loss.backward()

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Does not support mem_eff_attention",
    )
    def test_mem_eff_attention_backward_non_16_aligned_head_dim(self):
        # [FUNCTION DOC] For non-causal and causal attention with head dim 20 (not a multiple of
        # 16), compare forward value and input gradients of eager and compiled SDPA forced onto
        # the efficient attention backend. Resets dynamo and seeds the RNG each iteration. Skipped
        # without mem-efficient attention support.
        for is_causal in (False, True):
            torch._dynamo.reset()
            torch.manual_seed(0)

            def fn(q, k, v):
                # [FUNCTION DOC] HELPER: Scaled dot product attention using the loop's causal
                # flag, summed to a scalar.
                return F.scaled_dot_product_attention(
                    q, k, v, is_causal=is_causal
                ).sum()

            inputs = [
                torch.randn(
                    2,
                    4,
                    8,
                    20,
                    device=device_type,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                for _ in range(3)
            ]
            eager_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
            compiled_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
            compiled_fn = torch.compile(fn, backend="inductor")

            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                expected = fn(*eager_inputs)
                expected.backward()
                actual = compiled_fn(*compiled_inputs)
                actual.backward()

            self.assertEqual(actual, expected)
            for actual_input, expected_input in zip(compiled_inputs, eager_inputs):
                self.assertEqual(actual_input.grad, expected_input.grad)

    def test_non_contiguous_unaligned_input_indices(self):
        # [FUNCTION DOC] Unit test of the helper that removes unaligned indices from the static
        # input index list: a sliced tensor (offset by one row) is dropped from the list, while
        # aligned ones are kept.
        from torch._inductor.compile_fx import remove_unaligned_input_idxs

        inputs = [
            torch.ones(2, 2, device=device_type),
            torch.ones(2, 2, device=device_type)[1:],
        ]
        idxs = remove_unaligned_input_idxs(inputs, [1])
        self.assertEqual(idxs, [])

        inputs = [
            torch.ones(2, 2, device=device_type),
            torch.ones(2, 2, device=device_type),
            torch.ones(2, 2, device=device_type)[1:],
        ]
        idxs = remove_unaligned_input_idxs(inputs, [0, 2])
        self.assertEqual(idxs, [0])

    @skipIfXpu(msg="cudagraph is not supported on xpu")
    @skipIfCachingAllocatorDisabled
    @config.patch("triton.cudagraphs", True)
    def test_unused_cpu_input_cudagraphs(self):
        # [FUNCTION DOC] With cudagraphs enabled, lower a traced pointwise function that has an
        # unused CPU input and check that cudagraphs are not disabled (no disable reason), the
        # graph's device types is only the accelerator, and the compiled output matches eager.
        # Skipped on XPU or if the caching allocator is disabled.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Apply sin four times and cos to x and add one; y is unused.
            return x.sin().sin().sin().sin().cos() + 1

        fx_graph = torch.fx.symbolic_trace(fn)
        inp = [torch.randn(64, device=device_type), torch.randn(64, device="cpu")]
        compiled_fn, (graph,) = run_and_get_graph_lowering(
            torch._inductor.compile, fx_graph, inp
        )
        self.assertEqual(graph.disable_cudagraphs_reason, None)
        self.assertEqual(graph.device_types, {device_type})
        self.assertEqual(compiled_fn(*inp), fn(*inp))

    def test_epilogue_fusion_with_view(self):
        # [FUNCTION DOC] Compile a conv, flatten and linear model with max-autotune on device
        # index 0 inside a fresh inductor cache, and compare with eager on a random 32 by 3 by 64
        # by 64 input (tolerance 1e-3). Exercises epilogue fusion with a view.
        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Toy model with a conv layer (3 to 64 channels, 3 by 3, padding
            # 1), a linear layer with 262144 inputs and 100 outputs and a relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the conv, linear and relu layers.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
                self.linear = torch.nn.Linear(262144, 100)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the conv, flatten all dims after the batch dim,
                # then linear and relu.
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.relu(self.linear(x))

        m = ToyModel().to(device=f"{device_type}:0")
        input_tensor = torch.randn(32, 3, 64, 64).to(device=f"{device_type}:0")
        from torch._inductor.utils import fresh_cache

        with fresh_cache():
            cm = torch.compile(m, mode="max-autotune")
            out = cm(input_tensor)
            out2 = m(input_tensor)
            self.assertEqual(out, out2, atol=1e-3, rtol=1e-3)

    @skipIfXpu(msg="cudagraph is not supported on xpu")
    @skipIfCachingAllocatorDisabled
    @config.patch("triton.cudagraphs", True)
    def test_cpu_index(self):
        # [FUNCTION DOC] With cudagraphs enabled, check that indexing a GPU tensor with a CPU
        # arange index does not disable cudagraphs and that only the accelerator appears in the
        # graph device types, for the forward with and without grad and for the backward graph.
        # Skipped on XPU or if the caching allocator is disabled.
        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Index x with the first 32 integers created on the CPU.
            return x[torch.arange(32)]

        result, (graph,) = run_and_get_graph_lowering(
            fn, torch.randn(64, device=device_type)
        )
        self.assertEqual(graph.disable_cudagraphs_reason, None)
        self.assertEqual(graph.device_types, {device_type})

        inp = torch.randn(64, device=device_type, requires_grad=True)
        result, (graph,) = run_and_get_graph_lowering(fn, inp)
        self.assertEqual(graph.disable_cudagraphs_reason, None)
        self.assertEqual(graph.device_types, {device_type})

        result, (graph,) = run_and_get_graph_lowering(lambda: result.sum().backward())
        self.assertEqual(graph.disable_cudagraphs_reason, None)
        self.assertEqual(graph.device_types, {device_type})

    @unittest.skipIf(IS_FBCODE, "Not runnable in fbcode")
    def test_triton_interpret(self):
        # [FUNCTION DOC] Run a small compiled add in a subprocess that has the Triton interpreter
        # environment variable set and check it exits with success (output values are not
        # compared). Skipped in fbcode.
        import subprocess

        script = f"""
import os
os.environ["TRITON_INTERPRET"] = "1"
import torch

@torch.compile()
def foo(x):
    return x + 1

# somehow gives different results.. still, check that it doesn't error
foo(torch.rand([256], device=\"{device_type}\"))
"""

        subprocess.run([sys.executable, "-c", script], check=True)

    def test_reflection_pad_loop_order(self):
        # [FUNCTION DOC] Check that two reflection pads added together compile into one pointwise
        # kernel which iterates in contiguous order (no tiling): extracts the first kernel from
        # the generated code (static asserts stripped) and compares it with an expected text, and
        # checks the result against eager.
        # KNOWN ISSUE: the expected kernel text is a hard-coded golden string, so it is sensitive
        #   to codegen changes.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Reflection-pad both inputs by 5 on every side of the last two
            # dims and add them.
            a = torch.nn.functional.pad(x, (5, 5, 5, 5), mode="reflect")
            b = torch.nn.functional.pad(y, (5, 5, 5, 5), mode="reflect")
            return a + b

        cfn = torch.compile(fn)
        a = torch.rand((10, 10, 10), device=device_type)
        b = torch.rand((10, 10, 10), device=device_type)
        expect = fn(a, b)
        actual, code = run_and_get_code(cfn, a, b)
        self.assertEqual(expect, actual)

        # Expect the code iterates in contiguous order, and is not tiled
        lines = _strip_triton_static_asserts(code[0]).split("\n")
        start = lines.index("@triton.jit")
        kernel_code = "\n".join(lines[start : start + 14])
        self.assertExpectedInline(
            kernel_code,
            """\
@triton.jit
def triton_poi_fused_add_reflection_pad2d_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 20)
    x1 = ((xindex // 20) % 20)
    x2 = xindex // 400
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (99 + ((-1)*tl_math.abs((-9) + tl_math.abs((-5) + x0))) + ((-10)*tl_math.abs((-9) + tl_math.abs((-5) + x1))) + 100*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (99 + ((-1)*tl_math.abs((-9) + tl_math.abs((-5) + x0))) + ((-10)*tl_math.abs((-9) + tl_math.abs((-5) + x1))) + 100*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)""",
        )

    @skipCUDAIf(not SM80OrLater, "uses bfloat16 which requires SM >= 80")
    @skipCUDAIf(not SM90OrLater, "needs at least 6 GiB of GPU memory headroom")
    def test_chunked_unrolled_slice_gather_int32_overflow(self):
        # [FUNCTION DOC] Regression test for an int32 index miscompile: an unrolled loop over
        # chunks of a bf16 projection with a gather (cross-entropy-like log-sum-exp) fused in one
        # kernel produced an index that overflows 32 bits and caused illegal memory access. Runs
        # compiled without dynamic shapes, synchronizes, and checks results are close to eager
        # (loose tolerance). Needs SM80 and SM90 level GPUs.
        # Regression for an int32-indexing miscompile when an unrolled chunked
        # loop containing a per-chunk F.linear and a gather gets fused into one
        # pointwise kernel. Per-chunk buffer storage individually fits in int32,
        # but the cross-buffer index `V * x0` reaches `V * total_numel` which
        # overflows int32, and a separate kernel-level fusion synthesizes
        # exactly that expression. Result before the fix: cudaErrorIllegalAddress.
        torch.manual_seed(0)
        T, D, V_, CHUNKSZ = 16384, 128, 200008, 2048
        x = torch.randn(1, T, D, device=device_type, dtype=torch.bfloat16)
        targets = torch.randint(0, V_, (1, T), device=device_type, dtype=torch.int64)
        W = torch.randn(V_, D, device=device_type, dtype=torch.bfloat16) * 0.01
        bias = torch.zeros(V_, device=device_type, dtype=torch.bfloat16)

        def fwd(x, targets, W, bias):
            # [FUNCTION DOC] HELPER: For each chunk of 2048 rows, compute logits with the shared
            # weight plus bias, then the log-sum-exp minus the logit at the target index, written
            # into the matching slice of a float32 output buffer. Returns the buffer.
            seqlen = x.shape[-2]
            out = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
            for start in range(0, seqlen, CHUNKSZ):
                end = start + CHUNKSZ
                chunk_x = x[..., start:end, :]
                chunk_targets = targets[..., start:end]
                logits = torch.nn.functional.linear(chunk_x, W) + bias
                m = logits.amax(dim=-1, keepdim=True)
                tok = (
                    torch.log(
                        torch.sum(torch.exp(logits - m), dim=-1, dtype=torch.float32)
                    )
                    + m.squeeze(-1).to(torch.float32)
                    - torch.gather(logits, -1, chunk_targets[..., None])
                    .squeeze(-1)
                    .to(torch.float32)
                )
                out[..., start:end] = tok
            return out

        eager = fwd(x, targets, W, bias)
        opt = torch.compile(fwd, dynamic=False, fullgraph=True)
        compiled = opt(x, targets, W, bias)
        torch.accelerator.synchronize()
        # bf16 matmul + log_sum_exp leaves some slack; loose tol is fine here,
        # the test is about not crashing.
        self.assertTrue(torch.allclose(eager, compiled, atol=1e-2, rtol=1e-2))

    def test_int64_index_intermediate(self):
        # [FUNCTION DOC] Check that splitting a huge bf16 tensor into eight pieces and
        # concatenating them gives the same result as eager, both with and without marking dim 0
        # dynamic (intermediate indices need 64-bit).
        def foo(inp):
            # [FUNCTION DOC] HELPER: View the input as 3D with an 8192 by 8192 tail, split along
            # dim 1 into eight chunks of 1024, concatenate them along dim 0 and return it.
            view_23 = torch.ops.aten.view.default(inp, [-1, 8192, 8192])
            split_1 = torch.ops.aten.split.Tensor(view_23, 1024, 1)
            view_23 = None
            getitem_17 = split_1[0]
            getitem_18 = split_1[1]
            getitem_19 = split_1[2]
            getitem_20 = split_1[3]
            getitem_21 = split_1[4]
            getitem_22 = split_1[5]
            getitem_23 = split_1[6]
            getitem_24 = split_1[7]
            split_1 = None
            cat_1 = torch.ops.aten.cat.default(
                [
                    getitem_17,
                    getitem_18,
                    getitem_19,
                    getitem_20,
                    getitem_21,
                    getitem_22,
                    getitem_23,
                    getitem_24,
                ]
            )
            getitem_17 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = (
                getitem_22
            ) = getitem_23 = getitem_24 = None
            return cat_1

        for mark_dynamic in [False, True]:
            inp = torch.rand((65536, 8192), dtype=torch.bfloat16, device=device_type)
            if mark_dynamic:
                torch._dynamo.mark_dynamic(inp, 0)
            foo_c = torch.compile(foo)
            torch.testing.assert_allclose(foo(inp), foo_c(inp))

    @skipCUDAIf(
        not SM90OrLater, "uses bfloat16 atomic add instrs which requires SM >= 90"
    )
    def test_float8_e8m0fnu(self):
        # [FUNCTION DOC] Check float8 e8m0 handling: a round trip cast float32 to e8m0 and back
        # must match eager; then a uint8 tensor plus one viewed as e8m0 and flattened must match
        # eager and not produce a reshape call in the generated code. Skipped below SM90.
        device = device_type
        dtype = torch.float8_e8m0fnu
        hp_dtype = torch.float32  # and torch.bfloat16

        def foo(x0):
            # [FUNCTION DOC] HELPER: Cast x to the float8 e8m0 type, then back to the high
            # precision dtype (float32).
            x1 = x0.to(dtype)
            x2 = x1.to(hp_dtype)
            return x2

        x0 = torch.randn(16, 16, device=device, dtype=hp_dtype)
        foo_c = torch.compile(foo, backend="inductor", fullgraph=True)

        with torch.no_grad():
            y_c = foo_c(x0)

        self.assertEqual(foo(x0), y_c)

        dtype = torch.float8_e8m0fnu

        def foo(x0):
            # [FUNCTION DOC] HELPER: Add one to x, reinterpret the bytes as the float8 e8m0 type
            # and flatten to 256 elements. This replaces the earlier local helper of the same
            # name.
            x1 = x0 + 1
            x2 = x1.view(dtype).view([16 * 16])
            return x2

        x0 = torch.randint(0, 255, (16, 16), device=device, dtype=torch.uint8)
        foo_c = torch.compile(foo, backend="inductor", fullgraph=True)

        with torch.no_grad():
            result, code = run_and_get_code(foo_c, x0)

        FileCheck().check("call").check_not("torch.ops.aten.reshape.default(").run(
            code[0]
        )
        self.assertEqual(foo(x0), result)

    @unittest.skipIf(
        not config.is_fbcode(),
        "bfloat16 atomic add is only supported in fbcode today #97016",
    )
    @skipCUDAIf(
        not SM90OrLater, "uses bfloat16 atomic add instrs which requires SM >= 90"
    )
    def test_atomic_add_bfloat16(self):
        # [FUNCTION DOC] Check that the backward of a compiled index_select on bf16 with many
        # repeated indices uses an atomic add in Triton and that the forward result equals eager.
        # Only runs in fbcode on SM90 or later.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Select rows of x along dim 0 using the indices y.
            return torch.index_select(x, 0, y)

        x = torch.randn(
            2000, 384, dtype=torch.bfloat16, device=device_type, requires_grad=True
        )
        y = torch.ones(713268, dtype=torch.int64, device=device_type)
        x_ref = x.clone().detach().requires_grad_(True)
        y_ref = y.clone().detach()

        out, (_, bw_code) = run_fw_bw_and_get_code(lambda: torch.compile(f)(x, y))
        fc = FileCheck()
        fc.check("tl.atomic_add")
        fc.run(bw_code)

        self.assertEqual(f(x_ref, y_ref), out)

    def test_red_dtype_mismatch(self):
        # [FUNCTION DOC] Regression test for a reduction dtype mismatch: a graph ending in a max
        # over an int64 iota (with unused embedding and view ops before it) must match eager,
        # tested with persistent reductions on and off. Resets dynamo each iteration.
        # KNOWN ISSUE: when the loop turns persistent reductions off in the global inductor config
        #   it never restores it, so the setting leaks into later tests.
        for per in (True, False):
            torch._dynamo.reset()
            if not per:
                torch._inductor.config.triton.persistent_reductions = False

            def f(arg0_1, arg1_1):
                # [FUNCTION DOC] HELPER: Graph body that builds some unused embedding views and
                # returns only the maximum of an int64 iota of length 36 viewed as one row.
                # Returns a one-element tuple.
                embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1)
                view = torch.ops.aten.view.default(embedding, [64, 3072])
                unsqueeze = torch.ops.aten.unsqueeze.default(view, 0)
                expand = torch.ops.aten.expand.default(unsqueeze, [576, -1, -1])
                view_1 = torch.ops.aten.view.default(expand, [2, 8, 36, 64, 3072])
                permute = torch.ops.aten.permute.default(view_1, [0, 1, 3, 2, 4])
                clone = torch.ops.aten.clone.default(
                    permute, memory_format=torch.contiguous_format
                )
                view_2 = torch.ops.aten.view.default(clone, [2, 18432, 3072])
                iota = torch.ops.prims.iota.default(
                    36,
                    start=0,
                    step=1,
                    dtype=torch.int64,
                    device=device_type,
                    requires_grad=False,
                )
                view_3 = torch.ops.aten.view.default(iota, [1, 36])
                max_1 = torch.ops.aten.max.default(view_3)
                return (max_1,)

            x = torch.ones(1, 64, device=device_type, dtype=torch.int64)
            y = torch.randn(64, 3072, device=device_type, dtype=torch.bfloat16)
            out = f(x, y)
            self.assertEqual(torch.compile(f)(x, y), out)

    @skipCUDAIf(
        not SM90OrLater, "uses bfloat16 atomic add instrs which requires SM >= 90"
    )
    @unittest.skipIf(
        config.is_fbcode(),
        "bfloat16 atomic add is supported in fbcode, so we won't fallback",
    )
    def test_index_add_fallback(self):
        # [FUNCTION DOC] Check that outside fbcode the backward of a compiled bf16 index_select
        # falls back to the index_add kernel instead of Triton atomics, and the forward result
        # matches eager. Needs SM90 or later.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Select rows of x along dim 0 using the indices y.
            return torch.index_select(x, 0, y)

        x = torch.randn(
            2000, 384, dtype=torch.bfloat16, device=device_type, requires_grad=True
        )
        y = torch.ones(713268, dtype=torch.int64, device=device_type)
        x_ref = x.clone().detach().requires_grad_(True)
        y_ref = y.clone().detach()

        out, (_, bw_code) = run_fw_bw_and_get_code(lambda: torch.compile(f)(x, y))
        fc = FileCheck()
        fc.check("aten.index_add")
        fc.run(bw_code)

        self.assertEqual(f(x_ref, y_ref), out)

    @skipCUDAIf(
        not SM90OrLater, "uses bfloat16 atomic add instrs which requires SM >= 90"
    )
    @unittest.skipIf(
        config.is_fbcode(),
        "bfloat16 atomic add is supported in fbcode, so we won't fallback",
    )
    def test_index_add_fallback_direct(self):
        # [FUNCTION DOC] Compare compiled and eager index_add of bf16 rows at distinct random
        # indices (outside fbcode, SM90 or later).
        def f(x, idx, src):
            # [FUNCTION DOC] HELPER: Return x with the source rows added at the given indices
            # along dim 0.
            return torch.index_add(x, 0, idx, src)

        x = torch.randn(16, 256, dtype=torch.bfloat16, device=device_type)
        idx = torch.randperm(8, device=device_type)
        src = torch.randn(8, 256, dtype=torch.bfloat16, device=device_type)

        out = f(x, idx, src)
        compiled_out = torch.compile(f)(x, idx, src)
        self.assertEqual(out, compiled_out)

    @requires_multigpu()
    def test_not_initializing_wrong_device(self):
        # [FUNCTION DOC] Check that compiling and running forward and backward of a matmul on
        # device 1 does not allocate more memory on device 0 than before (peak active bytes).
        # Requires multiple GPUs.
        device_stats = torch.cuda.memory_stats("cuda:0")

        @torch.compile()
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Matrix multiply of x and y.
            return x @ y

        x = torch.rand([256, 256], device="cuda:1", requires_grad=True)
        y = torch.rand([256, 256], device="cuda:1", requires_grad=True)

        foo(x, y).sum().backward()

        device_stats2 = torch.cuda.memory_stats("cuda:0")
        self.assertTrue(
            device_stats2["active.all.peak"] <= device_stats["active.all.peak"]
        )

    @config.patch(
        {
            "triton.prefer_nd_tiling": True,
            "triton.max_tiles": 3,
        }
    )
    def test_3d_tiling(self):
        # [FUNCTION DOC] With nd tiling preferred and up to three tiles, add two strided views of
        # a 5D tensor and check the result equals the plain sum and the generated code uses a
        # third tile dimension (znumel). The block pointer and tile count variables are defined
        # but not used.
        full_size, view_size, num_block_pointers, num_tiles = (
            (5, 5, 5, 5, 5),
            (3, 3, 5, 3, 5),
            1,
            2,
        )
        GPU_TYPE = device_type

        def get_input() -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Create a random 5 by 5 by 5 by 5 by 5 tensor on the device
            # and return a strided view of shape 3, 3, 5, 3, 5 using the full tensor's strides.
            device = torch.device(GPU_TYPE)
            full = torch.randn(full_size).to(device)
            return torch.as_strided(full, view_size, full.stride())

        a, b = get_input(), get_input()

        opt_fn = torch.compile(functools.partial(torch.add))
        result, (code,) = run_and_get_code(opt_fn, a, b)
        self.assertEqual(result, a + b)
        self.assertIn("znumel", code)

    @unittest.skipIf(config.is_fbcode(), "Dependence on functorch.einops")
    def test_repeated_masked_load(self):
        # [FUNCTION DOC] Check that chunked nearest interpolation of a permuted, rearranged fp16
        # tensor (repeated masked loads) gives the eager result and that Dynamo records exactly
        # one unique graph. Clears the shared counters first and requires the Dynamo stats counter
        # to exist. Skipped in fbcode.
        counters.clear()

        target_size = (8, 2)
        mem_eff_temporal_upsampling_interp_chunks = 2
        from functorch.einops import rearrange

        x = torch.randn(1, 8, 12, 12, 4, dtype=torch.float16, device=device_type)
        x = x.permute(0, 1, 4, 2, 3)  # make non-contiguous
        x = rearrange(x, "b c t h w -> b c t (h w)")

        def interpolate_chunked(x):
            # [FUNCTION DOC] HELPER: Split x into two chunks along dim 1, interpolate each (as
            # float32, nearest mode, to size 8 by 2) and cast back to the original dtype, then
            # concatenate along dim 1.
            chunks = x.chunk(chunks=mem_eff_temporal_upsampling_interp_chunks, dim=1)
            r = []
            for t in chunks:
                r.append(
                    torch.nn.functional.interpolate(
                        t.float(), size=target_size, mode="nearest"
                    ).to(t.dtype)
                )
            return torch.cat(r, dim=1)

        out_eager = interpolate_chunked(x)
        out_compiled = torch.compile(interpolate_chunked)(x)

        self.assertEqual(out_eager, out_compiled)

        unique_graphs = counters["stats"].get("unique_graphs", None)
        self.assertIsNotNone(
            unique_graphs,
            "Expected Dynamo to record unique_graphs counter",
        )
        self.assertEqual(
            unique_graphs,
            1,
            "Repeated masked loads should compile to a single stable graph",
        )

    def test_max_autotune_nograd(self):
        # [FUNCTION DOC] Regression test (issue 155688): compile a model with default and with
        # max-autotune modes under no_grad and check both outputs match (previously max-autotune
        # raised a TypeError about a symbolic Identity not being a number).
        """
        https://github.com/pytorch/pytorch/issues/155688
        Smallest repro for max-autotune not working with no_grad
        Before adding __int__ function to torch.utils._sympy.functions.Identity,
        running the max_autotune mode would raise an error:
        TypeError: Expected a number but got Identity
        """

        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model made of five linear layers (inputs 4 to 8 wide, one output
            # each). Each step applies a linear and relu to the running tensor and concatenates
            # the result as an extra column.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the list of five linear layers.
                super().__init__()

                self.linear_layers = nn.ModuleList(
                    [
                        nn.Linear(4, 1, bias=True),
                        nn.Linear(5, 1, bias=True),
                        nn.Linear(6, 1, bias=True),
                        nn.Linear(7, 1, bias=True),
                        nn.Linear(8, 1, bias=True),
                    ]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: For each layer, apply the layer and relu to x and append
                # the result to x as a new column; return the final wider tensor.
                for layer in self.linear_layers:
                    x2 = layer(x)
                    x2 = F.relu(x2)
                    x = torch.cat((x, x2), dim=1)

                return x

        model = ToyModel().to(device_type)
        input_tensor = torch.randn((2, 4)).to(device_type)

        compile_default = torch.compile(model, mode="default")
        compile_max_autotune = torch.compile(model, mode="max-autotune")

        with torch.no_grad():
            default_output = compile_default(input_tensor)
            max_autotune_output = compile_max_autotune(input_tensor)

        self.assertEqual(default_output, max_autotune_output)

    @tf32_on_and_off(0.005)
    def test_adaptive_avg_pool3d_issue_157248(self):
        # [FUNCTION DOC] Regression test (issue 157248) for wrong results when a conv output is
        # unsqueezed to 5D and passed to adaptive average pooling in 3D. For five input sizes,
        # compares eager and inductor-compiled outputs under no_grad in eval mode, using subTest
        # per size. Runs with tf32 on and off at tolerance 0.005.
        """Test for GitHub issue #157248: Conv2d-unsqueeze-AdaptiveAvgPool3d produces incorrect results"""

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a 3 to 16 channel conv, followed by adding a dim at
            # position 1 and adaptive average pooling to 4 by 4 by 4.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the conv and pooling layers.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
                self.adaptive_pool = torch.nn.AdaptiveAvgPool3d((4, 4, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the conv, add a dim at position 1 (which produced
                # zero strides before the fix) and apply the pooling.
                x = self.conv(x)
                # This specific unsqueeze position was problematic due to zero strides
                x = x.unsqueeze(1)
                x = self.adaptive_pool(x)
                return x

        model = Model().to(device_type)
        model.eval()
        test_cases = [
            (1, 3, 8, 8),
            (2, 3, 16, 16),
            (1, 3, 32, 32),
            (1, 3, 15, 15),
            (2, 3, 13, 13),
        ]

        for batch, channels, h, w in test_cases:
            with self.subTest(input_shape=(batch, channels, h, w)):
                input_tensor = torch.randn(batch, channels, h, w, device=device_type)

                # Test eager mode
                with torch.no_grad():
                    eager_output = model(input_tensor)

                # Test compiled mode with inductor
                compiled_model = torch.compile(model, backend="inductor")
                with torch.no_grad():
                    compiled_output = compiled_model(input_tensor)

                # They should be identical (or very close)
                self.assertEqual(eager_output, compiled_output)

    @parametrize(
        "quantiles_shape,quantiles_strides,batch_size",
        [
            ((100, 10), (10, 1), 16),  # Contiguous C-order
            ((100, 10), (1, 100), 16),  # Transposed/F-order
            ((80, 12), (1, 80), 16),  # Transposed different size
            ((50, 20), (1, 50), 16),  # Transposed medium
            ((200, 8), (1, 200), 16),  # Transposed large x small
            ((25, 40), (1, 25), 16),  # Transposed small x large
            ((20, 5, 8), (40, 1, 5), 16),  # 3D case with mixed strides
            ((20, 5, 8), (1, 20, 100), 16),  # 3D case different stride order
        ],
    )
    def test_searchsorted_stride_permutations(
        self, quantiles_shape, quantiles_strides, batch_size
    ):
        # [FUNCTION DOC] Parametrized over shapes and strides of the sorted boundaries (including
        # transposed and 3D layouts) with batch size 16. Builds the boundaries as a strided view
        # of random data, sorts along dim 0, then checks a compiled fullgraph module that does a
        # transposed search gives the same result as eager under no_grad.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the transposed sorted boundaries as a frozen
            # parameter; forward runs a sorted search of the transposed input and transposes the
            # result back.
            def __init__(self, quantiles: torch.Tensor) -> None:
                # [FUNCTION DOC] HELPER: Check there is at least one boundary row (raises
                # AssertionError otherwise), transpose the boundaries and store them as a
                # parameter that does not require grad.
                super().__init__()
                if quantiles.shape[0] <= 0:
                    raise AssertionError(
                        f"Expected quantiles.shape[0] > 0, got {quantiles.shape[0]}"
                    )
                quantiles = quantiles.T
                self.q = torch.nn.Parameter(quantiles, requires_grad=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Search the transposed x in the stored boundaries and
                # return the transposed indices.
                return torch.searchsorted(self.q, x.T).T

        torch.manual_seed(42)

        # Create contiguous tensor first
        numel = 1
        for dim in quantiles_shape:
            numel *= dim
        data = torch.randn(numel, dtype=torch.float32, device=device_type)

        # Create tensor with specified shape and strides
        quantiles = torch.as_strided(
            data, size=quantiles_shape, stride=quantiles_strides
        )

        quantiles = torch.sort(quantiles, dim=0)[0]

        x_shape = (batch_size,) + quantiles_shape[1:]
        x = torch.randn(*x_shape, dtype=torch.float32, device=device_type)

        foo = Foo(quantiles)
        foo_compiled = torch.compile(Foo(quantiles), fullgraph=True)

        # Test eager vs compiled
        with torch.no_grad():
            eager = foo(x)
            compiled = foo_compiled(x)

        self.assertEqual(eager, compiled)

    def test_identity_load(self):
        # [FUNCTION DOC] Check that an index expression stays an identity-like load in compiled
        # code involving int32 and int64 concatenation and an arange: the code must contain the
        # expected simplified index pattern and the output equals eager.
        device = device_type

        def f(x, y):
            # [FUNCTION DOC] HELPER: Concatenate x without its first column with y plus a
            # constant, then combine with the last element as a third dim, add an arange and
            # reshape to a single row of 65536.
            y2 = torch.cat(
                [
                    x[:, 1:],
                    y[:, None] + 32 * 2048,
                ],
                dim=1,
            )

            x2 = x[:, 1:, None]
            y3 = y2[:, -1:, None]

            return (
                torch.cat([x2, y3], dim=1)
                + torch.arange(-2048, 0, device=device)[None, None, :]
            ).reshape(1, 32 * 2048)

        # This succeeds
        eager_out = f(
            torch.zeros(1, 32, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        # This crashes
        compile_out, code = run_and_get_code(
            torch.compile(f),
            torch.zeros(1, 32, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        # make sure the identity is maintained
        FileCheck().check("(1 + ((31)").run(code[0])

        self.assertEqual(eager_out, compile_out)

    @skipIfXpu(
        msg="_scaled_dot_product_efficient_attention returns log_sumexp in the query "
        "dtype instead of float32, tripping Inductor's fake kernel metadata check"
    )
    def test_qwen2_7b_sdpa_input_alignment_requires_recompile(self):
        # [FUNCTION DOC] Regression test for a Qwen2 7B repro: forward and backward of efficient
        # SDPA with a padded, expanded attention bias must remain correct with dynamic shapes when
        # a later call has a sequence length (6102) not divisible by 8, which requires a recompile
        # for alignment. Compares compiled against eager for both lengths. Skipped on XPU due to a
        # fake kernel metadata mismatch.
        # SDPA constraints ensures inputs have alignment (8).
        device = device_type

        def forward(q_proj, k_proj, attn_mask):
            # [FUNCTION DOC] HELPER: Build queries, repeated keys and values from projection
            # inputs, create an additive bias from the boolean mask (zeros and negative infinity)
            # padded to a multiple of 8 and expanded over heads, run the efficient SDPA forward
            # and then its backward with a zero gradient. Returns the backward outputs.
            scale = 0.08838834764831845  # 1/sqrt(128)

            B = attn_mask.size(0)
            S = attn_mask.size(3)
            D = 128
            d_model = q_proj.size(1)

            query_states = q_proj.view(B, S, -1, D).transpose(1, 2)  # [B, Hq, S, D]
            q = query_states.contiguous()

            Hkv = k_proj.size(1) // D
            Hq = query_states.size(1)

            nrepeats = Hq // Hkv
            key_states = k_proj.view(B, S, -1, D).transpose(1, 2)  # [B, Hkv, S, D]
            kv_repeated = key_states[:, :, None, :].expand(B, Hkv, nrepeats, S, D)
            kv_repeated = kv_repeated.contiguous()
            k = kv_repeated.reshape(B, Hq, S, D)
            v = k.clone()  # value tensor

            inf = torch.scalar_tensor(
                float("-inf"), dtype=torch.bfloat16, device=device
            )
            zero = torch.scalar_tensor(0.0, dtype=torch.bfloat16, device=device)
            where = torch.where(condition=attn_mask, input=zero, other=inf)
            pad_amount = 8 - (S % 8)
            padded = torch.nn.functional.pad(
                where, (0, pad_amount), value=0.0
            )  # pad last-dim
            sliced = padded[..., :S]  # back to [B,1,S,S]
            attn_bias = sliced.expand(B, Hq, S, S)

            sdpa_out, logsumexp, seed, offset = (
                torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q,
                    k,
                    v,
                    attn_bias,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                    compute_log_sumexp=True,
                )
            )

            zeros = torch.zeros(B, S, d_model, device=device, dtype=torch.bfloat16)
            zeros = zeros.reshape(B, S, Hq, D)
            grad_out = zeros.permute(0, 2, 1, 3)

            out = (
                torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(
                    grad_out,
                    q,
                    k,
                    v,
                    attn_bias,
                    sdpa_out,
                    logsumexp,
                    seed,
                    offset,
                    dropout_p=0.0,
                    scale=scale,
                    grad_input_mask=[True, True, True, False],
                )
            )
            return out

        B = 2
        S = 6144
        D = 128
        Hq = 28
        Hkv = 4

        example_inputs = (
            torch.randn((B * S, Hq * D), dtype=torch.bfloat16, device=device),  # q_proj
            torch.randn(
                (B * S, Hkv * D), dtype=torch.bfloat16, device=device
            ),  # k_proj
            torch.zeros((B, 1, S, S), dtype=torch.bool, device=device),  # attn_mask
        )
        correct = forward(*example_inputs)
        compiled = torch.compile(forward, dynamic=True)
        actual = compiled(*example_inputs)
        self.assertEqual(actual, correct)

        # run once more with seqlen that isn't divisible by 8
        S = 6102
        example_inputs = (
            torch.randn((S * B, Hq * D), dtype=torch.bfloat16, device=device),  # q_proj
            torch.randn(
                (S * B, Hkv * D), dtype=torch.bfloat16, device=device
            ),  # k_proj
            torch.zeros((B, 1, S, S), dtype=torch.bool, device=device),  # attn_mask
        )
        correct = forward(*example_inputs)
        actual = compiled(*example_inputs)
        self.assertEqual(actual, correct)

    @config.patch({"eager_numerics.division_rounding": True})
    def test_truediv_emulate_division_rounding(self):
        # [FUNCTION DOC] With division rounding emulation on, compare compiled division of 11 by 7
        # for all combinations of float16, bf16, float32 and float64 operand dtypes against eager,
        # comparing the exact decimal value of the results. Resets dynamo each iteration.
        from decimal import Decimal

        y, x = 7.0, 11.0

        @torch.compile
        def compiled_divide(x, y):
            # [FUNCTION DOC] HELPER: True division x over y.
            return x / y

        for y_dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            for x_dtype in [
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ]:
                y_ten = torch.tensor([y], dtype=y_dtype, device=device_type)
                x_ten = torch.tensor([x], dtype=x_dtype, device=device_type)

                torch._dynamo.reset()
                compiled_div = Decimal(compiled_divide(x_ten, y_ten).item())
                eager_div = Decimal((x_ten / y_ten).item())

                self.assertEqual(eager_div, compiled_div)

    @skipIfXpu(msg="triton dependency - torch-xpu-ops: 2554")
    @config.patch({"eager_numerics.division_rounding": False})
    @xfailIfROCm
    def test_truediv_base_not_bitwise_equivalent(self):
        # [FUNCTION DOC] With division rounding emulation off, a float32 division of 11 by 7
        # compiled by inductor gives a different value from eager and the generated code does not
        # use the round-to-nearest division intrinsic. Expected to fail on ROCm and skipped on
        # XPU.
        from decimal import Decimal

        y, x = 7.0, 11.0

        y_ten = torch.tensor([y], dtype=torch.float32, device=device_type)
        x_ten = torch.tensor([x], dtype=torch.float32, device=device_type)

        compile_out, code = run_and_get_code(
            torch.compile(lambda x, y: x / y),
            x_ten,
            y_ten,
        )
        compiled_div = Decimal(compile_out.item())
        eager_div = Decimal((x_ten / y_ten).item())

        self.assertNotEqual(eager_div, compiled_div)
        self.assertTrue("div_rn" not in code)

    @config.patch({"eager_numerics.disable_ftz": True})
    def test_disabling_ftz_yields_subnormals(self):
        # [FUNCTION DOC] With flush-to-zero disabled, check that a compiled float32 power of 2.0
        # to the -127 gives a positive (subnormal) result.
        from decimal import Decimal

        x = -127.0
        x_ten = torch.tensor([x], dtype=torch.float32, device=device_type)

        def fn(x):
            # [FUNCTION DOC] HELPER: Two raised to the power x.
            return 2.0**x

        compile_out = torch.compile(fn)(x_ten)
        compile_decimal = Decimal(compile_out.item())

        self.assertTrue(compile_decimal > Decimal(0))

    @skipIfXpu(msg="Decimal object comparison failed - torch-xpu-ops: 2810")
    @skipIfRocm(msg="ROCm preserves subnormals by default")
    @config.patch({"eager_numerics.disable_ftz": False})
    def test_not_disabling_ftz_yields_zero(self):
        # [FUNCTION DOC] With flush-to-zero not disabled, check that a compiled float32 power of
        # 2.0 to the -128 flushes to exactly zero. Skipped on XPU and ROCm.
        from decimal import Decimal

        x = -128.0
        x_ten = torch.tensor([x], dtype=torch.float32, device=device_type)

        def fn(x):
            # [FUNCTION DOC] HELPER: Two raised to the power x.
            return 2.0**x

        compile_out = torch.compile(fn)(x_ten)
        compile_decimal = Decimal(compile_out.item())

        self.assertEqual(compile_decimal, Decimal(0))

    @unittest.skipUnless(HAS_GPU, "requires GPU")
    def test_fused_slice_scatter_int32_const_overflow(self):
        # [FUNCTION DOC] End-to-end repro of an int32 address-constant overflow in a fused
        # slice_scatter backward plus matmul backward kernel over ten large inputs (9.5 million
        # rows). Asserts the sizes match the three triggering conditions, skips if the GPU lacks
        # the estimated free memory, then compiles the model without dynamic shapes, runs forward
        # and backward and always frees the tensors and empties the cache afterwards. Needs GPU.
        # End-to-end repro for the int32 address-constant overflow guarded
        # by SIMDKernelFeatures.any_index_expr_overflows_int32.
        # On unfixed builds the fused backward kernel raises
        #   ValueError('Scalar -2779057358 is out of range for type int32')
        # because slice_scatter.backward + mm.backward fuse into a kernel
        # whose simplified index expression contains a constant > 2**31.
        # Needs ~30 GB free GPU memory (leaves + grads); skipped otherwise.
        total_rows = 9_500_000
        slice_begin = 9_472_768
        slice_end = 9_499_648
        col_widths = [32, 32, 32, 32, 32, 32, 32, 32, 32, 22]
        total_cols = sum(col_widths)
        int32_max = 2**31 - 1

        # Three triggering conditions for the bug.
        self.assertLess(total_rows * max(col_widths), int32_max)
        self.assertLess((slice_end - slice_begin) * total_cols, int32_max)
        self.assertGreater(slice_begin * total_cols, int32_max)

        # ~2x leaf bytes for .grad, plus 1.3x safety margin.
        leaf_bytes = sum(total_rows * w * 4 for w in col_widths)
        need_gb = leaf_bytes * 2 / 1024**3 * 1.3
        free_gb = getattr(torch, device_type).mem_get_info()[0] / 1024**3
        if free_gb < need_gb:
            raise unittest.SkipTest(
                f"requires ~{need_gb:.1f} GB free GPU memory, have {free_gb:.1f} GB"
            )

        class SliceConcatDense(nn.Module):
            # [CLASS DOC] HELPER: Module with one square bias-free linear per column width;
            # forward slices a fixed row range from each input, concatenates the slices along
            # columns, applies each linear to its column block and returns the sum of all outputs
            # as a scalar.
            def __init__(self, widths):
                # [FUNCTION DOC] HELPER: Store the widths and create one square linear without
                # bias per width.
                super().__init__()
                self.widths = widths
                self.linears = nn.ModuleList(
                    [nn.Linear(w, w, bias=False) for w in widths]
                )

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Slice the same row range from every input, concatenate
                # along dim 1, apply each linear to its own block of columns and return the total
                # sum as a scalar.
                sliced = [t[slice_begin:slice_end, :] for t in inputs]
                cat = torch.cat(sliced, dim=1)
                out = cat.new_zeros(())
                col = 0
                for w, linear in zip(self.widths, self.linears):
                    out = out + linear(cat[:, col : col + w]).sum()
                    col += w
                return out

        torch._dynamo.reset()
        torch.manual_seed(0)
        inputs = [
            torch.randn(
                total_rows,
                w,
                device=device_type,
                dtype=torch.float32,
                requires_grad=True,
            )
            for w in col_widths
        ]
        model = SliceConcatDense(col_widths).to(device_type)
        compiled = torch.compile(model, dynamic=False)
        try:
            loss = compiled(inputs)
            loss.backward()
        finally:
            del inputs, model, compiled
            getattr(torch, device_type).empty_cache()

    @config.patch(
        {"triton.use_block_ptr": True, "triton.codegen_upcast_to_fp32": False}
    )
    def test_float16_reduction_with_int_output(self):
        # [FUNCTION DOC] With block pointers on and upcast to float32 off, a compiled argmax over
        # an fp16 input must not generate any conversion to float16, since the output is int64.
        @torch.compile
        def fn(input: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Index of the maximum along dim 0.
            return torch.argmax(input, dim=0)

        input = torch.randn(20, 20, device=device_type, dtype=torch.float16)
        _, code = run_and_get_code(fn, input)
        # There should not be any conversions to float16 in this code, since the input
        # is already float16 and the output is int64.
        self.assertNotIn(".to(tl.float16)", code[0])

    @config.patch("eager_numerics.division_rounding", True)
    def test_reciprocal_precision_rounding(self):
        # [FUNCTION DOC] With division rounding emulation on, check a compiled reciprocal of
        # float32 values matches eager through the common checker.
        # Test that reciprocal matches eager when division_rounding is enabled.
        # This requires OpDecompositions.reciprocal to use float32 constant so
        # that div_rn can be applied (the dtype check requires both operands float32).
        def fn(x):
            # [FUNCTION DOC] HELPER: Reciprocal of x.
            return torch.reciprocal(x)

        x = torch.randn(1000, device=device_type, dtype=torch.float32) + 0.1
        self.common(fn, [x])

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    @skipIfRocm(msg="PTX atan codegen is CUDA-specific")
    @skipIfXpu(msg="PTX atan codegen is CUDA-specific")
    def test_atan_special_psi_eager_parity(self):
        # [FUNCTION DOC] On CUDA only, check that compiled atan on special values (infinities,
        # NaN, signed zeros and a value that triggered a 1 ULP difference) is bitwise identical to
        # eager, and that a compiled atan followed by an in-place psi (digamma) is close to eager
        # and uses the approximate reciprocal instruction in the generated code.
        def atan_fn(x):
            # [FUNCTION DOC] HELPER: Arctangent of x.
            return torch.ops.aten.atan(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Apply atan, then write the psi function of that result into
            # the same tensor via the out argument and return it.
            out = atan_fn(x)
            return torch.ops.aten.special_psi(out, out=out)

        atan_x = torch.tensor(
            [
                -float("inf"),
                -10.0,
                -1.5516796112060547,
                -1.0,
                -0.0,
                0.0,
                1.0,
                1.5516796112060547,
                10.0,
                float("inf"),
                float("nan"),
            ],
            device=device_type,
        )
        actual_atan = torch.compile(atan_fn, backend="inductor", fullgraph=True)(atan_x)
        expected_atan = atan_fn(atan_x)
        self.assertEqual(
            actual_atan.view(torch.uint32),
            expected_atan.view(torch.uint32),
        )

        # Selected from the original seed-0 repro where a 1 ULP atan difference
        # was amplified by special_psi into an assert_close failure.
        x = torch.tensor([-1.5516796112060547], device=device_type)
        actual, code = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), x
        )
        expected = fn(x)

        torch.testing.assert_close(actual, expected)
        self.assertTrue(any("rcp.approx.ftz.f32" in src for src in code))

    def test_vector_norm_negative_dim_size_one(self):
        # [FUNCTION DOC] Regression test (issue 182181): vector norm over negative dims when those
        # dims have size one, via the tensor method and the linalg function, for one and two dims;
        # each compared to eager with the common checker.
        # Regression test for https://github.com/pytorch/pytorch/issues/182181
        def fn(x):
            # [FUNCTION DOC] HELPER: Norm over the last dim.
            return x.norm(dim=-1)

        x = torch.rand(8, 3, 1, device=device_type)
        self.common(fn, [x])

        # Multiple negative dims, all size 1
        def fn2(x):
            # [FUNCTION DOC] HELPER: Norm over the last two dims.
            return x.norm(dim=(-2, -1))

        y = torch.rand(8, 1, 1, device=device_type)
        self.common(fn2, [y])

        # torch.linalg.vector_norm directly
        def fn3(x):
            # [FUNCTION DOC] HELPER: Vector norm over the last dim using the linalg function.
            return torch.linalg.vector_norm(x, dim=-1)

        self.common(fn3, [x])

        def fn4(x):
            # [FUNCTION DOC] HELPER: Vector norm over the last two dims using the linalg function.
            return torch.linalg.vector_norm(x, dim=(-2, -1))

        self.common(fn4, [y])


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_GPU_AND_TRITON

    if HAS_GPU_AND_TRITON and not TEST_WITH_ASAN:
        run_tests(needs="filelock")
