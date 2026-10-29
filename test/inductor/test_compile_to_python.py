# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for torch._inductor.compile_to_python and load_from_python, which compile a
# decomposed ATen graph with inductor and return the generated Python source (plus an optional
# cache blob). Contains tracing and execution helpers, tiny test modules, golden-text tests of the
# generated entry point on CPU and CUDA (triton), and contract tests (argument validation, option
# pinning, cache behavior, graph partition form, dynamic shapes). No import-time side effects
# beyond imports.
import ast
import textwrap
from unittest import mock, skipIf

import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch._inductor import compile_to_python, config, load_from_python
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.standalone_compile import NoRunnableInductorModuleError
from torch._inductor.utils import fresh_cache
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import IS_BIG_GPU
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def _capture(m, x, tracing_mode="fake"):
    # [FUNCTION DOC] HELPER: Traces a module on an input into a flat-input ATen graph with make_fx
    # using the inductor decomposition table. Parameters and buffers are lifted to leading graph
    # inputs followed by the input tensor; the traced function re-parametrizes the module with the
    # supplied tensors (tied weights), runs it, and returns the flattened outputs. Tracing mode
    # defaults to fake and can be symbolic for dynamic dims. Runs the module once with grad
    # enabled, so pass a throwaway module. Returns the graph module.
    """Trace ``m(x)`` into a flat-input ATen graph (params+buffers then ``x`` lifted to
    inputs), mirroring how ``torch.compiler.precompile`` feeds a post-AOTAutograd inner
    graph to ``torch._inductor.compile_to_python``. The graph is decomposed against the
    inductor decomposition table (``compile_to_python`` drives inductor codegen directly
    and requires an already-decomposed graph) and traced under a single ``FakeTensorMode``
    (placeholders carry the fake ``val`` metadata inductor lowers against). Tracing runs
    ``m(x)`` once, so pass a throwaway module. ``tracing_mode="symbolic"`` produces a
    graph with dynamic dims."""
    pnames = [n for n, _ in m.named_parameters()]
    bnames = [n for n, _ in m.named_buffers()]
    pb = [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()]
    k = len(pnames)

    def flat_fn(flat):
        # [FUNCTION DOC] HELPER: Traced function: receives the flat list, rebuilds name to tensor
        # mappings for parameters (first entries) and buffers (next entries), calls the module on
        # the last entry under a stateless reparametrization, and returns the flattened list of
        # outputs.
        params = dict(zip(pnames, flat[:k]))
        buffers = dict(zip(bnames, flat[k : k + len(bnames)]))
        with stateless._reparametrize_module(
            m, {**params, **buffers}, tie_weights=True
        ):
            out = m(flat[-1])
        return pytree.tree_flatten(out)[0]

    with torch.enable_grad():
        return make_fx(
            flat_fn,
            decomposition_table=select_decomp_table(),
            tracing_mode=tracing_mode,
        )(pb + [x])


def _flat_inputs(m, x):
    # [FUNCTION DOC] HELPER: Returns the flat input list used with captured graphs: all
    # parameters, then all buffers, then the given input tensor.
    return (
        [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()] + [x]
    )


def _exec(src):
    # [FUNCTION DOC] HELPER: Compiles the given generated source text, executes it in a fresh
    # namespace named as a compiled module, and returns the module-level call entry point from
    # that namespace.
    ns = {"__name__": "_compiled"}
    exec(compile(src, "<compiled>", "exec"), ns)
    return ns["call"]


def _extract_call(src):
    # [FUNCTION DOC] HELPER: Parses the generated source and finds the function named call
    # (walking the whole syntax tree, including methods). Returns its source lines, dedented, with
    # a method-style signature rewritten to the flat single-argument form, so goldens do not
    # depend on whether graph partition wraps the body in a runner method. Raises AssertionError
    # if no such function exists.
    """Return the dedented source of the ``call`` entry point, normalized to the flat
    ``def call(args)`` signature. graph_partition (on by default in OSS, off in fbcode)
    wraps the body in a ``Runner.call(self, args)`` method; the body is byte-identical
    either way, so normalizing the signature makes the golden independent of the
    graph_partition default. The expect goldens lock this runtime entry point only: the
    rest of the emitted module (imports, the inert compile-time auto-tuning docstring, the
    ``# AOT ID`` global-counter comment) carries build- and ordering-dependent noise that
    should not be goldened."""
    mod = ast.parse(src)
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == "call":
            body = "\n".join(src.split("\n")[node.lineno - 1 : node.end_lineno])
            body = textwrap.dedent(body)
            return body.replace("def call(self, args):", "def call(args):", 1)
    raise AssertionError("generated module has no module-level def call")


class _Pointwise(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module applying a scale, shift and relu, which fuses into one
    # pointwise kernel.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns relu of the input times 2 plus 1.
        return torch.relu(x * 2.0 + 1.0)


class _SumDim1(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module that reduces the input by summing over dimension 1.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the sum of the input over dimension 1.
        return x.sum(dim=1)


class _Softmax(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module computing softmax over the last dimension.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns softmax of the input over the last dimension.
        return torch.softmax(x, dim=-1)


class TestInductorCompileToPythonCodegen(TestCase):
    # [CLASS DOC] Golden tests on CPU for the generated call body under the default inductor
    # config. Each test captures a flat graph, compiles to Python, compares the extracted entry
    # point with an expected string, and checks numerics against eager.
    # These golden the runtime ``call`` body emitted under DEFAULT inductor config (no
    # option overrides), so the test reflects what callers get out of the box. The
    # extern-kernel / CPU codegen body is deterministic and build-independent (CPU tensors);
    # assert_size_stride lines come from the default size_asserts=True (the same default the
    # rest of the inductor golden suite relies on). ``_extract_call`` normalizes the
    # entry-point signature, so the golden is the same whether graph_partition wraps the body
    # in a ``Runner`` method (OSS default) or emits a flat top-level ``def call`` (fbcode).
    def _inner_call(self, m, x):
        # [FUNCTION DOC] HELPER: Captures the module as a flat graph, compiles it to Python, and
        # returns the full source together with the extracted normalized call entry point source.
        # The cache return value is discarded.
        gm = _capture(m, x)
        src, _cache = compile_to_python(gm, _flat_inputs(m, x))
        return src, _extract_call(src)

    def test_addmm_extern_kernel_codegen(self):
        # [FUNCTION DOC] Linear layer with bias: the golden shows a single extern addmm kernel
        # call with size/stride assertions and buffer frees. Also checks the executed source
        # equals the eager output without grad.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    flat_1, flat_2, flat_3 = args
    args.clear()
    assert_size_stride_grouped((flat_2, flat_3, flat_1), ((3, ), (5, 4), (3, 4)), ((1, ), (4, 1), (4, 1)), 'input')
    buf0 = empty_strided_cpu((5, 3), (3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    extern_kernels.addmm(flat_2, flat_3, reinterpret_tensor(flat_1, (4, 3), (1, 4), 0), alpha=1, beta=1, out=buf0)
    del flat_1
    del flat_2
    del flat_3
    return (buf0, )""",
        )
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_matmul_extern_kernel_codegen(self):
        # [FUNCTION DOC] Linear layer without bias: the golden shows a single extern mm call on a
        # reinterpreted weight. Also checks the executed source equals eager output.
        m = torch.nn.Linear(4, 3, bias=False).eval()
        x = torch.randn(5, 4)
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    flat_1, flat_2 = args
    args.clear()
    assert_size_stride_grouped((flat_2, flat_1), ((5, 4), (3, 4)), ((4, 1), (4, 1)), 'input')
    buf0 = empty_strided_cpu((5, 3), (3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    extern_kernels.mm(flat_2, reinterpret_tensor(flat_1, (4, 3), (1, 4), 0), out=buf0)
    del flat_1
    del flat_2
    return (buf0, )""",
        )
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_addmm_relu_fused_pointwise_codegen(self):
        # [FUNCTION DOC] Linear plus relu: the golden shows an extern addmm followed by a fused
        # C++ pointwise kernel reusing the output buffer. Also checks numerics against eager.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    flat_1, flat_2, flat_3 = args
    args.clear()
    assert_size_stride_grouped((flat_2, flat_3, flat_1), ((3, ), (5, 4), (3, 4)), ((1, ), (4, 1), (4, 1)), 'input')
    buf0 = empty_strided_cpu((5, 3), (3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    extern_kernels.addmm(flat_2, flat_3, reinterpret_tensor(flat_1, (4, 3), (1, 4), 0), alpha=1, beta=1, out=buf0)
    del flat_1
    del flat_2
    del flat_3
    buf1 = buf0; del buf0  # reuse
    cpp_fused_0(buf1)
    return (buf1, )""",
        )
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_benchmark_harness_suppressed(self):
        # [FUNCTION DOC] Checks that the emitted source for a small linear layer does not contain
        # any profiling harness entry points (benchmark module function, argument getter, main
        # driver, performance printer), because benchmark_harness is pinned off.
        # #187858 pins benchmark_harness=False, so the emitted module is runnable rather
        # than an Inductor profiling harness: none of these debug entry points appear.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _call_src = self._inner_call(m, x)
        for marker in (
            "benchmark_compiled_module",
            "def get_args(",
            "compiled_module_main",
            "print_performance",
        ):
            self.assertNotIn(marker, src)

    def test_no_runnable_module_for_no_compute(self):
        # [FUNCTION DOC] A module that returns its input does no compute, so compile_to_python is
        # expected to raise NoRunnableInductorModuleError.
        # A graph with no compute (returns its input unchanged) lowers to no module-level
        # ``call``, so there is nothing runnable to inline.
        m = torch.nn.Identity().eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with self.assertRaises(NoRunnableInductorModuleError):
            compile_to_python(gm, _flat_inputs(m, x))


@requires_cuda_and_triton
class TestInductorCompileToPythonCudaCodegen(TestCase):
    # [CLASS DOC] Requires CUDA and triton. Same idea as the CPU golden class but for triton
    # kernels: the call body is goldened (arch independent) and the kernel text is checked with
    # substring assertions. Device 0 is assumed.
    # On CUDA, compile_to_python emits actual @triton.jit kernels rather than the CPU
    # extern_kernels / cpp_fused path the sibling class goldens. The expect goldens lock
    # the ``call()`` body only: it carries NO autotuning artifacts -- XBLOCK / num_warps
    # are chosen inside ``.run`` at launch time and the arch-specific DeviceProperties
    # live in the kernel decorator, not in ``call`` -- so it is arch-independent. The
    # hardware- and Triton-version-dependent kernel body is instead checked structurally
    # (assertIn). The device ordinal is hardcoded to 0 (these tests run on device 0, as
    # the rest of the inductor codegen-golden suite does). Default inductor config is used
    # (no option overrides); ``_extract_call`` normalizes the entry-point signature so the
    # golden is independent of the graph_partition default (Runner method in OSS, flat
    # ``def call`` in fbcode).
    def _inner_call(self, m, x):
        # [FUNCTION DOC] HELPER: Captures the module as a flat graph, compiles it to Python and
        # returns the source with the extracted normalized call entry point source.
        gm = _capture(m, x)
        src, _cache = compile_to_python(gm, _flat_inputs(m, x))
        return src, _extract_call(src)

    def test_pointwise_triton_kernel_codegen(self):
        # [FUNCTION DOC] Pointwise module on a CUDA tensor: golden call body launches a single
        # pointwise triton kernel under a device guard. Source must contain the triton jit
        # decorator, the pointwise heuristic decorator, the generic kernel name, load, store and
        # maximum operations, and the call body must have no extern kernels. Numerics are checked
        # against eager.
        m = _Pointwise().eval().cuda()
        x = torch.randn(128, 64, device="cuda")
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    flat_1, = args
    args.clear()
    assert_size_stride(flat_1, (128, 64), (64, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        flat_1 = copy_if_misaligned(flat_1)
        buf0 = empty_strided_cuda((128, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(flat_1, buf0, 8192, stream=raw_stream0)
        del flat_1
    return (buf0, )""",
        )
        # The pointwise fusion lowers to a single @triton.jit pointwise kernel; the
        # call drives it directly with no extern (BLAS/cuDNN) kernel.
        self.assertIn("@triton.jit", src)
        self.assertIn("@triton_heuristics.pointwise", src)
        # Kernel name is generic (triton_poi_fused_0) because the make_fx test graph
        # carries no source-node provenance; a real AOTAutograd graph would name it after
        # the fused ops (e.g. triton_poi_fused_add_mul_relu_0).
        self.assertIn("def triton_poi_fused_0(", src)
        self.assertIn("tl.load", src)
        self.assertIn("tl.store", src)
        self.assertIn("tl.maximum", src)  # the fused relu
        self.assertNotIn("extern_kernels", call_src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_reduction_triton_kernel_codegen(self):
        # [FUNCTION DOC] Row sum on CUDA: golden call body launches one persistent reduction
        # triton kernel. Source must contain the persistent reduction decorator, the generic
        # kernel name and a sum operation; no extern kernels in the call body. Numerics checked
        # against eager.
        m = _SumDim1().eval().cuda()
        x = torch.randn(64, 256, device="cuda")
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    flat_1, = args
    args.clear()
    assert_size_stride(flat_1, (64, 256), (256, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        flat_1 = copy_if_misaligned(flat_1)
        buf0 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_0.run(flat_1, buf0, 64, 256, stream=raw_stream0)
        del flat_1
    return (buf0, )""",
        )
        # A row reduction lowers to a (persistent) reduction Triton kernel doing the
        # cross-row ``tl.sum``; there is no extern kernel.
        self.assertIn("@triton_heuristics.persistent_reduction", src)
        # Generic kernel name (no source-node provenance on the make_fx test graph); a real
        # AOTAutograd graph would name it triton_per_fused_sum_0.
        self.assertIn("def triton_per_fused_0(", src)
        self.assertIn("tl.sum", src)
        self.assertNotIn("extern_kernels", call_src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_addmm_relu_fused_triton_epilogue_codegen(self):
        # [FUNCTION DOC] Linear plus relu on CUDA: golden shows an extern mm call followed by a
        # fused triton add kernel that carries the epilogue. Checks the call body has the extern
        # mm, the source has the triton jit decorator, the add kernel name and a maximum
        # operation, and numerics match eager.
        # CUDA counterpart of test_addmm_relu_fused_pointwise_codegen: the matmul is an
        # extern BLAS call but the relu epilogue fuses into a @triton.jit kernel
        # (triton_poi_fused_add_0 here; a real AOTAutograd graph, with source-node
        # provenance, would name it triton_poi_fused_addmm_relu_0) rather than the CPU
        # cpp_fused_relu_0.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval().cuda()
        x = torch.randn(5, 4, device="cuda")
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    flat_1, flat_2, flat_3 = args
    args.clear()
    assert_size_stride_grouped((flat_3, flat_1), ((5, 4), (3, 4)), ((4, 1), (4, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        flat_3 = copy_if_misaligned(flat_3)
        flat_1 = copy_if_misaligned(flat_1)
        buf0 = empty_strided_cuda((5, 3), (3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(flat_3, reinterpret_tensor(flat_1, (4, 3), (1, 4), 0), out=buf0)
        del flat_1
        del flat_3
        assert_size_stride(flat_2, (3, ), (1, ), 'input')
        flat_2 = copy_if_misaligned(flat_2)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_0.run(buf1, flat_2, 15, stream=raw_stream0)
        del flat_2
    return (buf1, )""",
        )
        self.assertIn("extern_kernels.mm", call_src)
        self.assertIn("@triton.jit", src)
        self.assertIn("def triton_poi_fused_add_0(", src)
        self.assertIn("tl.maximum", src)  # the fused relu epilogue
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_softmax_fused_reduction_triton_kernel(self):
        # [FUNCTION DOC] Softmax on CUDA: no golden. Checks the source contains the triton jit
        # decorator, persistent reduction heuristic, a sum, and an exp call, that the call body
        # has no extern kernels, and that numerics match eager.
        # softmax is the canonical multi-stage fusion: max, subtract, exp, sum, divide
        # all collapse into ONE persistent-reduction Triton kernel. Its exact name
        # (the decomposition route) varies, so this checks structure + numerics rather
        # than goldening the call body.
        m = _Softmax().eval().cuda()
        x = torch.randn(32, 128, device="cuda")
        src, call_src = self._inner_call(m, x)
        self.assertIn("@triton.jit", src)
        self.assertIn("@triton_heuristics.persistent_reduction", src)
        self.assertIn("tl.sum", src)  # the denominator reduction
        # the exp numerator: libdevice.exp / tl_math.exp / tl.exp depending on fast-math
        self.assertIn(".exp(", src)
        self.assertNotIn("extern_kernels", call_src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    @skipIf(
        not IS_BIG_GPU,
        "Skipping triton backend only since not big GPU (not enough SMs for "
        "max_autotune_gemm; the TRITON-only GEMM backend then has no choices)",
    )
    def test_max_autotune_excludes_benchmark_lowerings(self):
        # [FUNCTION DOC] Skipped unless on a big GPU. Compiles a large-K linear layer with max
        # autotune on and the triton-only GEMM backend inside a fresh cache, while a spy wraps the
        # subgraph choice benchmark-compile method to count calls. Asserts the benchmarking
        # compiles ran, the final source has a call function and contains no benchmark_ names, and
        # numerics match eager with loose tolerances.
        # max_autotune's decompose_k GEMM choice compiles each k-split as a nested
        # SubgraphChoiceCaller benchmark GraphLowering during autotuning. compile_to_python
        # reads the FINAL module's source off the compiled artifact, so those benchmark
        # modules are excluded by construction -- they never become the artifact, no
        # filtering needed. fresh_cache forces a cold autotune cache so the benchmark
        # compiles actually run, and the spy asserts they did, so the test cannot silently
        # degrade into a no-op. k >= 32*m and k >= 32*n makes decompose_k a candidate.
        from torch._inductor.codegen.subgraph import SubgraphChoiceCaller

        m = (
            torch.nn.Sequential(torch.nn.Linear(8192, 64, bias=False), torch.nn.ReLU())
            .eval()
            .cuda()
        )
        x = torch.randn(64, 8192, device="cuda")
        gm = _capture(m, x)
        orig = SubgraphChoiceCaller._compile_for_benchmarking
        bench_calls = []

        def _spy(choice, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Spy replacing the subgraph-choice benchmark compile method:
            # records one entry in the enclosing list per call, then delegates to the original
            # method with the same arguments and returns its result.
            bench_calls.append(1)
            return orig(choice, *args, **kwargs)

        with fresh_cache():
            with mock.patch.object(
                SubgraphChoiceCaller, "_compile_for_benchmarking", _spy
            ):
                src, _cache = compile_to_python(
                    gm,
                    _flat_inputs(m, x),
                    options={
                        "max_autotune": True,
                        "max_autotune_gemm_backends": "TRITON",
                    },
                )
        self.assertGreater(len(bench_calls), 0)  # benchmark lowerings actually ran
        self.assertIn("def call(", src)  # the final runnable module
        self.assertNotIn(
            "benchmark_", src
        )  # excluded by construction, never in the artifact
        with torch.no_grad():
            self.assertEqual(
                _exec(src)(_flat_inputs(m, x))[0], m(x), atol=1e-2, rtol=1e-2
            )

    @config.patch({"compile_threads": 1})
    def test_warm_load_without_eager_static_launcher_rehydration(self):
        # [FUNCTION DOC] Uses a single compile thread. Skips unless the FX graph cache and static
        # CUDA launcher are enabled. Compiles a pointwise module, checks the cache is bytes, then
        # in a fresh cache loads the source with the cache blob and runs it, asserting the bundler
        # counter for eagerly rehydrated static autotuners is zero and the output equals eager.
        # A cold load should use JIT instead of eagerly rehydrating the static launcher.
        if config.force_disable_caches or not config.fx_graph_cache:
            self.skipTest("requires inductor FxGraphCache enabled")
        if not config.use_static_cuda_launcher:
            self.skipTest("requires the static CUDA launcher")
        m = _Pointwise().eval().cuda()
        x = torch.randn(1024, 1024, device="cuda")
        src, cache = compile_to_python(_capture(m, x), _flat_inputs(m, x))
        self.assertIsInstance(cache, bytes)
        with fresh_cache():
            counters.clear()
            with torch.no_grad():
                out = load_from_python(src, cache)(_flat_inputs(m, x))
            rehydrated = counters["inductor"]["triton_bundler_load_static_autotuner"]
        self.assertEqual(rehydrated, 0)
        self.assertEqual(out[0], m(x))


class TestInductorCompileToPythonContract(TestCase):
    # [CLASS DOC] CPU-only contract and branch tests for compile_to_python and load_from_python:
    # input validation, option pin overrides, cache return, graph partition runner form, dynamic
    # shapes and warm-cache reuse.
    # Contract + branch coverage the codegen-golden classes above do not exercise: the
    # option pin-override, the cache return value, the graph_partition runner form, the
    # no-compute error path, and warm-cache reuse. All CPU, so no GPU is required.
    def test_rejects_non_graphmodule(self):
        # [FUNCTION DOC] Passing a string instead of a graph module must raise TypeError.
        with self.assertRaises(TypeError):
            compile_to_python("not a graph module", [])

    def test_pins_override_conflicting_user_options(self):
        # [FUNCTION DOC] Passing options that request the C++ wrapper and benchmark harness must
        # be overridden: the source still has a call function but no C-linkage text, AOTInductor
        # model class name or benchmark function, and the executed source matches eager.
        # The benchmark_harness/cpp_wrapper pins must beat conflicting user options so the
        # captured module stays the runnable python wrapper rather than a C++ wrapper or a
        # profiling harness. ``def call(`` matches both the flat ``def call(args)`` and the
        # graph_partition ``Runner.call(self, args)`` form, so this does not pin partitioning.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _cache = compile_to_python(
            _capture(m, x),
            _flat_inputs(m, x),
            options={
                "cpp_wrapper": True,
                "benchmark_harness": True,
            },
        )
        self.assertIn("def call(", src)
        self.assertNotIn('extern "C"', src)
        self.assertNotIn("AOTInductorModel", src)
        self.assertNotIn("benchmark_compiled_module", src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_load_from_python_standalone_and_warm(self):
        # [FUNCTION DOC] Skips unless the FX graph cache is enabled and caches are not force-
        # disabled. Compiles a linear layer and checks the cache is bytes. In separate fresh cache
        # directories, loads the source with no cache and then with the cache blob; both loaded
        # callables must match eager.
        # load_from_python(python_code, cache) is the inverse of compile_to_python. The
        # python_code is self-contained: it loads and runs with cache=None (JIT path). The
        # cache is a PURE ACCELERATOR -- passing it warms the kernel caches so exec loads
        # precompiled binaries instead of recompiling; both paths must match eager. Run each
        # in a fresh cache dir so the standalone path is a genuine cold load. The cache
        # bytes require force_disable_caches off AND fx_graph_cache on; those flags are
        # env-authoritative on PyTorch CI's cache-disabled shards (cannot be patched back
        # on), where compile_to_python returns None -- skip there (test_no_cache_when_caches
        # _disabled covers the None path).
        if config.force_disable_caches or not config.fx_graph_cache:
            self.skipTest("requires inductor FxGraphCache enabled")
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, cache = compile_to_python(_capture(m, x), _flat_inputs(m, x))
        self.assertIsInstance(cache, bytes)
        # Standalone: no cache, the module JIT-compiles its own kernels.
        with fresh_cache(), torch.no_grad():
            self.assertEqual(load_from_python(src)(_flat_inputs(m, x))[0], m(x))
        # Warm: the cache accelerates the same module; result is identical.
        with fresh_cache(), torch.no_grad():
            self.assertEqual(load_from_python(src, cache)(_flat_inputs(m, x))[0], m(x))

    def test_no_cache_when_caches_disabled(self):
        # [FUNCTION DOC] With force_disable_caches passed as an option, the returned cache must be
        # None while the returned source still runs and matches eager.
        # With caches disabled there is no saveable artifact, so cache is None; the source
        # still runs (the kernels JIT-compile from it on first call).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, cache = compile_to_python(
            _capture(m, x),
            _flat_inputs(m, x),
            options={"force_disable_caches": True},
        )
        self.assertIsNone(cache)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_graph_partition_runner_call_form(self):
        # [FUNCTION DOC] With graph partition requested, the source must contain the runner call
        # assignment form, and the executed module must still match eager.
        # graph_partition=True emits the Runner form (``call = runner.call``) instead of a
        # top-level ``def call``; the returned source must carry it and the emitted module
        # must still run.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _cache = compile_to_python(
            _capture(m, x), _flat_inputs(m, x), options={"graph_partition": True}
        )
        self.assertIn("call = runner.call", src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_dynamic_shapes_emits_symbolic_codegen(self):
        # [FUNCTION DOC] Captures a symbolic-mode graph; the extracted call body must mention a
        # symbolic size name and must not contain the concrete input shape. The same compiled
        # callable is then run at several batch sizes and each result must match eager.
        # A symbolically-traced graph carries symbolic sizes in its placeholder val
        # metadata, so the emitted call() is keyed on symbolic sizes (sN) rather than baked
        # constants and the single module runs at multiple shapes. Shapes come from the
        # graph (there is no dynamic_shapes knob); the other tests exercise the static path.
        # Symbol names are non-deterministic, so assert structure + multi-shape numerics
        # rather than goldening.
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        gm = _capture(m, x, tracing_mode="symbolic")
        src, _cache = compile_to_python(gm, _flat_inputs(m, x))
        call_src = _extract_call(src)
        self.assertRegex(call_src, r"\bs\d+\b")  # a symbolic size symbol is present
        self.assertNotIn("(8, 4)", call_src)  # the input shape is not baked in
        fn = _exec(src)
        for n in (8, 16, 5):
            xi = torch.randn(n, 4)
            with torch.no_grad():
                self.assertEqual(fn(_flat_inputs(m, xi))[0], m(xi))

    @config.patch({"compile_threads": 1})
    def test_warm_cache_still_yields_source(self):
        # [FUNCTION DOC] Uses a single compile thread; skips if caching is unavailable. In a fresh
        # cache, compiles the same graph twice, asserting zero FX graph cache hits after the first
        # and exactly one after the second, that both sources contain a call function, and that
        # the second source runs and matches eager.
        # On a warm cache the 2nd compile of the same graph is an FxGraphCache hit (no fresh
        # codegen), yet compile_to_python still returns a runnable module: Inductor
        # populates source_code on the restored artifact too. fresh_cache isolates an empty
        # cache dir so the first compile is a guaranteed miss and the second a guaranteed
        # hit; compile_threads=1 keeps codegen in-process so the hit counter is
        # deterministic. The warm path needs caching, env-disabled on some CI shards
        # (force_disable_caches cannot be patched back on), so skip there.
        if config.force_disable_caches or not config.fx_graph_cache:
            self.skipTest("requires inductor FxGraphCache enabled")
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with fresh_cache():
            counters.clear()
            src1, _ = compile_to_python(_capture(m, x), _flat_inputs(m, x))
            hits_after_first = counters["inductor"]["fxgraph_cache_hit"]
            src2, _ = compile_to_python(_capture(m, x), _flat_inputs(m, x))
            hits_after_second = counters["inductor"]["fxgraph_cache_hit"]
        self.assertEqual(hits_after_first, 0)
        self.assertEqual(hits_after_second, 1)
        self.assertIn("def call(", src1)
        self.assertIn("def call(", src2)
        with torch.no_grad():
            self.assertEqual(_exec(src2)(_flat_inputs(m, x))[0], m(x))


if __name__ == "__main__":
    run_tests()
