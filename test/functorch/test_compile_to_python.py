# Owner(s): ["oncall: pt2"]
# [MODULE DOC]
# End-to-end and unit tests for compile_to_python / load_from_python, which compose
# AOTAutograd's codegen'd runtime wrappers around an Inductor call into one standalone
# Python module. Defines module fixtures and helpers (graph capture, flat input ordering,
# source exec, composed-module marker checks), CPU end-to-end tests comparing against eager,
# CUDA mirrors (need CUDA and Triton), unit tests of the composer helpers, and guard tests
# that hand-build generated-source records to check that bad shapes are rejected. Importing
# registers parametrized tests; nothing else runs at import time.
import ast
import unittest

import torch
import torch._functorch.config as functorch_config
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._functorch._aot_autograd.codegen import GeneratedSource
from torch._functorch._aot_autograd.to_standalone_python import (
    _compose_standalone_module,
    _find_effectful_op,
    _known_helper_table,
    _module_level_names,
)
from torch._functorch.aot_autograd import compile_to_python, load_from_python
from torch._higher_order_ops.effects import _get_effect, hop_print
from torch._inductor.utils import fresh_cache
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def _capture(m, x, tracing_mode="real"):
    # [FUNCTION DOC] HELPER: Traces a module call on one input into a flat-input ATen graph
    # with make_fx under enable_grad. The traced function takes one list holding
    # parameters, then buffers, then the input, rebinds the module state from that list via
    # a stateless reparametrization (tied weights), runs the module on the last element and
    # returns the flattened outputs. Returns the traced GraphModule; tracing_mode selects
    # real or symbolic shapes.
    """Trace ``m(x)`` into a flat-input ATen graph (params+buffers then ``x`` lifted to
    inputs), the same shape ``torch.compiler.precompile`` feeds the AOT lowering. The
    flat-input ordering returned by ``_flat_inputs`` MUST match this order."""
    pnames = [n for n, _ in m.named_parameters()]
    bnames = [n for n, _ in m.named_buffers()]
    pb = [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()]
    k = len(pnames)

    def flat_fn(flat):
        # [FUNCTION DOC] HELPER: Traced function over a flat list: splits parameter and buffer
        # slices by name, runs the module under reparametrized state on the last list element, and
        # returns the flattened output leaves.
        params = dict(zip(pnames, flat[:k]))
        buffers = dict(zip(bnames, flat[k : k + len(bnames)]))
        with stateless._reparametrize_module(
            m, {**params, **buffers}, tie_weights=True
        ):
            out = m(flat[-1])
        return pytree.tree_flatten(out)[0]

    with torch.enable_grad():
        return make_fx(flat_fn, tracing_mode=tracing_mode)(pb + [x])


def _flat_inputs(m, x):
    # [FUNCTION DOC] HELPER: Returns a new list with the module parameters, then its buffers, then
    # the input, in the order the captured graph expects.
    return (
        [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()] + [x]
    )


def _exec(src):
    # [FUNCTION DOC] HELPER: Compiles the given source text under a fake file name, executes it in
    # a fresh namespace named for a compiled module, and returns the entry function bound to the
    # name call in that namespace. Raises KeyError if the source does not define it.
    ns = {"__name__": "_compiled"}
    exec(compile(src, "<compiled>", "exec"), ns)
    return ns["call"]


# A stand-in for the authoritative inner-call placeholder that compile_to_python threads
# into _compose_standalone_module. The guard tests below build wrappers by hand and never
# reference the real inner call, so any distinct object serves as its identity.
_SENTINEL_INNER_CALL = object()


def _make_holder(value):
    # [FUNCTION DOC] HELPER: Returns its argument unchanged; a plain importable module-level
    # callable used as a baked global in a guard test.
    # A plain importable module-level callable used by test_unwired_chain_wrapper_rejected
    # as a stand-in inner-ref global, so it resolves cleanly as source (isolating the
    # "unwired wrapper" rejection from a resolution failure).
    return value


class _NewObjEx:
    # [CLASS DOC] HELPER: Fixture class built through a custom construction hook that stores two
    # fields and exposes constructor args plus keyword args for reduction, so its baked-global
    # source uses the rebuild helper. Equality compares type and both fields.
    # A baked-global fixture whose reduce (__getnewargs_ex__ + dict state) emits a
    # ``_rebuild(...)`` call, used by test_rebuild_helper_spliced_and_runs_in_composed_module
    # to check the composed module splices and runs the _rebuild helper.
    def __new__(cls, a, b):
        # [FUNCTION DOC] HELPER: Creates the instance without running an initializer and stores
        # the two arguments as fields a and b.
        obj = object.__new__(cls)
        obj.a = a
        obj.b = b
        return obj

    def __getnewargs_ex__(self):
        # [FUNCTION DOC] HELPER: Returns the positional-args tuple holding field a and the keyword
        # mapping holding field b, used by pickle-style reduction.
        return ((self.a,), {"b": self.b})

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal when the other object is the same fixture type
        # and both fields match.
        return isinstance(other, _NewObjEx) and self.a == other.a and self.b == other.b


class _Pointwise(torch.nn.Module):
    # [CLASS DOC] HELPER: Module computing a relu of the input scaled by two plus one.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns relu applied to twice the input plus one.
        return torch.relu(x * 2.0 + 1.0)


class _ViewAlias(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose output is a flattened view that aliases the input.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input viewed as one flat dimension.
        return x.view(-1)


class _SumDim1(torch.nn.Module):
    # [CLASS DOC] HELPER: Module reducing the input by summing along dimension one.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the sum over dimension one.
        return x.sum(dim=1)


class _MultiOut(torch.nn.Module):
    # [CLASS DOC] HELPER: Module returning three outputs: doubled input, row sums, and
    # relu of the input.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns a tuple of the doubled input, the sum along dimension
        # one, and the relu of the input.
        return x * 2.0, x.sum(dim=1), torch.relu(x)


class _BufferMutate(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a zero-initialized buffer of four elements that is mutated
    # in place on every forward, used to exercise mutation copy-back.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the base module and registers a buffer named
        # b holding four zeros.
        super().__init__()
        self.register_buffer("b", torch.zeros(4))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Adds the scalar sum of the input to the buffer in place and
        # returns the input plus the updated buffer.
        self.b.add_(x.sum())
        return x + self.b


class _MatMul(torch.nn.Module):
    # [CLASS DOC] HELPER: Module wrapping a linear layer from four to three features,
    # whose addmm is autocast-sensitive.
    # addmm is an autocast-to-bf16 op, so a float32 input under torch.autocast engages
    # autocast and bakes the casts into the graph -- the setup the _DisableAutocast_ test
    # needs (a Linear's addmm behaves the same way).
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the base module and creates the linear layer from
        # four to three features.
        super().__init__()
        self.l = torch.nn.Linear(4, 3)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the linear layer to the input.
        return self.l(x)


def _compose(m, x):
    # [FUNCTION DOC] HELPER: Captures the module graph for the example input and returns the
    # result of compile_to_python on it with the flat inputs, i.e. the pair of composed source
    # and optional cache bytes.
    gm = _capture(m, x)
    return compile_to_python(gm, _flat_inputs(m, x))


def _assert_composed(test, src):
    # [FUNCTION DOC] HELPER: Asserts on the composed source text: it defines the flat-input
    # entry, captures the inner call, inlines the runtime wrapper as a def invoked by name, and
    # contains no pickle load, raw-storage loader or base64 blobs. Raises assertion failures on
    # the passed test case.
    # Structural markers proving this is the COMPOSED module (not just the inner inductor
    # output): the outer entry takes flat_inputs, the inner call is captured as
    # _inner_call, and the AOTAutograd orchestration is inlined as a real def that the
    # outer call invokes directly by name. (All wrappers are inlined now -- no _exec_wrapper
    # / source-string blobs anywhere; the chain-wrapper test checks that directly.)
    test.assertIn("def call(flat_inputs):", src)
    test.assertIn("_inner_call = call", src)
    test.assertIn("def _runtime_wrapper(", src)
    test.assertIn("return _runtime_wrapper(", src)
    # Auditability guarantee: no pickle.loads / base64 blob in the emitted module.
    # _load_from_bytes is the storage-reduce callable that embeds raw weight bytes and
    # base64 is the only other opaque-blob encoding that could smuggle them in, so the
    # absence of all three is what makes the comment's no-blob promise actually hold.
    test.assertNotIn("pickle.loads", src)
    test.assertNotIn("_load_from_bytes", src)
    test.assertNotIn("base64", src)


# Effectful targets the composition must reject. Both kinds must be caught: an effectful
# OpOverload (aten._print) and an effectful HigherOrderOperator (hop_print, ORDERED). A HOP
# is NOT an OpOverload, so an OpOverload-only gate would silently let the HOP through -- so
# the HOP subtest is the one that fails against the narrower gate.
_EFFECTFUL_TARGETS = [
    subtest(torch.ops.aten._print.default, name="op_overload"),
    subtest(hop_print, name="hop"),
]


@instantiate_parametrized_tests
class TestAOTCompileToPython(TestCase):
    # [CLASS DOC] End-to-end CPU tests of compile_to_python and load_from_python. Each composes a
    # small module or traced function and checks the emitted source markers plus equality with
    # eager results (and aliasing, versions or subclass types where relevant). Also covers
    # rejection errors, lock behavior and thread-local capture. Parameterized tests are
    # instantiated by the class decorator.
    # End-to-end coverage of the functorch composition layer: compile_to_python composes
    # AOTAutograd's codegen'd runtime wrappers (prelude/epilogue) around the inner Inductor
    # call into one standalone module, and the emitted module must match eager. All CPU.

    def test_pointwise_runs_like_eager(self):
        # [FUNCTION DOC] Composes a pointwise module, checks composed markers and that the cache
        # is bytes or None, and that executing the source matches eager.
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        src, cache = _compose(m, x)
        _assert_composed(self, src)
        # Return contract: cache is the opaque acceleration bytes or None.
        self.assertIsInstance(cache, (bytes, type(None)))
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_load_from_python_standalone_and_warm(self):
        # [FUNCTION DOC] Checks load_from_python runs the composed source standalone in a
        # fresh cache and, when a cache was produced, also with that cache as an accelerator;
        # both must match eager.
        # load_from_python is the inverse of compile_to_python: python_code runs standalone
        # (kernels JIT on first use), and the forwarded inductor cache is a pure accelerator
        # (warms the kernel caches so exec loads precompiled binaries). Both paths must match
        # eager; run each in a fresh cache dir so the standalone path is a genuine cold load.
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        src, cache = _compose(m, x)
        # Standalone: no cache, the module JIT-compiles its own kernels.
        with fresh_cache(), torch.no_grad():
            self.assertEqual(load_from_python(src)(_flat_inputs(m, x))[0], m(x))
        # Warm: the forwarded cache accelerates the same module; result is identical. cache
        # is None on cache-disabled shards, where only the standalone path applies.
        if cache is not None:
            with fresh_cache(), torch.no_grad():
                self.assertEqual(
                    load_from_python(src, cache)(_flat_inputs(m, x))[0], m(x)
                )

    def test_linear_addmm_runs_like_eager(self):
        # [FUNCTION DOC] Composes a single linear layer and checks the
        # executed result equals eager.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_sequential_linear_relu_runs_like_eager(self):
        # [FUNCTION DOC] Composes a linear followed by relu and checks the
        # executed result equals eager.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_reduction_runs_like_eager(self):
        # [FUNCTION DOC] Composes a sum-reduction module and checks the
        # executed result equals eager.
        m = _SumDim1().eval()
        x = torch.randn(6, 7)
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_dynamic_shapes_runs_at_multiple_shapes(self):
        # [FUNCTION DOC] Captures a pointwise module symbolically, composes it, and checks one
        # executed module matches eager at several batch sizes.
        # compile_to_python has no dynamic_shapes knob: it auto-detects symbolic shapes
        # (_graph_has_dynamic_shapes) and picks the internal shapes_mode accordingly. A
        # symbolically-traced graph selects "from_graph", composing one module keyed on
        # symbolic sizes rather than baked constants, and that single module runs at
        # multiple shapes. (A statically-traced graph selects "from_example_inputs", which
        # specializes to the example shapes instead.)
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        gm = _capture(m, x, tracing_mode="symbolic")
        src, _cache = compile_to_python(gm, _flat_inputs(m, x))
        _assert_composed(self, src)
        fn = _exec(src)
        for n in (8, 16, 5):
            xi = torch.randn(n, 4)
            with torch.no_grad():
                self.assertEqual(fn(_flat_inputs(m, xi))[0], m(xi))

    def test_multi_output_runs_like_eager(self):
        # [FUNCTION DOC] Composes a three-output module and checks the executed result has the
        # same count and order of outputs, each equal to eager.
        # Exercises the output epilogue's multi-output count/ordering: the composed module
        # must return all outputs in the captured order, each equal to eager.
        m = _MultiOut().eval()
        x = torch.randn(6, 7)
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        eager = m(x)
        with torch.no_grad():
            out = _exec(src)(_flat_inputs(m, x))
        self.assertEqual(len(out), len(eager))
        for got, want in zip(out, eager):
            self.assertEqual(got, want)

    def test_input_mutation_copy_back_runs_like_eager(self):
        # [FUNCTION DOC] Composes a module mutating a buffer in place and checks both the output
        # and the caller-passed buffer, which must end equal to the eager module buffer.
        # A buffer mutated in place exercises AOTAutograd's mutation epilogue (input copy-
        # back): the composed call must reflect the mutation onto the passed-in buffer
        # tensor, exactly as eager mutates m.b. Compare both the output AND the mutated
        # input.
        m = _BufferMutate().eval()
        x = torch.randn(4)
        src, _cache = _compose(m, x)
        _assert_composed(self, src)

        eager = _BufferMutate().eval()
        eager_out = eager(x)

        buf = torch.zeros(4)
        with torch.no_grad():
            composed_out = _exec(src)([buf, x])[0]
        self.assertEqual(composed_out, eager_out)
        self.assertEqual(buf, eager.b)

    def test_output_alias_regen_runs_like_eager(self):
        # [FUNCTION DOC] Composes a module returning a view and checks the alias generation helper
        # and view-metadata reconstruction text appear in the source, and that the output equals
        # eager and shares storage with the passed input.
        # An output that aliases an input exercises AOTAutograd's output-alias regeneration
        # (the _alias_fn / gen_alias_from_base path, which the orchestration closes over
        # directly). The composed output must both equal eager AND alias the input's
        # storage, exactly as eager's view does.
        m = _ViewAlias().eval()
        x = torch.randn(4, 4)
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        self.assertIn("gen_alias_from_base", src)
        # Pin the view-replay reconstruction branches this PR adds (the new
        # ViewMetaSequence._from_parts factory + the ViewMeta as_tuple branch). The
        # numerics/aliasing asserts alone would not catch a wrong reconstruction on this
        # tiny view, so assert the emission explicitly.
        self.assertIn("ViewMetaSequence._from_parts(", src)
        self.assertIn("ViewMeta(", src)
        xc = x.clone()
        with torch.no_grad():
            out = _exec(src)([xc])[0]
        self.assertEqual(out, m(x))
        self.assertEqual(
            out.untyped_storage().data_ptr(), xc.untyped_storage().data_ptr()
        )

    def test_dedup_mutated_duplicate_input_runs_like_eager(self):
        # [FUNCTION DOC] Composes a function mutating a duplicated input (same tensor passed
        # twice) and checks the outer-wrapper adapter and dedup arguments in the source, plus that
        # the mutation and version bump landed on the right tensor, matching eager.
        # A mutated DUPLICATE input drives AOTDedupeWrapper -- a CompilerWrapper applied
        # AROUND the orchestration (graph_compile._aot_stage2c_make_inference_function),
        # NOT an inner chain wrapper. The composed module must nest it OUTSIDE the
        # orchestration -- dedup(orchestration(inner)) -- so the orchestration sees the
        # DEDUPED args. Composed inside-out (the pre-fix bug) the version bump / copy-back
        # index the raw pre-dedup args and land on the wrong (duplicate) tensor, so the
        # version-counter assertion below fails; the ``_orchestration_entry`` adapter is
        # the structural marker of the correct outer nesting (absent pre-fix).
        def f(a, b, c, d):
            # [FUNCTION DOC] HELPER: Traced function that multiplies its fourth arg by two in
            # place and returns the first arg plus the fourth.
            d.mul_(2)
            return a + d

        x = torch.randn(4)
        y = torch.randn(4)
        gm = make_fx(f)(x, x, y, y)
        src, _cache = compile_to_python(gm, [x, x, y, y])
        self.assertIn("def call(flat_inputs):", src)
        self.assertIn("_orchestration_entry", src)  # outer-wrapper adapter over orch
        self.assertIn("deduped_args", src)  # the dedup wrapper is spliced

        xe, ye = x.clone(), y.clone()
        eager_out = f(xe, xe, ye, ye)

        xc, yc = x.clone(), y.clone()
        with torch.no_grad():
            composed_out = _exec(src)([xc, xc, yc, yc])[0]
        self.assertEqual(composed_out, eager_out)
        self.assertEqual(yc, ye)  # mutation landed on the right tensor
        self.assertEqual(yc._version, ye._version)  # and bumped the right version

    def test_synthetic_base_aliased_mutated_input_runs_like_eager(self):
        # [FUNCTION DOC] Composes a function mutating an input that aliases another input and
        # checks the outer-wrapper adapter and synthetic-base wrapper appear in the source and
        # that values and version counters match eager.
        # Aliased + mutated inputs drive AOTSyntheticBaseWrapper, also applied AROUND the
        # orchestration. It collapses the aliased inputs into a synthetic base BEFORE the
        # orchestration runs, so it must be composed OUTSIDE it (same requirement as dedup):
        # the ``_orchestration_entry`` adapter marks the correct nesting (absent pre-fix,
        # which inverted the chain to orchestration(synthetic_base(inner))).
        def f(a, b):
            # [FUNCTION DOC] HELPER: Traced function that doubles its first arg in place and
            # returns it plus the second arg.
            a.mul_(2)
            return a + b

        def make():
            # [FUNCTION DOC] HELPER: Builds a fresh base tensor and returns a slice of it plus the
            # base, so the two inputs alias.
            base = torch.arange(4, dtype=torch.float32) + 1
            return base[:], base  # a aliases b

        gm = make_fx(f)(*make())
        a0, b0 = make()
        src, _cache = compile_to_python(gm, [a0, b0])
        self.assertIn("_orchestration_entry", src)
        self.assertIn("_synthetic_base_wrapper", src)

        ae, be = make()
        eager_out = f(ae, be)

        ac, bc = make()
        with torch.no_grad():
            composed_out = _exec(src)([ac, bc])[0]
        self.assertEqual(composed_out, eager_out)
        self.assertEqual(ac, ae)
        self.assertEqual(bc, be)
        self.assertEqual(bc._version, be._version)

    def test_tensor_subclass_wrap_unwrap_runs_like_eager(self):
        # [FUNCTION DOC] Composes a function over a two-component tensor subclass input and checks
        # the executed result is the same subclass with components equal to eager.
        # The headline feature: a tensor-subclass input exercises AOTAutograd's subclass
        # flatten/unflatten wrapper plus baked subclass metadata. The composed module must
        # unwrap the subclass for the inner dense call and re-wrap the output as the same
        # subclass, matching eager.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(x):
            # [FUNCTION DOC] HELPER: Traced function returning twice the input plus one.
            return x * 2.0 + 1.0

        tt = TwoTensor(torch.randn(4, 4), torch.randn(4, 4))
        gm = make_fx(f, tracing_mode="real")(tt)
        src, _cache = compile_to_python(gm, [tt])
        _assert_composed(self, src)
        with torch.no_grad():
            out = _exec(src)([tt])[0]
        eager = f(tt)
        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, eager.a)
        self.assertEqual(out.b, eager.b)

    def test_multiple_subclass_inputs_runs_like_eager(self):
        # [FUNCTION DOC] Composes a function over two subclass inputs and checks the executed
        # result is the subclass with components equal to eager.
        # Two tensor-subclass inputs make the subclass wrapper flatten/unflatten more than
        # one subclass (each into its constituents), which the single-input subclass test
        # above does not. NOTE: this stays a single INNER chain wrapper -- a >= 2-link inner
        # chain (e.g. subclass + functionalized-RNG) is only exercised by the order-
        # inversion guard unit test, since a supported multi-inner-chain forward graph is
        # not readily constructible (subclass+RNG hits an internal AOTAutograd assertion,
        # and duplicate subclass inputs collapse into one subclass wrapper). Mutated
        # duplicate / aliased inputs DO add OUTER wrappers (dedup / synthetic base), but
        # those wrap the orchestration rather than the inner call -- see the dedup /
        # synthetic-base tests above.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(a, b):
            # [FUNCTION DOC] HELPER: Traced function returning twice the first arg plus
            # three times the second.
            return a * 2.0 + b * 3.0

        ta = TwoTensor(torch.randn(4, 4), torch.randn(4, 4))
        tb = TwoTensor(torch.randn(4, 4), torch.randn(4, 4))
        gm = make_fx(f, tracing_mode="real")(ta, tb)
        src, _cache = compile_to_python(gm, [ta, tb])
        _assert_composed(self, src)
        with torch.no_grad():
            out = _exec(src)([ta, tb])[0]
        eager = f(ta, tb)
        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, eager.a)
        self.assertEqual(out.b, eager.b)

    def test_autocast_disable_autocast_runs_like_eager(self):
        # [FUNCTION DOC] Composes a linear module traced and compiled under bf16 CPU
        # autocast and checks the source disables autocast and that the executed output
        # matches eager run under autocast.
        # disable_amp is read by AOTAutograd at compile time via _is_any_autocast_enabled,
        # so the inner compile MUST run under autocast for the orchestration to emit
        # _DisableAutocast_. The graph is also traced under autocast so the bf16 casts are
        # baked in; the orchestration then disables autocast at runtime to keep the dense
        # call from double-casting. Equivalence is checked against eager run UNDER autocast.
        m = _MatMul().eval()
        x = torch.randn(5, 4)
        pb = _flat_inputs(m, x)

        pnames = [n for n, _ in m.named_parameters()]
        k = len(pnames)
        bnames = [n for n, _ in m.named_buffers()]

        def flat_fn(flat):
            # [FUNCTION DOC] HELPER: Traced function over a flat list that rebinds
            # module state, runs the module under bf16 autocast on the last element, and
            # returns the flattened outputs.
            params = dict(zip(pnames, flat[:k]))
            buffers = dict(zip(bnames, flat[k : k + len(bnames)]))
            with stateless._reparametrize_module(
                m, {**params, **buffers}, tie_weights=True
            ):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out = m(flat[-1])
            return pytree.tree_flatten(out)[0]

        with torch.enable_grad():
            gm = make_fx(flat_fn, tracing_mode="real")(pb)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            src, _cache = compile_to_python(gm, pb)
        _assert_composed(self, src)
        self.assertIn("_DisableAutocast_", src)
        with torch.no_grad():
            out = _exec(src)(pb)[0]
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            eager = m(x)
        self.assertEqual(out, eager)

    def test_options_passthrough_runs_like_eager(self):
        # [FUNCTION DOC] Passes an option enabling NaN assertions through compile_to_python and
        # checks the source contains the NaN check, proving the option reached the inner compile,
        # and that numerics match eager.
        # compile_to_python forwards ``options`` straight to the inner inductor compile in a
        # single line. Use nan_asserts=True (default False) as the probe: it is observable in
        # the inner source as an ``isnan`` check, so the assertion below FAILS if the forward
        # were dropped (the option would fall back to the False default and emit no isnan) --
        # unlike an option whose value equals its default, which could not detect a dropped
        # forward. nan_asserts only adds runtime checks, so numerics still match eager.
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        gm = _capture(m, x)
        src, _cache = compile_to_python(
            gm, _flat_inputs(m, x), options={"nan_asserts": True}
        )
        _assert_composed(self, src)
        self.assertIn("isnan", src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_orchestration_inlined_as_real_def(self):
        # [FUNCTION DOC] Checks the runtime wrapper appears as a real top-level def invoked by
        # name with no leftover alias or exec helper, that it is called with a null context and a
        # no-op in its context slots, and that results match eager.
        # The orchestration is spliced as a real top-level ``def _runtime_wrapper`` that the
        # outer ``call`` invokes directly by name -- no string re-exec, no ``_orchestration``
        # alias. All wrappers are inlined now, so ``_exec_wrapper`` no longer exists in any
        # composed module; the module reads as ordinary code and must still exec like eager.
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        src, _cache = _compose(m, x)
        self.assertIn("def _runtime_wrapper(", src)
        self.assertIn("return _runtime_wrapper(", src)
        self.assertNotIn("_orchestration", src)  # redundant alias removed
        self.assertNotIn("_exec_wrapper", src)
        # Pin the deliberate drop of the first-invocation context / profiler prologue:
        # the orchestration is invoked with contextlib.nullcontext + a no-op in those two
        # positional slots. A future change re-threading a real context here would fail.
        self.assertIn(", contextlib.nullcontext, lambda: None,", src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_chain_wrapper_inlined_as_real_def(self):
        # [FUNCTION DOC] Composes a subclass-input function and checks the chain wrapper is
        # inlined as a def with the compiled function hoisted to an assignment from the inner
        # call, with no exec helper or source string blobs, and that numerics match eager.
        # A graph with a chain wrapper (tensor subclass -> ``inner_fn``, which closes over
        # the inner via a ``compiled_fn`` global) is now inlined too: the wrapper is a real
        # top-level def with ``compiled_fn`` hoisted to a module-scope assignment, no exec /
        # string blob anywhere. Numerics must match eager.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(x):
            # [FUNCTION DOC] HELPER: Traced function returning twice the input plus one.
            return x * 2.0 + 1.0

        tt = TwoTensor(torch.randn(4, 4), torch.randn(4, 4))
        gm = make_fx(f, tracing_mode="real")(tt)
        src, _cache = compile_to_python(gm, [tt])
        self.assertNotIn("_exec_wrapper", src)
        self.assertNotIn("_src = ", src)  # no re-exec'd source-string blobs
        self.assertIn("def inner_fn(", src)
        self.assertIn("compiled_fn = _inner_call", src)
        self.assertIn("def _runtime_wrapper(", src)
        with torch.no_grad():
            out = _exec(src)([tt])[0]
        eager = f(tt)
        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, eager.a)
        self.assertEqual(out.b, eager.b)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "functionalize_rng_ops threads CUDA RNG state via CUDARngStateHelper, which "
        "requires a CUDA device (the graph itself lowers through the CPU backend).",
    )
    def test_functionalized_rng_runs_like_eager(self):
        # [FUNCTION DOC] Under a CUDA-availability skip, composes a dropout module with
        # functionalized RNG ops enabled and checks the source mentions the CUDA RNG state helper
        # and that, with the same manual seed, the executed output equals eager.
        # functionalize_rng_ops rewrites the RNG op into a functional form during the inner
        # AOTAutograd lowering, producing a FunctionalizedRngRuntimeWrapper that threads RNG
        # state via CUDARngStateHelper. ``CUDARngStateHelper`` in the source is the RNG-specific
        # signal (it appears only under functionalize_rng_ops and exercises the helper-table
        # rows); the wrapper's ``_compiled_fn_`` inner-name is real but not asserted here, as
        # that token is the orchestration's first parameter and appears in every composed
        # module. Seeded so dropout's mask is deterministic for the eager comparison.
        class _Dropout(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying training-mode dropout with probability one half.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns dropout of the input in training mode.
                return torch.nn.functional.dropout(x, p=0.5, training=True)

        m = _Dropout()
        x = torch.randn(8, 4)

        def flat_fn(flat):
            # [FUNCTION DOC] HELPER: Traced function over a flat list that returns the flattened
            # output of the dropout module on the last element.
            return pytree.tree_flatten(m(flat[-1]))[0]

        with functorch_config.patch(functionalize_rng_ops=True):
            with torch.enable_grad():
                gm = make_fx(flat_fn, tracing_mode="real")([x])
            src, _cache = compile_to_python(gm, [x])
        _assert_composed(self, src)
        self.assertIn("CUDARngStateHelper", src)
        fn = _exec(src)
        torch.manual_seed(123)
        with torch.no_grad():
            out = fn([x])[0]
        torch.manual_seed(123)
        with torch.no_grad():
            eager = m(x)
        self.assertEqual(out, eager)

    def test_helpers_imported_from_standalone_runtime_surface(self):
        # [FUNCTION DOC] Composes a view-returning module and checks the source imports
        # the alias helper from the standalone runtime surface and does not reference
        # its internal functional-utils location.
        # End-to-end lock for the stability contract: a graph closing over a runtime helper
        # (here gen_alias_from_base, via output-alias regen) must import it from the
        # standalone_runtime surface, not its internal AOTAutograd location. A dropped or
        # aliased _known_helper_table entry would silently fall through to the internal
        # module. (ViewMetaSequence legitimately imports functional_utils, so this checks the
        # specific helper expression, not the bare module name.)
        m = _ViewAlias().eval()
        x = torch.randn(4, 4)
        src, _cache = _compose(m, x)
        self.assertIn(
            "from torch._functorch._aot_autograd.standalone_runtime import "
            "gen_alias_from_base",
            src,
        )
        self.assertNotIn("functional_utils.gen_alias_from_base", src)

    @parametrize("target", _EFFECTFUL_TARGETS)
    def test_rejects_effectful_op(self, target):
        # [FUNCTION DOC] Parametrized over an effectful op overload and an effectful higher-order
        # op: builds a graph calling it and checks compile_to_python raises NotImplementedError
        # mentioning an effectful op.
        # A graph carrying an effectful op is rejected up front with a concrete
        # NotImplementedError -- effect tokens thread through a calling convention the
        # standalone composition does not reproduce. Both an effectful OpOverload
        # (aten._print) and an effectful HigherOrderOperator (hop_print) must be caught;
        # the HOP case would slip past an OpOverload-only gate.
        g = fx.Graph()
        a = g.placeholder("a")
        g.call_function(target, ("hello",))
        g.output((a,))
        gm = fx.GraphModule(torch.nn.Module(), g)
        with self.assertRaisesRegex(NotImplementedError, "effectful op"):
            compile_to_python(gm, [torch.randn(3)])

    def test_rejects_non_graphmodule(self):
        # [FUNCTION DOC] Checks compile_to_python raises TypeError (expects a post-AOTAutograd
        # graph) when given a string instead of a graph module.
        # The effectful-op scan dereferences gm.graph before reaching inductor's own check,
        # so the functorch layer must reject a non-GraphModule with a clean TypeError rather
        # than an opaque AttributeError.
        with self.assertRaisesRegex(TypeError, "expects a post-AOTAutograd"):
            compile_to_python("not a graph module", [])

    def test_reentrant_compile_to_python_under_held_lock(self):
        # [FUNCTION DOC] Holds the compile lock while composing on the same thread to
        # show it is re-entrant (a non-reentrant lock would hang), then checks markers
        # and numerics against eager.
        # The entry point takes a re-entrant lock (RLock) so a nested on-thread compile (a
        # custom backend / inductor pass re-entering during compile) does not self-deadlock.
        # Holding the lock and compiling exercises that re-entry: a plain Lock would hang
        # here (surfacing as a CI timeout), so this pins the RLock choice behaviorally.
        from torch._functorch._aot_autograd.to_standalone_python import _COMPILE_LOCK

        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        with _COMPILE_LOCK:
            src, _cache = _compose(m, x)
        _assert_composed(self, src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_concurrent_compile_to_python_smoke(self):
        # [FUNCTION DOC] Captures two graphs serially, then composes them in two threads and
        # checks neither thread recorded an error and each composed module matches eager. Captured
        # tracebacks are joined into the failure message.
        # End-to-end concurrency smoke test: _COMPILE_LOCK serializes the entry point (the
        # underlying cache-state swap is process-global), so two threads compiling different
        # graphs run one-at-a-time and must each still produce their own correct module.
        # This exercises the lock end-to-end (no deadlock, both succeed); the thread-local
        # sink isolation it relies on is pinned by test_capture_sink_is_thread_local.
        #
        # Only compile_to_python runs concurrently. The graphs are captured SERIALLY up
        # front because make_fx (fx symbolic tracing) is NOT thread-safe: it mutates
        # process-global tracing state -- the CURRENT_PATCHER global in
        # torch.fx._symbolic_trace and the fx.traceback node-meta globals. Capturing inside
        # the worker threads would race that state against the make_fx run nested inside the
        # OTHER thread's compile_to_python (which _COMPILE_LOCK covers, but capture is not
        # the serialized entry point), intermittently raising "CURRENT_PATCHER is None in
        # finally block" -- a make_fx concurrency artifact unrelated to the lock under test.
        import threading
        import traceback

        specs = [
            (_Pointwise().eval(), torch.randn(8, 4)),
            (_SumDim1().eval(), torch.randn(6, 7)),
        ]
        captured = [(_capture(m, x), m, x) for m, x in specs]
        results: dict = {}
        errors: dict = {}

        def run(i, gm, m, x):
            # [FUNCTION DOC] HELPER: Thread body: composes the given graph and stores the source
            # with its module and input in the shared results mapping by index; on any exception
            # stores the formatted traceback in the errors mapping instead.
            try:
                results[i] = (compile_to_python(gm, _flat_inputs(m, x))[0], m, x)
            except Exception:
                # Surface the full traceback so a future failure is diagnosable from CI
                # logs; the bare {1: <Exception>} the swallow used to leave hid the cause.
                errors[i] = traceback.format_exc()

        threads = [
            threading.Thread(target=run, args=(i, gm, m, x))
            for i, (gm, m, x) in enumerate(captured)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, {}, "\n".join(errors.values()))
        for src, m, x in results.values():
            _assert_composed(self, src)
            with torch.no_grad():
                self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_capture_sink_is_thread_local(self):
        # [FUNCTION DOC] Starts two threads that synchronize with a two-party barrier while each
        # is inside the generated-source capture context, each compiles a tiny generated
        # function named for its key, and checks each sink holds only its own function name,
        # proving capture is thread-local.
        # The capture sink MUST be thread-local: two threads forced (via a barrier) to be
        # mid-capture simultaneously must each record ONLY their own codegen'd wrapper. A
        # process-global sink would bleed wrappers across threads and fail this -- the direct
        # pin for the thread-local contract that the lock-serialized smoke test above cannot
        # exercise (the lock prevents real overlap there).
        import threading

        from torch._functorch._aot_autograd.codegen import (
            _compile_and_exec_source,
            capture_generated_sources,
        )

        barrier = threading.Barrier(2)
        sinks: dict = {}

        def run(key):
            # [FUNCTION DOC] HELPER: Thread body: opens a capture sink, waits at the barrier,
            # compiles and executes a generated function named for the key, waits again, then
            # stores the captured function names in the shared mapping.
            into: list = []
            with capture_generated_sources(into):
                barrier.wait(
                    timeout=60
                )  # both threads now inside their capture context
                _compile_and_exec_source(
                    f"def {key}_fn(args):\n    return args\n", {}, f"{key}_fn", key
                )
                barrier.wait(
                    timeout=60
                )  # hold both contexts open across the other's codegen
            sinks[key] = [g.fn_name for g in into]

        threads = [threading.Thread(target=run, args=(k,)) for k in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sinks["a"], ["a_fn"])
        self.assertEqual(sinks["b"], ["b_fn"])


@instantiate_parametrized_tests
class TestComposerHelpers(TestCase):
    # [CLASS DOC] Unit tests for composer helpers: the stable-import table, module-level name
    # discovery, and the recursive search for effectful ops in a graph and its subgraphs.
    # Unit coverage of the composer's own helpers: the _known_helper_table stable-import
    # contract, _module_level_names (inner-binding collision seeding), and the recursive
    # _find_effectful_op scan. (Source-emission helper tests live in test_source_emit.py.)

    def test_known_helper_table_imports_are_stable_surface(self):
        # [FUNCTION DOC] Checks every helper-table entry imports either the torch package or the
        # standalone runtime surface, never internal AOTAutograd modules.
        # Stability contract: every runtime helper the composer recognizes must emit an
        # import via the stable standalone_runtime surface (or `import torch` for public
        # torch paths), never a deep AOTAutograd-internal module. Lock the table so a new
        # entry pointing at an unstable location is caught.
        for import_stmt, _expr in _known_helper_table().values():
            self.assertTrue(
                import_stmt == "import torch"
                or import_stmt.startswith(
                    "from torch._functorch._aot_autograd.standalone_runtime import "
                ),
                f"helper import {import_stmt!r} bypasses the standalone_runtime surface",
            )

    def test_module_level_names_excludes_deleted(self):
        # [FUNCTION DOC] Parses a small module that binds then deletes a name and checks the
        # module-level name finder keeps the surviving name and drops the deleted one.
        # Inductor's inner module binds then dels a name (async_compile = AsyncCompile();
        # del async_compile) at module scope. A del'd name does not persist, so it must not
        # be reserved -- otherwise a hoisted wrapper global of the same name would trip a
        # spurious _reserve collision.
        tree = ast.parse("a = 1\nb = 2\ndel a\n")
        names = _module_level_names(tree)
        self.assertIn("b", names)
        self.assertNotIn("a", names)

    @parametrize("target", _EFFECTFUL_TARGETS)
    def test_find_effectful_op_top_level(self, target):
        # [FUNCTION DOC] Parametrized over an effectful op overload and an effectful higher-order
        # op: builds a graph calling it and checks the effect search returns that exact target.
        # The scan must surface an effectful OpOverload AND an effectful
        # HigherOrderOperator (hop_print); the HOP case would return None under an
        # OpOverload-only isinstance check.
        g = fx.Graph()
        a = g.placeholder("a")
        g.call_function(target, ("hi",))
        g.output((a,))
        gm = fx.GraphModule(torch.nn.Module(), g)
        self.assertIs(_find_effectful_op(gm, _get_effect), target)

    def test_find_effectful_op_nested_in_subgraph(self):
        # [FUNCTION DOC] Builds a parent graph that references a child graph module through an
        # attribute fetch, where the child holds a print op, and checks the search finds it.
        # An effect nested inside a child GraphModule reached via get_attr must be found.
        child = fx.Graph()
        ca = child.placeholder("a")
        child.call_function(torch.ops.aten._print.default, ("hi",))
        child.output((ca,))
        child_gm = fx.GraphModule(torch.nn.Module(), child)

        parent = fx.Graph()
        pa = parent.placeholder("a")
        parent.get_attr("sub")
        parent.output((pa,))
        root = torch.nn.Module()
        root.sub = child_gm
        parent_gm = fx.GraphModule(root, parent)
        self.assertIs(
            _find_effectful_op(parent_gm, _get_effect), torch.ops.aten._print.default
        )

    def test_find_effectful_op_nested_in_container_arg(self):
        # [FUNCTION DOC] Builds a parent graph passing a child graph module inside a list keyword
        # arg of a call, where the child holds a print op, and checks the search finds it.
        # A child GraphModule reached via a container-nested node ARG (not get_attr) -- the
        # form HOPs use to pass a body callable -- must still be scanned for effects.
        child = fx.Graph()
        ca = child.placeholder("a")
        child.call_function(torch.ops.aten._print.default, ("hi",))
        child.output((ca,))
        child_gm = fx.GraphModule(torch.nn.Module(), child)

        parent = fx.Graph()
        pa = parent.placeholder("a")
        parent.call_function(torch.relu, (pa,), {"bodies": [child_gm]})
        parent.output((pa,))
        parent_gm = fx.GraphModule(torch.nn.Module(), parent)
        self.assertIs(
            _find_effectful_op(parent_gm, _get_effect), torch.ops.aten._print.default
        )

    def test_find_effectful_op_none_when_pure(self):
        # [FUNCTION DOC] Checks the effect search returns None for a
        # captured pure pointwise graph.
        m = _Pointwise().eval()
        gm = _capture(m, torch.randn(4, 4))
        self.assertIsNone(_find_effectful_op(gm, _get_effect))


@requires_cuda_and_triton
class TestAOTCompileToPythonCuda(TestCase):
    # [CLASS DOC] CUDA mirrors of key composition cases; only runs with CUDA and Triton available.
    # Verifies the composed wrappers work around Triton kernels and CUDA tensors.
    # The composition is device-agnostic source manipulation, but its wrappers must also
    # compose correctly around Inductor's @triton.jit kernels and on CUDA tensors. Mirror
    # the key e2e cases on CUDA; the inner-kernel codegen itself is covered by
    # test/inductor/test_compile_to_python.py's CUDA class.
    def test_pointwise_runs_like_eager(self):
        # [FUNCTION DOC] Composes a pointwise module on CUDA, checks the source contains a Triton
        # jit marker, and that the executed result matches eager.
        m = _Pointwise().eval().cuda()
        x = torch.randn(8, 4, device="cuda")
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        self.assertIn("@triton.jit", src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_output_alias_regen_runs_like_eager(self):
        # [FUNCTION DOC] Composes a view-returning module on CUDA and checks the alias helper
        # appears, the output equals eager and shares storage with the input.
        m = _ViewAlias().eval().cuda()
        x = torch.randn(4, 4, device="cuda")
        src, _cache = _compose(m, x)
        _assert_composed(self, src)
        self.assertIn("gen_alias_from_base", src)
        xc = x.clone()
        with torch.no_grad():
            out = _exec(src)([xc])[0]
        self.assertEqual(out, m(x))
        self.assertEqual(
            out.untyped_storage().data_ptr(), xc.untyped_storage().data_ptr()
        )

    def test_tensor_subclass_wrap_unwrap_runs_like_eager(self):
        # [FUNCTION DOC] Composes a function over a two-component subclass of CUDA tensors and
        # checks the output is the subclass with components equal to eager.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(x):
            # [FUNCTION DOC] HELPER: Traced function returning twice the input plus one.
            return x * 2.0 + 1.0

        tt = TwoTensor(
            torch.randn(4, 4, device="cuda"), torch.randn(4, 4, device="cuda")
        )
        gm = make_fx(f, tracing_mode="real")(tt)
        src, _cache = compile_to_python(gm, [tt])
        _assert_composed(self, src)
        with torch.no_grad():
            out = _exec(src)([tt])[0]
        eager = f(tt)
        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, eager.a)
        self.assertEqual(out.b, eager.b)

    def test_input_mutation_copy_back_runs_like_eager(self):
        # [FUNCTION DOC] Composes a buffer-mutating module on CUDA and checks both the output and
        # the copied-back caller buffer match eager.
        # The mutation epilogue's copy-back is the most plausibly device-sensitive wrapper
        # path (it writes updated values back onto the passed-in CUDA tensors), so mirror
        # the CPU mutation case on CUDA in addition to the pointwise/alias/subclass cases.
        m = _BufferMutate().eval().cuda()
        x = torch.randn(4, device="cuda")
        src, _cache = _compose(m, x)
        _assert_composed(self, src)

        eager = _BufferMutate().eval().cuda()
        eager_out = eager(x)

        buf = torch.zeros(4, device="cuda")
        with torch.no_grad():
            composed_out = _exec(src)([buf, x])[0]
        self.assertEqual(composed_out, eager_out)
        self.assertEqual(buf, eager.b)


class TestAOTComposeGuards(TestCase):
    # [CLASS DOC] Guard tests for the standalone module composer. Hand-built generated-source
    # records (with fixed orchestration and chain templates kept as class attributes) are passed
    # to the compose function to check each rejection raises NotImplementedError with the expected
    # message, and that valid shapes compose.
    # The composer's defensive guards (which reject rather than emit a subtly-wrong module)
    # only fire if AOTAutograd's codegen drifts, so drive them directly with hand-built
    # GeneratedSource objects rather than waiting for an upstream regression.
    _ORCH_SRC = (
        "def _runtime_wrapper(_compiled_fn_, _first_ctx_, _on_before_call_, args):\n"
        "    return _compiled_fn_(args)\n"
    )
    _CHAIN_SRC = "def inner_fn(args):\n    return compiled_fn(args)\n"

    def test_orchestration_signature_guard(self):
        # [FUNCTION DOC] Passes an orchestration source with an unexpected signature and checks
        # the compose function raises NotImplementedError about the orchestration signature.
        # The generated call invokes the orchestration positionally, so a changed signature
        # must fail loudly rather than silently pass wrong arguments.
        bad_orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            "def _runtime_wrapper(wrong, args):\n    return None\n",
            {},
            lambda: None,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "orchestration wrapper signature"
        ):
            _compose_standalone_module(
                "def call(args):\n    return args\n",
                [bad_orch],
                _SENTINEL_INNER_CALL,
            )

    def test_orchestration_extra_kwonly_param_rejected(self):
        # [FUNCTION DOC] Passes an orchestration source with the right positional params plus an
        # extra keyword-only param and checks the same signature error is raised.
        # The 4 positional params are intact but a keyword-only param is added. The standalone
        # call is purely positional, so a kw-only-with-default would be silently dropped; the
        # guard must compare the FULL signature and reject this, not just the positional list.
        kwonly_orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            "def _runtime_wrapper(_compiled_fn_, _first_ctx_, _on_before_call_, args, "
            "*, new_flag=None):\n    return _compiled_fn_(args)\n",
            {},
            lambda: None,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "orchestration wrapper signature"
        ):
            _compose_standalone_module(
                "def call(args):\n    return args\n",
                [kwonly_orch],
                _SENTINEL_INNER_CALL,
            )

    def test_empty_capture_rejected(self):
        # [FUNCTION DOC] Passes an empty captured list and checks the error saying exactly one
        # forward orchestration wrapper is required.
        # The real backstop for an incomplete capture (e.g. if a future change offloaded
        # wrapper codegen to a worker thread so nothing was captured): the composer requires
        # exactly one forward orchestration wrapper and rejects an empty capture rather than
        # emitting a partial module.
        with self.assertRaisesRegex(
            NotImplementedError, "exactly one forward orchestration wrapper"
        ):
            _compose_standalone_module(
                "def call(args):\n    return args\n", [], _SENTINEL_INNER_CALL
            )

    def test_orchestration_global_colliding_with_inner_rejected(self):
        # [FUNCTION DOC] Gives the orchestration a global whose name collides with a top-level
        # name of the inner module and checks the collision error is raised.
        # The inlined orchestration hoists its globals to module scope; a hoisted name that
        # shadows a top-level name the inner module already binds is rejected (rather than
        # silently rebinding it). ``aten`` is a real inner top-level binding; the resolved
        # expr (a helper) differs from the name, so it is hoisted and trips the guard.
        inner = "aten = 1\ndef call(args):\n    return args\n"
        orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            self._ORCH_SRC,
            {"aten": torch.autograd.graph.increment_version},
            lambda: None,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "collides with another top-level name in the composed"
        ):
            _compose_standalone_module(inner, [orch], _SENTINEL_INNER_CALL)

    def test_orchestration_def_name_colliding_with_inner_rejected(self):
        # [FUNCTION DOC] Gives the inner module a top-level name equal to the orchestration
        # function name and checks the collision error is raised before globals are hoisted.
        # Distinct from the hoisted-global collision above: the up-front _reserve loop
        # reserves every wrapper DEF name before any global is hoisted, so an inner module
        # that binds a top-level name equal to a wrapper's fn_name ("_runtime_wrapper")
        # trips that earlier guard rather than the hoist path.
        inner = "_runtime_wrapper = 1\ndef call(args):\n    return args\n"
        with self.assertRaisesRegex(
            NotImplementedError, "collides with another top-level name in the composed"
        ):
            _compose_standalone_module(inner, [self._orch()], _SENTINEL_INNER_CALL)

    def test_compose_rejects_baked_live_tensor(self):
        # [FUNCTION DOC] Gives the orchestration a global that is a live tensor and checks
        # composing raises NotImplementedError saying a live tensor cannot be baked.
        # The module's no-weights / no-pickle.loads promise: a wrapper closing over a live
        # tensor (with no import path) must RAISE at compose time rather than embed raw
        # bytes. This drives that rejection through the COMPOSE path (a wrapper global),
        # not just emit_value in isolation -- the _assert_composed markers only prove no
        # blob appears when nothing forces one, so this is what actually locks the promise.
        orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            "def _runtime_wrapper(_compiled_fn_, _first_ctx_, _on_before_call_, args):\n"
            "    return [_baked]\n",
            {"_baked": torch.randn(4)},
            lambda: None,
        )
        with self.assertRaisesRegex(NotImplementedError, "cannot bake a live Tensor"):
            _compose_standalone_module(
                "def call(args):\n    return args\n", [orch], _SENTINEL_INNER_CALL
            )

    def test_rebuild_helper_spliced_and_runs_in_composed_module(self):
        # [FUNCTION DOC] Bakes a fixture object whose reduction uses the rebuild helper; checks
        # the composed source includes the helper and call, and that executing it returns a value
        # equal to the baked object.
        # When a baked global reconstructs via the pickle-reduce-as-source path (_NewObjEx
        # emits ``_rebuild(...)``), the composer must splice the _rebuild helper into the
        # module (needs_rebuild) AND _rebuild must actually run at module-exec time to
        # reconstruct the value. The _emit_via_reduce unit tests cover the _rebuild logic in
        # isolation; this is the only coverage of the splice + in-module execution.
        baked = _NewObjEx(1, b=2)
        orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            "def _runtime_wrapper(_compiled_fn_, _first_ctx_, _on_before_call_, args):\n"
            "    return [_baked]\n",
            {"_baked": baked},
            lambda: None,
        )
        src = _compose_standalone_module(
            "def call(args):\n    return args\n", [orch], _SENTINEL_INNER_CALL
        )
        self.assertIn("def _rebuild", src)  # helper spliced (needs_rebuild=True)
        self.assertIn("_rebuild(", src)
        out = _exec(src)(
            []
        )  # exec the module; _rebuild runs to rebuild the baked value
        self.assertEqual(out[0], baked)

    def test_chain_head_order_inversion_guard(self):
        # [FUNCTION DOC] Feeds wrapper records in outer-first order (inverted) and checks the
        # composer raises NotImplementedError saying capture order must be innermost-to-outermost.
        # Inner-chain capture order is assumed innermost-to-outermost. Feed it OUTER-first
        # (inverted): the outer wrapper (wraps the inner wrapper) is captured before the
        # inner wrapper (which wraps the authoritative inner call ``inner_fn``), so the
        # "last with an inner-ref" head is actually wrapped by an earlier sibling.
        def inner_fn(args):
            # [FUNCTION DOC] HELPER: Stand-in inner call that returns its argument.
            return args

        def fn_a(args):
            # [FUNCTION DOC] HELPER: Stand-in compiled function that returns its argument.
            return args

        def fn_b(args):
            # [FUNCTION DOC] HELPER: Stand-in compiled function that returns its argument.
            return args

        def orch_fn():
            # [FUNCTION DOC] HELPER: Stand-in orchestration function that returns None.
            return None

        orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            self._ORCH_SRC,
            {},
            orch_fn,
        )
        # outer (fn_b) wraps the inner wrapper (fn_a); inner (fn_a) wraps the dense call.
        outer = GeneratedSource(
            "dedup_wrapper", "inner_fn", self._CHAIN_SRC, {"compiled_fn": fn_a}, fn_b
        )
        inner = GeneratedSource(
            "dedup_wrapper",
            "inner_fn",
            self._CHAIN_SRC,
            {"compiled_fn": inner_fn},
            fn_a,
        )
        with self.assertRaisesRegex(NotImplementedError, "innermost-to-outermost"):
            _compose_standalone_module(
                "def call(args):\n    return args\n", [outer, inner, orch], inner_fn
            )

    def _orch(self, origin_id=None):
        # [FUNCTION DOC] HELPER: Builds a valid generated-source record for the orchestration
        # wrapper using the class template, an empty globals mapping, a no-op function, and the
        # optional origin id, to pair with a deliberately broken sibling record.
        # A valid forward orchestration wrapper whose codegen'd signature matches what the
        # composer invokes positionally; the guard tests below pair it with a deliberately
        # broken sibling so the SIBLING is what trips the guard, not a missing orchestration.
        return GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            self._ORCH_SRC,
            {},
            lambda: None,
            origin_id,
        )

    def test_backward_wrapper_rejected(self):
        # [FUNCTION DOC] Passes a backward wrapper record plus a valid orchestration and checks
        # the composer raises NotImplementedError saying it cannot compose these runtime wrappers.
        # A backward wrapper is out of scope for forward lowering, so it is rejected up
        # front (before chain wiring) even when paired with a valid forward orchestration:
        # an "in backward" artifact_name must fail loudly rather than be spliced in.
        bwd = GeneratedSource(
            "backward_subclass_wrapper",
            "wrap_fn",
            "def wrap_fn(unwrapped_outs):\n    return unwrapped_outs\n",
            {},
            lambda args: args,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "cannot yet compose these runtime"
        ):
            _compose_standalone_module(
                "def call(args):\n    return args\n",
                [bwd, self._orch()],
                _SENTINEL_INNER_CALL,
            )

    def test_unwired_chain_wrapper_rejected(self):
        # [FUNCTION DOC] Passes a chain wrapper whose inner reference uses an unrecognized global
        # name and checks the composer raises NotImplementedError saying it could not wire it.
        # A chain wrapper that names the inner it wraps via a global NOT in _INNER_NAMES
        # (here "mystery_inner") is invisible to chain-head/inner-call detection, so it can
        # never be wired into the module. If _INNER_NAMES drifts out of sync with a new
        # AOTAutograd inner-ref global this is exactly the shape that arises; reject it
        # rather than silently emit a structurally-wrong module.
        def chain_fn(args):
            # [FUNCTION DOC] HELPER: Stand-in compiled function that returns its argument.
            return args

        # mystery_inner is bound to an importable module-level function so its global
        # resolves cleanly as source -- the point under test is the UNRECOGNIZED global
        # NAME (not in _INNER_NAMES), which is what leaves the wrapper unwired.
        mystery = GeneratedSource(
            "dedup_wrapper",
            "inner_fn",
            "def inner_fn(args):\n    return mystery_inner(args)\n",
            {"mystery_inner": _make_holder},
            chain_fn,
        )
        with self.assertRaisesRegex(NotImplementedError, "could not wire"):
            _compose_standalone_module(
                "def call(args):\n    return args\n",
                [mystery, self._orch()],
                _SENTINEL_INNER_CALL,
            )

    def test_multiple_orchestrations_rejected(self):
        # [FUNCTION DOC] Passes two orchestration records with the same origin id and checks the
        # composer raises the error requiring exactly one forward orchestration.
        # Two orchestration wrappers sharing one origin_id (so the origin filter keeps both)
        # is an impossible capture for a single forward; the composer requires exactly one
        # and must reject the ambiguous pair rather than pick one arbitrarily.
        with self.assertRaisesRegex(
            NotImplementedError, "exactly one forward orchestration"
        ):
            _compose_standalone_module(
                "def call(args):\n    return args\n",
                [self._orch(origin_id=5), self._orch(origin_id=5)],
                _SENTINEL_INNER_CALL,
            )

    def test_foreign_origin_wrapper_filtered_out(self):
        # [FUNCTION DOC] Passes a chain wrapper with a different origin id plus an
        # orchestration with the target id; checks the foreign wrapper is dropped from the
        # source while the orchestration is kept.
        # The capture sink is duration-scoped, so a re-entrant on-thread lowering can append
        # ITS wrappers (a different origin_id) during the window. The composer filters to the
        # target origin (the last orchestration's), so a foreign-origin wrapper must be
        # dropped from the emitted source while the target orchestration still composes.
        def foreign_fn(args):
            # [FUNCTION DOC] HELPER: Stand-in compiled function that returns its argument.
            return args

        foreign = GeneratedSource(
            "dedup_wrapper",
            "foreign_inner",
            "def foreign_inner(args):\n    return compiled_fn(args)\n",
            {"compiled_fn": lambda a: a},
            foreign_fn,
            origin_id=1,
        )
        src = _compose_standalone_module(
            "def call(args):\n    return args\n",
            [foreign, self._orch(origin_id=2)],
            _SENTINEL_INNER_CALL,
        )
        self.assertNotIn("foreign_inner", src)
        self.assertIn("_runtime_wrapper", src)

    def test_inner_call_guard_rejects_missing_call(self):
        # [FUNCTION DOC] Passes an inner module that binds no module-level call entry and checks
        # the composer raises NotImplementedError about the missing call.
        # The module splices ``_inner_call = call``, so the inner Inductor source MUST bind a
        # module-level ``call``. An inner module that binds only ``not_call`` would surface as
        # a bare NameError at exec; the guard turns that into a clear contract error.
        with self.assertRaisesRegex(
            NotImplementedError, "does not bind a module-level 'call'"
        ):
            _compose_standalone_module(
                "def not_call(args):\n    return args\n",
                [self._orch()],
                _SENTINEL_INNER_CALL,
            )

    def test_inner_call_guard_accepts_runner_assign(self):
        # [FUNCTION DOC] Composes an inner module that binds the entry through assignment from a
        # runner object method and checks it is accepted, the assignment survives in the source,
        # and executing the module returns the given argument list.
        # The other inductor codegen form binds the entry point as ``call = runner.call``
        # (the graph_partition Runner path) rather than ``def call``. The guard must accept
        # that Assign-with-Name-target form too, so this composes without raising.
        runner_inner = """\
class _R:
    def call(self, args):
        return args
runner = _R()
call = runner.call
"""
        src = _compose_standalone_module(
            runner_inner, [self._orch()], _SENTINEL_INNER_CALL
        )
        # ``_inner_call = call`` is emitted for any successful compose, so it only proves the
        # guard did not raise; assert the runner-specific binding survived into the source to
        # pin that the Assign form (not just some ``call``) was the accepted one, then exec to
        # confirm the spliced ``_inner_call = call`` actually resolves at runtime.
        self.assertIn("call = runner.call", src)
        self.assertEqual(_exec(src)([7]), [7])


if __name__ == "__main__":
    run_tests()
