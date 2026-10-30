# Owner(s): ["module: decompositions"]
# [MODULE DOC] Tests for the primitive operators (torch._prims) and reference
# implementations (torch._refs). Defines device-generic test classes for prims
# (broadcast, sum, cbrt, collapse, var, memory format strides, philox and functional
# rng wrappers), basic prim behavior, refs corner cases and a decomposition varargs
# check over OpInfo entries. Imports scipy only when available; two device-type test
# classes are instantiated at import time.

from functools import partial
from itertools import product
import unittest

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (parametrize, run_tests, TestCase, TEST_SCIPY,
                                                  set_default_dtype)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    dtypes,
    OpDTypes,
)
from torch.testing._internal.common_methods_invocations import (
    op_db,
)
from torch.testing._internal.common_device_type import (
    ops,
)

from torch.testing._internal.logging_tensor import LoggingTensor, capture_logs, log_input
import torch._prims as prims
from torch._prims_common import CUDARngStateHelper
from torch._prims.executor import make_traced
import torch._refs as refs


if TEST_SCIPY:
    import scipy.special

NVPRIM_ATEN_FALLBACK_WARNING = "fallback to aten executor"
GET_ISOLATED_GRAPHMODULE_ERROR = "get_isolated_graphmodule failed on decomposition"

class TestPrims(TestCase):
    # [CLASS DOC] Device-generic tests for prims. Several tests are CUDA only and compare
    # traced execution through the aten executor against eager results; others check
    # shapes, views, strides and rng state handling on any device.
    @onlyCUDA
    @dtypes(torch.float32)
    def test_broadcast_in_dim(self, device, dtype):
        # [FUNCTION DOC] CUDA only. Traces a wrapper around the broadcast_in_dim prim with the
        # aten executor and checks outputs for same-shape, added outer dimensions, expansion
        # and unsqueeze cases against the equivalent tensor ops. Checks reordering the dims
        # raises an error.
        def _wrapper(a, b, broadcast_dimensions):
            # [FUNCTION DOC] HELPER: Calls the broadcast_in_dim prim to broadcast the first tensor
            # to the shape of the second using the given mapping of dimensions.
            return prims.broadcast_in_dim(a, b.shape, broadcast_dimensions)

        traced = make_traced(_wrapper)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for executor in ('aten',):
            fn = partial(traced, executor=executor)
            # Same shape
            shape = (5, 5)
            a = make_arg(shape)
            b = make_arg(shape, low=0.0, high=0.0)
            result = fn(a, b, (0, 1))

            self.assertEqual(result.shape, a.shape)
            self.assertTrue(result.is_contiguous)
            self.assertEqual(a, result)

            # Error input: reordering dims
            with self.assertRaises(Exception):
                result = fn(a, b, (1, 0))

            # Adding outermost dimensions
            a = make_arg((5, 5))
            b = make_arg((3, 3, 5, 5), low=0.0, high=0.0)
            result = fn(a, b, (2, 3))

            self.assertEqual(result.shape, b.shape)
            self.assertEqual(a.broadcast_to(b.shape), result)

            # Expands
            a = make_arg((1, 5, 1))
            b = make_arg((3, 5, 7), low=0.0, high=0.0)
            result = fn(a, b, (0, 1, 2))

            self.assertEqual(result.shape, b.shape)
            self.assertEqual(a.expand_as(result), result)

            # Unsqueezes
            a = make_arg((1, 2, 3))
            b = make_arg((1, 2, 1, 3), low=0.0, high=0.0)
            result = fn(a, b, (0, 1, 3))

            self.assertEqual(result.shape, b.shape)
            self.assertEqual(a.unsqueeze(2), result)

    @onlyCUDA
    @dtypes(torch.float32)
    def test_broadcast_in_dim_sum(self, device, dtype):
        # [FUNCTION DOC] CUDA only. Traces a sum over both dims followed by a broadcast to a
        # scalar shape with the aten executor, and checks the result is zero-dimensional and
        # equals eager execution.
        def _wrapper(a):
            # [FUNCTION DOC] HELPER: Sums the input over dims zero and one with the sum prim, then
            # broadcasts the result to an empty shape; returns it.
            a_sum = prims.sum(a, [0, 1])
            a_bc = prims.broadcast_in_dim(a_sum, [], [])
            return a_bc

        traced = make_traced(_wrapper)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for executor in ('aten',):
            fn = partial(traced, executor=executor)
            shape = (5, 5)
            a = make_arg(shape)
            result = fn(a)

            self.assertEqual(result.shape, ())
            self.assertTrue(result.is_contiguous)
            self.assertEqual(_wrapper(a), result)

    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.float64, torch.long)
    def test_cbrt_prim(self, device, dtype):
        # [FUNCTION DOC] Skipped without SciPy. For float64 and long dtypes and many
        # batch/shape combos (including empty), compares the cbrt prim against scipy cube root
        # computed on the CPU copy, under a double default dtype. Device is not compared
        # strictly.
        make_arg = partial(make_tensor, device=device, dtype=dtype)
        batches = [(), (1,), (2,), (0, 1), (1, 1), (2, 2)]
        shapes = [(), (0,), (1,), (5,)]

        # Sets the default dtype to NumPy's default dtype of double
        with set_default_dtype(torch.double):
            # Tested here, as this OP is not currently exposed or tested in ATen
            for b, s in product(batches, shapes):
                x = make_arg(b + s)
                y = prims.cbrt(x)

                x_np = x.cpu().numpy()
                y_np = scipy.special.cbrt(x_np)

                self.assertEqual(y, y_np, exact_device=False)

    @dtypes(torch.float32)
    def test_collapse(self, device, dtype):
        # [FUNCTION DOC] Checks collapse (copy) and collapse_view (view) over several dim
        # ranges against reshape with the expected shapes. Checks a view of a transposed
        # tensor raises RuntimeError, the copy form still works, and invalid dim ranges raise
        # AssertionError for both prims.
        t = torch.rand(2, 2, 2)
        dim_ranges = [(0, 0), (0, 1), (1, 2), (0, 2)]
        expected_shapes = [(2, 2, 2), (4, 2), (2, 4), (8,)]

        for (start, end), shape in zip(dim_ranges, expected_shapes):
            expect = t.reshape(shape)

            copy = prims.collapse(t, start, end)
            self.assertEqual(copy, expect)
            self.assertFalse(copy._is_view())

            view = prims.collapse_view(t, start, end)
            self.assertEqual(view, expect)
            self.assertTrue(view._is_view())

        t_discontig = t.transpose(0, 1)
        with self.assertRaises(RuntimeError, msg="Attempting to view a collapsed tensor, but no such view exists!"):
            view = prims.collapse_view(t_discontig, 0, 2)

        copy = prims.collapse(t_discontig, 0, 1)
        self.assertEqual(copy, t_discontig.reshape(4, 2))

        error_dims = [(-1, 1), (0, 3), (1, -1)]
        for start, end in error_dims:
            for fn in [prims.collapse, prims.collapse_view]:
                with self.assertRaises(AssertionError):
                    fn(t, start, end)


    def test_aten_overload_to_prims(self, device):
        # [FUNCTION DOC] Under the refs mode context, traces a function of sigmoid and digamma
        # aten calls with make_fx and checks every call_function node targets an op in the
        # prims namespace.
        # This test is to ensure that the torch.ops.aten calls are replaced with refs
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsMode

        a = torch.randn(3, 3, device=device)

        def func(a):
            # [FUNCTION DOC] HELPER: Applies aten digamma then aten sigmoid to the input and
            # returns it.
            return torch.ops.aten.sigmoid.default(torch.ops.aten.digamma.default(a))

        with TorchRefsMode():
            gm = make_fx(func)(a)

        # Check that all call_function nodes are prims
        call_function_nodes = list(filter(lambda n: n.op == "call_function", gm.graph.nodes))
        all_prims_namespace = all(
            node.target.name().startswith("prims") for node in call_function_nodes
        )
        self.assertTrue(all_prims_namespace)

    @onlyCUDA
    @dtypes(torch.float32)
    @parametrize("correction", [0, 1])
    def test_var(self, device, dtype, correction):
        # [FUNCTION DOC] CUDA only. Parametrized by correction (0 or 1). Traces a wrapper
        # around the var prim over both dims with the aten executor and checks a scalar result
        # equal to eager.
        def _wrapper(a):
            # [FUNCTION DOC] HELPER: Computes the var prim over dims zero and one with the closed-
            # over correction value and returns it.
            return prims.var(a, [0, 1], correction=correction)

        traced = make_traced(_wrapper)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for executor in ('aten',):
            fn = partial(traced, executor=executor)
            shape = (5, 5)
            a = make_arg(shape)
            result = fn(a)

            self.assertEqual(result.shape, ())
            self.assertTrue(result.is_contiguous)
            self.assertEqual(_wrapper(a), result)

    @dtypes(torch.float32)
    def test_memory_format_strides(self, device, dtype):
        # [FUNCTION DOC] For many shapes and the contiguous, channels last and channels last
        # 3d formats, checks the refs empty strides equal torch empty strides, and checks
        # contiguous strides match the eager method on a noncontiguous input. The clone check
        # compares torch clone to itself (both sides use the torch function), so it cannot
        # fail.
        # KNOWN ISSUE: the clone portion compares torch clone with torch clone, not the refs
        #   version.
        shapes = (
            (),
            (0,),
            (1,),
            (5),
            (1, 0),
            (1, 1),
            (3, 7),
            (3, 0, 2),
            (1, 1, 2),
            (4, 1, 1),
            (7, 8, 9),
        )

        channels_last_shapes = (
            (0, 0, 0, 0),
            (1, 0, 3, 0),
            (0, 2, 3, 5),
            (2, 2, 2, 0),
            (5, 4, 3, 2),
            (8, 8, 7, 2),
            (9, 1, 3, 1),
            (4, 5, 8, 7)
        )

        channels_last_3d_shapes = (
            (0, 8, 7, 9, 2),
            (5, 0, 7, 9, 2),
            (5, 0, 7, 9, 0),
            (5, 8, 7, 9, 2),
            (5, 1, 7, 9, 2),
            (5, 1, 7, 9, 1),
        )

        pairs = (
            (shapes, torch.contiguous_format),
            (channels_last_shapes, torch.contiguous_format),
            (channels_last_3d_shapes, torch.contiguous_format),
            (channels_last_shapes, torch.channels_last),
            (channels_last_3d_shapes, torch.channels_last_3d),
        )

        for shapes, memory_format in pairs:
            for shape in shapes:
                # tests empty
                expected = torch.empty(shape, device=device, dtype=dtype, memory_format=memory_format)
                actual = refs.empty(shape, device=device, dtype=dtype, memory_format=memory_format)
                self.assertEqual(expected.stride(), actual.stride())

                # tests clone
                a = torch.testing.make_tensor(shape, device=device, dtype=dtype)
                expected = torch.clone(a, memory_format=memory_format)
                actual = torch.clone(a, memory_format=memory_format)
                self.assertEqual(expected.stride(), actual.stride())

                # tests contiguous
                a = torch.testing.make_tensor(shape, device=device, dtype=dtype, noncontiguous=True)
                expected = a.contiguous(memory_format=memory_format)
                actual = refs.contiguous(a, memory_format=memory_format)
                self.assertEqual(expected.stride(), actual.stride())

    @dtypes(torch.float32)
    def test_reshape_view_method(self, device, dtype):
        # [FUNCTION DOC] Checks refs reshape and refs view give the same result as the tensor
        # reshape and view methods for a five by five input and a new shape with extra size
        # one dims.
        make_arg = partial(make_tensor, device=device, dtype=dtype)
        a = make_arg((5, 5))
        new_shape = 1, 5, 1, 5
        result_eager = a.reshape(*new_shape)
        result_refs = refs.reshape(a, *new_shape)
        self.assertEqual(result_eager, result_refs)

        result_eager = a.view(*new_shape)
        result_refs = refs.view(a, *new_shape)
        self.assertEqual(result_eager, result_refs)


    @onlyCUDA
    @dtypes(torch.float32)
    def test_philox_rand(self, device, dtype):
        # [FUNCTION DOC] CUDA only. For two sizes, records the CUDA rng state and a reference
        # rand result per repeat, reseeds, then calls the philox_rand rngprim with the saved
        # seed and offset and checks each result equals the reference.
        sizes = (1000, 1000000)  # offsets of 4 and 8
        repeats = 2  # Checks multiple rand calls results with multiple philox_rand calls
        for size in sizes:
            torch.cuda.manual_seed(123)
            references = []
            results = []
            rng_states = []
            for _ in range(repeats):
                rng_states.append(CUDARngStateHelper.get_torch_state_as_tuple())
                references.append(torch.rand(size, device=device, dtype=dtype))

            torch.cuda.manual_seed(123)
            for idx in range(repeats):
                seed, offset = rng_states[idx]
                result, _ = torch.ops.rngprims.philox_rand((size,),
                                                           seed=seed,
                                                           offset=offset,
                                                           stride=None,
                                                           device=device,
                                                           dtype=dtype)
                results.append(result)

            for a, b in zip(references, results):
                self.assertEqual(a, b)


    @dtypes(torch.float32)
    def test_functional_rng_wrappers(self, device, dtype):
        # [FUNCTION DOC] Checks the run_and_save_rng_state and run_with_rng_state wrappers:
        # two saved runs after seeding reproduce two plain rand outputs, and replaying each
        # saved state reproduces the same values.

        torch.manual_seed(123)
        ref1 = torch.rand(10, device=device, dtype=dtype)
        ref2 = torch.rand(10, device=device, dtype=dtype)


        torch.manual_seed(123)
        rng_state1, res1 = torch._prims.rng_prims.run_and_save_rng_state(torch.rand, 10, device=device, dtype=dtype)
        rng_state2, res2 = torch._prims.rng_prims.run_and_save_rng_state(torch.rand, 10, device=device, dtype=dtype)

        res3 = torch._prims.rng_prims.run_with_rng_state(rng_state1, torch.rand, 10, device=device, dtype=dtype)
        res4 = torch._prims.rng_prims.run_with_rng_state(rng_state2, torch.rand, 10, device=device, dtype=dtype)

        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        self.assertEqual(ref1, res3)
        self.assertEqual(ref2, res4)

    def test_functional_rng_wrapper_with_positional_device(self):
        # [FUNCTION DOC] Calls run_and_save_rng_state on the inductor seeds prim passing the
        # device as a positional argument, and checks the returned state and result are on CPU
        # and the result has shape of one element.
        import torch._inductor.inductor_prims

        rng_state, result = torch._prims.rng_prims.run_and_save_rng_state(
            torch.ops.prims.inductor_seeds.default, 1, torch.device("cpu")
        )

        self.assertEqual(rng_state.device.type, "cpu")
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.shape, (1,))

class TestPrimsBasic(TestCase):
    # [CLASS DOC] Device-independent tests of basic prim behaviors: dispatch logging,
    # complex inputs and meta clone stride handling.
    def test_torch_ops(self):
        # [FUNCTION DOC] Checks the prims namespace sin matches torch sin, then logs the
        # dispatch of the prim on a logging tensor and compares the captured log text with an
        # expected string.
        r = make_tensor((2,), device='cpu', dtype=torch.float)
        self.assertEqual(torch.ops.prims.sin(r), torch.sin(r))

        r = LoggingTensor(r)
        with capture_logs() as logs:
            log_input("input", r)
            prims.sin(r)
        self.assertExpectedInline('\n'.join(logs), """\
$0: f32[2] = input('input')
$1: f32[2] = torch._ops.prims.sin.default($0)""")

    def test_mul_complex(self):
        # [FUNCTION DOC] Smoke test: multiplies a random tensor by a complex Python scalar
        # with the mul prim; only checks it does not raise.
        prims.mul(torch.randn(2), 1 + 1j)

    def test_clone_complex(self):
        # [FUNCTION DOC] Smoke test: with the python dispatcher on, adds a scalar to a
        # conjugated complex meta tensor; only checks it does not raise.
        with torch._dispatch.python.enable_python_dispatcher():
            x = torch.randn(4, dtype=torch.complex64, device='meta').conj()
            x + 1

    def test_clone_meta_stride_preservation_dense(self):
        # [FUNCTION DOC] Checks the meta clone function under preserve format keeps the
        # strides of a transposed (non-contiguous but dense) tensor.
        tensor = torch.randn(1, 5).t()
        meta_clone = prims._clone_meta(tensor, memory_format=torch.preserve_format)
        self.assertEqual(tensor.stride(), meta_clone.stride())

    def test_clone_meta_stride_preservation_sparse(self):
        # [FUNCTION DOC] Checks the meta clone function under preserve format on a strided
        # slice of a tensor gives the strides of the contiguous version.
        tensor = torch.arange(12).float().view(3, 4)[1:, ::2]
        meta_clone = prims._clone_meta(tensor, memory_format=torch.preserve_format)
        self.assertEqual(tensor.contiguous().stride(), meta_clone.stride())

    def test_check_deprecation_warning(self):
        # [FUNCTION DOC] Checks the common check helper emits a FutureWarning stating it will
        # be removed in the future, when called with a true condition.
        with self.assertWarnsRegex(FutureWarning, 'will be removed in the future'):
            torch._prims_common.check(True, lambda: 'message')


instantiate_device_type_tests(TestPrims, globals())


class TestRefs(TestCase):
    # [CLASS DOC] Device-generic tests for reference implementations (torch._refs).
    @dtypes(torch.float32)
    def test_constant_pad_nd_memory_format(self, device, dtype):
        # [FUNCTION DOC] Checks refs constant_pad_nd preserves channels last or contiguous
        # format for unambiguous cases, and for two ambiguous stride layouts compares strides
        # with the torch version and checks the expected channels last or contiguous result.
        # Test memory format is preserved in unambiguous cases
        for mf, ndim in (
                (torch.channels_last, 4),
                (torch.contiguous_format, 4),
                (torch.channels_last_3d, 5),
                (torch.contiguous_format, 5),
        ):
            a = torch.zeros([2] * ndim).to(memory_format=mf)
            res = refs.constant_pad_nd(a, pad=[1] * (2 * ndim))
            self.assertTrue(res.is_contiguous(memory_format=mf))

        # Ambiguous cases

        # is_channels_last_ and is_contiguous_, results in channels_last output
        a = torch.empty_strided((2, 1, 2, 2), stride=(4, 1, 2, 1))
        self.assertTrue(a.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(a.is_contiguous())
        actual = refs.constant_pad_nd(a, pad=[1] * 8)
        expect = torch.constant_pad_nd(a, pad=[1] * 8)
        self.assertEqual(actual.stride(), expect.stride())
        self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))

        # is_channels_last_contiguous_ but not is_channels_last_, results in
        # contiguous output
        a = torch.empty_strided((2, 1, 2, 2), stride=(4, 4, 2, 1))
        self.assertTrue(a.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(a.is_contiguous())
        actual = refs.constant_pad_nd(a, pad=[1] * 8)
        expect = torch.constant_pad_nd(a, pad=[1] * 8)
        self.assertEqual(actual.stride(), expect.stride())
        self.assertTrue(actual.is_contiguous())

    def test_unbind(self):
        # [FUNCTION DOC] Checks refs unbind on an empty-dim tensor returns the same as torch
        # unbind. Kept here as it would break assumptions in the backward tests if added to
        # OpInfo.
        # If unbind returns empty tuple, it breaks some assumptions in some backward tests in test_ops.py.
        # So can't put this test into common_methods_invocations.py.
        a = torch.rand([3, 0, 4])
        actual = refs.unbind(a, 1)
        expect = torch.unbind(a, 1)
        self.assertEqual(actual, expect)

    def test_logspace_with_complex_input(self):
        # [FUNCTION DOC] Compares refs logspace with torch logspace for a complex end value.
        actual = refs.logspace(2, 10 + 5j, steps=5)
        expect = torch.logspace(2, 10 + 5j, steps=5)
        self.assertEqual(actual, expect)

    def test_linspace_with_complex_input(self):
        # [FUNCTION DOC] Compares refs linspace with torch linspace for a complex end value.
        actual = refs.linspace(2, 10 + 5j, steps=5)
        expect = torch.linspace(2, 10 + 5j, steps=5)
        self.assertEqual(actual, expect)

    # From https://github.com/pytorch/pytorch/issues/109558
    def test_infinite_loop_from_py_dispatcher(self):
        # [FUNCTION DOC] Regression test for an infinite loop in the python dispatcher: with
        # it enabled, moves a tensor to the meta device; passes if it terminates without
        # error.
        # enables prim decomps
        with torch._dispatch.python.enable_python_dispatcher():
            x = torch.ones(4)
            x.to(device="meta")

    def test_inferred_tags(self):
        # [FUNCTION DOC] Checks the tags on the normal prim default overload are the
        # nondeterministic seeded tag and the pt2 compliant tag.
        self.assertEqual(torch.ops.prims.normal.default.tags, (torch.Tag.nondeterministic_seeded, torch.Tag.pt2_compliant_tag))



instantiate_device_type_tests(TestRefs, globals())


class TestDecomp(TestCase):
    # [CLASS DOC] Tests that decompositions via the refs mode work for method variants
    # taking varargs.
    @ops([op for op in op_db if op.supports_varargs], dtypes=OpDTypes.any_one)
    def test_decomposition_method_vararg(self, device, dtype, op):
        # [FUNCTION DOC] For OpInfo entries that support varargs, takes the first sample whose
        # last argument is a non-empty sequence, traces the factory or method variant under
        # the refs mode with that sequence expanded as varargs, then reruns the traced graph
        # and the original under the same seed and compares results.
        # some ops have vararg variants for the methods. this tests it.
        # we don't have tests for varargs in OpInfo, so we need to
        # improvise this a bit.
        # The rule for general functions (the special cases being e.g. tensor
        # creation functions taking shapes) is that things can be vararg
        # if the method has only one argument of sequence type.
        # e.g. permute can be called on a 3d tensor t as t.permute(0, 2, 1)
        #      as well as t.permute([0, 2, 1])
        #      when the signature in native_functions.yaml
        #      shows arguments Tensor self, IntList dims
        # we might need to adjust things for the factory functions or
        # have them do their own test
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsMode

        # filter out empty tuple as that cannot be the varargs
        sample_inputs = (si for si in op.sample_inputs(device, dtype, requires_grad=False)
                         if (si.args[-1] if si.args else si.input))

        # just run one test, we assume there is a suitable one in the tests
        sample_input = next(sample_inputs)
        all_args = (sample_input.input,) + sample_input.args

        # in general, the methods take varargs and not (always?) the function
        # variants, the exception to this rule are the factory functions
        if op.is_factory_function:
            fn = op.op
        else:
            fn = op.method_variant
        with TorchRefsMode():
            gm = make_fx(fn)(*all_args[:-1], *all_args[-1])

        # in case we add random factory functions
        torch.manual_seed(1)
        res = gm(*all_args[:-1], *all_args[-1])
        torch.manual_seed(1)
        expected = fn(*all_args[:-1], *all_args[-1])
        self.assertEqual(res, expected)


instantiate_device_type_tests(TestDecomp, globals())


if __name__ == "__main__":
    run_tests()
