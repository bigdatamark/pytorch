# Owner(s): ["module: dsl-native-ops"]
#
# Correctness tests for the CuTeDSL topk override (register + radix
# kernels).
#
# Radix kernel (K in {64,128,256,512,1024}) has two specialisations:
#   * Default (non-deterministic): atomic-counter gather + ord-only sort.
#     Returns correct top-K values; indices may differ from aten on ties
#     and may differ across successive calls. Tests verify values match
#     aten bit-exactly and ``gather(input, indices)`` reproduces values.
#   * Deterministic (under torch.use_deterministic_algorithms): prefix-sum
#     gather + lex ``(ord, -idx)`` sort. Bit-exact aten match on both
#     values and indices, stable across runs.
#
# Register kernel (K in {16,32}) is bit-exact on values and stable across
# runs. Index ordering on ties is ``(value desc, idx asc)``, which doesn't
# match aten's small-K CUDA topk; tests check value-equality + gather
# round-trip rather than index-equality.
# [MODULE DOC] CUDA-only correctness tests comparing the CuTeDSL topk override against aten
# (obtained by temporarily disabling the cutedsl DSL via the python_native backend). Covers the
# register kernel (K of 16 or 32) and the radix kernel (K from 64 to 1024), random, duplicate,
# extreme, NaN and N-dimensional inputs, out variants, fall-through for unsupported K or N,
# deterministic mode, autograd and a non-current device. Module constants list the supported K
# values. Skipped without CUDA or CuTeDSL.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


_RADIX_KS = (64, 128, 256, 512, 1024)
_REGISTER_KS = (16, 32)
_SUPPORTED_KS = _REGISTER_KS + _RADIX_KS


def _radix_min_n_for_k(k: int) -> int:
    # [FUNCTION DOC] HELPER: Returns the minimum row length N for which the radix kernel accepts
    # the given K: the per-K multiplier from the cutedsl_impl table times K. Imports the table
    # lazily inside the function.
    """Mirror of the radix kernel's per-K N gate; used to size test
    shapes so the override actually fires rather than falling through
    to aten."""
    from torch._native.ops.topk.cutedsl_impl import _RADIX_MIN_N_MULTIPLIER

    return _RADIX_MIN_N_MULTIPLIER[k] * k


def _test_n(k: int) -> int:
    # [FUNCTION DOC] HELPER: Chooses a row length N for which the override fires. For register K
    # values returns the upper end of the allowed N range for that K; for radix K values returns
    # the larger of the radix minimum N and 4096. Imports the range table lazily.
    """Pick a test N for which the override fires."""
    if k in _REGISTER_KS:
        # Register kernel has a (min, max) N range per K. Pick max so it
        # exercises the largest in-range tile.
        from torch._native.ops.topk.cutedsl_impl import _REGISTER_N_RANGE

        return _REGISTER_N_RANGE[k][1]
    return max(_radix_min_n_for_k(k), 4096)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestCuTeDSLTopK(TestCase):
    # [CLASS DOC] CUDA test case for the CuTeDSL topk override. Each test builds float32 CUDA
    # inputs sized by the test-N helper and compares against aten results computed with the
    # cutedsl DSL disabled. Many tests are parametrized over K by
    # instantiate_parametrized_tests.
    def _assert_topk_matches_aten(self, x: torch.Tensor, k: int) -> None:
        # [FUNCTION DOC] HELPER: Asserts topk on the input with the given K along the last dim
        # equals the aten reference in values (exact), that gathering the input at the returned
        # indices reproduces the values, and (for K of at least 2) that values are
        # non-increasing. Indices are not compared with aten because ties may differ.
        """Compare the override (default non-deterministic mode) against
        aten. Values must match bit-exactly; indices may differ on ties
        but ``gather(input, indices)`` must still reproduce values."""
        pn = torch.backends.python_native
        with pn.cutedsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        got_v, got_i = torch.topk(x, k, dim=-1)
        self.assertEqual(got_v, ref_v)
        gathered = torch.gather(x, -1, got_i)
        self.assertEqual(gathered, got_v)
        # descending
        if k >= 2:
            diffs = got_v[..., :-1] - got_v[..., 1:]
            self.assertTrue((diffs >= 0).all(), "output is not descending")

    @parametrize("k", _SUPPORTED_KS)
    def test_correctness_random_gaussian(self, k: int) -> None:
        # [FUNCTION DOC] Random normal float32 input with 256 rows; checks the override matches
        # aten for the parametrized K. Seeds the RNG for reproducibility.
        torch.manual_seed(0)
        # M chosen to exceed typical SM count so the override actually fires.
        x = torch.randn(256, _test_n(k), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", _SUPPORTED_KS)
    def test_correctness_with_duplicates(self, k: int) -> None:
        # [FUNCTION DOC] Input of small random integers (many duplicate values) to stress the
        # threshold-bin gather where tie-breaking of indices is non-deterministic; checks values
        # and gather round trip.
        """Duplicates stress the threshold-bin gather path where
        non-deterministic index tie-breaking occurs."""
        torch.manual_seed(1)
        x = torch.randint(0, 50, (256, _test_n(k)), device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", _SUPPORTED_KS)
    def test_correctness_with_extreme_values(self, k: int) -> None:
        # [FUNCTION DOC] Random input whose first four columns are set to positive infinity,
        # negative infinity, and plus and minus 1e38; checks the override against aten.
        torch.manual_seed(2)
        x = torch.randn(256, _test_n(k), device="cuda", dtype=torch.float32)
        x[:, 0] = float("inf")
        x[:, 1] = float("-inf")
        x[:, 2] = 1e38
        x[:, 3] = -1e38
        self._assert_topk_matches_aten(x, k)

    @parametrize("k", _SUPPORTED_KS)
    def test_correctness_with_nan(self, k: int) -> None:
        # [FUNCTION DOC] Input with a positive NaN, a negative-sign NaN (built from raw bits),
        # plus and minus infinity in the first four columns. Checks gather round trip, equal NaN
        # count per row versus aten, and bit-equal finite values after masking out NaNs. Index
        # and NaN position equality with aten is deliberately not checked because aten's
        # ordering of NaNs is unstable.
        """NaN of either sign must be classified as NaN (sorts at the top),
        matching aten's TopKTypeConfig<float> behaviour. The radix-ordinal
        encoder flags NaN explicitly so negative-sign NaN doesn't sort below
        -inf. We can't compare bit-exactly with aten because aten's final
        sort over the K gathered values uses fp32 ``<`` (NaN comparisons all
        false), so NaN positions within the top-K are unstable on aten's
        side. Instead we check the invariants: the same number of NaNs and
        the same set of finite values appear in our output and aten's."""
        import struct

        torch.manual_seed(10)
        x = torch.randn(256, _test_n(k), device="cuda", dtype=torch.float32)
        neg_nan = struct.unpack("<f", struct.pack("<I", 0xFFC00000))[0]
        x[:, 0] = float("nan")
        x[:, 1] = neg_nan
        x[:, 2] = float("inf")
        x[:, 3] = float("-inf")
        pn = torch.backends.python_native
        with pn.cutedsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        got_v, got_i = torch.topk(x, k, dim=-1)
        # gather round-trip: every output value must come from the input
        # at the reported index (works for NaN since assertEqual treats
        # NaN==NaN).
        self.assertEqual(torch.gather(x, -1, got_i), got_v)
        # Same NaN count per row as aten (proves negative-NaN didn't sink
        # below -inf).
        self.assertEqual(got_v.isnan().sum(dim=-1), ref_v.isnan().sum(dim=-1))
        # Finite tails must match aten bit-exactly (sorted descending).
        ref_finite = ref_v.masked_select(~ref_v.isnan()).reshape(256, -1)
        got_finite = got_v.masked_select(~got_v.isnan()).reshape(256, -1)
        self.assertEqual(got_finite, ref_finite)

    @parametrize("k", _SUPPORTED_KS)
    def test_nd_input(self, k: int) -> None:
        # [FUNCTION DOC] Uses a 3-dimensional input; checks the result matches aten and that the
        # values output has the leading dims preserved with last dim K.
        """Leading dims should be flattened to 2D; output should be
        reshaped back to self.shape[:-1] + (k,)."""
        torch.manual_seed(3)
        N = _test_n(k)
        x = torch.randn(4, 64, N, device="cuda", dtype=torch.float32)
        self._assert_topk_matches_aten(x, k)
        got_v, _ = torch.topk(x, k, dim=-1)
        self.assertEqual(got_v.shape, (4, 64, k))

    @parametrize("k", _SUPPORTED_KS)
    def test_out_variant(self, k: int) -> None:
        # [FUNCTION DOC] Calls topk with preallocated output tensors; checks the returned
        # objects are those same tensors, values equal the aten reference and gather round trip
        # holds.
        torch.manual_seed(4)
        x = torch.randn(256, _test_n(k), device="cuda", dtype=torch.float32)
        pn = torch.backends.python_native
        with pn.cutedsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)

        out_v = torch.empty(256, k, dtype=torch.float32, device="cuda")
        out_i = torch.empty(256, k, dtype=torch.int64, device="cuda")
        got_v, got_i = torch.topk(x, k, dim=-1, out=(out_v, out_i))
        self.assertIs(got_v, out_v)
        self.assertIs(got_i, out_i)
        self.assertEqual(got_v, ref_v)
        self.assertEqual(torch.gather(x, -1, got_i), got_v)

    def test_unsupported_k_falls_through(self) -> None:
        # [FUNCTION DOC] For K values 8, 100 and 2048 (outside the supported set) on 16384-wide
        # rows, checks both values and indices equal the aten result, i.e. the override falls
        # through to aten.
        """K values outside the supported set must hit aten and stay correct."""
        torch.manual_seed(5)
        x = torch.randn(256, 16384, device="cuda", dtype=torch.float32)
        pn = torch.backends.python_native
        for bad_k in (8, 100, 2048):
            with pn.cutedsl.disabled():
                ref = torch.topk(x, bad_k, dim=-1)
            got = torch.topk(x, bad_k, dim=-1)
            self.assertEqual(got.values, ref.values)
            self.assertEqual(got.indices, ref.indices)

    def test_register_n_out_of_range_falls_through(self) -> None:
        # [FUNCTION DOC] With K of 32 and rows of width 512 (beyond the register kernel's N
        # cap), checks values and indices exactly equal the aten result, so the override falls
        # through.
        """Register K values with N outside the per-K cap should fall through."""
        torch.manual_seed(8)
        pn = torch.backends.python_native
        # K=32 register accepts N <= 256; N=512 must fall through (radix
        # doesn't accept K=32 either).
        x = torch.randn(256, 512, device="cuda", dtype=torch.float32)
        with pn.cutedsl.disabled():
            ref = torch.topk(x, 32, dim=-1)
        got = torch.topk(x, 32, dim=-1)
        self.assertEqual(got.values, ref.values)
        self.assertEqual(got.indices, ref.indices)

    @parametrize("k", _RADIX_KS)
    def test_deterministic_mode_matches_aten_with_heavy_ties(self, k: int) -> None:
        # [FUNCTION DOC] Radix K values on tie-heavy integer data: with deterministic algorithms
        # enabled, two successive calls must be identical and must match aten exactly in both
        # values and indices. The previous deterministic setting is restored in finally.
        """Under ``torch.use_deterministic_algorithms`` the radix kernel
        uses prefix-sum gather + lex ``(ord, -idx)`` sort and must match
        aten bit-exactly on both values and indices, even with heavy ties."""
        torch.manual_seed(6)
        x = torch.randint(0, 4, (256, _test_n(k)), device="cuda", dtype=torch.float32)
        pn = torch.backends.python_native
        with pn.cutedsl.disabled():
            ref_v, ref_i = torch.topk(x, k, dim=-1)

        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            v1, i1 = torch.topk(x, k, dim=-1)
            # Determinism: two successive calls return identical results.
            v2, i2 = torch.topk(x, k, dim=-1)
        finally:
            torch.use_deterministic_algorithms(prior)
        self.assertEqual(v1, v2)
        self.assertEqual(i1, i2)
        # And bit-exact match to aten.
        self.assertEqual(v1, ref_v)
        self.assertEqual(i1, ref_i)

    @parametrize("k", _REGISTER_KS)
    def test_register_stable_with_heavy_ties(self, k: int) -> None:
        # [FUNCTION DOC] Register K values on tie-heavy data: checks values are stable across
        # two calls, indices are identical across calls, values equal aten and gather round trip
        # holds. Index equality with aten is not required.
        """Register kernel index order on ties is ``(value desc, idx asc)``,
        which doesn't match aten's small-K CUDA topk. Check the weaker
        contract: bit-exact values, gather round-trip, and stability
        across successive calls."""
        torch.manual_seed(9)
        x = torch.randint(0, 4, (256, _test_n(k)), device="cuda", dtype=torch.float32)
        pn = torch.backends.python_native
        with pn.cutedsl.disabled():
            ref_v, _ = torch.topk(x, k, dim=-1)
        v1, i1 = torch.topk(x, k, dim=-1)
        v2, i2 = torch.topk(x, k, dim=-1)
        self.assertEqual(v1, v2)
        self.assertEqual(i1, i2)
        self.assertEqual(v1, ref_v)
        self.assertEqual(torch.gather(x, -1, i1), v1)

    def test_autograd_passes_through(self) -> None:
        # [FUNCTION DOC] Runs topk with gradient tracking on a 256 by 4096 input with K of 256,
        # sums values and backpropagates; checks the gradient is ones scattered at the selected
        # indices and zeros elsewhere.
        """aten's derivative handles backward; override runs at CUDA key
        below Autograd, so grad should still be the standard scatter-of-1s."""
        torch.manual_seed(7)
        x = torch.randn(
            256, 4096, device="cuda", dtype=torch.float32, requires_grad=True
        )
        v, i = torch.topk(x, 256, dim=-1)
        v.sum().backward()
        self.assertIsNotNone(x.grad)
        expected = torch.zeros_like(x)
        expected.scatter_(-1, i, 1.0)
        self.assertEqual(x.grad, expected)

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "requires at least 2 visible CUDA devices",
    )
    def test_non_current_device(self) -> None:
        # [FUNCTION DOC] Needs at least two CUDA devices. With device 0 current and input on
        # device 1, K of 32: checks the override still works on the input's device, leaves the
        # current device unchanged, returns values on device 1 matching aten and satisfying
        # gather round trip. Restores the original current device in finally.
        """The override must launch on the input's device even when it isn't
        the current device. The Python-native dispatch path gets no automatic
        CUDA device guard, so without the guard the CuTeDSL kernel launches on
        the current device's stream and rejects the off-device input (mirrors
        the bmm override regression fixed in #187983)."""
        k = 32  # register kernel; _test_n(k) keeps the override in range
        old_device = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            x = torch.randn(256, _test_n(k), device="cuda:1", dtype=torch.float32)
            pn = torch.backends.python_native
            with pn.cutedsl.disabled():
                ref_v, _ = torch.topk(x, k, dim=-1)
            got_v, got_i = torch.topk(x, k, dim=-1)
            # current device unchanged; result stays on the input's device
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(got_v.device, torch.device("cuda:1"))
            self.assertEqual(got_v, ref_v)
            self.assertEqual(torch.gather(x, -1, got_i), got_v)
        finally:
            torch.cuda.set_device(old_device)


instantiate_parametrized_tests(TestCuTeDSLTopK)


if __name__ == "__main__":
    run_tests()
