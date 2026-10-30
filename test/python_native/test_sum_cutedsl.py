# Owner(s): ["module: dsl-native-ops"]
#
# Correctness + bitwise-equivalence tests for the CuTeDSL ``aten::sum``
# inner-tree override (``torch/_native/ops/sum``). The override is gated on
# ``PYTORCH_SUM_INNER_TREE`` and registered for CUDA; these tests are gated on
# CuTeDSL availability so the assertions exercise the CuTeDSL kernel.
#
# The bitwise tests pin the reduction output to precomputed SHA-256 hashes of
# deterministic row sums. The hashes are the same ones the (now removed) ATen
# CUDA inner-tree kernel produced -- the CuTeDSL kernel reproduces them
# bit-for-bit, which is the contract for this op.
# [MODULE DOC] CUDA tests for the CuTeDSL sum override in torch._native.ops.sum, enabled by the
# PYTORCH_SUM_INNER_TREE environment flag. Covers the dispatch predicate (accept and reject
# cases), agreement with aten, out variant, strided input, partial last block, fall-through for
# integer and complex dtypes, and bitwise checks of row sums against stored truncated SHA-256
# digests for float8, float16, float32 and float64 over many shapes. Skipped without CUDA or
# CuTeDSL.

import contextlib
import os
import unittest

import torch
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FP8,
    SM100OrLater,
    TEST_CUDA,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    skipIfRocm,
    TestCase,
)


def _cutedsl_impl():
    # [FUNCTION DOC] HELPER: Lazily imports and returns the torch._native.ops.sum.cutedsl_impl
    # module so importing this test file does not load it.
    from torch._native.ops.sum import cutedsl_impl

    return cutedsl_impl


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestSumCuteDSLOverride(TestCase):
    # [CLASS DOC] Test case for the CuTeDSL inner-tree sum override. Class attributes: the
    # accumulator dtype per input dtype, the list of (rows, columns) shapes, the table of
    # expected 16-hex-character digests keyed by dtype name and shape, and the mapping from
    # dtype names to dtypes. Tests turn the override on by setting an environment variable
    # inside a context manager.
    """Tests for the CuTeDSL inner-tree reduction sum override."""

    _ACC_DTYPES = {
        torch.float8_e4m3fn: torch.float32,
        torch.float16: torch.float32,
        torch.float32: torch.float32,
        torch.float64: torch.float64,
    }

    _SHAPES = [
        (1, 32),
        (1, 8192),
        (1, 16384),
        (128, 32),
        (128, 8192),
        (128, 16384),
        (4096, 32),
        (4096, 8192),
        (4096, 16384),
        (1, 27),
        (1, 6561),
        (1, 19683),
        (128, 27),
        (128, 6561),
        (128, 19683),
        (4096, 27),
        (4096, 6561),
        (4096, 19683),
        (1, 36),
        (1, 7776),
        (1, 46656),
        (128, 36),
        (128, 7776),
        (128, 46656),
        (4096, 36),
        (4096, 7776),
        (4096, 46656),
    ]

    # fmt: off
    _EXPECTED = {
        ("float8_e4m3fn", 1, 32): "df3f619804a92fdb",
        ("float8_e4m3fn", 1, 8192): "df3f619804a92fdb",
        ("float8_e4m3fn", 1, 16384): "df3f619804a92fdb",
        ("float8_e4m3fn", 128, 32): "f428ef9e3fbd9469",
        ("float8_e4m3fn", 128, 8192): "4834001f0135cd23",
        ("float8_e4m3fn", 128, 16384): "76e9b171932d1f9f",
        ("float8_e4m3fn", 4096, 32): "a5c21039e21a5323",
        ("float8_e4m3fn", 4096, 8192): "e34643d4c484ef9c",
        ("float8_e4m3fn", 4096, 16384): "289c5f23e410c913",
        ("float8_e4m3fn", 1, 27): "c68830a25204a09f",
        ("float8_e4m3fn", 1, 6561): "c68830a25204a09f",
        ("float8_e4m3fn", 1, 19683): "c68830a25204a09f",
        ("float8_e4m3fn", 128, 27): "2252a0a2c19ada94",
        ("float8_e4m3fn", 128, 6561): "6bf0fe73bdad39f5",
        ("float8_e4m3fn", 128, 19683): "03b2636d8210d4d0",
        ("float8_e4m3fn", 4096, 27): "41896635bfcb0c7c",
        ("float8_e4m3fn", 4096, 6561): "990ad1490c767a17",
        ("float8_e4m3fn", 4096, 19683): "c3cc65b050631480",
        ("float8_e4m3fn", 1, 36): "df3f619804a92fdb",
        ("float8_e4m3fn", 1, 7776): "df3f619804a92fdb",
        ("float8_e4m3fn", 1, 46656): "df3f619804a92fdb",
        ("float8_e4m3fn", 128, 36): "927688b93e357722",
        ("float8_e4m3fn", 128, 7776): "e46696a8a4613062",
        ("float8_e4m3fn", 128, 46656): "a10c4d305d19a761",
        ("float8_e4m3fn", 4096, 36): "efe1cee0d5e0bf4e",
        ("float8_e4m3fn", 4096, 7776): "d5588816762acbd4",
        ("float8_e4m3fn", 4096, 46656): "27fc2749145207e3",
        ("float16", 1, 32): "df3f619804a92fdb",
        ("float16", 1, 8192): "df3f619804a92fdb",
        ("float16", 1, 16384): "df3f619804a92fdb",
        ("float16", 128, 32): "f428ef9e3fbd9469",
        ("float16", 128, 8192): "4834001f0135cd23",
        ("float16", 128, 16384): "76e9b171932d1f9f",
        ("float16", 4096, 32): "a5c21039e21a5323",
        ("float16", 4096, 8192): "e34643d4c484ef9c",
        ("float16", 4096, 16384): "289c5f23e410c913",
        ("float16", 1, 27): "c68830a25204a09f",
        ("float16", 1, 6561): "c68830a25204a09f",
        ("float16", 1, 19683): "c68830a25204a09f",
        ("float16", 128, 27): "2252a0a2c19ada94",
        ("float16", 128, 6561): "6bf0fe73bdad39f5",
        ("float16", 128, 19683): "03b2636d8210d4d0",
        ("float16", 4096, 27): "41896635bfcb0c7c",
        ("float16", 4096, 6561): "990ad1490c767a17",
        ("float16", 4096, 19683): "c3cc65b050631480",
        ("float16", 1, 36): "df3f619804a92fdb",
        ("float16", 1, 7776): "df3f619804a92fdb",
        ("float16", 1, 46656): "df3f619804a92fdb",
        ("float16", 128, 36): "927688b93e357722",
        ("float16", 128, 7776): "e46696a8a4613062",
        ("float16", 128, 46656): "a10c4d305d19a761",
        ("float16", 4096, 36): "efe1cee0d5e0bf4e",
        ("float16", 4096, 7776): "d5588816762acbd4",
        ("float16", 4096, 46656): "27fc2749145207e3",
        ("float32", 1, 32): "df3f619804a92fdb",
        ("float32", 1, 8192): "df3f619804a92fdb",
        ("float32", 1, 16384): "df3f619804a92fdb",
        ("float32", 128, 32): "f428ef9e3fbd9469",
        ("float32", 128, 8192): "4834001f0135cd23",
        ("float32", 128, 16384): "76e9b171932d1f9f",
        ("float32", 4096, 32): "a5c21039e21a5323",
        ("float32", 4096, 8192): "e34643d4c484ef9c",
        ("float32", 4096, 16384): "289c5f23e410c913",
        ("float32", 1, 27): "c68830a25204a09f",
        ("float32", 1, 6561): "c68830a25204a09f",
        ("float32", 1, 19683): "c68830a25204a09f",
        ("float32", 128, 27): "2252a0a2c19ada94",
        ("float32", 128, 6561): "6bf0fe73bdad39f5",
        ("float32", 128, 19683): "03b2636d8210d4d0",
        ("float32", 4096, 27): "41896635bfcb0c7c",
        ("float32", 4096, 6561): "990ad1490c767a17",
        ("float32", 4096, 19683): "c3cc65b050631480",
        ("float32", 1, 36): "df3f619804a92fdb",
        ("float32", 1, 7776): "df3f619804a92fdb",
        ("float32", 1, 46656): "df3f619804a92fdb",
        ("float32", 128, 36): "927688b93e357722",
        ("float32", 128, 7776): "e46696a8a4613062",
        ("float32", 128, 46656): "a10c4d305d19a761",
        ("float32", 4096, 36): "efe1cee0d5e0bf4e",
        ("float32", 4096, 7776): "d5588816762acbd4",
        ("float32", 4096, 46656): "27fc2749145207e3",
        ("float64", 1, 32): "af5570f5a1810b7a",
        ("float64", 1, 8192): "af5570f5a1810b7a",
        ("float64", 1, 16384): "af5570f5a1810b7a",
        ("float64", 128, 32): "35a0993dedd64659",
        ("float64", 128, 8192): "05102af99bde7f20",
        ("float64", 128, 16384): "c54af6a4a43a21d4",
        ("float64", 4096, 32): "a5e02a44cc2dd361",
        ("float64", 4096, 8192): "e9ba5b9f0409fec2",
        ("float64", 4096, 16384): "6d8144f96bb8a210",
        ("float64", 1, 27): "e77817b649821c63",
        ("float64", 1, 6561): "e77817b649821c63",
        ("float64", 1, 19683): "e77817b649821c63",
        ("float64", 128, 27): "86e9b1f2d6fd4504",
        ("float64", 128, 6561): "e761055386fc7295",
        ("float64", 128, 19683): "3ce4cdd14e0b152f",
        ("float64", 4096, 27): "4926c5d5541728d1",
        ("float64", 4096, 6561): "417525b3b5d3de87",
        ("float64", 4096, 19683): "cd8ef6c48ad2c67f",
        ("float64", 1, 36): "af5570f5a1810b7a",
        ("float64", 1, 7776): "af5570f5a1810b7a",
        ("float64", 1, 46656): "af5570f5a1810b7a",
        ("float64", 128, 36): "f00a064b84ba50b4",
        ("float64", 128, 7776): "c227b341b6240d30",
        ("float64", 128, 46656): "af1f2b32fdaf1f62",
        ("float64", 4096, 36): "5cb245154bac020c",
        ("float64", 4096, 7776): "751916d8d6b9dcba",
        ("float64", 4096, 46656): "498bc246ba0b069a",
    }
    # fmt: on

    _DTYPE_MAP = {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    @contextlib.contextmanager
    def _inner_tree_flag(self):
        # [FUNCTION DOC] HELPER: Context manager that sets the PYTORCH_SUM_INNER_TREE
        # environment variable to the string 1 for its body and afterwards restores the previous
        # value, or removes the variable if it was unset (done in finally).
        old_value = os.environ.get("PYTORCH_SUM_INNER_TREE")
        os.environ["PYTORCH_SUM_INNER_TREE"] = "1"
        try:
            yield
        finally:
            if old_value is None:
                os.environ.pop("PYTORCH_SUM_INNER_TREE", None)
            else:
                os.environ["PYTORCH_SUM_INNER_TREE"] = old_value

    def _make_input(self, m, n, dtype):
        # [FUNCTION DOC] HELPER: Builds a deterministic CUDA input of m rows and n columns in
        # the given dtype. Columns alternate between -1 and 1; when m is above 1 each row adds
        # an offset cycling through five values from -0.5 to 0.5. Computed in float64 for
        # float64, otherwise float32, then cast to the dtype.
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        cols = torch.arange(n, device="cuda", dtype=compute_dtype)
        values = ((cols % 2) * 2 - 1).reshape(1, n)
        if m > 1:
            rows = torch.arange(m, device="cuda", dtype=compute_dtype).reshape(m, 1)
            values = values + ((rows % 5) - 2) / 4
        return values.to(dtype)

    def _eager_sum(self, x):
        # [FUNCTION DOC] HELPER: Row sum over dim 1 after converting the input to the accumulator
        # dtype from the class table (float32 for float8 and float16, same dtype otherwise);
        # the float32 result is the one the bitwise test hashes. Returns the sum tensor.
        acc_dtype = self._ACC_DTYPES[x.dtype]
        if x.dtype == torch.float8_e4m3fn:
            return x.to(torch.float32).sum(dim=1).to(acc_dtype)
        return x.to(acc_dtype).sum(dim=1)

    def _sha(self, t):
        # [FUNCTION DOC] HELPER: Returns the first 16 hex characters of the SHA-256 of the raw
        # bytes of the given tensor, after moving it to CPU and making it contiguous.
        import hashlib

        return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()[:16]

    def _make_order_sensitive_input(self, m, n, dtype):
        # [FUNCTION DOC] HELPER: Builds an m by n float64 CUDA tensor from the running element
        # index using two mod-based patterns (period 29 and 7) so floating point summation order
        # changes the result, then casts to the requested dtype.
        values = torch.arange(m * n, device="cuda", dtype=torch.float64).reshape(m, n)
        return (((values % 29) - 14) / 29 + ((values % 7) - 3) / 13).to(dtype)

    # --- predicate accept/reject ---

    def test_cond_accepts_covered_shapes(self):
        # [FUNCTION DOC] With the flag set, checks the override predicate accepts a 128 by 8192
        # input and a row-strided view of a 256 by 8192 input for both float32 and float64,
        # reducing dim 1.
        impl = _cutedsl_impl()
        with self._inner_tree_flag():
            for dtype in (torch.float32, torch.float64):
                x = torch.randn(128, 8192, device="cuda", dtype=dtype)
                self.assertTrue(impl._cond(x, [1]))
                # Strided outer input (contiguous reduced dim) is still covered.
                strided = torch.randn(256, 8192, device="cuda", dtype=dtype)[::2]
                self.assertTrue(impl._cond(strided, [1]))

    def test_cond_rejects_unsupported(self):
        # [FUNCTION DOC] Checks the predicate rejects: any input without the flag; multi-dim,
        # full (no dim) and dtype-casting reductions; a non-contiguous reduced dim; int64 and
        # complex64 inputs, all with the flag set.
        impl = _cutedsl_impl()
        x = torch.randn(128, 8192, device="cuda", dtype=torch.float32)
        # No flag -> reject.
        self.assertFalse(impl._cond(x, [1]))
        with self._inner_tree_flag():
            # Multi-dim and full reductions are out of scope.
            self.assertFalse(impl._cond(x, [0, 1]))
            self.assertFalse(impl._cond(x, None))
            # dtype-casting sum is out of scope.
            self.assertFalse(impl._cond(x, [1], dtype=torch.float64))
            # Non-contiguous reduced dim.
            self.assertFalse(impl._cond(x[:, ::2], [1]))
            # Integer / complex dtypes fall through to aten.
            self.assertFalse(
                impl._cond(torch.ones(17, 8192, device="cuda", dtype=torch.int64), [1])
            )
            self.assertFalse(
                impl._cond(
                    torch.ones(17, 8192, device="cuda", dtype=torch.complex64), [1]
                )
            )

    # --- correctness ---

    @parametrize("dtype", [torch.float32, torch.float64])
    def test_override_matches_aten(self, dtype):
        # [FUNCTION DOC] Parametrized over float32 and float64: an order-sensitive 128 by 8192
        # sum with the override must equal the aten result computed with the cutedsl DSL
        # disabled.
        x = self._make_order_sensitive_input(128, 8192, dtype)
        with torch.backends.python_native.cutedsl.disabled():
            ref = x.sum(dim=1)
        with self._inner_tree_flag():
            got = x.sum(dim=1)
        self.assertEqual(got, ref)

    def test_override_out_variant(self):
        # [FUNCTION DOC] Same comparison for float32 but the override writes into a preallocated
        # out tensor; checks out equals the aten reference.
        x = self._make_order_sensitive_input(128, 8192, torch.float32)
        with torch.backends.python_native.cutedsl.disabled():
            ref = x.sum(dim=1)
        out = torch.empty(128, device="cuda", dtype=torch.float32)
        with self._inner_tree_flag():
            torch.sum(x, dim=1, out=out)
        self.assertEqual(out, ref)

    def test_strided_outer_input(self):
        # [FUNCTION DOC] Sums every other row of an all-ones matrix with 32 columns (other rows
        # hold fives, skipped by the stride) with the override; expects 128 results all equal to
        # 32.
        base = torch.ones(256, 32, device="cuda", dtype=torch.float32)
        base[1::2, :] = 5
        x = base[::2, :]
        with self._inner_tree_flag():
            result = x.sum(dim=1)
        self.assertEqual(result, torch.full((128,), 32, device="cuda", dtype=x.dtype))

    def test_looped_kernel_partial_last_block(self):
        # [FUNCTION DOC] Sums a 129 by 256 matrix of ones with the override (row count not a
        # multiple of the block); expects 129 results of 256.
        x = torch.ones(129, 256, device="cuda", dtype=torch.float32)
        with self._inner_tree_flag():
            result = x.sum(dim=1)
        self.assertEqual(result, torch.full((129,), 256, device="cuda", dtype=x.dtype))

    @parametrize("dtype", [torch.int64, torch.complex64])
    def test_integer_and_complex_fall_through(self, dtype):
        # [FUNCTION DOC] Parametrized over int64 and complex64: sums a 17 by 8192 matrix of ones
        # with the flag set; the override declines these dtypes and aten must give 8192 per row.
        # CuTeDSL declines integer/complex; the call must fall through to aten.
        x = torch.ones(17, 8192, device="cuda", dtype=dtype)
        with self._inner_tree_flag():
            result = x.sum(dim=1)
        self.assertEqual(result, torch.full((17,), 8192, device="cuda", dtype=dtype))

    # --- bitwise equivalence ---

    @skipIfRocm
    @unittest.skipUnless(
        SM100OrLater, "SM100 hash is generated on GB200 with CUDA 12.8"
    )
    def test_cross_warp_order_sensitive_hash_sm100(self):
        # [FUNCTION DOC] Skipped on ROCm and before SM100. Sums an order-sensitive 17 by 8192
        # float32 input with the override and compares the truncated digest of the result with a
        # hard-coded value generated on GB200.
        x = self._make_order_sensitive_input(17, 8192, torch.float32)
        with self._inner_tree_flag():
            result = x.sum(dim=1)
        self.assertEqual(self._sha(result), "75d8b1a702344e90")

    @skipIfRocm
    def test_bitwise(self):
        # [FUNCTION DOC] Skipped on ROCm. For each dtype (float8 only when the platform supports
        # it) and each shape in the class list, builds the deterministic input, sums over dim 1
        # with the flag set, and compares the digest to the stored expected value in a subtest,
        # with a message naming dtype and shape.
        for dtype_name, dtype in self._DTYPE_MAP.items():
            if dtype == torch.float8_e4m3fn and not PLATFORM_SUPPORTS_FP8:
                continue
            for m, n in self._SHAPES:
                with self.subTest(dtype_name=dtype_name, m=m, n=n):
                    x = self._make_input(m, n, dtype)
                    with self._inner_tree_flag():
                        result = self._eager_sum(x)
                    sha = self._sha(result)
                    expected = self._EXPECTED[(dtype_name, m, n)]
                    self.assertEqual(
                        sha,
                        expected,
                        f"{dtype_name} m={m} n={n}: got {sha}, expected {expected}",
                    )


instantiate_parametrized_tests(TestSumCuteDSLOverride)

if __name__ == "__main__":
    run_tests()
