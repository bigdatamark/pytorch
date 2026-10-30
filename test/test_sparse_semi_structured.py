# Owner(s): ["module: sparse"]
# ruff: noqa: F841
# [MODULE DOC] Tests for semi-structured (2:4) sparse tensors: SparseSemiStructuredTensor and
# its CUTLASS and cuSPARSELt backends. Defines random 2:4 test-data generators, a
# torch.compile test class, inference tests (to_sparse_semi_structured, mm, linear,
# shape/dtype validation), training tests (pruning, tile/apply ops, sparse matmuls, SM8x
# only), CUTLASS-only op tests, and cuSPARSELt-only tests (fp8, alpha, search, padding, .to).
# At import time it probes CUDA/ROCm capability to fill the mapping of available backends,
# builds dtype and backend parametrization decorators, and instantiates the device-type test
# classes for cuda only when a matching backend exists. Running the file as a script calls
# run_tests.
import itertools
import random
import unittest

import pytest
import torch
import torch._dynamo.test_case
import torch.nn.functional as F
from torch import nn
from torch.sparse import (
    SparseSemiStructuredTensor,
    SparseSemiStructuredTensorCUSPARSELT,
    SparseSemiStructuredTensorCUTLASS,
    to_sparse_semi_structured,
)
from torch.sparse._semi_structured_conversions import (
    _compute_compressed_swizzled_bitmask,
    _sparse_semi_structured_tile,
    sparse_semi_structured_from_dense_cutlass,
)
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_FP8_SPARSE,
    xfailIfSM89PreCUDA13,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import all_types_and_complex
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_GPU

SEMI_STRUCTURED_SUPPORTED_BACKENDS = dict()

_IS_SM8X = False
_IS_SM9X = False
_IS_HIPSPARSELT_AVAILABLE = False
if torch.cuda.is_available():
    _IS_SM8X = torch.version.cuda is not None and (
        torch.cuda.get_device_capability(0)[0] == 8
    )
    _IS_SM9X = torch.version.cuda is not None and (
        torch.cuda.get_device_capability(0)[0] == 9
    )
    _IS_HIPSPARSELT_AVAILABLE = torch.version.hip is not None and tuple(
        int(v) for v in torch.version.hip.split(".")[:2]
    ) >= (7, 12)
    # CUTLASS kernels only work for Ampere
    if _IS_SM8X:
        SEMI_STRUCTURED_SUPPORTED_BACKENDS["cutlass"] = (
            SparseSemiStructuredTensorCUTLASS
        )

    # add cuSPASRELt tests if available
    if torch.backends.cusparselt.is_available() and (
        _IS_SM8X or _IS_SM9X or _IS_HIPSPARSELT_AVAILABLE
    ):
        SEMI_STRUCTURED_SUPPORTED_BACKENDS["cusparselt"] = (
            SparseSemiStructuredTensorCUSPARSELT
        )

inference_dtypes = dtypes(torch.float16, torch.bfloat16, torch.int8)
training_dtypes = dtypes(torch.float16, torch.bfloat16)
parametrize_backends = parametrize("backend", SEMI_STRUCTURED_SUPPORTED_BACKENDS)

atol_rtol_kw = {
    torch.float16: {
        "rtol": 1e-3,
        "atol": 1e-3,
    },
    torch.bfloat16: {
        "rtol": 1e-1,
        "atol": 1e-1,
    },
}


def sparse24_largest_mask_2d(original):
    # [FUNCTION DOC] HELPER: Prunes the 2D input to 2:4 sparsity using the CUTLASS static-sort
    # pruning, converts the result back to dense, and returns a boolean mask that is true
    # where values were kept. Requires CUDA and the CUTLASS backend.
    sparse = SparseSemiStructuredTensorCUTLASS.prune_dense_static_sort(original)
    return sparse.to_dense().bool()


def sparsify24_dense(original):
    # [FUNCTION DOC] HELPER: Returns the input with every element outside its 2:4 largest-
    # magnitude mask zeroed, by multiplying with the mask from sparse24_largest_mask_2d.
    return sparse24_largest_mask_2d(original) * original


def rand_sparse_semi_structured_mask(
    r, c, dtype=torch.float16, device="cuda", choice=None
):
    # [FUNCTION DOC] HELPER: Builds an r by c tensor of the given dtype and device whose
    # entries come in pairs that are each either zero-one or one-zero (1:2 sparsity, so also
    # valid 2:4). A fixed pair can be forced through the choice argument; otherwise each pair
    # is drawn with the global random module. Returns a contiguous tensor holding only zeros
    # and ones.
    """
    This function returns a 1:2 sparse matrix of size (r, c).
    Note that this means this matrix will also be 2:4 and 4:8 sparse as well.
    """

    choices = [[0, 1], [1, 0]]
    mask_entries = [choice or random.choice(choices) for i in range(r * c // 2)]

    return (
        torch.tensor(mask_entries, dtype=dtype, device=device)
        .reshape(r, c)
        .contiguous()
    )


def rand_sparse_semi_structured(r, c, dtype, device, choice=None):
    # [FUNCTION DOC] HELPER: Returns a random dense r by c tensor satisfying semi-structured
    # sparsity. Uses a 1:2 pattern for float32 and a 2:4 pattern for other dtypes, draws a
    # pattern per group (or uses the given choice), builds values with make_tensor, replaces
    # any natural zeros with one so zeros occur only under the mask, and zeroes masked
    # entries.
    pattern = "2by4" if dtype != torch.float32 else "1by2"
    if pattern == "1by2":
        ksparse = 2
        choices = [[0, 1], [1, 0]]
    elif pattern == "2by4":
        ksparse = 4
        choices = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
    mask_entries = [choice or random.choice(choices) for i in range(r * c // ksparse)]
    mask = torch.tensor(mask_entries, dtype=torch.bool).view(r, c).to(device)
    dense = make_tensor(r, c, dtype=dtype, device=device)
    dense[dense == 0] = 1  # To prevent zeros except where mask applied.
    dense = dense.masked_fill(~mask, 0)
    return dense


def rand_sparse_semi_structured_all_patterns(r, c, dtype, device):
    # [FUNCTION DOC] HELPER: Builds test data covering every valid 2:4 (or 1:2 for float32)
    # pattern. Each table row pairs an input group with the group that results after
    # compression and decompression. Randomly picks a row per group and returns two tensors:
    # the input with the first pattern applied, and the expected tensor after compression
    # with the second pattern applied on top. Natural zeros are replaced by one first.
    pattern = "2by4" if dtype != torch.float32 else "1by2"
    if pattern == "1by2":
        ksparse = 2
        choices = [
            [[0, 0], [0, 1]],
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
            [[1, 1], [1, 0]],
        ]
    elif pattern == "2by4":
        ksparse = 4
        choices = [
            [[0, 0, 0, 0], [0, 0, 1, 1]],
            [[0, 0, 0, 1], [0, 0, 1, 1]],
            [[0, 0, 1, 0], [0, 0, 1, 1]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            [[0, 1, 0, 0], [0, 1, 1, 0]],
            [[0, 1, 0, 1], [0, 1, 0, 1]],
            [[0, 1, 1, 0], [0, 1, 1, 0]],
            [[0, 1, 1, 1], [0, 1, 0, 1]],
            [[1, 0, 0, 0], [1, 0, 1, 0]],
            [[1, 0, 0, 1], [1, 0, 0, 1]],
            [[1, 0, 1, 0], [1, 0, 1, 0]],
            [[1, 0, 1, 1], [1, 0, 0, 1]],
            [[1, 1, 0, 0], [1, 1, 0, 0]],
            [[1, 1, 0, 1], [1, 1, 0, 0]],
            [[1, 1, 1, 0], [1, 1, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 0, 0]],
        ]
    mask_rows = [random.randint(0, len(choices) - 1) for i in range(r * c // ksparse)]

    COL_INV, COL_VAL = 0, 1
    mask_entries_inv = [choices[i][COL_INV] for i in mask_rows]
    mask_entries_val = [choices[i][COL_VAL] for i in mask_rows]
    mask_inv = torch.tensor(mask_entries_inv, dtype=torch.bool).view(r, c).to(device)
    mask_val = torch.tensor(mask_entries_val, dtype=torch.bool).view(r, c).to(device)
    dense = make_tensor(r, c, dtype=dtype, device=device)
    dense[dense == 0] = 1  # To prevent zeros except where mask below applied.
    dense_inv = dense.masked_fill(~mask_inv, 0)
    dense_val = dense_inv.masked_fill(~mask_val, 0)

    return dense_inv, dense_val


class SparseSemiStructuredTensorCompileTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] torch.compile tests for semi-structured sparse tensors. Derives from the
    # dynamo TestCase. Skips every test when no semi-structured backend is available. Tests
    # cover compiling an MLP with a sparse weight under inductor for each backend, compiling
    # pruning plus a matmul with gradients, and functionalization of the CUTLASS custom op.
    def setUp(self):
        # [FUNCTION DOC] Skips the test if no semi-structured backend was detected at import;
        # otherwise runs the base class setUp.
        if len(SEMI_STRUCTURED_SUPPORTED_BACKENDS) == 0:
            self.skipTest("semi-structured sparsity has no available backend!")
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] Runs the base class tearDown; adds no extra cleanup.
        super().tearDown()

    @staticmethod
    def _test_mlp_contiguous_relu_compile(backend, dense_input_shape):
        # [FUNCTION DOC] HELPER: Static method shared by the compile tests. Builds a small
        # model of a linear layer, contiguous call and relu on CUDA in half precision, masks
        # the weight with a repeating 2:4 pattern, records the dense output, swaps in a sparse
        # weight made with the chosen backend from_dense, records the sparse eager output,
        # then compiles with inductor (full graph) and compares to the dense output within
        # tolerance. Raises an error if eager and compiled outputs differ in strides.
        """
        Test nn.Linear + .contiguous() + nn.ReLU with SparseSemiStructuredTensor + torch.compile
        We expect:
            (1) The sparse tensor subclass should turn nn.Linear into `aten._structured_sparse_addmm` + `aten.contiguous()`
            (2) Inductor should fuse the .contiguous() call into the relu
        """

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Tiny module with one 128 by 128 linear layer followed by a
            # contiguous call and relu, used as the compile target.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the single linear layer of 128 input and 128
                # output features.
                super().__init__()
                self.linear = nn.Linear(128, 128)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer, makes the result
                # contiguous, and returns the relu of it.
                x = self.linear(x)
                x = x.contiguous()
                return torch.nn.functional.relu(x)

        input = torch.rand(dense_input_shape, device="cuda").half()
        model = Model().eval().cuda().half()
        mod_linear = model.linear
        m, n = mod_linear.weight.shape
        mask = torch.Tensor([1, 0, 0, 1]).tile((m, n // 4)).bool().cuda()
        # set masked weight
        mod_linear.weight = nn.Parameter(mod_linear.weight * mask)

        dense_result = model(input)
        mod_linear.weight = nn.Parameter(
            SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend].from_dense(mod_linear.weight)
        )
        sparse_result = model(input)

        model = torch.compile(model, backend="inductor", fullgraph=True)
        sparse_compile_result = model(input)

        # test that sparse_compile_result and dense_result are numerically close
        torch.testing.assert_close(
            dense_result, sparse_compile_result, rtol=1e-3, atol=1e-3
        )
        # assert sparse and sparse_compile have the same strides,
        # as meta registrations may return contiguous tensors when the output is transposed
        # https://github.com/pytorch/pytorch/pull/114477
        if sparse_result.stride() != sparse_compile_result.stride():
            raise AssertionError(
                f"stride mismatch: {sparse_result.stride()} != {sparse_compile_result.stride()}"
            )

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    @unittest.skipIf(
        "cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS,
        "cusparselt not supported on this machine",
    )
    @unittest.skipIf(
        torch.version.hip and not _IS_HIPSPARSELT_AVAILABLE,
        "HIPSPARSELt is not available for ROCm versions < 7.12"
    )
    def test_mlp_contiguous_relu_compile_cusparselt(self):
        # [FUNCTION DOC] Runs the shared MLP compile helper with the cusparselt backend for
        # four input shapes, including a three-dimensional batch. Skipped on Windows, when the
        # backend is unavailable, or on ROCm without hipSPARSELt support.
        """
        test for cuSPASRELt meta registrations (_cslt_sparse_mm) + torch.compile
        """
        for dense_input_shape in [(1, 128), (64, 128), (128, 128), (64, 128, 128)]:
            SparseSemiStructuredTensorCompileTest._test_mlp_contiguous_relu_compile(
                "cusparselt", dense_input_shape
            )

    @unittest.skipIf(
        "cutlass" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS,
        "cutlass not supported on this machine",
    )
    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_mlp_contiguous_relu_compile_cutlass(self):
        # [FUNCTION DOC] Runs the shared MLP compile helper with the cutlass backend for four
        # input shapes, including a three-dimensional batch. Skipped without the backend, on
        # Windows, or on ROCm.
        """
        test for CUTLASS meta registrations (_sparse_semi_structured_addmm) + torch.compile
        """
        for dense_input_shape in [(1, 128), (64, 128), (128, 128), (64, 128, 128)]:
            SparseSemiStructuredTensorCompileTest._test_mlp_contiguous_relu_compile(
                "cutlass", dense_input_shape
            )

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    @unittest.skipIf(
        "cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS,
        "cusparselt not supported on this machine",
    )
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
    def test_sp24_compile(self) -> None:
        # [FUNCTION DOC] Checks that pruning a requires-grad half tensor to cuSPARSELt 2:4
        # format, transposing it and multiplying with the original works both eagerly and
        # under torch.compile, including running backward on the output in both modes. Only
        # checks that nothing raises; no numerical comparison.
        x = torch.randn(
            [1024, 512], device="cuda", dtype=torch.float16, requires_grad=True
        )

        def fn(x):
            # [FUNCTION DOC] HELPER: Prunes the input with the cuSPARSELt static-sort pruning,
            # transposes the sparse result, and returns the dense input matrix-multiplied with
            # it.
            y = SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(x)
            y = y.t()
            return x @ y

        # Eager
        output = fn(x)
        output.backward(output)
        # Torch compile
        output = torch.compile(fn)(x)
        output.backward(output)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_cutlass_mm_functionalization_decomp(self):
        # [FUNCTION DOC] Under fake tensor mode and functional tensor mode, calls the
        # semi_structured cutlass_mm custom op on fake dense, packed and metadata inputs,
        # after registering the op, and checks the output shape is out_features by n. Verifies
        # the functionalization decomposition without running real CUTLASS kernels.
        """Test that semi_structured::cutlass_mm decomposes under FunctionalTensorMode.

        This verifies the register_torch_dispatch(FunctionalTensorMode) registration
        works correctly -- the custom op should decompose into aten._sparse_semi_structured_mm
        (or _addmm) + pad/narrow. Does not require actual CUTLASS execution (uses FakeTensorMode).
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        from torch.sparse.semi_structured import _ensure_cutlass_mm_registered

        _ensure_cutlass_mm_registered()

        out_features, k, n = 32, 64, 16
        with FakeTensorMode() as fake_mode:
            dense = torch.randn(k, n, device="cuda", dtype=torch.float16)
            packed = torch.randn(out_features, k // 2, device="cuda", dtype=torch.float16)
            meta = torch.randint(0, 10, (out_features, k // 16), device="cuda", dtype=torch.int16)

            with FunctionalTensorMode():
                dense_f = FunctionalTensor.to_functional(dense)
                packed_f = FunctionalTensor.to_functional(packed)
                meta_f = FunctionalTensor.to_functional(meta)

                result = torch.ops.semi_structured.cutlass_mm(
                    dense_f, packed_f, meta_f, None, out_features, 8, 8, False
                )

            self.assertEqual(result.shape, torch.Size([out_features, n]))


class TestSparseSemiStructured(TestCase):
    # [CLASS DOC] Inference tests for SparseSemiStructuredTensor, parametrized over dtype and
    # backend (and shapes). Setup skips the class when no backend is available or on Windows.
    # Tests cover conversion, mm with the sparse operand in each position/transposition,
    # linear and MLP numerics, values and indices accessors, and error cases for unsupported
    # shape, dtype or dimension.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp and skips the test if no backend is available or
        # the platform is Windows.
        super().setUp()
        if len(SEMI_STRUCTURED_SUPPORTED_BACKENDS) == 0:
            self.skipTest("semi-structured sparsity has no available backend!")
        if IS_WINDOWS:
            self.skipTest("torch.compile not supported on windows")

    @inference_dtypes
    @parametrize_backends
    def test_to_sparse_semi_structured(self, dtype, backend):
        # [FUNCTION DOC] Selects the backend via the class-level force-CUTLASS flag, converts
        # a 128 by 256 masked matrix with to_sparse_semi_structured, and asserts the result
        # keeps shape, device and dtype and is a SparseSemiStructuredTensor. Raises
        # AssertionError on mismatch.
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        A = rand_sparse_semi_structured_mask(128, 256, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        if A.shape != A_sparse.shape:
            raise AssertionError(f"shape mismatch: {A.shape} != {A_sparse.shape}")
        if A.device != A_sparse.device:
            raise AssertionError(f"device mismatch: {A.device} != {A_sparse.device}")
        if A.dtype != A_sparse.dtype:
            raise AssertionError(f"dtype mismatch: {A.dtype} != {A_sparse.dtype}")

        if not isinstance(A, torch.Tensor):
            raise AssertionError(f"A should be torch.Tensor, got {type(A)}")
        if not isinstance(A_sparse, SparseSemiStructuredTensor):
            raise AssertionError(
                f"A_sparse should be SparseSemiStructuredTensor, got {type(A_sparse)}"
            )

    @inference_dtypes
    @parametrize_backends
    @parametrize("dense_input_shape", [(128, 1), (128, 64), (128, 128)])
    def test_mm_sparse_first_NN(self, dense_input_shape, dtype, device, backend):
        # [FUNCTION DOC] Checks sparse-first mm (sparse times dense, no transposes). For int8
        # the expected outcome is an error: a RuntimeError about layout dispatch for CUTLASS,
        # the matching cuSPARSELt operation-not-supported error on CUDA, or on ROCm a result
        # matching a CPU reference. For other dtypes the sparse result must be close to the
        # dense mm result.
        """
        Ensure torch.mm(A_sparse, B) is correct for float16 and will throw error for int8
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        B = torch.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is torch.int8:
            if backend == "cutlass":
                with self.assertRaisesRegex(
                    RuntimeError, "spgemm_cutlass_dispatch_layouts"
                ):
                    sparse_result = torch.mm(A_sparse, B)
            else:
                if torch.version.hip:
                    dense_result = torch.mm(A.cpu(), B.cpu()).to(
                        device, dtype=torch.int8
                    )
                    sparse_result = torch.mm(A_sparse, B)
                    torch.testing.assert_close(
                        dense_result, sparse_result, rtol=1e-3, atol=1e-3
                    )
                else:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "CUDA error: operation not supported when calling `cusparseLtMatmulDescriptorInit",
                    ):
                        sparse_result = torch.mm(A_sparse, B)
        else:
            dense_result = torch.mm(A, B)
            sparse_result = torch.mm(A_sparse, B)
            torch.testing.assert_close(
                dense_result, sparse_result, rtol=1e-3, atol=1e-3
            )

    @inference_dtypes
    @parametrize_backends
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    def test_mm_sparse_first_NT(self, dense_input_shape, dtype, device, backend):
        # [FUNCTION DOC] Checks sparse-first mm with a transposed dense operand. For int8 with
        # the single-row shape it expects the same backend errors as the plain case (CUTLASS
        # layout error, cuSPARSELt unsupported on CUDA, CPU-matching result on ROCm); other
        # int8 shapes compare with a CPU reference; other dtypes compare with dense mm.
        # Tolerances are loose.
        """
        Ensure torch.mm(A_sparse, B.t()) is correct for float16/bfloat16
        and will throw an error for int8 + padding
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        B = torch.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is torch.int8 and dense_input_shape == (1, 128):
            # padding with int8 throws an error because transposing B yields a contiguous output
            # and row-row 2:4 sparse @ dense with NN is not supported by cuSPARSELt or CUTLASS.
            if backend == "cutlass":
                with self.assertRaisesRegex(
                    RuntimeError, "spgemm_cutlass_dispatch_layouts"
                ):
                    sparse_result = torch.mm(A_sparse, B.t())
            else:
                if torch.version.hip:
                    dense_result = torch.mm(A.cpu(), B.t().cpu()).to(
                        device, dtype=torch.int8
                    )
                    sparse_result = torch.mm(A_sparse, B.t())
                    torch.testing.assert_close(
                        dense_result, sparse_result, rtol=1e-3, atol=1e-3
                    )
                else:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "CUDA error: operation not supported when calling `cusparseLtMatmulDescriptorInit",
                    ):
                        sparse_result = torch.mm(A_sparse, B.t())
        elif dtype is torch.int8:
            # test transpose
            dense_result = torch.mm(A.cpu(), B.t().cpu()).to(device, dtype=torch.int8)
            sparse_result = torch.mm(A_sparse, B.t())
            torch.testing.assert_close(
                dense_result, sparse_result, rtol=1e-3, atol=1e-3
            )
        else:
            # test transpose
            dense_result = torch.mm(A, B.t())
            sparse_result = torch.mm(A_sparse, B.t())
            torch.testing.assert_close(
                dense_result, sparse_result, rtol=1e-3, atol=1e-3
            )

    @inference_dtypes
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize_backends
    def test_mm_sparse_first_TN(self, dtype, dense_input_shape, device, backend):
        # [FUNCTION DOC] Verifies that multiplying a transposed sparse matrix by a dense
        # matrix raises NotImplementedError saying the matmul operation is not supported.
        # Skips for CUTLASS on Windows.
        """
        Ensure torch.mm(A_sparse.t(), B) throws error
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(128, 256, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        B = torch.rand(dense_input_shape, device=A_sparse.device).to(dtype)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"`SparseSemiStructuredTensor.*` matmul: operation is not supported",
        ):
            torch.mm(A_sparse.t(), B)

    @inference_dtypes
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize_backends
    def test_mm_sparse_second_NT(self, dense_input_shape, dtype, device, backend):
        # [FUNCTION DOC] Checks mm of a dense matrix by the transposed sparse matrix. int8
        # reference is computed on CPU and copied to the device; other dtypes use device mm.
        # The sparse result must be close to the dense reference. Skips for CUTLASS on
        # Windows.
        """
        Ensure torch.mm(A, B_sparse.t()) is correct
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        B = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        B_sparse = to_sparse_semi_structured(B)

        A = torch.rand(dense_input_shape, device=B_sparse.device).to(dtype)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is torch.int8:
            dense_result = torch.mm(A.cpu(), B.t().cpu()).to(device, dtype=torch.int8)
            sparse_result = torch.mm(A, B_sparse.t())
        else:
            dense_result = torch.mm(A, B.t())
            sparse_result = torch.mm(A, B_sparse.t())

        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128)])
    @parametrize_backends
    def test_mm_sparse_second_NN(self, dense_input_shape, dtype, device, backend):
        # [FUNCTION DOC] Verifies that multiplying a dense matrix by an untransposed sparse
        # matrix raises NotImplementedError reporting the matmul operation as unsupported.
        # Skips for CUTLASS on Windows.
        """
        Ensure torch.mm(A, B_sparse) throws error
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        B = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        B_sparse = to_sparse_semi_structured(B)

        A = torch.rand(dense_input_shape, device=B_sparse.device).to(dtype)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"`SparseSemiStructuredTensor.*` matmul: operation is not supported",
        ):
            sparse_result = torch.mm(A, B_sparse)

    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128), (64, 128, 128)])
    @parametrize("inference_mode", [subtest(True), subtest(False)])
    @parametrize_backends
    def test_linear(self, dense_input_shape, inference_mode, device, backend):
        # [FUNCTION DOC] Compares a half-precision linear layer with a randomly 2:4-masked
        # weight against the same layer after the weight is replaced by its semi-structured
        # version, for several input shapes and with or without inference mode. Outputs must
        # be close. Skips for CUTLASS on Windows.
        """
        Test nn.Linear has the same numerics
        """
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        input = torch.rand((dense_input_shape), device=device).half()
        model = nn.Linear(128, 256).to(device).half()
        m, n = model.weight.shape
        mask = rand_sparse_semi_structured_mask(m, n, device=device, dtype=torch.bool)
        # set masked weight
        model.weight = nn.Parameter(model.weight * mask)

        dense_result = model(input)

        model.weight = nn.Parameter(to_sparse_semi_structured(model.weight))

        if inference_mode:
            with torch.inference_mode():
                sparse_result = model(input)
        else:
            sparse_result = model(input)

        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @parametrize("dense_input_shape", [(1, 128), (64, 128), (128, 128), (64, 128, 128)])
    @parametrize_backends
    def test_mlp(self, device, dense_input_shape, backend):
        # [FUNCTION DOC] Builds a two-layer half-precision MLP, masks both weights with random
        # 2:4 masks, records the dense output, converts both weights to semi-structured, and
        # checks the sparse model output is close to the dense one for several input shapes.
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        input = torch.rand(dense_input_shape, device=device).half()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 128),
            )
            .half()
            .to(device)
        )

        for i in range(2):
            m, n = model[i].weight.shape
            mask = rand_sparse_semi_structured_mask(
                m, n, device=device, dtype=torch.bool
            )
            # set masked weight
            model[i].weight = nn.Parameter(model[i].weight * mask)

        dense_result = model(input)

        for i in range(2):
            model[i].weight = nn.Parameter(to_sparse_semi_structured(model[i].weight))

        sparse_result = model(input)

        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @parametrize_backends
    def test_values(self, backend):
        # [FUNCTION DOC] Converts a fully ones-valued 128 by 128 masked matrix and checks the
        # values accessor returns a 128 by 64 tensor that is all ones. Raises AssertionError
        # otherwise. Skips for CUTLASS on Windows.
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(128, 128)
        A_sparse = to_sparse_semi_structured(A)
        if A_sparse.values().shape != (128, 64):
            raise AssertionError(
                f"values shape should be (128, 64), got {A_sparse.values().shape}"
            )
        if not (A_sparse.values() == 1).all():
            raise AssertionError("values should all be 1")

    @parametrize_backends
    def test_indices(self, backend):
        # [FUNCTION DOC] Converts a 128 by 128 masked matrix and checks the indices accessor
        # returns a tensor of shape 128 by 8. Raises AssertionError otherwise. Skips for
        # CUTLASS on Windows.
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(128, 128)
        A_sparse = to_sparse_semi_structured(A)
        if A_sparse.indices().shape != (128, 8):
            raise AssertionError(
                f"indices shape should be (128, 8), got {A_sparse.indices().shape}"
            )

    @inference_dtypes
    @parametrize_backends
    def test_min_sparse_shape(self, dtype, device, backend):
        # [FUNCTION DOC] Uses the backend minimum sparse rows/cols and dense columns from its
        # dtype-shape constraints table to build the smallest valid sparse matrix and dense
        # operand, and checks sparse mm matches dense mm. For int8 the dense reference is
        # computed on CPU and the operand is passed as a transposed contiguous copy to get a
        # supported layout.
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        config = SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend]._DTYPE_SHAPE_CONSTRAINTS[
            dtype
        ]
        A = rand_sparse_semi_structured_mask(
            config.sparse_min_rows, config.sparse_min_cols, dtype=dtype, device=device
        )
        A_sparse = to_sparse_semi_structured(A)
        B = torch.rand(
            (config.sparse_min_cols, config.dense_min_cols), device=device
        ).to(dtype)
        if dtype == torch.int8:
            dense_res = torch.mm(A.cpu(), B.cpu()).to(device, dtype=torch.int8)
            # int8 sparse matmul not supported for R/R -> R layout, so we transpose one of the arguments to get R/C -> R
            B_t = B.t().contiguous()
            sparse_res = torch.mm(A_sparse, B_t.t())
        else:
            dense_res = torch.mm(A, B)
            sparse_res = torch.mm(A_sparse, B)
        torch.testing.assert_close(sparse_res, dense_res, rtol=1e-3, atol=1e-3)

    @inference_dtypes
    @parametrize_backends
    def test_unsupported_shape(self, dtype, device, backend):
        # [FUNCTION DOC] Checks that converting a 2 by 2 matrix, too small for the backend,
        # raises RuntimeError mentioning the original tensor shape. Skips for CUTLASS on
        # Windows.
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(2, 2, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.shape"):
            A_sparse = to_sparse_semi_structured(A)

    @dtypes(*all_types_and_complex())
    @parametrize_backends
    def test_unsupported_dtype(self, dtype, device, backend):
        # [FUNCTION DOC] For every dtype, builds a masked matrix and converts it. If the dtype
        # is absent from the backend constraints table a RuntimeError mentioning the dtype is
        # expected; otherwise conversion must succeed. Skips for CUTLASS on Windows.
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype, device=device)

        if (
            dtype
            not in SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend]._DTYPE_SHAPE_CONSTRAINTS
        ):
            with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dtype"):
                A_sparse = to_sparse_semi_structured(A)
        else:
            A_sparse = to_sparse_semi_structured(A)

    @parametrize_backends
    def test_unsupported_dim(self, device, backend):
        # [FUNCTION DOC] Checks that converting a three-dimensional half tensor raises
        # RuntimeError mentioning the tensor dimension. Skips for CUTLASS on Windows.
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")
        A = torch.rand(128, 128, 128, device=device, dtype=torch.float16)

        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dim"):
            A_sparse = to_sparse_semi_structured(A)


def create_random_mask(shape) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Returns a boolean mask of the given 2D shape with exactly two
    # true entries in every group of four consecutive columns of each row. Uses a locally
    # seeded random generator (fixed seed), so the mask is deterministic. The column count is
    # expected to be a multiple of four.
    r = random.Random(0)
    mask = torch.zeros(shape, dtype=torch.bool)
    for line in range(mask.shape[0]):
        for col in range(0, mask.shape[1], 4):
            sparsity = r.choice(
                [
                    [False, False, True, True],
                    [False, True, False, True],
                    [True, False, False, True],
                    [False, True, True, False],
                    [True, False, True, False],
                    [True, True, False, False],
                ]
            )
            mask[line, col : col + 4] = torch.tensor(sparsity, dtype=torch.bool)
    return mask


class TestSparseSemiStructuredTraining(TestCase):
    # [CLASS DOC] Training-oriented tests for 2:4 sparse tensors (pruning, tile packing,
    # apply, and sparse matmuls with CUTLASS and cuSPARSELt). Setup skips unless the device is
    # SM8x (Ampere) and the platform is not Windows. Most tests are skipped on ROCm and in
    # RelWithAssert builds.
    def setUp(self):
        # [FUNCTION DOC] Calls base setUp, then skips the test unless running on an SM8x GPU
        # and not on Windows.
        super().setUp()
        if not _IS_SM8X:
            self.skipTest(
                "SparseSemiStructuredTensor training only supported on SM8x (Ampere)"
            )

        if IS_WINDOWS:
            self.skipTest("CUTLASS not supported on windows")

    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        not torch.backends.cusparselt.is_available(),
        "cuSPARSELt not available",
    )
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
    def test_prune_dense_static_sort(self, dtype) -> None:
        # [FUNCTION DOC] Prunes a random matrix with the tile op to get a 2:4 pattern, then
        # checks both backends: prune_dense_static_sort with the greedy algorithm must
        # reproduce the pruned matrix when densified, and a sparse tensor manually assembled
        # from packed data, metadata and bitmask (CUTLASS) or compressed buffers (cuSPARSELt)
        # must densify to the same result as the reference.
        # Ideally we would like to clone and compare, but that won't work because the sorting order will be different
        # instead we pass the pruned matrix to the CUDA implementation and preserve the sparsity pattern.
        dense = torch.randn(128, 128, device="cuda", dtype=dtype)
        pruned = _sparse_semi_structured_tile(dense)

        # CUTLASS
        reference_cutlass = SparseSemiStructuredTensorCUTLASS.prune_dense_static_sort(
            pruned, algorithm="largest_abs_values_greedy"
        )
        torch.testing.assert_close(pruned, reference_cutlass.to_dense())

        packed_cutlass, meta_cutlass = sparse_semi_structured_from_dense_cutlass(pruned)
        packed_t_cutlass, meta_t_cutlass = sparse_semi_structured_from_dense_cutlass(
            pruned.t().contiguous()
        )
        meta_cutlass = meta_cutlass.as_strided(
            reference_cutlass.meta.shape, reference_cutlass.meta.stride()
        )
        meta_t_cutlass = meta_t_cutlass.as_strided(
            reference_cutlass.meta_t.shape, reference_cutlass.meta_t.stride()
        )
        compressed_swizzled_bitmask = _compute_compressed_swizzled_bitmask(pruned)
        compressed_swizzled_bitmask = compressed_swizzled_bitmask.as_strided(
            reference_cutlass.compressed_swizzled_bitmask.shape,
            reference_cutlass.compressed_swizzled_bitmask.stride(),
        )
        cutlass = SparseSemiStructuredTensorCUTLASS(
            dense.shape,
            packed_cutlass,
            meta_cutlass,
            packed_t_cutlass,
            meta_t_cutlass,
            compressed_swizzled_bitmask,
        )
        torch.testing.assert_close(reference_cutlass.to_dense(), cutlass.to_dense())

        # CUSPARSELT
        reference_cusparselt = (
            SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(
                pruned, algorithm="largest_abs_values_greedy"
            )
        )
        torch.testing.assert_close(pruned, reference_cusparselt.to_dense())

        packed_cusparselt = torch._cslt_compress(pruned)
        packed_t_cusparselt = torch._cslt_compress(pruned.t().contiguous())
        cusparselt = SparseSemiStructuredTensorCUSPARSELT(
            dense.shape,
            packed_cusparselt,
            None,
            packed_t_cusparselt,
            None,
            compressed_swizzled_bitmask,
        )
        torch.testing.assert_close(
            reference_cusparselt.to_dense(), cusparselt.to_dense()
        )

    @training_dtypes
    @parametrize_backends
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
    def test_pruning_algo_largest_abs_values_greedy(self, dtype, backend) -> None:
        # [FUNCTION DOC] Prunes a padded matrix whose top-left 4 by 4 block has known values
        # with the greedy largest-absolute-value algorithm of the backend, derives the kept-
        # element mask by dividing the dense result by the input, and checks the top-left
        # block equals the expected pattern. Raises AssertionError otherwise.
        inp = torch.tensor(
            [[4, 3, 2, 1], [-1, -3, 0.6, 0.5], [1, 2, 3, 4], [10, 2, -1, 5]],
            device="cuda",
            dtype=dtype,
        )
        inp = F.pad(inp, (0, 128 - 4, 0, 128 - 4), "constant", 1)
        sInp = SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend].prune_dense_static_sort(
            inp, algorithm="largest_abs_values_greedy"
        )

        mask = sInp.to_dense() / inp
        expected = [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ]
        if mask[:4, :4].int().tolist() != expected:
            raise AssertionError(
                f"mask[:4, :4] mismatch: {mask[:4, :4].int().tolist()} != {expected}"
            )

    @training_dtypes
    @unittest.skipIf(
        not torch.backends.cusparselt.is_available(),
        "cuSPARSELt not available",
    )
    def test_gemm(self, dtype) -> None:
        # [FUNCTION DOC] Converts a randomly 2:4-masked 32 by 64 matrix to semi-structured and
        # checks the sparse matmul with a dense matrix matches the masked dense product within
        # dtype-specific tolerances.
        M, N, K = 32, 32, 64
        a = torch.randn([M, K], device="cuda", dtype=dtype)
        b = torch.randn([K, N], device="cuda", dtype=dtype)
        mask = rand_sparse_semi_structured_mask(M, K, dtype=torch.bool)

        a.masked_fill_(~mask, 0)

        a_sparse = to_sparse_semi_structured(a)

        masked_a = a * mask
        ref_out = masked_a @ b
        sp24_out = a_sparse @ b
        torch.testing.assert_close(ref_out, sp24_out, **atol_rtol_kw[dtype])

    @training_dtypes
    @parametrize_backends
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
    def test_pack_both_ways_meta_correctness(self, dtype, backend) -> None:
        # [FUNCTION DOC] Builds a matrix with exactly eight kept elements per 4 by 4 tile,
        # prunes it with the backend, and for CUTLASS also checks the result type and that its
        # metadata equals the metadata from the tile op on the mask. Finally compares the
        # sparse matmul with the masked dense matmul.
        M, N = 128, 256
        # Construct x to make sure we always have exactly 8 elements per 4x4 tile
        a = (4 * torch.arange(8))[:, None] + torch.arange(8)[None, :]
        a = a.repeat(M // 8, N // 8)
        if a.shape != (M, N):
            raise AssertionError(f"a.shape should be ({M}, {N}), got {a.shape}")
        a = a.cuda().to(dtype)
        b = torch.randn([a.shape[1], 128], device="cuda", dtype=dtype)

        a_sparse = SEMI_STRUCTURED_SUPPORTED_BACKENDS[backend].prune_dense_static_sort(
            a
        )

        mask_dense = sparse24_largest_mask_2d(a).to(dtype)

        if backend == "cutlass":
            if not isinstance(a_sparse, SparseSemiStructuredTensorCUTLASS):
                raise AssertionError(
                    f"a_sparse should be SparseSemiStructuredTensorCUTLASS, got {type(a_sparse)}"
                )
            (packed, meta, packed_t, meta_t, bitmask) = (
                torch._sparse_semi_structured_tile(mask_dense, use_cutlass=True)
            )

            sparse_mask = SparseSemiStructuredTensorCUTLASS(
                mask_dense.shape,
                packed=packed,
                meta=meta,
                packed_t=packed_t,
                meta_t=meta_t,
                compressed_swizzled_bitmask=bitmask,
            )
            torch.testing.assert_close(
                a_sparse.meta.view(torch.short), sparse_mask.meta
            )

        ref_gemm = (mask_dense * a) @ b
        pack_gemm = a_sparse @ b
        torch.testing.assert_close(ref_gemm, pack_gemm, **atol_rtol_kw[dtype])

    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
    def test_pack_both_ways_id(self, dtype) -> None:
        # [FUNCTION DOC] Tiles a random square matrix into packed and transposed-packed forms,
        # checks the sums of packed values agree, then multiplies an identity by each packed
        # form through the sparse linear op and compares both results with the masked dense
        # matrix. The failure message reports the position of the largest difference.
        N = 512
        torch.manual_seed(0)
        a = torch.randn([N, N], dtype=dtype, device="cuda")
        b = torch.eye(N, dtype=dtype, device="cuda")

        packed, meta, packed_t, meta_t = torch._sparse_semi_structured_tile(a)[:4]
        # Heuristic to ensure we pack the same values
        torch.testing.assert_close(
            packed.to(torch.float64).sum(), packed_t.to(torch.float64).sum()
        )

        mask_dense = sparse24_largest_mask_2d(a.to(dtype))

        ref_gemm = mask_dense * a
        # Test A@B
        pack_gemm = torch._sparse_semi_structured_linear(b.t(), packed, meta).t()
        max_diff = (ref_gemm - pack_gemm).abs().argmax()
        torch.testing.assert_close(
            ref_gemm,
            pack_gemm,
            **atol_rtol_kw[dtype],
            msg=f"packed is wrong at pos: ({max_diff // N}, {max_diff % N})",
        )
        # Test A.t@B
        pack_gemm = torch._sparse_semi_structured_linear(b.t(), packed_t, meta_t)
        max_diff = (ref_gemm - pack_gemm).abs().argmax()

        torch.testing.assert_close(
            ref_gemm,
            pack_gemm,
            **atol_rtol_kw[dtype],
            msg=f"packed_t is wrong at pos: ({max_diff // N}, {max_diff % N})",
        )

    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
    def test_pack_both_ways_edge_case1(self, dtype) -> None:
        # [FUNCTION DOC] Edge case where the pruning heuristic keeps seven values in a 4 by 4
        # tile rather than eight. Checks the first row of the packed output and first row of
        # the transposed packed output hold the value two followed by zero padding. Raises
        # AssertionError otherwise.
        # In this case, the heuristic will keep 7 values out of 16
        # instead of 8. let's see how the kernel handles this
        quad = torch.tensor(
            [
                [2, -1, -2, -3],  # Should be packed as `2 <null>`
                [-1, 8, -1, 6],
                [-1, -1, 4, 5],
                [-1, 3, 7, -1],
            ],
            dtype=dtype,
            device="cuda",
        )
        a = torch.randn([32, 64], dtype=dtype, device="cuda")
        a[:4, :4] = quad
        packed, meta, packed_t, meta_t = torch._sparse_semi_structured_tile(a)[:4]
        # Check first line in A
        if packed[0, 0].item() != 2:
            raise AssertionError(f"packed[0, 0] should be 2, got {packed[0, 0].item()}")
        if packed[0, 1].item() != 0:
            raise AssertionError(f"packed[0, 1] should be 0, got {packed[0, 1].item()}")
        # And first column in A.t
        if packed_t[0, 0].item() != 2:
            raise AssertionError(
                f"packed_t[0, 0] should be 2, got {packed_t[0, 0].item()}"
            )
        if packed_t[0, 1].item() != 0:
            raise AssertionError(
                f"packed_t[0, 1] should be 0, got {packed_t[0, 1].item()}"
            )

    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
    def test_sp24_apply(self, dtype) -> None:
        # [FUNCTION DOC] Checks that applying a bitmask produced by the tile op to the same
        # input via the apply op reproduces the packed and transposed packed values from the
        # tile op.
        M, N = 256, 1024
        x = torch.randn([M, N], dtype=dtype, device="cuda")
        (
            packed,
            meta,
            packed_t,
            meta_t,
            bitmask,
        ) = torch._sparse_semi_structured_tile(x)
        packed2, packed_t2 = torch._sparse_semi_structured_apply(x, bitmask)
        torch.testing.assert_close(packed, packed2)
        torch.testing.assert_close(packed_t, packed_t2)

    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
    def test_sp24_apply_dense(self, dtype) -> None:
        # [FUNCTION DOC] Checks the apply and apply-dense ops: builds a reference CUTLASS
        # sparse tensor from tile op output and densifies it, then rebuilds a sparse tensor
        # from the apply op output and compares both it and the apply-dense output with the
        # reference.
        M, N = 256, 1024
        x = torch.randn([M, N], dtype=dtype, device="cuda")
        (
            packed,
            meta,
            packed_t,
            meta_t,
            bitmask,
        ) = torch._sparse_semi_structured_tile(x)

        expected = SparseSemiStructuredTensorCUTLASS(
            x.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        ).to_dense()

        packed2, packed_t2 = torch._sparse_semi_structured_apply(x, bitmask)
        sparse = SparseSemiStructuredTensorCUTLASS(
            x.shape,
            packed=packed2,
            meta=meta,
            packed_t=packed_t2,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        )

        dense = torch._sparse_semi_structured_apply_dense(x, bitmask)

        torch.testing.assert_close(dense, expected)
        torch.testing.assert_close(sparse.to_dense(), expected)

    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
    def test_sp24_matmuls(self, dtype) -> None:
        # [FUNCTION DOC] Wraps the tile op output for two random matrices as CUTLASS sparse
        # tensors and checks four matmul forms (sparse times dense, dense times sparse, dense
        # times transposed sparse, transposed sparse times dense) against dense products with
        # the 2:4 mask applied. Uses loose tolerances.
        M, N, K = 64, 256, 1024
        a = torch.randn([M, K], device="cuda", dtype=dtype)
        b = torch.randn([K, N], device="cuda", dtype=dtype)
        a_m = sparse24_largest_mask_2d(a)
        b_m = sparse24_largest_mask_2d(b)
        (packed, meta, packed_t, meta_t, bitmask) = torch._sparse_semi_structured_tile(
            a
        )
        a_s = SparseSemiStructuredTensorCUTLASS(
            a.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        )
        (packed, meta, packed_t, meta_t, bitmask) = torch._sparse_semi_structured_tile(
            b
        )
        b_s = SparseSemiStructuredTensorCUTLASS(
            b.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        )

        torch.testing.assert_close(a_s @ b, (a * a_m) @ b, rtol=1e-1, atol=1.5e-1)
        torch.testing.assert_close(a @ b_s, a @ (b * b_m), rtol=1e-1, atol=1.5e-1)
        torch.testing.assert_close(
            a @ a_s.t(), a @ (a * a_m).t(), rtol=1e-1, atol=1.5e-1
        )
        torch.testing.assert_close(a_s.t() @ a, (a * a_m).t() @ a, rtol=1e-1, atol=1e-1)

    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        not torch.backends.cusparselt.is_available(),
        "cuSPARSELt not available",
    )
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
    def test_sp24_matmuls_mat_vec(self) -> None:
        # [FUNCTION DOC] Checks that multiplying a semi-structured matrix by a one-dimensional
        # vector raises NotImplementedError. Requires cuSPARSELt.
        a = torch.randn([64, 128], device="cuda", dtype=torch.float16)
        b = torch.randn([128], device="cuda", dtype=torch.float16)
        a_m = sparse24_largest_mask_2d(a)
        a_s = to_sparse_semi_structured(a)

        with pytest.raises(NotImplementedError):
            torch.testing.assert_close(a_s @ b, (a * a_m) @ b, **atol_rtol_kw[a.dtype])

    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        not torch.backends.cusparselt.is_available(),
        "cuSPARSELt not available",
    )
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
    def test_sp24_matmuls_bmm(self) -> None:
        # [FUNCTION DOC] Checks that multiplying a semi-structured matrix by a three-
        # dimensional batched tensor raises NotImplementedError. Requires cuSPARSELt.
        a = torch.randn([64, 128], device="cuda", dtype=torch.float16)
        b = torch.randn([5, 6, 128], device="cuda", dtype=torch.float16)
        a_m = sparse24_largest_mask_2d(a)
        a_s = to_sparse_semi_structured(a)

        with pytest.raises(NotImplementedError):
            torch.testing.assert_close(a_s @ b, (a * a_m) @ b, **atol_rtol_kw[a.dtype])


class TestSparseSemiStructuredCUTLASS(TestCase):
    # [CLASS DOC] CUTLASS-specific tests (sparse linear, mm and addmm ops, and conversion
    # kernels). Setup forces the CUTLASS backend through the class-level flag and skips when
    # CUTLASS is not enabled; tearDown resets the flag.
    """
    This contains CUTLASS specific tests for
         - torch._sparse_semi_structured_linear
    """

    def setUp(self):
        # [FUNCTION DOC] Calls base setUp, forces the CUTLASS backend, and skips when CUTLASS
        # is unavailable.
        super().setUp()
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
        if "cutlass" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
            self.skipTest("CUTLASS not enabled")

    def tearDown(self):
        # [FUNCTION DOC] Resets the force-CUTLASS flag to false and calls base tearDown.
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
        super().tearDown()

    @unittest.skipIf(
        TEST_WITH_ROCM or IS_WINDOWS, "ROCm and Windows doesn't support CUTLASS"
    )
    @inference_dtypes
    def test_linear_cutlass(self, device, dtype):
        # [FUNCTION DOC] Sweeps batch shapes, power-of-two sizes, optional bias and
        # activations (none, relu, silu) through the CUTLASS sparse linear op and compares to
        # a float32 dense linear. Skips silu for int8. Tolerances vary by dtype. Skipped on
        # ROCm and Windows.
        # KNOWN ISSUE: The float32 branch restores fp32_precision but sets the allow_tf32
        #   flag; it is never exercised since float32 is not among the inference dtypes.
        def run_test(
            batch_shape,
            m,
            n,
            k,
            device,
            dtype,
            dtype_out,
            add_bias,
            activation,
            rtol,
            atol,
        ):
            # [FUNCTION DOC] HELPER: Runs one configuration: builds a sparse weight and random
            # input, optional bias, computes the float32 dense reference with the activation,
            # compresses the weight to values and metadata, calls the sparse linear op (with
            # output dtype for int8), and asserts closeness within the given tolerances.
            weight = rand_sparse_semi_structured(m, k, dtype, device)
            input = make_tensor((*batch_shape, n, k), dtype=dtype, device=device)
            bias = (
                make_tensor((m,), dtype=dtype_out, device=device) if add_bias else None
            )

            dtype_dense = torch.float32
            input_dense = input.to(dtype_dense)
            weight_dense = weight.to(dtype_dense)
            bias_dense = bias.to(dtype_dense) if add_bias else None
            output0 = torch.nn.functional.linear(
                input_dense, weight_dense, bias=bias_dense
            )
            if activation == "relu":
                relu = torch.nn.ReLU()
                output0 = relu(output0)
            elif activation == "silu":
                silu = torch.nn.SiLU()
                output0 = silu(output0)

            compressed = to_sparse_semi_structured(weight)

            weight_sparse = compressed.values()
            meta = compressed.indices()

            output1 = torch._sparse_semi_structured_linear(
                input,
                weight_sparse,
                meta,
                bias=bias,
                activation=activation,
                out_dtype=dtype_out if dtype == torch.int8 else None,
            )
            torch.testing.assert_close(
                output1.to(dtype_dense), output0, rtol=rtol, atol=atol
            )

        if dtype == torch.float32:
            # Inputs are converted to TF32 internally for sparse GEMM,
            # so make dense GEMM to do the same for matching results.
            orig = torch.backends.cuda.matmul.fp32_precision
            torch.backends.cuda.matmul.allow_tf32 = True

        batch_shapes = [[], [3], [3, 1]]
        dtype_out = {
            torch.int8: torch.int32,
            torch.half: torch.half,
            torch.bfloat16: torch.bfloat16,
            torch.float32: torch.float32,
        }
        activations = [None, "relu", "silu"]
        rtol, atol = 1e-3, 1e-3
        if dtype == torch.bfloat16:
            rtol, atol = 5e-3, 5e-3
        elif dtype == torch.float32:
            rtol, atol = 1e-3, 75e-2
        for batch_shape, m, n, k, add_bias, activation in itertools.product(
            batch_shapes, range(3), range(3), range(3), (False, True), activations
        ):
            if activation == "silu" and dtype == torch.int8:
                continue  # SiLU not supported for integer inputs

            m = 2**m * 32
            n = 2**n * 32
            k = 2**k * 128
            run_test(
                batch_shape,
                m,
                n,
                k,
                device,
                dtype,
                dtype_out[dtype],
                add_bias,
                activation,
                rtol,
                atol,
            )

        if dtype == torch.float32:
            torch.backends.cuda.matmul.fp32_precision = orig

    @unittest.skipIf(
        TEST_WITH_ROCM or IS_WINDOWS, "ROCm and Windows doesn't support CUTLASS"
    )
    @parametrize("backend", ["cutlass"])
    @inference_dtypes
    def test_sparse_semi_structured_ops_cutlass(self, device, dtype, backend):
        # [FUNCTION DOC] Sweeps sizes and presence of an input vector through the CUTLASS
        # sparse mm and addmm ops, comparing with float32 dense mm or addmm using alpha and
        # beta. Also sets the force flag and skips on Windows. Skipped on ROCm and Windows.
        # KNOWN ISSUE: The float32 branch restores fp32_precision but sets the allow_tf32
        #   flag; it is never exercised since float32 is not among the inference dtypes.
        SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
        if backend == "cutlass" and IS_WINDOWS:
            self.skipTest("CUTLASS not supported on Windows")

        def run_test(m, n, k, device, dtype, dtype_out, use_input, rtol, atol):
            # [FUNCTION DOC] HELPER: Runs one size configuration: builds a sparse first
            # operand and a transposed dense second operand, optional bias input with alpha
            # and beta (floating or integer values), computes the dense float32 reference,
            # compresses the first operand, calls the sparse mm or addmm op with the given
            # output dtype, and asserts closeness.
            mat1 = rand_sparse_semi_structured(m, k, dtype, device)
            # mat2 transposed as int8 case supports only row-major/column-major combination
            mat2 = make_tensor((n, k), dtype=dtype, device=device).t()
            input = (
                make_tensor((m,), dtype=dtype_out, device=device) if use_input else None
            )

            if use_input:
                if dtype.is_floating_point:
                    alpha = 1.3
                    beta = -0.7
                else:
                    alpha = 2
                    beta = -3

            dtype_dense = torch.float32
            mat1_dense = mat1.to(dtype_dense)
            mat2_dense = mat2.to(dtype_dense)
            if not use_input:
                output0 = torch.mm(mat1_dense, mat2_dense)
            else:
                input_dense = input.to(dtype_dense)[:, None]
                output0 = torch.addmm(
                    input_dense, mat1_dense, mat2_dense, alpha=alpha, beta=beta
                )

            compressed = to_sparse_semi_structured(mat1)

            mat1_sparse = compressed.values()
            mat1_meta = compressed.indices()

            if not use_input:
                output1 = torch._sparse_semi_structured_mm(
                    mat1_sparse, mat1_meta, mat2, out_dtype=dtype_out
                )
            else:
                output1 = torch._sparse_semi_structured_addmm(
                    input,
                    mat1_sparse,
                    mat1_meta,
                    mat2,
                    alpha=alpha,
                    beta=beta,
                    out_dtype=dtype_out,
                )
            torch.testing.assert_close(
                output1.to(dtype_dense), output0, rtol=rtol, atol=atol
            )

        if dtype == torch.float32:
            # Inputs are converted to TF32 internally for sparse GEMM,
            # so make dense GEMM to do the same for matching results.
            orig = torch.backends.cuda.matmul.fp32_precision
            torch.backends.cuda.matmul.allow_tf32 = True

        dtype_out = {
            torch.int8: torch.int32,
            torch.half: torch.half,
            torch.bfloat16: torch.bfloat16,
            torch.float32: torch.float32,
        }
        rtol, atol = 1e-3, 1e-3
        if dtype == torch.bfloat16:
            rtol, atol = 5e-3, 5e-3
        elif dtype == torch.float32:
            rtol, atol = 1e-3, 75e-2
        for m, n, k, use_input in itertools.product(
            range(3), range(3), range(3), (False, True)
        ):
            m = 2**m * 32
            n = 2**n * 32
            k = 2**k * 128
            run_test(m, n, k, device, dtype, dtype_out[dtype], use_input, rtol, atol)

        if dtype == torch.float32:
            torch.backends.cuda.matmul.fp32_precision = orig

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @inference_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_conversions(self, device, dtype):
        # [FUNCTION DOC] For several matrix shapes, checks that the compressed tensor's
        # metadata equals the metadata from the ATen conversion operator exactly and that
        # densifying returns the original matrix exactly. Needs a GPU with triton; skipped on
        # ROCm.
        def run_test(r, c, device, dtype):
            # [FUNCTION DOC] HELPER: Converts one random sparse matrix and checks exact
            # equality of the metadata with the ATen operator output and of the densified
            # result with the original.
            dense_ref = rand_sparse_semi_structured(r, c, dtype, device)

            compressed = to_sparse_semi_structured(dense_ref)

            # The torch.ops.aten._to_sparse_semi_structured operator
            # uses CUTLASS to perform conversion from given dense
            # matrix to the pair of corresponding sparse and metadata
            # matrices, with the later used here as a reference to
            # compare the metadata matrix produced by conversion
            # performed by SparseSemiStructuredTensor class
            # constructor against.
            _, meta_ref = torch.ops.aten._to_sparse_semi_structured(dense_ref)

            meta = compressed.indices()
            torch.testing.assert_close(meta, meta_ref, rtol=0, atol=0)

            dense = compressed.to_dense()
            torch.testing.assert_close(dense, dense_ref, rtol=0, atol=0)

        shapes = [[32, 128], [32, 256], [64, 128], [64, 256]]
        for r, c in shapes:
            run_test(r, c, device, dtype)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @inference_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_conversions_all_patterns(self, device, dtype):
        # [FUNCTION DOC] Builds data covering all valid 2:4 patterns, converts the input to
        # semi-structured, densifies, and checks it exactly equals the expected result. Needs
        # a GPU with triton; skipped on ROCm.
        r, c = 32, 128

        dense_inv, dense_val = rand_sparse_semi_structured_all_patterns(
            r, c, dtype, device
        )

        compressed = to_sparse_semi_structured(dense_inv)
        dense = compressed.to_dense()

        torch.testing.assert_close(dense, dense_val, rtol=0, atol=0)


CUSPARSELT_MIXED_DTYPE_SUPPORT = [torch.float16, torch.bfloat16, torch.int32]


def to_float8(x, dtype=torch.float8_e4m3fn):
    # [FUNCTION DOC] HELPER: Quantizes a tensor to float8. Computes a scale as the dtype max
    # divided by the clamped absolute maximum, scales and clamps to the dtype range, and
    # returns the float8 tensor with the reciprocal scale as float32, as needed by the scaled
    # mm op.
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()


class TestSparseSemiStructuredCUSPARSELT(TestCase):
    # [CLASS DOC] cuSPARSELt-specific tests for the compress and sparse mm ops, fp8 and scaled
    # mm, alpha scaling, algorithm search, padding of outputs, tensor .to behaviour, and
    # cloning. Setup disables CUTLASS forcing and skips when the cusparselt backend is not
    # available.
    """
    This contains cuSPARSELt specific tests for
        torch._cslt_compress
        torch._cslt_sparse_mm
    """

    def setUp(self):
        # [FUNCTION DOC] Calls base setUp, clears the force-CUTLASS flag, and skips when
        # cuSPARSELt is not enabled.
        super().setUp()
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
        if "cusparselt" not in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
            self.skipTest("cuSPARSELt not enabled")

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @xfailIfSM89PreCUDA13
    @parametrize("dense_input_shape", [(256, 128)])
    def test_sparse_fp8fp8_mm(self, dense_input_shape, device):
        # [FUNCTION DOC] Skips on old cuSPARSELt without fp8. Converts a masked matrix and a
        # dense operand to float8 and checks that mm of a semi-structured fp8 tensor raises
        # NotImplementedError pointing at the scaled mm op.
        if torch.backends.cusparselt.version() < 602 and not torch.version.hip:
            self.skipTest("fp8 matmul requires cuSPARSELt v0.6.2+")
        A = rand_sparse_semi_structured_mask(256, 128, dtype=torch.float16)
        B = torch.rand(dense_input_shape, device=device).to(torch.float16).t()

        A_fp8, A_scale = to_float8(A)
        B_fp8, B_scale = to_float8(B)
        A_fp8_sparse = to_sparse_semi_structured(A_fp8)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"`SparseSemiStructuredTensor.*_scaled_mm",
        ):
            dense_result = torch.mm(A_fp8_sparse, B_fp8)

    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @xfailIfSM89PreCUDA13
    def test_sparse_semi_structured_scaled_mm_fp8(self, device) -> None:
        # [FUNCTION DOC] Compares the scaled mm op on a float8 masked matrix with the semi-
        # structured version of the same matrix, with unit scales and float8 output, after
        # casting results to float32. Tolerance is loose.
        (k, l, m) = (32, 64, 32)
        x = rand_sparse_semi_structured_mask(
            k, l, dtype=torch.float8_e4m3fn, device=device
        )
        y = torch.full((m, l), 0.25, device=device, dtype=torch.float8_e4m3fn).t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        out_fp8 = torch._scaled_mm(
            x, y, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float8_e4m3fn
        )

        x_sparse = to_sparse_semi_structured(x)
        out_fp8_sparse = torch._scaled_mm(
            x_sparse, y, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float8_e4m3fn
        )
        # this fails on ROCm currently because hipblaslt doesn't have amax op
        out_fp32 = out_fp8.to(torch.float32)
        out_fp32_sparse = out_fp8_sparse.to(torch.float32)
        torch.testing.assert_close(out_fp32, out_fp32_sparse, rtol=1e-1, atol=1e-1)

    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @xfailIfSM89PreCUDA13
    @parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("dense_input_shape", [(256, 128)])
    def test_sparse_semi_structured_scaled_mm(
        self, dense_input_shape, device, out_dtype
    ):
        # [FUNCTION DOC] Quantizes a masked matrix and dense operand to float8 with scales,
        # runs the scaled mm op with both dense and semi-structured first operand for the
        # given output dtype, and checks the results are close.
        A = rand_sparse_semi_structured_mask(256, 128, dtype=torch.float16)
        B = torch.rand(dense_input_shape, device=device).to(torch.float16).t()

        A_fp8, A_scale = to_float8(A)
        B_fp8, B_scale = to_float8(B)

        A_fp8_sparse = to_sparse_semi_structured(A_fp8)

        dense_result = torch._scaled_mm(
            A_fp8, B_fp8, scale_a=A_scale, scale_b=B_scale, out_dtype=out_dtype
        )
        sparse_result = torch._scaled_mm(
            A_fp8_sparse, B_fp8, scale_a=A_scale, scale_b=B_scale, out_dtype=out_dtype
        )
        torch.testing.assert_close(dense_result, sparse_result, rtol=7e-2, atol=7e-2)

    @parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.int32])
    @parametrize("dense_input_shape", [(128, 128)])
    def test_cslt_sparse_mm_mixed_dtype(self, dense_input_shape, out_dtype, device):
        # [FUNCTION DOC] Compresses an int8 masked matrix with the cslt compress op, runs the
        # cslt sparse mm op with a mixed output dtype (half, bfloat16 or int32), and compares
        # with an int64 CPU matmul cast to that dtype.
        A = rand_sparse_semi_structured_mask(128, 128, dtype=torch.int8)
        A_compressed = torch._cslt_compress(A)

        B = torch.rand(dense_input_shape, device=device).to(torch.int8)

        dense_result = torch.mm(
            A.cpu().to(torch.int64), B.t().cpu().to(torch.int64)
        ).to(device, dtype=out_dtype)
        sparse_result = torch._cslt_sparse_mm(A_compressed, B.t(), out_dtype=out_dtype)
        torch.testing.assert_close(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

    @unittest.skip("cuSPARSELt v0.6.x does not support bfloat/float16 alpha scaling")
    @training_dtypes
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    def test_cslt_sparse_mm_alpha(self, dtype, device):
        # [FUNCTION DOC] Intended to check alpha and bias in the cslt sparse mm op against a
        # dense reference scaled per row. Unconditionally skipped because cuSPARSELt 0.6.x
        # lacks half alpha scaling.
        A = torch.Tensor([0, 0, 1, 1]).tile((128, 64)).to(dtype).cuda()
        B = torch.ones((256, 128), device=device).to(dtype)
        alpha = torch.Tensor([2 ** (-i) for i in range(128)]).cuda()
        bias = torch.ones(128, device=device).to(dtype)

        A_compressed = torch._cslt_compress(A)
        sparse_result = torch._cslt_sparse_mm(A_compressed, B, alpha=alpha, bias=bias)

        alpha_scaled = torch.stack([alpha] * 128).t()
        dense_result = alpha_scaled * torch.mm(A.to(torch.float32), B.to(torch.float32))
        dense_result = dense_result.to(dtype)

        torch.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    @parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.int32])
    @unittest.skipIf(
        torch.version.hip and not _IS_HIPSPARSELT_AVAILABLE,
        "HIPSPARSELt is not available for ROCm versions < 7.12"
    )
    def test_cslt_sparse_mm_alpha_compile_autotune(self, device, out_dtype):
        # [FUNCTION DOC] Runs the cslt sparse mm op under torch.compile in max-autotune mode
        # with an int8 operand, per-row alpha and a chosen output dtype, and compares with a
        # CPU int64 reference scaled by alpha.
        A = torch.Tensor([0, 0, 1, 1]).tile((128, 64)).to(torch.int8).to(device)
        B = torch.ones((128, 256), device=device, dtype=torch.int8).t()
        alpha = torch.Tensor([2 ** (-i) for i in range(128)]).cuda()

        A_compressed = torch._cslt_compress(A)

        cslt_sparse_mm_c = torch.compile(torch._cslt_sparse_mm, mode="max-autotune")
        sparse_result = cslt_sparse_mm_c(
            A_compressed, B, alpha=alpha, out_dtype=out_dtype
        )

        # disable this otherwise inductor will attempt to reorder strides and pass a contiguous B
        @torch.compiler.disable
        def get_dense_result():
            # [FUNCTION DOC] HELPER: Computes the dense reference on CPU: int64 matmul of the
            # operands, scaled by the stacked alpha values, cast to the output dtype. Kept
            # outside compilation so strides of B stay unchanged.
            alpha_scaled = torch.stack([alpha] * 128).t().cpu().float()
            dense_result = alpha_scaled * torch.mm(
                A.to(torch.int64).cpu(), B.to(torch.int64).cpu()
            )
            dense_result = dense_result.to(out_dtype)
            return dense_result

        torch.testing.assert_close(
            sparse_result.cpu(), get_dense_result(), rtol=1e-3, atol=1e-3
        )

    @parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.int32])
    @unittest.skipIf(
        torch.version.hip and not _IS_HIPSPARSELT_AVAILABLE,
        "HIPSPARSELt is not available for ROCm versions < 7.12"
    )
    def test_cslt_sparse_mm_alpha_mixed_dtype(self, out_dtype, device):
        # [FUNCTION DOC] Runs the cslt sparse mm op with int8 inputs, per-row alpha (all ones
        # for int32 output) and a mixed output dtype, and compares with a CPU int64 reference
        # scaled by alpha.
        A = torch.Tensor([0, 0, 10, 10]).tile((128, 64)).to(torch.int8).cuda()
        B = torch.ones((128, 256), device=device).to(torch.int8).t()
        alpha = torch.Tensor(
            [2 ** (-i) if out_dtype is not torch.int32 else 1 for i in range(128)]
        ).cuda()

        A_compressed = torch._cslt_compress(A)
        sparse_result = torch._cslt_sparse_mm(
            A_compressed, B, alpha=alpha, out_dtype=out_dtype
        ).cpu()

        alpha_scaled = torch.stack([alpha] * 128).t()
        dense_result = alpha_scaled.cpu() * torch.mm(
            A.to(torch.int64).cpu(), B.to(torch.int64).cpu()
        )
        dense_result = dense_result.to(out_dtype)

        torch.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(
        torch.version.hip and not _IS_HIPSPARSELT_AVAILABLE,
        "HIPSPARSELt is not available for ROCm versions < 7.12"
    )
    @inference_dtypes
    def test_cslt_sparse_mm_search(self, device, dtype):
        # [FUNCTION DOC] Calls the cslt algorithm search op for a compressed matrix, uses the
        # returned algorithm id in the sparse mm op, and checks the result matches a float32
        # dense product cast to the dtype.
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        B = torch.ones((128, 128), device=device).to(dtype)

        A_compressed = torch._cslt_compress(A)
        alg_id = torch._cslt_sparse_mm_search(A_compressed, B.t())
        sparse_result = torch._cslt_sparse_mm(A_compressed, B.t(), alg_id=alg_id)
        dense_result = torch.mm(A.to(torch.float32), B.to(torch.float32))
        dense_result = dense_result.to(dtype)
        torch.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(
        torch.version.hip and not _IS_HIPSPARSELT_AVAILABLE,
        "HIPSPARSELt is not available for ROCm versions < 7.12"
    )
    @inference_dtypes
    def test_csrc_cslt_sparse_mm_search(self, device, dtype):
        # [FUNCTION DOC] Same as the search test but calls the low-level C binding for the
        # search, which returns algorithm id, split-k and split-k mode, and passes all three
        # to the sparse mm op; checks against a dense product.
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        B = torch.ones((128, 128), device=device).to(dtype)

        A_compressed = torch._cslt_compress(A)
        alg_id, split_k, split_k_mode, _ = torch._C._cusparselt.mm_search(
            A_compressed, B.t(), None, None, None, False
        )
        sparse_result = torch._cslt_sparse_mm(
            A_compressed,
            B.t(),
            alg_id=alg_id,
            split_k=split_k,
            split_k_mode=split_k_mode,
        )
        dense_result = torch.mm(A.to(torch.float32), B.to(torch.float32))
        dense_result = dense_result.to(dtype)
        torch.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(
        torch.version.hip and not _IS_HIPSPARSELT_AVAILABLE,
        "HIPSPARSELt is not available for ROCm versions < 7.12"
    )
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("dense_input_shape", [(128, 1), (128, 3)])
    def test_mm_padded_output_shape(self, dtype, dense_input_shape, device):
        # [FUNCTION DOC] Checks that mm with a sparse first operand and dense columns not a
        # multiple of the minimum (padding path) returns the same output shape as the dense
        # mm.
        """Explicitly verify the padded path in cusparselt_mm trims output to the correct shape.

        Shapes like (128, 1) and (128, 3) have n % dense_min_cols != 0 (dense_min_cols=8),
        triggering the need_pad branch and the narrow(0, 0, out_cols) dim slicing.
        """
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)
        B = torch.rand(dense_input_shape, device=device).to(dtype)

        dense_result = torch.mm(A, B)
        sparse_result = torch.mm(A_sparse, B)

        self.assertEqual(sparse_result.shape, dense_result.shape)

    @unittest.skipIf(
        torch.version.hip and not _IS_HIPSPARSELT_AVAILABLE,
        "HIPSPARSELt is not available for ROCm versions < 7.12"
    )
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("dense_input_shape", [(1, 128), (3, 128)])
    def test_mm_padded_output_shape_sparse_second(self, dtype, dense_input_shape, device):
        # [FUNCTION DOC] Checks that mm of a dense matrix with the transposed sparse matrix,
        # with rows needing padding, returns the same output shape as the dense computation.
        """Verify cusparselt_mm trims output correctly for dense @ sparse.t() (Branch B).

        Shapes like (1, 128) and (3, 128) have m % dense_min_rows != 0 (dense_min_rows=8),
        triggering need_pad with should_transpose_dense=True. The narrow(0, 0, out_rows)
        path must correctly return shape (m, n_out) without a trailing .t().
        """
        B = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        B_sparse = to_sparse_semi_structured(B)
        A = torch.rand(dense_input_shape, device=device).to(dtype)

        dense_result = torch.mm(A, B.t())
        sparse_result = torch.mm(A, B_sparse.t())

        self.assertEqual(sparse_result.shape, dense_result.shape)

    @unittest.skipIf(
        torch.version.hip and not _IS_HIPSPARSELT_AVAILABLE,
        "HIPSPARSELt is not available for ROCm versions < 7.12"
    )
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("dense_input_shape", [(1, 128), (3, 128)])
    def test_addmm_padded_output_shape_sparse_second(self, dtype, dense_input_shape, device):
        # [FUNCTION DOC] Checks that addmm with a bias, a dense matrix and a transposed sparse
        # matrix, with padding-triggering shapes, returns the same output shape as the dense
        # addmm.
        """Verify cusparselt_mm trims output correctly for addmm with dense @ sparse.t() (Branch B + bias).

        Covers the semi_sparse_addmm path (used by nn.Linear) with should_transpose_dense=True
        and padding-triggering input shapes.
        """
        B = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        B_sparse = to_sparse_semi_structured(B)
        A = torch.rand(dense_input_shape, device=device).to(dtype)
        bias = torch.rand(256, device=device).to(dtype)

        dense_result = torch.addmm(bias, A, B.t())
        sparse_result = torch.addmm(bias, A, B_sparse.t())

        self.assertEqual(sparse_result.shape, dense_result.shape)

    @unittest.skipIf(
        torch.version.hip and not _IS_HIPSPARSELT_AVAILABLE,
        "HIPSPARSELt is not available for ROCm versions < 7.12"
    )
    def test_cusparselt_backend(self):
        # [FUNCTION DOC] Checks cuSPARSELt is available and, on CUDA, that its version is at
        # least 602. Raises AssertionError otherwise.
        if not torch.backends.cusparselt.is_available():
            raise AssertionError("cusparselt backend should be available")

        # PyTorch CUDA 12.4+ using cuSPARSELt v0.6.2+
        if torch.backends.cusparselt.version() < 602 and not torch.version.hip:
            raise AssertionError(
                f"cusparselt version should be >= 602, got {torch.backends.cusparselt.version()}"
            )

    @inference_dtypes
    def test_cslt_sparse_tensor_clone(self, dtype):
        # [FUNCTION DOC] Clones a semi-structured tensor and checks the packed data of the
        # clone equals the original.
        # KNOWN ISSUE: data_ptr is referenced without being called, so the not-equal check
        #   compares bound methods and does not verify separate storage.
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype).cuda()
        A_sparse = to_sparse_semi_structured(A)
        A_clone = A_sparse.clone()
        self.assertNotEqual(A_sparse.packed.data_ptr, A_clone.packed.data_ptr)
        self.assertEqual(A_sparse.packed, A_clone.packed)

    @dtypes(torch.float16, torch.bfloat16)
    def test_semi_sparse_to_device_cpu(self, dtype):
        # [FUNCTION DOC] Checks converting a semi-structured tensor to cpu yields a plain
        # dense CPU tensor (not the sparse subclass) equal to the densified original.
        """Test .to('cpu') converts sparse semi-structured tensor to dense on CPU."""
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype).cuda()
        A_sparse = to_sparse_semi_structured(A)
        A_dense = A_sparse.to_dense()

        result = A_sparse.to("cpu")
        self.assertEqual(result.device.type, "cpu")
        self.assertFalse(isinstance(result, SparseSemiStructuredTensor))
        torch.testing.assert_close(result, A_dense.cpu(), rtol=1e-3, atol=1e-3)

    @dtypes(torch.float16, torch.bfloat16)
    def test_semi_sparse_to_device_cuda(self, dtype):
        # [FUNCTION DOC] Checks that moving a semi-structured tensor to its own device with
        # copy raises NotImplementedError.
        """Test .to('cuda') / .to(device) raises NotImplementedError."""
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype).cuda()
        A_sparse = to_sparse_semi_structured(A)

        with self.assertRaises(NotImplementedError):
            A_sparse.to(A_sparse.device, copy=True)

    @dtypes(torch.float16, torch.bfloat16)
    def test_semi_sparse_to_dtype(self, dtype):
        # [FUNCTION DOC] Checks that converting a semi-structured tensor to float32 alone
        # raises NotImplementedError.
        """Test .to(dtype) raises NotImplementedError (only to('cpu') is supported)."""
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype).cuda()
        A_sparse = to_sparse_semi_structured(A)

        with self.assertRaises(NotImplementedError):
            A_sparse.to(torch.float32)

    @dtypes(torch.float16, torch.bfloat16)
    def test_semi_sparse_to_device_and_dtype(self, dtype):
        # [FUNCTION DOC] Checks converting to cpu and float32 together with copy gives a dense
        # CPU float32 tensor (not sparse) equal to the densified original cast to float32.
        """Test .to(device, dtype) converts both device and dtype."""
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype).cuda()
        A_sparse = to_sparse_semi_structured(A)
        A_dense = A_sparse.to_dense()

        result = A_sparse.to("cpu", torch.float32, copy=True)
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.dtype, torch.float32)
        self.assertFalse(isinstance(result, SparseSemiStructuredTensor))
        torch.testing.assert_close(
            result, A_dense.cpu().to(torch.float32), rtol=1e-3, atol=1e-3
        )

    @dtypes(torch.float16, torch.bfloat16)
    def test_semi_sparse_to_same_dtype_noop(self, dtype):
        # [FUNCTION DOC] Checks that converting to the same dtype with copy raises
        # NotImplementedError.
        # KNOWN ISSUE: The test name says noop but the test expects an error.
        """Test .to(same_dtype) raises NotImplementedError (only to('cpu') is supported)."""
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype).cuda()
        A_sparse = to_sparse_semi_structured(A)

        with self.assertRaises(NotImplementedError):
            A_sparse.to(dtype, copy=True)

    @dtypes(torch.float16, torch.bfloat16)
    def test_semi_sparse_to_kwargs(self, dtype):
        # [FUNCTION DOC] Checks the keyword form of the conversion with device and dtype gives
        # a dense CPU float32 tensor equal to the densified original.
        """Test .to() with keyword arguments (device=, dtype=)."""
        A = rand_sparse_semi_structured_mask(128, 128, dtype=dtype).cuda()
        A_sparse = to_sparse_semi_structured(A)
        A_dense = A_sparse.to_dense()

        result = A_sparse.to(device="cpu", dtype=torch.float32)
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.dtype, torch.float32)
        self.assertFalse(isinstance(result, SparseSemiStructuredTensor))
        torch.testing.assert_close(
            result, A_dense.cpu().to(torch.float32), rtol=1e-3, atol=1e-3
        )

    @unittest.skipIf(
        torch.version.hip and not _IS_HIPSPARSELT_AVAILABLE,
        "HIPSPARSELt is not available for ROCm versions < 7.12"
    )
    def test_cslt_sparse_tensor_with_alg_id(self, device):
        # [FUNCTION DOC] Compresses a matrix, finds an algorithm id via search, builds a semi-
        # structured tensor with that id, checks the stored id attribute matches and that the
        # sparse mm result is close to the dense one.
        A = rand_sparse_semi_structured_mask(256, 128, dtype=torch.float16).cuda()
        A_compressed = torch._cslt_compress(A)
        B = torch.ones((128, 128), device=device).to(torch.float16)
        alg_id = torch._cslt_sparse_mm_search(A_compressed, B)
        A_sparse = to_sparse_semi_structured(A, alg_id=alg_id)
        self.assertEqual(A_sparse.alg_id_cusparselt, alg_id)
        dense_result = torch.mm(A, B).to(torch.float16)
        sparse_result = torch.mm(A_sparse, B).to(torch.float16)
        torch.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)


    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8_SPARSE,
        "FP8 sparse requires cuSPARSELt v0.6.2+ on SM 8.9+ or MI350+ (gfx950) on ROCm",
    )
    def test_cslt_compress_fp8(self, device):
        # [FUNCTION DOC] Quantizes a masked half matrix to float8 and checks the cslt compress
        # op returns a float8 tensor.
        A = rand_sparse_semi_structured_mask(256, 128, dtype=torch.float16)
        A_fp8, _ = to_float8(A)
        compressed = torch._cslt_compress(A_fp8)
        self.assertEqual(compressed.dtype, torch.float8_e4m3fn)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8_SPARSE,
        "FP8 sparse requires cuSPARSELt v0.6.2+ on SM 8.9+ or MI350+ (gfx950) on ROCm",
    )
    @parametrize("dense_input_shape", [(256, 128)])
    def test_cslt_sparse_mm_fp8_to_fp32(self, dense_input_shape, device):
        # [FUNCTION DOC] Compresses a float8 sparse matrix, multiplies it with a float8 dense
        # operand through the cslt sparse mm op with float32 output, and checks the output
        # dtype is float32 and the values are close to a float32 dense product.
        A = rand_sparse_semi_structured_mask(256, 128, dtype=torch.float16)
        B = torch.rand(dense_input_shape, device=device).to(torch.float16).t()

        A_fp8, _ = to_float8(A)
        B_fp8, _ = to_float8(B)

        compressed = torch._cslt_compress(A_fp8)
        sparse_result = torch._cslt_sparse_mm(compressed, B_fp8, out_dtype=torch.float32)

        self.assertEqual(sparse_result.dtype, torch.float32)

        dense_result = torch.mm(A_fp8.to(torch.float32), B_fp8.to(torch.float32))
        torch.testing.assert_close(sparse_result, dense_result, rtol=1e-1, atol=1e-1)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8_SPARSE,
        "FP8 sparse requires cuSPARSELt v0.6.2+ on SM 8.9+ or MI350+ (gfx950) on ROCm",
    )
    @unittest.skipIf(not TEST_WITH_ROCM, "ROCm-specific out_dtype restriction")
    @parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
    def test_cslt_sparse_mm_fp8_unsupported_out_dtype_rocm(self, out_dtype, device):
        # [FUNCTION DOC] ROCm only: checks the cslt sparse mm op with fp8 inputs raises
        # RuntimeError saying the output dtype must be float32 when the requested output dtype
        # is half, bfloat16 or float8.
        A = rand_sparse_semi_structured_mask(256, 128, dtype=torch.float16)
        B = torch.rand(128, 128, device=device).to(torch.float16).t()

        A_fp8, _ = to_float8(A)
        B_fp8, _ = to_float8(B)

        compressed = torch._cslt_compress(A_fp8)
        with self.assertRaisesRegex(
            RuntimeError,
            "Unsupported out_dtype passed, must be float32 for fp8 inputs on ROCm",
        ):
            torch._cslt_sparse_mm(compressed, B_fp8, out_dtype=out_dtype)

if len(SEMI_STRUCTURED_SUPPORTED_BACKENDS) > 0:
    instantiate_device_type_tests(TestSparseSemiStructured, globals(), only_for="cuda")
if "cutlass" in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
    instantiate_device_type_tests(
        TestSparseSemiStructuredCUTLASS, globals(), only_for="cuda"
    )
    instantiate_device_type_tests(
        TestSparseSemiStructuredTraining, globals(), only_for="cuda"
    )
if "cusparselt" in SEMI_STRUCTURED_SUPPORTED_BACKENDS:
    instantiate_device_type_tests(
        TestSparseSemiStructuredCUSPARSELT, globals(), only_for="cuda"
    )

if __name__ == "__main__":
    run_tests()
