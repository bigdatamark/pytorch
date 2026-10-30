# Owner(s): ["module: scatter & gather ops"]
# [MODULE DOC] Tests for scatter and gather operators: gather, scatter_, scatter_add_ and
# scatter_reduce_ across dtypes, reductions and deterministic mode, plus expanded-index fast paths
# on CPU. A second half tests the CUDA CuTeDSL scatter_add override from torch._native (dispatch
# predicates and end-to-end numerics against a naive reference), which needs CUDA and the CuTeDSL
# package. Import-time effects: asserts the default dtype is float32, instantiates device-generic
# tests for TestScatterGather and parametrized tests for the two CuTeDSL classes.

import random
import unittest
from typing import NamedTuple

import torch

from torch.testing import make_tensor
from torch.testing._internal.common_utils import \
    (instantiate_parametrized_tests, parametrize, run_tests, skipIfNoCuteDSL,
     subtest, TestCase, DeterministicGuard, TEST_CUDA, TEST_WITH_ROCM, serialTest)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, onlyCPU, onlyAccelerator, dtypes, dtypesIfCUDA,
     toleranceOverride, tol,)
from torch.testing._internal.common_dtype import \
    (all_passthru_types, all_passthru_types_and, get_all_dtypes,)

from torch.testing._internal.common_cuda import gfx_arch_supports_opportunistic_fastatomics, SM90OrLater

# Protects against includes accidentally setting the default dtype
if torch.get_default_dtype() is not torch.float32:
    raise AssertionError("default dtype should be float32")


# Note: test_scatter_gather_ops.py
# This test file tests scatter and gather operations,
#   like torch.scatter and torch.gather.

class TestScatterGather(TestCase):
    # [CLASS DOC] Device-generic tests of gather and the scatter family against slow loop-based
    # reference implementations built from random shapes and random valid index tensors.
    # Fills an index tensor with valid indices
    def _fill_indices(self, idx, dim, dim_size, elems_per_row, m, n, o, unique_indices=True):
        # [FUNCTION DOC] HELPER: Fills the preallocated integer index tensor idx in place. For
        # each position over the two non-scatter dimensions it assigns a vector of elems_per_row
        # indices into dim_size along dimension dim: a random permutation prefix (distinct values)
        # when unique_indices is true, otherwise independent random draws (repeats allowed). The
        # arguments m, n, o are the three sizes of the source tensor; dim must be 0, 1 or 2.
        # Returns nothing.
        for i in range(1 if dim == 0 else m):
            for j in range(1 if dim == 1 else n):
                for k in range(1 if dim == 2 else o):
                    ii = [i, j, k]
                    ii[dim] = slice(0, idx.size(dim) + 1)
                    if unique_indices:
                        idx[tuple(ii)] = torch.randperm(dim_size)[0:elems_per_row]
                    else:
                        idx[tuple(ii)] = torch.randint(dim_size, (elems_per_row,))

    @dtypes(*all_passthru_types())
    @dtypesIfCUDA(*all_passthru_types_and(torch.chalf))
    def test_gather(self, device, dtype):
        # [FUNCTION DOC] Builds a random 3-D source tensor, a long index tensor along a random
        # dimension filled with distinct valid indices, and compares torch.gather against an
        # explicit triple loop reference with exact equality. Also, for dtypes where max is
        # defined (not complex, not barebones unsigned), checks gather with the argmax indices
        # returned by max reproduces the max values.
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        src = make_tensor((m, n, o), device=device, dtype=dtype)
        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = make_tensor(idx_size, device=device, dtype=torch.long)
        self._fill_indices(idx, dim, src.size(dim), elems_per_row, m, n, o)

        actual = torch.gather(src, dim, idx)
        expected = torch.zeros(idx_size, device=device, dtype=dtype)
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    expected[i, j, k] = src[tuple(ii)]
        self.assertEqual(actual, expected, atol=0, rtol=0)

        # Guarded because torch.max isn't defined for complex or barebones
        # unsigned tensors.
        max_unsupported = (
            dtype.is_complex
            or dtype in (torch.uint16, torch.uint32, torch.uint64)
        )
        if not max_unsupported:
            src = make_tensor((3, 4, 5), device=device, dtype=dtype)
            expected, idx = src.max(2, True)
            actual = torch.gather(src, 2, idx)
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @serialTest()
    @dtypes(torch.int8, torch.bfloat16)
    def test_gather_large(self, device, dtype):
        # [FUNCTION DOC] Stress test of gather on large 2-D shapes for int8 and bfloat16 (the
        # vectorized path). For each shape and both dims, compares gather against plain advanced
        # indexing and, on non-CPU devices, against a CPU result, using contiguous, strided,
        # misaligned-by-offset and misaligned-row-stride sources and contiguous or strided
        # expanded indices. Frees tensors each iteration and empties the accelerator cache.
        # Finally checks a gather along dim one with an index of extent one that could wrongly
        # take a fast path. Serial test (not run in parallel).
        # KNOWN ISSUE: for dim one the line that indexes the second misaligned view uses the first
        # misaligned tensor instead, so that variant of the advanced-indexing check is not
        # exercised.
        # test larger shapes to check vectorized implementation
        for (m, n, k) in ((4096, 3072, 4096), (4096, 3072, 4100), (4, 4, 16384 * 8192)):
            if torch.accelerator.is_available():
                torch.accelerator.empty_cache()
            src = make_tensor((m, k), device=device, dtype=dtype)
            alloc0 = torch.empty(src.nelement() * 2, device=device, dtype=dtype)
            discontig = alloc0.view(m, 2 * k)[:, ::2].copy_(src)
            alloc1 = torch.empty(src.nelement() + 1, device=device, dtype=dtype)
            misaligned = alloc1[1:].view(m, k).copy_(src)
            alloc2 = torch.empty(m, k + 4, device=device, dtype=dtype)
            misaligned1 = alloc2[:, :-4].copy_(src)
            num_ind = n
            for dim in (0, 1):
                max_ind = src.shape[dim]
                ind0 = torch.randint(max_ind, (num_ind,), device=device)
                ind_discontig0 = torch.empty(num_ind * 2, device=device, dtype=torch.int64)[::2].copy_(ind0)
                shape_ind = [1] * src.ndim
                shape_ind[dim] = ind0.shape[0]
                shape_out = list(src.shape)
                shape_out[dim] = ind0.shape[0]
                ind = ind0.view(shape_ind).expand(shape_out)
                ind_discontig = ind_discontig0.view(shape_ind).expand(shape_out)
                res = torch.gather(src, dim=dim, index=ind)
                ref = src[ind0] if dim == 0 else src[:, ind0]
                self.assertEqual(res, ref, atol=0, rtol=0)
                if device != 'cpu':
                    ref_cpu = src.cpu()[ind0.cpu()] if dim == 0 else src.cpu()[:, ind0.cpu()]
                    self.assertEqual(res.cpu(), ref_cpu, atol=0, rtol=0)
                res = torch.gather(src, dim=dim, index=ind_discontig)
                self.assertEqual(res, ref, atol=0, rtol=0)
                res_ind = src[ind_discontig0] if dim == 0 else src[:, ind_discontig0]
                self.assertEqual(res_ind, ref, atol=0, rtol=0)
                res_ind_neg = src[ind0 - src.shape[dim]] if dim == 0 else src[:, ind0 - src.shape[1]]
                self.assertEqual(res_ind_neg, ref, atol=0, rtol=0)
                res = torch.gather(discontig, dim=dim, index=ind)
                self.assertEqual(res, ref, atol=0, rtol=0)
                res_ind = discontig[ind0] if dim == 0 else discontig[:, ind0]
                self.assertEqual(res_ind, ref, atol=0, rtol=0)
                res = torch.gather(misaligned, dim=dim, index=ind)
                self.assertEqual(res, ref, atol=0, rtol=0)
                res_ind = misaligned[ind0] if dim == 0 else misaligned[:, ind0]
                self.assertEqual(res_ind, ref, atol=0, rtol=0)
                res_ind = misaligned1[ind0] if dim == 0 else misaligned[:, ind0]
                self.assertEqual(res_ind, ref, atol=0, rtol=0)
                res_gather = torch.gather(misaligned1, dim=dim, index=ind)
                self.assertEqual(res_gather, ref, atol=0, rtol=0)
            del src, alloc0, alloc1, alloc2
            del discontig, misaligned, misaligned1
        # test gather along 1st dim that can accidentally trigger fast path
        # because due to index dimension in the gather dim being 1
        # an unexpected squashing in tensorIterator happens
        src = make_tensor((16, 2, 16), device=device, dtype=dtype)
        ind = torch.randint(2, (16, 1), device=device).view(16, 1, 1).expand(16, 1, 16)
        res = torch.gather(src, dim=1, index=ind)
        if device != 'cpu':
            ref_cpu = torch.gather(src.cpu(), dim=1, index=ind.cpu())
            self.assertEqual(res.cpu(), ref_cpu, atol=0, rtol=0)


    @dtypes(torch.bool)
    def test_gather_bool(self, device, dtype):
        # [FUNCTION DOC] Checks gather on a small 2x2 boolean tensor with a given long index
        # matches the expected boolean result exactly.
        src = torch.tensor(((False, True), (True, True)), device=device, dtype=dtype)
        idx = torch.tensor(((0, 0), (1, 0)), device=device, dtype=torch.long)
        actual = torch.gather(src, 1, idx)
        expected = torch.tensor(((False, False), (True, True)), device=device, dtype=dtype)
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @parametrize("sparse_grad", [False, True])
    @dtypes(torch.float32, torch.float64)
    def test_gather_backward_with_empty_index_tensor(self, device, dtype, sparse_grad):
        # [FUNCTION DOC] Checks that backward through gather with an empty index (zero columns)
        # works for dense and sparse_grad modes: the gradient (converted to dense for sparse) is
        # all zeros with the input shape. Parametrized over sparse_grad and float dtypes.
        dim = -1
        input = torch.rand([10, 5], dtype=dtype, device=device, requires_grad=True)
        index = torch.randint(0, 2, [3, 0], dtype=torch.int64, device=device)
        res = torch.gather(input, dim, index, sparse_grad=sparse_grad)
        res.sum().backward()
        grad = input.grad.to_dense() if sparse_grad else input.grad
        expected_grad = torch.zeros_like(input, requires_grad=False)
        self.assertEqual(grad, expected_grad, atol=0, rtol=0)

    def _test_scatter_base(self, fn, *, device, dtype, is_scalar, reduction,
                           unique_indices=True, include_self=True):
                               # [FUNCTION DOC] HELPER: Shared driver for the scatter tests. fn is
                               # an unbound Tensor method (scatter_, scatter_add_ or
                               # scatter_reduce_), is_scalar selects a scalar or tensor source,
                               # reduction is a reduce name or None, and unique_indices and
                               # include_self tune the index and self handling. It builds a random
                               # base, index and source, computes the expected result in Python
                               # loops (using 64-bit integers for small signed ints on
                               # add/mul/mean to avoid overflow, tracking per-element counts for
                               # include_self false and for mean), and calls the skip hook of the
                               # test case if the reference would overflow the dtype. Then runs fn
                               # on a clone and compares, with loose tolerance for half and
                               # bfloat16 and a small one on ROCm with fast atomics, else exact.
                               # Finally checks that an empty index leaves the destination
                               # unchanged. Fails via assertions.
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = torch.empty(tuple(idx_size), device=device, dtype=torch.long)
        self._fill_indices(idx, dim, ([m, n, o])[dim], elems_per_row, m, n, o, unique_indices)

        if is_scalar:
            src = random.random()
        else:
            src_size = [random.randint(1, 5) + s for s in idx_size]
            src = make_tensor(tuple(src_size), device=device, dtype=dtype)

        base = make_tensor((m, n, o), device=device, dtype=dtype)
        # for signed integers, avoid undefined behavior in expected output
        use_int64_expected = (
            dtype in (torch.int8, torch.int16, torch.int32)
            and reduction in {"add", "sum", "multiply", "prod", "mean"}
        )
        expected = base.to(torch.int64, copy=True) if use_int64_expected else base.clone()

        counts = torch.zeros(base.shape, dtype=torch.long, device=device) + include_self
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    if fn is torch.Tensor.scatter_add_:
                        expected[tuple(ii)] += src[i, j, k]
                    else:
                        # method may be 'scatter_', 'scatter', 'scatter_reduce'
                        # or 'scatter_reduce_', the former two might have a reduction argument
                        # while the latter two always do
                        value = src if is_scalar else src[i, j, k]
                        if use_int64_expected:
                            value = int(value) if is_scalar else value.to(torch.int64)

                        if ((not include_self) and counts[tuple(ii)] == 0):
                            expected[tuple(ii)] = value
                        else:
                            if reduction == "add" or reduction == "sum":
                                expected[tuple(ii)] += value
                            elif reduction == "multiply" or reduction == "prod":
                                expected[tuple(ii)] *= value
                            elif reduction == "amax":
                                expected[tuple(ii)] = max(expected[tuple(ii)], value)
                            elif reduction == "amin":
                                expected[tuple(ii)] = min(expected[tuple(ii)], value)
                            elif reduction == "mean":
                                expected[tuple(ii)] += value
                            else:
                                expected[tuple(ii)] = value

                        counts[tuple(ii)] += 1

        if use_int64_expected:
            iinfo = torch.iinfo(dtype)
            expected_out_of_range = torch.any(
                (expected < iinfo.min) | (expected > iinfo.max)
            )
            if bool(expected_out_of_range.item()):
                min_observed = expected.min().item()
                max_observed = expected.max().item()
                self.skipTest(
                    f"Skipping {fn.__name__} reference case with {dtype=} and "
                    f"{reduction=} because it would overflow {dtype}: observed "
                    f"expected range [{min_observed}, {max_observed}], allowed "
                    f"[{iinfo.min}, {iinfo.max}]"
                )
            else:
                # note: this cast is not UB because we checked above
                expected = expected.to(dtype)

        if (reduction == "mean"):
            counts.masked_fill_(counts == 0, 1)
            if (dtype.is_floating_point or dtype.is_complex):
                expected /= counts
            else:
                expected.div_(counts, rounding_mode="floor")

        # note: only call `fn` once we know expected values in `base`
        # are safe from signed overflow (UB). Otherwise, test is skipped above
        if reduction is not None:
            if fn is torch.Tensor.scatter_reduce_:
                actual = fn(base.clone(), dim, idx, src, reduce=reduction, include_self=include_self)
            else:
                actual = fn(base.clone(), dim, idx, src, reduce=reduction)
        else:
            actual = fn(base.clone(), dim, idx, src)

        if dtype == torch.float16 or dtype == torch.bfloat16:
            # Some CUDA kernels (e.g. indexing_backward_kernel_stride_1) that are called during
            # the test use fp32 for internal accumulation for improved accuracy. When using 16 bit
            # precision types can be small differences
            self.assertEqual(actual, expected, atol=0.04, rtol=0.05)
        else:
            # When we are running opportunistic_fastatomics, we will expect some floating point rounding
            # errors as the order of operation is not guaranteed.
            if TEST_WITH_ROCM and gfx_arch_supports_opportunistic_fastatomics() \
                    and not torch.are_deterministic_algorithms_enabled():
                self.assertEqual(actual, expected, atol=1e-9, rtol=1e-6)
            else:
                self.assertEqual(actual, expected, atol=0, rtol=0)

        # Tests empty index
        dst = make_tensor((2, 2), device=device, dtype=dtype)
        idx = torch.tensor((), device=device, dtype=torch.long)
        src = make_tensor((2, 2), device=device, dtype=dtype)
        if reduction is not None:
            actual = fn(dst, 0, idx, src, reduce=reduction)
        else:
            actual = fn(dst, 0, idx, src)
        self.assertEqual(actual, dst, atol=0, rtol=0)

    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter_(self, device, dtype):
        # [FUNCTION DOC] Runs the base scatter_ test with a tensor source and no reduction, once
        # with deterministic algorithms off and once on, for half, float and complex dtypes.
        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
                self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                        is_scalar=False, reduction=None)

    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter__scalar(self, device, dtype):
        # [FUNCTION DOC] Runs the base scatter_ test with a scalar source value and no reduction.
        self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                is_scalar=True, reduction=None)

    # FIXME: RuntimeError: "cuda_scatter_gather_base_kernel_reduce_multiply" not implemented for 'ComplexFloat'
    @toleranceOverride({torch.float16: tol(atol=1e-2, rtol=0)})
    @dtypesIfCUDA(torch.float16, torch.float32)
    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter__reductions(self, device, dtype):
        # [FUNCTION DOC] Runs the base scatter_ test for the add and multiply reductions with both
        # tensor and scalar sources. Float16 uses a tolerance override; CUDA skips complex (a
        # known missing kernel).
        for reduction in ("add", "multiply"):
            self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                    is_scalar=False, reduction=reduction)
            self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                    is_scalar=True, reduction=reduction)

    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter_add_(self, device, dtype):
        # [FUNCTION DOC] Runs the base scatter_add_ test with a tensor source, with deterministic
        # mode off and on.
        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
                self._test_scatter_base(torch.Tensor.scatter_add_, device=device, dtype=dtype,
                                        is_scalar=False, reduction=None)

    @dtypes(torch.float32)
    def test_scatter_add_mult_index_base(self, device, dtype):
        # [FUNCTION DOC] Scatter-adds a ones matrix with an all-zero index along dim zero and
        # along dim one into zeros (every element lands on a single row or column). Checks the
        # first row or column holds the full count, in both deterministic modes.
        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
                m, n = 30, 40
                idx = torch.zeros(m, n, device=device, dtype=torch.long)
                src = torch.ones(m, n, device=device, dtype=dtype)
                res0 = torch.zeros(m, n, device=device, dtype=dtype).scatter_add_(0, idx, src)
                res1 = torch.zeros(m, n, device=device, dtype=dtype).scatter_add_(1, idx, src)

                self.assertEqual(res0[0, :], m * torch.ones(n, device=device, dtype=dtype), atol=0, rtol=0)
                self.assertEqual(res1[:, 0], n * torch.ones(m, device=device, dtype=dtype), atol=0, rtol=0)

    @serialTest()
    @onlyAccelerator
    @dtypes(torch.float32, torch.half, torch.bfloat16)
    def test_scatter_add_large(self, device, dtype):
        # [FUNCTION DOC] Accelerator-only, serial test of scatter_add_ on large shapes for the
        # vectorized or TMA path: compares against a CPU reference with dtype dependent tolerances
        # for contiguous input, strided source, misaligned self, scatter along dim one and a
        # sliced view whose row stride differs from its width. Uses a smaller last shape for
        # dtypes wider than two bytes to avoid running out of memory, and deletes tensors each
        # iteration.
        # test larger shapes that exercise the vectorized/TMA scatter_add path
        # and verify large launch configs don't produce invalid kernel launches
        if dtype in (torch.float32, torch.float64):
            atol, rtol = 1e-5, 1e-3
        elif dtype == torch.half:
            atol, rtol = 2e-2, 2e-2
        else:
            atol, rtol = 0.2, 0.5
        # The large shape tests grid.y clipping at maxGridSize[1]=65535.
        # Use it only for small dtypes to avoid OOM on CI (22GB GPUs).
        shapes = [(4096, 3072, 4096), (4096, 3072, 4100)]
        if dtype.itemsize <= 2:
            shapes.append((4, 4, 16384 * 8192))
        else:
            shapes.append((4, 4, 16384 * 256))
        for (m, n, k) in shapes:
            if torch.accelerator.is_available():
                torch.accelerator.empty_cache()
            self_tensor = torch.zeros(m, k, device=device, dtype=dtype)
            src = make_tensor((n, k), device=device, dtype=dtype)
            # contiguous + aligned (should hit fast path on dim=0)
            idx_1d = torch.randint(m, (n,), device=device)
            idx_2d = idx_1d.unsqueeze(1).expand(n, k)
            res = self_tensor.clone().scatter_add_(0, idx_2d, src)
            ref_cpu = torch.zeros(m, k, dtype=dtype).scatter_add_(0, idx_2d.cpu(), src.cpu())
            self.assertEqual(res.cpu(), ref_cpu, atol=atol, rtol=rtol)

            # discontiguous src (should fall back to element-wise)
            alloc = torch.empty(n, 2 * k, device=device, dtype=dtype)
            src_discontig = alloc[:, ::2].copy_(src)
            res = self_tensor.clone().scatter_add_(0, idx_2d, src_discontig)
            self.assertEqual(res.cpu(), ref_cpu, atol=atol, rtol=rtol)

            # misaligned self (should fall back)
            alloc_self = torch.empty(m * k + 1, device=device, dtype=dtype)
            self_misaligned = alloc_self[1:].view(m, k)
            self_misaligned.zero_()
            res = self_misaligned.scatter_add_(0, idx_2d, src)
            self.assertEqual(res.cpu(), ref_cpu, atol=atol, rtol=rtol)

            # dim=1 (fast path is dim=0 only)
            self_t1 = torch.zeros(k, m, device=device, dtype=dtype)
            src_t1 = make_tensor((k, n), device=device, dtype=dtype)
            idx_1d_d1 = torch.randint(m, (n,), device=device)
            idx_2d_d1 = idx_1d_d1.unsqueeze(0).expand(k, n)
            res_d1 = self_t1.clone().scatter_add_(1, idx_2d_d1, src_t1)
            ref_d1 = torch.zeros(k, m, dtype=dtype).scatter_add_(1, idx_2d_d1.cpu(), src_t1.cpu())
            self.assertEqual(res_d1.cpu(), ref_d1, atol=atol, rtol=rtol)

            # sliced tensors: contiguous within slice, aligned, but stride != D
            # this passes eligibility checks and must use the actual stride
            K = 256
            self_full = torch.zeros(m, K, device=device, dtype=dtype)
            src_full = make_tensor((n, K), device=device, dtype=dtype)
            self_sliced = self_full[:, 64:192]  # shape [m, 128], stride [256, 1]
            src_sliced = src_full[:, 64:192]
            idx_1d_s = torch.randint(m, (n,), device=device)
            idx_2d_s = idx_1d_s.unsqueeze(1).expand(n, 128)
            res = self_sliced.clone().scatter_add_(0, idx_2d_s, src_sliced)
            ref_cpu = self_sliced.cpu().clone().scatter_add_(0, idx_2d_s.cpu(), src_sliced.cpu())
            self.assertEqual(res.cpu(), ref_cpu, atol=atol, rtol=rtol)

            del self_tensor, src, alloc, alloc_self, self_misaligned
            del self_t1, src_t1, self_full, src_full

    # FIXME: discrepancy between bool ReduceAdd on CUDA and CPU (a + b on CPU and buggy a && b on CUDA)
    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_bool=False))
    def test_scatter_reduce_sum(self, device, dtype):
        # [FUNCTION DOC] Runs the base scatter_reduce_ test with the sum reduction for non-bool
        # dtypes, over both include_self values and both deterministic modes, with repeated
        # indices allowed.
        for include_self in (True, False):
            for deterministic in [False, True]:
                with DeterministicGuard(deterministic):
                    self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                            is_scalar=False, reduction='sum', unique_indices=False,
                                            include_self=include_self)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_prod(self, device, dtype):
        # [FUNCTION DOC] Runs the base scatter_reduce_ test with the prod reduction, over both
        # include_self values, with repeated indices allowed (CUDA excludes complex and bool
        # dtypes).
        for include_self in (True, False):
            self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                    is_scalar=False, reduction='prod', unique_indices=False,
                                    include_self=include_self)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_bool=False))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_mean(self, device, dtype):
        # [FUNCTION DOC] Runs the base scatter_reduce_ test with the mean reduction, for both
        # include_self values and both deterministic modes, with repeated indices allowed.
        for include_self in (True, False):
            for deterministic in [False, True]:
                with DeterministicGuard(deterministic):
                    self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                            is_scalar=False, reduction='mean', unique_indices=False,
                                            include_self=include_self)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_amax(self, device, dtype):
        # [FUNCTION DOC] Runs the base scatter_reduce_ test with amax for both include_self
        # values. For floating dtypes also scatters a hand-built source holding nan and infinities
        # into zeros and checks nan propagates, infinities win, and the untouched initial zero
        # survives only when include_self is true.
        for include_self in (True, False):
            self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                    is_scalar=False, reduction='amax', unique_indices=False,
                                    include_self=include_self)
            # simple test for nan/inf propagation
            if (dtype.is_floating_point):
                input = torch.zeros(3, device=device, dtype=dtype)
                src = torch.tensor([1, float('nan'), -float('inf'), -float('inf'), 2, float('inf')], device=device, dtype=dtype)
                idx = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
                input.scatter_reduce_(0, idx, src, 'amax', include_self=include_self)
                expected_result = torch.tensor([float('nan'), -float('inf'), float('inf')], device=device, dtype=dtype)
                if (include_self):
                    expected_result[1] = 0
                self.assertEqual(input, expected_result)


    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_amin(self, device, dtype):
        # [FUNCTION DOC] Same as the amax test but for amin: base comparison plus the nan and
        # infinity propagation check, where the initial zero survives (in the last slot) only when
        # include_self is true.
        for include_self in (True, False):
            self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                    is_scalar=False, reduction='amin', unique_indices=False,
                                    include_self=include_self)
            # simple test for nan/inf propagation
            if (dtype.is_floating_point):
                input = torch.zeros(3, device=device, dtype=dtype)
                src = torch.tensor([1, float('nan'), -2, -float('inf'), float('inf'), float('inf')], device=device, dtype=dtype)
                idx = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
                input.scatter_reduce_(0, idx, src, 'amin', include_self=include_self)
                expected_result = torch.tensor([float('nan'), -float('inf'), float('inf')], device=device, dtype=dtype)
                if (include_self):
                    expected_result[2] = 0
                self.assertEqual(input, expected_result)

    @onlyCPU
    @dtypes(torch.float32, torch.float64, torch.bfloat16, torch.float16)
    def test_scatter_expanded_index(self, device, dtype):
        # [FUNCTION DOC] CPU-only test that scatter_add and scatter_reduce (all five reductions,
        # both include_self values) give the same result for an expanded (stride zero) index as
        # for the same index made contiguous, over several input ranks.
        def helper(input_size, idx_size):
            # [FUNCTION DOC] HELPER: For one input size and one index length, builds a random
            # input and an index that is expanded along dim zero (with some indices masked to zero
            # to create empty rows), and compares the results of scatter_add and scatter_reduce
            # with expanded versus contiguous copies of the index.
            input = torch.randn(input_size, device=device).to(dtype=dtype)
            input2 = input.clone()

            shape = [1] * len(input_size)
            shape[0] = idx_size
            dim_size = input_size[0]
            idx = torch.randint(0, dim_size, shape)

            # The fast path on scatter when index is expanded
            # will depend on sorted index where the collected src indice
            # for each row in input will be mapped to rowptrs in a CSR format.
            # Create some empty rows by masking:
            mask = (idx > 1) * (idx < 4)
            idx[mask] = 0

            expanded_shape = input_size
            expanded_shape[0] = idx_size
            idx = idx.expand(expanded_shape)
            idx2 = idx.contiguous()
            src = torch.randn(expanded_shape, device=device).to(dtype=dtype)

            out = input.scatter_add(0, idx, src)
            out2 = input2.scatter_add(0, idx2, src)
            self.assertEqual(out, out2)

            for reduce in ["sum", "prod", "mean", "amax", "amin"]:
                for include_self in [True, False]:
                    out = input.scatter_reduce(0, idx, src, reduce=reduce, include_self=include_self)
                    out2 = input2.scatter_reduce(0, idx2, src, reduce=reduce, include_self=include_self)
                    self.assertEqual(out, out2)

        helper([50, 17], 100)
        helper([50, 1], 100)
        helper([50, 8, 7], 100)
        helper([50, 3, 4, 5], 100)

    @dtypes(torch.float32)
    def test_scatter_add_broadcasted_index_deterministic(self, device, dtype):
        # [FUNCTION DOC] Checks that deterministic-mode scatter_add_ with an index that is a
        # broadcast view of a one-dimensional index, along dims zero and one, equals the non-
        # deterministic result.
        for d in (0, 1):
            inp = torch.randn(3, 4, 5, device=device, dtype=dtype)
            idx_1d = torch.randint(3, (10,), device=device)
            src_shape = list(inp.shape)
            src_shape[d] = 10
            src = torch.randn(src_shape, device=device, dtype=dtype)
            idx_view_shape = [1] * inp.ndim
            idx_view_shape[d] = 10
            idx = idx_1d.view(idx_view_shape).expand(src_shape)
            ref = inp.clone().scatter_add_(d, idx, src)
            with DeterministicGuard(True):
                res = inp.clone().scatter_add_(d, idx, src)
            self.assertEqual(res, ref)


    @onlyCPU
    @dtypes(torch.float32, torch.float64, torch.bfloat16)
    def test_gather_expanded_index(self, device, dtype):
        # [FUNCTION DOC] CPU-only test of gather with expanded (stride zero) index: first a small
        # column index case, then random cases where the result with an expanded index must equal
        # the result with its contiguous copy, plus an unsqueezed-index case that the expanded-
        # index kernel cannot handle (referencing a known upstream issue) and so must fall back
        # correctly.
        # Test when index is [N, 1], which would have stride [1, 0]
        # should be excluded from the fast path when index ix expanded
        input = torch.arange(25).view(5, 5)
        input2 = input.to(dtype=dtype)

        idx = torch.arange(5).view(5, 1)
        out = torch.gather(input, 0, idx)
        out2 = torch.gather(input2, 0, idx)

        self.assertEqual(out.to(dtype=dtype), out2)

        def helper(input_size, idx_size):
            # [FUNCTION DOC] HELPER: Builds a random input and an expanded index along dim zero
            # and checks gather gives identical results for the expanded index and its contiguous
            # copy; then repeats with an index built by the nested unsqueeze helper whose size and
            # stride pattern is unsupported by the expanded-index fast path.
            input = torch.randn(input_size, device=device).to(dtype=dtype)
            input2 = input.clone()

            shape = [1] * len(input_size)
            shape[0] = idx_size
            dim_size = input_size[0]
            idx = torch.randint(0, dim_size, shape)

            # Test the fast path on gather when index is expanded
            expanded_shape = input_size
            expanded_shape[0] = idx_size
            idx = idx.expand(expanded_shape)
            idx2 = idx.contiguous()

            out = torch.gather(input, 0, idx)
            out2 = torch.gather(input2, 0, idx2)

            self.assertEqual(out, out2)

            # test unsqueezed index
            # expanded_index kernel can not handle the case:
            # the size > 1 and stride == 1 at a dimension.
            # for example: the index with size of [1, 8, 7],  stride of [1, 1, 0].
            # see https://github.com/pytorch/pytorch/issues/129093
            def unsqueeze_helper(idx, dim):
                # [FUNCTION DOC] HELPER: Recursively builds an index whose dimensions are added
                # with unsqueeze steps (the base case for two dims transposes a column vector),
                # producing an index with a size above one and stride one in some dimension.
                if dim == 2:
                    return idx.unsqueeze(1).t()
                else:
                    return unsqueeze_helper(idx, dim - 1).unsqueeze(dim - 1)

            idx = torch.randint(0, dim_size, (input.shape[1],))
            idx = unsqueeze_helper(idx, len(input_size))
            expanded_shape[0] = 1
            idx = idx.expand(expanded_shape)
            idx2 = idx.contiguous()
            out = torch.gather(input, 0, idx)
            out2 = torch.gather(input2, 0, idx2)
            self.assertEqual(out, out2)

        helper([50, 17], 100)
        helper([50, 1], 100)
        helper([50, 8, 7], 100)
        helper([50, 3, 4, 5], 100)

# ---------------------------------------------------------------------------
# CuTeDSL scatter_add override tests. Two surfaces:
#   (a) dispatch conds: call the eligibility predicates directly to verify
#       the routing matrix (dtypes, shapes, outer-strided, int32 index,
#       deterministic mode). No kernel execution.
#   (b) correctness: run torch.scatter_add on shapes the conds accept and
#       compare to a naive reference. Covers both TMA and vec-scatter
#       paths via the shape parameters.
# ---------------------------------------------------------------------------


def _expanded_idx(index_1d, trailing_shape):
    # [FUNCTION DOC] HELPER: Builds an index tensor for the CuTeDSL scatter_add override by
    # expanding a one-dimensional index with a new trailing dimension for each size in
    # trailing_shape, so every axis after the first has stride zero. Returns the expanded view.
    """Build the expanded-1D index (stride 0 on every axis > 0)."""
    idx = index_1d
    for s in trailing_shape:
        idx = idx.unsqueeze(-1).expand(*idx.shape, s)
    return idx


def _make_override_triple(
    M_out, M_src, shape_tail, *, dtype=torch.float32,
    idx_dtype=torch.int64, outer_pad_self=0, outer_pad_src=0,
):
    # [FUNCTION DOC] HELPER: Allocates CUDA operands for the expanded-one-dimensional scatter_add
    # pattern. Returns a tuple of the destination (M_out rows), the expanded index, the random
    # source (M_src rows) and the raw one-dimensional random index (values below M_out, dtype
    # idx_dtype). Trailing dimensions come from shape_tail. A nonzero outer_pad_self or
    # outer_pad_src allocates wider rows and slices them, giving a view whose row stride exceeds
    # its row width.
    """Build a (self, idx_2d, src, idx_1d) triple for the expanded-1D
    scatter_add pattern. ``outer_pad_*`` widens the allocated buffer's
    inner dim, then slices, producing a view with stride(-1)==1 but
    outer stride != prod(shape[1:]).
    """
    N = 1
    for s in shape_tail:
        N *= s
    if outer_pad_self:
        big = torch.zeros(M_out, N + outer_pad_self, device="cuda", dtype=dtype)
        self_t = big[:, :N].view(M_out, *shape_tail)
    else:
        self_t = torch.zeros(M_out, *shape_tail, device="cuda", dtype=dtype)
    if outer_pad_src:
        big = torch.randn(M_src, N + outer_pad_src, device="cuda", dtype=dtype)
        src = big[:, :N].view(M_src, *shape_tail)
    else:
        src = torch.randn(M_src, *shape_tail, device="cuda", dtype=dtype)
    index_1d = torch.randint(0, M_out, (M_src,), device="cuda", dtype=idx_dtype)
    return self_t, _expanded_idx(index_1d, shape_tail), src, index_1d


def _naive_scatter_add(self_t, idx_1d, src, alpha=1.0):
    # [FUNCTION DOC] HELPER: Reference scatter_add. Clones self_t in float32, adds alpha times
    # each source row to the destination row named by the matching entry of the one-dimensional
    # index using a Python loop, and returns the result cast back to the dtype of self_t. Works
    # for any stride layout.
    """Reference implementation. Works on any stride pattern."""
    ref = self_t.clone().float()
    src_f = src.float()
    for i, r in enumerate(idx_1d.tolist()):
        ref[r] += alpha * src_f[i]
    return ref.to(self_t.dtype)


def _override_tol(dtype):
    # [FUNCTION DOC] HELPER: Returns the comparison tolerance for a dtype: tight for float32 and a
    # loose value for every other dtype.
    return 1e-4 if dtype == torch.float32 else 0.5


def _misaligned_view(rows, cols, dtype):
    # [FUNCTION DOC] HELPER: Allocates an uninitialized CUDA tensor with one extra element and
    # returns a contiguous (rows, cols) view starting one element in, so its data pointer is
    # misaligned for 16-byte access. Raises AssertionError if the pointer turns out to be 16-byte
    # aligned.
    """Allocate (rows*cols + 1) elements and slice from index 1, producing
    a contiguous (rows, cols) view whose data_ptr is one element past a
    16B-aligned base."""
    t = torch.empty(rows * cols + 1, device="cuda", dtype=dtype)[1:].view(rows, cols)
    # Sanity check: if the allocator ever hands us a base such that the
    # +1-element slice is still 16B-aligned, the test silently passes
    # without exercising the misalignment path.
    if t.data_ptr() % 16 == 0:
        raise AssertionError(
            f"test bug: data_ptr() should be misaligned, got {t.data_ptr() % 16=}"
        )
    return t


def _outer_padded_view(rows, cols, dtype):
    # [FUNCTION DOC] HELPER: Returns a zero CUDA tensor view with the requested rows and cols
    # whose row stride is one larger than the width (a slice of a wider allocation).
    """Allocate (rows, cols + 1) zeros and slice the first `cols` columns,
    producing a view with stride(0) == cols + 1 (not cols)."""
    return torch.zeros(rows, cols + 1, device="cuda", dtype=dtype)[:, :cols]


# Joint matrix for the data_ptr / row-stride alignment cond checks. Each
# case names a misalignment site and what the cond should return. M_out /
# M_src / D are fixed; D=128 -> row_bytes = 128 * elem_size, which is
# 16B-aligned for fp32 and bf16, so any cond rejection here comes from
# the new data_ptr / row-stride checks (not the row_bytes check).
_M_OUT, _M_SRC, _D = 100, 50, 128


class _AlignmentCase(NamedTuple):
    # [CLASS DOC] HELPER: Named tuple describing one alignment test case: a label, which operand
    # and kind of misalignment (self or src, data pointer or row stride, or both pointers), the
    # dtype, and the expected TMA and vector-scatter eligibility. Read-only properties expose the
    # fixed shape (M_out, M_src, D) from the module constants.
    name: str
    # which buffer is misaligned: "self_dp", "src_dp", "both_dp",
    # "self_rs", "src_rs"
    site: str
    dtype: torch.dtype
    expected_tma: bool
    expected_vec: bool

    @property
    def M_out(self):
        # [FUNCTION DOC] HELPER: Returns the number of destination rows used by alignment cases.
        return _M_OUT

    @property
    def M_src(self):
        # [FUNCTION DOC] HELPER: Returns the number of source rows used by alignment cases.
        return _M_SRC

    @property
    def D(self):
        # [FUNCTION DOC] HELPER: Returns the row width (elements) used by alignment cases.
        return _D


_ALIGNMENT_CASES = [
    # data_ptr misalignment via storage_offset=1.
    _AlignmentCase("fp32_self_dp", "self_dp", torch.float32, False, True),
    _AlignmentCase("fp32_src_dp", "src_dp", torch.float32, False, False),
    _AlignmentCase("fp32_both_dp", "both_dp", torch.float32, False, False),
    _AlignmentCase("bf16_self_dp", "self_dp", torch.bfloat16, False, False),
    # Outer row-stride misalignment via shape (D+1) slice.
    _AlignmentCase("fp32_self_rs", "self_rs", torch.float32, False, True),
    _AlignmentCase("fp32_src_rs", "src_rs", torch.float32, False, False),
    _AlignmentCase("bf16_self_rs", "self_rs", torch.bfloat16, False, False),
]


def _build_alignment_case(case):
    # [FUNCTION DOC] HELPER: Builds the CUDA destination and source tensors for an alignment case,
    # making the operand named by case.site misaligned (via a one element offset, or via a padded
    # row stride) and the other one aligned. Raises ValueError for an unknown site. Returns the
    # pair.
    """Return (self, src) tensors for an alignment case."""
    dtype = case.dtype
    if case.site == "self_dp":
        self_t = _misaligned_view(case.M_out, case.D, dtype).zero_()
        src = torch.randn(case.M_src, case.D, device="cuda", dtype=dtype)
    elif case.site == "src_dp":
        self_t = torch.zeros(case.M_out, case.D, device="cuda", dtype=dtype)
        src = _misaligned_view(case.M_src, case.D, dtype).normal_()
    elif case.site == "both_dp":
        self_t = _misaligned_view(case.M_out, case.D, dtype).zero_()
        src = _misaligned_view(case.M_src, case.D, dtype).normal_()
    elif case.site == "self_rs":
        self_t = _outer_padded_view(case.M_out, case.D, dtype)
        src = torch.randn(case.M_src, case.D, device="cuda", dtype=dtype)
    elif case.site == "src_rs":
        self_t = torch.zeros(case.M_out, case.D, device="cuda", dtype=dtype)
        src = _outer_padded_view(case.M_src, case.D, dtype).normal_()
    else:
        raise ValueError(f"unknown site: {case.site}")
    return self_t, src


@unittest.skipUnless(TEST_CUDA, "needs CUDA")
@skipIfNoCuteDSL
class TestScatterAddOverrideConds(TestCase):
    # [CLASS DOC] CUDA tests (skipped without CUDA or CuTeDSL) of the dispatch eligibility
    # predicates for the CuTeDSL scatter_add kernels: each test builds operands, queries both the
    # TMA and vector-scatter predicates and checks the routing decision. No scatter kernel is run.
    """Unit tests for the dispatch predicates in
    torch._native.ops.scatter_add.cutedsl_impl."""

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class fixture: imports the cutedsl_impl module of torch._native, stores
        # it on the class as impl, and registers its dispatch overrides.
        super().setUpClass()
        from torch._native.ops.scatter_add import cutedsl_impl
        cls.impl = cutedsl_impl
        cutedsl_impl.register_to_dispatch()

    def _conds(self, self_t, idx, src, dim=0):
        # [FUNCTION DOC] HELPER: Returns a pair of booleans, whether the TMA and the vector-
        # scatter predicates accept the destination, index and source for the given dim (default
        # zero).
        # _is_*_supported take (self, dim, index, src); they build the
        # analysis iter internally and pattern-match its strides against
        # the kernel's expected layout.
        return (
            self.impl._is_tma_supported(self_t, dim, idx, src),
            self.impl._is_vec_scatter_supported(self_t, dim, idx, src),
        )

    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @parametrize("N", [128, 256, 512, 1024])
    def test_accepts_contig_supported_shapes(self, dtype, N):
        # [FUNCTION DOC] Checks both predicates accept contiguous operands with 16-byte aligned
        # rows, over three dtypes and four row widths.
        self_t, idx, src, _ = _make_override_triple(100, 50, (N,), dtype=dtype)
        tma, vec = self._conds(self_t, idx, src)
        self.assertTrue(tma)
        self.assertTrue(vec)

    def test_rejects_row_bytes_not_16_aligned(self):
        # [FUNCTION DOC] Checks that a float32 row width of 129 (row bytes not a multiple of 16)
        # is rejected by both predicates.
        # fp32 D=129 -> row_bytes=516, 516 % 16 != 0 -> TMA rejects
        # (cp.reduce.async.bulk requires 16-aligned gmem operands).
        # 129 % 4 != 0 -> vec also rejects.
        self_t, idx, src, _ = _make_override_triple(100, 50, (129,))
        self.assertEqual(self._conds(self_t, idx, src), (False, False))

    @parametrize("case", _ALIGNMENT_CASES, name_fn=lambda c: c.name)
    def test_alignment_cond_matrix(self, case):
        # [FUNCTION DOC] Parametrized over the alignment cases: builds the misaligned operands and
        # checks the pair of predicate answers equals the case expectation. Skips bfloat16 cases
        # before sm90, where the architecture gate would mask the alignment check. Seeds the RNG.
        # Joint coverage of the data_ptr / row-stride alignment checks
        # in `_alignment_contract_ok`. src always needs 16B alignment
        # (LDG.128 / TMA load); self/dst needs 16B for TMA but only 4B
        # for vec-scatter, so dtype changes the vec-scatter expectation
        # whenever the dst misalignment is sub-4B (bf16 storage_offset=1
        # is 2B-aligned; bf16 row-stride pad of 1 elem leaves a 2B
        # remainder).
        #
        # bf16 cases require sm_90+: pre-sm_90 vec-scatter and TMA both
        # gate bf16 unconditionally, which would mask the alignment cond
        # under test (False return value would come from the SM gate,
        # not _alignment_contract_ok).
        if case.dtype is torch.bfloat16 and not SM90OrLater:
            self.skipTest("bf16 alignment cond requires sm_90+")
        torch.manual_seed(0)
        self_t, src = _build_alignment_case(case)
        idx = _expanded_idx(
            torch.randint(0, case.M_out, (case.M_src,), device="cuda", dtype=torch.int64),
            (case.D,),
        )
        self.assertEqual(
            self._conds(self_t, idx, src), (case.expected_tma, case.expected_vec),
            msg=lambda msg: f"{msg}\n{case.name}: expected (TMA={case.expected_tma}, vec={case.expected_vec})",
        )

    def test_out_cond_rejects_misaligned_out(self):
        # [FUNCTION DOC] Checks that the predicates for the out variant reject a misaligned out
        # tensor even when self is aligned, since the kernel runs on out. Builds a bfloat16
        # misaligned out and asserts both the TMA and vector-scatter out conditions return false.
        # The .out impl runs the kernel on ``out``, not ``self`` (see
        # _make_impls.out_impl). A misaligned ``out`` with aligned
        # ``self`` must be rejected -- otherwise the kernel runs on
        # ``out``'s misaligned data_ptr and faults with
        # cudaErrorMisalignedAddress (the exact bug this diff prevents,
        # just on the .out variant).
        M_out, M_src, D = 100, 50, 128
        self_t = torch.zeros(M_out, D, device="cuda", dtype=torch.bfloat16)
        src = torch.randn(M_src, D, device="cuda", dtype=torch.bfloat16)
        # Misaligned out: one-element (2B) offset from a 16B boundary.
        # bf16 -> 2B-aligned, which fails both the TMA 16B and the
        # vec-scatter 4B destination alignment checks.
        out_mis = (
            torch.empty(M_out * D + 1, device="cuda", dtype=torch.bfloat16)[1:]
            .view(M_out, D)
        )
        self.assertNotEqual(
            out_mis.data_ptr() % 16, 0,
            msg=lambda msg: f"{msg}\ntest bug: out should be misaligned, got {out_mis.data_ptr() % 16=}",
        )
        idx = _expanded_idx(
            torch.randint(0, M_out, (M_src,), device="cuda", dtype=torch.int64),
            (D,),
        )
        self.assertFalse(
            self.impl._tma_out_cond(self_t, 0, idx, src, out=out_mis),
            msg=".out TMA cond accepted misaligned `out`",
        )
        self.assertFalse(
            self.impl._vs_out_cond(self_t, 0, idx, src, out=out_mis),
            msg=".out vec-scatter cond accepted misaligned `out`",
        )

    def test_tma_accepts_row_not_chunk_multiple(self):
        # [FUNCTION DOC] Checks a bfloat16 row of 528 bytes (16-aligned, not a multiple of the
        # chunk size) is accepted by both predicates.
        # bf16 D=264 -> row_bytes=528, 16-aligned but not a multiple of
        # chunk_bytes (512). TMA descriptor handles the partial final
        # chunk via OOB-clamp-to-zero; cond should accept.
        self_t, idx, src, _ = _make_override_triple(100, 50, (264,), dtype=torch.bfloat16)
        self.assertEqual(self._conds(self_t, idx, src), (True, True))

    def test_rejects_unsupported_dtype(self):
        # [FUNCTION DOC] Checks both predicates reject a float64 operand.
        self_t, idx, src, _ = _make_override_triple(100, 50, (128,), dtype=torch.float64)
        self.assertEqual(self._conds(self_t, idx, src), (False, False))

    def test_rejects_dim_nonzero_on_default_contig(self):
        # [FUNCTION DOC] Checks both predicates reject scatter along dim one for ordinary
        # contiguous operands, whose inner axis is then not packed.
        # Default-contiguous (M, N) tensor with dim=1 permutes to (N, M) with
        # outer stride 1 and inner stride N -- inner is not packed, so the
        # post-permute eligibility check still rejects.
        self_t, idx, src, _ = _make_override_triple(100, 50, (128,))
        self.assertEqual(self._conds(self_t, idx, src, dim=1), (False, False))

    def test_accepts_dim_after_permute(self):
        # [FUNCTION DOC] Checks both predicates accept scatter along dim one when the operands are
        # transposed views, which become contiguous after moving that dimension to the front.
        # User-facing tensors are (N, M)-shaped views of (M, N)-contiguous
        # backing storage. After ``movedim(1, 0)`` the layout becomes
        # (M, N)-contiguous, which is what the kernel expects.
        M_self, M_src, N = 100, 50, 128
        big_self = torch.zeros(M_self, N, device="cuda").transpose(0, 1)
        big_src = torch.randn(M_src, N, device="cuda").transpose(0, 1)
        idx_1d = torch.randint(0, M_self, (M_src,), device="cuda", dtype=torch.int64)
        # Index broadcast pattern along the inner axis of the permuted view
        # = N axis of the user's tensor.
        idx = idx_1d.unsqueeze(0).expand(N, M_src)
        self.assertEqual(self._conds(big_self, idx, big_src, dim=1), (True, True))

    def test_rejects_non_expanded_index(self):
        # [FUNCTION DOC] Checks both predicates reject a fully materialized two-dimensional index,
        # which has nonzero strides on every axis.
        # Materialized 2D index (not a broadcast view) has nonzero stride
        # on every axis -> the stride(i) == 0 check fails.
        self_t = torch.zeros(100, 128, device="cuda")
        src = torch.randn(50, 128, device="cuda")
        bad_idx = torch.randint(0, 100, (50, 128), device="cuda", dtype=torch.int64)
        self.assertEqual(self._conds(self_t, bad_idx, src), (False, False))

    def test_rejects_inner_non_contig(self):
        # [FUNCTION DOC] Checks both predicates reject permuted operands whose last axis is not
        # stride one.
        # permute(1, 0) -> trailing axis has stride 100 (not 1).
        big_self = torch.zeros(128, 100, device="cuda").permute(1, 0)
        big_src = torch.randn(128, 50, device="cuda").permute(1, 0)
        idx = _expanded_idx(
            torch.randint(0, 100, (50,), device="cuda", dtype=torch.int64), (128,)
        )
        self.assertEqual(self._conds(big_self, idx, big_src), (False, False))

    def test_accepts_outer_strided(self):
        # [FUNCTION DOC] Checks both predicates accept a destination whose rows are strided
        # (padded), asserting first that such a destination is indeed not contiguous.
        self_t, idx, src, _ = _make_override_triple(
            100, 50, (128,), outer_pad_self=64
        )
        self.assertFalse(self_t.is_contiguous())
        self.assertEqual(self._conds(self_t, idx, src), (True, True))

    @parametrize("idx_dtype", [torch.int32, torch.int64])
    def test_accepts_int32_and_int64_index(self, idx_dtype):
        # [FUNCTION DOC] Checks both predicates accept 32-bit and 64-bit index dtypes.
        self_t, idx, src, _ = _make_override_triple(
            100, 50, (128,), idx_dtype=idx_dtype
        )
        self.assertEqual(self._conds(self_t, idx, src), (True, True))

    def test_rejects_deterministic_mode(self):
        # [FUNCTION DOC] Under deterministic mode, checks the full dispatch-level conditions (not
        # the bare support checks) for TMA and vector-scatter both return false.
        # The determinism gate lives in _base_cond_ok (the shared
        # prelude), not the per-path support_check. Call the wrapped
        # cond to exercise the full dispatch-layer predicate.
        self_t, idx, src, _ = _make_override_triple(100, 50, (128,))
        with DeterministicGuard(True):
            self.assertFalse(self.impl._tma_cond(self_t, 0, idx, src))
            self.assertFalse(self.impl._vs_cond(self_t, 0, idx, src))

    def test_rejects_empty_self(self):
        # [FUNCTION DOC] Checks both predicates reject a destination with zero rows and a nonempty
        # source, so that the standard error path reports the out of range index.
        # self.shape[0] == 0 with a non-empty src would have every index
        # out of range. Cond must reject so aten raises the index error
        # instead of our kernel silently writing OOB.
        self_t = torch.zeros(0, 128, device="cuda", dtype=torch.float32)
        src = torch.randn(5, 128, device="cuda", dtype=torch.float32)
        idx = _expanded_idx(
            torch.zeros(5, device="cuda", dtype=torch.int64), (128,)
        )
        self.assertEqual(self._conds(self_t, idx, src), (False, False))


@unittest.skipUnless(TEST_CUDA, "needs CUDA")
@skipIfNoCuteDSL
class TestScatterAddOverrideCorrectness(TestCase):
    # [CLASS DOC] CUDA end-to-end tests (skipped without CUDA or CuTeDSL) running scatter_add on
    # shapes the CuTeDSL predicates accept, and comparing with the naive reference.
    """End-to-end: torch.scatter_add vs a naive reference on shapes the
    conds accept (TMA-eligible and vec-scatter-only)."""

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class fixture: imports cutedsl_impl and registers its dispatch overrides.
        super().setUpClass()
        from torch._native.ops.scatter_add import cutedsl_impl
        cutedsl_impl.register_to_dispatch()

    @parametrize("dtype,shape_tail", [
        subtest((torch.float32, (128,)), name="fp32_N128"),       # TMA, single chunk
        subtest((torch.float32, (1024,)), name="fp32_N1024"),     # TMA, full chunks
        subtest((torch.bfloat16, (128,)), name="bf16_N128"),
        subtest((torch.bfloat16, (1024,)), name="bf16_N1024"),
        subtest((torch.float16, (256,)), name="fp16_N256"),
        subtest((torch.float32, (132,)), name="fp32_N132_partial"),   # TMA partial final chunk
        subtest((torch.bfloat16, (264,)), name="bf16_N264_partial"),  # TMA partial final chunk
        subtest((torch.float32, (4, 32)), name="fp32_3d"),        # nD flatten
    ])
    def test_matches_reference(self, dtype, shape_tail):
        # [FUNCTION DOC] Parametrized over dtype and trailing shape (including partial chunk and
        # multi-dimensional tails): runs the functional scatter_add on seeded random operands and
        # compares with the naive reference using the per-dtype tolerance.
        torch.manual_seed(0)
        self_t, idx, src, idx_1d = _make_override_triple(200, 100, shape_tail, dtype=dtype)
        got = torch.scatter_add(self_t, 0, idx, src)
        ref = _naive_scatter_add(
            self_t.reshape(self_t.shape[0], -1),
            idx_1d,
            src.reshape(src.shape[0], -1),
        ).reshape(self_t.shape)
        tol = _override_tol(dtype)
        self.assertEqual(got, ref, atol=tol, rtol=tol)

    @parametrize("path", ["tma", "vec"])
    def test_inner_dim_change_no_recompile(self, path):
        # [FUNCTION DOC] For each path (tma needs sm90 or later, else skipped), drives the kernel
        # host entry directly over five inner sizes after clearing the compile cache, checking
        # numerics each time, that cache misses stay at most one, and that exactly one in-memory
        # entry serves every size; proves the inner dimension is a runtime argument and not part
        # of the cache key.
        # The kernels take the slice extent N as a runtime arg, so a single
        # compile must serve every inner-dim size. Call the kernel host entry
        # at several N and assert the compile cache runs exactly one real
        # compile (misses advances once, then only hits). We drive the host
        # entry directly rather than through torch.scatter_add: TMA is
        # registered first and strictly narrower, so on sm_90+ it would win
        # every contiguous shape and the vec cache would never move.
        from torch._native.ops.scatter_add import tma_kernel, vec_scatter_kernel
        if path == "tma":
            if not SM90OrLater:
                self.skipTest("TMA path requires sm_90+")
            compile_fn = tma_kernel._compile_tma_scatter
            run = tma_kernel.tma_scatter_add_into
        else:
            compile_fn = vec_scatter_kernel._compile_vec_scatter
            run = vec_scatter_kernel.vec_scatter_add_into

        torch.manual_seed(0)
        # Same dtype and contiguity across all sizes so only N varies.
        Ns = [128, 256, 512, 1024, 2048]
        compile_fn.cache_clear()
        for N in Ns:
            self_t, _, src, idx_1d = _make_override_triple(200, 100, (N,))
            out = self_t.clone()
            run(out, idx_1d, src)
            self.assertEqual(out, _naive_scatter_add(self_t, idx_1d, src),
                             atol=1e-4, rtol=1e-4)
            misses = compile_fn.cache_info().misses
            # Every N after the first must be served from cache: misses
            # tops out at 1 (0 if the .o was already warm on disk). If N
            # were still in the key, misses would climb with each size.
            self.assertLessEqual(misses, 1, msg=f"recompiled at N={N}")
        # The decisive invariant, independent of on-disk cache state: one
        # in-memory entry serves every N. If N were in the key there'd be
        # len(Ns) entries.
        self.assertEqual(compile_fn.cache_info().currsize, 1)

    @parametrize("variant", ["functional", "out", "inplace"])
    def test_op_variants(self, variant):
        # [FUNCTION DOC] Parametrized over functional, out and in-place variants of scatter_add:
        # checks each result equals the naive reference, and that the in-place call returns the
        # same object.
        torch.manual_seed(0)
        self_t, idx, src, idx_1d = _make_override_triple(200, 100, (128,))
        ref = _naive_scatter_add(self_t, idx_1d, src)
        if variant == "functional":
            got = torch.scatter_add(self_t, 0, idx, src)
        elif variant == "out":
            got = torch.empty_like(self_t)
            torch.scatter_add(self_t, 0, idx, src, out=got)
        else:
            got = self_t.clone()
            ret = got.scatter_add_(0, idx, src)
            self.assertIs(ret, got)
        self.assertEqual(got, ref)

    @parametrize("outer_pad_self,outer_pad_src", [(64, 0), (0, 64), (64, 64)])
    def test_outer_strided(self, outer_pad_self, outer_pad_src):
        # [FUNCTION DOC] Parametrized over which operand has padded outer stride; checks
        # functional scatter_add equals the naive reference.
        torch.manual_seed(0)
        self_t, idx, src, idx_1d = _make_override_triple(
            200, 100, (128,),
            outer_pad_self=outer_pad_self, outer_pad_src=outer_pad_src,
        )
        got = torch.scatter_add(self_t, 0, idx, src)
        ref = _naive_scatter_add(self_t, idx_1d, src)
        self.assertEqual(got, ref)

    def test_repeated_index_accumulates(self):
        # [FUNCTION DOC] Sends all indices to row zero and checks that row equals the column-wise
        # sum of the source and all other rows remain zero.
        # All indices target row 0: result[0] == sum(src).
        torch.manual_seed(0)
        src = torch.randn(64, 128, device="cuda")
        self_t = torch.zeros(10, 128, device="cuda")
        idx = _expanded_idx(
            torch.zeros(64, device="cuda", dtype=torch.int64), (128,)
        )
        got = torch.scatter_add(self_t, 0, idx, src)
        self.assertEqual(got[0], src.sum(0), atol=1e-3, rtol=1e-3)
        self.assertEqual(got[1:].abs().sum().item(), 0)

    def test_matches_reference_dim1_after_permute(self):
        # [FUNCTION DOC] Checks scatter_add along dim one on transposed views equals the naive
        # reference computed after moving that dimension to the front and back.
        # Build operands whose scatter axis (dim=1) becomes axis 0 after
        # ``movedim``, landing on the kernel-eligible (M, N)-contiguous
        # layout. Concretely: take a contiguous (M, N) buffer and view it
        # as (N, M) via transpose; scatter_add along dim=1 then sees
        # exactly the original buffer after movedim(1, 0).
        torch.manual_seed(0)
        M_out, M_src, N = 200, 100, 128
        self_t = torch.zeros(M_out, N, device="cuda", dtype=torch.float32).transpose(0, 1)
        src = torch.randn(M_src, N, device="cuda", dtype=torch.float32).transpose(0, 1)
        idx_1d = torch.randint(0, M_out, (M_src,), device="cuda", dtype=torch.int64)
        idx = idx_1d.unsqueeze(0).expand(N, M_src)
        got = torch.scatter_add(self_t, 1, idx, src)
        # Reference: same scatter with dim=0 on the (M, N) view, then
        # transpose back to match self_t's layout.
        ref = _naive_scatter_add(
            self_t.movedim(1, 0), idx_1d, src.movedim(1, 0)
        ).movedim(0, 1)
        self.assertEqual(got, ref, atol=1e-4, rtol=1e-4)

    def test_empty_index_is_noop(self):
        # [FUNCTION DOC] Checks empty index and source leave the input unchanged for the
        # functional, out and in-place variants (no kernel launch failure).
        # End-to-end: empty index/src must produce the input unchanged
        # without any kernel-launch crash. In current TI behavior,
        # empty 2D shapes coalesce to ndim=1 so eligibility rejects and
        # aten handles the no-op. The numel==0 guard in _run is a
        # belt-and-braces defense (mirrors aten's pattern) for the case
        # where eligibility passes but the iter has no work.
        torch.manual_seed(0)
        self_t = torch.randn(200, 128, device="cuda", dtype=torch.float32)
        idx = torch.empty(0, 128, device="cuda", dtype=torch.int64)
        src = torch.empty(0, 128, device="cuda", dtype=torch.float32)
        for variant in ("functional", "out", "inplace"):
            ref = self_t.clone()
            if variant == "functional":
                got = torch.scatter_add(self_t, 0, idx, src)
            elif variant == "out":
                got = torch.empty_like(self_t)
                torch.scatter_add(self_t, 0, idx, src, out=got)
            else:
                got = self_t.clone()
                got.scatter_add_(0, idx, src)
            self.assertEqual(got, ref)

    def test_deterministic_mode_uses_aten(self):
        # [FUNCTION DOC] Under deterministic mode checks scatter_add falls back to the reference
        # implementation and still matches the naive result.
        # Under use_deterministic_algorithms(True), the cond rejects and
        # scatter_add falls through to aten's deterministic path. Verify
        # end-to-end numerics match a naive reference.
        torch.manual_seed(0)
        self_t, idx, src, idx_1d = _make_override_triple(200, 100, (128,))
        ref = _naive_scatter_add(self_t, idx_1d, src)
        with DeterministicGuard(True):
            got = torch.scatter_add(self_t, 0, idx, src)
        self.assertEqual(got, ref)

    @parametrize("case", _ALIGNMENT_CASES, name_fn=lambda c: c.name)
    def test_alignment_correctness(self, case):
        # [FUNCTION DOC] Regression for misaligned-address crashes: for each alignment case
        # (bfloat16 skipped before sm90) checks the dispatch predicates equal the expected values
        # and that the scatter_add result equals the naive reference.
        # End-to-end regression for the SEV-class crash where a
        # misaligned data_ptr / row stride bypassed the cond and faulted
        # in the TMA / vec-scatter kernel (cudaErrorMisalignedAddress
        # from cp.reduce.async.bulk or LDG.128). With the alignment cond
        # in place, every case here either runs via vec-scatter (when
        # the relaxed dst-only 4B contract is met -- expected_vec=True)
        # or falls through to aten; both must produce correct numerics.
        from torch._native.ops.scatter_add import cutedsl_impl

        # See test_alignment_cond_matrix.
        if case.dtype is torch.bfloat16 and not SM90OrLater:
            self.skipTest("bf16 alignment cond requires sm_90+")
        torch.manual_seed(0)
        self_t, src = _build_alignment_case(case)
        idx_1d = torch.randint(0, case.M_out, (case.M_src,), device="cuda", dtype=torch.int64)
        idx = _expanded_idx(idx_1d, (case.D,))

        # Confirm the cond observed in the unit test matches what the
        # dispatch-layer cond returns end-to-end (no skew between
        # eligibility check and kernel execution).
        self.assertEqual(
            cutedsl_impl._is_tma_supported(self_t, 0, idx, src), case.expected_tma,
        )
        self.assertEqual(
            cutedsl_impl._is_vec_scatter_supported(self_t, 0, idx, src), case.expected_vec,
        )

        ref = _naive_scatter_add(self_t, idx_1d, src)
        got = torch.scatter_add(self_t, 0, idx, src)
        tol = _override_tol(case.dtype)
        self.assertEqual(got, ref, atol=tol, rtol=tol, msg=case.name)

    def test_misaligned_out_correctness(self):
        # [FUNCTION DOC] Regression for the out variant with a misaligned out tensor: the result
        # must fall back and match the reference, and the returned object is the out tensor
        # itself.
        # End-to-end regression for the .out variant: when ``out`` is
        # misaligned the .out cond must reject so the call falls through
        # to aten (the kernel runs on ``out``, so a misaligned ``out``
        # would otherwise fault with cudaErrorMisalignedAddress -- same
        # SEV-class bug as the functional path, just on .out).
        torch.manual_seed(0)
        M_out, M_src, D = 200, 100, 128
        self_t = torch.zeros(M_out, D, device="cuda", dtype=torch.bfloat16)
        src = torch.randn(M_src, D, device="cuda", dtype=torch.bfloat16)
        # bf16 -> 2B-aligned data_ptr; fails both 16B (TMA) and 4B
        # (vec-scatter) destination alignment.
        out_mis = _misaligned_view(M_out, D, torch.bfloat16)
        self.assertNotEqual(out_mis.data_ptr() % 16, 0)
        idx_1d = torch.randint(0, M_out, (M_src,), device="cuda", dtype=torch.int64)
        idx = _expanded_idx(idx_1d, (D,))

        ref = _naive_scatter_add(self_t, idx_1d, src)
        ret = torch.scatter_add(self_t, 0, idx, src, out=out_mis)
        self.assertIs(ret, out_mis)
        tol = _override_tol(torch.bfloat16)
        self.assertEqual(out_mis, ref, atol=tol, rtol=tol)

    @parametrize("dtype,N", [
        # chunk_bytes < 128 (small D)
        subtest((torch.float32, 8), name="fp32_N8_cb32"),
        subtest((torch.float32, 16), name="fp32_N16_cb64"),
        subtest((torch.bfloat16, 8), name="bf16_N8_cb16"),
        subtest((torch.bfloat16, 16), name="bf16_N16_cb32"),
        # 128 < chunk_bytes < 512 but not a multiple of 128 -- stage 1
        # also misaligns here without the stage-stride padding.
        subtest((torch.float32, 36), name="fp32_N36_cb144"),
        subtest((torch.bfloat16, 72), name="bf16_N72_cb144"),
    ])
    @unittest.skipUnless(SM90OrLater, "TMA path requires sm_90+")
    def test_smem_stage_alignment_multi_iter(self, dtype, N):
        # [FUNCTION DOC] Requires sm90 or later and parametrized over small or non-128-aligned
        # chunk sizes: uses more source rows than the grid cap so a block loops at least twice,
        # asserts the TMA predicate accepts, then checks scatter_add equals the naive reference.
        # Guards against misalignment of the second pipeline stage in shared memory.
        # Regression for TMA stage-1 smem misalignment: stage 1 of the
        # 2-stage pipeline buffer lands at offset chunk_bytes, so any
        # chunk_bytes that isn't 128B-aligned (small D < 128, or D's
        # row_bytes not a multiple of 128) makes cp.async.bulk fault
        # with cudaErrorMisalignedAddress the first time a CTA writes
        # stage 1. Stage 1 is only reached when a CTA does >= 2
        # iterations, i.e. M_src > grid_x. _plan_grid caps grid_x at
        # sm*64, so M_src must exceed that to expose the bug.
        from torch._native.ops.scatter_add import cutedsl_impl

        torch.manual_seed(0)
        sm = torch.cuda.get_device_properties(0).multi_processor_count
        M_out, M_src = 1024, sm * 64 + 256
        self_t, idx, src, idx_1d = _make_override_triple(
            M_out, M_src, (N,), dtype=dtype
        )
        # Assert the TMA path is the one that actually runs -- otherwise
        # the test would silently regress to vec-scatter (which has no
        # smem stage stride and can't reproduce this bug).
        self.assertTrue(
            cutedsl_impl._is_tma_supported(self_t, 0, idx, src),
            msg="test bug: TMA cond rejected; this test would not exercise the regressed path",
        )
        ref = _naive_scatter_add(self_t, idx_1d, src)
        got = torch.scatter_add(self_t, 0, idx, src)
        torch.cuda.synchronize()
        tol = _override_tol(dtype)
        self.assertEqual(got, ref, atol=tol, rtol=tol)



instantiate_parametrized_tests(TestScatterAddOverrideConds)
instantiate_parametrized_tests(TestScatterAddOverrideCorrectness)


# Generic Device Test Framework instantiation, see
#   https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests
#   for details.
instantiate_device_type_tests(TestScatterGather, globals())

if __name__ == '__main__':
    run_tests()
