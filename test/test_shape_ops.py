# Owner(s): ["module: tests"]
# [MODULE DOC] Device-generic tests for shape-manipulation ops: unbind, tolist, movedim,
# diag/diagonal/trace, clamp/clip, flip family, rot90, nonzero, sparse_dim/dense_dim and unfold.
# Many tests compare against NumPy. Instantiates per-device classes at import time.

import random
import unittest
import warnings
from functools import partial
from itertools import chain, combinations, permutations, product

import numpy as np

import torch
from torch import nan
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    dtypesIfXPU,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyAccelerator,
    onlyCPU,
)
from torch.testing._internal.common_dtype import (
    all_passthru_types,
    all_passthru_types_and,
    all_types,
    all_types_and,
    all_types_and_complex_and,
)
from torch.testing._internal.common_utils import (
    IS_JETSON,
    run_tests,
    skipIfTorchDynamo,
    TEST_PRIVATEUSE1_DEVICE_TYPE,
    TestCase,
    torch_to_numpy_dtype_dict,
)


# TODO: replace with make_tensor
def _generate_input(shape, dtype, device, with_extremal):
    # [FUNCTION DOC] HELPER: Builds a random test tensor of the given shape, dtype and device.
    # An empty shape tuple gives an empty tensor. Floating/complex dtypes get scaled normal
    # values with roughly half the entries zeroed, and when with_extremal is set some entries
    # become nan, inf and -inf. Bool gets random True entries; other dtypes get random ints in
    # the range 15 to 99. bfloat16 is generated in float32 then cast.
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around torch.randn not being implemented for bfloat16
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(
                    30, 100
                )
            x[torch.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[torch.randn(*shape) > 0.5] = float("nan")
                x[torch.randn(*shape) > 0.5] = float("inf")
                x[torch.randn(*shape) > 0.5] = float("-inf")
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex("nan")
                x[torch.randn(*shape) > 0.5] = complex("inf")
                x[torch.randn(*shape) > 0.5] = complex("-inf")
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x


class TestShapeOps(TestCase):
    # [CLASS DOC] Device-generic test case covering shape-related ops. Methods take a device
    # argument (and a dtype when decorated) supplied by device-type instantiation; helper methods
    # generate random shapes, clamp baselines and NumPy comparisons.
    # TODO: update to work on CUDA, too
    @onlyCPU
    def test_unbind(self, device):
        # [FUNCTION DOC] CPU only: unbinds a random 4-d tensor along every dim via function and
        # method forms, checks the number of slices equals the dim size and that slices match
        # select.
        # KNOWN ISSUE: the slice-comparison loop iterates up to the dim index rather than the dim
        # size, so it checks only some of the slices.
        x = torch.rand(2, 3, 4, 5)
        for dim in range(4):
            res = torch.unbind(x, dim)
            res2 = x.unbind(dim)
            self.assertEqual(x.size(dim), len(res))
            self.assertEqual(x.size(dim), len(res2))
            for i in range(dim):
                self.assertEqual(x.select(dim, i), res[i])
                self.assertEqual(x.select(dim, i), res2[i])

    # TODO: update to work on CUDA, too?
    @skipIfTorchDynamo("TorchDynamo fails with an unknown error")
    @onlyCPU
    def test_tolist(self, device):
        # [FUNCTION DOC] CPU only: checks tolist round-trips for empty, 1-d, 2-d tensors, a float
        # storage, and a non-contiguous 3-d slice (which is asserted non-contiguous first).
        list0D = []
        tensor0D = torch.tensor(list0D)
        self.assertEqual(tensor0D.tolist(), list0D)

        table1D = [1.0, 2.0, 3.0]
        tensor1D = torch.tensor(table1D)
        storage = torch.Storage(table1D)
        self.assertEqual(tensor1D.tolist(), table1D)
        self.assertEqual(storage.tolist(), table1D)
        self.assertEqual(tensor1D.tolist(), table1D)
        self.assertEqual(storage.tolist(), table1D)

        table2D = [[1, 2], [3, 4]]
        tensor2D = torch.tensor(table2D)
        self.assertEqual(tensor2D.tolist(), table2D)

        tensor3D = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        tensorNonContig = tensor3D.select(1, 1)
        self.assertFalse(tensorNonContig.is_contiguous())
        self.assertEqual(tensorNonContig.tolist(), [[3, 4], [7, 8]])

    @dtypes(torch.int64, torch.float, torch.complex128)
    def test_movedim_invalid(self, device, dtype):
        # [FUNCTION DOC] Checks movedim and its alias moveaxis raise errors: IndexError for
        # out-of-range source or destination, RuntimeError for length mismatch between source and
        # destination, and for repeated dims in either side. Uses a random 4-d tensor.
        shape = self._rand_shape(4, min_size=5, max_size=10)
        x = _generate_input(shape, dtype, device, False)

        for fn in [torch.movedim, torch.moveaxis]:
            # Invalid `source` and `destination` dimension
            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 5, 0)

            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 0, 5)

            # Mismatch in size of `source` and `destination`
            with self.assertRaisesRegex(
                RuntimeError, "movedim: Invalid source or destination dims:"
            ):
                fn(x, (1, 0), (0,))

            with self.assertRaisesRegex(
                RuntimeError, "movedim: repeated dim in `source`"
            ):
                fn(x, (0, 0), (0, 1))

            with self.assertRaisesRegex(
                RuntimeError, "movedim: repeated dim in `source`"
            ):
                fn(x, (0, 1, 0), (0, 1, 2))

            with self.assertRaisesRegex(
                RuntimeError, "movedim: repeated dim in `destination`"
            ):
                fn(x, (0, 1), (1, 1))

            with self.assertRaisesRegex(
                RuntimeError, "movedim: repeated dim in `destination`"
            ):
                fn(x, (0, 1, 2), (1, 0, 1))

    @dtypes(torch.int64, torch.float, torch.complex128)
    def test_movedim(self, device, dtype):
        # [FUNCTION DOC] Compares movedim and moveaxis against NumPy for 0 to 4 dims. Covers
        # integer source/destination pairs and random-length permutation sequences, randomly
        # rewriting some dims as negative; also same-position moves and empty sequences on a fixed
        # 5-d tensor.
        for fn in [torch.moveaxis, torch.movedim]:
            for nd in range(5):
                shape = self._rand_shape(nd, min_size=5, max_size=10)
                x = _generate_input(shape, dtype, device, with_extremal=False)
                for random_negative in [True, False]:
                    for src_dim, dst_dim in permutations(range(nd), r=2):
                        random_prob = random.random()

                        if random_negative and random_prob > 0.66:
                            src_dim = src_dim - nd
                        elif random_negative and random_prob > 0.33:
                            dst_dim = dst_dim - nd
                        elif random_negative:
                            src_dim = src_dim - nd
                            dst_dim = dst_dim - nd

                        # Integer `source` and `destination`
                        torch_fn = partial(fn, source=src_dim, destination=dst_dim)
                        np_fn = partial(
                            np.moveaxis, source=src_dim, destination=dst_dim
                        )
                        self.compare_with_numpy(
                            torch_fn, np_fn, x, device=None, dtype=None
                        )

                    if nd == 0:
                        continue

                    def make_index_negative(sequence, idx):
                        # [FUNCTION DOC] HELPER: Intended to turn one entry of a sequence into its
                        # negative dim form.
                        # KNOWN ISSUE: it ignores its idx argument, reads the enclosing random_idx
                        # instead, and returns the enclosing src_sequence rather than the modified
                        # copy, so the destination case yields the source sequence unchanged (and
                        # the negative-dim edit is lost).
                        sequence = list(sequence)
                        sequence[random_idx] = sequence[random_idx] - nd
                        return tuple(src_sequence)

                    for src_sequence in permutations(
                        range(nd), r=random.randint(1, nd)
                    ):
                        # Sequence `source` and `destination`
                        dst_sequence = tuple(
                            random.sample(range(nd), len(src_sequence))
                        )

                        # Randomly change a dim to a negative dim representation of itself.
                        random_prob = random.random()
                        if random_negative and random_prob > 0.66:
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            src_sequence = make_index_negative(src_sequence, random_idx)
                        elif random_negative and random_prob > 0.33:
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            dst_sequence = make_index_negative(dst_sequence, random_idx)
                        elif random_negative:
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            dst_sequence = make_index_negative(dst_sequence, random_idx)
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            src_sequence = make_index_negative(src_sequence, random_idx)

                        torch_fn = partial(
                            fn, source=src_sequence, destination=dst_sequence
                        )
                        np_fn = partial(
                            np.moveaxis, source=src_sequence, destination=dst_sequence
                        )
                        self.compare_with_numpy(
                            torch_fn, np_fn, x, device=None, dtype=None
                        )

            # Move dim to same position
            x = torch.randn(2, 3, 5, 7, 11)
            torch_fn = partial(fn, source=(0, 1), destination=(0, 1))
            np_fn = partial(np.moveaxis, source=(0, 1), destination=(0, 1))
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

            torch_fn = partial(fn, source=1, destination=1)
            np_fn = partial(np.moveaxis, source=1, destination=1)
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

            # Empty Sequence
            torch_fn = partial(fn, source=(), destination=())
            np_fn = partial(np.moveaxis, source=(), destination=())
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

    @dtypes(torch.float, torch.bool)
    def test_diag(self, device, dtype):
        # [FUNCTION DOC] Builds a random 100x100 matrix (bool or float) and checks diag with the
        # out argument gives the same result as the functional call.
        if dtype is torch.bool:
            x = torch.rand(100, 100, device=device) >= 0.5
        else:
            x = torch.rand(100, 100, dtype=dtype, device=device)

        res1 = torch.diag(x)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.diag(x, out=res2)
        self.assertEqual(res1, res2)

    def test_diagonal(self, device):
        # [FUNCTION DOC] Checks diagonal equals diag of a random 100x100 matrix, with default and
        # offset 17.
        x = torch.randn((100, 100), device=device)
        result = torch.diagonal(x)
        expected = torch.diag(x)
        self.assertEqual(result, expected)

        x = torch.randn((100, 100), device=device)
        result = torch.diagonal(x, 17)
        expected = torch.diag(x, 17)
        self.assertEqual(result, expected)

    @onlyCPU
    @dtypes(torch.float)
    def test_diagonal_multidim(self, device, dtype):
        # [FUNCTION DOC] CPU only: compares diagonal of a random 4-d tensor with NumPy for several
        # offset/dim argument sets (shape and values), and for a permuted non-contiguous input.
        x = torch.randn(10, 11, 12, 13, dtype=dtype, device=device)
        xn = x.numpy()
        for args in [(2, 2, 3), (2,), (-2, 1, 2), (0, -2, -1)]:
            result = torch.diagonal(x, *args)
            expected = xn.diagonal(*args)
            self.assertEqual(expected.shape, result.shape)
            self.assertEqual(expected, result)
        # test non-contiguous
        xp = x.permute(1, 2, 3, 0)
        result = torch.diagonal(xp, 0, -2, -1)
        expected = xp.numpy().diagonal(0, -2, -1)
        self.assertEqual(expected.shape, result.shape)
        self.assertEqual(expected, result)

    @dtypes(*all_types())
    @dtypesIfCUDA(*all_types_and(torch.half))
    @dtypesIfXPU(*all_types_and(torch.half))
    def test_trace(self, device, dtype):
        # [FUNCTION DOC] Checks trace against NumPy trace (using the dtype of a sum) for several
        # rectangular and square integer/float matrices.
        def test(shape):
            # [FUNCTION DOC] HELPER: Makes a random tensor of the given shape and asserts its trace
            # matches the NumPy reference computed on the CPU copy.
            tensor = make_tensor(shape, dtype=dtype, device=device, low=-9, high=9)
            expected_dtype = tensor.sum().dtype
            expected_dtype = torch_to_numpy_dtype_dict[expected_dtype]

            result = np.trace(tensor.cpu().numpy(), dtype=expected_dtype)
            expected = torch.tensor(result, device=device)
            self.assertEqual(tensor.trace(), expected)

        shapes = (
            [10, 1],
            [1, 10],
            [100, 100],
            [20, 100],
            [100, 20],
        )
        for shape in shapes:
            test(shape)

    def generate_clamp_baseline(self, device, dtype, *, min_vals, max_vals, with_nans):
        # [FUNCTION DOC] HELPER: Returns a random 100 element tensor in the range -25 to 25 cast to
        # dtype (optionally with random entries set to nan) and the NumPy-computed clip of it with
        # the given min and max (tensors are converted to NumPy first).
        """
        Creates a random tensor for a given device and dtype, and computes the expected clamped
        values given the min_vals and/or max_vals.
        If with_nans is provided, then some values are randomly set to nan.
        """
        X = torch.rand(100, device=device).mul(50).add(-25)  # uniform in [-25, 25]
        X = X.to(dtype)
        if with_nans:
            mask = torch.randint(0, 2, X.shape, dtype=torch.bool, device=device)
            X[mask] = nan

        if isinstance(min_vals, torch.Tensor):
            min_vals = min_vals.cpu().numpy()

        if isinstance(max_vals, torch.Tensor):
            max_vals = max_vals.cpu().numpy()

        # Use NumPy implementation as reference
        X_clamped = torch.tensor(
            np.clip(X.cpu().numpy(), a_min=min_vals, a_max=max_vals), device=device
        )
        return X, X_clamped

    # Tests clamp and its alias, clip
    @dtypes(torch.int64, torch.float32)
    def test_clamp(self, device, dtype):
        # [FUNCTION DOC] Checks clamp, clip, and their method and in-place variants against the
        # NumPy baseline for all min/max combinations except both None; also the out variant for
        # the two function forms. Inputs are cloned so in-place ops do not alter the baseline
        # input.
        op_list = (
            torch.clamp,
            torch.Tensor.clamp,
            torch.Tensor.clamp_,
            torch.clip,
            torch.Tensor.clip,
            torch.Tensor.clip_,
        )

        # min/max argument product
        args = product((-10, None), (10, None))

        for op in op_list:
            for min_val, max_val in args:
                if min_val is None and max_val is None:
                    continue

                X, Y_expected = self.generate_clamp_baseline(
                    device, dtype, min_vals=min_val, max_vals=max_val, with_nans=False
                )

                # Test op
                X1 = X.clone()  # So that the in-place ops do not change X
                Y_actual = op(X1, min_val, max_val)
                self.assertEqual(Y_expected, Y_actual)

                # Test op-out behavior (out does not exist for method versions)
                if op in (torch.clamp, torch.clip):
                    Y_out = torch.empty_like(X)
                    op(X, min=min_val, max=max_val, out=Y_out)
                    self.assertEqual(Y_expected, Y_out)

    def test_clamp_propagates_nans(self, device):
        # [FUNCTION DOC] Same op variants and bound combinations as the clamp test, but with nan
        # values in the input; checks the positions of nan in the result match the reference.
        op_list = (
            torch.clamp,
            torch.Tensor.clamp,
            torch.Tensor.clamp_,
            torch.clip,
            torch.Tensor.clip,
            torch.Tensor.clip_,
        )

        # min/max argument product
        args = product((-10, None), (10, None))

        for op in op_list:
            for min_val, max_val in args:
                if min_val is None and max_val is None:
                    continue

                X, Y_expected = self.generate_clamp_baseline(
                    device,
                    torch.float,
                    min_vals=min_val,
                    max_vals=max_val,
                    with_nans=True,
                )
                Y_expected = torch.isnan(Y_expected)

                # Test op
                X1 = X.clone()  # So that the in-place ops do not change X
                Y_actual = op(X1, min_val, max_val)
                self.assertEqual(Y_expected, torch.isnan(Y_actual))

                # Test op-out behavior (out does not exist for method versions)
                if op in (torch.clamp, torch.clip):
                    Y_out = torch.empty_like(X)
                    op(X, min_val, max_val, out=Y_out)
                    self.assertEqual(Y_expected, torch.isnan(Y_out))

    def test_clamp_scalar_nan_bounds(self, device):
        # [FUNCTION DOC] Checks that a nan scalar as min or max makes the whole output nan.
        x = torch.ones(3, device=device)
        y = torch.clamp(x, None, float("nan"))
        self.assertTrue(torch.isnan(y).all())
        y = torch.clamp(x, float("nan"), None)
        self.assertTrue(torch.isnan(y).all())

    def test_clamp_raises_arg_errors(self, device):
        # [FUNCTION DOC] Checks clamp (method, in-place method, function) raises a runtime error
        # when neither min nor max is given.
        X = torch.randn(100, dtype=torch.float, device=device)
        error_msg = "At least one of 'min' or 'max' must not be None"
        with self.assertRaisesRegex(RuntimeError, error_msg):
            X.clamp()
        with self.assertRaisesRegex(RuntimeError, error_msg):
            X.clamp_()
        with self.assertRaisesRegex(RuntimeError, error_msg):
            torch.clamp(X)

    @dtypes(*all_passthru_types())
    @dtypesIfCUDA(*all_passthru_types_and(torch.chalf))
    def test_flip(self, device, dtype):
        # [FUNCTION DOC] Checks flip over many data cases: results for single/multiple dims,
        # permuted dim orders, negative dims, non-contiguous, expanded, transposed, empty and
        # size-1 inputs, vectorized image cases on CPU, plus quantized variants for float; then
        # checks output size is preserved for several dim combinations.
        make_from_data = partial(torch.tensor, device=device, dtype=dtype)
        make_from_size = partial(make_tensor, device=device, dtype=dtype)

        def test_flip_impl(input_t, dims, output_t):
            # [FUNCTION DOC] HELPER: For an input, dims and expected output, asserts flip gives the
            # expected result (also with negative first dim and every permutation of tuple dims),
            # and repeats for quantized copies when the dtype is float.
            def all_t():
                # [FUNCTION DOC] HELPER: Generator yielding the plain input/output pair and, for
                # float, per-tensor quantized pairs (quint8, qint8, qint32).
                yield input_t, output_t
                if dtype is torch.float:
                    # We generate quantized versions as well
                    for qdtype in (torch.quint8, torch.qint8, torch.qint32):
                        qinput_t = torch.quantize_per_tensor(input_t, 0.1, 5, qdtype)
                        qoutput_t = torch.quantize_per_tensor(output_t, 0.1, 5, qdtype)
                        yield qinput_t, qoutput_t

            for in_t, out_t in all_t():
                self.assertEqual(in_t.flip(dims), out_t)
                n = in_t.ndim
                if not isinstance(dims, tuple):
                    # Wrap dim
                    self.assertEqual(in_t.flip(-n + dims), out_t)
                else:
                    # Permute dimensions
                    for p_dims in permutations(dims):
                        self.assertEqual(in_t.flip(p_dims), out_t)
                        if len(p_dims) > 0:
                            # Wrap 1st dim
                            self.assertEqual(
                                in_t.flip((-n + p_dims[0],) + p_dims[1:]), out_t
                            )

        def gen_data():
            # [FUNCTION DOC] HELPER: Generator of (input, dims, expected output) triples: basic
            # 2x2x2 cases (contiguous and not), expanded, transposed, rectangular, CPU image cases
            # built with NumPy reversal, and no-op edge cases (size zero, empty dims, zero
            # elements, size one).
            # Basic tests
            data = make_from_data([1, 2, 3, 4, 5, 6, 7, 8]).view(2, 2, 2)
            nonctg = make_from_size((2, 2, 2), noncontiguous=True).copy_(data)

            dims_result = (
                (0, make_from_data([5, 6, 7, 8, 1, 2, 3, 4]).view(2, 2, 2)),
                (1, make_from_data([3, 4, 1, 2, 7, 8, 5, 6]).view(2, 2, 2)),
                (2, make_from_data([2, 1, 4, 3, 6, 5, 8, 7]).view(2, 2, 2)),
                ((0, 1), make_from_data([7, 8, 5, 6, 3, 4, 1, 2]).view(2, 2, 2)),
                ((0, 1, 2), make_from_data([8, 7, 6, 5, 4, 3, 2, 1]).view(2, 2, 2)),
            )
            for in_tensor, (dims, out_tensor) in product((data, nonctg), dims_result):
                yield in_tensor, dims, out_tensor

            # Expanded
            in_t = make_from_data([1, 2, 3]).view(3, 1).expand(3, 2)
            dims = 0
            out_t = make_from_data([3, 3, 2, 2, 1, 1]).view(3, 2)
            yield in_t, dims, out_t
            # Noop on expanded dimension
            yield in_t, 1, in_t

            # Transposed
            in_t = (
                make_from_data([1, 2, 3, 4, 5, 6, 7, 8]).view(2, 2, 2).transpose(0, 1)
            )
            dims = (0, 1, 2)
            out_t = make_from_data([8, 7, 4, 3, 6, 5, 2, 1]).view(2, 2, 2)
            yield in_t, dims, out_t

            # Rectangular case
            in_t = make_from_data([1, 2, 3, 4, 5, 6]).view(2, 3)
            dims = 0
            out_t = make_from_data([[4, 5, 6], [1, 2, 3]])
            yield in_t, dims, out_t
            dims = 1
            out_t = make_from_data([[3, 2, 1], [6, 5, 4]])
            yield in_t, dims, out_t

            # vectorized NCHW cases (images)
            if device == "cpu" and dtype != torch.bfloat16:
                for mf in [torch.contiguous_format, torch.channels_last]:
                    for c in [2, 3, 8, 16]:
                        in_t = make_from_size((2, c, 32, 32)).contiguous(
                            memory_format=mf
                        )
                        np_in_t = in_t.numpy()

                        np_out_t = np_in_t[:, :, :, ::-1].copy()
                        out_t = torch.from_numpy(np_out_t)
                        yield in_t, 3, out_t

                        np_out_t = np_in_t[:, :, ::-1, :].copy()
                        out_t = torch.from_numpy(np_out_t)
                        yield in_t, 2, out_t

                        # non-contig cases
                        in_tt = in_t[..., ::2, :]
                        np_in_t = in_tt.numpy()
                        np_out_t = np_in_t[:, :, :, ::-1].copy()
                        out_t = torch.from_numpy(np_out_t)
                        yield in_tt, 3, out_t

                        in_tt = in_t[..., ::2]
                        np_in_t = in_tt.numpy()
                        np_out_t = np_in_t[:, :, :, ::-1].copy()
                        out_t = torch.from_numpy(np_out_t)
                        yield in_tt, 3, out_t

            # Noops (edge cases)

            # Size 0
            in_t = make_from_data(())
            yield in_t, 0, in_t
            yield in_t, (), in_t

            # dims = ()
            in_t = make_from_size((3, 2, 1))
            yield in_t, (), in_t

            # Zero elements, non-zero size
            in_t = make_from_size((3, 0, 2))
            for i in range(in_t.ndim):
                yield in_t, i, in_t

            # Size 1
            in_t = make_from_size(())
            yield in_t, 0, in_t
            in_t = make_from_size((1,))
            yield in_t, 0, in_t

        for in_tensor, dims, out_tensor in gen_data():
            test_flip_impl(in_tensor, dims, out_tensor)

        # test for shape
        size = [2, 3, 4]
        data = make_from_size(size)
        possible_dims = range(len(size))
        test_dims = chain(
            combinations(possible_dims, 1), combinations(possible_dims, 2)
        )

        for dims in test_dims:
            self.assertEqual(size, list(data.flip(dims).size()))

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_flip_errors(self, device, dtype):
        # [FUNCTION DOC] Checks flip raises: runtime error for a repeated dim, type error with no
        # dims, index error for out-of-range dims.
        make_arg = partial(make_tensor, dtype=dtype, device=device)
        data = make_arg((2, 2, 2))

        # not allow flip on the same dim more than once
        self.assertRaises(RuntimeError, lambda: data.flip(0, 1, 1))
        # not allow empty list as input
        self.assertRaises(TypeError, lambda: data.flip())

        # not allow dim > max dim
        self.assertRaises(IndexError, lambda: data.flip(0, 1, 2, 3))
        self.assertRaises(IndexError, lambda: data.flip(3))

    def _rand_shape(self, dim, min_size, max_size):
        # [FUNCTION DOC] HELPER: Returns a tuple of dim random sizes between min_size and max_size
        # inclusive (as 0-d tensors from a random integer draw).
        return tuple(torch.randint(min_size, max_size + 1, (dim,)))

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_flip_numpy(self, device, dtype):
        # [FUNCTION DOC] Compares flip with NumPy flip for random 3-d and 4-d shapes, over all
        # combinations of axes.
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        for ndim in [3, 4]:
            shape = self._rand_shape(ndim, 5, 10)
            data = make_arg(shape)

            # Axis to sample for given shape.
            for i in range(1, ndim + 1):
                # Check all combinations of `i` axis.
                for flip_dim in combinations(range(ndim), i):
                    torch_fn = partial(torch.flip, dims=flip_dim)
                    np_fn = partial(np.flip, axis=flip_dim)
                    self.compare_with_numpy(torch_fn, np_fn, data)

    @onlyAccelerator  # CPU is too slow
    @largeTensorTest("17GB")  # 4 tensors of 4GB (in, out) x (torch, numpy) + 1GB
    @largeTensorTest(
        "81GB", "cpu"
    )  # even for CUDA test, sufficient system memory is required
    @unittest.skipIf(IS_JETSON, "Too large for Jetson")
    def test_flip_large_tensor(self, device):
        # [FUNCTION DOC] Accelerator only, needs large memory: flips a byte tensor with more than
        # 2^32 elements along dim 0 and compares with NumPy, then frees the input.
        t_in = torch.empty(2**32 + 1, dtype=torch.uint8).random_()
        torch_fn = partial(torch.flip, dims=(0,))
        np_fn = partial(np.flip, axis=0)
        self.compare_with_numpy(torch_fn, np_fn, t_in)
        del t_in

    @onlyCPU
    @unittest.expectedFailure
    @dtypes(torch.quint4x2, torch.quint2x4)
    def test_flip_unsupported_dtype(self, dtype):
        # [FUNCTION DOC] CPU only, marked expected failure: flip on a quantized sub-byte tensor is
        # unsupported and is expected to raise.
        scale, zero_point = 0.1, 5
        qt = torch.quantize_per_tensor(
            torch.randn(16, 16), scale=scale, zero_point=zero_point, dtype=dtype
        )
        torch.flip(qt, dims=(0,))

    def _test_fliplr_flipud(self, torch_fn, np_fn, min_dim, max_dim, device, dtype):
        # [FUNCTION DOC] HELPER: For each rank from min_dim to max_dim, builds random data of that
        # rank (normal for float/complex, small ints otherwise) and compares the torch function
        # with the NumPy function.
        for dim in range(min_dim, max_dim + 1):
            shape = self._rand_shape(dim, 5, 10)
            # Randomly scale the input
            if dtype.is_floating_point or dtype.is_complex:
                data = torch.randn(*shape, device=device, dtype=dtype)
            else:
                data = torch.randint(0, 10, shape, device=device, dtype=dtype)
            self.compare_with_numpy(torch_fn, np_fn, data)

    @dtypes(torch.int64, torch.double, torch.cdouble)
    def test_fliplr(self, device, dtype):
        # [FUNCTION DOC] Compares fliplr with NumPy for ranks 2 to 4.
        self._test_fliplr_flipud(torch.fliplr, np.fliplr, 2, 4, device, dtype)

    @dtypes(torch.int64, torch.double, torch.cdouble)
    def test_fliplr_invalid(self, device, dtype):
        # [FUNCTION DOC] Checks fliplr raises a runtime error for a 1-d and a 0-d input.
        x = torch.randn(42).to(dtype)
        with self.assertRaisesRegex(RuntimeError, "Input must be >= 2-d."):
            torch.fliplr(x)
        with self.assertRaisesRegex(RuntimeError, "Input must be >= 2-d."):
            torch.fliplr(torch.tensor(42, device=device, dtype=dtype))

    @dtypes(torch.int64, torch.double, torch.cdouble)
    def test_flipud(self, device, dtype):
        # [FUNCTION DOC] Compares flipud with NumPy for ranks 1 to 4.
        self._test_fliplr_flipud(torch.flipud, np.flipud, 1, 4, device, dtype)

    @dtypes(torch.int64, torch.double, torch.cdouble)
    def test_flipud_invalid(self, device, dtype):
        # [FUNCTION DOC] Checks flipud raises a runtime error for a 0-d input.
        with self.assertRaisesRegex(RuntimeError, "Input must be >= 1-d."):
            torch.flipud(torch.tensor(42, device=device, dtype=dtype))

    def test_rot90(self, device):
        # [FUNCTION DOC] Checks rot90 on a 2x2 tensor for k of 0..3, default args, reversed dims,
        # k modulo behavior (including negative k), a 3-d case, and index/runtime errors for
        # out-of-range, repeated, or wrong-count dims.
        data = torch.arange(1, 5, device=device).view(2, 2)
        self.assertEqual(torch.tensor([1, 2, 3, 4]).view(2, 2), data.rot90(0, [0, 1]))
        self.assertEqual(torch.tensor([2, 4, 1, 3]).view(2, 2), data.rot90(1, [0, 1]))
        self.assertEqual(torch.tensor([4, 3, 2, 1]).view(2, 2), data.rot90(2, [0, 1]))
        self.assertEqual(torch.tensor([3, 1, 4, 2]).view(2, 2), data.rot90(3, [0, 1]))

        # test for default args k=1, dims=[0, 1]
        self.assertEqual(data.rot90(), data.rot90(1, [0, 1]))

        # test for reversed order of dims
        self.assertEqual(data.rot90(3, [0, 1]), data.rot90(1, [1, 0]))

        # test for modulo of k
        self.assertEqual(data.rot90(5, [0, 1]), data.rot90(1, [0, 1]))
        self.assertEqual(data.rot90(3, [0, 1]), data.rot90(-1, [0, 1]))
        self.assertEqual(data.rot90(-5, [0, 1]), data.rot90(-1, [0, 1]))

        # test for dims out-of-range error
        self.assertRaises(IndexError, lambda: data.rot90(1, [0, -3]))
        self.assertRaises(IndexError, lambda: data.rot90(1, [0, 2]))

        # test tensor with more than 2D
        data = torch.arange(1, 9, device=device).view(2, 2, 2)
        self.assertEqual(
            torch.tensor([2, 4, 1, 3, 6, 8, 5, 7]).view(2, 2, 2), data.rot90(1, [1, 2])
        )
        self.assertEqual(data.rot90(1, [1, -1]), data.rot90(1, [1, 2]))

        # test for errors
        self.assertRaises(IndexError, lambda: data.rot90(1, [0, 3]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [1, 1]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, 1, 2]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0]))

    @skipIfTorchDynamo("TorchDynamo fails with an unknown error")
    @dtypes(torch.cfloat, torch.cdouble)
    def test_complex_rot90(self, device, dtype):
        # [FUNCTION DOC] Compares rot90 for k from 0 to 3 on random complex tensors with NumPy.
        shape = self._rand_shape(random.randint(2, 4), 5, 10)
        for rot_times in range(4):
            data = torch.randn(*shape, device=device, dtype=dtype)
            torch_fn = partial(torch.rot90, k=rot_times, dims=[0, 1])
            np_fn = partial(np.rot90, k=rot_times, axes=[0, 1])
            self.compare_with_numpy(torch_fn, np_fn, data)

    # TODO: update once warning flag is available to always trigger ONCE warnings
    # Ensures nonzero does not throw a warning, even when the as_tuple argument
    #   is not provided
    def test_nonzero_no_warning(self, device):
        # [FUNCTION DOC] Checks nonzero (function and method) emits no warnings.
        t = torch.randn((2, 2), device=device)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.nonzero(t)
            t.nonzero()
            self.assertEqual(len(w), 0)

    @dtypes(*all_types_and(torch.half, torch.bool, torch.bfloat16))
    def test_nonzero(self, device, dtype):
        # [FUNCTION DOC] For several shapes, builds a random 0/1 tensor, checks nonzero in plain,
        # out and tuple forms against NumPy, that a wrong-dtype out raises (except on xla), and
        # that an out on another device raises on cuda, xpu, or the privateuse1 device.
        shapes = [
            torch.Size((12,)),
            torch.Size((12, 1)),
            torch.Size((1, 12)),
            torch.Size((6, 2)),
            torch.Size((3, 2, 2)),
            torch.Size((5, 5, 5)),
        ]

        def gen_nontrivial_input(shape, dtype, device):
            # [FUNCTION DOC] HELPER: Returns random 0/1 values of the shape; for bfloat16 generates
            # in float then casts, since integer random generation for it is not supported on
            # Windows.
            if dtype != torch.bfloat16:
                return torch.randint(2, shape, device=device, dtype=dtype)
            else:
                # windows does not work for bfloat16 randing
                return torch.randint(2, shape, device=device, dtype=torch.float).to(
                    dtype
                )

        for shape in shapes:
            tensor = gen_nontrivial_input(shape, dtype, device)
            dst1 = torch.nonzero(tensor, as_tuple=False)
            dst2 = tensor.nonzero(as_tuple=False)
            dst3 = torch.empty([], dtype=torch.long, device=device)
            torch.nonzero(tensor, out=dst3)
            if self.device_type != "xla":
                # xla does not raise runtime error
                self.assertRaisesRegex(
                    RuntimeError,
                    "scalar type Long",
                    lambda: torch.nonzero(
                        tensor, out=torch.empty([], dtype=torch.float, device=device)
                    ),
                )
            if (
                self.device_type == "cuda"
                or self.device_type == "xpu"
                or self.device_type == TEST_PRIVATEUSE1_DEVICE_TYPE
            ):
                self.assertRaisesRegex(
                    RuntimeError,
                    "on the same device",
                    lambda: torch.nonzero(
                        tensor, out=torch.empty([], dtype=torch.long)
                    ),
                )
            np_array = (
                tensor.cpu().numpy()
                if dtype != torch.bfloat16
                else tensor.float().cpu().numpy()
            )
            np_result = torch.from_numpy(np.stack(np_array.nonzero())).t()
            self.assertEqual(dst1.cpu(), np_result, atol=0, rtol=0)
            self.assertEqual(dst2.cpu(), np_result, atol=0, rtol=0)
            self.assertEqual(dst3.cpu(), np_result, atol=0, rtol=0)
            tup1 = torch.nonzero(tensor, as_tuple=True)
            tup2 = tensor.nonzero(as_tuple=True)
            tup1 = torch.stack(tup1).t().cpu()
            tup2 = torch.stack(tup2).t().cpu()
            self.assertEqual(tup1, np_result, atol=0, rtol=0)
            self.assertEqual(tup2, np_result, atol=0, rtol=0)

    def test_nonzero_astuple_out(self, device):
        # [FUNCTION DOC] Checks that as_tuple with out raises, that as_tuple false with out equals
        # the default, that scripting a function using the as_tuple keyword raises, and that
        # tracing it works and matches eager results.
        t = torch.randn((3, 3, 3), device=device)
        out = torch.empty_like(t, dtype=torch.long)

        with self.assertRaises(RuntimeError):
            torch.nonzero(t, as_tuple=True, out=out)

        self.assertEqual(
            torch.nonzero(t, as_tuple=False, out=out), torch.nonzero(t, out=out)
        )

        # Verifies that JIT script cannot handle the as_tuple kwarg
        # See Issue https://github.com/pytorch/pytorch/issues/45499.
        def _foo(t):
            # [FUNCTION DOC] HELPER: Calls nonzero in tuple, non-tuple and out forms and returns
            # all three results; scripting it is expected to fail.
            tuple_result = torch.nonzero(t, as_tuple=True)
            nontuple_result = torch.nonzero(t, as_tuple=False)
            out = torch.empty_like(nontuple_result)
            torch.nonzero(t, as_tuple=False, out=out)
            return tuple_result, nontuple_result, out

        with self.assertRaises(RuntimeError):
            torch.jit.script(_foo)

        # Verifies that JIT tracing works fine
        traced_foo = torch.jit.trace(_foo, t)
        traced_tuple, traced_nontuple, traced_out = traced_foo(t)
        expected_tuple = torch.nonzero(t, as_tuple=True)
        expected_nontuple = torch.nonzero(t)

        self.assertEqual(traced_tuple, expected_tuple)
        self.assertEqual(traced_nontuple, expected_nontuple)
        self.assertEqual(traced_out, expected_nontuple)

    def test_nonzero_discontiguous(self, device):
        # [FUNCTION DOC] Checks nonzero on a non-contiguous input equals the contiguous result,
        # that an out tensor's storage is reused, and that a non-contiguous out keeps its strides.
        shape = (4, 4)
        tensor = torch.randint(2, shape, device=device)
        tensor_nc = torch.empty(shape[0], shape[1] * 2, device=device)[:, ::2].copy_(
            tensor
        )
        dst1 = tensor.nonzero(as_tuple=False)
        dst2 = tensor_nc.nonzero(as_tuple=False)
        self.assertEqual(dst1, dst2, atol=0, rtol=0)
        dst3 = torch.empty_like(dst1)
        data_ptr = dst3.data_ptr()
        # expect dst3 storage to be reused
        torch.nonzero(tensor, out=dst3)
        self.assertEqual(data_ptr, dst3.data_ptr())
        self.assertEqual(dst1, dst3, atol=0, rtol=0)
        # discontiguous out
        dst4 = torch.empty(
            dst1.size(0), dst1.size(1) * 2, dtype=torch.long, device=device
        )[:, ::2]
        data_ptr = dst4.data_ptr()
        strides = dst4.stride()
        torch.nonzero(tensor, out=dst4)
        self.assertEqual(data_ptr, dst4.data_ptr())
        self.assertEqual(dst1, dst4, atol=0, rtol=0)
        self.assertEqual(strides, dst4.stride())

    def test_nonzero_non_diff(self, device):
        # [FUNCTION DOC] Checks nonzero output does not require grad even if the input does.
        x = torch.randn(10, requires_grad=True)
        nz = x.nonzero()
        self.assertFalse(nz.requires_grad)

    @dtypes(torch.int64, torch.float, torch.complex128)
    def test_sparse_dense_dim(self, device, dtype):
        # [FUNCTION DOC] Checks sparse_dim is 0 and dense_dim equals the rank for dense tensors of
        # shapes 0-d, 1-d and 2-d.
        for shape in [(), (2,), (2, 3)]:
            if dtype.is_complex or dtype.is_floating_point:
                x = torch.rand(shape, device=device, dtype=dtype)
            else:
                x = torch.randint(-9, 9, shape, device=device, dtype=dtype)
            self.assertEqual(x.sparse_dim(), 0)
            self.assertEqual(x.dense_dim(), len(shape))

    def test_unfold_all_devices_and_dtypes(self, device):
        # [FUNCTION DOC] For each dtype checks unfold of an empty tensor gives the expected shape.
        # KNOWN ISSUE: the bool and non-bool branches are identical.
        for dt in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            if dt == torch.bool:
                x = torch.empty((0, 1, 3, 0), dtype=dt, device=device)
                self.assertEqual((0, 1, 1, 0, 3), x.unfold(2, 3, 2).shape)
            else:
                x = torch.empty((0, 1, 3, 0), dtype=dt, device=device)
                self.assertEqual((0, 1, 1, 0, 3), x.unfold(2, 3, 2).shape)

    def test_unfold_scalars(self, device):
        # [FUNCTION DOC] Checks unfold on a 0-d tensor returns a 1-d tensor of the requested size.
        x = torch.tensor(0.5, device=device)
        # unfold on a 0-dimensional tensor should always return a 1-d dimensional
        # tensor of shape [size] (i.e., the second parameter to unfold)

        self.assertEqual(torch.empty(0, device=device), x.unfold(0, 0, 1))
        self.assertEqual(torch.empty(0, device=device), x.unfold(0, 0, 2))
        self.assertEqual(torch.tensor([0.5], device=device), x.unfold(0, 1, 1))

    def test_unfold_errors(self, device):
        # [FUNCTION DOC] Checks unfold raises a runtime error for negative size and negative step.
        x = torch.arange(1.0, 8, device=device)
        with self.assertRaisesRegex(RuntimeError, "size is -1 but must be >= 0"):
            x.unfold(0, -1, 1)
        with self.assertRaisesRegex(RuntimeError, "step is -1 but must be > 0"):
            x.unfold(0, 1, -1)

    def test_unfold_backward_errors(self, device):
        # [FUNCTION DOC] Checks the unfold_backward op raises a value error for step zero and a
        # runtime error for negative size.
        grad_in = torch.randn(2, 3, device=device)
        input_sizes = [6]

        with self.assertRaisesRegex(ValueError, "step is 0 but must be > 0"):
            torch.ops.aten.unfold_backward(grad_in, input_sizes, 0, 3, 0)

        with self.assertRaisesRegex(RuntimeError, "size is -1 but must be >= 0"):
            torch.ops.aten.unfold_backward(grad_in, input_sizes, 0, -1, 1)


instantiate_device_type_tests(TestShapeOps, globals())

if __name__ == "__main__":
    run_tests()
