# Owner(s): ["module: masked operators"]
# [MODULE DOC] Tests for torch.masked operators. Builds reference implementations that apply
# reductions and normalizations slice by slice over masked-in elements, collects the masked
# ops from the op database, and checks strided results against references and sparse COO/CSR
# layouts against strided results. Also tests the masked where for sparse inputs. Device-type
# test classes are instantiated at import time for all devices except meta.

"""Tests for masked operations.
"""

import itertools
import torch
from typing import Any
from functools import wraps
import unittest
from torch.testing._internal.common_utils import skipIfTorchDynamo


from torch.testing._internal.common_utils import \
    (TestCase, parametrize, suppress_warnings, _TestParametrizer, run_tests)
from torch.testing._internal.common_methods_invocations import \
    (op_db, SampleInput)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, onlyNativeDeviceTypes, precisionOverride)


def apply_masked_reduction_along_dim(op, input, *args, **kwargs):
    # [FUNCTION DOC] Reference implementation of a masked reduction. Pops the mask and the
    # argument position of dim from the keyword arguments, locates dim in positional or keyword
    # arguments, and replaces it with None before calling the op. Enumerates every elementary
    # slice (all index combinations over non-reduced dims, full range over reduced dims), gathers
    # only masked-in elements into a flat tensor and applies the op. Empty slices keep nan for
    # floating dtype, else 0. Returns the result with reduced dims kept as size one, or dropped
    # unless keepdim is requested. A zero-dim input is passed straight to the op and cast to the
    # dtype. Raises AssertionError if dim is given both positionally and by keyword.
    """Applies reduction op along given dimension to strided x
    elements that are valid according to mask tensor.

    The op is applied to each elementary slice of input with args and
    kwargs with the following constraints:

    1. Prior applying the op:

      A. if kwargs contains an item with key 'dim_position' then it is
         removed from kwargs. The value of 'dim_position' is an
         integer that describes the dim argument position: while
         typically the dim argument appears at the 0-th position of
         the op arguments (excluding input), for instance, sum(input,
         dim), then there exists reductions that have extra arguments
         prior the dim argument, for instance, norm(input, ord, dim).

      B. if args or kwargs contains dim or keepdim arguments, these
         will be removed or replaced with None so that the op is
         applied to elementary slice using the default dim and keepdim
         value.

    2. The elementary slice of the input is defined as the flattened
      slice that has no masked out elements and when op is applied,
      the result will be a scalar value (assuming keepdim=False). For
      example, an input tensor to a reduction operation op having
      dim=0 and keepdim=True argument:

       [[1 * 2 * *]
        [* 3 4 * 5]]

      (* denotes masked out elements) has the following elementary
      slices: [1, 2] and [3, 4, 5]. The result of
      apply_masked_reduction_along_dim is

       [[op([1, 2], *args0, **kwargs, dim=None, keepdim=False)]
        [op([3, 4, 5], *args0, **kwargs, dim=None, keepdim=False)]]

      where args0 is args where dim value is replaced with None if
      present.

      Using the same example data, if the op is called with dim=(0, 1)
      and keepdim=False, there is one elementary slice: [1, 2, 3, 4,
      5]; and the corresponding result of the op is:

        op([1, 2, 3, 4, 5], *args0, **kwargs, dim=None, keepdim=False)

    3. If the elementary slice is empty, the corresponding output
      value is nan if dtype is float, otherwise, 0.  An empty
      elementary slice corresponds to fully masked-out output, so, the
      corresponding specific value of the output will not be important
      because we used masked equality check for comparing the results
      of masked operations.
    """
    # eliminate mask and dim_position keyword arguments:
    mask = kwargs.pop('mask', None)
    dim_pos = kwargs.pop('dim_position', 0)

    dtype = kwargs.get('dtype', input.dtype)
    if input.ndim == 0:
        # scalar input is an elementary slice
        return op(input, *args, **kwargs).to(dtype=dtype)

    # eliminate keepdim keyword argument if specified:
    keepdim = kwargs.pop('keepdim', False)

    # eliminate dim argument that may appear both as args or kwargs
    # element:
    if dim_pos < len(args):
        # dim is specified in args
        if 'dim' in kwargs:
            raise AssertionError(f"'dim' should not be in kwargs: {(args, kwargs)}")
        dim = args[dim_pos]
        args0 = args[:dim_pos] + (None,) + args[dim_pos + 1:]
    else:
        # dim may be specified in kwargs
        dim = kwargs.pop('dim', None)
        args0 = args

    # dimensions along which the reduction operation is applied:
    dim_ = torch.masked._canonical_dim(dim, input.ndim)
    # slices in product(*ranges) define all elementary slices:
    ranges: list[Any] = []
    # shape of output for the keepdim=True case:
    shape = []
    for i in range(input.ndim):
        if i in dim_:
            ranges.append((slice(None),))
            shape.append(1)
        else:
            ranges.append(range(input.shape[i]))
            shape.append(input.shape[i])

    # keepdim=True version of the output, filled with nan or 0:
    output = input.new_full(shape, float('nan') if dtype.is_floating_point else 0, dtype=dtype)

    # apply op to all elementary slices:
    if mask is None:
        inpmask = input.new_ones([], dtype=torch.bool).expand(input.shape)
    else:
        inpmask = torch.masked._input_mask(input, mask=mask)
    for s in itertools.product(*ranges):
        # data of an elementary slice is 1D sequence and has only
        # masked-in elements:
        data = input[s].flatten()[inpmask[s].flatten().argwhere()]
        if not data.numel():
            # empty elementary slice
            continue
        output[s][0] = op(data, *args0, **kwargs)

    if not keepdim:
        # reshape output for the keepdim=False case
        shape = [shape[i] for i in range(len(shape)) if i not in dim_]
        output = output.reshape(shape)
    return output


def apply_masked_normalization_along_dim(op, input, *args, **kwargs):
    # [FUNCTION DOC] Reference implementation of a masked normalization such as softmax. Pops mask
    # and dim position, returns the op result directly for zero-dim input, otherwise takes dim
    # from the positional arguments, runs the op with dim replaced by zero on each 1D slice along
    # dim restricted to its masked-in elements, and writes into a zeros tensor of the output dtype
    # so masked-out positions stay zero. Returns the full-shape output.
    """Applies normalization op along given dimension to strided x
    elements that are valid according to mask tensor.
    """
    mask = kwargs.pop('mask', None)
    dim_pos = kwargs.pop('dim_position', 0)
    if input.ndim == 0:  # scalar input
        return op(input, *args, **kwargs)
    dtype = kwargs.get('dtype', input.dtype)
    dim = args[dim_pos]
    args0 = args[:dim_pos] + (0,) + args[dim_pos + 1:]
    output = torch.zeros_like(input, dtype=dtype)
    if mask is None:
        inpmask = input.new_ones([], dtype=torch.bool).expand(input.shape)
    else:
        inpmask = torch.masked._input_mask(input, mask=mask)
    dim_ = dim % input.ndim
    left_ranges = tuple(map(range, input.shape[:dim_]))
    right_ranges = tuple(map(range, input.shape[dim_ + 1:]))
    for s in itertools.product(*(left_ranges + ((slice(None),),) + right_ranges)):
        indices = inpmask[s].argwhere()
        output[s][indices] = op(input[s][indices], *args0, **kwargs)
    return output


reference_functions = dict(
    norm=lambda *args, **kwargs: apply_masked_reduction_along_dim(torch.linalg.vector_norm, *args, **dict(kwargs, dim_position=1)),
    var=lambda *args, **kwargs: apply_masked_reduction_along_dim(torch.var, *args, **dict(kwargs, dim_position=0)),
    std=lambda *args, **kwargs: apply_masked_reduction_along_dim(torch.std, *args, **dict(kwargs, dim_position=0)),
    softmax=lambda *args, **kwargs: apply_masked_normalization_along_dim(torch.softmax, *args, **kwargs),
    log_softmax=lambda *args, **kwargs: apply_masked_normalization_along_dim(torch.log_softmax, *args, **kwargs),
    softmin=lambda *args, **kwargs: apply_masked_normalization_along_dim(torch.nn.functional.softmin, *args, **kwargs),
    normalize=lambda *args, **kwargs: apply_masked_normalization_along_dim(
        torch.nn.functional.normalize, *args, **dict(kwargs, dim_position=1)),
)

masked_ops = [op for op in op_db if op.name.startswith('masked.')]
masked_ops_with_references = [op for op in masked_ops if op.name.rsplit('.', 1)[-1] in reference_functions]
masked_ops_with_non_strided_support = [op for op in masked_ops if op.supports_sparse or op.supports_sparse_csr]


def _tensor_to_strided(obj):
    # [FUNCTION DOC] HELPER: Returns a non-tensor object or an already strided tensor unchanged,
    # and converts any other layout tensor to dense.
    # after gh-59958 is resolved, replace the usage of this function
    # with torch.Tensor.to_dense
    if torch.is_tensor(obj):
        if obj.layout == torch.strided:
            return obj
        return obj.to_dense()
    return obj


def to_strided(obj):
    # [FUNCTION DOC] Maps the strided-conversion helper over every leaf of a nested structure and
    # returns the structure with all tensors dense.
    """Convert the tensor content of object to strided tensor content.
    """
    return torch.utils._pytree.tree_map(_tensor_to_strided, obj)


def to_sparse_coo(obj):
    # [FUNCTION DOC] Maps the tensor to_sparse method over every leaf of a nested structure, so
    # every leaf must be a tensor; returns the structure with sparse COO tensors.
    """Convert the tensor content of object to sparse coo tensor content.
    """
    return torch.utils._pytree.tree_map(torch.Tensor.to_sparse, obj)


def to_sparse_csr(obj):
    # [FUNCTION DOC] Maps the tensor to_sparse_csr method over every leaf of a nested structure,
    # so every leaf must be a tensor; returns the structure with CSR tensors.
    """Convert the tensor content of object to sparse csr tensor content.
    """
    return torch.utils._pytree.tree_map(torch.Tensor.to_sparse_csr, obj)


class mask_layouts(_TestParametrizer):
    # [CLASS DOC] Parametrizer used as a test decorator. Each decorated test is generated
    # three times, once per layout (strided, sparse COO, sparse CSR), and receives the layout
    # plus a generator of sample inputs whose masks are provided in all layouts the op
    # supports. Stateless.
    """Decorator class for parametrization of test function with an input
    layout argument and an extra argument of sample inputs generator.
    The sample_inputs generator provides samples with all supported
    layouts for the mask argument.
    """
    def _parametrize_test(self, test, generic_cls, device_cls):
        # [FUNCTION DOC] Parametrizer hook. Defines a wrapper for the test and yields one
        # entry per layout with the wrapper, a layout name derived from the layout string, the
        # layout as a parameter and an empty decorator list.

        @wraps(test)
        def wrap(self, layout, device, dtype, op):
            # [FUNCTION DOC] HELPER: Picks the sample input function for the layout (regular
            # for strided, sparse variants otherwise), raising SkipTest when the op lacks
            # support for that layout and NotImplementedError for any other layout. Calls the
            # wrapped test with layout, device, dtype, op and a sample input generator.
            # KNOWN ISSUE: the layout name is built with a character-set strip rather than a
            # prefix removal; it gives the intended names only because the names start with a
            # letter not in that set.
            layout_name = str(layout).lstrip('torch.')
            if layout == torch.strided:
                # strided layouts are always supported
                sample_inputs_func = op.sample_inputs
            elif layout == torch.sparse_coo:
                if not op.supports_sparse:
                    raise unittest.SkipTest(f"{op.name} does not support inputs with {layout_name} layout")
                sample_inputs_func = op.sample_inputs_sparse_coo
            elif layout == torch.sparse_csr:
                if not op.supports_sparse_csr:
                    raise unittest.SkipTest(f"{op.name} does not support inputs with {layout_name} layout")
                sample_inputs_func = op.sample_inputs_sparse_csr
            else:
                raise NotImplementedError(f'{layout}')

            def sample_inputs_generator():
                # [FUNCTION DOC] HELPER: Generator over the op's samples. Samples without a
                # mask are yielded as is. For masked samples it yields the original when its
                # input layout matches, and copies with the mask converted to dense (non-
                # strided layouts), to sparse COO (if the layout is not COO and the op
                # supports sparse) or to CSR (if the layout is not CSR, supported and the
                # input is 2D).
                for sample_input in sample_inputs_func(device, dtype):
                    mask = sample_input.kwargs.get('mask')
                    if mask is None:
                        yield sample_input
                    else:
                        if layout == sample_input.input.layout:
                            yield sample_input
                        if layout != torch.strided:
                            sample_input_kwargs = sample_input.kwargs.copy()
                            sample_input_kwargs.update(mask=mask.to_dense())
                            yield SampleInput(sample_input.input.clone(),
                                              args=sample_input.args,
                                              kwargs=sample_input_kwargs)
                        if layout != torch.sparse_coo and op.supports_sparse:
                            sample_input_kwargs = sample_input.kwargs.copy()
                            sample_input_kwargs.update(mask=mask.to_sparse())
                            yield SampleInput(sample_input.input.clone(),
                                              args=sample_input.args,
                                              kwargs=sample_input_kwargs)
                        if layout != torch.sparse_csr and op.supports_sparse_csr and sample_input.input.ndim == 2:
                            sample_input_kwargs = sample_input.kwargs.copy()
                            sample_input_kwargs.update(mask=mask.to_sparse_csr())
                            yield SampleInput(sample_input.input.clone(),
                                              args=sample_input.args,
                                              kwargs=sample_input_kwargs)

            test(self, layout, device, dtype, op, sample_inputs_generator())

        for layout in (torch.strided, torch.sparse_coo, torch.sparse_csr):
            yield (wrap, str(layout).lstrip('torch.'), {'layout': layout}, lambda _: [])


class TestMasked(TestCase):
    # [CLASS DOC] Device-generic tests for masked operators: comparison to slice-wise
    # references, consistency across mask and input layouts, and sparse masked where.
    # Instantiated per device type with meta excluded.

    def assertEqualMasked(self, actual, expected, mask):
        # [FUNCTION DOC] HELPER: Densifies the actual result, zeroes both actual and expected
        # outside the mask when a mask is given, then asserts equality (device ignored).
        strided = to_strided(actual)
        if mask is not None:
            strided = torch.where(mask, strided, strided.new_zeros([]))
            expected = torch.where(mask, expected, expected.new_zeros([]))
        self.assertEqual(strided, expected, exact_device=False)

    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(masked_ops_with_references)
    @precisionOverride({torch.bfloat16: 5e-4, torch.float16: 5e-4})
    def test_reference_masked(self, device, dtype, op):
        # [FUNCTION DOC] For each masked op that has a reference, runs the op and the
        # reference on every sample input and compares them only at the output mask positions.
        # Variance and std are skipped for integer inputs. Runs on native device types with
        # warnings suppressed and loosened precision for half and bfloat16.
        op_name = op.name.rsplit('.', 1)[-1]
        ref_op = reference_functions[op_name]
        sample_inputs = op.sample_inputs(device, dtype)
        for sample_input in sample_inputs:
            t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs
            if op_name in {'var', 'std'} and not (t_inp.dtype.is_floating_point or t_inp.dtype.is_complex):
                # torch.var/torch.std does not support integer inputs
                continue
            actual = op.op(t_inp, *t_args, **t_kwargs)
            expected = ref_op(t_inp, *t_args, **t_kwargs)
            if t_kwargs.get('mask') is None:
                outmask = None
            else:
                outmask = torch.masked._output_mask(op.op, t_inp, *t_args, **t_kwargs)
            self.assertEqualMasked(actual, expected, outmask)

    @mask_layouts()
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(masked_ops_with_non_strided_support)
    @precisionOverride({torch.bfloat16: 5e-3, torch.float16: 5e-3})
    def test_mask_layout(self, layout, device, dtype, op, sample_inputs):
        # [FUNCTION DOC] For every sample, runs the masked op and requires the result layout
        # to equal the tested layout (AssertionError otherwise). Then converts inputs to
        # strided, runs the op again and compares at the output mask, checking sparse results
        # agree with dense ones.
        for sample in sample_inputs:
            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            actual = op.op(t_inp, *t_args, **t_kwargs)

            if actual.layout != layout:
                raise AssertionError(f"actual.layout should be {layout}, got {actual.layout}")

            # check masked invariance:
            #  op(inp, mask).to_dense() == op(inp.to_dense(), mask.to_dense()) at outmask
            #
            r_inp, r_args, r_kwargs = to_strided((t_inp, t_args, t_kwargs))
            if r_kwargs.get('mask') is None:
                outmask = None
            else:
                outmask = torch.masked._output_mask(op.op, r_inp, *r_args, **r_kwargs)
            expected = op.op(r_inp, *r_args, **r_kwargs)
            self.assertEqualMasked(actual, expected, outmask)

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1992")
    @parametrize("sparse_kind,fill_value", [('coo', 0), ('hybrid_coo', 0),
                                            ('coo', 123), ('hybrid_coo', 123),
                                            ('csr', 0), ('csr', 123)],
                 name_fn=lambda sparse_kind, fill_value: f'{sparse_kind}_fill_value_{fill_value}')
    def test_where(self, sparse_kind, fill_value):
        # [FUNCTION DOC] Tests the masked where on sparse COO, hybrid COO and CSR masks and
        # inputs for two fill values. Builds a sparse mask and input with some explicitly
        # stored masked-out or zero values, computes the expected sparse result with
        # placeholder entries replaced by zero, and compares. Also checks the dense equivalent
        # of the where over the output mask. Raises AssertionError for unknown sparse kind or
        # layout. Skipped under Dynamo.

        is_hybrid = False
        if sparse_kind == 'coo':

            def to_sparse(dense):
            # [FUNCTION DOC] HELPER: Converts a dense tensor to COO, two sparse dims.
                return dense.to_sparse(2)

            def set_values(sparse, index, value):
            # [FUNCTION DOC] HELPER: Overwrites the stored COO values at an index (mutates).
                sparse._values()[index] = value

        elif sparse_kind == 'hybrid_coo':
            is_hybrid = True

            def to_sparse(dense):
            # [FUNCTION DOC] HELPER: Converts a dense tensor to hybrid COO, one sparse dim.
                return dense.to_sparse(1)

            def set_values(sparse, index, value):
            # [FUNCTION DOC] HELPER: Overwrites the stored COO values at an index (mutates).
                sparse._values()[index] = value

        elif sparse_kind == 'csr':

            def to_sparse(dense):
            # [FUNCTION DOC] HELPER: Converts a dense tensor to CSR layout.
                return dense.to_sparse_csr()

            def set_values(sparse, index, value):
            # [FUNCTION DOC] HELPER: Overwrites the stored CSR values at an index (mutates).
                sparse.values()[index] = value

        else:
            raise AssertionError(f"unexpected sparse_kind: {sparse_kind}")

        mask = torch.tensor([[1, 0, 1, 0, 0],
                             [1, 1, 1, 1, 0],
                             [0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0],
                             [1, 1, 0, 0, 0]]).to(dtype=bool)
        mask = to_sparse(mask)
        # make some specified mask elements as explicit masked-out masks:
        if is_hybrid:
            set_values(mask, (1, 1), False)
            set_values(mask, (-2, -2), False)
        else:
            set_values(mask, 3, False)
            set_values(mask, -3, False)

        input = torch.tensor([[1, 0, 0, 0, -1],
                              [2, 3, 0, 0, -2],
                              [0, 4, 5, 0, -3],
                              [0, 0, 6, 7, 0],
                              [0, 8, 9, 0, -3],
                              [10, 11, 0, 0, -5]])
        input = to_sparse(input)
        # make specified input elements have zero values:
        if is_hybrid:
            set_values(input, (1, 1), 0)
            set_values(input, (-1, 0), 0)
            F = fill_value
        else:
            set_values(input, 3, 0)
            set_values(input, -3, 0)
            F = 0

        # expected where result:
        Z = 99
        # Z value corresponds to masked-in elements that are not
        # specified in the input and it will be replaced with a zero
        tmp = torch.tensor([[1, F, Z, F, F],
                            [2, F, Z, Z, F],
                            [F, 4, F, Z, F],
                            [0, 0, 0, 0, 0],
                            [F, F, 9, F, F],
                            [Z, 11, F, F, F]])
        tmp = to_sparse(tmp)


        sparse = torch.masked._where(mask, input,
                                     torch.tensor(fill_value, dtype=input.dtype, device=input.device))

        if tmp.layout == torch.sparse_coo:
            expected_sparse = torch.sparse_coo_tensor(
                tmp.indices(),
                torch.where(tmp.values() != Z, tmp.values(), tmp.values().new_full([], 0)),
                input.shape)
            outmask = torch.sparse_coo_tensor(sparse.indices(),
                                              sparse.values().new_full(sparse.values().shape, 1).to(dtype=bool),
                                              sparse.shape)._coalesced_(True)
        elif tmp.layout == torch.sparse_csr:
            expected_sparse = torch.sparse_csr_tensor(
                tmp.crow_indices(),
                tmp.col_indices(),
                torch.where(tmp.values() != Z, tmp.values(), tmp.values().new_full([], 0)),
                input.shape)
            outmask = torch.sparse_csr_tensor(sparse.crow_indices(), sparse.col_indices(),
                                              sparse.values().new_full(sparse.values().shape, 1).to(dtype=bool),
                                              sparse.shape)
        else:
            raise AssertionError("unexpected sparse layout")

        self.assertEqual(sparse, expected_sparse)

        # check invariance:
        #  torch.where(mask.to_dense(), input.to_dense(), fill_value)
        #    == where(mask, input, fill_value).to_dense(fill_value)
        expected = torch.where(mask.to_dense(), input.to_dense(), torch.full(input.shape, F))
        dense = torch.where(outmask.to_dense(), sparse.to_dense(), torch.full(sparse.shape, F))
        self.assertEqual(dense, expected)


instantiate_device_type_tests(TestMasked, globals(), except_for='meta')

if __name__ == "__main__":
    run_tests()
