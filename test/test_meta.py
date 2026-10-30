# Owner(s): ["module: decompositions"]
# ruff: noqa: F841
# [MODULE DOC] Tests that meta-tensor behavior agrees with real tensor behavior. It contains
# tests of MetaConverter (conversion of real tensors, views, autograd leaves, complex and
# channels-last tensors into metadata-equal meta tensors), plus operator-database-driven cross-
# reference tests (TestMeta) that run each operator on real inputs and on meta inputs, via torch
# function mode, torch dispatch mode and the python dispatcher, and compare dtype, shape,
# strides, storage offset, grad and conj/neg flags. Large module-level tables list expected
# failures and skips per operator and dtype, per device type. At import time it defines dtype
# aliases, builds the combined foreach operator list, and (only if the PYTORCH_COLLECT_EXPECT
# environment variable is 1) registers an exit hook that prints suggested expected-failure and
# skip tables. Device-specific test classes are instantiated at the bottom of the file. In the
# main block, if the XLA or text comparison environment variables name a file, it prints which of
# the listed operators are skipped or expected to fail in the dispatch tables and exits, else it
# runs the test runner.
# KNOWN ISSUE: the dispatch device skip table for CUDA lists the softmax integer overload twice;
# duplicate dictionary keys are silently collapsed.

import itertools
import torch
import os
import numpy as np
from enum import Enum
from torch.overrides import resolve_name
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._pytree import tree_map, tree_map_only, tree_flatten, tree_unflatten
from torch.utils import _pytree as pytree
from torch._subclasses.meta_utils import MetaConverter, assert_metadata_eq, is_sparse_any
import torch.utils._python_dispatch
from torch._dispatch.python import enable_python_dispatcher
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx.experimental import _config as exp_config
from torch.testing import make_tensor
from torch.testing._internal.common_utils import unMarkDynamoStrictTest
from torch.testing._internal.common_utils import (
    TestCase,
    skipIfCrossRef,
    skipIfTorchDynamo,
    suppress_warnings,
    TEST_WITH_TORCHDYNAMO,
    run_tests,
    parametrize,
    xfailIfTorchDynamo,
    skipIfXpu,
)
from torch.testing._internal.common_device_type import (
    ops,
    instantiate_device_type_tests,
    onlyCUDA,
    onlyCPU,
    OpDTypes,
    skipOps,
    xfail,
    skip,
)
from torch.testing._internal.common_dtype import (
    complex_types,
    integral_types_and,
    complex_types_and,
    all_types_and_complex_and,
    integral_types,
)
from torch.testing._internal.common_methods_invocations import (
    binary_ufuncs, op_db, foreach_unary_op_db, foreach_binary_op_db,
    foreach_pointwise_op_db, foreach_reduce_op_db, foreach_other_op_db)
from torch.testing._internal.opinfo.core import S, SampleInput
from torchgen.yaml_utils import YamlLoader
from torchgen.model import OperatorName

import copy
import sys
import yaml
import atexit
import re
from collections import defaultdict
from collections.abc import Iterable
import unittest
import warnings
import weakref
from functools import partial, wraps

bf16 = torch.bfloat16
f64 = torch.float64
f32 = torch.float32
f16 = torch.float16
c32 = torch.complex32
c64 = torch.complex64
c128 = torch.complex128
i8 = torch.int8
i16 = torch.int16
i32 = torch.int32
i64 = torch.int64
b8 = torch.bool
u8 = torch.uint8
u16 = torch.uint16
u32 = torch.uint32
u64 = torch.uint64

foreach_op_db = (
    foreach_unary_op_db +
    foreach_binary_op_db +
    foreach_pointwise_op_db +
    foreach_reduce_op_db +
    foreach_other_op_db
)


class TestMetaConverter(TestCase):
    # [CLASS DOC] Tests for MetaConverter. Each test builds a real tensor (view, leaf, non-leaf,
    # channels-last, complex, strided) with a fresh converter, converts it, and checks that
    # metadata (shape, strides, dtype, grad flags, view structure) matches the original. Also
    # checks version counter sharing between views and that the converter's memo tables release
    # entries when tensors die.
    def assertSameVersionCounter(self, m1, m2):
        # [FUNCTION DOC] HELPER: Asserts that two meta views share one version counter: reads the
        # first one's version, requires the second to equal it, bumps the base in place under no-
        # grad, and requires the first version to change and the two to remain equal. Used
        # because storage identity cannot be tested directly.
        # Cannot easily test m1 and m2 have same storage due to
        # lack of Storage bindings.  Use version counter.
        vc = m1._version
        self.assertEqual(m2._version, vc)
        # Doing it this way ensures that we get VC bump even with leaves
        with torch.no_grad():
            m1._base.add_(3)
        self.assertNotEqual(m1._version, vc)
        self.assertEqual(m2._version, m1._version)

    def assertMetadataMatches(self, m1, m2):
        # [FUNCTION DOC] HELPER: Compares the metadata of two tensors using the metadata
        # comparison utility, with this test case's equality assertion as the checker.
        assert_metadata_eq(self.assertEqual, m1, m2)

    def test_view_of_non_leaf(self):
        # [FUNCTION DOC] Converts two separate slices of a non-leaf tensor to meta with one
        # converter and checks both are views of a non-leaf base, are distinct objects, match the
        # metadata of their originals, and share a version counter.
        x = torch.randn(4, requires_grad=True)
        y = x.neg()
        z1 = y[:]
        z2 = y[:]
        to_meta = MetaConverter()
        m1 = to_meta(z1)
        m2 = to_meta(z2)

        # check the test is actually testing what it claims
        self.assertTrue(m1._is_view())
        self.assertFalse(m1._base.is_leaf)

        self.assertIsNot(m1, m2)
        self.assertMetadataMatches(m1, z1)
        self.assertMetadataMatches(m2, z2)
        self.assertSameVersionCounter(m1, m2)

    def test_view_of_leaf(self):
        # [FUNCTION DOC] Same as the previous view test but the base is a leaf requiring grad:
        # two slices of it are converted, must be views of a leaf base, distinct objects with
        # matching metadata and a shared version counter.
        x = torch.randn(4, requires_grad=True)
        z1 = x[:]
        z2 = x[:]
        to_meta = MetaConverter()
        m1 = to_meta(z1)
        m2 = to_meta(z2)

        # check the test is actually testing what it claims
        self.assertTrue(m1._is_view())
        self.assertTrue(m1._base.is_leaf)

        self.assertIsNot(m1, m2)
        self.assertMetadataMatches(m1, z1)
        self.assertMetadataMatches(m2, z2)
        self.assertSameVersionCounter(m1, m2)

    def test_view_of_view_of_leaf(self):
        # [FUNCTION DOC] Converts the base tensor and then a view of a view (whose middle view
        # requires grad), and checks that the converted tensors' metadata matches the originals
        # and that the final view is not a leaf.
        x = torch.randn(8)
        y = x.view(2, 4)
        y.requires_grad = True
        z = y.view(2, 2, 2)

        to_meta = MetaConverter()
        mx = to_meta(x)
        mz = to_meta(z)

        self.assertFalse(z.is_leaf)

        self.assertMetadataMatches(mx, x)
        self.assertMetadataMatches(mz, z)

    def test_leaf(self):
        # [FUNCTION DOC] Converts a leaf tensor that requires grad and checks the result is a
        # leaf requiring grad with matching metadata.
        x = torch.randn(4, requires_grad=True)
        to_meta = MetaConverter()
        m = to_meta(x)

        # check the test is actually testing what it claims
        self.assertTrue(m.is_leaf)
        self.assertTrue(m.requires_grad)

        self.assertMetadataMatches(m, x)

    def test_non_leaf(self):
        # [FUNCTION DOC] Converts a non-leaf tensor requiring grad and checks the result is a
        # non-leaf requiring grad with matching metadata.
        x = torch.randn(4, requires_grad=True)
        y = x.neg()
        to_meta = MetaConverter()
        m = to_meta(y)

        # check the test is actually testing what it claims
        self.assertFalse(m.is_leaf)
        self.assertTrue(m.requires_grad)

        self.assertMetadataMatches(m, y)

    def test_requires_grad_false(self):
        # [FUNCTION DOC] Converts a tensor not requiring grad and checks the result does not
        # require grad and metadata matches.
        x = torch.randn(4, requires_grad=False)
        to_meta = MetaConverter()
        m = to_meta(x)

        # check the test is actually testing what it claims
        self.assertFalse(m.requires_grad)

        self.assertMetadataMatches(m, x)

    def test_channels_last(self):
        # [FUNCTION DOC] Converts a channels-last formatted empty tensor and checks the result is
        # a leaf with matching metadata (including strides).
        x = torch.empty(2, 3, 4, 5, memory_format=torch.channels_last)
        to_meta = MetaConverter()
        m = to_meta(x)

        # check the test is actually testing what it claims
        self.assertTrue(m.is_leaf)

        self.assertMetadataMatches(m, x)

    def test_channels_last_leaf(self):
        # [FUNCTION DOC] Converts a channels-last leaf tensor that requires grad and checks the
        # result requires grad, is a leaf, and has matching metadata.
        x = torch.empty(2, 3, 4, 5, memory_format=torch.channels_last, requires_grad=True)
        to_meta = MetaConverter()
        m = to_meta(x)

        # check the test is actually testing what it claims
        self.assertTrue(m.requires_grad)
        self.assertTrue(m.is_leaf)

        self.assertMetadataMatches(m, x)

    def test_channels_last_non_leaf(self):
        # [FUNCTION DOC] Converts a non-leaf result of arithmetic on a channels-last
        # tensor requiring grad; checks strides were preserved by the arithmetic, the meta
        # copy is a non-leaf requiring grad with matching metadata, and that autograd can
        # compute a gradient of its sum with respect to itself without error (regression
        # for a linked issue).
        x = torch.empty(2, 3, 4, 5, memory_format=torch.channels_last, requires_grad=True)
        y = x + 2

        # sanity
        self.assertEqual(x.stride(), y.stride())
        self.assertFalse(y.is_leaf)

        to_meta = MetaConverter()
        m = to_meta(y)

        # check the test is actually testing what it claims
        self.assertTrue(m.requires_grad)
        self.assertFalse(m.is_leaf)

        self.assertMetadataMatches(m, y)

        # Check that we can autograd with m as input without erroring;
        # see https://github.com/pytorch/pytorch/issues/87956
        loss = m.sum()
        torch.autograd.grad(loss, m)

    def test_empty_strided_non_dense_leaf(self):
        # [FUNCTION DOC] Converts a leaf tensor with non-dense strides that requires grad and
        # checks it stays a leaf requiring grad with matching metadata.
        x = torch.empty_strided((2, 2), (4, 2), requires_grad=True)

        to_meta = MetaConverter()
        m = to_meta(x)

        # check the test is actually testing what it claims
        self.assertTrue(m.requires_grad)
        self.assertTrue(m.is_leaf)

        self.assertMetadataMatches(m, x)

    def test_view_mutate(self):
        # [FUNCTION DOC] Converts a view of a zeros tensor and then performs in-place addition of
        # a grad-requiring random tensor on the real view and of a meta grad-requiring tensor on
        # the meta view. Only checks that neither raises.
        x = torch.zeros(4)
        y = x.view(2, 2)

        to_meta = MetaConverter()
        m = to_meta(y)

        y.add_(torch.randn(2, 2, requires_grad=True))
        m.add_(torch.randn(2, 2, device='meta', requires_grad=True))

    def test_non_leaf_torture(self):
        # [FUNCTION DOC] Builds a leaf tensor requiring grad whose storage is re-pointed, under
        # no-grad, with an offset and a different size and stride, then converts it and checks it
        # is a leaf requiring grad with matching metadata.
        x = torch.empty(20, requires_grad=True)
        with torch.no_grad():
            x.set_(x.storage(), 10, (2,), (2,))

        to_meta = MetaConverter()
        m = to_meta(x)

        # check the test is actually testing what it claims
        self.assertTrue(m.requires_grad)
        self.assertTrue(m.is_leaf)

        self.assertMetadataMatches(m, x)

    # NB: complex stuff is not actually exercised right now because
    # we have a blanket exclusion for complex conversion

    def test_view_as_real(self):
        # [FUNCTION DOC] Converts the real-view of a complex tensor and checks metadata
        # matches. Complex conversion is noted in a comment as not really exercised because of
        # a blanket exclusion.
        x = torch.randn(4, dtype=torch.complex64)
        y = torch.view_as_real(x)
        m = MetaConverter()(y)
        self.assertMetadataMatches(m, y)

    def test_complex_noncontiguous_bug(self):
        # [FUNCTION DOC] Regression test: converts a noncontiguous slice of a half-precision
        # complex tensor and checks metadata matches.
        x = torch.randn((2, 2, 4, 9), dtype=torch.complex32)[:, 0, :, :]
        m = MetaConverter()(x)
        self.assertMetadataMatches(m, x)

    def test_view_as_complex(self):
        # [FUNCTION DOC] Converts the complex view of a float tensor with trailing dimension two
        # and checks metadata matches.
        x = torch.randn((4, 2), dtype=torch.float32)
        y = torch.view_as_complex(x)
        m = MetaConverter()(y)
        self.assertMetadataMatches(m, y)

    def test_view_dtype(self):
        # [FUNCTION DOC] Converts a float tensor reinterpreted as 32-bit integers through the
        # dtype view and checks metadata matches.
        x = torch.randn(4, dtype=torch.float32)
        y = x.view(dtype=torch.int32)
        m = MetaConverter()(y)
        self.assertMetadataMatches(m, y)

    def test_imag(self):
        # [FUNCTION DOC] Converts the imaginary-part view of a complex tensor and
        # checks metadata matches.
        x = torch.randn(4, dtype=torch.complex64)
        y = x.imag
        m = MetaConverter()(y)
        self.assertMetadataMatches(m, y)

    def test_inplace_set_storage(self):
        # [FUNCTION DOC] Sets a scalar-shaped meta tensor to use the storage of a real tensor and
        # checks the real storage size did not change.
        x = torch.tensor([0, 1], dtype=torch.int64)
        storage = x.untyped_storage()
        ssize = storage.size()
        meta = torch.empty((), dtype=torch.int64)
        meta.set_(storage, 0, (), ())
        self.assertEqual(storage.size(), ssize)

    @xfailIfTorchDynamo
    def test_weakref(self):
        # [FUNCTION DOC] Checks memoization and lifetime. Converting the same tensor twice
        # returns the same object, and the converter's tensor and storage memos and the describer
        # lookup tables each hold one entry. Deleting the real tensor empties the describer
        # tables but not the memos; deleting the meta tensors empties the memos. Repeats with
        # four tensors held in lists and checks entries drop as lists are deleted. Marked
        # expected failure under Dynamo.
        x = torch.randn(4, 4, 4)
        m = MetaConverter()
        y = m(x)
        z = m(x)
        self.assertIs(y, z)
        self.assertEqual(len(m.tensor_memo), 1)
        self.assertEqual(len(m.storage_memo), 1)
        self.assertEqual(len(m.describer.lookup_tensor), 1)
        self.assertEqual(len(m.describer.lookup_storage), 1)
        del x
        # Entries from Tensor -> int get deallocated when the real tensor
        # disappears...
        self.assertEqual(len(m.describer.lookup_tensor), 0)
        self.assertEqual(len(m.describer.lookup_storage), 0)
        del y
        del z
        # ... but the int -> FakeTensor entries don't die until the fake
        # tensors themselves die (because the user may have held onto the
        # int key and are expecting to get a consistent fake tensor in
        # this case)
        self.assertEqual(len(m.tensor_memo), 0)
        self.assertEqual(len(m.storage_memo), 0)
        li = []
        r = []
        for i in range(4):
            li.append(torch.rand([i]))
            r.append(m(li[-1]))
        self.assertEqual(len(m.tensor_memo), 4)
        self.assertEqual(len(m.storage_memo), 4)
        self.assertEqual(len(m.describer.lookup_tensor), 4)
        self.assertEqual(len(m.describer.lookup_storage), 4)
        del li
        self.assertEqual(len(m.describer.lookup_tensor), 0)
        self.assertEqual(len(m.describer.lookup_storage), 0)
        del r
        self.assertEqual(len(m.tensor_memo), 0)
        self.assertEqual(len(m.storage_memo), 0)

    @xfailIfTorchDynamo
    def test_tensor_outlives_converter(self):
        # [FUNCTION DOC] Checks that deleting a converter after converting a tensor lets the
        # converter be garbage collected (a weak reference to it becomes empty) even though a
        # converted tensor still exists. Marked expected failure under Dynamo.
        m = MetaConverter()
        ref = weakref.ref(m)
        x = torch.randn([4, 4])
        y = m(x)
        del m
        self.assertIs(ref(), None)

aten = torch.ops.aten

CHECK_STRIDES = {
    torch.Tensor.__getitem__,
    aten._fft_c2c.default,
    aten._fft_c2r.default,
    aten._fft_r2c.default,
}

CHECK_ALL_STRIDES = {
    aten.unsqueeze.default
}

CHECK_STRIDES_SKIPS = {
    aten._conj_physical.default,
    aten._linalg_svd.default,
    aten.binary_cross_entropy.default,
    aten.complex.default,
    aten.polar.default,
    aten.copysign.Tensor,
    aten.div.Tensor_mode,
    aten.floor_divide.default,
    aten.heaviside.default,
    aten.ldexp.Tensor,
    aten.lerp.Scalar,
    aten.lerp.Tensor,
    aten.logaddexp.default,
    aten.logical_and.default,
    aten.logical_or.default,
    aten.logical_xor.default,
    aten.pow.Scalar,
    aten.prelu.default,
    aten.special_xlog1py.default,
    aten.xlogy.Tensor,
    aten.nll_loss2d_forward.default,

    # channel_last and channel_last_3d related failures
    aten.convolution.default,

    # following ops fails if include_storage_offset = True, but these are a bit edge casey
    # we should still fix them, leaving them here for tracking.
    # aten._reshape_alias.default,  # repro with test_dispatch_symbolic_meta_outplace_all_strides_matmul_cuda_float32
    # aten.view.default,  # repro with test_dispatch_symbolic_meta_outplace_all_strides_unflatten_cuda_float32
}

CHECK_CONJ_SKIPS = {
    # The conj bit is not copied, see:
    # https://github.com/pytorch/pytorch/pull/101836
    aten.linalg_lu_solve.out,
}

class CheckStrides(Enum):
    # [CLASS DOC] Enumeration of how strictly output strides are compared: not at all, only
    # significant strides (ignoring size-one dimensions and similar), or all strides.
    NONE = 0
    SIGNIFICANT = 1
    ALL = 2

def should_check_strides(func):
    # [FUNCTION DOC] Decides the stride-check level for an operator. Membership in the check-
    # all set gives all, the check set gives significant, the skip set gives none. Anything
    # that is not an operator overload gives none; prims namespace and every remaining
    # overload give significant.
    # KNOWN ISSUE: the view-detection branch and the final fallthrough both return significant,
    # so the view check has no effect (a TODO about tensor iterator is left in the code).
    if func in CHECK_ALL_STRIDES:
        return CheckStrides.ALL
    if func in CHECK_STRIDES:
        return CheckStrides.SIGNIFICANT
    if func in CHECK_STRIDES_SKIPS:
        return CheckStrides.NONE
    if not isinstance(func, torch._ops.OpOverload):
        return CheckStrides.NONE
    # Prims are expected to model strides correctly
    if func.namespace == "prims":
        return CheckStrides.SIGNIFICANT
    # Check if it's a view, by testing if any of the returns have
    # a non-empty alias set
    if any(r.alias_info.before_set for r in func._schema.returns if r.alias_info):
        return CheckStrides.SIGNIFICANT
    # TODO: check for TensorIterator
    return CheckStrides.SIGNIFICANT

def assert_ref_meta_equal(test_case, func, meta_rs, rs, msg_callable):
    # [FUNCTION DOC] Compares real outputs with meta outputs of one operator call. Flattens both
    # with pytree, requires equal leaf counts, then for each real tensor leaf checks the meta
    # leaf is a tensor with the same dtype, shape, storage offset and requires_grad, strides per
    # the stride-check level, and conj and neg bits (conj skipped for operators in the conj skip
    # set). Raises a runtime error with the output index and the message transformed by
    # msg_callable on the first mismatch. Non-tensor leaves are skipped.
    # KNOWN ISSUE: the message for the not-a-tensor check reads that the real result is a Tensor,
    # which is worded confusingly.
    flat_meta_rs = pytree.tree_leaves(meta_rs)
    flat_rs = pytree.tree_leaves(rs)
    test_case.assertEqual(len(flat_meta_rs), len(flat_rs))
    for i, meta_r, r in zip(range(len(flat_rs)), flat_meta_rs, flat_rs):
        def test_assert(cond, msg):
            # [FUNCTION DOC] HELPER: If the condition is false, raises a runtime error naming the
            # current output index and the message passed through the enclosing msg_callable.
            if not cond:
                raise RuntimeError(f"output {i}: {msg_callable(msg)}")
        if not isinstance(r, torch.Tensor):
            continue
        test_assert(isinstance(meta_r, torch.Tensor), f"but real {i}th result is Tensor")
        test_assert(meta_r.dtype == r.dtype, f"for element {i}, was {meta_r.dtype} but real dtype was {r.dtype}")
        test_assert(meta_r.shape == r.shape, f"for element {i}, was {meta_r.shape} but real shape was {r.shape}")
        # See https://github.com/pytorch/pytorch/issues/78050
        if should_check_strides(func) == CheckStrides.ALL:
            same_strides, _ = torch._prims_common.check_all_strides(meta_r, r)
            test_assert(same_strides, f"for element {i}, was {meta_r.stride()} but real stride was {r.stride()}")
        elif should_check_strides(func) == CheckStrides.SIGNIFICANT:
            same_strides, _ = torch._prims_common.check_significant_strides(meta_r, r)
            test_assert(same_strides, f"for element {i}, was {meta_r.stride()} but real stride was {r.stride()}")
        test_assert(
            meta_r.storage_offset() == r.storage_offset(),
            f"for element {i}, was {meta_r.storage_offset()} but real storage_offset was {r.storage_offset()}")
        test_assert(meta_r.requires_grad == r.requires_grad,
                    f"for element {i}, was {meta_r.requires_grad} but real requires_grad was {r.requires_grad}")
        if func not in CHECK_CONJ_SKIPS:
            test_assert(meta_r.is_conj() == r.is_conj(),
                        f"for element {i}, was {meta_r.is_conj()} but real is_conj was {r.is_conj()}")
        test_assert(meta_r.is_neg() == r.is_neg(), f"for element {i}, was {meta_r.is_neg()} but real is_neg was {r.is_neg()}")


# This environment variable controls whether or not we print expected failure
# lists at the end of a test suite run.  The intended usage looks like this:
#
# 1. Run `PYTORCH_COLLECT_EXPECT=1 python test/test_meta.py` on a CUDA build
#    of PyTorch that has LAPACK/MAGMA installed.  You can filter `-k test_meta`
#    or `-k test_dispatch_meta` to only focus on one or another list
# 2. Given the printed skip/xfail list, add them to the corresponding lists;
#    torch.* entries go in meta_function and aten.* entries go in meta_dispatch.
#    If there are preexisting entries, you need to merge in the entries.
#
# This is somewhat manual but typically you shouldn't need to do this, unless
# you've made a major change (e.g., added a new dtype to PyTorch) and need to
# refresh the lists.  If you want to do it from scratch, just clear out the
# preexisting lists before running.
#
# WARNING: Python dict literals will silently ignore duplicate keys
COLLECT_EXPECT = os.getenv('PYTORCH_COLLECT_EXPECT', '0') == '1'

seen_succeeded = {}
seen_failed = {}
failed_reasons = defaultdict(set)
def print_seen():
    # [FUNCTION DOC] HELPER: Exit hook used when expectation collection is enabled. For each
    # operator that failed on some dtypes it resolves a printable name, splits failed dtypes into
    # those that never succeeded (expected failures) and those that both failed and succeeded
    # (skips), optionally appends the recorded failure reasons as a comment, sorts the lines and
    # prints two dictionary-style tables for pasting into the expectation lists. Reads module-
    # level global records only.
    expected_failures = []
    skips = []

    def fmt_dtypes(dtypes):
        # [FUNCTION DOC] HELPER: Formats a collection of dtypes as a brace-wrapped comma-
        # separated sorted list of their short abbreviations.
        r = ', '.join(sorted(dtype_abbrs[d] for d in dtypes))
        return '{' + r + '}'

    for op, failed_dtypes in seen_failed.items():
        ops = resolve_name(op)
        succeeded_dtypes = seen_succeeded.get(op, set())
        expected_failures_dtypes = failed_dtypes - succeeded_dtypes
        skips_dtypes = failed_dtypes & succeeded_dtypes
        reasons = ""
        if failed_reasons[op]:
            reasons = "  # " + ", ".join(sorted(failed_reasons[op]))
        if expected_failures_dtypes:
            expected_failures.append(f"    {ops}: {fmt_dtypes(expected_failures_dtypes)},{reasons}")
        if skips_dtypes:
            skips.append(f"    {ops}: {fmt_dtypes(skips_dtypes)},")
    expected_failures.sort()
    skips.sort()
    nl = '\n'
    print(f"""\
expected_failures = {{
{nl.join(expected_failures)}
}}

skips = {{
{nl.join(skips)}
}}
""")
if COLLECT_EXPECT:
    atexit.register(print_seen)

# Success forces pass; failure forces fail; skip unconditionally skips testing
TestExpect = Enum("TestExpect", ("SUCCESS", "XFAILURE", "SKIP"))

# unlike print produce strides
def verbose_print(e):
    # [FUNCTION DOC] HELPER: Returns a string form of a (possibly nested) structure of values
    # where each dense tensor is shown together with its strides; sparse tensors and non-tensors
    # are left as they are.
    class Lit:
        # [CLASS DOC] HELPER: Wrapper around a string whose repr is exactly that string, so a
        # custom text appears unquoted inside a printed container.
        def __init__(self, s):
            # [FUNCTION DOC] HELPER: Stores the text.
            self.s = s

        def __repr__(self):
            # [FUNCTION DOC] HELPER: Returns the stored text unchanged.
            return self.s

    def go(t):
        # [FUNCTION DOC] HELPER: Maps a leaf to a text wrapper including strides when it is a
        # dense tensor, and returns sparse tensors and other values unchanged.
        if is_sparse_any(t):
            return t
        elif isinstance(t, torch.Tensor):
            return Lit(f"{t} stride={t.stride()}")
        else:
            return t

    return repr(tree_map(go, e))

def run_meta_crossref(
    test_case,
    test_expect,
    func,
    args,
    kwargs,
    *,
    dtype,
    device_type,
    run_symbolic_meta: bool
):
    # [FUNCTION DOC] Core cross-reference check for one operator call. Runs the real function on
    # the given args and kwargs (any failure raises an assertion error saying the operator-info
    # sample is broken). Unless the expectation is skip, args are converted to meta tensors first
    # (conversion failure raises a runtime error). If every tensor converted, it patches the meta
    # args for special cases: tensor_split and repeat_interleave without an output size keep real
    # data-dependent args, indexing keeps real integer-8 or boolean index tensors, and list-
    # length ctc_loss forces expected success; a device keyword becomes the meta device. It then
    # runs the function on meta args with warnings suppressed, optionally under the python
    # dispatcher for symbolic meta mode. A failure on meta returns the real result if failure was
    # expected; otherwise it records the dtype as failed (and the missing-kernel reason when the
    # error is a not-implemented one), returns quietly in collect mode, or raises a runtime error
    # describing the call. If meta ran, it compares outputs with the output comparison helper; a
    # mismatch is handled the same way (expected failure returns, collect mode records and
    # returns, else re-raises); a match records success and raises a runtime error for an
    # unexpected success when failure was expected and collect mode is off. Returns the real
    # result. Mutates the module-level seen and reason records.
    to_meta = MetaConverter()
    do_meta = test_expect is not TestExpect.SKIP
    if do_meta:
        try:
            meta_args = tree_map(to_meta, args)
            meta_kwargs = tree_map(to_meta, kwargs)
        except Exception as e:
            raise RuntimeError(
                f"failed to convert args to meta; "
                f"originally (*{args}, **{kwargs})") from e
    try:
        rs = func(*args, **kwargs)
    except Exception as e:
        raise AssertionError("Original OpInfo is broken") from e

    # TODO: also handle cases where func raise an exception

    # For now, only attempt if we managed to convert all tensor types
    # (if any of them failed, we're in a mixed device situation and
    # this isn't well supported)
    if do_meta and to_meta.successful():
        # Special cases
        if func is torch.tensor_split:
            # Use original indices_or_sections, this argument is data dependent
            meta_args = (meta_args[0], args[1]) + meta_args[2:]
        elif func is torch.Tensor.__getitem__:
            # Ensure boolean tensors use original
            if len(args) != 2:
                raise AssertionError(f"expected len(args) == 2, got {len(args)}")
            flat_args = pytree.tree_leaves(args[1])
            flat_meta_args, spec = tree_flatten(meta_args[1])
            flat_new_args = []
            for a, ma in zip(flat_args, flat_meta_args):
                flat_new_args.append(a if isinstance(a, torch.Tensor) and a.dtype in [torch.int8, torch.bool] else ma)
            meta_args = (meta_args[0], tree_unflatten(flat_new_args, spec))
        elif func in (torch.ops.aten.repeat_interleave.Tensor, torch.ops.aten.repeat_interleave.Tensor_out):
            if kwargs.get("output_size", None) is None:
                meta_args = args
                if func is torch.ops.aten.repeat_interleave.Tensor_out:
                    meta_kwargs["out"] = kwargs["out"]
        elif func in (torch.ops.aten.index.Tensor, torch.ops.aten.index.Tensor_out):
            # Don't convert boolean tensors to meta as they will have nonzero
            # called on them
            indices = []
            for meta_index, real_index in zip(meta_args[1], args[1]):
                if meta_index is not None and meta_index.dtype in [torch.int8, torch.bool]:
                    indices.append(real_index)
                else:
                    indices.append(meta_index)
            meta_args = (meta_args[0], indices)
        elif func is torch.nn.functional.ctc_loss and all([isinstance(args[2], list), isinstance(args[3], list)]):
            # torch.ops.aten._ctc_loss.IntList has a meta kernel but
            # torch.ops.aten._ctc_loss.Tensor does not
            test_expect = TestExpect.SUCCESS

        if kwargs.get("device", None) is not None:
            meta_kwargs["device"] = "meta"

        try:
            # Suppress warnings, this doesn't matter for test_meta.py
            # but it does matter if you want to use this decorator
            # for cross-ref testing, as some tests may be looking at
            # errors
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if run_symbolic_meta:
                    # Run the decomps and meta kernels registered
                    # to the python dispatcher instead of the regular dispatcher.
                    # This should be the same set of kernels
                    # that fake tensor runs in dynamic shapes mode.
                    with enable_python_dispatcher():
                        meta_rs = func(*meta_args, **meta_kwargs)
                else:
                    meta_rs = func(*meta_args, **meta_kwargs)
        except Exception as e:
            if test_expect is TestExpect.XFAILURE:
                return rs
            seen_failed.setdefault(func, set()).add(dtype)
            if isinstance(e, NotImplementedError):
                m = RE_NOT_IMPLEMENTED_MSG.search(e.args[0])
                if m:
                    failed_reasons[func].add(m.group(1))
            if COLLECT_EXPECT:
                return rs
            raise RuntimeError(f"""\
failed to run: {resolve_name(func)}(
*{verbose_print(meta_args)},
**{verbose_print(meta_kwargs)}
)""") from e
        else:
            try:
                delim = ',\n  '
                assert_ref_meta_equal(test_case, func, meta_rs, rs, lambda msg: f"""\
meta disagrees with real impl:
{resolve_name(func)}(
  {delim.join(map(verbose_print, meta_args))},
  {delim.join(k + ": " + verbose_print(v) for k, v in meta_kwargs.items())}
) = (
  {verbose_print(meta_rs)}
)
{msg}
""")
            except Exception:
                if test_expect is TestExpect.XFAILURE:
                    return rs
                seen_failed.setdefault(func, set()).add(dtype)
                if COLLECT_EXPECT:
                    return rs
                raise
            else:
                seen_succeeded.setdefault(func, set()).add(dtype)
                if test_expect is TestExpect.XFAILURE and not COLLECT_EXPECT:
                    raise RuntimeError(f"unexpected success {resolve_name(func)} {meta_args} {meta_kwargs}")

    return rs



RE_NOT_IMPLEMENTED_MSG = re.compile(r"Could not run '([^']+)' with arguments ")

meta_function_expected_failures = {
    torch.Tensor.to_sparse : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.allclose : {f64, f16, c128, c64, bf16, f32},
    torch.argwhere : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.corrcoef : {f64, i32, c128, i64, i16, u8, c64, bf16, f16, i8, f32},
    torch.cov : {f64, i32, c128, i64, i16, u8, c64, bf16, i8, f32, f16},
    torch.functional.istft : {f64, c64, c128, f32},
    torch.geqrf : {f64, c64, c128, f32},
    torch.masked_select : {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.nonzero : {f64, i32, c128, i64, i16, c32, f16, u8, c64, bf16, b8, i8, f32},
    torch.Tensor.nonzero : {f64, i32, c128, i64, i16, c32, f16, u8, c64, bf16, b8, i8, f32},
    torch.Tensor.item : {f64, i32, c128, i64, i16, f16, u8, c32, c64, bf16, b8, i8, f32},
    torch.bincount : {i32, i64, u8, i16, i8},
    torch.functional.unique : {f64, i32, i64, u8, i16, f16, bf16, b8, i8, f32, u16, u32, u64},
    torch.functional.unique_consecutive : {f64, i32, i64, u8, i16, f16, bf16, b8, i8, f32, u16, u32, u64},
    torch.histogram : {f64, f32},
    torch.histogramdd : {f64, f32},
    torch.nn.functional.ctc_loss : {f64, f32},
    torch.nn.functional.gaussian_nll_loss : {f16, f64, bf16, f32},
    torch.linalg.lstsq : {f64, f32, c128, c64},
}

meta_function_expected_failures_conditional = {
    torch.repeat_interleave: lambda dtype, *args, **kwargs: (
        not isinstance(kwargs.get("repeats", None), int)
        and (kwargs.get("output_size", None) is None)
    ),
}

"""
# This is some sample code for how we could dump these dicts into YAML
# file for easier reading/writing
import yaml
print(yaml.dump(
  {resolve_name(k): [dtype_abbrs[d] for d in v]
   for k, v in meta_function_expected_failures.items()}, default_flow_style=None))
import sys
sys.exit()
"""

meta_function_skips = {
    torch.Tensor.__rmatmul__ : {bf16, c128, f64, f32, f16, c64},
    torch.Tensor.matmul : {f64, f32, c128, c64},
    torch.functional.atleast_2d : {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.functional.atleast_3d : {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.functional.cartesian_prod : {bf16, i8, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.functional.einsum : {bf16, c128, f64, f32, f16, c64},
    torch.inner : {f16, bf16, i8, i64, u8, c128, f64, i16, f32, i32, c64},
    torch.linalg.matrix_norm : {c128, f32, c64, f64},
    torch.linalg.matrix_rank : {c128, c64},
    torch.linalg.svd : {c128, c64},
    torch.matmul : {bf16, c128, f64, f32, f16, c64},
    torch.nanquantile : {f64, f32},
    torch.narrow : {bf16, i8, i64, u8, c128, b8, f64, i16, i32, f32, f16, c32, c64},
    torch.nn.functional.batch_norm : {f64, f32},
    torch.nn.functional.binary_cross_entropy : {bf16, f64, f32, f16},
    torch.nn.functional.dropout3d : {bf16, f64, f32, f16},
    torch.nn.functional.local_response_norm : {bf16, f64, f32, f16},
    torch.svd : {c128, c64},
    torch.take_along_dim : {bf16, i8, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.vstack : {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.diff : {b8},
    torch.equal : {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.nanmean : {bf16, f64, f32, f16, c32, c64, c128},
    torch.nn.functional.cross_entropy : {bf16, f64, f32},
    torch.nn.functional.nll_loss : {bf16, f64, f32},
    torch.linalg.cond : {c128, c64, f32, f64},
    torch.linalg.vecdot : {bf16, f64, f32, f16},
    torch.empty : {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.Tensor.addbmm_: {bf16, c128, c64, f32, f64, i16, i32, i64, i8, u8},
    torch.nn.functional.one_hot : {i64},
}


meta_function_device_expected_failures = defaultdict(dict)
meta_function_device_expected_failures_only_outplace = defaultdict(dict)
meta_function_device_skips = defaultdict(dict)

meta_function_device_expected_failures['cpu'] = {
    # TODO: The decomps for these batch norm ops return different dtypes depending
    # on the device. We should make this work better with meta tensors.
    torch.native_batch_norm: {bf16, f16},
    torch._native_batch_norm_legit: {bf16, f16},
    torch.ops.aten._batch_norm_with_update: {bf16, f16},
    torch.native_layer_norm: {bf16, f16},
}

meta_function_device_expected_failures['cuda'] = {
    torch.corrcoef: {bf16, f16},  # aten::_local_scalar_dense
    torch.cov: {f16},  # aten::_local_scalar_dense
    torch.functional.unique: {f16},  # aten::_unique2, aten::unique_dim
    torch.functional.unique_consecutive: {f16},  # aten::unique_consecutive
    torch.geqrf: {f32, f64},  # aten::geqrf
}

meta_function_device_skips['cpu'] = {
    # TODO: The decomps for these batch norm ops return different dtypes depending
    # on the device. We should make this work better with meta tensors.
    torch.native_batch_norm: {f32, f64},
    torch._native_batch_norm_legit: {f32, f64},
    torch.ops.aten._batch_norm_with_update: {f32, f64},
}

meta_function_device_skips['cuda'] = {
    torch.inner: {f16},
    torch.linalg.matrix_rank: {f32, f64},
    torch.linalg.svd: {f32, f64},
    torch.nn.functional.cross_entropy: {f16},
    torch.nn.functional.interpolate: {f16},
    torch.nn.functional.nll_loss: {f16},
    torch.svd: {f32, f64},
}

# This is a __torch_function__ mode that, when enabled, interposes every
# Torch API call and runs the operator as normal, and then reruns it
# with meta inputs, and then checks that everything about the output agrees.
# Most of the logic deals with faithfully replicating the original tensor
# as a meta tensor, which is nontrivial because there are a lot of subsystems
# that may potentially be exercised.
#
# That being said, this class is a little overkill for what it is doing in
# this test file (since I could have just inlined __torch_function__ on the
# OpInfo call, and OpInfos generally have very regular inputs), but it will be
# useful for more comprehensive testing e.g., as seen in
# https://github.com/pytorch/pytorch/pull/75994  The big benefit is it is
# A LOT more efficient that torch dispatch mode (at the cost of less coverage)
class MetaCrossRefFunctionMode(torch.overrides.TorchFunctionMode):
    # [CLASS DOC] Torch function mode that, for each intercepted torch API call, runs the real
    # call, reruns it with meta tensors and compares the outputs via the cross-reference helper.
    # Holds the test case, the device type string, the dtype under test and whether the variant
    # is in-place. The expectation (success, expected failure, or skip) for each function is
    # looked up from the module-level function skip and failure tables, including device-specific
    # and conditional tables.
    test_case: TestCase
    device_type: str
    dtype: torch.dtype

    def __init__(self, test_case, *, device, dtype, inplace):
        # [FUNCTION DOC] Stores the test case, the device type derived from the given device, the
        # dtype and the in-place flag.
        self.test_case = test_case
        self.device_type = torch.device(device).type
        self.dtype = dtype
        self.inplace = inplace

    def __torch_function__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Mode hook. While tracing, for script methods, or when the Python
        # dispatch key is excluded in thread-local state, just calls the function normally with
        # no comparison. Otherwise picks the expectation in priority order: skip tables (generic
        # then device), expected-failure tables (generic then device), the conditional failure
        # table (predicate called with the dtype and call arguments), the out-of-place-only
        # device failure table (ignored for in-place variants); default is success. Then runs the
        # cross-reference check with symbolic meta off and returns the real result it produces.
        kwargs = kwargs or {}

        if (
            torch.jit.is_tracing() or isinstance(func, torch.ScriptMethod) or
            # meta converter doesn't work correctly when no_dispatch() is on, so
            # skip running the crossref test in this case
            torch._C._dispatch_tls_local_exclude_set().has(torch._C.DispatchKey.Python)
        ):
            return func(*args, **kwargs)

        if self.dtype in meta_function_skips.get(func, set()):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_function_device_skips[self.device_type].get(func, set()):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_function_expected_failures.get(func, set()):
            test_expect = TestExpect.XFAILURE
        elif self.dtype in meta_function_device_expected_failures[self.device_type].get(func, set()):
            test_expect = TestExpect.XFAILURE
        elif meta_function_expected_failures_conditional.get(func, lambda *_, **__: False)(self.dtype, *args, **kwargs):
            test_expect = TestExpect.XFAILURE
        elif not self.inplace and \
                self.dtype in meta_function_device_expected_failures_only_outplace[self.device_type].get(func, set()):
            test_expect = TestExpect.XFAILURE
        else:
            test_expect = TestExpect.SUCCESS

        return run_meta_crossref(
            self.test_case, test_expect, func, args,
            kwargs, dtype=self.dtype, device_type=self.device_type, run_symbolic_meta=False
        )

# these always fail
meta_dispatch_expected_failures = {
    aten.allclose.default: {f16, bf16, f32, f64, c64, c128},  # NotImplementedError: 'aten::_local_scalar_dense'
    aten.geqrf.default : {c64, c128, f64, f32},
    aten.linalg_lstsq.default : {c64, c128, f64, f32},
    aten.masked_select.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.masked_select.out : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.nonzero.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, c32, b8, i16, u8},
    aten.nonzero.out : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, c32, b8, i16, u8},
    aten._to_sparse.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten._to_sparse.sparse_dim : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten._ctc_loss.Tensor : {f32, f64},  # Shape of second output depends on data.
    aten._histogramdd_bin_edges.default : {f32, f64},
    aten._histogramdd_from_bin_cts.default : {f32, f64},
    aten._histogramdd_from_bin_tensors.default : {f32, f64},
    aten._local_scalar_dense.default : {c32, c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten._unique2.default : {i8, f64, i64, f16, bf16, f32, i32, b8, i16, u8, u16, u32, u64},
    aten.bincount.default : {i64, i8, i32, i16, u8},
    aten.equal.default : {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.histogram.bin_ct : {f32, f64},
    aten.histogram.bins_tensor : {f32, f64},
    aten.unique_consecutive.default : {i8, f64, i64, f16, bf16, f32, i32, b8, i16, u8, u16, u32, u64},
    aten.unique_dim.default : {i8, f64, i64, f16, bf16, f32, i32, b8, i16, u8, u16, u32, u64},
    aten.upsample_nearest3d.vec : {bf16, f32, f64, u8},

}

# these sometimes pass and sometimes fail
meta_dispatch_skips = {
    aten.index.Tensor: {i64, bf16, f16, u8, b8, f32, i8, f64, i16, i32, c32, c64, c128},  # at::nonzero doesn't have a Meta function
    aten._to_copy.default: {i64, bf16, f16, u8, b8, f32, i8, f64, i16, i32, c32, c64, c128},
    aten.empty.memory_format: {b8, bf16, c128, c64, c32, f16, f32, f64, i16, i32, i64, i8, u8},
    aten.addbmm_.default: {bf16, c128, c64, f32, f64, i16, i32, i64, i8, u8},
}

# For CompositeImplicitAutograd functions that fail before hitting the Mode
meta_dispatch_early_skips = set({
    torch.Tensor.float_power_,
    # Errors out in one of the tests, while ProxyTensor passes...
    torch.Tensor.cumprod_,
    torch.Tensor.cumsum_,
})

meta_inplace_skips = set({
    # Errors out in one of the tests, while ProxyTensor passes...
    torch.Tensor.cumprod_,
    torch.Tensor.cumsum_,
})

meta_dispatch_device_expected_failures = defaultdict(dict)
meta_dispatch_device_skips = defaultdict(dict)

meta_dispatch_device_expected_failures['cpu'] = {
    # TODO: The decomps for these batch norm ops return different dtypes depending
    # on the device. We should make this work better with meta tensors.
    aten.native_batch_norm.default: {bf16, f16},
    aten._native_batch_norm_legit.default: {bf16, f16},
    aten._native_batch_norm_legit.no_stats: {bf16, f16},
    aten._batch_norm_with_update.default: {bf16, f16},

    aten.native_layer_norm.default: {bf16, f16},
}

meta_dispatch_device_expected_failures['cuda'] = {
    aten._unique2.default: {f16},  # aten::_unique2
    aten._use_cudnn_ctc_loss.default: {f32, f64},  # aten::_use_cudnn_ctc_loss
    aten._use_cudnn_ctc_loss.Tensor: {f32, f64},  # aten::_use_cudnn_ctc_loss.Tensor
    aten._use_miopen_ctc_loss.default: {f32, f64},  # aten::_use_miopen_ctc_loss
    aten._use_miopen_ctc_loss.Tensor: {f32, f64},  # aten::_use_miopen_ctc_loss.Tensor
    aten.cudnn_grid_sampler.default: {f16, f32, f64},  # aten::cudnn_grid_sampler
    aten.geqrf.default: {f32, f64},  # aten::geqrf
    aten.linalg_eigvalsh.out: {f32, f64},  # aten::linalg_eigvalsh.out
    aten.log_sigmoid_forward.default: {bf16, f16, f64, f32},
    aten.log_sigmoid_forward.output : {bf16, f16, f64, f32},  # aten::log_sigmoid_forward.output
    aten.unique_consecutive.default: {f16},  # aten::unique_consecutive
    aten.unique_dim.default: {f16},  # aten::unique_dim
    aten.upsample_nearest3d.vec: {f16},  # aten::upsample_nearest3d.vec
}

meta_dispatch_device_skips['cpu'] = {
    aten._embedding_bag_forward_only.default: {bf16, f16, f32, f64},

    # TODO: The decomps for these batch norm ops return different dtypes depending
    # on the device. We should make this work better with meta tensors.
    aten.native_batch_norm.default: {f32, f64},
    aten._native_batch_norm_legit.default: {f32, f64},
    aten._native_batch_norm_legit.no_stats: {f32, f64},
    aten._batch_norm_with_update.default: {f32, f64},

    # If the computation dtype is different from the input
    # dtype this will fail. CPU execution may also have a
    # a different output from other devices.
    aten.native_batch_norm.out: {bf16, f16, f32, f64}
}

meta_dispatch_device_skips['cuda'] = {
    aten._conj.default: {c32, f16},  # file issue
    aten._linalg_svd.default: {c64, c128},  # aten::linalg_eigvalsh.out
    aten.cudnn_batch_norm.default: {f32, f64},
    aten.log_softmax.int : {c32, c64},
    aten.softmax.int : {c32, c64},
    aten.softmax.int : {c32, c64},

    # ROCm stuff; technically this should be expected failure but it's
    # not worth it; these should get unified anyway
    aten.miopen_batch_norm.default: {f32},
}

def get_strided_args(args):
    # [FUNCTION DOC] Generator that yields tuples of argument lists to try different memory
    # layouts. For each argument that is a dense, contiguous, non-sparse-compressed tensor it
    # builds variants, and every other argument gets a single-item variant. Yields the cartesian
    # product over all arguments.

    def get_strided_variants(t, include_storage_offset=False):
        # [FUNCTION DOC] HELPER: Returns a list of layouts of a tensor: itself, a transposed-
        # layout copy when it has more than one dimension, a non-dense strided one (every other
        # element along the last dimension) when it has at least one dimension, channels-last for
        # four-dimensional, channels-last-3d for five-dimensional, and when requested a copy
        # placed at storage offset one.
        variants = []

        # contiguous
        variants.append(t)

        # transposed
        if t.ndim > 1:
            perm = list(reversed(range(t.ndim)))
            transposed = torch.empty(
                t.shape[::-1], device=t.device, dtype=t.dtype, requires_grad=t.requires_grad
            ).permute(perm).copy_(t)
            variants.append(transposed)

        # nondense
        if t.ndim > 0:
            nondense = torch.repeat_interleave(t, 2, dim=-1)[..., ::2]
            variants.append(nondense)

        # channel_last
        if t.ndim == 4:
            variants.append(t.contiguous(memory_format=torch.channels_last))

        # channel_last_3d
        if t.ndim == 5:
            variants.append(t.contiguous(memory_format=torch.channels_last_3d))

        # storage_offset
        if include_storage_offset:
            buffer = torch.empty(t.numel() + 1, device=t.device, dtype=t.dtype, requires_grad=t.requires_grad)
            buffer = buffer.as_strided(t.shape, t.stride(), storage_offset=1)
            buffer.copy_(t)
            variants.append(buffer)

        return variants

    strided_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor) and not arg.is_sparse_csr and arg.is_contiguous():
            strided_arg_variants = get_strided_variants(arg)
        else:
            strided_arg_variants = [arg]
        strided_args.append(strided_arg_variants)

    yield from itertools.product(*strided_args)

class MetaCrossRefDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    # [CLASS DOC] Torch dispatch mode performing the same real-versus-meta comparison as the
    # function mode but at the operator-overload level. Attributes: test case, device type,
    # dtype, flags for symbolic meta, in-place and whether the operator supports an out argument,
    # and the precision and relative tolerance saved from the test case. A class-level set
    # remembers operator packets known to have no out overload. Also checks out-overloads of
    # operators that have none at the user level.
    test_case: TestCase
    device: torch.device
    dtype: torch.dtype
    aten_olp_no_out_overload: set = set()

    def __init__(self, test_case, *, device, dtype, symbolic_meta: bool, inplace: bool, supports_out: bool):
        # [FUNCTION DOC] Stores the test case, the precision and relative tolerance read from it
        # (saved because thread-local state is not visible later), the device type from the
        # device, the dtype and the symbolic meta, in-place and supports-out flags.
        self.test_case = test_case
        # save TLS
        self.precision = test_case.precision
        self.rel_tol = test_case.rel_tol
        self.device_type = torch.device(device).type
        self.dtype = dtype
        self.symbolic_meta = symbolic_meta
        self.inplace = inplace
        self.supports_out = supports_out

    @staticmethod
    def try_resolve_aten_out_overload(ol, args, kwargs, num_outputs):
        # [FUNCTION DOC] Finds an out-variant overload that matches a call to the given
        # overload. Returns a triple of the overload, the names of its out arguments and a
        # keyword dictionary, or a triple of nothing when none is found. Packets already known
        # to lack out overloads return early, and are added to the class-level cache when no
        # overload of the packet takes an out argument. A candidate matches when it has more
        # arguments than positional args supplied, the positional argument types equal, its
        # trailing out arguments count equals num_outputs, and each remaining argument is either
        # given in kwargs, has a default, or is optional (optional boolean gets false, other
        # optional gets nothing).

        ol_args = ol._schema.arguments
        olp: OpOverloadPacket = ol._overloadpacket

        if olp in MetaCrossRefDispatchMode.aten_olp_no_out_overload:
            return (None, None, None)

        candidate_ols = []
        for candidate_ol in olp.op_overloads():
            if any(arg.is_out for arg in candidate_ol._schema.arguments):
                candidate_ols.append(candidate_ol)

        if not candidate_ols:
            MetaCrossRefDispatchMode.aten_olp_no_out_overload.add(olp)
            return (None, None, None)

        # Now match based on args, kwargs and number of required outputs
        candidate_ol: OpOverload = None
        for candidate_ol in candidate_ols:
            candidate_ol_args = candidate_ol._schema.arguments

            if (len(args) >= len(candidate_ol_args)):
                continue

            # Positional arguments must have the same type
            if not all(
                ol_args[pos_arg_ind].type == candidate_ol_args[pos_arg_ind].type
                for pos_arg_ind in range(len(args))
            ):
                continue

            # Number of outputs must match
            candidate_out_names = [out_arg.name for out_arg in candidate_ol_args[-num_outputs:] if out_arg.is_out]
            if len(candidate_out_names) != num_outputs:
                continue

            # Now try and match kwargs. Just need to ensure that the
            # remaining kwargs allow an out overload to be called. For example
            # we can throw away parameters like `dtype` that may be passed to the
            # functional version of the op since the `dtype` will already be present
            # in the `out` argument
            new_kwargs = {}
            kwargs_match = True
            for arg in candidate_ol_args[len(args):-num_outputs]:
                if arg.name not in kwargs:
                    if arg.has_default_value():
                        new_kwargs[arg.name] = arg.default_value
                    elif isinstance(arg.type, torch.OptionalType):
                        if isinstance(arg.type.getElementType(), torch.BoolType):
                            new_kwargs[arg.name] = False
                        else:
                            new_kwargs[arg.name] = None
                    else:
                        kwargs_match = False
                        break
                else:
                    new_kwargs[arg.name] = kwargs[arg.name]

            if kwargs_match:
                return candidate_ol, candidate_out_names, new_kwargs

        return None, None, None

    def _get_expected_test_result(self, func: OpOverload):
        # [FUNCTION DOC] Looks up expectation for an operator overload: skip if the dtype is in
        # the generic or device skip tables, expected failure if in the failure tables, else
        # success. Returns the expectation value.
        if self.dtype in meta_dispatch_skips.get(func, set()):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_dispatch_device_skips[self.device_type].get(func, set()):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_dispatch_expected_failures.get(func, set()):
            test_expect = TestExpect.XFAILURE
        elif self.dtype in meta_dispatch_device_expected_failures[self.device_type].get(func, set()):
            test_expect = TestExpect.XFAILURE
        else:
            test_expect = TestExpect.SUCCESS
        return test_expect

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Mode hook. Restores the saved precision and tolerance on the test case,
        # determines the expectation and runs the cross-reference check with this mode's symbolic
        # setting to get the real result. If the variant is out-of-place, the operator lacks an
        # out argument, the expectation is success and the result is a tensor or iterable, it
        # tries to find a matching out overload; if found it fills the out arguments with the
        # real result (one tensor or one per output), looks up that overload's expectation and
        # runs the cross-reference check on it too. Returns the real result.
        kwargs = kwargs or {}
        self.test_case.precision = self.precision
        self.test_case.rel_tol = self.rel_tol

        test_expect = self._get_expected_test_result(func)

        expected = run_meta_crossref(
            self.test_case,
            test_expect,
            func,
            args,
            kwargs,
            dtype=self.dtype,
            device_type=self.device_type,
            run_symbolic_meta=self.symbolic_meta,
        )

        # This is to test torch ops that do not have an out parameter but have
        # aten op overloads that have out parameters. Additionally, Python decompositions
        # may register OpOverloadPacket's so decompositions need to be tested
        # to ensure all OpOverloads still function for the Meta key (e.g. if a python decomposition
        # is registered for an aten op aten.foo with overloads [default, out], the python
        # function needs to support receiving `out` arguments)
        if (
            not self.inplace and
            not self.supports_out and
            test_expect == TestExpect.SUCCESS and
            (torch.is_tensor(expected) or isinstance(expected, Iterable))
        ):

            # check to see if there is a potential out overload
            num_outputs = 1 if torch.is_tensor(expected) else len(expected)
            func_out_overload, out_param_names, kwargs = self.try_resolve_aten_out_overload(func, args, kwargs, num_outputs)

            if func_out_overload:

                if num_outputs == 1:
                    kwargs[out_param_names[0]] = expected
                else:
                    for ind, out_param_name in enumerate(out_param_names):
                        kwargs[out_param_name] = expected[ind]

                test_expect = self._get_expected_test_result(func_out_overload)

                run_meta_crossref(
                    self.test_case,
                    test_expect,
                    func_out_overload,
                    args,
                    kwargs,
                    dtype=self.dtype,
                    device_type=self.device_type,
                    run_symbolic_meta=self.symbolic_meta,
                )

        return expected

# NB: we're running these tests only on CUDA because there are some
# inconsistencies between CUDA and CPU, and running on CUDA makes it easier
# to ignore the CPU case when inconsistencies arise.  Ideally we deal
# with the inconsistencies but this takes time.
@unMarkDynamoStrictTest
class TestMeta(TestCase):
    # [CLASS DOC] Operator-database-driven meta tests, generated per device and dtype. Out-of-
    # place and in-place tests run each sample input under the function mode; dispatch tests run
    # them under the dispatch mode, with and without symbolic meta, and with all stride layouts
    # on CUDA only. Further tests check specific meta kernels directly (quantized empty,
    # nan_to_num, broadcast errors, custom op without autograd kernel, huber and normalization
    # backward shapes). Per-operator expected failures and skips are given by decorators and the
    # module-level tables.
    # Copies inputs to inplace operations to avoid inplace modifications
    #   to leaves requiring gradient
    def _get_safe_inplace(self, inplace_variant):
        # [FUNCTION DOC] HELPER: Wraps an in-place callable so that its first argument (a
        # tensor, or each tensor of a list) is cloned before the call, protecting leaves that
        # require grad from in-place modification. Returns the wrapper, which keeps the
        # metadata of the original.
        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the wrapped in-place function on clones of the first
            # argument (a list is cloned element-wise) with the remaining arguments unchanged,
            # returning its result.
            if isinstance(t, list):
                return inplace_variant([x.clone() for x in t], *args, **kwargs)
            else:
                return inplace_variant(t.clone(), *args, **kwargs)

        return _fn

    @skipIfCrossRef
    @suppress_warnings
    @skipOps((
        skip('sparse.sampled_addmm'),
        skip('sparse.mm', variant_name='reduce'),
        skip('to'),
        xfail('_foreach_neg', dtypes=(b8,)),
        xfail('_foreach_ceil', dtypes=complex_types_and(b8)),
        xfail('_foreach_erf', dtypes=complex_types()),
        xfail('_foreach_erfc', dtypes=complex_types()),
        xfail('_foreach_floor', dtypes=complex_types_and(b8)),
        xfail('_foreach_round', dtypes=complex_types_and(b8)),
        xfail('_foreach_frac', dtypes=integral_types_and(b8) + complex_types()),
        xfail('_foreach_trunc', dtypes=complex_types_and(b8)),
        xfail('_foreach_sign', dtypes=complex_types()),
        xfail('_foreach_lgamma', dtypes=complex_types()),
        xfail('_foreach_add'),
        xfail('_foreach_sub'),
        xfail('_foreach_clamp_min', dtypes=complex_types_and(b8)),
        xfail('_foreach_clamp_max', dtypes=complex_types_and(b8)),
        xfail('_foreach_minimum', dtypes=complex_types_and(b8)),
        xfail('_foreach_maximum', dtypes=complex_types_and(b8)),
        xfail('_foreach_pow', dtypes=(b8,)),
        xfail('_foreach_addcmul', dtypes=(b8,)),
        xfail('_foreach_addcdiv', dtypes=integral_types_and(b8)),
        xfail('_foreach_max', dtypes=(c128, c64)),
        xfail('_foreach_norm', dtypes=integral_types_and(b8)),
        xfail('_foreach_lerp', dtypes=integral_types_and(b8)),
    ))
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_meta_outplace(self, device, dtype, op):
        # [FUNCTION DOC] Out-of-place cross-reference test for one operator and dtype. Skips
        # scaled matrix multiply operators (no meta support) and, under Dynamo, a few flaky
        # operators. For each sample input it calls the operator inside the function mode and,
        # when the result is a tensor and the operator supports an out argument, calls it again
        # with that result as output. For samples with a device keyword on a like-operator, it
        # also runs with a fixed random seed once as is and once with the first argument moved to
        # the meta device, and checks the two results are equal (except for empty_like, which is
        # not deterministic). Raises if the first argument of such a sample is not a tensor.
        if "_scaled_mm" in op.name:
            raise unittest.SkipTest("_scaled_mm dose not support meta device")
        skip_op_names = (
            "fft.ihfft",
            "fft.ihfft2",
            "linalg.lu_solve",
        )
        if TEST_WITH_TORCHDYNAMO and op.name in skip_op_names:
            raise unittest.SkipTest("flaky")
        # run the OpInfo sample inputs, cross-referencing them with the
        # meta implementation and check the results are the same.  All
        # the heavy lifting happens in MetaCrossRefFunctionMode
        func = op.get_op()
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in samples:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            with MetaCrossRefFunctionMode(self, dtype=dtype, device=device, inplace=False):
                expected = func(*args, **kwargs)
                if isinstance(expected, torch.Tensor) and op.supports_out:
                    func(*args, **kwargs, out=expected)

            # Special test for functions taking "device" kwarg
            # The crossref tests that replacing the device with "meta" works
            # This part makes sure that *_like functions work well with a "meta"
            # Tensor and their original device argument.
            if "device" in kwargs and "_like" in op.name:
                with torch.random.fork_rng():
                    torch.manual_seed(123)
                    ref = func(*args, **kwargs)

                # *_like functions take a Tensor as first argument
                if not isinstance(args[0], torch.Tensor):
                    raise AssertionError(f"expected args[0] to be Tensor, got {type(args[0])}")
                with torch.random.fork_rng():
                    torch.manual_seed(123)
                    args[0] = args[0].to(device="meta")
                    meta = func(*args, **kwargs)

                # empty_like is not deterministic
                if op.name != "empty_like":
                    self.assertEqual(ref, meta)

    @skipIfCrossRef
    @suppress_warnings
    @skipOps((
        xfail('abs', dtypes=(c128, c64, c32)),
        xfail('as_strided', variant_name='partial_views'),
        xfail('float_power', dtypes=[bf16, f16, f32]),
        xfail('square', dtypes=(b8,)),
        xfail('_foreach_exp', dtypes=integral_types_and(b8)),
        xfail('_foreach_acos', dtypes=integral_types_and(b8)),
        xfail('_foreach_asin', dtypes=integral_types_and(b8)),
        xfail('_foreach_atan', dtypes=integral_types_and(b8)),
        xfail('_foreach_cos', dtypes=integral_types_and(b8)),
        xfail('_foreach_cosh', dtypes=integral_types_and(b8)),
        xfail('_foreach_log', dtypes=integral_types_and(b8)),
        xfail('_foreach_log10', dtypes=integral_types_and(b8)),
        xfail('_foreach_log2', dtypes=integral_types_and(b8)),
        xfail('_foreach_tan', dtypes=integral_types_and(b8)),
        xfail('_foreach_tanh', dtypes=integral_types_and(b8)),
        xfail('_foreach_sin', dtypes=integral_types_and(b8)),
        xfail('_foreach_sinh', dtypes=integral_types_and(b8)),
        xfail('_foreach_neg', dtypes=(b8,)),
        xfail('_foreach_sqrt', dtypes=integral_types_and(b8)),
        xfail('_foreach_rsqrt', dtypes=integral_types_and(b8)),
        xfail('_foreach_ceil', dtypes=complex_types_and(b8)),
        xfail('_foreach_erf', dtypes=integral_types_and(b8) + complex_types()),
        xfail('_foreach_erfc', dtypes=integral_types_and(b8) + complex_types()),
        xfail('_foreach_expm1', dtypes=integral_types_and(b8)),
        xfail('_foreach_floor', dtypes=complex_types_and(b8)),
        xfail('_foreach_log1p', dtypes=integral_types_and(b8)),
        xfail('_foreach_round', dtypes=complex_types_and(b8)),
        xfail('_foreach_frac', dtypes=integral_types_and(b8) + complex_types()),
        xfail('_foreach_reciprocal', dtypes=integral_types_and(b8)),
        xfail('_foreach_sigmoid', dtypes=integral_types_and(b8)),
        xfail('_foreach_trunc', dtypes=complex_types_and(b8)),
        xfail('_foreach_abs', dtypes=complex_types()),
        xfail('_foreach_sign', dtypes=complex_types()),
        skip('_foreach_lgamma', dtypes=integral_types_and(b8)),
        xfail('_foreach_lgamma', dtypes=complex_types() + integral_types_and(b8)),
        xfail('_foreach_add', dtypes=all_types_and_complex_and(b8, bf16, f16)),
        xfail('_foreach_sub'),
        xfail('_foreach_mul', dtypes=(b8,)),
        xfail('_foreach_div', dtypes=integral_types_and(b8)),
        xfail('_foreach_clamp_min', dtypes=complex_types_and(b8)),
        xfail('_foreach_clamp_max', dtypes=complex_types_and(b8)),
        xfail('_foreach_minimum', dtypes=complex_types_and(b8)),
        xfail('_foreach_maximum', dtypes=complex_types_and(b8)),
        xfail('_foreach_pow', dtypes=(b8,)),
        xfail('_foreach_addcmul', dtypes=(b8,)),
        xfail('_foreach_addcdiv', dtypes=integral_types_and(b8)),
        xfail('_foreach_norm'),
        xfail('_foreach_lerp', dtypes=integral_types_and(b8)),
    ))
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_meta_inplace(self, device, dtype, op):
        # [FUNCTION DOC] In-place cross-reference test. Skips if the operator has no in-place
        # variant, if it promotes integers to float and the dtype is not floating point, or if
        # it is in the in-place skip set. Wraps the in-place function so inputs are cloned,
        # then for every sample not broadcasting its input runs it inside the function mode
        # with the in-place flag on.
        func = op.get_inplace()
        if not func:
            self.skipTest("No inplace variable for this op")
        if op.promotes_int_to_float and not dtype.is_floating_point:
            self.skipTest("Op promotes to float, which is impossible for inplace with non-float input")
        if func in meta_inplace_skips:
            self.skipTest("Skipped")
        func = self._get_safe_inplace(func)
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in samples:
            if sample_input.broadcasts_input:
                continue
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            with MetaCrossRefFunctionMode(self, dtype=dtype, device=device, inplace=True):
                expected = func(*args, **kwargs)

    def _run_dispatch_meta_test(self, device, dtype, op, symbolic_meta, inplace, all_stride_variants=False):
        # [FUNCTION DOC] HELPER: Shared driver for the dispatch-mode tests. Skips scaled matrix
        # multiply operators, picks the in-place or out-of-place function (skipping the test when
        # no in-place variant exists or integers would be promoted to float), skips functions in
        # the dispatch early-skip set, and wraps in-place functions to clone inputs. For each
        # sample (in-place samples that broadcast the input are skipped) it optionally expands
        # the arguments into stride layout variants (only when five or fewer tensors are
        # present), and for each variant calls the function with the dispatch mode pushed using
        # the given symbolic, in-place and supports-out settings; for tensor results of out-of-
        # place operators supporting out it also calls with the result as the output.
        if "_scaled_mm" in op.name:
            raise unittest.SkipTest("_scaled_mm dose not support meta device")
        if inplace:
            func = op.get_inplace()
            if not func:
                self.skipTest("No inplace variable for this op")
            if op.promotes_int_to_float and not dtype.is_floating_point:
                self.skipTest("Op promotes to float, which is impossible for inplace with non-float input")
        else:
            func = op.get_op()

        if func in meta_dispatch_early_skips:
            self.skipTest("Function is in dispatch early skips")

        if inplace:
            func = self._get_safe_inplace(func)

        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in samples:
            if inplace and sample_input.broadcasts_input:
                continue

            sample_args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs

            if all_stride_variants and sum(isinstance(arg, torch.Tensor) for arg in sample_args) <= 5:
                # test inputs <= 5 tensors to avoid combinatorial explosion
                strided_args = get_strided_args(sample_args)
            else:
                strided_args = [sample_args]

            for args in strided_args:
                with MetaCrossRefDispatchMode.push(
                    self, dtype=dtype, device=device,
                    symbolic_meta=symbolic_meta, inplace=inplace,
                     supports_out=op.supports_out):
                    expected = func(*args, **kwargs)

                    if not inplace and isinstance(expected, torch.Tensor) and op.supports_out:
                        func(*args, **kwargs, out=expected)


    @skipIfCrossRef
    @suppress_warnings
    @skipOps((
        skip('sparse.sampled_addmm'),
        skip('sparse.mm', variant_name='reduce'),
        xfail('nn.functional.binary_cross_entropy'),
        xfail('empty_strided'),
        xfail('_foreach_neg', dtypes=(b8,)),
        xfail('_foreach_ceil', dtypes=complex_types_and(b8)),
        xfail('_foreach_erf', dtypes=complex_types()),
        xfail('_foreach_erfc', dtypes=complex_types()),
        xfail('_foreach_floor', dtypes=complex_types_and(b8)),
        xfail('_foreach_round', dtypes=complex_types_and(b8)),
        xfail('_foreach_frac', dtypes=integral_types_and(b8) + complex_types()),
        xfail('_foreach_trunc', dtypes=complex_types_and(b8)),
        xfail('_foreach_sign', dtypes=complex_types()),
        xfail('_foreach_lgamma', dtypes=complex_types()),
        xfail('_foreach_sub'),
        xfail('_foreach_clamp_min', dtypes=complex_types_and(b8)),
        xfail('_foreach_clamp_max', dtypes=complex_types_and(b8)),
        xfail('_foreach_minimum', dtypes=complex_types_and(b8)),
        xfail('_foreach_maximum', dtypes=complex_types_and(b8)),
        xfail('_foreach_pow', dtypes=(b8,)),
        xfail('_foreach_addcmul', dtypes=(b8,)),
        xfail('_foreach_addcdiv', dtypes=integral_types_and(b8)),
        xfail('_foreach_max', dtypes=(c128, c64)),
        xfail('_foreach_norm', dtypes=integral_types_and(b8)),
        xfail('_foreach_lerp', dtypes=integral_types_and(b8)),
    ))
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_dispatch_meta_outplace(self, device, dtype, op):
        # [FUNCTION DOC] Dispatch-mode meta test for the out-of-place variant with regular (non-
        # symbolic) meta kernels; delegates to the shared driver.
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=False, inplace=False)

    @skipIfCrossRef
    @suppress_warnings
    @skipOps((
        xfail('abs', dtypes=(c128, c64, c32)),
        xfail('as_strided', variant_name='partial_views'),
        xfail('square', dtypes=(b8,)),
        xfail('_foreach_exp', dtypes=integral_types_and(b8)),
        xfail('_foreach_acos', dtypes=integral_types_and(b8)),
        xfail('_foreach_asin', dtypes=integral_types_and(b8)),
        xfail('_foreach_atan', dtypes=integral_types_and(b8)),
        xfail('_foreach_cos', dtypes=integral_types_and(b8)),
        xfail('_foreach_cosh', dtypes=integral_types_and(b8)),
        xfail('_foreach_log', dtypes=integral_types_and(b8)),
        xfail('_foreach_log10', dtypes=integral_types_and(b8)),
        xfail('_foreach_log2', dtypes=integral_types_and(b8)),
        xfail('_foreach_tan', dtypes=integral_types_and(b8)),
        xfail('_foreach_tanh', dtypes=integral_types_and(b8)),
        xfail('_foreach_sin', dtypes=integral_types_and(b8)),
        xfail('_foreach_sinh', dtypes=integral_types_and(b8)),
        xfail('_foreach_neg', dtypes=(b8,)),
        xfail('_foreach_sqrt', dtypes=integral_types_and(b8)),
        xfail('_foreach_rsqrt', dtypes=integral_types_and(b8)),
        xfail('_foreach_ceil', dtypes=complex_types_and(b8)),
        xfail('_foreach_erf', dtypes=integral_types_and(b8) + complex_types()),
        xfail('_foreach_erfc', dtypes=integral_types_and(b8) + complex_types()),
        xfail('_foreach_expm1', dtypes=integral_types_and(b8)),
        xfail('_foreach_floor', dtypes=complex_types_and(b8)),
        xfail('_foreach_log1p', dtypes=integral_types_and(b8)),
        xfail('_foreach_round', dtypes=complex_types_and(b8)),
        xfail('_foreach_frac', dtypes=integral_types_and(b8) + complex_types()),
        xfail('_foreach_reciprocal', dtypes=integral_types_and(b8)),
        xfail('_foreach_sigmoid', dtypes=integral_types_and(b8)),
        xfail('_foreach_trunc', dtypes=complex_types_and(b8)),
        xfail('_foreach_abs', dtypes=complex_types()),
        xfail('_foreach_sign', dtypes=complex_types()),
        xfail('_foreach_lgamma', dtypes=complex_types() + integral_types_and(b8)),
        xfail('_foreach_add', dtypes=(b8,)),
        xfail('_foreach_sub'),
        xfail('_foreach_mul', dtypes=(b8,)),
        xfail('_foreach_div', dtypes=integral_types_and(b8)),
        xfail('_foreach_clamp_min', dtypes=complex_types_and(b8)),
        xfail('_foreach_clamp_max', dtypes=complex_types_and(b8)),
        xfail('_foreach_minimum', dtypes=complex_types_and(b8)),
        xfail('_foreach_maximum', dtypes=complex_types_and(b8)),
        xfail('_foreach_pow', dtypes=(b8,)),
        xfail('_foreach_addcmul', dtypes=(b8,)),
        xfail('_foreach_addcdiv', dtypes=integral_types_and(b8)),
        xfail('_foreach_norm'),
        xfail('_foreach_lerp', dtypes=integral_types_and(b8)),
    ))
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_dispatch_meta_inplace(self, device, dtype, op):
        # [FUNCTION DOC] Dispatch-mode meta test for the in-place variant with regular (non-
        # symbolic) meta kernels; delegates to the shared driver.
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=False, inplace=True)

    @skipIfCrossRef
    @suppress_warnings
    @skipOps((
        skip('sparse.sampled_addmm'),
        skip('sparse.mm', variant_name='reduce'),
        xfail('nn.functional.binary_cross_entropy'),
        xfail('empty_strided'),
        xfail('_foreach_neg', dtypes=(b8,)),
        xfail('_foreach_ceil', dtypes=complex_types_and(b8)),
        xfail('_foreach_erf', dtypes=complex_types()),
        xfail('_foreach_erfc', dtypes=complex_types()),
        xfail('_foreach_floor', dtypes=complex_types_and(b8)),
        xfail('_foreach_round', dtypes=complex_types_and(b8)),
        xfail('_foreach_frac', dtypes=integral_types_and(b8) + complex_types()),
        xfail('_foreach_trunc', dtypes=complex_types_and(b8)),
        xfail('_foreach_sign', dtypes=complex_types()),
        xfail('_foreach_lgamma', dtypes=complex_types()),
        xfail('_foreach_sub'),
        xfail('_foreach_clamp_min', dtypes=complex_types_and(b8)),
        xfail('_foreach_clamp_max', dtypes=complex_types_and(b8)),
        xfail('_foreach_minimum', dtypes=complex_types_and(b8)),
        xfail('_foreach_maximum', dtypes=complex_types_and(b8)),
        xfail('_foreach_pow', dtypes=(b8,)),
        xfail('_foreach_addcmul', dtypes=(b8,)),
        xfail('_foreach_addcdiv', dtypes=integral_types_and(b8)),
        xfail('_foreach_max', dtypes=(c128, c64)),
        xfail('_foreach_norm', dtypes=integral_types_and(b8)),
        xfail('_foreach_lerp', dtypes=integral_types_and(b8)),
    ))
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_dispatch_symbolic_meta_outplace(self, device, dtype, op):
        # [FUNCTION DOC] Dispatch-mode meta test for the out-of-place variant using symbolic meta
        # (python dispatcher); delegates to the shared driver.
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=False)


    @skipIfCrossRef
    @suppress_warnings
    @skipOps((
        xfail('abs', dtypes=(c128, c64, c32)),
        xfail('as_strided', variant_name='partial_views'),
        xfail('square', dtypes=(b8,)),
        xfail('_foreach_exp', dtypes=integral_types_and(b8)),
        xfail('_foreach_acos', dtypes=integral_types_and(b8)),
        xfail('_foreach_asin', dtypes=integral_types_and(b8)),
        xfail('_foreach_atan', dtypes=integral_types_and(b8)),
        xfail('_foreach_cos', dtypes=integral_types_and(b8)),
        xfail('_foreach_cosh', dtypes=integral_types_and(b8)),
        xfail('_foreach_log', dtypes=integral_types_and(b8)),
        xfail('_foreach_log10', dtypes=integral_types_and(b8)),
        xfail('_foreach_log2', dtypes=integral_types_and(b8)),
        xfail('_foreach_tan', dtypes=integral_types_and(b8)),
        xfail('_foreach_tanh', dtypes=integral_types_and(b8)),
        xfail('_foreach_sin', dtypes=integral_types_and(b8)),
        xfail('_foreach_sinh', dtypes=integral_types_and(b8)),
        xfail('_foreach_neg', dtypes=(b8,)),
        xfail('_foreach_sqrt', dtypes=integral_types_and(b8)),
        xfail('_foreach_rsqrt', dtypes=integral_types_and(b8)),
        xfail('_foreach_ceil', dtypes=complex_types_and(b8)),
        xfail('_foreach_erf', dtypes=integral_types_and(b8) + complex_types()),
        xfail('_foreach_erfc', dtypes=integral_types_and(b8) + complex_types()),
        xfail('_foreach_expm1', dtypes=integral_types_and(b8)),
        xfail('_foreach_floor', dtypes=complex_types_and(b8)),
        xfail('_foreach_log1p', dtypes=integral_types_and(b8)),
        xfail('_foreach_round', dtypes=complex_types_and(b8)),
        xfail('_foreach_frac', dtypes=integral_types_and(b8) + complex_types()),
        xfail('_foreach_reciprocal', dtypes=integral_types_and(b8)),
        xfail('_foreach_sigmoid', dtypes=integral_types_and(b8)),
        xfail('_foreach_trunc', dtypes=complex_types_and(b8)),
        xfail('_foreach_abs', dtypes=complex_types()),
        xfail('_foreach_sign', dtypes=complex_types()),
        skip('_foreach_lgamma', dtypes=integral_types_and(b8)),
        xfail('_foreach_lgamma', dtypes=complex_types() + integral_types_and(b8)),
        xfail('_foreach_add', dtypes=(b8,)),
        xfail('_foreach_sub'),
        xfail('_foreach_mul', dtypes=(b8,)),
        xfail('_foreach_div', dtypes=integral_types_and(b8)),
        xfail('_foreach_clamp_min', dtypes=complex_types_and(b8)),
        xfail('_foreach_clamp_max', dtypes=complex_types_and(b8)),
        xfail('_foreach_minimum', dtypes=complex_types_and(b8)),
        xfail('_foreach_maximum', dtypes=complex_types_and(b8)),
        xfail('_foreach_pow', dtypes=(b8,)),
        xfail('_foreach_addcmul', dtypes=(b8,)),
        xfail('_foreach_addcdiv', dtypes=integral_types_and(b8)),
        xfail('_foreach_norm'),
        xfail('_foreach_lerp', dtypes=integral_types_and(b8)),
    ))
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_dispatch_symbolic_meta_inplace(self, device, dtype, op):
        # [FUNCTION DOC] Dispatch-mode meta test for the in-place variant using symbolic meta
        # (python dispatcher); delegates to the shared driver.
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=True)

    @skipIfCrossRef
    @suppress_warnings
    # only test one dtype, as output stride behavior is the same for all dtypes
    @skipOps((
        xfail('view_as_complex'),
        skip('sparse.sampled_addmm'),
        xfail('nn.functional.binary_cross_entropy'),
        xfail('narrow_copy'),
        xfail('view_copy'),
        xfail('view'),
        xfail('view_as'),
        xfail('empty_strided'),
        skip('normal'),
        xfail('take_along_dim'),
        xfail('kron'),
        xfail('nn.functional.channel_shuffle'),
        xfail('_foreach_sub'),
        xfail('_foreach_clamp_min', dtypes=complex_types_and(b8)),
        xfail('_foreach_clamp_max', dtypes=complex_types_and(b8)),
        xfail('_foreach_minimum', dtypes=complex_types_and(b8)),
        xfail('_foreach_maximum', dtypes=complex_types_and(b8)),
        xfail('_foreach_addcmul', dtypes=(b8,)),
        xfail('_foreach_addcdiv', dtypes=integral_types() + complex_types_and(b8)),
        xfail('_foreach_lerp', dtypes=integral_types_and(b8)),
    ))
    @ops(itertools.chain(op_db, foreach_op_db), dtypes=OpDTypes.any_common_cpu_cuda_one)
    # Only test on CUDA, as CUDA kernel's stride is the reference
    @onlyCUDA
    def test_dispatch_symbolic_meta_outplace_all_strides(self, device, dtype, op):
        # [FUNCTION DOC] Symbolic-meta out-of-place dispatch test that also tries several stride
        # layouts of the inputs. CUDA only, using one common dtype per operator, since CUDA
        # strides are the reference; delegates to the shared driver.
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=False, all_stride_variants=True)

    @skipIfCrossRef
    @suppress_warnings
    # only test one dtype, as output stride behavior is the same for all dtypes
    @skipOps((
        xfail('abs', dtypes=(c128, c64, c32)),
        xfail('as_strided', variant_name='partial_views'),
        xfail('_foreach_add', dtypes=integral_types() + complex_types_and(b8, bf16, f16, f64)),
        xfail('_foreach_sub'),
        xfail('_foreach_mul', dtypes=(b8,)),
        xfail('_foreach_div', dtypes=integral_types_and(b8)),
        xfail('_foreach_clamp_min', dtypes=complex_types_and(b8)),
        xfail('_foreach_clamp_max', dtypes=complex_types_and(b8)),
        xfail('_foreach_minimum', dtypes=complex_types_and(b8)),
        xfail('_foreach_maximum', dtypes=complex_types_and(b8)),
        xfail('_foreach_addcmul', dtypes=integral_types() + complex_types_and(b8)),
        xfail('_foreach_addcdiv', dtypes=integral_types() + complex_types_and(b8)),
        xfail('_foreach_norm'),
        xfail('_foreach_lerp', dtypes=integral_types_and(b8)),
    ))
    @ops(itertools.chain(op_db, foreach_op_db), dtypes=OpDTypes.any_common_cpu_cuda_one)
    # Only test on CUDA, as CUDA kernel's stride is the reference
    @onlyCUDA
    def test_dispatch_symbolic_meta_inplace_all_strides(self, device, dtype, op):
        # [FUNCTION DOC] Symbolic-meta in-place dispatch test that also tries several
        # stride layouts of the inputs. CUDA only, one common dtype per operator; delegates
        # to the shared driver.
        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=True, all_stride_variants=True)

    @skipIfCrossRef
    @suppress_warnings
    # only test one dtype, as output stride behavior is the same for all dtypes
    @skipOps((
        xfail('complex'),
        xfail('heaviside'),
        xfail('isclose'),
        xfail('polar'),
        xfail('_refs.copysign'),
        xfail('_refs.floor_divide'),
        xfail('_refs.isclose'),
        xfail('_refs._conversions.complex'),
        xfail('_refs._conversions.polar'),
    ))
    @ops(binary_ufuncs, allowed_dtypes=(torch.float32,))
    # Only test on CUDA, as CUDA kernel's stride is the reference
    @onlyCUDA
    def test_binary_ufuncs_mixed_dtype(self, device, dtype, op):
        # [FUNCTION DOC] CUDA-only test, float32 only, of binary pointwise operators with a
        # float32 first input and a float16 second input. It copies the operator info, replaces
        # its sample generator with one yielding that single mixed-dtype pair, and runs the
        # shared dispatch driver with symbolic meta, out-of-place.
        make_arg = partial(
            make_tensor,
            device=device,
        )

        def sample_input(op, device, dtype, requires_grad, **kwargs):
            # [FUNCTION DOC] HELPER: Generator that yields one sample input: a float tensor of
            # the test dtype and a half-precision tensor, both with the fixed size, made on the
            # test device. Extra arguments are ignored.
            yield SampleInput(
                make_arg((S,), dtype=dtype), make_arg((S,), dtype=torch.float16)
            )

        op = copy.copy(op)
        op.sample_inputs_func = sample_input

        self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=False)


    def test_empty_quantized(self):
        # [FUNCTION DOC] Creates a huge quantized empty tensor on the meta device and
        # checks that it is on meta.
        r = torch.empty(2 ** 52, device='meta', dtype=torch.qint8)
        self.assertEqual(r.device.type, 'meta')

    def test_nan_to_num(self):
        # [FUNCTION DOC] Applies nan_to_num to a meta tensor holding special float values and
        # checks the result is on meta.
        t = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14], device='meta')
        r = t.nan_to_num()
        self.assertEqual(r.device.type, 'meta')

    def test_inplace_masked_fill_error(self):
        # [FUNCTION DOC] Checks that in-place masked fill on a meta tensor with a mask that
        # cannot be broadcast in place raises a runtime error mentioning the broadcast mismatch.
        t = torch.randn(3, 3, device='meta')
        with self.assertRaisesRegex(RuntimeError, "doesn't match the broadcast"):
            t.masked_fill_((t > 0).unsqueeze(0), 0.1)

    def test_inplace_bin_ops_error(self):
        # [FUNCTION DOC] For each of several in-place binary operators (add, sub, mul, div,
        # logical and, or, xor), checks calling it on a meta tensor with an operand whose
        # broadcast shape is larger raises a runtime error mentioning the broadcast mismatch.
        t = torch.randn(3, 3, device='meta')
        for op in (torch.Tensor.add_, torch.Tensor.sub_, torch.Tensor.mul_, torch.Tensor.div_,
                   torch.Tensor.logical_and_, torch.Tensor.logical_or_, torch.Tensor.logical_xor_):
            with self.assertRaisesRegex(RuntimeError, "doesn't match the broadcast"):
                op(t, t.clone().unsqueeze(0))

    @onlyCPU
    def test_meta_autograd_no_error(self):
        # [FUNCTION DOC] CPU-only. In temporary scoped library registrations it defines a custom
        # operator with both CPU and meta implementations that add one, and calls it on a meta
        # tensor. The test passes if there is no error, since a fallthrough exists for the
        # autograd meta key of custom operators. Registrations are removed on exit.
        with torch.library._scoped_library("meta_test", "DEF") as lib:
            with torch.library._scoped_library("meta_test", "IMPL", "CPU") as impl_cpu:
                with torch.library._scoped_library("meta_test", "IMPL", "Meta") as impl_meta:
                    def foo_impl(x):
                        # [FUNCTION DOC] HELPER: Implementation registered for the custom
                        # operator on CPU and meta; returns its input plus one.
                        return x + 1

                    lib.define("foo(Tensor a) -> Tensor")
                    impl_meta.impl("foo", foo_impl)
                    impl_cpu.impl("foo", foo_impl)

                    a = torch.ones(2, device='meta')
                    # The point of the test is that this should not error:
                    # We have a fallthrough kernel registered to the AutogradMeta
                    # key for custom ops, so it's fine that `foo()` doesn't have
                    # an autograd kernel.
                    b = torch.ops.meta_test.foo.default(a)

    def test_huber_loss_backward(self):
        # [FUNCTION DOC] Calls the huber loss backward operator on three huge meta tensors and
        # checks the result is on meta with the shape of the first input.
        inps = [torch.rand(2**52, device='meta') for _ in range(3)]
        r = torch.ops.aten.huber_loss_backward(*inps, 0, 1.0)
        self.assertEqual(r.device.type, 'meta')
        self.assertEqual(r.shape, inps[0].shape)

    def _norm_backwards_test_helper(self, op, args, output_mask, expected_shapes):
        # [FUNCTION DOC] HELPER: Shared checker for normalization backward on meta tensors. Calls
        # the operator functionally with the output mask and checks gradient shapes against the
        # expected shapes (a missing expected shape means a none gradient). Then calls it again
        # passing preallocated empty out tensors, one per mask entry, and checks the shapes of
        # those outputs against the expected shapes.

        dtype = torch.float32
        device = "meta"

        # test functional call
        grads = op(*args, output_mask)

        def assertEqualShapes(res, exp):
            # [FUNCTION DOC] HELPER: Asserts a functional result is none when the expected shape
            # is none, otherwise that its shape equals the expected shape.
            self.assertIsNone(res) if exp is None else self.assertEqual(exp, res.shape)

        assertEqualShapes(grads[0], expected_shapes[0])
        assertEqualShapes(grads[1], expected_shapes[1])
        assertEqualShapes(grads[2], expected_shapes[2])

        out_kwargs = {
            f"out{i}": torch.empty(0, device=device, dtype=dtype)
            for i in range(len(output_mask))
        }

        # test call with out parameters
        grads = op(*args, output_mask, **out_kwargs)

        def assertEqualShapes(res, exp):
            # [FUNCTION DOC] HELPER: Redefinition for the out-parameter phase: when an
            # expected shape exists, asserts the result's shape equals it; when it is
            # none, nothing is checked.
            self.assertEqual(exp, res.shape) if exp is not None else True

        assertEqualShapes(out_kwargs["out0"], expected_shapes[0])
        assertEqualShapes(out_kwargs["out1"], expected_shapes[1])
        assertEqualShapes(out_kwargs["out2"], expected_shapes[2])

    @onlyCPU
    @parametrize("output_mask", list(itertools.product([True, False], [True, False], [True, False])))
    def test_layer_norm_backward(self, output_mask):
        # [FUNCTION DOC] CPU-only, parameterized over all eight combinations of the three-
        # entry output mask. For each layer norm sample input on the meta device (missing
        # weight and bias padded with none), builds a gradient of ones, zero mean and
        # reciprocal std-dev tensors of the reduced shape, computes expected gradient shapes
        # (input shape, weight shape, bias shape or none depending on the mask and presence),
        # and runs the shared normalization backward checker on the layer norm backward
        # operator. Each sample is a subtest.
        from torch.testing._internal.common_methods_invocations import sample_inputs_layer_norm

        device = "meta"
        dtype = torch.float32

        samples = sample_inputs_layer_norm(None, device, dtype, requires_grad=False)

        for sample in samples:
            with self.subTest(sample=sample):
                # handle optional weight and bias
                if len(sample.args) != 3:
                    sample.args = (*sample.args, *([None] * (3 - len(sample.args))))

                grad_out = torch.ones_like(sample.input)
                normalized_shape, weight, bias = sample.args
                ndims_after_reduction = sample.input.ndim - len(normalized_shape)
                mean_shape = grad_out.shape[:ndims_after_reduction]
                mean = torch.zeros(mean_shape, device=device, dtype=dtype)
                rstd = torch.zeros(mean_shape, device=device, dtype=dtype)

                expected_shapes = (
                    sample.input.shape if output_mask[0] else None,
                    weight.shape if output_mask[1] and weight is not None else None,
                    bias.shape if output_mask[2] and bias is not None else None)

                args = [grad_out, sample.input, normalized_shape, mean, rstd, weight, bias]

                self._norm_backwards_test_helper(torch.ops.aten.native_layer_norm_backward,
                                                 args, output_mask, expected_shapes)

    @onlyCPU
    @parametrize("output_mask", list(itertools.product([True, False], [True, False], [True, False])))
    def test_group_norm_backward(self, output_mask):
        # [FUNCTION DOC] CPU-only, parameterized over the eight output masks. For each group norm
        # sample on meta it builds ones gradient, zero mean and std-dev per batch and group, and
        # a zero weight with channel size; expected shapes are the input shape and the weight
        # shape (used for both weight and bias gradients) as selected by the mask. Runs the
        # shared checker on the group norm backward operator with batch, channel, spatial size
        # and group count arguments.
        from torch.testing._internal.common_methods_invocations import sample_inputs_group_norm

        # input, (args) num_groups, (kwargs) weight, bias eps
        device = "meta"
        dtype = torch.float32
        samples = sample_inputs_group_norm(None, device, dtype, requires_grad=False)

        for sample in samples:
            with self.subTest(sample=sample):
                grad_out = torch.ones_like(sample.input)
                N, C = sample.input.shape[:2]
                HxW = torch.prod(torch.as_tensor(sample.input.shape[2:]), dtype=torch.int32).item()
                group = sample.args[0]
                mean = torch.zeros((N, group), device=device, dtype=dtype)
                rstd = torch.zeros((N, group), device=device, dtype=dtype)
                weight = torch.zeros((C), device=device, dtype=dtype)

                args = [grad_out, sample.input, mean, rstd, weight, N, C, HxW, group]

                expected_shapes = (
                    sample.input.shape if output_mask[0] else None,
                    weight.shape if output_mask[1] else None,
                    weight.shape if output_mask[2] else None)

                # test functional call
                self._norm_backwards_test_helper(torch.ops.aten.native_group_norm_backward,
                                                 args, output_mask, expected_shapes)

    @onlyCPU
    @parametrize("output_mask", list(itertools.product([True], [True, False], [True, False])))
    def test_batch_norm_backward(self, output_mask):
        # [FUNCTION DOC] CPU-only, parameterized over output masks whose first entry is always
        # true. For each batch norm sample on meta with at least two dimensions it builds a ones
        # gradient and, in training mode, zero saved mean and inverse std-dev of channel size
        # (none otherwise), and the argument list for the batch norm backward operator. Expected
        # shapes are the input shape, then the channel size shape for the weight and bias
        # gradients per the mask. Runs the shared checker.
        from torch.testing._internal.common_methods_invocations import sample_inputs_batch_norm

        # input, (args) num_groups, (kwargs) weight, bias eps
        device = "meta"
        dtype = torch.float32
        samples = sample_inputs_batch_norm(None, device, dtype, requires_grad=False)

        for sample in samples:
            with self.subTest(sample=sample):

                if sample.input.dim() < 2:
                    continue

                grad_out = torch.ones_like(sample.input)
                running_mean, running_var, weight, bias = sample.args
                train = sample.kwargs.get("training", True)
                save_mean = torch.zeros((sample.input.shape[1], ), device=device, dtype=dtype) if train else None
                save_invstd = torch.zeros((sample.input.shape[1], ), device=device, dtype=dtype) if train else None

                args = [grad_out, sample.input, weight, running_mean, running_var,
                        save_mean, save_invstd, train, sample.kwargs.get("eps", 1e-5)]

                expected_shapes = (
                    sample.input.shape,
                    torch.Size([sample.input.shape[1]]) if output_mask[1] else None,
                    torch.Size([sample.input.shape[1]]) if output_mask[2] else None)

                self._norm_backwards_test_helper(torch.ops.aten.native_batch_norm_backward,
                                                 args, output_mask, expected_shapes)

    def test_fill__alias_relationship(self):
        # [FUNCTION DOC] On a huge meta tensor, checks the in-place fill operator
        # returns the very same object (same identity) while the out-of-place fill
        # returns a different object.
        inps = torch.rand(2**52, device='meta')
        r = torch.ops.aten.fill_(inps, 1.0)
        # aten.fill_ returns an alias
        self.assertEqual(id(inps), id(r))

        # aten.fill returns a new tensor
        r2 = torch.ops.aten.fill(inps, 1.0)
        self.assertNotEqual(id(inps), id(r2))

    def test_meta__fused_moving_avg_obs_fq_helper(self, device):
        # [FUNCTION DOC] Runs the fused moving-average observer fake-quantize helper operator on
        # a real tensor and on its meta conversion, with the observer/fake-quant flags from a
        # module, for three keyword settings (default, per-row off with symmetric off, per-row
        # off with symmetric on). For each, checks the first two outputs have equal sizes and
        # strides on real and meta. Takes a device argument.
        from torch.ao.quantization import FusedMovingAvgObsFakeQuantize
        to_meta = MetaConverter()

        x = torch.randn(5, 5, device=device)
        running_min_op = torch.tensor(float("inf"), device=device)
        running_max_op = torch.tensor(float("-inf"), device=device)
        avg_const = 0.01
        scale = torch.tensor([1.0], device=device)
        zero_point = torch.tensor([0], dtype=torch.int, device=device)

        mod = FusedMovingAvgObsFakeQuantize()
        torch.ao.quantization.enable_fake_quant(mod)
        torch.ao.quantization.enable_observer(mod)
        mod.to(device)

        meta_x = to_meta(x)

        args = [
            x,
            mod.observer_enabled,
            mod.fake_quant_enabled,
            running_min_op,
            running_max_op,
            scale,
            zero_point,
            avg_const,
            0,
            255,
            0,
        ]

        meta_args = args.copy()
        meta_args[0] = meta_x

        kwargss = [
            {},
            {"per_row_fake_quant": False, "symmetric_quant": False},
            {"per_row_fake_quant": False, "symmetric_quant": True},
        ]

        for kwargs in kwargss:
            ref_out = aten._fused_moving_avg_obs_fq_helper.default(*args, **kwargs)
            meta_out = aten._fused_moving_avg_obs_fq_helper.default(*meta_args, **kwargs)

            self.assertEqual(ref_out[0].size(), meta_out[0].size())
            self.assertEqual(ref_out[0].stride(), meta_out[0].stride())
            self.assertEqual(ref_out[1].size(), meta_out[1].size())
            self.assertEqual(ref_out[1].stride(), meta_out[1].stride())

    def test_cdist_forward(self, device):
        # [FUNCTION DOC] Calls the cdist forward operator for each of the three compute modes
        # (none, one, two) on real and meta-converted inputs and checks the meta result is on
        # meta with the same shape as the real one.
        to_meta = MetaConverter()
        x1 = torch.rand([3, 2], device=device)
        x2 = torch.rand([2, 2], device=device)
        p = 2.0
        for compute_mode in (None, 1, 2):
            ref = aten._cdist_forward.default(x1, x2, p, compute_mode)
            res = aten._cdist_forward.default(to_meta(x1), to_meta(x2), p, compute_mode)
            self.assertEqual(res.device.type, 'meta')
            self.assertEqual(ref.shape, res.shape)

    def test_quantized_embedding_bag(self):
        # [FUNCTION DOC] Builds a random float table, packs it with the byte prepack quantization
        # operator, and creates random indices and offsets. Calls the rowwise-offsets embedding-
        # bag byte operator with all inputs moved to meta, sum mode, no per-sample weights and
        # including the last offset. Checks the result shape is 32 by 128, dtype float32 and that
        # its storage has null data pointer.
        tab_shape = [8, 128]
        emb_size, ind_len, off_len = tab_shape[0], 32, 33
        f_table = torch.from_numpy((np.random.random_sample(tab_shape) + 1).astype(np.float32))
        q_table = torch.ops.quantized.embedding_bag_byte_prepack(f_table)
        indices = torch.from_numpy(np.random.randint(low=0, high=emb_size, size=ind_len)).int()
        max_length = len(indices) // (off_len - 1)
        if max_length > 20:
            max_length = 20
        np_lengths = np.random.randint(0, max_length + 1, size=off_len - 1).astype(np.int32)
        offsets = torch.cat([torch.zeros([1]), torch.cumsum(torch.from_numpy(np_lengths), 0)]).int()

        eb = torch.ops.quantized.embedding_bag_byte_rowwise_offsets(
            q_table.to(device="meta"),
            indices.to(device="meta"),
            offsets.to(device="meta"),
            mode=0,  # sum
            per_sample_weights=None,
            include_last_offset=True,
        )
        self.assertEqual(eb.shape, [32, 128])
        self.assertEqual(eb.dtype, torch.float32)
        self.assertEqual(eb.untyped_storage().data_ptr(), 0)

    # Tests mean and max.
    # Can't easily test sum, because there is a fast path for sum which
    # causes offset2bag to not get allocated... but the backward function
    # needs it, and the offset2bag computation lives inside the
    # derivatives.yaml formula directly, so there is no way to access it.
    # To test sum, need to manually compute offset2bag
    @parametrize("mode", [1, 2])
    def test_embedding_bag_dense_backward(self, mode):
        # [FUNCTION DOC] Parameterized over mean and max modes. Runs the embedding bag forward to
        # get intermediate tensors, then the dense backward operator on real tensors and on meta-
        # converted tensors, and checks the real gradient converted to meta equals the meta
        # result. Sum mode is not tested because of its fast path (see comment above).
        weight = torch.randn(4, 3, requires_grad=True)
        indices = torch.tensor([1, 0, 2, 1, 3])
        offsets = torch.tensor([0, 2, 3, 5])
        scale_grad_by_freq = False
        sparse = False
        per_sample_weights = None
        include_last_offset = False
        padding_idx = -1

        output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag.default(
            weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx
        )
        grad = torch.randn_like(output)

        # Call the function with example inputs
        grad_weight = torch.ops.aten._embedding_bag_dense_backward.default(
            grad, indices, offset2bag, bag_size, maximum_indices, weight.size(0),
            scale_grad_by_freq, mode, per_sample_weights, padding_idx
        )
        meta_grad_weight = torch.ops.aten._embedding_bag_dense_backward.default(
            grad.to('meta'), indices.to('meta'), offset2bag.to('meta'), bag_size.to('meta'),
            maximum_indices.to('meta'), weight.size(0),
            scale_grad_by_freq, mode, per_sample_weights, padding_idx
        )
        self.assertEqual(grad_weight.to('meta'), meta_grad_weight)

    def test_segment_reduce_backward(self):
        # [FUNCTION DOC] Calls the segment reduce backward operator with max reduction on real
        # and meta tensors and checks shape, stride, dtype and layout agree. Repeats with
        # noncontiguous gradient and data (a column slice of a two-column tensor).
        grad = torch.ones(16, dtype=torch.float)
        output = torch.ones(16, dtype=torch.float)
        data = torch.ones(16, dtype=torch.float)
        reduce_str = 'max'
        lengths = torch.ones(16, dtype=torch.long)

        out = torch.ops.aten._segment_reduce_backward(grad, output, data, reduce_str, lengths=lengths)
        out_meta = torch.ops.aten._segment_reduce_backward(
            grad.to(device='meta'),
            output.to(device='meta'),
            data.to(device='meta'),
            reduce_str,
            lengths=lengths.to(device='meta'),
        )
        self.assertEqual(out.shape, out_meta.shape)
        self.assertEqual(out.stride(), out_meta.stride())
        self.assertEqual(out.dtype, out_meta.dtype)
        self.assertEqual(out.layout, out_meta.layout)

        # noncontiguous
        grad = torch.ones(16, 2, dtype=torch.float)[:, 1]
        data = torch.ones(16, 2, dtype=torch.float)[:, 1]
        out = torch.ops.aten._segment_reduce_backward(grad, output, data, reduce_str, lengths=lengths)
        out_meta = torch.ops.aten._segment_reduce_backward(
            grad.to(device='meta'),
            output.to(device='meta'),
            data.to(device='meta'),
            reduce_str,
            lengths=lengths.to(device='meta'),
        )
        self.assertEqual(out.shape, out_meta.shape)
        self.assertEqual(out.stride(), out_meta.stride())
        self.assertEqual(out.dtype, out_meta.dtype)
        self.assertEqual(out.layout, out_meta.layout)

    def test_embedding_bag_dense_backward_per_sample_weights(self):
        # [FUNCTION DOC] Runs embedding bag forward with per-sample weights, then the per-sample-
        # weights backward operator on real tensors and on meta-converted ones, and checks the
        # real result converted to meta equals the meta result.
        weight = torch.randn(4, 3, requires_grad=True)
        indices = torch.tensor([1, 0, 2, 1, 3])
        offsets = torch.tensor([0, 2, 3, 5])
        scale_grad_by_freq = False
        sparse = False
        mode = 0
        per_sample_weights = torch.randn(5, requires_grad=True)
        include_last_offset = False
        padding_idx = -1

        output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag.default(
            weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx
        )
        grad = torch.randn_like(output)

        # Call the function with example inputs
        grad_weight = torch.ops.aten._embedding_bag_per_sample_weights_backward.default(
            grad, weight, indices, offsets, offset2bag, mode, padding_idx
        )
        meta_grad_weight = torch.ops.aten._embedding_bag_per_sample_weights_backward.default(
            grad.to('meta'), weight.to('meta'), indices.to('meta'),
            offsets.to('meta'), offset2bag.to('meta'), mode, padding_idx
        )
        self.assertEqual(grad_weight.to('meta'), meta_grad_weight)

    def _assert_fft_meta_stride_matches_eager(self, op, *args):
        # [FUNCTION DOC] HELPER: Converts every tensor among the arguments to meta, calls
        # the operator on real and meta arguments, and asserts the sizes and strides of
        # the outputs are equal.
        to_meta = MetaConverter()
        meta_args = tree_map_only(torch.Tensor, to_meta, args)
        ref_out = op(*args)
        meta_out = op(*meta_args)
        self.assertEqual(ref_out.size(), meta_out.size())
        self.assertEqual(ref_out.stride(), meta_out.stride())

    @onlyCUDA
    @unittest.skipIf(torch.version.hip, "cuFFT-specific stride behavior")
    def test_fft_multi_dim_cufft_stride_matches_meta(self, device):
        # [FUNCTION DOC] CUDA-only, skipped on ROCm. Checks that meta stride of multi-dimensional
        # complex-to-complex and complex-to-real FFT operators matches the strides produced on
        # the GPU for five-dimensional inputs, by using the stride comparison helper for each.
        self._assert_fft_meta_stride_matches_eager(
            aten._fft_c2c.default,
            torch.randn((5, 5, 5, 5, 5), device=device, dtype=torch.complex64),
            [1, 2, 3, 4],
            0,
            True,
        )
        self._assert_fft_meta_stride_matches_eager(
            aten._fft_c2r.default,
            torch.randn((5, 5, 5, 5, 3), device=device, dtype=torch.complex64),
            [0, 1, 2, 3, 4],
            0,
            5,
        )

    # opinfo test is using aten.fill_, it's not testing aten.fill
    @onlyCUDA
    def test_fill_stride(self):
        # [FUNCTION DOC] CUDA-only (comment says the operator-database test covers only the in-
        # place fill). For each stride layout variant of a small four-dimensional random tensor
        # with scalar value, compares size and stride of the out-of-place fill on real and meta-
        # converted arguments.
        # KNOWN ISSUE: the tensors are created on the default device, not on CUDA, despite the
        # CUDA-only decorator.
        to_meta = MetaConverter()
        sample_args = [torch.rand(2, 2, 2, 2), 1.0]

        for args in get_strided_args(sample_args):
            meta_args = to_meta(args)
            ref_out = torch.ops.aten.fill(*args)
            meta_out = torch.ops.aten.fill(*meta_args)
            self.assertEqual(ref_out.size(), meta_out.size())
            self.assertEqual(ref_out.stride(), meta_out.stride())


    def test_map_location_deserialize(self):
        # [FUNCTION DOC] Saves a random tensor to an in-memory buffer and loads it back with a
        # location mapping to the meta device; checks the loaded tensor is on meta with the same
        # shape and dtype and that its storage data pointer is zero.
        import io

        t = torch.rand(10)
        b = io.BytesIO()

        torch.save(t, b)
        b.seek(0)
        r = torch.load(b, map_location=torch.device("meta"))
        self.assertEqual(r.device.type, 'meta')
        self.assertEqual(r.shape, t.shape)
        self.assertEqual(r.dtype, t.dtype)
        self.assertEqual(r.storage().data_ptr(), 0)

    def test_embedding_bag_byte_prepack(self):
        # [FUNCTION DOC] For three embedding sizes, packs a random float weight with the byte
        # prepack operator on a meta tensor and checks the result has eight more entries in the
        # last dimension, float32 dtype and a null storage pointer.
        batch_size = 10
        num_embeddings = 80
        embedding_dim = [128, 256, 512]
        res_shape = [[batch_size, num_embeddings, ed + 8] for ed in embedding_dim]
        for ed, rs in zip(embedding_dim, res_shape):
            weight = torch.randn(batch_size, num_embeddings, ed, dtype=torch.float32)
            res = torch.ops.quantized.embedding_bag_byte_prepack(weight.to(device="meta"))
            self.assertEqual(res.shape, rs)
            self.assertEqual(res.dtype, torch.float32)
            self.assertEqual(res.untyped_storage().data_ptr(), 0)

    def test_embedding_bag_byte_unpack(self):
        # [FUNCTION DOC] For three embedding sizes (each with eight extra packed entries),
        # unpacks a meta tensor with the byte unpack operator and checks the result has eight
        # fewer entries in the last dimension, float32 dtype and a null storage pointer.
        batch_size = 10
        num_embeddings = 80
        embedding_dim = [128, 256, 512]
        res_shape = [[batch_size, num_embeddings, ed] for ed in embedding_dim]
        for ed, rs in zip(embedding_dim, res_shape):
            packed_weight = torch.randn(batch_size, num_embeddings, ed + 8, dtype=torch.float32)
            res = torch.ops.quantized.embedding_bag_byte_unpack(packed_weight.to(device="meta"))
            self.assertEqual(res.shape, rs)
            self.assertEqual(res.dtype, torch.float32)
            self.assertEqual(res.untyped_storage().data_ptr(), 0)

    def test_index_select_out(self):
        # [FUNCTION DOC] Under the python dispatcher, calls index_select with a preallocated out
        # tensor, all on meta, and checks the output shape is ten by sixteen.
        def f():
            # [FUNCTION DOC] HELPER: Makes meta input, index and out tensors and returns the
            # result of index_select with the out argument.
            input = torch.randn([8, 16], device='meta')
            index = torch.tensor([2, 1, 6, 7, 3, 1, 7, 5, 6, 7], device='meta')
            out = torch.empty([10, 16], device='meta')
            return torch.index_select(input=input, dim=0, index=index, out=out)
        with enable_python_dispatcher():
            out = f()
            self.assertEqual(out.shape, [10, 16])

    def test_local_scalar_dense_call(self):
        # [FUNCTION DOC] Checks that calling item on a meta tensor raises a runtime error saying
        # it cannot be called on meta tensors.
        with self.assertRaisesRegex(RuntimeError, "cannot be called on meta tensors"):
            meta_tensor = torch.randn(1, device='meta')
            meta_tensor.item()

    def test_triangular_solve_out(self):
        # [FUNCTION DOC] Computes triangular solve on a real example to get reference
        # outputs, converts the outputs to meta, calls triangular solve on meta inputs with
        # those meta outputs as the out argument, and checks shape and dtype of both outputs
        # match the reference.
        # Get what's the expected output for the given example.
        A = torch.randn(2, 2).triu()
        b = torch.randn(2, 3)
        out = torch.triangular_solve(b, A)

        # Call the function again, transforming every tensor input (including the out tensor)
        # into a meta tensor.
        meta_out = tree_map_only(torch.Tensor, lambda t: t.to("meta"), out)
        torch.triangular_solve(b.to("meta"), A.to("meta"), out=meta_out)

        self.assertEqual(out[0].shape, meta_out[0].shape)
        self.assertEqual(out[0].dtype, meta_out[0].dtype)

        self.assertEqual(out[1].shape, meta_out[1].shape)
        self.assertEqual(out[1].dtype, meta_out[1].dtype)

    def test_meta_consistency_out_dtype_mismatch_pow_Tensor_Scalar(self):
        # [FUNCTION DOC] Checks consistency of errors between CPU and meta for pow of a float32
        # tensor by an integer with a float64 out tensor. If one device raises and the other does
        # not, raises a runtime error chained to the real exception.
        S = (5,)

        def run(device):
            # [FUNCTION DOC] HELPER: Runs the pow call with a float32 random tensor, an integer
            # exponent and a float64 out tensor on the given device; returns the exception if one
            # is raised, otherwise nothing.
            a = torch.rand(S, device=device, dtype=torch.float32)
            b = 2
            out = torch.empty(S, device=device, dtype=torch.float64)

            try:
                torch.pow(a, b, out=out)
            except Exception as e:
                return e

        cpu_err = run("cpu")
        meta_err = run("meta")

        if cpu_err is None and meta_err is not None:
            raise RuntimeError("cpu didn't fail, but meta did.") from meta_err
        elif cpu_err is not None and meta_err is None:
            raise RuntimeError("cpu failed, but meta didn't.") from cpu_err

    def test_nonzero(self):
        # [FUNCTION DOC] With the config flag making meta nonzero assume every element nonzero,
        # checks nonzero on a meta tensor of 2x3x4 has int64 dtype, meta device, shape 24 by 3
        # and strides 1 and 24.
        t = torch.randn(2, 3, 4, device='meta')
        with exp_config.patch(meta_nonzero_assume_all_nonzero=True):
            nz = t.nonzero()
        self.assertEqual(nz.dtype, torch.int64)
        self.assertEqual(nz.device.type, 'meta')
        self.assertEqual(nz.shape, torch.Size([24, 3]))
        self.assertEqual(nz.stride(), torch.Size([1, 24]))


    def test_stride_for_index_Tensor(self):
        # [FUNCTION DOC] Indexes a channels-last-derived 5D tensor with two index tensors in
        # eager mode and under fake tensor mode, and checks the resulting strides are equal.
        from torch._subclasses import FakeTensorMode
        x = torch.randn((24, 16, 32, 32)).to(memory_format=torch.channels_last)
        x = x.view(2, 12, 16, 32, 32)

        i1 = torch.arange(2).unsqueeze(-1)
        i2 = torch.argsort(torch.rand(2, 12), dim=-1)[:, :3]

        out = x[i1, i2]

        mode = FakeTensorMode()
        with mode:
            f_x = mode.from_tensor(x)
            f_i1 = mode.from_tensor(i1)
            f_i2 = mode.from_tensor(i2)
            f_out = f_x[f_i1, f_i2]

        self.assertEqual(out.stride(), f_out.stride())


    @parametrize("in_dtype", [torch.float32, torch.float16])
    @parametrize("bias_dtype", [torch.float32, torch.float16, None])
    def test_mixed_dtype_for_native_layer_norm_backward(self, in_dtype, bias_dtype):
        # [FUNCTION DOC] Parameterized over input dtype (float32 or float16) and bias dtype
        # (float32, float16 or none); skips the half input with float32 bias combination.
        # Computes layer norm and gradients with respect to input (and weight and bias when
        # present) on meta tensors and checks the gradient dtypes equal the input dtype first and
        # the bias dtype for weight and bias.
        if in_dtype == torch.float16 and bias_dtype == torch.float32:
            self.skipTest(f"not supported input dtype is {in_dtype} and bias dtype is {bias_dtype}")
        device = "meta"

        def fn(input, weight, bias, need_grad_input):
            # [FUNCTION DOC] HELPER: Applies layer norm over the last dimension, back-
            # propagates a gradient of ones, and returns the gradients with respect to the
            # given list of tensors.
            outputs = torch.nn.functional.layer_norm(input, input.shape[-1:], weight, bias)
            grad_outs = torch.ones_like(outputs)
            grad_ins = torch.autograd.grad(outputs, need_grad_input, grad_outs)
            return grad_ins

        input = torch.randn([4, 8, 5], dtype=in_dtype, device=device, requires_grad=True)
        need_grad_input = [input]

        if bias_dtype:
            weight = torch.randn(
                [5], dtype=bias_dtype, device=device, requires_grad=True
            )
            bias = torch.randn(
                [5], dtype=bias_dtype, device=device, requires_grad=True
            )
            need_grad_input.append(weight)
            need_grad_input.append(bias)
        else:
            weight = None
            bias = None

        outs = fn(input, weight, bias, need_grad_input)
        out_dtype = [t.dtype for t in outs]
        if bias_dtype:
            self.assertEqual(out_dtype, [in_dtype, bias_dtype, bias_dtype])
        else:
            self.assertEqual(out_dtype, [in_dtype,])

    @parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
    def test_scaled_dot_product_flash_attention_for_cpu_logsumexp_dtype(self, dtype):
        # [FUNCTION DOC] Parameterized over four float dtypes. Calls the CPU flash attention
        # operator on random query, key and value on CPU and on meta, and checks both the output
        # dtype and the log-sum-exp dtype agree.
        B, H, L, E = 2, 4, 8, 16

        def run(device):
            # [FUNCTION DOC] HELPER: Makes random query, key and value of the fixed batch, head,
            # length and embedding sizes on the device, calls the CPU flash attention operator,
            # and returns the dtypes of the output and the log-sum-exp.
            query = torch.randn(B, H, L, E, device=device, dtype=dtype)
            key = torch.randn(B, H, L, E, device=device, dtype=dtype)
            value = torch.randn(B, H, L, E, device=device, dtype=dtype)

            output, logsumexp = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
                query, key, value
            )
            return output.dtype, logsumexp.dtype

        cpu_output_dtype, cpu_logsumexp_dtype = run("cpu")
        meta_output_dtype, meta_logsumexp_dtype = run("meta")

        self.assertEqual(cpu_output_dtype, meta_output_dtype)
        self.assertEqual(cpu_logsumexp_dtype, meta_logsumexp_dtype)

class TestMetaKernelConv(TestCase):
    # [CLASS DOC] Tests of raw meta kernels for convolution backward.
    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_convolution_backward_meta_kernel_channels_last(self):
        # [FUNCTION DOC] Calls convolution backward directly on meta tensors (no fake tensor
        # mode). With channels-last gradient output and contiguous input and weight it requires
        # input and weight gradients to be contiguous; with contiguous gradient output and
        # channels-last input it requires both gradients to be channels-last contiguous.
        # Skipped under Dynamo.
        """Test the meta kernel directly (device='meta', no FakeTensorMode).
        This exercises the @register_meta path used by torch.export, which
        does NOT go through the FakeTensor intercept in fake_impls.py.
        """
        # channels_last grad_output + contiguous input/weight -> contiguous
        grad_out = torch.empty(2, 3, 4, 4, device="meta").to(
            memory_format=torch.channels_last
        )
        inp = torch.empty(2, 3, 4, 4, device="meta")
        w = torch.empty(3, 3, 3, 3, device="meta")
        gi, gw, _ = torch.ops.aten.convolution_backward(
            grad_out,
            inp,
            w,
            [3],
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
            [True, True, True],
        )
        self.assertTrue(gi.is_contiguous())
        self.assertTrue(gw.is_contiguous())

        # contiguous grad_output + channels_last input -> channels_last
        grad_out2 = torch.empty(2, 3, 4, 4, device="meta")
        inp2 = torch.empty(2, 3, 4, 4, device="meta").to(
            memory_format=torch.channels_last
        )
        gi2, gw2, _ = torch.ops.aten.convolution_backward(
            grad_out2,
            inp2,
            w,
            [3],
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
            [True, True, True],
        )
        self.assertTrue(gi2.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(gw2.is_contiguous(memory_format=torch.channels_last))




class TestMetaKernelRegistrations(TestCase):
    # [CLASS DOC] Tests that individual meta kernels and decompositions match eager CPU behavior
    # (shapes, strides, dtypes and raised errors). Most tests are skipped under Dynamo because
    # they exercise raw meta kernels.
    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_aminmax_out_dtype_mismatch(self):
        # [FUNCTION DOC] Calls aminmax with out tensors of float64 for a float meta input and
        # checks a runtime error saying the out tensor dtype was unexpected.
        inp = torch.rand(10, 10, device="meta")
        out_min = torch.empty(10, dtype=torch.float64, device="meta")
        out_max = torch.empty(10, dtype=torch.float64, device="meta")

        with self.assertRaisesRegex(RuntimeError, "Expected out tensor to have dtype"):
            torch.ops.aten.aminmax.out(
                inp, dim=-1, keepdim=False, min=out_min, max=out_max
            )

    def test_grid_sampler_2d_cpu_fallback_meta(self):
        # [FUNCTION DOC] Checks the CPU-fallback grid sampler forward and both backward outputs:
        # for contiguous, channels-last and transposed-contiguous inputs the fake tensor results
        # must have equal shape, stride and dtype to eager results. Then with a five-dimensional
        # input and grid, checks eager, fake forward and fake backward all raise a runtime error
        # expecting a four-dimensional input.
        # The meta kernels' whole contract is output metadata, so compare
        # shape/stride/dtype against eager for the forward and both backward
        # outputs. Covers contiguous, channels_last and non-contiguous inputs.
        from torch._subclasses.fake_tensor import FakeTensorMode

        def check(inp, grid):
            # [FUNCTION DOC] HELPER: Runs the CPU-fallback grid sampler forward and backward in
            # eager mode and in fake tensor mode for the given input and grid and asserts shape,
            # stride and dtype of the three outputs agree.
            expected = torch._grid_sampler_2d_cpu_fallback(inp, grid, 0, 0, True)
            grad_out = torch.randn_like(expected)
            bwd = torch.ops.aten._grid_sampler_2d_cpu_fallback_backward
            exp_gi, exp_gg = bwd(grad_out, inp, grid, 0, 0, True)
            with FakeTensorMode() as mode:
                f_inp, f_grid = mode.from_tensor(inp), mode.from_tensor(grid)
                f_out = torch._grid_sampler_2d_cpu_fallback(f_inp, f_grid, 0, 0, True)
                f_gi, f_gg = bwd(
                    mode.from_tensor(grad_out), f_inp, f_grid, 0, 0, True
                )
            for fake, real in ((f_out, expected), (f_gi, exp_gi), (f_gg, exp_gg)):
                self.assertEqual(fake.shape, real.shape)
                self.assertEqual(fake.stride(), real.stride())
                self.assertEqual(fake.dtype, real.dtype)

        inp = torch.randn(2, 3, 4, 5)
        grid = torch.rand(2, 6, 7, 2) * 2 - 1
        check(inp, grid)
        check(inp.contiguous(memory_format=torch.channels_last), grid)
        check(inp.transpose(2, 3).contiguous().transpose(2, 3), grid)

        # The C++ impls run check_grid_sampler_common/_2d themselves, so the
        # metas must reject the same inputs rather than silently returning a
        # shape eager would refuse to compute. Use 5D input and grid: the last
        # grid dim still equals input.ndim - 2, so check_grid_sampler_common
        # passes and only check_grid_sampler_2d can reject it.
        bad_inp, bad_grid = torch.randn(2, 3, 4, 5, 6), torch.rand(2, 6, 7, 8, 3)
        bwd = torch.ops.aten._grid_sampler_2d_cpu_fallback_backward
        msg = "expected 4D input"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch._grid_sampler_2d_cpu_fallback(bad_inp, bad_grid, 0, 0, True)
        with FakeTensorMode() as mode:
            f_inp, f_grid = mode.from_tensor(bad_inp), mode.from_tensor(bad_grid)
            with self.assertRaisesRegex(RuntimeError, msg):
                torch._grid_sampler_2d_cpu_fallback(f_inp, f_grid, 0, 0, True)
            # the backward meta runs the same checks
            with self.assertRaisesRegex(RuntimeError, msg):
                bwd(mode.from_tensor(torch.randn(2, 3, 6, 7)), f_inp, f_grid, 0, 0, True)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_make_dep_token(self):
        # [FUNCTION DOC] Creates a dependency token on CPU and on meta and checks the shapes and
        # dtypes are equal. Skipped under Dynamo.
        cpu_result = torch.ops.aten._make_dep_token(device=torch.device("cpu"))
        meta_result = torch.ops.aten._make_dep_token(device=torch.device("meta"))
        self.assertEqual(cpu_result.shape, meta_result.shape)
        self.assertEqual(cpu_result.dtype, meta_result.dtype)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_rrelu_backward_small_range(self):
        # [FUNCTION DOC] Compares the rrelu backward from the native operator and from the Python
        # decomposition for a tiny lower and upper range with training mode, and checks they are
        # equal. Skipped under Dynamo.
        from torch._decomp.decompositions import rrelu_with_noise_backward

        x = torch.randn(5, requires_grad=True)
        lower, upper = 0.125, 0.125 + torch.finfo(torch.float32).eps
        noise = torch.rand(5)
        grad = torch.ones(5)
        cpp_result = torch.ops.aten.rrelu_with_noise_backward(
            grad, x, noise, lower, upper, True, False
        )
        decomp_result = rrelu_with_noise_backward(
            grad, x, noise, lower, upper, True, False
        )
        self.assertEqual(cpp_result, decomp_result)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_linalg_eig_strides_cpu(self):
        # [FUNCTION DOC] Computes eigen decomposition of a three by three matrix eagerly and
        # under fake tensor mode, and checks the eigenvector shape, stride and dtype are equal.
        from torch._subclasses.fake_tensor import FakeTensorMode

        matrix = torch.randn(3, 3)
        _, eigvecs = torch.linalg.eig(matrix)
        with FakeTensorMode():
            matrix_fake = torch.randn(3, 3, device="cpu")
            _, eigvecs_fake = torch.linalg.eig(matrix_fake)
        self.assertEqual(eigvecs.stride(), eigvecs_fake.stride())
        self.assertEqual(eigvecs.shape, eigvecs_fake.shape)
        self.assertEqual(eigvecs.dtype, eigvecs_fake.dtype)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_zero_preserves_input_strides(self):
        # [FUNCTION DOC] Zeroes a transposed tensor with the zero operator on CPU, a meta tensor
        # with the same strides, and a fake tensor, and checks shape and stride of results match
        # the CPU result in both cases (strides are preserved).
        from torch._subclasses.fake_tensor import FakeTensorMode

        x = torch.randn(5, 4).t()
        cpu_result = torch.ops.aten.zero.default(x)

        meta_x = torch.empty_strided(x.shape, x.stride(), device="meta")
        meta_result = torch.ops.aten.zero.default(meta_x)
        self.assertEqual(cpu_result.shape, meta_result.shape)
        self.assertEqual(cpu_result.stride(), meta_result.stride())

        with FakeTensorMode() as mode:
            fake_x = mode.from_tensor(x)
            fake_result = torch.ops.aten.zero.default(fake_x)
        self.assertEqual(cpu_result.shape, fake_result.shape)
        self.assertEqual(cpu_result.stride(), fake_result.stride())

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_randint_like_tensor_dtype_kwarg(self):
        # [FUNCTION DOC] Calls random-integer-like with a tensor upper bound and an explicit
        # int32 dtype on CPU and meta and checks dtype and shape agree.
        x_cpu = torch.randn(3, 4)
        high_cpu = torch.tensor(10)
        y_cpu = torch.ops.aten.randint_like.Tensor(x_cpu, high_cpu, dtype=torch.int32)
        x_meta = torch.randn(3, 4, device="meta")
        high_meta = torch.tensor(10, device="meta")
        y_meta = torch.ops.aten.randint_like.Tensor(
            x_meta, high_meta, dtype=torch.int32
        )
        self.assertEqual(y_cpu.dtype, y_meta.dtype)
        self.assertEqual(y_cpu.shape, y_meta.shape)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_randint_like_tensor_preserves_default(self):
        # [FUNCTION DOC] Calls random-integer-like with a tensor upper bound and no dtype on
        # half-precision inputs on CPU and meta and checks dtype and shape agree (the input
        # dtype is preserved).
        x_cpu = torch.randn(3, 4, dtype=torch.float16)
        high_cpu = torch.tensor(10)
        y_cpu = torch.ops.aten.randint_like.Tensor(x_cpu, high_cpu)
        x_meta = torch.randn(3, 4, device="meta", dtype=torch.float16)
        high_meta = torch.tensor(10, device="meta")
        y_meta = torch.ops.aten.randint_like.Tensor(x_meta, high_meta)
        self.assertEqual(y_cpu.dtype, y_meta.dtype)
        self.assertEqual(y_cpu.shape, y_meta.shape)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_pad_sequence_decomp_left(self):
        # [FUNCTION DOC] Pads a list of two one-dimensional integer tensors on the left with
        # batch first using the decomposition, and checks the values equal the expected
        # padded matrix. Then runs the decomposition on meta tensors and checks the shape
        # equals the CPU shape.
        from torch._decomp import decompositions

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        cpu_result = decompositions.pad_sequence(
            [a, b], batch_first=True, padding_value=0.0, padding_side="left"
        )
        expected = torch.tensor([[1, 2, 3], [0, 4, 5]])
        self.assertEqual(cpu_result, expected)
        a_meta = torch.randn(3, device="meta")
        b_meta = torch.randn(2, device="meta")
        meta_result = decompositions.pad_sequence(
            [a_meta, b_meta], batch_first=True, padding_value=0.0, padding_side="left"
        )
        self.assertEqual(cpu_result.shape, meta_result.shape)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_pad_sequence_decomp_left_not_batch_first(self):
        # [FUNCTION DOC] Checks left padding with the decomposition when batch first is false
        # gives the expected transposed padded matrix.
        from torch._decomp import decompositions

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        result = decompositions.pad_sequence(
            [a, b], batch_first=False, padding_value=0.0, padding_side="left"
        )
        expected = torch.tensor([[1, 0], [2, 4], [3, 5]])
        self.assertEqual(result, expected)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_pad_sequence_decomp_mixed_dtype_padding_value(self):
        # [FUNCTION DOC] For float and complex first sequences paired with an int32 sequence and
        # a negative float padding value, checks the decomposition's padded result equals the
        # native pad sequence operator result with right padding and batch first false.
        from torch._decomp import decompositions

        for first_sequence in (
            torch.tensor([0, 0.4]),
            torch.tensor([0, 0.4 + 0j]),
        ):
            sequences = [first_sequence, torch.tensor([0], dtype=torch.int32)]
            result = decompositions.pad_sequence(
                sequences, batch_first=False, padding_value=-0.7
            )
            expected = torch.ops.aten.pad_sequence.default(
                sequences, False, -0.7, "right"
            )
            self.assertEqual(result, expected)

    @skipIfXpu(msg="https://github.com/pytorch/pytorch/issues/181490")
    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_padded_dense_to_jagged_total_L_zero(self):
        # [FUNCTION DOC] Skipped on XPU (linked issue) and under Dynamo. Calls the padded dense
        # to jagged forward operator with total length zero and all-zero offsets on CPU and in
        # fake tensor mode, and checks shapes and dtypes agree.
        from torch._subclasses.fake_tensor import FakeTensorMode

        cpu_padded = torch.randn(2, 3, 4)
        cpu_offsets = [torch.tensor([0, 0, 0])]
        cpu_result = torch.ops.aten._padded_dense_to_jagged_forward(
            cpu_padded, cpu_offsets, total_L=0
        )
        with FakeTensorMode():
            fake_padded = torch.randn(2, 3, 4)
            fake_offsets = [torch.tensor([0, 0, 0])]
            fake_result = torch.ops.aten._padded_dense_to_jagged_forward(
                fake_padded, fake_offsets, total_L=0
            )
        self.assertEqual(cpu_result.shape, fake_result.shape)
        self.assertEqual(cpu_result.dtype, fake_result.dtype)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_padded_dense_to_jagged_total_L_none(self):
        # [FUNCTION DOC] Calls the padded dense to jagged forward operator with no total
        # length on CPU and in fake tensor mode using a shape environment that allows dynamic
        # output shapes, and checks the rank and dtype of results agree (the data-dependent
        # size is not compared).
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        cpu_padded = torch.randn(2, 3, 4)
        cpu_offsets = [torch.tensor([0, 1, 3])]
        cpu_result = torch.ops.aten._padded_dense_to_jagged_forward(
            cpu_padded, cpu_offsets, total_L=None
        )
        shape_env = ShapeEnv(allow_dynamic_output_shape_ops=True)
        with FakeTensorMode(shape_env=shape_env):
            fake_padded = torch.randn(2, 3, 4)
            fake_offsets = [torch.tensor([0, 1, 3])]
            fake_result = torch.ops.aten._padded_dense_to_jagged_forward(
                fake_padded, fake_offsets, total_L=None
            )
        self.assertEqual(len(cpu_result.shape), len(fake_result.shape))
        self.assertEqual(cpu_result.dtype, fake_result.dtype)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_segment_reduce_2d(self):
        # [FUNCTION DOC] Runs segment reduce with sum along axis zero on CPU and meta data and
        # checks shape and dtype agree.
        data_cpu = torch.randn(10, 5)
        lengths = torch.tensor([3, 4, 3])
        cpu_result = torch.segment_reduce(data_cpu, "sum", lengths=lengths, axis=0)
        data_meta = torch.randn(10, 5, device="meta")
        meta_result = torch.segment_reduce(
            data_meta, "sum", lengths=lengths, axis=0
        )
        self.assertEqual(cpu_result.shape, meta_result.shape)
        self.assertEqual(cpu_result.dtype, meta_result.dtype)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_segment_reduce_batched_shape(self):
        # [FUNCTION DOC] Runs segment reduce on a meta tensor with a two-dimensional lengths
        # tensor along axis one in unsafe mode and checks the shape is ten by three. Only meta is
        # verified because the CPU run is noted to crash.
        # axis=1 with 2D lengths exercises the shape fix (using data.shape
        # as base instead of lengths.shape). CPU segfaults for this case
        # so we verify shape on meta only.
        data = torch.randn(10, 5, device="meta")
        lengths = torch.ones(2, 3, dtype=torch.long)
        result = torch.segment_reduce(
            data, "sum", lengths=lengths, axis=1, unsafe=True
        )
        self.assertEqual(result.shape, (10, 3))

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_avg_pool3d_divisor_zero_error(self):
        # [FUNCTION DOC] Checks that average pooling in three dimensions with a zero divisor
        # override raises a runtime error mentioning the divisor, for both CPU and meta inputs.
        x_cpu = torch.randn(1, 1, 4, 4, 4)
        with self.assertRaisesRegex(RuntimeError, "divisor"):
            torch.nn.functional.avg_pool3d(x_cpu, kernel_size=2, divisor_override=0)
        x_meta = torch.randn(1, 1, 4, 4, 4, device="meta")
        with self.assertRaisesRegex(RuntimeError, "divisor"):
            torch.nn.functional.avg_pool3d(x_meta, kernel_size=2, divisor_override=0)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_avg_pool3d_backward_divisor_zero_error(self):
        # [FUNCTION DOC] Checks that the three-dimensional average pooling backward operator
        # with a zero divisor override raises a runtime error mentioning the divisor, for both
        # CPU and meta tensors.
        grad_cpu = torch.randn(1, 1, 2, 2, 2)
        with self.assertRaisesRegex(RuntimeError, "divisor"):
            torch.ops.aten.avg_pool3d_backward(
                grad_cpu,
                torch.randn(1, 1, 4, 4, 4),
                [2, 2, 2], [2, 2, 2], [0, 0, 0], True, True, 0,
            )
        grad_meta = torch.randn(1, 1, 2, 2, 2, device="meta")
        with self.assertRaisesRegex(RuntimeError, "divisor"):
            torch.ops.aten.avg_pool3d_backward(
                grad_meta,
                torch.randn(1, 1, 4, 4, 4, device="meta"),
                [2, 2, 2], [2, 2, 2], [0, 0, 0], True, True, 0,
            )

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_fill_tensor_dim_check(self):
        # [FUNCTION DOC] Checks that in-place fill with a two-dimensional value tensor raises a
        # runtime error mentioning the zero-dimension requirement, on CPU and on meta.
        x_cpu = torch.randn(3, 4)
        value_cpu = torch.randn(2, 3)
        with self.assertRaisesRegex(RuntimeError, "0-dimension"):
            x_cpu.fill_(value_cpu)
        x_meta = torch.randn(3, 4, device="meta")
        value_meta = torch.randn(2, 3, device="meta")
        with self.assertRaisesRegex(RuntimeError, "0-dimension"):
            x_meta.fill_(value_meta)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_fill_tensor_scalar_ok(self):
        # [FUNCTION DOC] Checks in-place fill with a zero-dimensional tensor value works on CPU
        # and meta and that the result shapes and dtypes agree.
        x_cpu = torch.randn(3, 4)
        value_cpu = torch.tensor(1.0)
        cpu_result = x_cpu.fill_(value_cpu)
        x_meta = torch.randn(3, 4, device="meta")
        value_meta = torch.tensor(1.0, device="meta")
        meta_result = x_meta.fill_(value_meta)
        self.assertEqual(cpu_result.shape, meta_result.shape)
        self.assertEqual(cpu_result.dtype, meta_result.dtype)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_fill_out_of_place_tensor_dim_check(self):
        # [FUNCTION DOC] Checks that out-of-place fill with a two-dimensional value tensor raises
        # a runtime error mentioning the zero-dimension requirement, on CPU and on meta.
        x_cpu = torch.randn(3, 4)
        value_cpu = torch.randn(2, 3)
        with self.assertRaisesRegex(RuntimeError, "0-dimension"):
            torch.fill(x_cpu, value_cpu)
        x_meta = torch.randn(3, 4, device="meta")
        value_meta = torch.randn(2, 3, device="meta")
        with self.assertRaisesRegex(RuntimeError, "0-dimension"):
            torch.fill(x_meta, value_meta)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_fill_out_of_place_tensor_scalar_ok(self):
        # [FUNCTION DOC] Checks out-of-place fill with a zero-dimensional tensor value works on
        # CPU and meta and that result shapes and dtypes agree.
        x_cpu = torch.randn(3, 4)
        value_cpu = torch.tensor(1.0)
        cpu_result = torch.fill(x_cpu, value_cpu)
        x_meta = torch.randn(3, 4, device="meta")
        value_meta = torch.tensor(1.0, device="meta")
        meta_result = torch.fill(x_meta, value_meta)
        self.assertEqual(cpu_result.shape, meta_result.shape)
        self.assertEqual(cpu_result.dtype, meta_result.dtype)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_weight_int8pack_mm_inner_dim_mismatch(self):
        # [FUNCTION DOC] Checks the int8 weight-packed matrix multiply operator raises a runtime
        # error, on CPU and meta, when the inner dimensions of the two matrices differ.
        A_cpu = torch.randn(4, 8)
        B_cpu = torch.randint(-128, 127, (3, 16), dtype=torch.int8)
        scales_cpu = torch.randn(3)
        with self.assertRaises(RuntimeError):
            torch.ops.aten._weight_int8pack_mm(A_cpu, B_cpu, scales_cpu)
        A_meta = torch.randn(4, 8, device="meta")
        B_meta = torch.randint(-128, 127, (3, 16), device="meta", dtype=torch.int8)
        scales_meta = torch.randn(3, device="meta")
        with self.assertRaises(RuntimeError):
            torch.ops.aten._weight_int8pack_mm(A_meta, B_meta, scales_meta)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_weight_int8pack_mm_scales_mismatch(self):
        # [FUNCTION DOC] Checks the int8 weight-packed matrix multiply operator raises a runtime
        # error, on CPU and meta, when the scales length does not match the weight rows.
        A_cpu = torch.randn(4, 8)
        B_cpu = torch.randint(-128, 127, (3, 8), dtype=torch.int8)
        scales_cpu = torch.randn(5)
        with self.assertRaises(RuntimeError):
            torch.ops.aten._weight_int8pack_mm(A_cpu, B_cpu, scales_cpu)
        A_meta = torch.randn(4, 8, device="meta")
        B_meta = torch.randint(-128, 127, (3, 8), device="meta", dtype=torch.int8)
        scales_meta = torch.randn(5, device="meta")
        with self.assertRaises(RuntimeError):
            torch.ops.aten._weight_int8pack_mm(A_meta, B_meta, scales_meta)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_miopen_batch_norm_save_dtype(self):
        # [FUNCTION DOC] On meta tensors with half-precision input and float32 weights, calls the
        # ROCm-only batch norm operator and checks the saved mean and saved variance outputs have
        # the dtype of the weight, as the native code does.
        # miopen_batch_norm is ROCm-only with no CPU implementation.
        # Test that meta kernel outputs use weight's dtype (matching C++ behavior).
        input_t = torch.randn(2, 3, 4, 4, device="meta", dtype=torch.float16)
        weight = torch.randn(3, device="meta", dtype=torch.float32)
        bias = torch.randn(3, device="meta", dtype=torch.float32)
        running_mean = torch.randn(3, device="meta", dtype=torch.float32)
        running_var = torch.randn(3, device="meta", dtype=torch.float32)
        result = torch.ops.aten.miopen_batch_norm(
            input_t, weight, bias, running_mean, running_var, True, 0.1, 1e-5
        )
        output, save_mean, save_var = result
        self.assertEqual(save_mean.dtype, weight.dtype)
        self.assertEqual(save_var.dtype, weight.dtype)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_reflection_pad2d_channels_last(self):
        # [FUNCTION DOC] Pads a channels-last four-dimensional tensor with reflection on CPU
        # and under fake tensor mode; checks shape and stride match and the fake result is
        # channels-last contiguous.
        from torch._subclasses.fake_tensor import FakeTensorMode

        x_cpu = torch.randn(1, 3, 4, 4).to(memory_format=torch.channels_last)
        cpu_result = torch.nn.functional.pad(x_cpu, (1, 1, 1, 1), mode="reflect")
        with FakeTensorMode():
            x_fake = torch.randn(1, 3, 4, 4, device="cpu").to(
                memory_format=torch.channels_last
            )
            fake_result = torch.nn.functional.pad(
                x_fake, (1, 1, 1, 1), mode="reflect"
            )
        self.assertEqual(cpu_result.shape, fake_result.shape)
        self.assertEqual(cpu_result.stride(), fake_result.stride())
        self.assertTrue(fake_result.is_contiguous(memory_format=torch.channels_last))

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_reflection_pad2d_contiguous(self):
        # [FUNCTION DOC] Pads a contiguous four-dimensional tensor with reflection on CPU and
        # meta and checks shape and stride match and the meta result is contiguous.
        x_cpu = torch.randn(1, 3, 4, 4)
        cpu_result = torch.nn.functional.pad(x_cpu, (1, 1, 1, 1), mode="reflect")
        x_meta = torch.randn(1, 3, 4, 4, device="meta")
        meta_result = torch.nn.functional.pad(x_meta, (1, 1, 1, 1), mode="reflect")
        self.assertEqual(cpu_result.shape, meta_result.shape)
        self.assertEqual(cpu_result.stride(), meta_result.stride())
        self.assertTrue(meta_result.is_contiguous())

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_replication_pad2d_channels_last(self):
        # [FUNCTION DOC] Pads a channels-last four-dimensional tensor with replication on CPU
        # and under fake tensor mode; checks shape and stride match and the fake result is
        # channels-last contiguous.
        from torch._subclasses.fake_tensor import FakeTensorMode

        x_cpu = torch.randn(1, 3, 4, 4).to(memory_format=torch.channels_last)
        cpu_result = torch.nn.functional.pad(x_cpu, (1, 1, 1, 1), mode="replicate")
        with FakeTensorMode():
            x_fake = torch.randn(1, 3, 4, 4, device="cpu").to(
                memory_format=torch.channels_last
            )
            fake_result = torch.nn.functional.pad(
                x_fake, (1, 1, 1, 1), mode="replicate"
            )
        self.assertEqual(cpu_result.shape, fake_result.shape)
        self.assertEqual(cpu_result.stride(), fake_result.stride())
        self.assertTrue(fake_result.is_contiguous(memory_format=torch.channels_last))

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_replication_pad2d_contiguous(self):
        # [FUNCTION DOC] Pads a contiguous four-dimensional tensor with replication on CPU and
        # meta and checks shape and stride match and the meta result is contiguous.
        x_cpu = torch.randn(1, 3, 4, 4)
        cpu_result = torch.nn.functional.pad(x_cpu, (1, 1, 1, 1), mode="replicate")
        x_meta = torch.randn(1, 3, 4, 4, device="meta")
        meta_result = torch.nn.functional.pad(
            x_meta, (1, 1, 1, 1), mode="replicate"
        )
        self.assertEqual(cpu_result.shape, meta_result.shape)
        self.assertEqual(cpu_result.stride(), meta_result.stride())
        self.assertTrue(meta_result.is_contiguous())

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    def test_mkldnn_rnn_backward_dtype(self):
        # [FUNCTION DOC] Skips if MKLDNN or its bfloat16 support is unavailable. Runs an LSTM
        # layer forward in bfloat16 on CPU, then the layer backward on CPU and on meta-converted
        # tensors, and checks every returned gradient has equal shape and dtype across the two.
        if not torch.backends.mkldnn.is_available():
            self.skipTest("MKLDNN not available")
        if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
            self.skipTest("MKLDNN bf16 not supported on this CPU")
        hidden_size = 16
        seq_len, batch, input_size = 5, 2, 8
        input_t = torch.randn(seq_len, batch, input_size, dtype=torch.bfloat16)
        weight_ih = torch.randn(hidden_size * 4, input_size, dtype=torch.bfloat16)
        weight_hh = torch.randn(hidden_size * 4, hidden_size, dtype=torch.bfloat16)
        bias_ih = torch.randn(hidden_size * 4, dtype=torch.bfloat16)
        bias_hh = torch.randn(hidden_size * 4, dtype=torch.bfloat16)
        hx = torch.randn(1, batch, hidden_size, dtype=torch.bfloat16)
        cx = torch.randn(1, batch, hidden_size, dtype=torch.bfloat16)
        output, hy, cy, workspace = torch.ops.aten.mkldnn_rnn_layer(
            input_t, weight_ih, weight_hh, bias_ih, bias_hh, hx, cx,
            False, [], 2, hidden_size, 1, True, False, False, True,
        )
        grad_output = torch.randn_like(output)
        grad_hy = torch.randn_like(hy)
        grad_cy = torch.randn_like(cy)
        cpu_result = torch.ops.aten.mkldnn_rnn_layer_backward(
            input_t, weight_ih, weight_hh, bias_ih, bias_hh, hx, cx,
            output, hy, cy, grad_output, grad_hy, grad_cy,
            False, 2, hidden_size, 1, True, True, False, [], False, workspace,
        )
        meta_result = torch.ops.aten.mkldnn_rnn_layer_backward(
            input_t.to("meta"), weight_ih.to("meta"), weight_hh.to("meta"),
            bias_ih.to("meta"), bias_hh.to("meta"), hx.to("meta"), cx.to("meta"),
            output.to("meta"), hy.to("meta"), cy.to("meta"),
            grad_output.to("meta"), grad_hy.to("meta"), grad_cy.to("meta"),
            False, 2, hidden_size, 1, True, True, False, [], False,
            workspace.to("meta"),
        )
        for cpu_t, meta_t in zip(cpu_result, meta_result):
            self.assertEqual(cpu_t.shape, meta_t.shape)
            self.assertEqual(cpu_t.dtype, meta_t.dtype)

    @skipIfTorchDynamo("tests raw meta kernel, not dynamo")
    @unittest.skip(
        "Disabled due to CI failures; see "
        "https://github.com/pytorch/pytorch/issues/190240"
    )
    def test_mkldnn_rnn_backward_gru_bias_shape(self):
        # [FUNCTION DOC] Permanently skipped (disabled for CI failures, linked issue).
        # Calls the layer backward on meta tensors in GRU mode and checks both bias
        # gradients have a length of four times the hidden size, per the comment on
        # shuffled bias handling in the native code.
        # GRU backward is not supported on CPU via oneDNN, so we verify
        # the meta kernel's bias shape against the expected value from
        # the C++ _shuffle_bias logic (num_bias_gates=4 for GRU).
        hidden_size = 16
        seq_len, batch, input_size = 5, 2, 8
        input_t = torch.randn(seq_len, batch, input_size, device="meta")
        weight_ih = torch.randn(hidden_size * 3, input_size, device="meta")
        weight_hh = torch.randn(hidden_size * 3, hidden_size, device="meta")
        bias_ih = torch.randn(hidden_size * 3, device="meta")
        bias_hh = torch.randn(hidden_size * 3, device="meta")
        hx = torch.randn(1, batch, hidden_size, device="meta")
        cx = torch.randn(1, batch, hidden_size, device="meta")
        output = torch.randn(seq_len, batch, hidden_size, device="meta")
        hy = torch.randn(1, batch, hidden_size, device="meta")
        cy = torch.randn(1, batch, hidden_size, device="meta")
        grad_output = torch.randn(seq_len, batch, hidden_size, device="meta")
        grad_hy = torch.randn(1, batch, hidden_size, device="meta")
        grad_cy = torch.randn(1, batch, hidden_size, device="meta")
        workspace = torch.randn(10, device="meta")
        result = torch.ops.aten.mkldnn_rnn_layer_backward(
            input_t, weight_ih, weight_hh, bias_ih, bias_hh, hx, cx,
            output, hy, cy, grad_output, grad_hy, grad_cy,
            False, 3, hidden_size, 1, True, True, False, [], False, workspace,
        )
        diff_x, diff_w1, diff_w2, diff_b1, diff_b2, diff_hx, diff_cx = result
        expected_bias_shape = torch.Size([4 * hidden_size])
        self.assertEqual(diff_b1.shape, expected_bias_shape)
        self.assertEqual(diff_b2.shape, expected_bias_shape)


instantiate_device_type_tests(TestMeta, globals())


def print_op_str_if_not_supported(op_str):
    # [FUNCTION DOC] HELPER: Parses an operator name string with optional overload, looks up the
    # aten overload (default when none is named), and prints the overload with a skip marker if
    # it is in the dispatch skip tables (generic or CUDA), and prints it plain if in the dispatch
    # expected-failure tables (generic or CUDA). Used by the main block when comparing against
    # external operator lists.
    op = OperatorName.parse(op_str)
    packet = getattr(torch.ops.aten, str(op.name))
    overload = getattr(packet, op.overload_name if op.overload_name else "default")
    if any(overload in d for d in [meta_dispatch_skips, meta_dispatch_device_skips['cuda']]):
        print(f"{overload}  # SKIP")
    if any(overload in d for d in [meta_dispatch_expected_failures, meta_dispatch_device_expected_failures['cuda']]):
        print(overload)


if __name__ == "__main__":
    COMPARE_XLA = os.getenv('PYTORCH_COMPARE_XLA', None)
    if COMPARE_XLA is not None:
        with open(COMPARE_XLA) as f:
            d = yaml.load(f, Loader=YamlLoader)
            ops = d.get("full_codegen", []) + d.get("supported", []) + d.get("autograd", [])
            for op_str in ops:
                print_op_str_if_not_supported(op_str)
        sys.exit(0)

    COMPARE_TEXT = os.getenv('PYTORCH_COMPARE_TEXT', None)
    if COMPARE_TEXT is not None:
        with open(COMPARE_TEXT) as f:
            for op_str in f:
                print_op_str_if_not_supported(op_str.strip())
        sys.exit(0)

    run_tests()
