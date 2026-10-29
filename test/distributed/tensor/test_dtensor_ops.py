# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] OpInfo-driven correctness tests for DTensor ops. For every op in the OpInfo
# database it runs the op on plain tensors and on DTensors built from many sharding combinations
# (via DTensorConverter) and compares results (and tries a backward). Test flavors: multi-threaded
# ranks, LocalTensor mode with a fake process group, unbacked-symint fullgraph compile, single-dim
# sharding strategy validation, and aot_eager compile smoke tests. Import-time side effects:
# overwrites the OpInfo size constants L, M, S, XS in the common methods module (24, 12, 4, 2) so
# shapes shard evenly across 4 ranks, defines the expected-failure and skip sets per flavor, and
# instantiates device-type test classes for the cpu device only (DEVICE_TYPE is hard-coded to cpu).

import copy
import random
import re
import threading
import warnings

import torch
import torch._dynamo
import torch.distributed as dist
import torch.testing._internal.common_methods_invocations as common_ops
from torch.distributed._local_tensor import LocalTensorMode, reconcile_args
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed.tensor._ops.single_dim_strategy import _ShardingPlaceholder
from torch.overrides import resolve_name
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    skip,
    skipOps,
    xfail,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_ops_unbacked import ops_dde_xfail, ops_unbacked_skip
from torch.testing._internal.common_utils import (
    freeze_rng_state,
    np,
    run_tests,
    SEED,
    suppress_warnings,
    TEST_NUMPY,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorConverter,
    DTensorOpTestBase,
    validate_sharding_rule_sample,
    validate_sharding_rule_sample_backward,
)
from torch.utils import _pytree as pytree
from torch.utils._debug_mode import _OpCall, DebugMode
from torch.utils._pytree import tree_flatten, tree_map


# rewrite common size variables to sth can be sharded evenly
# we can enable uneven shards later, but need to adjust more on
# sample inputs (i.e. view/reshape need to adjust shape size as well)
common_ops.L = 24
common_ops.M = 12
common_ops.S = 4
common_ops.XS = 2


def repurpose_ops(op_db, base_test_name, derived_test_name):
    # [FUNCTION DOC] Returns a deep copy of the OpInfo list in which every decorator whose
    # cls_name equals base_test_name now has cls_name set to derived_test_name, so skip and
    # xfail decorators registered for the base class also apply to a derived test class. The
    # input list is not modified.
    """
    Copies op info database and for the decorators that applied to base test class updates
    them to apply to derived test class. The class update is required because decorators are applied
    only if the class name matches (it doesn't consider base classes).

    Specifically we use this function to create two test classes (one for multi-threaded and one for
    local tensor flavors) that share common test body but different rules for skip or fail.

    Args:
        op_db: List of OpInfo objects to be repurposed.
        base_test_name: The original test class name to be replaced.
        derived_test_name: The new test class name to set in decorators.

    Returns:
        list: A new list of OpInfo objects with updated target class names for the
        decorator.
    """
    repurposed_ops = []
    for opinfo in op_db:
        opinfo_copy = copy.deepcopy(opinfo)
        for decorator in list(opinfo_copy.decorators):
            if hasattr(decorator, "cls_name") and decorator.cls_name == base_test_name:
                decorator.cls_name = derived_test_name
        repurposed_ops.append(opinfo_copy)
    return repurposed_ops


# Re-generate this failed list, turn on dry_run of the below func
# check_dtensor_func(self, test, op, dry_run=True), then run sth
# like python test/distributed/tensor/test_dtensor_ops.py > failed.expect
dtensor_fails = {
    # view/reshape ops: rejects flatten/split of sharded dims without redistribution
    xfail("repeat_interleave"),
    xfail("unbind"),
    xfail("unflatten"),
    # factory/creation ops: test harness can't convert non-tensor args to DTensor
    xfail("arange"),
    xfail("broadcast_shapes"),
    xfail("eye"),
    xfail("full"),
    xfail("linspace"),
    xfail("logspace"),
    xfail("ones"),
    xfail("scalar_tensor"),
    xfail("signal.windows.bartlett"),
    xfail("signal.windows.blackman"),
    xfail("signal.windows.cosine"),
    xfail("signal.windows.exponential"),
    xfail("signal.windows.gaussian"),
    xfail("signal.windows.general_cosine"),
    xfail("signal.windows.general_hamming"),
    xfail("signal.windows.hamming"),
    xfail("signal.windows.hann"),
    xfail("signal.windows.kaiser"),
    xfail("signal.windows.nuttall"),
    xfail("zeros"),
    # random/stochastic ops: different RNG states between DTensor and reference
    xfail("bernoulli"),
    xfail("cauchy"),
    xfail("exponential"),
    xfail("nn.functional.alpha_dropout"),
    xfail("nn.functional.dropout"),
    xfail("normal"),
    xfail("normal", "in_place"),
    xfail("normal", "number_mean"),
    xfail("rand_like"),
    xfail("randint"),
    xfail("randint_like"),
    xfail("randn"),
    xfail("randn_like"),
    xfail("uniform"),
    # mixed Tensor/DTensor inputs: op creates plain Tensors mixed with DTensor args
    xfail("__getitem__"),
    xfail("nn.functional.fractional_max_pool2d"),
    xfail("nn.functional.fractional_max_pool3d"),
    xfail("pca_lowrank"),
    xfail("quantile"),
    xfail("svd_lowrank"),
    # dynamic output shape: output shape depends on data values
    xfail("combinations"),
    xfail("linalg.lstsq"),
    xfail("linalg.lstsq", "grad_oriented"),
    xfail("masked_select"),
    xfail("nn.functional.ctc_loss"),
    # 0-dim tensor edge cases: strategies don't handle scalar tensors
    xfail("transpose"),
    # conv stride+padding: TP convolution rejects stride != 1 with padding
    xfail("nn.functional.conv1d"),
    xfail("nn.functional.conv2d"),
    xfail("nn.functional.conv3d"),
    xfail("nn.functional.conv_transpose1d"),
    xfail("nn.functional.conv_transpose2d"),
    xfail("nn.functional.conv_transpose3d"),
    # "cannot resize variables that require grad" from test harness
    xfail("resize_"),
    xfail("resize_as_"),
    # DTensorConverter can't convert sparse tensor inputs
    xfail("sparse.sampled_addmm"),
    xfail("sparse.mm", "reduce"),
    # meta tensor data not allocated yet during tensor_split
    xfail("tensor_split"),
    xfail("torch.ops.aten._scaled_dot_product_flash_attention_for_cpu"),
    # /TODO(whc) debug/triage
    # ops inside this might even fail without dtensor
    # tests, as we rescale op db common test size factor (i.e. L, M, S)
    # which triggered the original function run failures with input
    # generation becomes wrong, we skip them for now but should enable later.
    # TODO: need to clean this list and remove all cases
    skip("argwhere"),
    skip("cumprod"),
    skip("__rmatmul__"),
    skip("meshgrid", "list_of_tensors"),
    skip("meshgrid", "variadic_tensors"),
    skip("nn.functional.scaled_dot_product_attention"),
    skip("nn.functional.softmin"),
    skip("nn.functional.embedding"),
    skip("nn.functional.embedding_bag"),
    skip("nn.functional.feature_alpha_dropout", "with_train"),
    skip("nn.functional.feature_alpha_dropout", "without_train"),
    skip("nn.functional.hinge_embedding_loss"),
    skip("nn.functional.cosine_embedding_loss"),
    skip("fft.hfft"),
    skip("fft.hfft2"),
    skip("fft.hfft2"),
    skip("fft.hfftn"),
    skip("fft.ifftn"),
    skip("fft.irfft"),
    skip("frexp"),
    skip("istft"),
    skip("isclose"),
    skip("isreal"),
    skip("matmul"),
    skip("masked.mean"),
    skip("masked.var"),
    skip("masked.std"),
    skip("masked.normalize"),
    skip("prod"),
    skip("_segment_reduce", "lengths"),
    skip("_segment_reduce", "offsets"),
    # TODO: fix the following ops
    skip("empty"),
    skip("empty_strided"),
    skip("empty_like"),
    skip("empty_permuted"),
    skip("new_empty"),
    skip("new_empty_strided"),
}

dtensor_multi_threaded_fails = {
    xfail("nn.functional.dropout2d"),
    xfail("nn.functional.dropout3d"),
    skip("nn.functional.multi_head_attention_forward"),
    # Nondeterministic: DTensor vs reference sample from the shared RNG at
    # different points, so an xfail is flaky (occasional "unexpected success").
    skip("multinomial"),
    # Flaky in CI: https://github.com/pytorch/pytorch/issues/167252
    skip("full_like"),
    # Flaky in CI: https://github.com/pytorch/pytorch/issues/179779
    skip("bmm"),
    # Flaky in CI: https://github.com/pytorch/pytorch/issues/180522
    skip("baddbmm"),
}

# Ops that fail to compile with DTensor + torch.compile(fullgraph=True).
# These are compile-time failures, NOT numeric correctness issues.
dtensor_compiled_fails = {
    xfail("cartesian_prod"),
    xfail("flatten"),
    xfail("kron"),
    xfail("ravel"),
    xfail("reshape"),
    xfail("reshape_as"),
    xfail("view"),
    xfail("view_as"),
    # No sharding strategy for aten.transpose_copy.int on 0-d inputs.
    xfail("transpose_copy"),
    # View-type ops that decompose into as_strided (at autograd level).
    # DTensor only answers as_strided when it is a permutation of the base
    # dims, which these are not.
    xfail("atleast_1d"),
    xfail("atleast_2d"),
    xfail("atleast_3d"),
    xfail("broadcast_tensors"),
    xfail("broadcast_to"),
    xfail("diagonal"),
    xfail("dsplit"),
    xfail("expand"),
    xfail("expand_as"),
    xfail("hsplit"),
    xfail("linalg.diagonal"),
    xfail("narrow"),
    xfail("select"),
    xfail("slice"),
    xfail("squeeze"),
    xfail("squeeze", "multiple"),
    xfail("unsqueeze"),
    xfail("vsplit"),
    # Decompositions that use plain tensor constructors (e.g. arange),
    # causing mixed tensor/DTensor errors during Dynamo's fake prop.
    xfail("corrcoef"),
    xfail("cov"),
    xfail("nn.functional.interpolate", "bicubic"),
    xfail("nn.functional.interpolate", "bilinear"),
    xfail("nn.functional.interpolate", "linear"),
    xfail("nn.functional.interpolate", "nearest"),
    xfail("nn.functional.interpolate", "nearest-exact"),
    xfail("nn.functional.interpolate", "trilinear"),
    xfail("nn.functional.upsample_bilinear"),
    xfail("nn.functional.upsample_nearest"),
    # Data-dependent outputs (SymBool, unbacked shapes) that raise
    # during DTensor's fake prop.
    xfail("equal"),
    xfail("hash_tensor"),
    xfail("item"),
    xfail("nonzero_static"),
    # Decompositions with .is_cuda checks that fail during sharding
    # propagation for aten.is_cuda / prim::device.
    xfail("nn.functional.binary_cross_entropy_with_logits"),
    xfail("nn.functional.gaussian_nll_loss"),
    xfail("nn.functional.logsigmoid"),
    xfail("scatter"),
    # False positives: these have no sharding strategy and their
    # eager DTensor failure is registered elsewhere.
    xfail("nn.functional.multilabel_soft_margin_loss"),
    # Flaky in CI: https://github.com/pytorch/pytorch/issues/181204
    skip("norm", "nuc"),
    # Flaky in CI: https://github.com/pytorch/pytorch/issues/176973
    skip("histc"),
    xfail("nn.functional.linear_cross_entropy", "chunked"),
    xfail("nn.functional.linear_cross_entropy", "chunked_none"),
}

# Ops that compile successfully but fail numeric checks in eager DTensor tests.
# These are excluded from TestCompiledDTensorOps skip list since we don't check numerics.
dtensor_numeric_only_fails = {
    xfail("arange"),
    xfail("broadcast_shapes"),
    xfail("eye"),
    xfail("full"),
    xfail("full_like"),
    xfail("index_fill"),
    xfail("linspace"),
    xfail("logspace"),
    xfail("nn.functional.huber_loss"),
    xfail("nn.functional.max_unpool3d", "grad"),
    xfail("nn.functional.smooth_l1_loss"),
    xfail("nn.functional.softshrink"),
    xfail("nn.functional.threshold"),
    xfail("ones"),
    xfail("randint"),
    xfail("randn"),
    xfail("scalar_tensor"),
    xfail("signal.windows.bartlett"),
    xfail("signal.windows.blackman"),
    xfail("signal.windows.cosine"),
    xfail("signal.windows.exponential"),
    xfail("signal.windows.gaussian"),
    xfail("signal.windows.general_cosine"),
    xfail("signal.windows.general_hamming"),
    xfail("signal.windows.hamming"),
    xfail("signal.windows.hann"),
    xfail("signal.windows.kaiser"),
    xfail("signal.windows.nuttall"),
    xfail("sparse.mm", "reduce"),
    xfail("sparse.sampled_addmm"),
    xfail("squeeze_copy"),
    xfail("stack"),
    xfail("unsafe_chunk"),
    xfail("zeros"),
}

# Ops in dtensor_fails that have no sharding strategy (NotImplementedError).
# These will error during sharding propagation and affect unbacked tests too.
dtensor_fails_no_strategy = {
    xfail("_chunk_cat"),
    xfail("_unsafe_masked_index"),
    xfail("_unsafe_masked_index_put_accumulate"),
    xfail("addbmm"),
    xfail("allclose"),
    xfail("as_strided"),
    xfail("as_strided", "partial_views"),
    xfail("as_strided_copy"),
    xfail("as_strided_scatter"),
    xfail("block_diag"),
    xfail("cdist"),
    xfail("complex"),
    xfail("exponential"),
    xfail("fft.ihfft2"),
    xfail("fft.ihfftn"),
    xfail("geometric"),
    xfail("histogram"),
    xfail("histogramdd"),
    xfail("isin"),
    xfail("linalg.matrix_power"),
    # Full-matrix op; matrix dims can't be sharded, like matrix_exp/matrix_power.
    xfail("linalg.matrix_sqrth"),
    xfail("linalg.polar"),
    xfail("linspace", "tensor_overload"),
    xfail("log_normal"),
    xfail("logspace", "tensor_overload"),
    xfail("masked_scatter"),
    xfail("nanquantile"),
    xfail("nn.functional.bilinear"),
    xfail("nn.functional.linear_cross_entropy", "chunked"),
    xfail("nn.functional.linear_cross_entropy", "chunked_none"),
    xfail("nn.functional.multi_margin_loss"),
    xfail("nn.functional.multilabel_margin_loss"),
    xfail("nn.functional.pad", "reflect"),
    xfail("nn.functional.pad", "replicate"),
    xfail("nn.functional.pad", "replicate_negative"),
    xfail("nn.functional.pdist"),
    xfail("nn.functional.rrelu"),
    xfail("nn.functional.unfold"),
    xfail("nonzero"),
    xfail("polar"),
    xfail("put"),
    xfail("renorm"),
    xfail("scatter_reduce", "amax"),
    xfail("scatter_reduce", "amin"),
    xfail("scatter_reduce", "mean"),
    xfail("scatter_reduce", "prod"),
    xfail("scatter_reduce", "sum"),
    xfail("searchsorted"),
    xfail("squeeze_copy"),
    xfail("stft"),
    xfail("take"),
    xfail("to_sparse"),
    xfail("unfold"),
    xfail("unfold_copy"),
    xfail("unique"),
    xfail("unique_consecutive"),
}

# Add a list of ops that are currently failing BW pass
skip_bw = [
    None,  # corresponds to the transpose ops 'H' and 'T'
    "torch.bucketize",
    "torch.conj_physical",
    "torch.eq",
    "torch.isfinite",
    "torch.isnan",
]


OP_DB_WORLD_SIZE = 4
# DEVICE_TYPE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() >= OP_DB_WORLD_SIZE else "cpu"
# TODO: debug cuda illegal memory access issue and re-enable cuda tests
DEVICE_TYPE = "cpu"


class TestDTensorOps(TestCase):
    # [CLASS DOC] Shared base for the op test flavors, not collected itself (__test__ is False and
    # is set True on subclasses). Provides sample iteration, the crossref comparison of DTensor
    # versus plain results, and helpers; subclasses implement assertEqualOnRank. world_size is 4
    # and the mesh is stored on the instance when a test runs.
    __test__ = False

    def __init_subclass__(cls, **kwargs):
        # [FUNCTION DOC] HELPER: Runs normal subclass setup and marks each subclass as collectable
        # by setting its test flag to True.
        super().__init_subclass__(**kwargs)
        cls.__test__ = True

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the number of ranks, OP_DB_WORLD_SIZE (4).
        return OP_DB_WORLD_SIZE

    def iter_valid_samples(
        self,
        op,
        dtype,
        requires_grad=False,
        sample_filter=None,
        needs_deepcopy=False,
    ):
        # [FUNCTION DOC] HELPER: Generator over the op's sample inputs for the module device
        # type and the given dtype and requires_grad. Yields (args, kwargs) where args is a list
        # of the sample input followed by its extra args. Samples rejected by sample_filter (called
        # with args and kwargs) are skipped; with needs_deepcopy the args and kwargs are deep
        # copied and samples that raise NotImplementedError when copied are skipped.
        """
        Iterate over valid samples for an op, yielding (args, kwargs) tuples.

        Args:
            op: The OpInfo object
            dtype: The dtype to use for sample inputs
            requires_grad: Whether tensors should require grad
            sample_filter: Optional callable(args, kwargs) -> bool to filter samples
            needs_deepcopy: If True, yields deepcopied args/kwargs and skips
                            samples that can't be deepcopied
        """
        samples = op.sample_inputs(DEVICE_TYPE, dtype, requires_grad=requires_grad)
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            if sample_filter and not sample_filter(args, kwargs):
                continue

            if needs_deepcopy:
                try:
                    args = copy.deepcopy(args)
                    kwargs = copy.deepcopy(kwargs)
                except NotImplementedError:
                    continue

            yield args, kwargs

    def run_opinfo_test(
        self, dtype, op, requires_grad=True, sample_filter=None, sample_lock=None
    ):
        # [FUNCTION DOC] Creates a one-dimensional device mesh of world_size ranks (stored on the
        # instance) and runs the DTensor crossref comparison of the op over all its samples
        # through check_dtensor_func (errors propagate). With a sample_lock (multi-threaded
        # ranks), the samples are materialized once under the lock with fixed seeds for torch,
        # random and numpy (restoring the python and numpy RNG state afterwards), followed by a
        # barrier, so that all ranks see identical non-tensor arguments.
        self.mesh = init_device_mesh(DEVICE_TYPE, (self.world_size,))

        def valid_samples():
            # [FUNCTION DOC] HELPER: Returns the sample generator for the op with the outer
            # dtype, requires_grad and sample filter.
            return self.iter_valid_samples(
                op,
                dtype,
                requires_grad=requires_grad,
                sample_filter=sample_filter,
            )

        def iter_samples():
            # [FUNCTION DOC] HELPER: Generator of samples; without a lock it yields lazily, with
            # the lock it yields from a list materialized under frozen, fixed RNG state.
            if sample_lock is None:
                yield from valid_samples()
                return

            # Threaded ranks share process-global RNG state. Materialize samples
            # with a fixed RNG state so non-tensor args match across ranks.
            with sample_lock:
                python_rng_state = random.getstate()
                numpy_rng_state = np.random.get_state() if TEST_NUMPY else None
                try:
                    with freeze_rng_state():
                        torch.manual_seed(SEED)
                        random.seed(SEED)
                        if TEST_NUMPY:
                            np.random.seed(SEED)
                        samples = list(valid_samples())
                finally:
                    random.setstate(python_rng_state)
                    if numpy_rng_state is not None:
                        np.random.set_state(numpy_rng_state)

            dist.barrier()
            yield from samples

        def test():
            # [FUNCTION DOC] HELPER: Runs the crossref comparison for every sample.
            for args, kwargs in iter_samples():
                self.run_dtensor_crossref(op.op, args, kwargs)

        self.check_dtensor_func(test, op)

    def assert_ref_dtensor_equal(self, dtensor_rs, rs):
        # [FUNCTION DOC] HELPER: Flattens both results with pytree and asserts the same number of
        # leaves. For each reference leaf that is a tensor, asserts (on rank 0 via
        # assertEqualOnRank) that the DTensor-side leaf is a tensor with the same shape and
        # requires_grad flag and equal values. Non-tensor reference leaves are ignored.
        flat_dtensor_rs = pytree.tree_leaves(dtensor_rs)
        flat_rs = pytree.tree_leaves(rs)
        self.assertEqual(len(flat_dtensor_rs), len(flat_rs))
        for dtensor_r, r in zip(flat_dtensor_rs, flat_rs):
            if not isinstance(r, torch.Tensor):
                continue

            self.assertIsInstance(dtensor_r, torch.Tensor)
            self.assertEqualOnRank(
                dtensor_r.shape,
                r.shape,
                f"Shape mismatch! original shape:{r.shape}, dtensor shape: {dtensor_r.shape}",
            )
            self.assertEqualOnRank(
                dtensor_r.requires_grad,
                r.requires_grad,
                "op result requires_grad mismatch!"
                f"original requires_grad: {r.requires_grad}, "
                f"dtensor requires_grad: {dtensor_r.requires_grad}",
            )

            self.assertEqualOnRank(dtensor_r, r)

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0) -> None:
        # [FUNCTION DOC] HELPER: Abstract hook; subclasses must provide it. Raises
        # NotImplementedError here.
        raise NotImplementedError

    def run_dtensor_crossref(self, func, args, kwargs):
        # [FUNCTION DOC] Runs func on the plain args to get the reference result, then on every
        # DTensor argument combination produced by DTensorConverter. Skips combinations whose
        # result has an empty tensor, converts results to full tensors, tries a backward on the
        # local part (any failure there is only printed on rank 0 for ops not in skip_bw), and
        # asserts equality with the reference. Raises RuntimeError if args could not all be
        # converted, and wraps any failure in a RuntimeError naming the op and arguments.
        # Returns the reference result.
        to_dtensor = DTensorConverter(self.mesh, args, kwargs)

        def concat_res_if_necessary(func, res: object) -> object:
            # [FUNCTION DOC] HELPER: For ops whose name contains split, concatenates the sequence
            # result along dim (the third arg when three args are given, else 0) so a single
            # tensor results; otherwise returns the result unchanged.
            # concat the result on corresponding dim for ops like
            # split, so that we can call backward on a single tensor
            if (resolve_name(func) is not None) and ("split" in resolve_name(func)):
                dim = args[2] if len(args) == 3 else 0
                return torch.cat(res, dim=dim)
            else:
                return res

        # TODO: also handle cases where func raise an exception
        op_args, op_kwargs = reconcile_args(args, kwargs)
        rs = func(*op_args, **op_kwargs)
        rs = concat_res_if_necessary(func, rs)

        def to_replicate(e: object) -> object:
            # [FUNCTION DOC] HELPER: Returns the full tensor of a DTensor, else the value as is.
            return e.full_tensor() if isinstance(e, DTensor) else e

        # Suppress warnings, this doesn't matter for test_meta.py
        # but it does matter if you want to use this decorator
        # for cross-ref testing, as some tests may be looking at
        # errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # for every comb of sharding choices, we test if it works
            for dtensor_args, dtensor_kwargs in to_dtensor:
                # Only attempt if we managed to convert all tensors to DTensor
                # (if any of them failed, we're in a mixed tensor situation and
                # this is not allowed in DTensor)
                try:
                    if to_dtensor.successful():
                        # Handle special cases first if there's any
                        # Suppress warnings, this doesn't matter for test_meta.py
                        # but it does matter if you want to use this decorator
                        # for cross-ref testing, as some tests may be looking at
                        # errors
                        dtensor_rs = func(*dtensor_args, **dtensor_kwargs)

                        # we need to skip tests containing tensors of zero elements for now.
                        # see issue: https://github.com/pytorch/PiPPy/issues/470
                        # TODO remove this once issue above fixed.
                        flat_args = pytree.tree_leaves(dtensor_rs)
                        if any(
                            isinstance(e, torch.Tensor) and e.numel() == 0
                            for e in flat_args
                        ):
                            continue

                        # redistribute/all_gather the results to compare with normal output
                        dtensor_rs = tree_map(to_replicate, dtensor_rs)
                        dtensor_rs = concat_res_if_necessary(func, dtensor_rs)
                        try:
                            if resolve_name(func) not in skip_bw:
                                if isinstance(dtensor_rs, DTensor):
                                    dtensor_rs.to_local().sum().backward()
                                elif isinstance(dtensor_rs, tuple):
                                    dtensor_rs[0].to_local().sum().backward()

                        except Exception as e:
                            # TODO(anj): Remove this guard exception after gaining more confidence.
                            if torch.distributed.get_rank() == 0:
                                print(
                                    f"failed to run BW: {resolve_name(func)}, {func}, {str(e)})"
                                )
                        self.assert_ref_dtensor_equal(dtensor_rs, rs)
                    else:
                        raise RuntimeError(
                            f"Failed to convert args to DTensor; "
                            f"originally (*{args}, **{kwargs})"
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"{str(e)}\n\nFailed to run: {resolve_name(func)}, with (*{dtensor_args}, **{dtensor_kwargs})"
                    ) from e
        return rs

    def check_dtensor_func(self, test_func, opinfo, dry_run=False):
        # [FUNCTION DOC] Calls test_func. If it raises, re-raises unless dry_run is set, in which
        # case rank 0 prints an xfail entry for the op (with the variant name if present) to help
        # regenerate the expected-failure list.
        try:
            test_func()
        except Exception:
            if not dry_run:
                raise
            if dist.get_rank() == 0:
                if opinfo.variant_test_name:
                    print(f"xfail('{opinfo.name}', '{opinfo.variant_test_name}'),")
                else:
                    print(f"xfail('{opinfo.name}'),")

    def run_one_hot(self):
        # [FUNCTION DOC] HELPER: Finds the single OpInfo for one_hot (raises AssertionError if
        # not exactly one) and runs the opinfo test with int64, no grad, skipping samples with
        # num_classes of minus one.
        ops = [op for op in op_db if op.name == "nn.functional.one_hot"]
        if len(ops) != 1:
            raise AssertionError(f"Expected 1 op, got {len(ops)}")
        op = ops[0]
        # num_classes = -1 appears to have a bug with dtensor.max().item()
        self.run_opinfo_test(
            torch.int64,
            op,
            requires_grad=False,
            sample_filter=lambda args, kwargs: kwargs.get("num_classes") != -1,
        )

    def run_mean(self):
        # [FUNCTION DOC] HELPER: Builds a 1-D mesh and a float (2w+1, 2w) arange tensor. For an
        # evenly shardable case (sharded on dim 1, mean over dim 1) and an uneven case (sharded
        # on dim 0, mean over dim 0) computes the DTensor mean under a debug mode, asserts the
        # full tensor equals the plain mean, and asserts the debug log shows a partial-average to
        # replicate redistribution for the even case and a shard-0 to replicate redistribution
        # for the uneven case.
        self.mesh = init_device_mesh(DEVICE_TYPE, (self.world_size,))

        shape = [2 * self.world_size + 1, 2 * self.world_size]
        tensor = (
            torch.arange(shape[0] * shape[1], dtype=torch.float32)
            .reshape(shape)
            .to(DEVICE_TYPE)
        )

        for is_evenly_shardable in [True, False]:
            if is_evenly_shardable:
                placement = [Shard(1)]
                reduce_dim = 1
            else:
                placement = [Shard(0)]
                reduce_dim = 0
            dtensor = distribute_tensor(tensor, self.mesh, placement)

            with DebugMode(record_torchfunction=False) as debug_mode:
                mean = dtensor.mean(dim=reduce_dim)
                full_tensor = mean.full_tensor()

            self.assertEqual(full_tensor, tensor.mean(dim=reduce_dim))

            if is_evenly_shardable:
                self.assertTrue("P(avg)->R" in debug_mode.debug_string())
            else:
                self.assertTrue("S(0)->R" in debug_mode.debug_string())

    def test_embedding_error_msg(self):
        # [FUNCTION DOC] Builds a (2, 2) dp/tp mesh, shards a weight on the tp sub-mesh and an
        # index tensor on the full 2-D mesh, and asserts calling aten embedding on the two (mesh
        # mismatch) raises a RuntimeError whose message starts with the expected sharding
        # propagation failure text.
        self.mesh_2d = init_device_mesh(
            DEVICE_TYPE, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        self.mesh_1d = self.mesh_2d["tp"]

        weight_global = torch.randn(2048, 256, device=DEVICE_TYPE)
        weight_dtensor = distribute_tensor(weight_global, self.mesh_1d, [Shard(0)])

        input_global = torch.randint(0, 2048, (16, 2048), device=DEVICE_TYPE)
        input_dtensor = distribute_tensor(
            input_global, self.mesh_2d, [Shard(0), Replicate()]
        )

        expected_error_msg = (
            "Sharding propagation failed for aten.embedding.default"
            "(Spec(f32[2048, 256](S(0))), Spec(i64[16, 2048](S(0)R))) "
            "on DeviceMesh((dp=2, tp=2), "
        )
        with self.assertRaisesRegex(RuntimeError, re.escape(expected_error_msg)):
            _ = torch.ops.aten.embedding.default(weight_dtensor, input_dtensor)


class TestMultiThreadedDTensorOps(DTensorOpTestBase, TestDTensorOps):
    # [CLASS DOC] Op tests running one thread per rank. Holds a repurposed op database and a
    # class-wide lock used to materialize samples consistently across threads.
    _op_db = repurpose_ops(op_db, "TestDTensorOps", "TestMultiThreadedDTensorOps")
    _op_db_sample_lock = threading.Lock()

    @suppress_warnings
    @ops(_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        dtensor_fails | dtensor_multi_threaded_fails | dtensor_fails_no_strategy,
    )
    def test_dtensor_op_db(self, dtype, op):
        # [FUNCTION DOC] Per-op test for float: runs the opinfo test with the shared sample lock,
        # skipping the multi-threaded, base and no-strategy failure sets.
        self.run_opinfo_test(dtype, op, sample_lock=self.__class__._op_db_sample_lock)

    def test_mean(self):
        # [FUNCTION DOC] Runs the mean check.
        self.run_mean()

    def test_one_hot(self):
        # [FUNCTION DOC] Runs the one_hot check.
        self.run_one_hot()


class TestLocalDTensorOps(TestDTensorOps):
    # [CLASS DOC] Op tests running all ranks in a single process with LocalTensor mode over a
    # fake process group of world_size ranks.
    _op_db = repurpose_ops(op_db, "TestDTensorOps", "TestLocalDTensorOps")

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Runs base setUp and initializes a fake process group (rank 0) and
        # stores its default group.
        super().setUp()
        torch.distributed.init_process_group("fake", rank=0, world_size=self.world_size)
        self.fake_pg = torch.distributed.distributed_c10d._get_default_group()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs base tearDown, clears the sharding propagation cache, and
        # destroys the process group, ignoring an AssertionError if there is none.
        super().tearDown()
        # Clear sharding propagation cache to avoid stale mesh references
        # between tests that destroy and recreate process groups
        from torch.distributed.tensor.debug import _clear_sharding_prop_cache

        _clear_sharding_prop_cache()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    @suppress_warnings
    @ops(_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        dtensor_fails | dtensor_fails_no_strategy,
    )
    def test_dtensor_op_db(self, dtype, op):
        # [FUNCTION DOC] Per-op test for float: runs the opinfo test, skipping the base and
        # no-strategy failure sets.
        self.run_opinfo_test(dtype, op)

    def test_mean(self):
        # [FUNCTION DOC] Runs the mean check inside local tensor mode for all ranks.
        with LocalTensorMode(frozenset(range(self.world_size))):
            self.run_mean()

    def test_one_hot(self):
        # [FUNCTION DOC] Runs the one_hot check.
        self.run_one_hot()

    def run_opinfo_test(self, dtype, op, requires_grad=True, sample_filter=None):
        # [FUNCTION DOC] HELPER: Runs the base opinfo test inside local tensor mode covering all
        # ranks (no sample lock parameter).
        with LocalTensorMode(frozenset(range(self.world_size))):
            super().run_opinfo_test(dtype, op, requires_grad, sample_filter)

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        # [FUNCTION DOC] HELPER: Plain equality assertion; the rank argument is ignored since a
        # single process holds all ranks.
        self.assertEqual(x, y, msg)


# Ops where DTensor shard prop has DDEs with unbacked (base tensor passes).
# This list only contains ops NOT in ops_dde_xfail - those are base tensor issues.
ops_unbacked_dtensor_dde = {
    xfail("lu_unpack"),
    xfail("__getitem__"),
    xfail("__rmatmul__"),
    xfail("_batch_norm_with_update"),
    xfail("_segment_reduce", "lengths"),
    xfail("_segment_reduce", "offsets"),
    xfail("_native_batch_norm_legit"),
    xfail("_unsafe_masked_index"),
    xfail("addmm"),
    xfail("addmm", "decomposed"),
    xfail("addr"),
    xfail("alias_copy"),
    xfail("aminmax"),
    xfail("argsort"),
    xfail("argwhere"),
    xfail("as_strided"),
    xfail("as_strided", "partial_views"),
    xfail("as_strided_copy"),
    xfail("block_diag"),
    xfail("bucketize"),
    skip("broadcast_to"),
    xfail("bucketize"),
    xfail("cartesian_prod"),
    xfail("combinations"),
    xfail("constant_pad_nd"),
    xfail("cumprod"),
    xfail("diagonal_scatter"),
    xfail("dist"),
    xfail("fill"),
    xfail("flatten"),
    xfail("flip"),
    xfail("fliplr"),
    xfail("flipud"),
    xfail("float"),
    xfail("gather"),
    xfail("histc"),
    xfail("index_add"),
    xfail("index_copy"),
    xfail("index_put"),
    xfail("index_reduce", "amax"),
    xfail("index_reduce", "amin"),
    xfail("index_reduce", "mean"),
    xfail("index_reduce", "prod"),
    xfail("index_select"),
    xfail("kthvalue"),
    xfail("linalg.lu"),
    xfail("linalg.lu_factor"),
    xfail("linalg.lu_factor_ex"),
    xfail("lu"),
    xfail("masked_fill"),
    xfail("masked_scatter"),
    xfail("masked_select"),
    xfail("matmul"),
    xfail("matrix_exp"),
    xfail("max", "reduction_with_dim"),
    xfail("min", "reduction_with_dim"),
    xfail("mode"),
    xfail("msort"),
    xfail("mv"),
    xfail("narrow"),
    xfail("narrow_copy"),
    xfail("new_empty"),
    xfail("new_empty_strided"),
    xfail("new_full"),
    xfail("new_ones"),
    xfail("native_batch_norm"),
    xfail("new_zeros"),
    xfail("nn.functional.batch_norm"),
    xfail("nn.functional.conv1d"),
    xfail("nn.functional.conv2d"),
    xfail("nn.functional.conv3d"),
    xfail("nn.functional.conv_transpose1d"),
    xfail("nn.functional.conv_transpose2d"),
    xfail("nn.functional.conv_transpose3d"),
    xfail("nn.functional.cosine_embedding_loss"),
    xfail("nn.functional.glu"),
    xfail("nn.functional.hinge_embedding_loss"),
    xfail("nn.functional.interpolate", "nearest"),
    xfail("nn.functional.interpolate", "nearest-exact"),
    xfail("nn.functional.linear"),
    xfail("nn.functional.logsigmoid"),
    xfail("nn.functional.margin_ranking_loss"),
    xfail("nn.functional.multilabel_soft_margin_loss"),
    xfail("nn.functional.pad", "constant"),
    xfail("nn.functional.poisson_nll_loss"),
    xfail("nn.functional.soft_margin_loss"),
    xfail("nn.functional.triplet_margin_loss"),
    xfail("nn.functional.triplet_margin_with_distance_loss"),
    xfail("nn.functional.upsample_nearest"),
    xfail("nonzero_static"),
    xfail("permute_copy"),
    xfail("prod"),
    xfail("quantile"),
    xfail("ravel"),
    xfail("reshape"),
    xfail("reshape_as"),
    xfail("rsub"),
    xfail("rot90"),
    xfail("scatter"),
    xfail("scatter_add"),
    xfail("select_scatter"),
    xfail("sort"),
    xfail("squeeze_copy"),
    xfail("std_mean"),
    xfail("topk"),
    xfail("transpose_copy"),
    xfail("unflatten"),
    xfail("unsqueeze_copy"),
    xfail("vdot"),
    xfail("view"),
    xfail("view_as"),
    xfail("view_as_complex"),
    xfail("torch.ops.aten._scaled_dot_product_flash_attention_for_cpu"),
}


class TestUnbackedDTensorOps(TestDTensorOps):
    # [CLASS DOC] Op tests with a fake process group that mark DTensor dims of size at least 2
    # as unbacked and run the op under an eager-backend fullgraph compile to detect
    # data-dependent errors. Only compilation is exercised; values are not compared.
    """
    Test suite for DTensor ops with unbacked symints.

    This runs correctness tests with tensor dimensions marked as unbacked
    and the op compiled with fullgraph=True to catch DDEs during tracing.
    """

    _op_db = repurpose_ops(op_db, "TestDTensorOps", "TestUnbackedDTensorOps")

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Runs base setUp and initializes a fake process group at rank 0.
        super().setUp()
        torch.distributed.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs base tearDown, clears the sharding propagation cache, resets
        # dynamo, and destroys the process group, ignoring an AssertionError.
        super().tearDown()
        from torch.distributed.tensor.debug import _clear_sharding_prop_cache

        _clear_sharding_prop_cache()
        torch._dynamo.reset()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        # [FUNCTION DOC] HELPER: Plain equality assertion; the rank argument is ignored.
        self.assertEqual(x, y, msg)

    def _has_valid_unbacked_dims(self, t: torch.Tensor) -> bool:
        # [FUNCTION DOC] HELPER: True if the tensor has at least one dim and some dim of size at
        # least 2 (so it can be marked unbacked).
        """Check if tensor has dimensions that can be marked as unbacked."""
        return t.ndim > 0 and any(s >= 2 for s in t.shape)

    def _sample_has_valid_unbacked_dims(self, args, kwargs) -> bool:
        # [FUNCTION DOC] HELPER: Sample filter: True if any tensor in the flattened args and
        # kwargs has a dim that can be marked unbacked.
        """Check if any tensor in args/kwargs has valid unbacked dimensions."""
        all_tensors = [
            x for x in tree_flatten((args, kwargs))[0] if isinstance(x, torch.Tensor)
        ]
        return any(self._has_valid_unbacked_dims(t) for t in all_tensors)

    def _mark_unbacked(self, t: torch.Tensor) -> None:
        # [FUNCTION DOC] HELPER: Marks every dim of size at least 2 of the tensor as unbacked in
        # dynamo (mutates dynamo markers on the tensor).
        """Mark all eligible dimensions of a tensor as unbacked."""
        for i in range(t.ndim):
            if t.shape[i] >= 2:
                torch._dynamo.decorators.mark_unbacked(t, i)

    def run_dtensor_crossref(self, func, args, kwargs):
        # [FUNCTION DOC] Override of the crossref: computes the plain reference result, then for
        # each DTensor argument combination marks eligible dims of the DTensor arguments as
        # unbacked, resets dynamo, and calls func through an eager-backend fullgraph compile.
        # Any failure (or failed conversion) is re-raised as a RuntimeError naming the op and
        # arguments. Returns the reference result.
        # KNOWN ISSUE: the compiled output is never compared with the reference, and the local
        # to_replicate helper is unused.
        """
        Override to add unbacked marking and fullgraph compilation.

        Same as parent but:
        1. Marks DTensor dimensions as unbacked before running
        2. Wraps the op in @torch.compile(backend="eager", fullgraph=True)
        """
        to_dtensor = DTensorConverter(self.mesh, args, kwargs)

        def concat_res_if_necessary(func, res: object) -> object:
            # [FUNCTION DOC] HELPER: Concatenates the result of split-like ops along a dim (the
            # third arg if three args, else 0); otherwise returns the result.
            if (resolve_name(func) is not None) and ("split" in resolve_name(func)):
                dim = args[2] if len(args) == 3 else 0
                return torch.cat(res, dim=dim)
            return res

        op_args, op_kwargs = reconcile_args(args, kwargs)
        rs = func(*op_args, **op_kwargs)
        rs = concat_res_if_necessary(func, rs)

        def to_replicate(e: object) -> object:
            # [FUNCTION DOC] HELPER: Returns the full tensor of a DTensor, else the value as is
            # (unused in this override).
            return e.full_tensor() if isinstance(e, DTensor) else e

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for dtensor_args, dtensor_kwargs in to_dtensor:
                try:
                    if not to_dtensor.successful():
                        raise RuntimeError(
                            f"Failed to convert args to DTensor; "
                            f"originally (*{args}, **{kwargs})"
                        )

                    pytree.tree_map_only(
                        lambda x: isinstance(x, DTensor)
                        and self._has_valid_unbacked_dims(x),
                        self._mark_unbacked,
                        (dtensor_args, dtensor_kwargs),
                    )

                    # Compile with fullgraph=True to catch DDEs
                    torch._dynamo.reset()

                    @torch.compile(backend="eager", fullgraph=True)
                    def compiled_func(*a, **kw):
                        # [FUNCTION DOC] HELPER: Calls the op under test with the given arguments.
                        return func(*a, **kw)

                    compiled_func(*dtensor_args, **dtensor_kwargs)

                except Exception as e:
                    raise RuntimeError(
                        f"{str(e)}\n\nFailed to run: {resolve_name(func)}, "
                        f"with (*{dtensor_args}, **{dtensor_kwargs})"
                    ) from e
        return rs

    @suppress_warnings
    @ops(_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        ops_dde_xfail
        | ops_unbacked_dtensor_dde
        | dtensor_fails_no_strategy
        | ops_unbacked_skip,
    )
    def test_unbacked_dtensor_op_db(self, dtype, op):
        # [FUNCTION DOC] Per-op test for float: runs the opinfo test without grad on samples that
        # have markable dims, skipping the unbacked-specific, no-strategy and skip sets.
        # Filter to samples with valid unbacked dimensions
        self.run_opinfo_test(
            dtype,
            op,
            requires_grad=False,
            sample_filter=self._sample_has_valid_unbacked_dims,
        )


class TestSingleDimStrategies(DTensorOpTestBase):
    # [CLASS DOC] Validates registered single-dim sharding strategies against real op results on
    # a 2-rank multi-threaded mesh.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the fixed world size, 2.
        return 2

    def _extract_aten_op_and_args(self, torch_op, args, kwargs):
        # [FUNCTION DOC] HELPER: Runs torch_op on the DTensor args under a debug mode that stores
        # the original args, skipping the test if the op raises. Returns the op, args and kwargs
        # of the first recorded call whose op name contains aten; skips the test if none.
        with DebugMode(store_original_args=True) as debug_mode:
            try:
                torch_op(*args, **kwargs)
            except Exception:
                self.skipTest(f"Op {torch_op} failed on replicated DTensors")

        for op in debug_mode.operators:
            if isinstance(op, _OpCall) and "aten" in str(op.op):
                return op.op, op.args, op.kwargs

        self.skipTest(f"Op {torch_op} failed to extract aten op")

    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        {
            # Value validation cannot compare nondeterministic or
            # uninitialized outputs shard-by-shard.
            skip("exponential"),
            skip("geometric"),
            skip("log_normal"),
            skip("nn.functional.rrelu"),
            skip("normal", "in_place"),
            skip("uniform"),
        },
    )
    def test_single_dim_strategy(self, dtype, op):
        # [FUNCTION DOC] Takes the first sample of the op (skipping if sampling fails or is
        # empty), replicates its tensors on a 2-rank mesh, finds the underlying aten call, and
        # skips if it has no registered single-dim strategy. Enumerates the strategy's placement
        # rules (placeholders turned into shards), skips ones with invalid shards (not a shard,
        # scalar, out-of-range or empty or non-divisible dim), and asserts each remaining rule
        # gives correct forward results on full tensors and, when a backward check applies, a
        # correct backward. Seeds the RNG with 42.
        torch.manual_seed(42)
        mesh = init_device_mesh(DEVICE_TYPE, (self.world_size,))
        sharding_prop = DTensor._op_dispatcher.sharding_propagator

        try:
            samples = list(op.sample_inputs(DEVICE_TYPE, dtype, requires_grad=False))
        except Exception:
            self.skipTest(f"Failed to get sample inputs for {op.name}")
        if not samples:
            self.skipTest(f"No sample inputs for {op.name}")

        sample = samples[0]
        args = (sample.input,) + tuple(sample.args)

        # create Replicated DTensors
        try:
            dtensor_args, dtensor_kwargs = pytree.tree_map_only(
                torch.Tensor,
                lambda t: distribute_tensor(t, mesh, (Replicate(),)),
                (args, sample.kwargs),
            )
        except Exception:
            self.skipTest(f"Failed to create replicate DTensors for {op.name}")

        # extract aten op/args/kwargs
        aten_op, aten_args, aten_kwargs = self._extract_aten_op_and_args(
            op.op, dtensor_args, dtensor_kwargs
        )

        single_dim_strats = sharding_prop.op_single_dim_strategy_funcs
        if aten_op not in single_dim_strats:
            self.skipTest(f"No single-dim strategy for {op.name}: {aten_op}")

        # extract tensor_meta, full tensors
        all_tensor_meta = []

        def _collect_tensor_meta(dt):
            # [FUNCTION DOC] HELPER: Appends the DTensor's tensor meta to the outer list and
            # returns it.
            meta = dt._spec.tensor_meta
            all_tensor_meta.append(meta)
            return meta

        args_meta, kwargs_meta = pytree.tree_map_only(
            DTensor, _collect_tensor_meta, (aten_args, aten_kwargs)
        )
        full_args, full_kwargs = pytree.tree_map_only(
            torch.Tensor, lambda t: t.full_tensor(), (aten_args, aten_kwargs)
        )

        # enumerate strategies, replace placeholders with Shard
        strategies = pytree.tree_map_only(
            _ShardingPlaceholder,
            lambda s: Shard(s.dim),
            single_dim_strats[aten_op](aten_op, args_meta, kwargs_meta),
        )
        n_inputs = len(all_tensor_meta)
        for strategy in strategies:
            input_placements = strategy[-n_inputs:]
            output_placements = strategy[:-n_inputs]

            # skip strategies with invalid shards
            def is_invalid_shard(meta, p):
                # [FUNCTION DOC] HELPER: True if placement p cannot be used to shard a tensor
                # of this meta: not a shard, zero dims, dim out of range, empty dim, or dim size
                # not divisible by the world size.
                ndim = len(meta.shape)
                if (
                    not isinstance(p, Shard)
                    or ndim == 0
                    or p.dim >= ndim
                    or meta.shape[p.dim] == 0
                    or meta.shape[p.dim] % self.world_size != 0
                ):
                    return True
                return False

            if any(
                is_invalid_shard(t, p)
                for t, p in zip(all_tensor_meta, input_placements)
            ):
                continue

            self.assertTrue(
                validate_sharding_rule_sample(
                    aten_op,
                    full_args,
                    full_kwargs,
                    input_placements,
                    tuple(output_placements),
                    mesh,
                ),
                lambda msg: f"{msg}\n{op.name}: forward {input_placements} -> {tuple(output_placements)} failed",
            )

            bwd = validate_sharding_rule_sample_backward(
                aten_op,
                full_args,
                full_kwargs,
                input_placements,
                mesh,
            )
            if bwd is not None:
                self.assertTrue(
                    bwd,
                    lambda msg: f"{msg}\n{op.name}: backward {input_placements} failed",
                )


class TestCompiledDTensorOps(TestDTensorOps):
    # [CLASS DOC] Compile smoke tests: each op is compiled with fullgraph on the aot_eager
    # backend on DTensor inputs under a fake process group; outputs are not compared.
    """
    Test DTensor ops compile successfully with aot_eager backend.
    Uses fake PG for speed - focuses on compilation, not output correctness.
    """

    _op_db = repurpose_ops(op_db, "TestDTensorOps", "TestCompiledDTensorOps")

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Runs base setUp and initializes a fake process group at rank 0.
        super().setUp()
        torch.distributed.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs base tearDown, clears the sharding propagation cache, resets
        # dynamo, and destroys the process group, ignoring an AssertionError.
        super().tearDown()
        from torch.distributed.tensor.debug import _clear_sharding_prop_cache

        _clear_sharding_prop_cache()
        torch._dynamo.reset()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        # [FUNCTION DOC] HELPER: Does nothing, so output comparison is skipped.
        # Skip output comparison - we only care that compilation succeeds
        pass

    def run_dtensor_crossref(self, func, args, kwargs):
        # [FUNCTION DOC] Override: for each fully converted DTensor argument combination, resets
        # dynamo and compiles func with fullgraph on aot_eager, then calls it. A linear algebra
        # error (or the lu_factor runtime error) from the compiled call is tolerated only if the
        # eager call raises the same type and message; anything else is re-raised. Returns
        # nothing.
        """
        Override to compile with aot_eager and verify compilation succeeds.
        Does not check output correctness.
        """
        to_dtensor = DTensorConverter(self.mesh, args, kwargs)

        def is_accepted_linalg_error(exc):
            # [FUNCTION DOC] HELPER: True if the exception is a linalg error or a runtime error
            # whose message starts with the lu_factor prefix.
            return isinstance(exc, torch._C._LinAlgError) or (
                # lu_factor intentionally raises a plain RuntimeError from
                # _linalg_check_errors for singular factors.
                isinstance(exc, RuntimeError)
                and str(exc).startswith("torch.linalg.lu_factor:")
            )

        for dtensor_args, dtensor_kwargs in to_dtensor:
            if not to_dtensor.successful():
                continue

            torch._dynamo.reset()

            @torch.compile(backend="aot_eager", fullgraph=True)
            def compiled_func(*a, **kw):
                # [FUNCTION DOC] HELPER: Calls the op under test with the given arguments.
                return func(*a, **kw)

            try:
                compiled_func(*dtensor_args, **dtensor_kwargs)
            except (torch._C._LinAlgError, RuntimeError) as compiled_exc:
                if not is_accepted_linalg_error(compiled_exc):
                    raise
                try:
                    func(*dtensor_args, **dtensor_kwargs)
                except (torch._C._LinAlgError, RuntimeError) as eager_exc:
                    # Some valid OpInfo samples can raise for particular DTensor
                    # placements. This test is a compile smoke, so accept
                    # runtime errors that eager DTensor raises in the same way.
                    if (
                        is_accepted_linalg_error(eager_exc)
                        and type(compiled_exc) is type(eager_exc)
                        and str(compiled_exc) == str(eager_exc)
                    ):
                        continue
                raise

    @suppress_warnings
    @ops(_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        (
            dtensor_fails
            | dtensor_fails_no_strategy
            | dtensor_multi_threaded_fails
            | dtensor_compiled_fails
        )
        - dtensor_numeric_only_fails,
    )
    def test_compiled_dtensor_op_db(self, dtype, op):
        # [FUNCTION DOC] Per-op test for float: runs the opinfo test without grad, skipping the
        # union of eager, no-strategy, threaded and compile failure sets minus the ops whose
        # failure is numeric only.
        self.run_opinfo_test(dtype, op, requires_grad=False)


# only instantiate tests for DEVICE_TYPE alone (i.e. either CPU or GPU)
instantiate_device_type_tests(
    TestMultiThreadedDTensorOps, globals(), only_for=(DEVICE_TYPE,)
)

instantiate_device_type_tests(TestLocalDTensorOps, globals(), only_for=(DEVICE_TYPE,))

instantiate_device_type_tests(
    TestUnbackedDTensorOps, globals(), only_for=(DEVICE_TYPE,)
)

instantiate_device_type_tests(
    TestSingleDimStrategies, globals(), only_for=(DEVICE_TYPE,)
)

instantiate_device_type_tests(
    TestCompiledDTensorOps, globals(), only_for=(DEVICE_TYPE,)
)

if __name__ == "__main__":
    run_tests()
