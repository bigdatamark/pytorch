# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-process tests for DTensor view-like operators (view, reshape, flatten,
# unflatten, permute, transpose, expand, tile, ...) and their sharding propagation in
# torch.distributed.tensor._ops._view_ops. Covers the dimension-map rules built by view_groups
# and dim_maps, propagate_shape_and_sharding, _StridedShard outputs of flatten, and the
# behavior of downstream ops (sum, softmax, layer_norm, transpose, select, unbind, stack,
# slice, ...) on _StridedShard inputs. Main class: TestViewOps on a 6 rank world, plus
# TestViewOpsWithLocalTensor built via create_local_tensor_test_class. Import has no side
# effects beyond imports and that class creation; run_tests is called when run as a script.
# KNOWN ISSUE: the skipped_tests list for the local tensor class names test_dtensor_flatten_1d
# and test_dtensor_flatten_2d, which do not exist in TestViewOps, so those entries are stale.

import contextlib
import itertools
import math
from typing import cast

import torch
import torch.distributed as dist
from torch import rand, randn, Tensor
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._ops._view_ops import (
    _ViewShardingPropagator,
    Broadcast,
    dim_maps,
    Flatten,
    InputDim,
    propagate_shape_and_sharding,
    Repeat,
    Singleton,
    Split,
    view_groups,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import _StridedShard, Placement
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorContinuousTestBase,
    LocalDTensorContinuousTestBase,
)
from torch.utils import _pytree as pytree


class TestViewOps(DTensorContinuousTestBase):
    # [CLASS DOC] Test case for DTensor view ops, run with a world size of 6 ranks. Tests
    # check dim-map rules against expected rule tuples, run ops over many input placements
    # asserting zero collective communication and equality with single-device results, and
    # check that flatten/unflatten produce the right Shard or _StridedShard placements and
    # local tensors for uneven and even sizes. Some tests set self.device_mesh themselves.
    world_size = 6

    @staticmethod
    def _get_all_factorizations(n):
        # [FUNCTION DOC] HELPER: Returns a list of every factorization of n into factors
        # greater than 1, kept as non-decreasing tuples, with at least two factors each
        # (so n itself alone and the empty case for n equal to 1 are excluded). Uses a
        # recursive search; no side effects.
        """Return all ways to factor n into a sorted tuple of integers > 1.

        Each factorization has length >= 2 (non-trivial unflatten).
        Only sorted (non-decreasing) factorizations are returned since permutations
        exercise the same split_factor logic with different constants.

        Examples:
            12 -> [(2, 6), (3, 4), (2, 2, 3)]
            36 -> [(2, 18), (3, 12), (4, 9), (6, 6), (2, 2, 9), (2, 3, 6), (3, 3, 4), (2, 2, 3, 3)]
        """

        def get_sorted_factorizations(remaining, min_factor=2):
            # [FUNCTION DOC] HELPER: Recursively lists tuples of non-decreasing factors (each at
            # least min_factor) whose product is remaining; returns a list holding the empty
            # tuple when remaining is 1.
            if remaining == 1:
                return [()]
            result = []
            for f in range(min_factor, remaining + 1):
                if remaining % f == 0:
                    for sub in get_sorted_factorizations(remaining // f, f):
                        result.append((f,) + sub)
            return result

        return [f for f in get_sorted_factorizations(n) if len(f) >= 2]

    def test_view_groups(self):
        # [FUNCTION DOC] Checks view_groups on pairs of input and output shapes against the
        # expected tuples of dimension rules: Split of an InputDim or of a Flatten for
        # unflatten and mixed cases, Flatten for merges, Singleton for size-1 output dims,
        # skipped size-1 input dims, and a -1 in the output shape resolved to a plain
        # InputDim mapping. Pure shape logic; no distributed state.
        self.assertEqual(
            view_groups([2, 3], [3, 2]),
            (
                Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 0),
                Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 1),
            ),
        )
        self.assertEqual(
            view_groups([3, 4, 5], [12, 5]),
            (Flatten((InputDim(0), InputDim(1))), InputDim(2)),
        )
        self.assertEqual(
            view_groups([2, 3, 4, 5, 7], [12, 70]),
            (
                Split(
                    Flatten(
                        (
                            InputDim(0),
                            InputDim(1),
                            InputDim(2),
                            InputDim(3),
                            InputDim(4),
                        )
                    ),
                    (12, 70),
                    0,
                ),
                Split(
                    Flatten(
                        (
                            InputDim(0),
                            InputDim(1),
                            InputDim(2),
                            InputDim(3),
                            InputDim(4),
                        )
                    ),
                    (12, 70),
                    1,
                ),
            ),
        )
        self.assertEqual(
            view_groups([2, 3, 4, 5, 7], [3, 8, 7, 5]),
            (
                Split(Flatten((InputDim(0), InputDim(1), InputDim(2))), (3, 8), 0),
                Split(Flatten((InputDim(0), InputDim(1), InputDim(2))), (3, 8), 1),
                Split(Flatten((InputDim(3), InputDim(4))), (7, 5), 0),
                Split(Flatten((InputDim(3), InputDim(4))), (7, 5), 1),
            ),
        )
        self.assertEqual(
            view_groups([3, 4, 8, 3], [12, 4, 2, 3]),
            (
                Flatten((InputDim(0), InputDim(1))),
                Split(InputDim(2), (4, 2), 0),
                Split(InputDim(2), (4, 2), 1),
                InputDim(3),
            ),
        )
        self.assertEqual(
            view_groups([3, 24], [1, 3, 2, 4, 1, 3, 1]),
            (
                Singleton(),
                InputDim(0),
                Split(InputDim(1), (2, 4, 3), 0),
                Split(InputDim(1), (2, 4, 3), 1),
                Singleton(),
                Split(InputDim(1), (2, 4, 3), 2),
                Singleton(),
            ),
        )
        self.assertEqual(
            view_groups([1, 1, 3, 2, 1, 1], [6, 1, 1, 1]),
            (
                Flatten((InputDim(2), InputDim(3))),
                InputDim(4),
                InputDim(5),
                Singleton(),
            ),
        )
        self.assertEqual(
            view_groups([1, 1, 12, 1, 1, 1, 2, 5, 1], [3, 4, 1, 10]),
            (
                Split(InputDim(2), (3, 4), 0),
                Split(InputDim(2), (3, 4), 1),
                InputDim(3),
                Flatten((InputDim(6), InputDim(7))),
            ),
        )
        self.assertEqual(
            view_groups([2, 3, 4], [2, -1, 4]),
            (InputDim(0), InputDim(1), InputDim(2)),
        )

    def call_dt_test(self, op, args, kwargs, device_mesh: DeviceMesh):
        # [FUNCTION DOC] HELPER: Runs op on DTensors for many input placements and checks the
        # result. The mesh must have dim names outer and inner. Computes the dim-map rules and
        # the single-device output, then excludes dims that cannot be sharded (the repeated
        # input dim, non-first flattened dims, and the unbind dim). Tries Replicate plus Shard
        # of each eligible dim on every mesh dim, and for Tensor.view also a _StridedShard on
        # the outer mesh dim with Shard on the inner (built by distributing over inner then
        # outer and clearing use_strided_shard_as_shard_order). Asserts zero collective counts
        # and, on rank 0, that the full output tensor equals the single-device output.
        dim_map = dim_maps[op]
        rules = dim_map(*args, **kwargs)
        outputs = op(*args, **kwargs)
        flat_args = pytree.arg_tree_leaves(*args)
        in_shape = flat_args[0].shape

        no_shard_dims = set()
        for rule in rules:
            if isinstance(rule, Repeat):
                if isinstance(rule.input_dim, InputDim):
                    no_shard_dims.add(rule.input_dim.input_dim)
            elif isinstance(rule, Flatten):
                for dim in rule.input_dims[1:]:
                    if isinstance(dim, InputDim):
                        no_shard_dims.add(dim.input_dim)
            elif isinstance(rule, Split):
                if isinstance(rule.input_dim, Flatten):
                    for dim in rule.input_dim.input_dims[1:]:
                        if isinstance(dim, InputDim):
                            no_shard_dims.add(dim.input_dim)

        if op == torch.unbind:
            no_shard_dims.add(kwargs.get("dim", 0))

        sharding_choices = cast(list[Placement], [Replicate()]) + [
            Shard(i) for i, s in enumerate(in_shape) if s > 1 and i not in no_shard_dims
        ]

        all_sharding_choices = itertools.product(
            *(device_mesh.ndim * [sharding_choices])
        )

        outer_mesh = device_mesh["outer"]
        inner_mesh = device_mesh["inner"]
        inner_mesh_size = inner_mesh.size()
        strided_sharding_choices = [
            (_StridedShard(i, split_factor=inner_mesh_size), Shard(i))
            for i, s in enumerate(in_shape)
            if s > 1 and i not in no_shard_dims
        ]

        for in_shard in itertools.chain(all_sharding_choices, strided_sharding_choices):
            if isinstance(in_shard[0], _StridedShard):
                if op != Tensor.view:
                    continue
                # cannot produce DTensor using ``distribute_tensor()``
                # with ``_StridedShard``. Need to distribute the input
                # over inner mesh dim first, then distribute the
                # _local_tensor over the outer mesh dim.
                in_dt = distribute_tensor(args[0], inner_mesh, (in_shard[1],))
                in_dt = distribute_tensor(
                    in_dt._local_tensor, outer_mesh, (Shard(in_shard[0].dim),)
                )
                in_dt = DTensor.from_local(
                    in_dt._local_tensor,
                    device_mesh,
                    in_shard,
                )
                # These are literal _StridedShard placements (simulating
                # flatten output), not shard-order encodings.  Override the
                # auto-detection which may incorrectly set the flag to True
                # when split_factor happens to match a mesh dim size.
                in_dt._spec.use_strided_shard_as_shard_order = False
            else:
                in_dt = distribute_tensor(args[0], device_mesh, in_shard)

            comm_mode = CommDebugMode()
            with comm_mode:
                out_dt = op(in_dt, *args[1:], **kwargs)

            self.assertEqual(
                comm_mode.get_total_counts(), 0, "Expected no redistribution."
            )

            full_out = out_dt.full_tensor()

            if dist.get_rank() == 0:
                self.assertEqual(outputs, full_out)

    def dimmap_test(self, op, args, expected_rule_output):
        # [FUNCTION DOC] HELPER: Asserts that the dim-map rule for op, applied to args, equals
        # expected_rule_output, then runs call_dt_test with empty kwargs on self.device_mesh
        # (which the caller must have set) to verify sharded execution.
        rules = dim_maps[op](*args)
        self.assertEqual(rules, expected_rule_output)
        self.call_dt_test(op, args, {}, self.device_mesh)

    def test_illegal_views(self):
        # [FUNCTION DOC] On a 1D mesh of 6 ranks, checks which flattening views of a sharded
        # DTensor are accepted. Shard(0) of (6, 256) and Shard(1) of (6, 256) flatten fine;
        # Shard(0) of an uneven (8, 256) and a Shard(1) unflatten of (8, 256) to (8, 2, -1)
        # must raise RuntimeError containing "Sharding propagation failed".
        device_mesh = self.build_device_mesh()
        # 1D mesh [6] (see above)
        tensor = torch.randn((6, 256))
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        shard = dtensor.redistribute(device_mesh=device_mesh, placements=[Shard(dim=0)])
        # view should be legal, since sharding is even and flatten includes only one sharded dim
        shard.view(-1)

        shard = dtensor.redistribute(device_mesh=device_mesh, placements=[Shard(dim=1)])
        # Shard(1) on the last flattened dim with uneven sharding is now allowed
        # (produces _StridedShard output)
        shard.view(-1)

        # 8 is the uneven case since mesh dim is 6
        tensor = torch.randn((8, 256))
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        shard = dtensor.redistribute(device_mesh=device_mesh, placements=[Shard(dim=0)])
        with self.assertRaisesRegex(RuntimeError, "Sharding propagation failed"):
            shard.view(-1)

        # assuming world size is 4+, tensor is shardable on dim 1 with size 256
        # but not viewable when the resulting dim 1 has size 2
        tensor = torch.randn((8, 256))
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        shard = dtensor.redistribute(device_mesh=device_mesh, placements=[Shard(dim=1)])
        with self.assertRaisesRegex(RuntimeError, "Sharding propagation failed"):
            shard.view(8, 2, -1)

    def test_split_flatten_non_first_shard_falls_back_to_replicate(self):
        # [FUNCTION DOC] Calls propagate_shape_and_sharding directly for Shard(1) on shape
        # (32, 4) viewed as (4, 32) with a mesh of size 4 in strict view mode. Asserts both
        # the required input placement and the output placement are Replicate, i.e. sharding
        # a non-first flattened dim under Split(Flatten) falls back to replication.
        rule = view_groups((32, 4), (4, 32))
        input_tgt, output = propagate_shape_and_sharding(
            [Shard(1)],
            (32, 4),
            rule,
            (4,),
            strict_view=True,
        )
        self.assertEqual(input_tgt, [Replicate()])
        self.assertEqual(output, [Replicate()])

    def test_double_shard_split_validation(self):
        # [FUNCTION DOC] Checks divisibility validation for two Shard(0) placements through a
        # Split rule on mesh sizes (2, 3). Viewing (24,) as (6, 4) succeeds and keeps
        # Shard(0), Shard(0) for both input and output; viewing (12,) as (3, 4) in strict view
        # mode raises RuntimeError saying the size is not evenly divisible by the mesh
        # dimension.
        """[Shard(0), Shard(0)] through Split correctly validates divisibility."""
        # Compatible: reshape (24,)→(6,4) with [Shard(0), Shard(0)] on mesh (2,3)
        # submesh_size = 2*3 = 6, split_id=0 out_size=6, 6%6==0 → passes
        result = propagate_shape_and_sharding(
            [Shard(0), Shard(0)],
            (24,),
            dim_maps[torch.Tensor.view](torch.empty(24), [6, 4]),
            (2, 3),
        )
        self.assertEqual(len(result), 2)
        input_tgt, output = result
        self.assertEqual(list(input_tgt), [Shard(0), Shard(0)])
        self.assertEqual(list(output), [Shard(0), Shard(0)])

        # Incompatible: reshape (12,)→(3,4) with [Shard(0), Shard(0)] on mesh (2,3)
        # out_size=3 is not divisible by mesh dim 0 size=2 → error
        with self.assertRaisesRegex(
            RuntimeError, "not evenly divisible by mesh dimension"
        ):
            propagate_shape_and_sharding(
                [Shard(0), Shard(0)],
                (12,),
                dim_maps[torch.Tensor.view](torch.empty(12), [3, 4]),
                (2, 3),
                strict_view=True,
            )

    def test_view_ops(self):
        # [FUNCTION DOC] Builds a 2D mesh (world_size/2 by 2) named outer and inner, stored
        # on self.device_mesh, then runs dimmap_test for many ops and argument forms:
        # atleast_1d/2d/3d, broadcast_to (including -1 and the invalid-size error, which must
        # raise AssertionError), Tensor.expand, flatten, movedim, permute, ravel, Tensor.repeat,
        # reshape, tile, transpose, unsqueeze and Tensor.view. Each case verifies the expected
        # rule tuple and zero-communication sharded results.
        mesh_shape = (dist.get_world_size() // 2, 2)
        self.device_mesh = init_device_mesh(
            self.device_type, mesh_shape=mesh_shape, mesh_dim_names=("outer", "inner")
        )
        self.dimmap_test(torch.atleast_1d, (randn(()),), (Singleton(),))
        self.dimmap_test(torch.atleast_1d, (randn(24),), (InputDim(0),))
        self.dimmap_test(torch.atleast_1d, (randn(24, 36),), (InputDim(0), InputDim(1)))

        self.dimmap_test(torch.atleast_2d, (randn(()),), (Singleton(), Singleton()))
        self.dimmap_test(torch.atleast_2d, (randn(24),), (Singleton(), InputDim(0)))
        self.dimmap_test(torch.atleast_2d, (randn(24, 36),), (InputDim(0), InputDim(1)))
        self.dimmap_test(
            torch.atleast_2d,
            (randn(24, 36, 48),),
            (InputDim(0), InputDim(1), InputDim(2)),
        )

        self.dimmap_test(
            torch.atleast_3d,
            (randn(()),),
            (Singleton(), Singleton(), Singleton()),
        )
        self.dimmap_test(
            torch.atleast_3d,
            (randn(24),),
            (Singleton(), InputDim(0), Singleton()),
        )
        self.dimmap_test(
            torch.atleast_3d,
            (randn(24, 36),),
            (InputDim(0), InputDim(1), Singleton()),
        )
        self.dimmap_test(
            torch.atleast_3d,
            (randn(24, 36, 42),),
            (InputDim(0), InputDim(1), InputDim(2)),
        )
        self.dimmap_test(
            torch.atleast_3d,
            (randn(24, 36, 42, 24),),
            (InputDim(0), InputDim(1), InputDim(2), InputDim(3)),
        )

        with self.assertRaises(AssertionError):
            dim_maps[torch.broadcast_to](randn(24, 36), (1, 2, 4))

        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 36), (1, 24, 36)),
            (Singleton(), InputDim(0), InputDim(1)),
        )
        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 36), (42, 24, 36)),
            (Broadcast(Singleton(), 42), InputDim(0), InputDim(1)),
        )
        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 1, 36), (12, 24, 24, 36)),
            (
                Broadcast(Singleton(), 12),
                InputDim(0),
                Broadcast(InputDim(1), 24),
                InputDim(2),
            ),
        )
        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 36), (-1, 36)),
            (InputDim(0), InputDim(1)),
        )
        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 1, 36), (-1, 1, 36)),
            (InputDim(0), InputDim(1), InputDim(2)),
        )

        self.dimmap_test(
            torch.broadcast_to,
            (randn(36, 1, 24), (12, 36, 42, 24)),
            (
                Broadcast(Singleton(), 12),
                InputDim(0),
                Broadcast(InputDim(1), 42),
                InputDim(2),
            ),
        )

        self.dimmap_test(
            Tensor.expand,
            (randn(24, 1, 36, 1), 36, 24, 42, -1, 24),
            (
                Broadcast(Singleton(), 36),
                InputDim(0),
                Broadcast(InputDim(1), 42),
                InputDim(2),
                Broadcast(InputDim(3), 24),
            ),
        )

        self.dimmap_test(
            Tensor.expand,
            (randn(24, 1, 36, 1), (36, 24, 42, -1, 24)),
            (
                Broadcast(Singleton(), 36),
                InputDim(0),
                Broadcast(InputDim(1), 42),
                InputDim(2),
                Broadcast(InputDim(3), 24),
            ),
        )

        self.dimmap_test(
            torch.flatten,
            (randn(24, 36),),
            (Flatten((InputDim(0), InputDim(1))),),
        )
        self.dimmap_test(torch.flatten, (randn(42),), (InputDim(0),))
        self.dimmap_test(torch.flatten, (randn(()),), (Singleton(),))

        self.dimmap_test(
            torch.movedim,
            (randn(12, 24, 48, 96), 1, 2),
            (InputDim(0), InputDim(2), InputDim(1), InputDim(3)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(6, 12, 24), 1, 0),
            (InputDim(1), InputDim(0), InputDim(2)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(24, 12, 6), (1, 2), (0, 1)),
            (InputDim(1), InputDim(2), InputDim(0)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(24, 6, 12), (0, 2, 1), (2, 1, 0)),
            (InputDim(1), InputDim(2), InputDim(0)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(24, 12), (1, 0), (0, 1)),
            (InputDim(1), InputDim(0)),
        )

        self.dimmap_test(
            torch.movedim,
            (randn(36, 24, 12), (1, 2), (0, 1)),
            (InputDim(1), InputDim(2), InputDim(0)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(36, 24, 12), (1, 2), (-3, -2)),
            (InputDim(1), InputDim(2), InputDim(0)),
        )

        self.dimmap_test(
            torch.permute,
            (randn(24, 36, 42), (2, 0, 1)),
            (InputDim(2), InputDim(0), InputDim(1)),
        )
        self.dimmap_test(
            torch.permute,
            (randn(24, 36, 42), (-1, -3, -2)),
            (InputDim(2), InputDim(0), InputDim(1)),
        )

        self.dimmap_test(
            torch.ravel,
            (randn(24, 36),),
            (Flatten((InputDim(0), InputDim(1))),),
        )
        self.dimmap_test(torch.ravel, (randn(42),), (InputDim(0),))
        self.dimmap_test(torch.ravel, (randn(()),), (Singleton(),))

        self.dimmap_test(
            Tensor.repeat,
            (randn(24, 36), 1, 2, 1, 1, 2),
            (
                Singleton(),
                Broadcast(Singleton(), 2),
                Singleton(),
                InputDim(0),
                Repeat(InputDim(1), 2),
            ),
        )

        self.dimmap_test(
            torch.reshape,
            (randn(6, 12, 24), (72, 24)),
            (Flatten((InputDim(0), InputDim(1))), InputDim(2)),
        )

        self.dimmap_test(
            torch.tile,
            (randn(24, 36), (1, 2, 1, 1, 2)),
            (
                Singleton(),
                Broadcast(Singleton(), 2),
                Singleton(),
                InputDim(0),
                Repeat(InputDim(1), 2),
            ),
        )
        self.dimmap_test(
            torch.tile,
            (randn(42, 24, 36), (1, 3)),
            (InputDim(0), InputDim(1), Repeat(InputDim(2), 3)),
        )

        self.dimmap_test(
            torch.transpose,
            (randn(24, 60, 42, 60), 2, 0),
            (InputDim(2), InputDim(1), InputDim(0), InputDim(3)),
        )
        self.dimmap_test(
            torch.transpose,
            (randn(24, 60, 42, 60), -1, 0),
            (InputDim(3), InputDim(1), InputDim(2), InputDim(0)),
        )

        self.dimmap_test(
            torch.unsqueeze,
            (randn(42, 24, 36), 1),
            (InputDim(0), Singleton(), InputDim(1), InputDim(2)),
        )
        self.dimmap_test(
            Tensor.view,
            (randn(6, 12, 24), 72, 24),
            (Flatten((InputDim(0), InputDim(1))), InputDim(2)),
        )

        self.dimmap_test(Tensor.view, (randn(1, 1, 12), -1), (InputDim(2),))

        self.dimmap_test(
            Tensor.view,
            (randn(1, 1, 42, 24), -1),
            (Flatten((InputDim(2), InputDim(3))),),
        )

        self.dimmap_test(
            Tensor.view,
            (randn(1, 1, 42, 1, 24, 1), -1),
            (Flatten((InputDim(2), InputDim(input_dim=3), InputDim(4))),),
        )

        self.dimmap_test(
            Tensor.view,
            (randn(48, 35, 26), (24, 4, 35, 13)),
            (
                Split(
                    Flatten(input_dims=(InputDim(0), InputDim(1), InputDim(2))),
                    group_shape=(24, 4, 35, 13),
                    split_id=0,
                ),
                Split(
                    Flatten(input_dims=(InputDim(0), InputDim(1), InputDim(2))),
                    group_shape=(24, 4, 35, 13),
                    split_id=1,
                ),
                Split(
                    Flatten(input_dims=(InputDim(0), InputDim(1), InputDim(2))),
                    group_shape=(24, 4, 35, 13),
                    split_id=2,
                ),
                Split(
                    Flatten(input_dims=(InputDim(0), InputDim(1), InputDim(2))),
                    group_shape=(24, 4, 35, 13),
                    split_id=3,
                ),
            ),
        )

    # TODO: Currently functional collectives on complex numbers are not fully supported,
    # so we are having a standalone test for view_as_complex and view_as_real combined.
    # Once complex numbers are supported, we can add the following to the dim_map test.
    #
    # self.dimmap_test(
    #     torch.view_as_complex,
    #     (randn(24, 13, 2),),
    #     (
    #         InputDim(0),
    #         Flatten((InputDim(1), InputDim(2))),
    #     ),
    # )
    # self.dimmap_test(
    #     torch.view_as_real,
    #     (torch.randn(24, 13, dtype=torch.cfloat),),
    #     (
    #         InputDim(0),
    #         Split(InputDim(1), (13, 2), 0),
    #         Split(InputDim(1), (13, 2), 1),
    #     ),
    # )

    def test_complex_view_ops(self):
        # [FUNCTION DOC] Uses a 2D mesh with world_size/2 rows and 2 columns. Verifies the
        # dim-map rules of view_as_complex (InputDim then Flatten of last two dims) and
        # view_as_real (Split of the last dim by 13 and 2), then for each combination of
        # Replicate or Shard(0) per mesh dim runs view_as_complex then view_as_real on a
        # DTensor, asserting zero collectives and a full tensor equal to the local result.
        self.device_mesh = DeviceMesh(
            self.device_type, torch.arange(dist.get_world_size()).view(-1, 2)
        )
        inp = randn(24, 13, 2)
        intermediate = torch.view_as_complex(inp)
        out = torch.view_as_real(intermediate)

        # test dim_map correctness
        expected_view_as_complex_rule = (
            InputDim(0),
            Flatten((InputDim(1), InputDim(2))),
        )
        view_as_complex_rule = dim_maps[torch.view_as_complex](inp)
        self.assertEqual(view_as_complex_rule, expected_view_as_complex_rule)
        expected_view_as_real_rule = (
            InputDim(0),
            Split(InputDim(1), (13, 2), 0),
            Split(InputDim(1), (13, 2), 1),
        )
        view_as_real_rule = dim_maps[torch.view_as_real](intermediate)
        self.assertEqual(view_as_real_rule, expected_view_as_real_rule)

        # test sharded computation correctness
        # NOTE: For the input to torch.view_as_complex, sharding
        #       on the last two dimensions is not supported.
        sharding_choices: list[Placement] = [Replicate(), Shard(0)]
        all_sharding_choices = itertools.product(
            *(self.device_mesh.ndim * [sharding_choices])
        )

        for inp_shard in all_sharding_choices:
            inp_dt = distribute_tensor(inp, self.device_mesh, inp_shard)

            comm_mode = CommDebugMode()
            with comm_mode:
                intermediate_dt = torch.view_as_complex(inp_dt)
                out_dt = torch.view_as_real(intermediate_dt)

            self.assertEqual(
                comm_mode.get_total_counts(), 0, "Expected no redistribution."
            )
            self.assertEqual(out, out_dt.full_tensor())

    def test_view_as_complex_no_pmax_pmin(self):
        # [FUNCTION DOC] On a 1D mesh over all ranks, applies view_as_complex to an (8, 2)
        # DTensor with Partial sum, which stays Partial with reduce op sum, and with Partial
        # max, which must be redistributed so the result placement is Replicate.
        """
        view_as_complex converts real tensors to complex. Complex numbers don't
        have a total ordering, so P(max) and P(min) placements are invalid.
        This test verifies that these placements are not propagated.
        """
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        inp = torch.randn(8, 2)

        # Create input with P(sum) - this should work
        inp_dt_psum = DTensor.from_local(
            inp.clone(), device_mesh, [Partial("sum")], run_check=False
        )
        result_psum = torch.view_as_complex(inp_dt_psum)
        # P(sum) should propagate since sum is valid for complex
        self.assertIsInstance(result_psum.placements[0], Partial)
        self.assertEqual(result_psum.placements[0].reduce_op, "sum")

        # Create input with P(max) - should redistribute to Replicate
        inp_dt_pmax = DTensor.from_local(
            inp.clone(), device_mesh, [Partial("max")], run_check=False
        )
        result_pmax = torch.view_as_complex(inp_dt_pmax)
        # P(max) should NOT propagate - output should be Replicate
        self.assertIsInstance(result_pmax.placements[0], Replicate)

    def test_dtensor_view_op_uneven(self):
        # [FUNCTION DOC] For first dim sizes 1 and world_size+1 (uneven shards), shards a
        # (n, 2, 2, 2) tensor on dim 0 and applies several views that keep dim 0 unchanged,
        # including viewing as its own shape. Under CommDebugMode asserts no communication
        # happened and the local tensor of each view shares the data pointer of the original.
        """
        When the sharded dimension is unchanged, the view op should not trigger any communication.
        And the behavior should be the same as operating under single-device.

        Test two uneven cases for view op:
            1) the sharded tensor dim is 1 so that only the first rank has an non-empty shard.
            2) the sharded tensor dim is uneven such that some ranks have full shards,
                smaller non-empty shards, and empty shards.
        """
        dim0_sizes = [1, self.world_size + 1]
        for dim0_size in dim0_sizes:
            p = torch.randn(dim0_size, 2, 2, 2)
            mesh = init_device_mesh(self.device_type, (self.world_size,))
            dtensor = distribute_tensor(p, mesh, [Shard(0)])

            with CommDebugMode() as comm_mode:
                view = dtensor.view(dim0_size, 2, 4)
                self.assertEqual(len(comm_mode.get_comm_counts()), 0)
                # when no communication happens, the data pointer should be the same.
                self.assertEqual(
                    view.to_local().data_ptr(), dtensor.to_local().data_ptr()
                )

                view = dtensor.view(dim0_size, 4, 2)
                self.assertEqual(
                    view.to_local().data_ptr(), dtensor.to_local().data_ptr()
                )
                self.assertEqual(len(comm_mode.get_comm_counts()), 0)

                view = dtensor.view(dim0_size, 8)
                self.assertEqual(
                    view.to_local().data_ptr(), dtensor.to_local().data_ptr()
                )
                self.assertEqual(len(comm_mode.get_comm_counts()), 0)

                view = dtensor.view(dtensor.shape)
                self.assertEqual(
                    view.to_local().data_ptr(), dtensor.to_local().data_ptr()
                )
                self.assertEqual(len(comm_mode.get_comm_counts()), 0)

    def _get_viewed_tensor_dims(self, tensor_dims, flatten_start, flatten_end):
        # [FUNCTION DOC] HELPER: Returns a tuple of dims where the entries from flatten_start up
        # to but excluding flatten_end are replaced by their product (omitted if that range is
        # empty), keeping leading and trailing dims. Accepts a tuple or list.
        if isinstance(tensor_dims, tuple):
            tensor_dims = list(tensor_dims)
        flatten_dims = tensor_dims[flatten_start:flatten_end]
        if len(flatten_dims) > 0:
            flatten_dim = math.prod(flatten_dims)
        else:
            flatten_dim = None
        leading_dims = tensor_dims[:flatten_start]
        trailing_dims = tensor_dims[flatten_end:]
        view_shapes = []
        if len(leading_dims) > 0:
            view_shapes.extend(leading_dims)
        if flatten_dim is not None:
            view_shapes.append(flatten_dim)
        if len(trailing_dims) > 0:
            view_shapes.extend(trailing_dims)
        return tuple(view_shapes)

    def _split_flatten_replicate_fallback_mesh_dims(self, rules, placements):
        # [FUNCTION DOC] HELPER: Returns the list of mesh dim indices whose Shard placement is
        # expected to fall back to Replicate. For each Shard placement, finds the first
        # Split rule whose input is a Flatten containing the sharded tensor dim, and records the
        # mesh dim when that tensor dim is not the first flattened dim. Raises AssertionError
        # if a Flatten holds anything other than InputDim entries.
        fallback_mesh_dims = []
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Shard):
                continue

            for rule in rules:
                if not (
                    isinstance(rule, Split) and isinstance(rule.input_dim, Flatten)
                ):
                    continue

                input_dims = rule.input_dim.input_dims
                if not all(isinstance(input_dim, InputDim) for input_dim in input_dims):
                    raise AssertionError(f"Expected InputDim list, got {input_dims}")
                flattened_dims = cast(list[InputDim], input_dims)
                if any(
                    input_dim.input_dim == placement.dim for input_dim in flattened_dims
                ):
                    first_dim = flattened_dims[0].input_dim
                    if placement.dim != first_dim:
                        fallback_mesh_dims.append(mesh_dim)
                    break
        return fallback_mesh_dims

    def _test_dtensor_flatten_split_case(self, in_shape, out_shape, placements, mesh):
        # [FUNCTION DOC] HELPER: Distributes an arange tensor of in_shape with the given
        # placements and views it as out_shape under CommDebugMode. A RuntimeError is accepted
        # only if its message says "is not supported yet" or "is not evenly divisible by mesh
        # dimension", then the case ends. Otherwise checks the full tensor equals the reference
        # view; if fallback mesh dims were predicted, expects some communication and Replicate
        # on those mesh dims, else expects zero communication.
        """Test a single Split(Flatten) view case.

        Representable cases must produce zero communication and correct output. Cases
        that require sharding a non-first flattened input dim fall back to Replicate
        with communication, because the plain Shard is not exactly representable.
        Other unrepresentable cases must raise RuntimeError with a specific message.
        """
        rules = view_groups(list(in_shape), list(out_shape))
        fallback_mesh_dims = self._split_flatten_replicate_fallback_mesh_dims(
            rules, placements
        )
        nelem = math.prod(in_shape)
        global_tensor = torch.arange(nelem).view(in_shape)
        in_dt = distribute_tensor(global_tensor, mesh, placements, src_data_rank=None)
        comm_mode = CommDebugMode()
        try:
            with comm_mode:
                out_dt = in_dt.view(list(out_shape))
        except RuntimeError as e:
            self.assertRegex(
                str(e),
                r"is not supported yet"
                r"|is not evenly divisible by mesh dimension",
            )
            return
        expected = global_tensor.view(list(out_shape))
        self.assertEqual(out_dt.full_tensor(), expected)
        if fallback_mesh_dims:
            self.assertGreater(comm_mode.get_total_counts(), 0)
            for mesh_dim in fallback_mesh_dims:
                self.assertIsInstance(out_dt.placements[mesh_dim], Replicate)
        else:
            self.assertEqual(comm_mode.get_total_counts(), 0)

    def test_dtensor_flatten_split_multi_mesh(self):
        # [FUNCTION DOC] Sweeps 1D (world_size) and 2D (3, 2) meshes, each shard mesh dim,
        # 2D input shapes with sizes around twice the mesh size, and every factorization of
        # the element count as an output shape whose view rules include a Split of a Flatten.
        # For each shard dim divisible by the mesh size, runs _test_dtensor_flatten_split_case
        # in a subTest with Shard on that mesh dim and Replicate elsewhere.
        """Test views producing Split(Flatten) rules.

        Complements test_dtensor_flatten_multi_mesh (pure Flatten rules) and
        test_dtensor_unflatten_multi_mesh (pure Split(InputDim) rules) by
        covering the hybrid case: an input dim crosses two output groups,
        so view_groups produces Split(Flatten(...)) rules.

        Uses {2*M-1, 2*M, 2*M+1} dim values (M = mesh size for the shard
        mesh dim) to cover even/uneven divisibility, following the same
        pattern as _run_flatten_single_shard.
        """
        for mesh_shape in [(self.world_size,), (3, 2)]:
            if self.world_size < math.prod(mesh_shape):
                continue
            mesh = init_device_mesh(self.device_type, mesh_shape)
            for shard_mesh_dim in range(mesh.ndim):
                M = mesh.size(shard_mesh_dim)
                dim_vals = [2 * M - 1, 2 * M, 2 * M + 1]
                for a, b in itertools.product(dim_vals, repeat=2):
                    in_shape = (a, b)
                    total = a * b
                    all_factors = self._get_all_factorizations(total)
                    for out_shape in all_factors:
                        if in_shape == out_shape:
                            continue
                        rules = view_groups(list(in_shape), list(out_shape))
                        if not any(
                            isinstance(r, Split) and isinstance(r.input_dim, Flatten)
                            for r in rules
                        ):
                            continue
                        for shard_dim in range(len(in_shape)):
                            if in_shape[shard_dim] % M != 0:
                                continue
                            placements = tuple(
                                Shard(shard_dim) if i == shard_mesh_dim else Replicate()
                                for i in range(mesh.ndim)
                            )
                            with self.subTest(
                                in_shape=in_shape,
                                out_shape=out_shape,
                                shard=shard_dim,
                                mesh_dim=shard_mesh_dim,
                                mesh_shape=mesh_shape,
                            ):
                                self._test_dtensor_flatten_split_case(
                                    in_shape,
                                    out_shape,
                                    placements,
                                    mesh,
                                )

    def test_dtensor_flatten_multi_mesh(self):
        # [FUNCTION DOC] Drives flatten tests over a 1D mesh (patterns S, R) and a 2D (3, 2)
        # mesh (patterns SR, RS, SS, RR), for tensor ranks 2 to 4 and every flatten range of
        # at least two dims. One shard mesh dim goes to _run_flatten_single_shard, two to
        # _run_flatten_ss, and all-replicate patterns run _test_dtensor_flatten_replicate over
        # dim size combinations around twice the first mesh size. Skips meshes larger than the
        # world size.
        """Test flatten operations across 1D and 2D meshes with all placement patterns.

        Iterates over 1D and 2D mesh configurations to test single-shard (S, SR, RS),
        multi-shard (SS), and replicate (R, RR) patterns with even/uneven tensor dim sizes.
        """
        cases = [
            ((6,), [("S",), ("R",)]),
            ((3, 2), [("S", "R"), ("R", "S"), ("S", "S"), ("R", "R")]),
        ]
        for mesh_shape, patterns in cases:
            if self.world_size < math.prod(mesh_shape):
                continue
            mesh = init_device_mesh(self.device_type, mesh_shape)
            mesh_ndim = len(mesh_shape)
            for pattern in patterns:
                shard_mesh_dims = [i for i, p in enumerate(pattern) if p == "S"]
                num_shard = len(shard_mesh_dims)
                for tensor_ndim in [2, 3, 4]:
                    for flatten_start in range(tensor_ndim):
                        for flatten_end in range(flatten_start + 2, tensor_ndim + 1):
                            if num_shard == 1:
                                self._run_flatten_single_shard(
                                    mesh,
                                    mesh_ndim,
                                    shard_mesh_dims[0],
                                    tensor_ndim,
                                    flatten_start,
                                    flatten_end,
                                )
                            elif num_shard == 2:
                                self._run_flatten_ss(
                                    mesh,
                                    tensor_ndim,
                                    flatten_start,
                                    flatten_end,
                                )
                            else:
                                even = 2 * mesh.size(0)
                                dim_vals = [even - 1, even, even + 1]
                                all_dims = list(
                                    itertools.product(dim_vals, repeat=tensor_ndim)
                                )
                                rep_placements = tuple([Replicate()] * mesh_ndim)
                                for tensor_dims in all_dims:
                                    with self.subTest(
                                        dims=tensor_dims,
                                        flat=(flatten_start, flatten_end),
                                        mesh=mesh_shape,
                                    ):
                                        self._test_dtensor_flatten_replicate(
                                            tensor_dims,
                                            flatten_start,
                                            flatten_end,
                                            mesh,
                                            rep_placements,
                                        )

    def _run_flatten_single_shard(
        self,
        mesh,
        mesh_ndim,
        shard_mesh_dim,
        tensor_ndim,
        flatten_start,
        flatten_end,
    ):
        # [FUNCTION DOC] HELPER: For each shard dim inside the flatten range and every tensor
        # shape with dim sizes one below, equal to, and one above twice the shard mesh dim
        # size, runs _test_dtensor_flatten_single_shard in a subTest. Expects RuntimeError "is
        # not evenly divisible by mesh dimension" when the sharded dim size is not divisible
        # and that dim is not the last one of the flatten range.
        even_val = 2 * mesh.size(shard_mesh_dim)
        dim_vals = [even_val - 1, even_val, even_val + 1]
        all_dims = list(itertools.product(dim_vals, repeat=tensor_ndim))
        for shard_dim in range(flatten_start, flatten_end):
            for tensor_dims in all_dims:
                ctx = contextlib.nullcontext()
                if tensor_dims[shard_dim] % mesh.size(
                    shard_mesh_dim
                ) != 0 and shard_dim != (flatten_end - 1):
                    ctx = self.assertRaisesRegex(
                        RuntimeError,
                        "is not evenly divisible by mesh dimension",
                    )
                with (
                    self.subTest(
                        dims=tensor_dims,
                        shard=shard_dim,
                        mesh_dim=shard_mesh_dim,
                        flat=(flatten_start, flatten_end),
                    ),
                    ctx,
                ):
                    self._test_dtensor_flatten_single_shard(
                        tensor_dims,
                        flatten_start,
                        flatten_end,
                        mesh,
                        shard_dim,
                        shard_mesh_dim,
                    )

    def _run_flatten_ss(self, mesh, tensor_ndim, flatten_start, flatten_end):
        # [FUNCTION DOC] HELPER: For a 2D mesh and every ordered pair of shard dims inside the
        # flatten range, builds shapes where the sharded dims take sizes around twice their
        # mesh size and other dims equal twice the product of both mesh sizes, and runs
        # _test_dtensor_flatten_2d_ss with placements Shard of each dim. Expects a RuntimeError
        # ("is not evenly divisible by mesh dimension") when the first sharded dim is not
        # divisible by mesh dim 0, or when the second, after dividing by mesh dim 0 if both
        # shard the same dim, is not divisible by mesh dim 1 and is not the last flattened dim.
        for shard_dim0 in range(flatten_start, flatten_end):
            for shard_dim1 in range(shard_dim0, flatten_end):
                dim0_values = [
                    2 * mesh.size(0) - 1,
                    2 * mesh.size(0),
                    2 * mesh.size(0) + 1,
                ]
                dim1_values = [
                    2 * mesh.size(1) - 1,
                    2 * mesh.size(1),
                    2 * mesh.size(1) + 1,
                ]
                other_dim_value = 2 * mesh.size(0) * mesh.size(1)
                for dim0_val in dim0_values:
                    for dim1_val in dim1_values:
                        tensor_dims = [other_dim_value] * tensor_ndim
                        tensor_dims[shard_dim0] = dim0_val
                        if shard_dim0 != shard_dim1:
                            tensor_dims[shard_dim1] = dim1_val
                        tensor_dims = tuple(tensor_dims)
                        local_tensor_dims = list(tensor_dims)
                        placements = (Shard(shard_dim0), Shard(shard_dim1))
                        ctx = contextlib.nullcontext()
                        if local_tensor_dims[shard_dim0] % mesh.size(0) != 0:
                            ctx = self.assertRaisesRegex(
                                RuntimeError,
                                "is not evenly divisible by mesh dimension",
                            )
                        local_tensor_dims[shard_dim0] = local_tensor_dims[
                            shard_dim0
                        ] // mesh.size(0)
                        if local_tensor_dims[shard_dim1] % mesh.size(
                            1
                        ) != 0 and shard_dim1 != (flatten_end - 1):
                            ctx = self.assertRaisesRegex(
                                RuntimeError,
                                "is not evenly divisible by mesh dimension",
                            )
                        with (
                            self.subTest(
                                dims=tensor_dims,
                                shard0=shard_dim0,
                                shard1=shard_dim1,
                            ),
                            ctx,
                        ):
                            self._test_dtensor_flatten_2d_ss(
                                tensor_dims,
                                flatten_start,
                                flatten_end,
                                mesh,
                                placements,
                            )

    def _test_dtensor_flatten_single_shard(
        self,
        tensor_dims,
        flatten_start,
        flatten_end,
        mesh,
        shard_dim,
        shard_mesh_dim,
    ):
        # [FUNCTION DOC] HELPER: Shards an arange tensor of tensor_dims on shard_dim along
        # shard_mesh_dim (Replicate on other mesh dims), flattens the range, and asserts: the
        # output placement on that mesh dim is Shard at flatten_start if shard_dim is the first
        # flattened dim, else _StridedShard with split_factor equal to the product of the dims
        # before it in the range (must exceed 1); the local tensor matches a direct
        # distribution of the flattened reference; and no communication occurred.
        mesh_ndim = mesh.ndim
        placements = tuple(
            Shard(shard_dim) if i == shard_mesh_dim else Replicate()
            for i in range(mesh_ndim)
        )
        nelem = math.prod(tensor_dims)
        global_inps: Tensor = torch.arange(nelem).view(tensor_dims)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        viewed_tensor_dims = self._get_viewed_tensor_dims(
            tensor_dims, flatten_start, flatten_end
        )
        comm_mode = CommDebugMode()
        with comm_mode:
            inps_viewed = inps.view(viewed_tensor_dims)
        if shard_dim == flatten_start:
            expected_placement = Shard(flatten_start)
        else:
            split_factor = math.prod(tensor_dims[flatten_start:shard_dim])
            self.assertGreater(split_factor, 1)
            expected_placement = _StridedShard(
                dim=flatten_start, split_factor=split_factor
            )
        expected_placements = tuple(
            expected_placement if i == shard_mesh_dim else Replicate()
            for i in range(mesh_ndim)
        )
        expected_local_tensor = distribute_tensor(
            global_inps.view(viewed_tensor_dims),
            mesh,
            expected_placements,
            src_data_rank=None,
        )._local_tensor
        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(inps_viewed._local_tensor, expected_local_tensor)
        self.assertEqual(comm_mode.get_total_counts(), 0)

    def _test_dtensor_flatten_replicate(
        self, tensor_dims, flatten_start, flatten_end, mesh, placements
    ):
        # [FUNCTION DOC] HELPER: Distributes an arange tensor as Replicate, redistributes to the
        # given placements, flattens the range via view, and asserts all output placements are
        # Replicate, the local tensor equals the replicated flattened reference, and no
        # communication happened.
        mesh_ndim = len(placements)
        nelem = math.prod(tensor_dims)
        global_inps: Tensor = torch.arange(nelem).view(tensor_dims)
        global_inps_replicate: DTensor = distribute_tensor(
            global_inps, mesh, tuple([Replicate()] * mesh_ndim)
        )
        inps = global_inps_replicate.redistribute(mesh, placements)
        viewed_tensor_dims = self._get_viewed_tensor_dims(
            tensor_dims, flatten_start, flatten_end
        )
        comm_mode = CommDebugMode()
        with comm_mode:
            inps_viewed = inps.view(viewed_tensor_dims)
        expected_placements = tuple([Replicate()] * mesh_ndim)
        expected_local_tensor = (
            distribute_tensor(
                global_inps.view(viewed_tensor_dims),
                mesh,
                tuple([Replicate()] * mesh_ndim),
            )
            .redistribute(mesh, expected_placements)
            ._local_tensor
        )
        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(inps_viewed._local_tensor, expected_local_tensor)
        self.assertEqual(comm_mode.get_total_counts(), 0)

    def _get_expected_placements_ss(
        self,
        tensor_dims,
        flatten_start,
        flatten_end,
        mesh,
        placements,
    ):
        # [FUNCTION DOC] HELPER: Computes the expected output placements of flattening with all
        # Shard placements. Walks mesh dims in order tracking local dim sizes: a Shard on the
        # first flattened dim is kept; otherwise it becomes _StridedShard at flatten_start with
        # split_factor equal to the product of the current local dims before it in the range
        # (asserted greater than 1). An uneven dim is allowed only if
        # _is_last_shard_in_flatten_range holds, then the local size is rounded up; else it is
        # divided exactly. Returns a tuple of placements.
        local_tensor_dims = list(tensor_dims)
        expected_placements = []
        for idx, placement in enumerate(placements):
            self.assertIsInstance(placement, Shard)
            shard_dim = placement.dim
            if shard_dim == flatten_start:
                # S(flatten_start), S(flatten_start) qualifies
                expected_placement = placement
            else:
                split_factor = math.prod(local_tensor_dims[flatten_start:shard_dim])
                self.assertGreater(split_factor, 1)
                expected_placement = _StridedShard(
                    dim=flatten_start, split_factor=split_factor
                )
            if local_tensor_dims[shard_dim] % mesh.size(idx) != 0:
                # uneven shard on last flattened dim is supported
                self.assertTrue(
                    _ViewShardingPropagator._is_last_shard_in_flatten_range(
                        idx, placements, flatten_start, flatten_end
                    )
                )
                local_tensor_dims[shard_dim] = math.ceil(
                    local_tensor_dims[shard_dim] * 1.0 / mesh.size(idx)
                )
            else:
                local_tensor_dims[shard_dim] = local_tensor_dims[
                    shard_dim
                ] // mesh.size(idx)
            expected_placements.append(expected_placement)
        return tuple(expected_placements)

    def _test_dtensor_flatten_2d_ss(
        self,
        tensor_dims,
        flatten_start,
        flatten_end,
        mesh,
        placements,
    ):
        # [FUNCTION DOC] HELPER: Distributes an arange tensor with the given all-Shard
        # placements, flattens the range via view under CommDebugMode, and asserts the output
        # placements equal _get_expected_placements_ss, the local tensor equals a direct
        # distribution of the flattened reference using those placements, and no communication
        # occurred.
        nelem = math.prod(tensor_dims)
        global_inps: Tensor = torch.arange(nelem).view(tensor_dims)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        viewed_tensor_dims = self._get_viewed_tensor_dims(
            tensor_dims, flatten_start, flatten_end
        )

        comm_mode = CommDebugMode()
        with comm_mode:
            inps_viewed = inps.view(viewed_tensor_dims)

        expected_placements = self._get_expected_placements_ss(
            tensor_dims,
            flatten_start,
            flatten_end,
            mesh,
            placements,
        )

        expected_local_tensor = distribute_tensor(
            global_inps.view(viewed_tensor_dims),
            mesh,
            expected_placements,
            src_data_rank=None,
        )._local_tensor
        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(inps_viewed._local_tensor, expected_local_tensor)
        self.assertEqual(comm_mode.get_total_counts(), 0)

    def test_dtensor_flatten_shard_outside_range(self):
        # [FUNCTION DOC] Checks that sharding a dim outside the flatten range survives the view
        # as a plain Shard with the dim index adjusted (unchanged before the range, reduced by
        # the number of merged dims after it). Runs a table of cases on a 1D mesh, then
        # 4D cases on a (3, world_size/3) mesh with the shard on each mesh dim in turn, always
        # asserting expected placements, matching local tensors and zero communication.
        """Test that Shard on a dim outside the flatten range passes through correctly.

        When flattening dims [flatten_start, flatten_end), a Shard on a dim outside
        this range should be preserved with adjusted dim index:
        - Shard before range: dim unchanged
        - Shard after range: dim shifts by -(flatten_end - flatten_start - 1)
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        # Use sizes divisible by mesh size to avoid uneven-shard complications
        dim_size = mesh.size(0) * 2
        test_cases = [
            # (tensor_dims, flatten_start, flatten_end, shard_dim, expected_shard_dim)
            # Shard before range
            ([dim_size] * 3, 1, 3, 0, 0),
            ([dim_size] * 4, 1, 3, 0, 0),
            ([dim_size] * 4, 2, 4, 0, 0),
            ([dim_size] * 4, 2, 4, 1, 1),
            # Shard after range
            ([dim_size] * 4, 0, 2, 2, 1),
            ([dim_size] * 4, 0, 2, 3, 2),
            ([dim_size] * 4, 0, 3, 3, 1),
            ([dim_size] * 4, 1, 3, 3, 2),
        ]

        for (
            tensor_dims,
            flatten_start,
            flatten_end,
            shard_dim,
            expected_shard_dim,
        ) in test_cases:
            with self.subTest(
                dims=tensor_dims, shard=shard_dim, flat=(flatten_start, flatten_end)
            ):
                placements = (Shard(shard_dim),)
                nelem = math.prod(tensor_dims)
                global_inps = torch.arange(nelem).view(tensor_dims)
                dt = distribute_tensor(
                    global_inps, mesh, placements, src_data_rank=None
                )

                flat_dims = self._get_viewed_tensor_dims(
                    tensor_dims, flatten_start, flatten_end
                )
                comm_mode = CommDebugMode()
                with comm_mode:
                    dt_flat = dt.view(flat_dims)

                expected_placements = (Shard(expected_shard_dim),)
                self.assertEqual(dt_flat.placements, expected_placements)
                expected_local = distribute_tensor(
                    global_inps.view(flat_dims),
                    mesh,
                    expected_placements,
                    src_data_rank=None,
                )._local_tensor
                self.assertEqual(dt_flat._local_tensor, expected_local)
                self.assertEqual(comm_mode.get_total_counts(), 0)

        # 2D mesh: test shard outside range with (Shard, Replicate) and (Replicate, Shard)
        mesh_2d = init_device_mesh(self.device_type, (3, self.world_size // 3))
        dim_size_2d = mesh_2d.size(0) * mesh_2d.size(1) * 2
        test_cases_2d = [
            # (tensor_dims, flatten_start, flatten_end, shard_dim)
            ([dim_size_2d] * 4, 0, 2, 2),  # shard after range
            ([dim_size_2d] * 4, 0, 2, 3),  # shard after range
            ([dim_size_2d] * 4, 1, 3, 0),  # shard before range
            ([dim_size_2d] * 4, 2, 4, 0),  # shard before range
            ([dim_size_2d] * 4, 2, 4, 1),  # shard right before range
        ]
        for tensor_dims, flatten_start, flatten_end, shard_dim in test_cases_2d:
            num_merged = flatten_end - flatten_start - 1
            expected_shard_dim = (
                shard_dim if shard_dim < flatten_start else shard_dim - num_merged
            )
            for mesh_dim_idx in range(2):
                with self.subTest(
                    dims=tensor_dims, shard=shard_dim, mesh_dim=mesh_dim_idx
                ):
                    placements = tuple(
                        Shard(shard_dim) if i == mesh_dim_idx else Replicate()
                        for i in range(2)
                    )
                    nelem = math.prod(tensor_dims)
                    global_inps = torch.arange(nelem).view(tensor_dims)
                    dt = distribute_tensor(
                        global_inps, mesh_2d, placements, src_data_rank=None
                    )

                    flat_dims = self._get_viewed_tensor_dims(
                        tensor_dims, flatten_start, flatten_end
                    )
                    comm_mode = CommDebugMode()
                    with comm_mode:
                        dt_flat = dt.view(flat_dims)

                    expected_placements = tuple(
                        Shard(expected_shard_dim) if i == mesh_dim_idx else Replicate()
                        for i in range(2)
                    )
                    self.assertEqual(dt_flat.placements, expected_placements)
                    expected_local = distribute_tensor(
                        global_inps.view(flat_dims),
                        mesh_2d,
                        expected_placements,
                        src_data_rank=None,
                    )._local_tensor
                    self.assertEqual(dt_flat._local_tensor, expected_local)
                    self.assertEqual(comm_mode.get_total_counts(), 0)

    def test_dtensor_flatten_unflatten_roundtrip(self):
        # [FUNCTION DOC] On a 1D mesh with evenly divisible dims, checks two round trips for a
        # table of shapes, flatten ranges and shard dims: shard then flatten then unflatten, and
        # start from the flattened shape with the mapped shard dim, unflatten then flatten again.
        # Asserts the final placements equal the starting placements and the local tensor equals
        # the original local tensor.
        """Flatten then unflatten should recover the original placements and data.

        Tests the full round-trip: DTensor -> flatten -> unflatten -> compare with original.
        Covers sharding on dims before, within, and after the flatten range.
        Also tests unflatten -> flatten direction.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        dim_size = mesh.size(0) * 2  # evenly divisible

        # flatten -> unflatten round-trip
        test_cases = [
            # (tensor_dims, flatten_start, flatten_end, shard_dim)
            # Shard on first of flattened dims
            ([dim_size, dim_size], 0, 2, 0),
            # Shard on second of flattened dims
            ([dim_size, dim_size], 0, 2, 1),
            # 3D: shard in range (leading)
            ([dim_size, dim_size, dim_size], 0, 2, 0),
            # 3D: shard in range (second)
            ([dim_size, dim_size, dim_size], 0, 2, 1),
            # 3D: shard outside range (after)
            ([dim_size, dim_size, dim_size], 0, 2, 2),
            # 3D: shard before range
            ([dim_size, dim_size, dim_size], 1, 3, 0),
            # 3D: shard in range (first of second pair)
            ([dim_size, dim_size, dim_size], 1, 3, 1),
            # 3D: shard in range (second of second pair)
            ([dim_size, dim_size, dim_size], 1, 3, 2),
            # 4D: shard first of 3 flattened
            ([dim_size] * 4, 0, 3, 0),
            # 4D: shard last of 3 flattened
            ([dim_size] * 4, 0, 3, 2),
            # 4D: shard outside range (after)
            ([dim_size] * 4, 0, 3, 3),
            # 4D: shard before range
            ([dim_size] * 4, 1, 3, 0),
            # 4D: shard after range
            ([dim_size] * 4, 1, 3, 3),
        ]

        for tensor_dims, flatten_start, flatten_end, shard_dim in test_cases:
            with self.subTest(dims=tensor_dims, shard=shard_dim, direction="flatten"):
                placements = (Shard(shard_dim),)
                nelem = math.prod(tensor_dims)
                global_inps = torch.arange(nelem).view(tensor_dims)
                dt = distribute_tensor(
                    global_inps, mesh, placements, src_data_rank=None
                )

                # Flatten
                flat_dims = self._get_viewed_tensor_dims(
                    tensor_dims, flatten_start, flatten_end
                )
                dt_flat = dt.view(flat_dims)

                # Unflatten back
                dt_roundtrip = dt_flat.view(tensor_dims)

                self.assertEqual(dt_roundtrip.placements, placements)
                self.assertEqual(dt_roundtrip._local_tensor, dt._local_tensor)

        # unflatten -> flatten round-trip
        # Start with a flattened tensor, unflatten it, then flatten back
        for tensor_dims, flatten_start, flatten_end, shard_dim in test_cases:
            with self.subTest(dims=tensor_dims, shard=shard_dim, direction="unflatten"):
                flat_dims = self._get_viewed_tensor_dims(
                    tensor_dims, flatten_start, flatten_end
                )
                # Compute the shard dim in the flattened view
                if shard_dim < flatten_start:
                    flat_shard_dim = shard_dim
                elif shard_dim < flatten_end:
                    flat_shard_dim = (
                        flatten_start  # all flattened dims merge to flatten_start
                    )
                else:
                    flat_shard_dim = shard_dim - (flatten_end - flatten_start - 1)

                placements = (Shard(flat_shard_dim),)
                nelem = math.prod(flat_dims)
                global_inps = torch.arange(nelem).view(flat_dims)
                dt = distribute_tensor(
                    global_inps, mesh, placements, src_data_rank=None
                )

                # Unflatten
                dt_unflat = dt.view(tensor_dims)

                # Flatten back
                dt_roundtrip = dt_unflat.view(flat_dims)

                self.assertEqual(dt_roundtrip.placements, placements)
                self.assertEqual(dt_roundtrip._local_tensor, dt._local_tensor)

    def generate_tensor_dims_1d(
        self, tensor_ndim, flatten_start, flatten_end, shard_dim, mesh
    ):
        # [FUNCTION DOC] HELPER: Generator. For the sharded dim taking sizes one below, equal
        # to, and one above twice the mesh size (other dims one above), yields a tuple of the
        # unflattened dims, the local dims (sharded dim rounded up per rank), and the flattened
        # dims where the range [flatten_start, flatten_end) is merged into its product.
        tensor_dims_unflatten = [2 * mesh.size(0) + 1] * tensor_ndim
        for tensor_dim in [
            2 * mesh.size(0) - 1,
            2 * mesh.size(0),
            2 * mesh.size(0) + 1,
        ]:
            tensor_dims_unflatten[shard_dim] = tensor_dim
            local_tensor_dims_unflatten = list(tensor_dims_unflatten)
            local_tensor_dims_unflatten[shard_dim] = math.ceil(
                tensor_dim * 1.0 / mesh.size(0)
            )
            nelem_flatten = math.prod(tensor_dims_unflatten[flatten_start:flatten_end])
            tensor_dims_flatten = (
                tensor_dims_unflatten[0:flatten_start]
                + [nelem_flatten]
                + tensor_dims_unflatten[flatten_end:]
            )
            yield (
                tensor_dims_unflatten,
                local_tensor_dims_unflatten,
                tensor_dims_flatten,
            )

    def generate_tensor_dims_1d_after_flatten(
        self, tensor_ndim, unflatten_dim, shard_dim, mesh
    ):
        # [FUNCTION DOC] HELPER: Generator yielding lists of tensor dims. All dims start as the
        # square of the mesh size; the unflatten dim and the shard dim each take that value minus
        # one, itself, and plus one (all nine combinations; if both indices coincide the shard
        # value wins). Each yielded list is a fresh copy.
        tensor_dims = [mesh.size(0) * mesh.size(0)] * tensor_ndim
        for unflatten_dim_value in [
            mesh.size(0) * mesh.size(0) - 1,
            mesh.size(0) * mesh.size(0),
            mesh.size(0) * mesh.size(0) + 1,
        ]:
            for shard_dim_value in [
                mesh.size(0) * mesh.size(0) - 1,
                mesh.size(0) * mesh.size(0),
                mesh.size(0) * mesh.size(0) + 1,
            ]:
                tensor_dims[unflatten_dim] = unflatten_dim_value
                tensor_dims[shard_dim] = shard_dim_value
                yield list(tensor_dims)

    def test_dtensor_unflatten_1d(self):
        # [FUNCTION DOC] On a 1D mesh runs three unflatten checks. First, for ranks 2 to 4 and
        # flatten ranges, round-trips from a flattened shape with a Shard or _StridedShard input
        # via _test_dtensor_unflatten_1d_shard, expecting RuntimeError "is not evenly divisible
        # by mesh dimension" when the sharded size is uneven and not the last dim of the range.
        # Second, tries every factorization of one dim via
        # _test_dtensor_unflatten_1d_shard_arbitrary. Third, checks Replicate is preserved and
        # shape correct when a size-12 dim is viewed as (3, 4).
        mesh: DeviceMesh = init_device_mesh(self.device_type, (self.world_size,))

        # flatten -> view -> unflatten
        for tensor_ndim in [2, 3, 4]:
            for flatten_start in range(tensor_ndim):
                for flatten_end in range(flatten_start + 2, tensor_ndim + 1):
                    for shard_dim in range(flatten_start, flatten_end):
                        expected_placements = (Shard(shard_dim),)
                        for (
                            tensor_dims_unflatten,
                            local_tensor_dims_unflatten,
                            tensor_dims_flatten,
                        ) in self.generate_tensor_dims_1d(
                            tensor_ndim, flatten_start, flatten_end, shard_dim, mesh
                        ):
                            ctx = contextlib.nullcontext()
                            is_uneven = (
                                tensor_dims_unflatten[shard_dim] % mesh.size(0) != 0
                            )
                            is_last_dim = shard_dim == flatten_end - 1
                            if is_uneven and not is_last_dim:
                                ctx = self.assertRaisesRegex(
                                    RuntimeError,
                                    "is not evenly divisible by mesh dimension",
                                )
                            with (
                                self.subTest(
                                    dims=tensor_dims_unflatten,
                                    shard=shard_dim,
                                    flat=(flatten_start, flatten_end),
                                ),
                                ctx,
                            ):
                                self._test_dtensor_unflatten_1d_shard(
                                    tensor_dims_unflatten,
                                    local_tensor_dims_unflatten,
                                    tensor_dims_flatten,
                                    flatten_start,
                                    expected_placements,
                                    mesh,
                                )

        # any factoring on unflatten_dim
        for tensor_ndim in [1, 2, 3, 4]:
            for unflatten_dim in range(tensor_ndim):
                for shard_dim in range(tensor_ndim):
                    for tensor_dims in self.generate_tensor_dims_1d_after_flatten(
                        tensor_ndim, unflatten_dim, shard_dim, mesh
                    ):
                        with self.subTest(
                            dims=tensor_dims, shard=shard_dim, unflatten=unflatten_dim
                        ):
                            placements = (Shard(shard_dim),)
                            self._test_dtensor_unflatten_1d_shard_arbitrary(
                                tensor_dims,
                                unflatten_dim,
                                placements,
                                mesh,
                            )

        # Replicate: unflatten should preserve Replicate placement
        for tensor_ndim in [1, 2, 3, 4]:
            for unflatten_dim in range(tensor_ndim):
                tensor_dims = [6] * tensor_ndim
                tensor_dims[unflatten_dim] = 12  # will unflatten to (3, 4)
                global_tensor = torch.arange(math.prod(tensor_dims)).view(tensor_dims)
                dt = distribute_tensor(
                    global_tensor, mesh, (Replicate(),), src_data_rank=None
                )
                # Unflatten dimension to (3, 4)
                unflatten_shape = list(tensor_dims)
                unflatten_shape[unflatten_dim : unflatten_dim + 1] = [3, 4]
                dt_unflattened = dt.view(unflatten_shape)
                self.assertEqual(dt_unflattened.placements, (Replicate(),))
                self.assertEqual(dt_unflattened.shape, torch.Size(unflatten_shape))

    def _test_dtensor_unflatten_1d_shard(
        self,
        tensor_dims_unflatten,
        local_tensor_dims_unflatten,
        tensor_dims_flatten,
        flatten_start,
        expected_placements,
        mesh,
    ):
        # [FUNCTION DOC] HELPER: Builds the flattened-side input placement: Shard at
        # flatten_start when the product of local dims between flatten_start and the expected
        # shard dim is 1 (asserting they coincide), else _StridedShard at flatten_start with
        # that product as split_factor. Distributes the flattened arange tensor, views it as
        # the unflattened shape, and asserts the placements equal expected_placements and the
        # local tensor equals a direct distribution of the unflattened reference.
        shard_dim = expected_placements[0].dim
        split_factor = math.prod(local_tensor_dims_unflatten[flatten_start:shard_dim])
        if split_factor == 1:
            self.assertEqual(shard_dim, flatten_start)
            placements = (Shard(flatten_start),)
        else:
            placements = (_StridedShard(flatten_start, split_factor=split_factor),)
        nelem = math.prod(tensor_dims_flatten)
        global_inps = torch.arange(nelem).view(tensor_dims_flatten)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        inps_viewed = inps.view(tensor_dims_unflatten)
        self.assertEqual(inps_viewed.placements, expected_placements)
        expected_local_tensor = distribute_tensor(
            torch.arange(nelem).view(tensor_dims_unflatten),
            mesh,
            expected_placements,
            src_data_rank=None,
        )._local_tensor
        self.assertEqual(inps_viewed._local_tensor, expected_local_tensor)

    def _test_dtensor_unflatten_1d_shard_arbitrary(
        self, tensor_dims, unflatten_dim, placements, mesh
    ):
        # [FUNCTION DOC] HELPER: Requires a plain Shard (not _StridedShard) placement. Distributes
        # an arange tensor and views it with unflatten_dim split into each non-trivial
        # factorization of its size (skips primes). When the sharded dim is the unflattened one,
        # a RuntimeError "is not evenly divisible by mesh dimension" is expected if the size is
        # uneven or the first factor is not divisible by the mesh size. Otherwise asserts the
        # output is Shard at the dim shifted by the number of added dims, no communication, and
        # the expected local tensor.
        shard_dim = placements[0].dim
        self.assertIsInstance(placements[0], Shard)
        self.assertNotIsInstance(placements[0], _StridedShard)

        tensor_dims = list(tensor_dims)  # Make a mutable copy

        # Get all non-trivial factorizations of unflatten_dim size
        unflatten_size = tensor_dims[unflatten_dim]
        factorizations = self._get_all_factorizations(unflatten_size)

        # Skip if no non-trivial factorization exists (prime number)
        if not factorizations:
            return

        # Create the global tensor once (reused for all factorizations)
        nelem = math.prod(tensor_dims)
        global_inps = torch.arange(nelem).view(tensor_dims)

        # Distribute the tensor
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)

        # Test each factorization (can be any length >= 2)
        for factors in factorizations:
            tensor_dims_unflatten = (
                tensor_dims[:unflatten_dim]
                + list(factors)
                + tensor_dims[unflatten_dim + 1 :]
            )

            # Determine if we expect an error
            first_factor = factors[0]

            ctx = contextlib.nullcontext()
            expect_error = False

            if shard_dim == unflatten_dim:
                # When unflattening the sharded dimension, check factor alignment
                uneven_shard = tensor_dims[shard_dim] % mesh.size(0) != 0
                first_factor_aligned = first_factor % mesh.size(0) == 0

                expect_error = not first_factor_aligned or uneven_shard

            if expect_error:
                ctx = self.assertRaisesRegex(
                    RuntimeError, "is not evenly divisible by mesh dimension"
                )

            with ctx:
                comm_mode = CommDebugMode()
                with comm_mode:
                    inps_viewed = inps.view(tensor_dims_unflatten)

                # Number of new dimensions added by unflatten
                num_new_dims = len(factors) - 1

                # Compute expected placements after unflatten
                if shard_dim < unflatten_dim:
                    expected_placements = (Shard(shard_dim),)
                elif shard_dim == unflatten_dim:
                    expected_placements = (Shard(unflatten_dim),)
                else:
                    expected_placements = (Shard(shard_dim + num_new_dims),)

                self.assertEqual(inps_viewed.placements, expected_placements)
                self.assertEqual(comm_mode.get_total_counts(), 0)

                expected_local = distribute_tensor(
                    global_inps.view(tensor_dims_unflatten),
                    mesh,
                    expected_placements,
                    src_data_rank=None,
                )._local_tensor
                self.assertEqual(inps_viewed._local_tensor, expected_local)

    def test_dtensor_unflatten_multi_mesh(self):
        # [FUNCTION DOC] Iterates over 2D (3, 2) and 3D (2, 2, 2) meshes (skipped if larger than
        # the world size) with a number of leading Replicate mesh dims. All-replicate cases
        # unflatten with every factorization and prefix dims and assert Replicate placements
        # and the new shape. Sharded cases choose, for each factorization, which output factors
        # the shard mesh dims land on, expect RuntimeError (not evenly divisible, or strided
        # shard-order unsupported) when an uneven factor is not on the last shard and last
        # factor, and otherwise call _test_dtensor_unflatten_factors.
        """Test unflatten across 2D and 3D meshes with all placement patterns.

        Iterates over 2D and 3D mesh configurations to test multi-shard (SS, SSS),
        mixed (R+SS, RR+SS), and replicate (RR, RRR) patterns.
        """
        # (mesh_shape, num_replicate_dims)
        cases = [
            ((3, 2), 0),  # (SS, SS)
            ((3, 2), 1),  # (R, SS)
            ((3, 2), 2),  # (R, R)
            ((2, 2, 2), 0),  # (SS, SS, SS)
            ((2, 2, 2), 1),  # (R, SS, SS)
            ((2, 2, 2), 2),  # (R, R, SS)
            ((2, 2, 2), 3),  # (R, R, R)
        ]
        for mesh_shape, num_rep in cases:
            if self.world_size < math.prod(mesh_shape):
                continue
            mesh = init_device_mesh(self.device_type, mesh_shape)
            mesh_ndim = len(mesh_shape)
            num_shard = mesh_ndim - num_rep
            flattened_size = math.prod(mesh_shape) * 12
            factorizations = self._get_all_factorizations(flattened_size)

            if num_shard == 0:
                # All-replicate: test that Replicate is preserved
                for factors in factorizations:
                    for prefix_size in [0, 1, 2]:
                        with self.subTest(
                            factors=factors,
                            prefix=prefix_size,
                            mesh=mesh_shape,
                        ):
                            tensor_dims_unflatten = (
                                [6] * prefix_size + list(factors) + [3]
                            )
                            nelem_flatten = math.prod(factors)
                            tensor_dims_flatten = (
                                [6] * prefix_size + [nelem_flatten] + [3]
                            )
                            nelem = math.prod(tensor_dims_flatten)
                            global_tensor = torch.arange(nelem).view(
                                tensor_dims_flatten
                            )
                            rep_placements = tuple([Replicate()] * mesh_ndim)
                            dt = distribute_tensor(
                                global_tensor,
                                mesh,
                                rep_placements,
                                src_data_rank=None,
                            )
                            dt_unflattened = dt.view(tensor_dims_unflatten)
                            self.assertEqual(
                                dt_unflattened.placements,
                                rep_placements,
                            )
                            self.assertEqual(
                                dt_unflattened.shape,
                                torch.Size(tensor_dims_unflatten),
                            )
            else:
                # Sharded cases: pick shard indices from factorizations
                min_factors = num_shard + 1  # need prefix + num_shard shard dims
                valid = [f for f in factorizations if len(f) >= min_factors]
                for factors in valid:
                    n = len(factors)
                    for shard_indices in itertools.combinations(
                        range(1, n),
                        num_shard,
                    ):
                        # Error detection: uneven on non-(last-shard AND last-factor)
                        expect_error = False
                        for rank, si in enumerate(shard_indices):
                            mesh_dim = num_rep + rank
                            is_last_shard = rank == num_shard - 1
                            is_last_factor = si == n - 1
                            if factors[si] % mesh.size(mesh_dim) != 0 and not (
                                is_last_shard and is_last_factor
                            ):
                                expect_error = True
                                break
                        with self.subTest(
                            factors=factors,
                            shard=shard_indices,
                            mesh=mesh_shape,
                        ):
                            if expect_error:
                                with self.assertRaisesRegex(
                                    RuntimeError,
                                    "is not evenly divisible by mesh dimension|"
                                    "do not support inputs with use_strided_shard_as_shard_order",
                                ):
                                    self._test_dtensor_unflatten_factors(
                                        factors,
                                        shard_indices,
                                        num_rep,
                                        mesh,
                                    )
                            else:
                                self._test_dtensor_unflatten_factors(
                                    factors,
                                    shard_indices,
                                    num_rep,
                                    mesh,
                                )

    def _test_dtensor_unflatten_factors(
        self,
        factors,
        shard_indices,
        num_rep,
        mesh,
    ):
        # [FUNCTION DOC] HELPER: Builds a flattened tensor of the product of factors (plus a
        # trailing size-3 dim when at least two mesh dims shard), with num_rep leading Replicate
        # placements and _StridedShard(0) on the remaining mesh dims whose split_factor is the
        # product of the factors before the target index divided by earlier shard mesh sizes.
        # Clears use_strided_shard_as_shard_order and recomputes shard_order on the spec, views
        # as the unflattened shape, and asserts Shard(index) placements, zero communication and
        # matching local tensor.
        """Test unflatten with _StridedShard placements for any mesh dimensionality.

        Args:
            factors: Tuple of factors representing the unflatten shape
            shard_indices: Tuple of indices within factors for each sharded mesh dim
            num_rep: Number of leading Replicate mesh dims
            mesh: DeviceMesh
        """
        num_shard = len(shard_indices)
        # SS cases need a suffix dim outside the flatten range
        suffix = [3] if num_shard >= 2 else []
        tensor_dims_unflatten = list(factors) + suffix
        nelem_flatten = math.prod(factors)
        tensor_dims_flatten = [nelem_flatten] + suffix

        # Build input placements and expected output placements
        placements = []
        expected_placements = []
        for i in range(mesh.ndim):
            if i < num_rep:
                placements.append(Replicate())
                expected_placements.append(Replicate())
            else:
                rank = i - num_rep
                si = shard_indices[rank]
                # split_factor = prod(factors[:si]) / prod(earlier mesh dims)
                sf = math.prod(factors[:si])
                for j in range(rank):
                    sf //= mesh.size(num_rep + j)
                placements.append(_StridedShard(0, split_factor=sf))
                expected_placements.append(Shard(si))

        placements = tuple(placements)
        expected_placements = tuple(expected_placements)

        nelem = math.prod(tensor_dims_flatten)
        global_inps = torch.arange(nelem).view(tensor_dims_flatten)
        inps = distribute_tensor(global_inps, mesh, placements, src_data_rank=None)
        # These _StridedShard placements represent literal strided layout
        # (as produced by flatten), not shard-order encoding.
        inps._spec.use_strided_shard_as_shard_order = False
        inps._spec.shard_order = DTensorSpec.compute_default_shard_order(
            inps._spec.placements
        )

        comm_mode = CommDebugMode()
        with comm_mode:
            inps_viewed = inps.view(tensor_dims_unflatten)

        self.assertEqual(inps_viewed.placements, expected_placements)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        expected_local_tensor = distribute_tensor(
            torch.arange(nelem).view(tensor_dims_unflatten),
            mesh,
            expected_placements,
            src_data_rank=None,
        )._local_tensor
        self.assertEqual(inps_viewed._local_tensor, expected_local_tensor)

    def test_dtensor_flatten_unflatten_2d_reversed_mesh(self):
        # [FUNCTION DOC] On a (2, 3) mesh (requires world size 6, else AssertionError from the
        # equality check) runs flatten with Shard on two flattened dims for 3D tensors via
        # _test_dtensor_flatten_2d_ss, then unflatten of several factor triples with
        # _StridedShard inputs via _test_dtensor_unflatten_factors, expecting RuntimeError
        # "is not evenly divisible" when the first sharded factor is uneven for mesh dim 0 or
        # the second is uneven for mesh dim 1 and not last.
        """Test flatten/unflatten with reversed mesh shape (2, 3) to catch ordering bugs."""
        self.assertEqual(self.world_size, 6)
        mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        dim_size = mesh.size(0) * mesh.size(1) * 2  # divisible by both mesh dims

        # Flatten with (S, S) pattern
        tensor_ndim = 3
        for flatten_start in range(tensor_ndim):
            for flatten_end in range(flatten_start + 2, tensor_ndim + 1):
                for shard_dim0 in range(flatten_start, flatten_end):
                    for shard_dim1 in range(shard_dim0, flatten_end):
                        with self.subTest(
                            shard0=shard_dim0,
                            shard1=shard_dim1,
                            flat=(flatten_start, flatten_end),
                        ):
                            tensor_dims = tuple([dim_size] * tensor_ndim)
                            placements = (Shard(shard_dim0), Shard(shard_dim1))
                            self._test_dtensor_flatten_2d_ss(
                                tensor_dims,
                                flatten_start,
                                flatten_end,
                                mesh,
                                placements,
                            )

        # Unflatten with (SS, SS) pattern — representative factorizations
        factors_list = [(6, 4, 3), (4, 6, 2), (3, 2, 6), (2, 6, 4)]
        for factors in factors_list:
            for shard_idx0 in range(1, len(factors) - 1):
                for shard_idx1 in range(shard_idx0 + 1, len(factors)):
                    with self.subTest(factors=factors, s0=shard_idx0, s1=shard_idx1):
                        factor0 = factors[shard_idx0]
                        factor1 = factors[shard_idx1]
                        uneven0 = factor0 % mesh.size(0) != 0
                        is_last1 = shard_idx1 == len(factors) - 1
                        uneven1 = factor1 % mesh.size(1) != 0 and not is_last1
                        if uneven0 or uneven1:
                            with self.assertRaisesRegex(
                                RuntimeError, "is not evenly divisible"
                            ):
                                self._test_dtensor_unflatten_factors(
                                    factors,
                                    (shard_idx0, shard_idx1),
                                    0,
                                    mesh,
                                )
                        else:
                            self._test_dtensor_unflatten_factors(
                                factors,
                                (shard_idx0, shard_idx1),
                                0,
                                mesh,
                            )

    def test_dtensor_unflatten_ss_and_s_same_dim(self):
        # [FUNCTION DOC] On a (3, 2) mesh shards a (4, 6, 3) tensor with Shard(1), Shard(0),
        # flattens dims 0 and 1 and asserts no communication, shape (24, 3), a _StridedShard of
        # split_factor 4 on mesh dim 0 and Shard on mesh dim 1. Then views dim 0 as (12, 2) and
        # asserts the placements stay (_StridedShard(0, 4), Shard(0)), no communication, and the
        # local tensor matches direct distribution.
        """Test unflatten when _StridedShard and Shard both shard the same tensor dim."""
        mesh = init_device_mesh(self.device_type, (3, self.world_size // 3))

        global_tensor = torch.arange(4 * 6 * 3).view(4, 6, 3)
        inps = distribute_tensor(global_tensor, mesh, [Shard(1), Shard(0)])

        # Flatten dims 0,1 -> [24, 3] with [_StridedShard(0, 4), Shard(0)]
        comm_mode = CommDebugMode()
        with comm_mode:
            flattened = inps.flatten(0, 1)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(flattened.shape, (24, 3))
        self.assertIsInstance(flattened.placements[0], _StridedShard)
        self.assertEqual(flattened.placements[0].split_factor, 4)
        self.assertIsInstance(flattened.placements[1], Shard)

        # Unflatten dim 0 (24) -> (12, 2): both mesh dims stay on output dim 0
        expected_placements = (_StridedShard(0, split_factor=4), Shard(0))
        comm_mode = CommDebugMode()
        with comm_mode:
            unflattened = flattened.view(12, 2, 3)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(unflattened.placements, expected_placements)
        expected_local = distribute_tensor(
            torch.arange(4 * 6 * 3).view(12, 2, 3),
            mesh,
            expected_placements,
            src_data_rank=None,
        )._local_tensor
        self.assertEqual(unflattened._local_tensor, expected_local)

    def test_flatten_then_sum_non_shard_dim(self):
        # [FUNCTION DOC] Flattens a Shard to a _StridedShard and checks sum: reducing a
        # non-sharded dim keeps _StridedShard with unchanged dim and split_factor; reducing the
        # strided dim gives a Partial placement; reducing a lower dim moves _StridedShard(1)
        # to _StridedShard(0). Each result's full tensor equals the single-device sum.
        """Verify _StridedShard correctness through sum on a non-shard dim.

        sum uses map_placements_after_reduction which must preserve
        _StridedShard (with split_factor) when remapping dims after
        the reduced dim is removed.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)  # _StridedShard(dim=0)
        flat_full = full.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)

        # reduce dim 1 (non-shard): _StridedShard(dim=0) stays at dim 0
        result = dt_flat.sum(dim=1)
        self.assertIsInstance(result.placements[0], _StridedShard)
        self.assertEqual(result.placements[0].dim, 0)
        self.assertEqual(
            result.placements[0].split_factor, dt_flat.placements[0].split_factor
        )
        self.assertEqual(result.full_tensor(), flat_full.sum(dim=1))

        # reduce dim 0 (shard dim): should produce Partial
        result2 = dt_flat.sum(dim=0)
        self.assertTrue(result2.placements[0].is_partial())
        self.assertEqual(result2.full_tensor(), flat_full.sum(dim=0))

        # _StridedShard on higher dim: reduce a dim below it to trigger remap
        shape2 = (3, 5, self.world_size * 2)
        full2 = torch.randn(*shape2, device=self.device_type)
        dt2 = distribute_tensor(full2, mesh, [Shard(2)])
        dt2_flat = dt2.flatten(1, 2)  # _StridedShard(dim=1)
        flat_full2 = full2.flatten(1, 2)

        self.assertIsInstance(dt2_flat.placements[0], _StridedShard)
        self.assertEqual(dt2_flat.placements[0].dim, 1)

        # reduce dim 0: _StridedShard(dim=1) remaps to _StridedShard(dim=0)
        result3 = dt2_flat.sum(dim=0)
        self.assertIsInstance(result3.placements[0], _StridedShard)
        self.assertEqual(result3.placements[0].dim, 0)
        self.assertEqual(
            result3.placements[0].split_factor, dt2_flat.placements[0].split_factor
        )
        self.assertEqual(result3.full_tensor(), flat_full2.sum(dim=0))

    def test_flatten_then_softmax(self):
        # [FUNCTION DOC] Flattens a sharded tensor to get a _StridedShard placement, then
        # asserts softmax over the strided dim and over the last dim each give a full tensor
        # equal to the single-device softmax.
        """Verify _StridedShard correctness through softmax.

        softmax uses replicate_reduction_dims which only checks isinstance(p, Shard).
        _StridedShard on the softmax dim may not be replicated, producing wrong results.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)
        flat_full = full.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)

        # softmax on shard dim
        result = torch.softmax(dt_flat, dim=0)
        self.assertEqual(result.full_tensor(), torch.softmax(flat_full, dim=0))

        # softmax on non-shard dim
        result = torch.softmax(dt_flat, dim=-1)
        self.assertEqual(result.full_tensor(), torch.softmax(flat_full, dim=-1))

    def test_flatten_then_layer_norm(self):
        # [FUNCTION DOC] With a _StridedShard input, asserts layer_norm over only the last dim
        # and layer_norm normalizing over the whole shape (covering the strided dim) both match
        # the single-device result.
        """Verify _StridedShard correctness through layer_norm.

        layer_norm uses _replicate_dims_start_at which only checks isinstance(p, Shard).
        _StridedShard on normalized dims may not be replicated.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)
        flat_full = full.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)

        # layer_norm on last dim (non-shard): shard dim not in normalized dims
        result = torch.nn.functional.layer_norm(dt_flat, [6])
        self.assertEqual(
            result.full_tensor(),
            torch.nn.functional.layer_norm(flat_full, [6]),
        )

        # layer_norm covering shard dim
        result = torch.nn.functional.layer_norm(dt_flat, list(dt_flat.shape))
        self.assertEqual(
            result.full_tensor(),
            torch.nn.functional.layer_norm(flat_full, list(flat_full.shape)),
        )

    def test_flatten_then_transpose(self):
        # [FUNCTION DOC] Checks transpose and t on _StridedShard inputs. transpose of dims 0 and
        # 1 matches single device. For t, the result stays _StridedShard with the dim swapped
        # (0 to 1 and 1 to 0) and split_factor preserved, for even and uneven shapes, with full
        # tensors equal to the reference.
        """Verify _StridedShard correctness through transpose.

        aten.transpose.int goes through view op propagation which handles
        _StridedShard. aten.t uses a single-dim strategy in _matrix_ops.py
        which swaps the dim for both Shard and _StridedShard.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)
        flat_full = full.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)

        # transpose shard dim with another dim (aten.transpose.int)
        result = dt_flat.transpose(0, 1)
        self.assertEqual(result.full_tensor(), flat_full.transpose(0, 1))

        # t() on 2D tensor (aten.t via single-dim strategy in _matrix_ops.py)
        # _StridedShard(dim=0) → _StridedShard(dim=1), split_factor preserved
        result = dt_flat.t()
        self.assertIsInstance(result.placements[0], _StridedShard)
        self.assertEqual(result.placements[0].dim, 1)
        self.assertEqual(
            result.placements[0].split_factor, dt_flat.placements[0].split_factor
        )
        self.assertEqual(result.full_tensor(), flat_full.t())

        # t() with _StridedShard(dim=1) → _StridedShard(dim=0)
        # Shard(2) on (4, 6, ws*2) + flatten(1,2) → _StridedShard(dim=1)
        shape1 = (4, 6, self.world_size * 2)
        full1 = torch.randn(*shape1, device=self.device_type)
        dt1 = distribute_tensor(full1, mesh, [Shard(2)])
        dt1_flat = dt1.flatten(1, 2)  # (4, 6*ws*2) with _StridedShard(dim=1)
        self.assertIsInstance(dt1_flat.placements[0], _StridedShard)
        self.assertEqual(dt1_flat.placements[0].dim, 1)

        result1 = dt1_flat.t()
        self.assertIsInstance(result1.placements[0], _StridedShard)
        self.assertEqual(result1.placements[0].dim, 0)
        self.assertEqual(
            result1.placements[0].split_factor, dt1_flat.placements[0].split_factor
        )
        self.assertEqual(result1.full_tensor(), full1.flatten(1, 2).t())

        # t() with uneven shapes: shard dim not divisible by world_size
        # _StridedShard(dim=0) → _StridedShard(dim=1)
        shape2 = (
            3,
            self.world_size * 2,
            5,
        )  # flatten(0,1) gives (3*ws*2, 5), uneven in dim 0
        full2 = torch.randn(*shape2, device=self.device_type)
        dt2 = distribute_tensor(full2, mesh, [Shard(1)])
        dt2_flat = dt2.flatten(0, 1)
        self.assertIsInstance(dt2_flat.placements[0], _StridedShard)

        result2 = dt2_flat.t()
        self.assertIsInstance(result2.placements[0], _StridedShard)
        self.assertEqual(result2.placements[0].dim, 1)
        self.assertEqual(
            result2.placements[0].split_factor, dt2_flat.placements[0].split_factor
        )
        self.assertEqual(result2.full_tensor(), full2.flatten(0, 1).t())

        # _StridedShard(dim=1) → _StridedShard(dim=0), uneven
        shape3 = (
            5,
            3,
            self.world_size * 2,
        )  # flatten(1,2) gives (5, 3*ws*2), uneven in dim 1
        full3 = torch.randn(*shape3, device=self.device_type)
        dt3 = distribute_tensor(full3, mesh, [Shard(2)])
        dt3_flat = dt3.flatten(1, 2)
        self.assertIsInstance(dt3_flat.placements[0], _StridedShard)
        self.assertEqual(dt3_flat.placements[0].dim, 1)

        result3 = dt3_flat.t()
        self.assertIsInstance(result3.placements[0], _StridedShard)
        self.assertEqual(result3.placements[0].dim, 0)
        self.assertEqual(
            result3.placements[0].split_factor, dt3_flat.placements[0].split_factor
        )
        self.assertEqual(result3.full_tensor(), full3.flatten(1, 2).t())

    def test_flatten_then_nll_loss(self):
        # [FUNCTION DOC] Builds flattened _StridedShard input and target DTensors (the target
        # broadcast from rank 0 for consistency), once for a 2D (N flattened) layout and once for
        # a 4D layout flattening spatial dims, applies log_softmax then nll_loss and asserts the
        # full tensor matches the single-device loss.
        """Verify _StridedShard correctness through nll_loss.

        nll_loss_forward_strategy uses replicate_reduction_dims on the channel
        dim and _skip_dim to build target placements.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        num_classes = 10

        # 2D input: (N, C) — _StridedShard on batch dim (dim 0), which is
        # below channel_dim (dim 1), so _skip_dim preserves it unchanged.
        shape = (4, self.world_size * 2)
        full_input = torch.randn(*shape, num_classes, device=self.device_type)
        full_target = torch.randint(0, num_classes, shape, device=self.device_type)
        dist.broadcast(full_target, src=0)

        dt_input = distribute_tensor(full_input, mesh, [Shard(1)])
        dt_input_flat = dt_input.flatten(0, 1)
        dt_target = distribute_tensor(full_target, mesh, [Shard(1)])
        dt_target_flat = dt_target.flatten(0, 1)

        self.assertIsInstance(dt_input_flat.placements[0], _StridedShard)

        input_full = full_input.flatten(0, 1)
        target_full = full_target.flatten(0, 1)

        result = torch.nn.functional.nll_loss(
            torch.log_softmax(dt_input_flat, dim=-1), dt_target_flat
        )
        expected = torch.nn.functional.nll_loss(
            torch.log_softmax(input_full, dim=-1), target_full
        )
        self.assertEqual(result.full_tensor(), expected)

        # 3D input: (N, C, D) — _StridedShard on spatial dim (dim 2), which
        # is above channel_dim (dim 1), so _skip_dim must shift it from
        # dim 2 to dim 1 while preserving split_factor.
        N, C, D1, D2 = 2, num_classes, 3, self.world_size * 2
        full_input_3d = torch.randn(N, C, D1, D2, device=self.device_type)
        full_target_3d = torch.randint(0, C, (N, D1, D2), device=self.device_type)
        dist.broadcast(full_target_3d, src=0)

        dt_input_3d = distribute_tensor(full_input_3d, mesh, [Shard(3)])
        dt_input_3d_flat = dt_input_3d.flatten(2, 3)
        dt_target_3d = distribute_tensor(full_target_3d, mesh, [Shard(2)])
        dt_target_3d_flat = dt_target_3d.flatten(1, 2)

        self.assertIsInstance(dt_input_3d_flat.placements[0], _StridedShard)
        self.assertEqual(dt_input_3d_flat.placements[0].dim, 2)

        input_3d_full = full_input_3d.flatten(2, 3)
        target_3d_full = full_target_3d.flatten(1, 2)

        result_3d = torch.nn.functional.nll_loss(
            torch.log_softmax(dt_input_3d_flat, dim=1), dt_target_3d_flat
        )
        expected_3d = torch.nn.functional.nll_loss(
            torch.log_softmax(input_3d_full, dim=1), target_3d_full
        )
        self.assertEqual(result_3d.full_tensor(), expected_3d)

    def test_flatten_then_select(self):
        # [FUNCTION DOC] On a _StridedShard(dim=2) tensor, select on dim 0 or 1 keeps
        # _StridedShard at dim 1 with the same split_factor; select on the strided dim gives a
        # non-_StridedShard placement. All results match single-device values.
        """Verify _StridedShard correctness through select.

        select removes a dim, so select_int_strategy must detect
        _StridedShard (which is_sharded() misses) and call
        shift_shard_dims_after_remove to adjust the shard dim.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (2, 3, 4, self.world_size * 2)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(3)])
        dt_flat = dt.flatten(2, 3)  # (2, 3, 4*ws*2) with _StridedShard(dim=2)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)
        orig_split_factor = dt_flat.placements[0].split_factor

        # select on dim 0: _StridedShard dim shifts from 2 to 1, split_factor preserved
        result = dt_flat.select(0, 0)
        expected = full.flatten(2, 3).select(0, 0)
        self.assertIsInstance(result.placements[0], _StridedShard)
        self.assertEqual(result.placements[0].dim, 1)
        self.assertEqual(result.placements[0].split_factor, orig_split_factor)
        self.assertEqual(result.full_tensor(), expected)

        # select on dim 1: _StridedShard dim shifts from 2 to 1, split_factor preserved
        result1 = dt_flat.select(1, 0)
        expected1 = full.flatten(2, 3).select(1, 0)
        self.assertIsInstance(result1.placements[0], _StridedShard)
        self.assertEqual(result1.placements[0].dim, 1)
        self.assertEqual(result1.placements[0].split_factor, orig_split_factor)
        self.assertEqual(result1.full_tensor(), expected1)

        # select on the _StridedShard dim: must unshard to Replicate
        result2 = dt_flat.select(2, 0)
        expected2 = full.flatten(2, 3).select(2, 0)
        self.assertNotIsInstance(result2.placements[0], _StridedShard)
        self.assertEqual(result2.full_tensor(), expected2)

    def test_flatten_then_unbind(self):
        # [FUNCTION DOC] Unbinds _StridedShard tensors on dims before and after the strided dim.
        # Each output keeps _StridedShard with the dim shifted down by one when the unbound dim
        # is lower (2 to 1) or unchanged otherwise, split_factor preserved, and full tensors
        # equal the single-device unbind outputs.
        """Verify _StridedShard correctness through unbind.

        unbind removes a dim via shift_shard_dims_after_remove, which
        must preserve split_factor when shifting _StridedShard dims.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (2, 3, 4, self.world_size * 2)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(3)])
        dt_flat = dt.flatten(2, 3)  # (2, 3, 4*ws*2) with _StridedShard(dim=2)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)
        orig_split_factor = dt_flat.placements[0].split_factor

        # unbind on dim 0: _StridedShard dim shifts from 2 to 1
        results = torch.unbind(dt_flat, dim=0)
        expected = torch.unbind(full.flatten(2, 3), dim=0)
        for r, e in zip(results, expected):
            self.assertIsInstance(r.placements[0], _StridedShard)
            self.assertEqual(r.placements[0].dim, 1)
            self.assertEqual(r.placements[0].split_factor, orig_split_factor)
            self.assertEqual(r.full_tensor(), e)

        # unbind on dim 1: _StridedShard dim shifts from 2 to 1
        results1 = torch.unbind(dt_flat, dim=1)
        expected1 = torch.unbind(full.flatten(2, 3), dim=1)
        for r, e in zip(results1, expected1):
            self.assertIsInstance(r.placements[0], _StridedShard)
            self.assertEqual(r.placements[0].dim, 1)
            self.assertEqual(r.placements[0].split_factor, orig_split_factor)
            self.assertEqual(r.full_tensor(), e)

        # unbind on dim above shard dim: _StridedShard dim stays unchanged
        shape2 = (4, self.world_size * 2, 3, 5)
        full2 = torch.randn(*shape2, device=self.device_type)
        dt2 = distribute_tensor(full2, mesh, [Shard(1)])
        dt2_flat = dt2.flatten(0, 1)  # (4*ws*2, 3, 5) with _StridedShard(dim=0)
        self.assertIsInstance(dt2_flat.placements[0], _StridedShard)
        self.assertEqual(dt2_flat.placements[0].dim, 0)
        orig_split_factor2 = dt2_flat.placements[0].split_factor

        results2 = torch.unbind(dt2_flat, dim=2)
        expected2 = torch.unbind(full2.flatten(0, 1), dim=2)
        for r, e in zip(results2, expected2):
            self.assertIsInstance(r.placements[0], _StridedShard)
            self.assertEqual(r.placements[0].dim, 0)
            self.assertEqual(r.placements[0].split_factor, orig_split_factor2)
            self.assertEqual(r.full_tensor(), e)

    def test_flatten_then_stack(self):
        # [FUNCTION DOC] Stacks a _StridedShard tensor with itself: stacking at dim 0 shifts the
        # strided dim from 0 to 1, stacking at dim 2 leaves it at 0; split_factor is preserved and
        # full tensors match the single-device stack.
        """Verify _StridedShard correctness through stack.

        stack inserts a new dim via shift_shard_dims_after_insert, which
        must preserve split_factor when shifting _StridedShard dims.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)
        orig_split_factor = dt_flat.placements[0].split_factor

        # stack along dim=0: _StridedShard dim shifts from 0 to 1
        result = torch.stack([dt_flat, dt_flat], dim=0)
        expected = torch.stack([full.flatten(0, 1), full.flatten(0, 1)], dim=0)
        self.assertIsInstance(result.placements[0], _StridedShard)
        self.assertEqual(result.placements[0].dim, 1)
        self.assertEqual(result.placements[0].split_factor, orig_split_factor)
        self.assertEqual(result.full_tensor(), expected)

        # stack along dim=2 (after shard dim): _StridedShard dim stays at 0
        result2 = torch.stack([dt_flat, dt_flat], dim=2)
        expected2 = torch.stack([full.flatten(0, 1), full.flatten(0, 1)], dim=2)
        self.assertIsInstance(result2.placements[0], _StridedShard)
        self.assertEqual(result2.placements[0].dim, 0)
        self.assertEqual(result2.placements[0].split_factor, orig_split_factor)
        self.assertEqual(result2.full_tensor(), expected2)

    def test_flatten_then_slice(self):
        # [FUNCTION DOC] Slices a _StridedShard tensor to its first half along the strided dim
        # and to the first three columns along a non-sharded dim, asserting each full tensor
        # equals the same slice of the single-device flattened tensor.
        """Verify _StridedShard correctness through slice.

        aten.slice.Tensor goes through gen_slice_strategy which uses
        is_tensor_dim_sharded (misses _StridedShard). When the slice dim
        matches the _StridedShard dim, the strategy should redistribute
        to Replicate first, otherwise results are silently wrong.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)
        flat_full = full.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)

        # slice on the _StridedShard dim (dim 0)
        result = dt_flat[: dt_flat.shape[0] // 2]
        expected = flat_full[: flat_full.shape[0] // 2]
        self.assertEqual(result.full_tensor(), expected)

        # slice on non-shard dim
        result2 = dt_flat[:, :3]
        expected2 = flat_full[:, :3]
        self.assertEqual(result2.full_tensor(), expected2)

    def test_flatten_then_slice_scatter(self):
        # [FUNCTION DOC] Applies slice_scatter with a doubled slice as source on a
        # _StridedShard tensor, once along the strided dim (first half) and once along a
        # non-sharded dim (first three columns), comparing full tensors with the single-device
        # slice_scatter.
        """Verify _StridedShard correctness through slice_scatter.

        gen_slice_scatter_strategy uses replicate_tensor_dim which only
        checks isinstance(p, Shard), missing _StridedShard. When the
        scatter dim matches the _StridedShard dim and all strategies are
        filtered, replicate_tensor_dim should replicate it but doesn't.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)
        flat_full = full.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)

        # slice_scatter on the _StridedShard dim (dim 0)
        half = dt_flat.shape[0] // 2
        src = dt_flat[:half] * 2
        src_full = flat_full[:half] * 2
        result = dt_flat.slice_scatter(src, dim=0, start=0, end=half)
        expected = flat_full.slice_scatter(src_full, dim=0, start=0, end=half)
        self.assertEqual(result.full_tensor(), expected)

        # slice_scatter on non-shard dim
        src2 = dt_flat[:, :3] * 2
        src2_full = flat_full[:, :3] * 2
        result2 = dt_flat.slice_scatter(src2, dim=1, start=0, end=3)
        expected2 = flat_full.slice_scatter(src2_full, dim=1, start=0, end=3)
        self.assertEqual(result2.full_tensor(), expected2)

    def test_flatten_then_slice_backward(self):
        # [FUNCTION DOC] Runs backward on the sum of the first half slice along the strided dim
        # of a flattened DTensor requiring grad, and on the equivalent plain tensor, then asserts
        # the DTensor gradient's full tensor equals the flattened gradient of the plain tensor.
        """Verify _StridedShard correctness through slice backward.

        slice_backward_rules only checks isinstance(p, Shard), missing
        _StridedShard. When the slice dim matches the _StridedShard dim,
        gradients are computed on the wrong local shards.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full = torch.randn(*shape, device=self.device_type)
        full_req = full.clone().requires_grad_(True)

        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1).requires_grad_(True)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)

        flat_full = full_req.flatten(0, 1)
        half = dt_flat.shape[0] // 2

        # backward through slice on the _StridedShard dim
        (dt_flat[:half].sum()).backward()
        (flat_full[:half].sum()).backward()
        self.assertEqual(dt_flat.grad.full_tensor(), full_req.grad.flatten(0, 1))

    def test_flatten_then_cat_on_strided_shard_dim(self):
        # [FUNCTION DOC] Concatenates a _StridedShard tensor with itself on the strided dim
        # (dim 0), on a non-sharded dim, and for a _StridedShard(dim=1) tensor on that dim,
        # asserting full tensors equal the single-device cat.
        """Verify _StridedShard correctness through cat on the shard dim.

        cat_strategy uses is_tensor_dim_sharded which calls is_shard(),
        missing _StridedShard. When cat dim == _StridedShard dim, the
        strategy won't unshard first, producing wrong results.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)
        flat_full = full.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)
        self.assertEqual(dt_flat.placements[0].dim, 0)

        # cat on the _StridedShard dim (dim 0)
        result = torch.cat([dt_flat, dt_flat], dim=0)
        expected = torch.cat([flat_full, flat_full], dim=0)
        self.assertEqual(result.full_tensor(), expected)

        # cat on non-shard dim should work without redistribution
        result2 = torch.cat([dt_flat, dt_flat], dim=1)
        expected2 = torch.cat([flat_full, flat_full], dim=1)
        self.assertEqual(result2.full_tensor(), expected2)

        # _StridedShard on a higher dim (dim=1), cat on that dim
        shape2 = (3, 5, self.world_size * 2)
        full2 = torch.randn(*shape2, device=self.device_type)
        dt2 = distribute_tensor(full2, mesh, [Shard(2)])
        dt2_flat = dt2.flatten(1, 2)  # (3, 5*ws*2) with _StridedShard(dim=1)
        flat_full2 = full2.flatten(1, 2)

        self.assertIsInstance(dt2_flat.placements[0], _StridedShard)
        self.assertEqual(dt2_flat.placements[0].dim, 1)

        result3 = torch.cat([dt2_flat, dt2_flat], dim=1)
        expected3 = torch.cat([flat_full2, flat_full2], dim=1)
        self.assertEqual(result3.full_tensor(), expected3)

    def test_flatten_then_split_on_strided_shard_dim(self):
        # [FUNCTION DOC] Splits a _StridedShard tensor into half-size chunks along the strided
        # dim, into size-3 chunks along a non-sharded dim, and a _StridedShard(dim=1) tensor along
        # dim 1, comparing each chunk's full tensor with the single-device split.
        """Verify _StridedShard correctness through split on the shard dim.

        split_strategy uses is_tensor_dim_sharded which calls is_shard(),
        missing _StridedShard. When split dim == _StridedShard dim, the
        strategy won't unshard first, producing wrong results.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)
        flat_full = full.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)
        self.assertEqual(dt_flat.placements[0].dim, 0)

        # split on the _StridedShard dim (dim 0)
        half = dt_flat.shape[0] // 2
        results = torch.split(dt_flat, half, dim=0)
        expected = torch.split(flat_full, half, dim=0)
        for r, e in zip(results, expected):
            self.assertEqual(r.full_tensor(), e)

        # split on non-shard dim
        results2 = torch.split(dt_flat, 3, dim=1)
        expected2 = torch.split(flat_full, 3, dim=1)
        for r, e in zip(results2, expected2):
            self.assertEqual(r.full_tensor(), e)

        # _StridedShard on a higher dim (dim=1), split on that dim
        shape2 = (3, 5, self.world_size * 2)
        full2 = torch.randn(*shape2, device=self.device_type)
        dt2 = distribute_tensor(full2, mesh, [Shard(2)])
        dt2_flat = dt2.flatten(1, 2)  # (3, 5*ws*2) with _StridedShard(dim=1)
        flat_full2 = full2.flatten(1, 2)

        self.assertIsInstance(dt2_flat.placements[0], _StridedShard)
        self.assertEqual(dt2_flat.placements[0].dim, 1)

        half2 = dt2_flat.shape[1] // 2
        results3 = torch.split(dt2_flat, half2, dim=1)
        expected3 = torch.split(flat_full2, half2, dim=1)
        for r, e in zip(results3, expected3):
            self.assertEqual(r.full_tensor(), e)

    def test_flatten_then_unbind_on_strided_shard_dim(self):
        # [FUNCTION DOC] Asserts unbind along the _StridedShard dim (dim 0) of a flattened
        # tensor raises RuntimeError, matching the behavior for a plain Shard dim.
        """Verify _StridedShard is detected by unbind on the shard dim.

        gen_unbind_strategy uses is_tensor_dim_sharded which calls is_shard(),
        missing _StridedShard. Unbinding on a _StridedShard dim should raise
        RuntimeError (same as unbinding on a regular Shard dim).
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (2, self.world_size * 2, 3)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)
        self.assertEqual(dt_flat.placements[0].dim, 0)

        # unbind on the _StridedShard dim (dim 0) — should raise
        with self.assertRaises(RuntimeError):
            torch.unbind(dt_flat, dim=0)

    def test_flatten_then_add_strided_shard_inputs(self):
        # [FUNCTION DOC] Adds two independently flattened _StridedShard DTensors under
        # CommDebugMode and asserts the full result equals the single-device sum and that no
        # collective ran (no unnecessary all-gather).
        """Verify _StridedShard is treated as shard in placement merge.

        Binary ops (e.g. add) with two _StridedShard inputs go through
        _derive_follow_placements_from_tuple_strategy → merge_placement.
        merge_placement uses .is_shard() which misses _StridedShard,
        causing it to fall through to the Replicate branch and triggering
        an unnecessary all-gather before the elementwise op.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full_a = torch.randn(*shape, device=self.device_type)
        full_b = torch.randn(*shape, device=self.device_type)

        dt_a = distribute_tensor(full_a, mesh, [Shard(1)])
        dt_b = distribute_tensor(full_b, mesh, [Shard(1)])
        dt_a_flat = dt_a.flatten(0, 1)
        dt_b_flat = dt_b.flatten(0, 1)

        self.assertIsInstance(dt_a_flat.placements[0], _StridedShard)
        self.assertIsInstance(dt_b_flat.placements[0], _StridedShard)

        # add should work without redistribution
        comm_mode = CommDebugMode()
        with comm_mode:
            result = dt_a_flat + dt_b_flat

        expected = full_a.flatten(0, 1) + full_b.flatten(0, 1)
        self.assertEqual(result.full_tensor(), expected)
        # If merge_placement treats _StridedShard as Replicate, an
        # all-gather would be emitted here.
        self.assertEqual(comm_mode.get_total_counts(), 0)

    def test_flatten_then_redistribute_backward_partial(self):
        # [FUNCTION DOC] Redistributes a flattened _StridedShard DTensor requiring grad to
        # Replicate, sums, runs backward, and asserts the gradient's full tensor is all ones in
        # the flattened shape.
        """Verify backward normalize handles _StridedShard → Partial.

        In the backward pass of Redistribute, the code normalizes
        Shard/Replicate → Partial to just Replicate (to avoid a useless
        reduce). The check uses .is_shard() which misses _StridedShard,
        so _StridedShard → Partial would keep the Partial placement
        instead of normalizing to Replicate.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full = torch.randn(*shape, device=self.device_type)

        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1).requires_grad_(True)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)

        # redistribute _StridedShard → Replicate, then backward
        dt_rep = dt_flat.redistribute(mesh, [Replicate()])
        loss = dt_rep.sum()
        loss.backward()

        # grad should be all-ones with _StridedShard placement (same as input)
        expected_grad = torch.ones_like(full.flatten(0, 1))
        self.assertEqual(dt_flat.grad.full_tensor(), expected_grad)

    def test_unpack_hook_tp_with_strided_shard(self):
        # [FUNCTION DOC] Imports _unpack_hook_tp and calls it with the mesh, dim 0 and a
        # flattened _StridedShard DTensor. Asserts the result is a DTensor with a Replicate
        # placement whose full tensor equals the flattened input.
        """_unpack_hook_tp must redistribute _StridedShard to Replicate.

        _unpack_hook_tp restores activations before recomputation in BWD.
        If it only checks .is_shard() it misses _StridedShard, returning the
        tensor still sharded and producing wrong gradients.
        """
        from torch.distributed.tensor.parallel.input_reshard import _unpack_hook_tp

        mesh = init_device_mesh(self.device_type, (self.world_size,))
        shape = (4, self.world_size * 2, 6)
        full = torch.randn(*shape, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])
        dt_flat = dt.flatten(0, 1)

        self.assertIsInstance(dt_flat.placements[0], _StridedShard)

        result = _unpack_hook_tp(mesh, 0, dt_flat)
        self.assertIsInstance(result, DTensor)
        self.assertTrue(result.placements[0].is_replicate())
        self.assertEqual(result.full_tensor(), full.flatten(0, 1))

    def test_view_redistribution(self):
        # [FUNCTION DOC] Asserts that viewing a (4, 4) DTensor sharded on dim 0 over the world
        # size as (-1, 8) raises RuntimeError "Sharding propagation failed" instead of
        # redistributing.
        """
        This test is added to demonstrate "incorrect" view ops behavior if redistribution happens.
        """

        x = torch.randn(4, 4)
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        dtensor_x = distribute_tensor(x, mesh, (Shard(0),))

        with self.assertRaisesRegex(RuntimeError, "Sharding propagation failed"):
            dtensor_x.view(-1, 8)

    def test_squeeze_(self):
        # [FUNCTION DOC] On a (3, 2) mesh builds a (1, 4) local tensor per rank as
        # [Partial, Shard(1)], applies in-place squeeze_ on dim 0 through the base class helper
        # _test_op_on_dtensor, and asserts the DTensor becomes shape (8,), stride (1,) and
        # placements [Partial, Shard(0)]. A second in-place squeeze_ asserts no communication.
        mesh_2d = init_device_mesh(self.device_type, (3, 2), mesh_dim_names=("a", "b"))
        self.init_manual_seed_for_rank()
        x = torch.randn((1, 4), device=self.device_type)
        dist_x = DTensor.from_local(x, mesh_2d, [Partial(), Shard(1)])
        self._test_op_on_dtensor(
            torch.ops.aten.squeeze_.dim,
            dist_x,
            0,
        )
        # check DTensor subclass metadata as well as placements
        self.assertEqual(dist_x.shape, torch.Size([8]))
        self.assertEqual(
            dist_x.stride(),
            (1,),
        )
        self.assertEqual(dist_x.placements, [Partial(), Shard(0)])

        # squeeze_ should not trigger any communication
        y = torch.randn((1, 4), device=self.device_type)
        dist_y = DTensor.from_local(y, mesh_2d, [Partial(), Shard(1)])
        with CommDebugMode() as comm_mode:
            torch.ops.aten.squeeze_.dim(dist_y, 0)
        self.assertEqual(comm_mode.get_total_counts(), 0)

    def test_squeeze_variants(self):
        # [FUNCTION DOC] Table of subTests for squeeze overloads (default, dim, dims, in-place)
        # on a 1D mesh, each asserting zero communication and the resulting shape and placements:
        # squeezing unsharded singleton dims, replicated and Partial inputs, a non-singleton
        # dim as a no-op, never removing a sharded dim whose local size is 1, shifting the shard
        # index, and that squeezing a sharded singleton dim raises RuntimeError "requires
        # redistribution" for all six ops.
        """Test squeeze.default, squeeze.dim, and squeeze.dims with DTensor."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # squeeze.dims on sharded tensor - squeeze non-sharded dims
        with self.subTest("dims_sharded"):
            x = torch.randn(self.world_size, 1, 1, 8, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Shard(0)])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze((1, 2))
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([self.world_size, 8]))
            self.assertEqual(result.placements, (Shard(0),))
            self.assertEqual(result.to_local().shape, torch.Size([1, 8]))

        # squeeze.dim on sharded tensor - squeeze non-sharded dim
        with self.subTest("dim_sharded"):
            x = torch.randn(self.world_size, 1, 8, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Shard(0)])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze(1)
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([self.world_size, 8]))
            self.assertEqual(result.placements, (Shard(0),))
            self.assertEqual(result.to_local().shape, torch.Size([1, 8]))

        # squeeze.default on replicated tensor
        with self.subTest("default_replicated"):
            x = torch.randn(4, 1, 1, 8, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Replicate()])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze()
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([4, 8]))
            self.assertEqual(result.placements, (Replicate(),))

        # squeeze.dims on replicated tensor
        with self.subTest("dims_replicated"):
            x = torch.randn(2, 1, 3, 1, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Replicate()])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze((1, 3))
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([2, 3]))
            self.assertEqual(result.placements, (Replicate(),))

        # squeeze non-singleton dim is a no-op
        with self.subTest("non_singleton_noop"):
            x = torch.randn(self.world_size, 4, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Shard(0)])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze(1)
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([self.world_size, 4]))
            self.assertEqual(result.placements, (Shard(0),))

        # Partial passes through squeeze unchanged
        with self.subTest("partial_max_passthrough"):
            x = torch.randn(1, 4, device=self.device_type)
            dt = DTensor.from_local(x, mesh, [Partial("max")])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze(0)
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([4]))
            self.assertEqual(result.placements, (Partial("max"),))

        # Partial("sum") also passes through
        with self.subTest("partial_sum_passthrough"):
            x = torch.randn(1, device=self.device_type)
            dt = DTensor.from_local(x, mesh, [Partial("sum")])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze()
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([]))
            self.assertEqual(result.placements, (Partial("sum"),))

        # squeeze must not remove sharded dim (local size 1, global size > 1)
        with self.subTest("preserve_sharded_dim_default"):
            x = (
                torch.arange(self.world_size * 8, device=self.device_type)
                .reshape(self.world_size, 8)
                .float()
            )
            dt = distribute_tensor(x, mesh, [Shard(0)])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze()
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([self.world_size, 8]))
            self.assertEqual(result._local_tensor.shape, torch.Size([1, 8]))
            self.assertEqual(result.placements, (Shard(0),))
            self.assertEqual(result.full_tensor(), x)

        # same as above but via squeeze.dim (single int arg)
        with self.subTest("preserve_sharded_dim_explicit"):
            x = (
                torch.arange(self.world_size * 8, device=self.device_type)
                .reshape(self.world_size, 8)
                .float()
            )
            dt = distribute_tensor(x, mesh, [Shard(0)])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze(0)
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([self.world_size, 8]))
            self.assertEqual(result._local_tensor.shape, torch.Size([1, 8]))
            self.assertEqual(result.placements, (Shard(0),))
            self.assertEqual(result.full_tensor(), x)

        # squeeze.dims with mixed singleton/non-singleton dims
        with self.subTest("mixed_dims"):
            x = torch.randn(1, 4, 1, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Replicate()])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze((0, 1, 2))
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([4]))
            self.assertEqual(result.placements, (Replicate(),))
            self.assertEqual(result.full_tensor(), x.squeeze())

        # sharded non-singleton dim preserved, shard index shifts
        with self.subTest("shard_index_shift"):
            x = torch.randn(1, self.world_size, 1, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Shard(1)])
            with CommDebugMode() as comm_mode:
                result = dt.squeeze((0, 1, 2))
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.shape, torch.Size([self.world_size]))
            self.assertEqual(result._local_tensor.shape, torch.Size([1]))
            self.assertEqual(result.placements, (Shard(0),))
            self.assertEqual(result.full_tensor(), x.squeeze((0, 2)))

        # Squeezing a sharded singleton dim requires redistribution and
        # must error rather than silently allgathering (#174136).
        with self.subTest("squeeze_sharded_dim_errors"):
            x = torch.randn(1, 4, device=self.device_type)
            dt = distribute_tensor(x, mesh, [Shard(0)])
            # All 6 squeeze ATen ops must error:
            for op, args in [
                (dt.squeeze, ()),  # squeeze.default
                (dt.squeeze, (0,)),  # squeeze.dim
                (dt.squeeze, ((0,),)),  # squeeze.dims
                (dt.squeeze_, ()),  # squeeze_.default
                (dt.squeeze_, (0,)),  # squeeze_.dim
                (dt.squeeze_, ((0,),)),  # squeeze_.dims
            ]:
                with self.assertRaisesRegex(RuntimeError, "requires redistribution"):
                    op(*args)

    def test_squeeze_comm_free_cases(self):
        # [FUNCTION DOC] Checks squeeze needs no communication when no sharded dim is removed:
        # out-of-place and in-place squeeze of an unsharded leading singleton with the shard dim
        # reindexed, squeeze after explicit redistribute to Replicate, and a (dp, tp) 2D mesh
        # where both Shard dims reindex. Asserts shape, placements and full tensor.
        """Squeeze is comm-free when no sharded dim is removed."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Non-sharded singleton removed (out-of-place), shard dim reindexes
        x = torch.randn(1, self.world_size, device=self.device_type)
        dt = distribute_tensor(x, mesh, [Shard(1)])
        with CommDebugMode() as comm_mode:
            result = dt.squeeze(0)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(result.shape, torch.Size([self.world_size]))
        self.assertEqual(result.placements, (Shard(0),))
        self.assertEqual(result.full_tensor(), x.squeeze(0))

        # Same but inplace — exercises the dispatch reindex path
        x_inp = torch.randn(1, self.world_size, device=self.device_type)
        dt_inp = distribute_tensor(x_inp, mesh, [Shard(1)])
        with CommDebugMode() as comm_mode:
            dt_inp.squeeze_(0)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(dt_inp.shape, torch.Size([self.world_size]))
        self.assertEqual(dt_inp.placements, (Shard(0),))
        self.assertEqual(dt_inp.full_tensor(), x_inp.squeeze(0))

        # Explicit redistribute first, then squeeze
        x2 = torch.randn(1, 4, device=self.device_type)
        dt2 = distribute_tensor(x2, mesh, [Shard(0)])
        dt2_rep = dt2.redistribute(mesh, [Replicate()])
        with CommDebugMode() as comm_mode:
            result2 = dt2_rep.squeeze(0)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(result2.shape, torch.Size([4]))
        self.assertEqual(result2.placements, (Replicate(),))
        self.assertEqual(result2.full_tensor(), x2.squeeze(0))

        # 2D mesh: both shard dims reindex
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        x3 = torch.randn(1, 2, self.world_size // 2, device=self.device_type)
        dt3 = distribute_tensor(x3, mesh_2d, [Shard(1), Shard(2)])
        with CommDebugMode() as comm_mode:
            result3 = dt3.squeeze(0)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(result3.shape, torch.Size([2, self.world_size // 2]))
        self.assertEqual(result3.placements, (Shard(0), Shard(1)))
        self.assertEqual(result3.full_tensor(), x3.squeeze(0))

    def test_storage_offset_slice(self):
        # [FUNCTION DOC] Slices a replicated 10-element DTensor from index 1 with no
        # communication and asserts the DTensor and its local tensor both report storage offset
        # 1, shape (9,), and the expected values.
        """
        Test that storage_offset is properly tracked on DTensor when slicing
        a replicated tensor.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create a replicated DTensor
        tensor = torch.randn(10, device=self.device_type)
        dtensor = distribute_tensor(tensor, mesh, [Replicate()])

        # Perform a slice operation [1:]
        with CommDebugMode() as comm_mode:
            sliced_dtensor = dtensor[1:]
            # Slicing should not trigger any communication
            self.assertEqual(comm_mode.get_total_counts(), 0)

        # Verify that the DTensor's storage_offset matches the expected value
        self.assertEqual(sliced_dtensor.storage_offset(), 1)

        # Verify that the local tensor also has the correct storage_offset
        self.assertEqual(sliced_dtensor.to_local().storage_offset(), 1)

        # Verify the shape is correct
        self.assertEqual(sliced_dtensor.shape, torch.Size([9]))

        # Verify the values are correct
        expected = tensor[1:]
        self.assertEqual(sliced_dtensor.full_tensor(), expected)

    def test_storage_offset_shard_dim0_slice_dim1(self):
        # [FUNCTION DOC] Slices columns from index 2 of a (12, 8) tensor sharded on dim 0, with
        # no communication, and asserts storage offset 2 on both the DTensor and its local
        # tensor, shape (12, 6), and matching values.
        """
        Test that storage_offset is properly tracked when tensor is sharded on dim 0
        and sliced on dim 1.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create a 2D tensor and shard on dim 0
        tensor = torch.randn(12, 8, device=self.device_type)
        dtensor = distribute_tensor(tensor, mesh, [Shard(0)])

        # Perform a slice operation [:, 2:]
        with CommDebugMode() as comm_mode:
            sliced_dtensor = dtensor[:, 2:]
            # Slicing should not trigger any communication
            self.assertEqual(comm_mode.get_total_counts(), 0)

        # The storage_offset should be 2 (skipping 2 elements in each row)
        self.assertEqual(sliced_dtensor.storage_offset(), 2)

        # Verify that the local tensor also has the correct storage_offset
        self.assertEqual(sliced_dtensor.to_local().storage_offset(), 2)

        # Verify the shape is correct
        expected_shape = torch.Size([12, 6])
        self.assertEqual(sliced_dtensor.shape, expected_shape)

        # Verify the values are correct
        expected = tensor[:, 2:]
        self.assertEqual(sliced_dtensor.full_tensor(), expected)

    def test_storage_offset_shard_dim1_slice_dim0(self):
        # [FUNCTION DOC] Slices rows from index 2 of a (10, 12) tensor sharded on dim 1, with
        # no communication, and asserts storage offset equals 2 times the local dim 1 size on
        # the DTensor and its local tensor, shape (8, 12), and matching values.
        """
        Test that storage_offset is properly tracked when tensor is sharded on dim 1
        and sliced on dim 0.
        """
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create a 2D tensor and shard on dim 1
        tensor = torch.randn(10, 12, device=self.device_type)
        dtensor = distribute_tensor(tensor, mesh, [Shard(1)])

        # Perform a slice operation [2:, :]
        with CommDebugMode() as comm_mode:
            sliced_dtensor = dtensor[2:, :]
            # Slicing should not trigger any communication
            self.assertEqual(comm_mode.get_total_counts(), 0)

        local_dim1_size = 12 // self.world_size
        expected_offset = 2 * local_dim1_size
        self.assertEqual(sliced_dtensor.storage_offset(), expected_offset)

        self.assertEqual(sliced_dtensor.to_local().storage_offset(), expected_offset)

        # Verify the shape is correct
        expected_shape = torch.Size([8, 12])
        self.assertEqual(sliced_dtensor.shape, expected_shape)

        # Verify the values are correct
        expected = tensor[2:, :]
        self.assertEqual(sliced_dtensor.full_tensor(), expected)

    def test_view_groups_unbacked_symint(self):
        # [FUNCTION DOC] Checks view_groups with unbacked symbolic ints from a ShapeEnv: the same
        # symbol forwards as InputDim, splits and flattens with concrete sizes work, size-1 dims
        # are dropped, a product of two symbols flattens, distinct symbols map one to one, and a
        # split by a concrete prefix does not need unbacked divisibility guards.
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()

        def fresh_sym():
            # [FUNCTION DOC] HELPER: Returns a new unbacked symbolic int from the shape_env.
            return shape_env.create_unbacked_symint()

        # Same symbol forwards correctly
        u0 = fresh_sym()
        self.assertEqual(
            view_groups([4, u0, 3], [4, u0, 3]),
            (InputDim(0), InputDim(1), InputDim(2)),
        )

        # Same symbol with concrete split
        u1 = fresh_sym()
        self.assertEqual(
            view_groups([u1, 12], [u1, 3, 4]),
            (
                InputDim(0),
                Split(InputDim(1), (3, 4), 0),
                Split(InputDim(1), (3, 4), 1),
            ),
        )

        # Same symbol with concrete flatten
        u2 = fresh_sym()
        self.assertEqual(
            view_groups([u2, 3, 4], [u2, 12]),
            (InputDim(0), Flatten((InputDim(1), InputDim(2)))),
        )

        # Concrete split with trailing symbol
        u3 = fresh_sym()
        self.assertEqual(
            view_groups([6, u3], [2, 3, u3]),
            (
                Split(InputDim(0), (2, 3), 0),
                Split(InputDim(0), (2, 3), 1),
                InputDim(1),
            ),
        )

        # Singletons with symbolic dims
        u4 = fresh_sym()
        self.assertEqual(
            view_groups([1, 1, u4, 3], [u4, 3]),
            (InputDim(2), InputDim(3)),
        )

        # Flatten symbolic dims using same product expression
        u5 = fresh_sym()
        u6 = fresh_sym()
        self.assertEqual(
            view_groups([u5, u6], [u5 * u6]),
            (Flatten((InputDim(0), InputDim(1))),),
        )

        # Partial fallback: concrete prefix resolves, symbolic suffix falls back
        u7 = fresh_sym()
        u8 = fresh_sym()
        result = view_groups([4, u7, u8], [4, u7 * u8])
        # dim 0 resolves as InputDim(0), dims 1-2 flatten
        self.assertEqual(
            result,
            (InputDim(0), Flatten((InputDim(1), InputDim(2)))),
        )

        # Different unbacked symbols (1-to-1): fallback simplifies to InputDim
        u9 = fresh_sym()
        u10 = fresh_sym()
        result = view_groups([4, u9], [4, u10])
        self.assertEqual(result, (InputDim(0), InputDim(1)))

        # Splitting a product by a concrete prefix should not guard on
        # unbacked divisibility such as ``8 % (8 * u0) == 0``.
        u11 = fresh_sym()
        self.assertEqual(
            view_groups([8 * u11, 256], [8, u11, 256]),
            (
                Split(InputDim(0), (8, u11), 0),
                Split(InputDim(0), (8, u11), 1),
                InputDim(1),
            ),
        )

    def test_view_groups_unbacked_sharding_propagation(self):
        # [FUNCTION DOC] Checks propagate_shape_and_sharding with symbolic shapes on mesh sizes
        # (2, 3): shards forward through identical, split and flatten rules; a strict view
        # with non-leftmost shard yields _StridedShard whose symbolic split_factor keeps the
        # symbol and its overridden hint (4), raising RuntimeError "symbolic _StridedShard" when
        # no hint exists; unflattening a symbolic _StridedShard gives Shard(1); and a
        # non-leftmost shard under a flatten falls back to Replicate.
        """Test that sharding is correctly propagated through view_groups with symbolic shapes."""
        from torch.fx.experimental.symbolic_shapes import (
            free_symbols,
            optimization_hint,
            ShapeEnv,
        )

        shape_env = ShapeEnv()

        def fresh_sym():
            # [FUNCTION DOC] HELPER: Returns a new unbacked symbolic int from the shape_env.
            return shape_env.create_unbacked_symint()

        mesh_sizes = (2, 3)

        # InputDim forwarding: Shard(1) on symbolic dim should propagate
        u0 = fresh_sym()
        from_shape = (4, u0, 6)
        to_shape = (4, u0, 6)
        rule = view_groups(from_shape, to_shape)
        input_placements = [Shard(1), Replicate()]
        inp_tgt, out_plc = propagate_shape_and_sharding(
            input_placements, from_shape, rule, mesh_sizes
        )
        self.assertEqual(out_plc, [Shard(1), Replicate()])

        # Same symbol with concrete split: Shard(0) on symbolic dim should forward
        u1 = fresh_sym()
        from_shape = (u1, 12)
        to_shape = (u1, 3, 4)
        rule = view_groups(from_shape, to_shape)
        input_placements = [Shard(0), Replicate()]
        inp_tgt, out_plc = propagate_shape_and_sharding(
            input_placements, from_shape, rule, mesh_sizes
        )
        self.assertEqual(out_plc, [Shard(0), Replicate()])

        # Shard on concrete dim that gets split: should propagate to split_id=0
        u2 = fresh_sym()
        from_shape = (6, u2)
        to_shape = (2, 3, u2)
        rule = view_groups(from_shape, to_shape)
        input_placements = [Shard(0), Replicate()]
        inp_tgt, out_plc = propagate_shape_and_sharding(
            input_placements, from_shape, rule, mesh_sizes
        )
        self.assertEqual(out_plc, [Shard(0), Replicate()])

        # Flatten [u, 16] -> [u*16] with Shard(0) on unbacked dim (leftmost)
        u3 = fresh_sym()
        for ms in mesh_sizes:  # needed for sharding
            torch._check(u3 % ms == 0)
        from_shape = (u3, 16)
        to_shape = (u3 * 16,)
        rule = view_groups(from_shape, to_shape)
        input_placements = [Shard(0), Replicate()]
        inp_tgt, out_plc = propagate_shape_and_sharding(
            input_placements, from_shape, rule, mesh_sizes
        )
        self.assertEqual(out_plc, [Shard(0), Replicate()])

        # Strict view with a non-leftmost shard needs _StridedShard. Preserve
        # the symbolic split_factor as semantic placement metadata; list-based
        # lowering can still use the explicit hint as a guarded unroll count.
        u4_hint = fresh_sym()
        torch._dynamo.override_optimization_hint(u4_hint, 4)
        from_shape = (u4_hint, 16, 8)
        to_shape = (u4_hint * 16, 8)
        rule = view_groups(from_shape, to_shape)
        input_placements = [Shard(1), Replicate()]
        inp_tgt, out_plc = propagate_shape_and_sharding(
            input_placements, from_shape, rule, mesh_sizes, strict_view=True
        )
        self.assertIsInstance(out_plc[0], _StridedShard)
        self.assertEqual(out_plc[0].dim, 0)
        self.assertEqual(out_plc[1], Replicate())
        self.assertEqual(free_symbols(out_plc[0].split_factor), free_symbols(u4_hint))
        self.assertEqual(optimization_hint(out_plc[0].split_factor), 4)
        self.assertTrue(free_symbols(u4_hint))

        u4_unhinted = fresh_sym()
        from_shape = (u4_unhinted, 16, 8)
        to_shape = (u4_unhinted * 16, 8)
        rule = view_groups(from_shape, to_shape)
        with self.assertRaisesRegex(RuntimeError, "symbolic _StridedShard"):
            propagate_shape_and_sharding(
                input_placements, from_shape, rule, mesh_sizes, strict_view=True
            )

        # Unflattening a symbolic _StridedShard should match the Split prefix
        # factor by guarded hint without specializing the symbol.
        u5_hint = fresh_sym()
        torch._dynamo.override_optimization_hint(u5_hint, 8)
        from_shape = (u5_hint * 2048, 256)
        to_shape = (u5_hint, 2048, 256)
        rule = view_groups(from_shape, to_shape)
        input_placements = [_StridedShard(0, split_factor=u5_hint), Replicate()]
        inp_tgt, out_plc = propagate_shape_and_sharding(
            input_placements, from_shape, rule, mesh_sizes, strict_view=True
        )
        self.assertEqual(out_plc, [Shard(1), Replicate()])
        self.assertTrue(free_symbols(u5_hint))

        # Flatten [16, u] -> [16*u] with Shard(1) on unbacked dim (non-leftmost):
        # should force replicate since non-leftmost dims can't propagate through flatten
        u4 = fresh_sym()
        from_shape = (16, u4)
        to_shape = (16 * u4,)
        rule = view_groups(from_shape, to_shape)
        input_placements = [Shard(1), Replicate()]
        inp_tgt, out_plc = propagate_shape_and_sharding(
            input_placements, from_shape, rule, mesh_sizes
        )
        self.assertEqual(out_plc, [Replicate(), Replicate()])

    def test_input_dim_rejects_int_comparison(self):
        # [FUNCTION DOC] Verifies InputDim equality and hash: equal to another InputDim of the
        # same index, unequal otherwise, hash matches for equal values and differs from the raw
        # int hash, and comparing with an int in either order raises TypeError mentioning
        # .input_dim.
        """InputDim.__eq__ should raise TypeError when compared with int.

        Regression guard: shard.dim == in_dim (int == InputDim) silently
        returned False for over 3 years, making downstream validation dead code.
        """
        dim = InputDim(0)
        # InputDim == InputDim is fine
        self.assertEqual(dim, InputDim(0))
        self.assertNotEqual(dim, InputDim(1))
        # hash contract: equal InputDims must have equal hashes
        self.assertEqual(hash(dim), hash(InputDim(0)))
        # hash is salted so it won't collide with raw int in dicts/sets
        self.assertNotEqual(hash(dim), hash(0))
        # InputDim == int raises (catches `dim == some_int`)
        with self.assertRaisesRegex(TypeError, "Did you mean to use .input_dim"):
            _ = dim == 0
        # int == InputDim also raises (catches the original bug: `shard.dim == in_dim`)
        with self.assertRaisesRegex(TypeError, "Did you mean to use .input_dim"):
            _ = 0 == dim

    def _assert_strided_shard_flag(self, dt, expected_flag):
        # [FUNCTION DOC] HELPER: Asserts the DTensor spec attribute
        # use_strided_shard_as_shard_order of dt equals expected_flag.
        """Assert use_strided_shard_as_shard_order matches expected_flag."""
        self.assertEqual(dt._spec.use_strided_shard_as_shard_order, expected_flag)

    def test_strided_shard_propagates_through_chained_ops(self):
        # [FUNCTION DOC] Seeds the RNG, flattens a Shard(1) tensor into _StridedShard, applies
        # relu, add, mul and abs, asserting the use_strided_shard_as_shard_order flag stays False
        # at each step and the final full tensor equals the single-device chain.
        """Verify use_strided_shard_as_shard_order=False propagates through
        chained pointwise ops after flatten produces _StridedShard."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        torch.manual_seed(42)
        B, num_heads, S, head_dim = 2, 6, 4, 8
        full = torch.randn(B, num_heads, S, head_dim, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])

        # flatten(0,1): Shard(1) on non-first flatten dim -> _StridedShard(0)
        flat = dt.flatten(0, 1)
        self.assertIsInstance(flat.placements[0], _StridedShard)
        self._assert_strided_shard_flag(flat, False)

        # Chain several pointwise ops and verify flag propagates
        activated = flat.relu()
        self._assert_strided_shard_flag(activated, False)

        added = activated + 1.0
        self._assert_strided_shard_flag(added, False)

        scaled = added * 0.5
        self._assert_strided_shard_flag(scaled, False)

        result = scaled.abs()
        self._assert_strided_shard_flag(result, False)

        # full_tensor triggers _StridedShard -> Replicate redistribution
        expected = full.flatten(0, 1).relu().add(1.0).mul(0.5).abs()
        self.assertEqual(result.full_tensor(), expected)

    def test_strided_shard_propagates_through_reduction(self):
        # [FUNCTION DOC] Flattens a Shard(1) tensor to _StridedShard, sums over the last dim,
        # asserts the use_strided_shard_as_shard_order flag stays False and the full result
        # equals the single-device sum.
        """Verify use_strided_shard_as_shard_order=False propagates through
        reduction ops that don't reduce the _StridedShard dim."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        torch.manual_seed(42)
        # shape: (2, 6, 4, 8), shard on dim 1
        B, num_heads, S, head_dim = 2, 6, 4, 8
        full = torch.randn(B, num_heads, S, head_dim, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])

        # flatten(0,1) -> (12, 4, 8) with _StridedShard(0)
        flat = dt.flatten(0, 1)
        self.assertIsInstance(flat.placements[0], _StridedShard)
        self._assert_strided_shard_flag(flat, False)

        # sum over dim -1 (head_dim), which is not the strided shard dim
        reduced = flat.sum(dim=-1)
        self._assert_strided_shard_flag(reduced, False)

        expected = full.flatten(0, 1).sum(dim=-1)
        self.assertEqual(reduced.full_tensor(), expected)

    def test_strided_shard_propagates_through_matmul(self):
        # [FUNCTION DOC] Flattens a (2, 6, 4, 8) Shard(1) tensor into _StridedShard, asserts the
        # use_strided_shard_as_shard_order flag is False, multiplies by a replicated (8, 16)
        # weight with matmul, asserts the flag is still False on the result, and the full tensor
        # equals the single-device matmul.
        """Verify use_strided_shard_as_shard_order=False propagates when a
        _StridedShard tensor participates in matmul."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        torch.manual_seed(42)
        B, num_heads, S, head_dim = 2, 6, 4, 8
        full = torch.randn(B, num_heads, S, head_dim, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(1)])

        # flatten(0,1) -> (12, 4, 8) with _StridedShard(0)
        flat = dt.flatten(0, 1)
        self.assertIsInstance(flat.placements[0], _StridedShard)
        self._assert_strided_shard_flag(flat, False)

        # matmul with a replicated weight: (12, 4, 8) @ (8, 16) -> (12, 4, 16)
        weight = torch.randn(head_dim, 16, device=self.device_type)
        weight_dt = distribute_tensor(weight, mesh, [Replicate()])
        result = torch.matmul(flat, weight_dt)
        self._assert_strided_shard_flag(result, False)

        expected = torch.matmul(full.flatten(0, 1), weight)
        self.assertEqual(result.full_tensor(), expected)

    def test_strided_shard_propagates_2d_mesh(self):
        # [FUNCTION DOC] On a (world_size/2, 2) mesh, flattens a (3, 4, 8) tensor placed as
        # Shard(0), Replicate, asserts the use_strided_shard_as_shard_order flag is False after
        # flatten and after relu, and the full result equals the single-device flatten and relu.
        """Verify use_strided_shard_as_shard_order=False propagates on a 2D mesh."""
        mesh = init_device_mesh(self.device_type, (self.world_size // 2, 2))
        torch.manual_seed(42)
        # shape: (3, 4, 8), shard dim 0 on mesh dim 0, replicate on mesh dim 1
        full = torch.randn(3, 4, 8, device=self.device_type)
        dt = distribute_tensor(full, mesh, [Shard(0), Replicate()])

        # flatten(0,1) -> (12, 8); Shard(0) stays Shard(0), Replicate stays
        flat = dt.flatten(0, 1)
        self._assert_strided_shard_flag(flat, False)

        result = flat.relu()
        self._assert_strided_shard_flag(result, False)

        expected = full.flatten(0, 1).relu()
        self.assertEqual(result.full_tensor(), expected)


TestViewOpsWithLocalTensor = create_local_tensor_test_class(
    TestViewOps,
    skipped_tests=[
        # Comparing data pointers is not supported for local tensor
        "test_dtensor_view_op_uneven",
        # These tests use ShapeEnv directly, not local tensor tests
        "test_view_groups_unbacked_symint",
        "test_view_groups_unbacked_sharding_propagation",
        # Too many test cases for LocalTensorMode dispatch overhead
        "test_dtensor_flatten_1d",
        "test_dtensor_flatten_2d",
        "test_dtensor_flatten_multi_mesh",
        "test_dtensor_flatten_split_multi_mesh",
    ],
    base_class=LocalDTensorContinuousTestBase,
)


if __name__ == "__main__":
    run_tests()
