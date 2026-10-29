# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for DTensor placement types (Shard, _StridedShard, Partial, Replicate):
# type predicates, equality, constructor rules, tensor splitting helpers and behavior with
# unbacked symbolic sizes under a fake tensor mode.
import copy
import itertools

import sympy

import torch
from torch._subclasses import FakeTensorMode
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._ops.utils import is_tensor_shardable
from torch.distributed.tensor.placement_types import (
    _hint_proves_even_shard,
    _StridedShard,
    _StridedShardOffsetMode,
    Partial,
    Replicate,
    Shard,
)
from torch.fx.experimental.symbolic_shapes import (
    free_symbols,
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
    optimization_hint,
    ShapeEnv,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# Basic functionality test for Placement types.
class PlacementTypesTestCase(TestCase):
    # [CLASS DOC] Test case covering basic placement behavior and symbolic-shape handling.
    def test_type_identification(self):
        # [FUNCTION DOC] Checks is_shard, is_partial and is_replicate for each placement kind,
        # before and after a deep copy; _StridedShard reports none of the three.
        shard = Shard(3)
        strided_shard = _StridedShard(dim=3, split_factor=7)
        partial_sum = Partial("sum")
        partial_max = Partial("max")
        replicate = Replicate()

        ident_tests = (
            (shard, True, False, False),
            (strided_shard, False, False, False),
            (partial_sum, False, True, False),
            (partial_max, False, True, False),
            (replicate, False, False, True),
        )
        for do_deepcopy in (False, True):
            for placement, is_shard, is_partial, is_replicate in ident_tests:
                if do_deepcopy:
                    placement = copy.deepcopy(placement)
                self.assertEqual(placement.is_shard(), is_shard)
                self.assertEqual(placement.is_partial(), is_partial)
                self.assertEqual(placement.is_replicate(), is_replicate)

    def test_equality(self):
        # [FUNCTION DOC] Checks each placement equals itself and differs from placements of other
        # kinds, dims, split factors or reduce ops (each equivalence class has one member).
        equivalence_classes = (
            (Shard(3),),
            (Shard(4),),
            (_StridedShard(dim=3, split_factor=1),),
            (_StridedShard(dim=3, split_factor=2),),
            (_StridedShard(dim=4, split_factor=9),),
            (Replicate(),),
            (Partial("sum"),),
            (Partial("max"),),
        )
        for eq_class in equivalence_classes:
            # Each item in the equivalence class should be equal to every other item in
            # its class.
            for lhs, rhs in itertools.product(eq_class, eq_class):
                self.assertEqual(lhs, rhs)

            # Each item in the equivalence class should not be equal to any item in any
            # other class.
            for other_class in equivalence_classes:
                if other_class is eq_class:
                    continue
                for lhs, rhs in itertools.product(eq_class, other_class):
                    self.assertNotEqual(lhs, rhs)

    def test_strided_shard_kwonly_argument(self):
        # [FUNCTION DOC] Checks _StridedShard rejects a positional split factor with TypeError
        # while the keyword form is accepted.
        with self.assertRaises(TypeError):
            _StridedShard(3, 4)
        _StridedShard(3, split_factor=4)

    def test_select_split_tensor_matches_split_tensor(self):
        # [FUNCTION DOC] For dims 0 and 1 and several size and chunk-count combinations (even,
        # uneven, fewer elements than chunks), checks _select_split_tensor for each index returns a
        # tensor equal to the corresponding entry of _split_tensor without padding.
        """
        Test that _select_split_tensor produces the same result as indexing
        into _split_tensor. This validates that any alternative implementation
        (e.g., the narrow-based SymInt path) matches the canonical _split_tensor.
        """
        # Test various tensor sizes and num_chunks combinations
        test_cases = [
            # (dim_size, num_chunks) - covers even splits, uneven splits, edge cases
            (16, 4),  # even split
            (17, 4),  # uneven split, last chunk smaller
            (15, 4),  # uneven split
            (7, 4),  # fewer elements than chunks would like
            (3, 4),  # very few elements
            (1, 4),  # single element
            (8, 1),  # single chunk
            (8, 8),  # one element per chunk
            (8, 16),  # more chunks than elements
        ]

        for dim in [0, 1]:
            shard = Shard(dim)
            for dim_size, num_chunks in test_cases:
                # Create a tensor with distinct values for easy debugging
                if dim == 0:
                    tensor = torch.arange(dim_size * 4).reshape(dim_size, 4)
                else:
                    tensor = torch.arange(4 * dim_size).reshape(4, dim_size)

                # Get ground truth from _split_tensor
                shards, _ = shard._split_tensor(
                    tensor, num_chunks, with_padding=False, contiguous=False
                )

                # Compare _select_split_tensor against _split_tensor for each index
                for idx in range(num_chunks):
                    selected = shard._select_split_tensor(
                        tensor,
                        num_chunks,
                        idx,
                        with_padding=False,
                        contiguous=False,
                        clone=False,
                    )
                    self.assertEqual(
                        selected,
                        shards[idx],
                        msg=lambda msg: f"{msg}\nMismatch for dim={dim}, dim_size={dim_size}, "
                        f"num_chunks={num_chunks}, idx={idx}",
                    )

    def test_select_split_tensor_symint_with_padding_raises(self):
        # [FUNCTION DOC] Checks _select_split_tensor with an unbacked SymInt index and padding
        # enabled raises GuardOnDataDependentSymNode.
        """
        Test that _select_split_tensor raises GuardOnDataDependentSymNode when
        called with a SymInt index and with_padding=True.

        This is expected because with_padding=True requires indexing into a
        Python list with the SymInt, which is not supported.
        """
        from torch.fx.experimental.symbolic_shapes import (
            GuardOnDataDependentSymNode,
            ShapeEnv,
        )

        shape_env = ShapeEnv()
        symint_index = shape_env.create_unbacked_symint()

        shard = Shard(0)
        tensor = torch.arange(16).reshape(4, 4)

        with self.assertRaises(GuardOnDataDependentSymNode):
            shard._select_split_tensor(
                tensor,
                num_chunks=4,
                index=symint_index,
                with_padding=True,
            )

    def test_hinted_unbacked_even_shard_skips_padding(self):
        # [FUNCTION DOC] In a fake mode, with an unbacked size hinted as 8 and 4 chunks, checks
        # _maybe_pad_tensor and _maybe_unpad_tensor leave shapes unchanged (padding is skipped).
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
        shard = Shard(0)

        with fake_mode:
            dim_size = fake_mode.shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(dim_size, 8)
            local_tensor = torch.empty(dim_size // 4, 4)
            full_tensor = torch.empty(dim_size, 4)

            padded = shard._maybe_pad_tensor(local_tensor, dim_size, 4)
            unpadded = shard._maybe_unpad_tensor(full_tensor, dim_size, 4)

        self.assertEqual(padded.shape, local_tensor.shape)
        self.assertEqual(unpadded.shape, full_tensor.shape)

    def test_hinted_unbacked_even_chunk_preserves_symbolic_shard_size(self):
        # [FUNCTION DOC] With a hinted unbacked symbolic size, checks _custom_chunk into 8 chunks
        # returns 8 chunks whose sizes equal the symbolic size divided by 8 and depend only on the
        # same unbacked symbols as the original batch symbol.
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())

        with fake_mode:
            batch = fake_mode.shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(batch, 4)
            dim_size = 4096 * batch - 2 * torch.sym_min(1024 * batch, 2048 * batch)
            tensor = torch.empty(dim_size, 256)
            expected_chunk_size = dim_size // 8

            chunks = Shard._custom_chunk(tensor, 8, dim=0)

        self.assertEqual(len(chunks), 8)
        for chunk in chunks:
            self.assertEqual(chunk.size(0).node.expr, expected_chunk_size.node.expr)
            self.assertEqual(
                free_unbacked_symbols(chunk.size(0).node.expr),
                free_unbacked_symbols(batch.node.expr),
            )

    def test_strided_shard_preserves_symbolic_split_factor(self):
        # [FUNCTION DOC] Checks a _StridedShard built with an unbacked symbolic split factor keeps
        # the symbol (free symbols, hint of 4, integer value 4), adds no shape env replacements,
        # and records a deferred runtime assert that the symbol equals 4.
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())

        with fake_mode:
            split_factor = fake_mode.shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(split_factor, 4)
            placement = _StridedShard(dim=0, split_factor=split_factor)

        self.assertEqual(placement._split_factor_int(), 4)
        self.assertEqual(
            free_symbols(placement.split_factor), free_symbols(split_factor)
        )
        self.assertEqual(optimization_hint(placement.split_factor), 4)
        self.assertEqual(fake_mode.shape_env.replacements, {})
        deferred_asserts = [
            runtime_assert.expr
            for asserts in fake_mode.shape_env.deferred_runtime_asserts.values()
            for runtime_assert in asserts
        ]
        self.assertIn(sympy.Eq(split_factor.node.expr, 4), deferred_asserts)

    def test_unbacked_even_fallback_hint_requires_override(self):
        # [FUNCTION DOC] Checks _hint_proves_even_shard is False for an unbacked size with no
        # override and True after an optimization hint of 8 is set for 4 shards.
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())

        with fake_mode:
            dim_size = fake_mode.shape_env.create_unbacked_symint()
            self.assertFalse(_hint_proves_even_shard(dim_size, 4))

            torch._dynamo.override_optimization_hint(dim_size, 8)
            self.assertTrue(_hint_proves_even_shard(dim_size, 4))

    def test_hinted_unbacked_shardability_adds_runtime_check(self):
        # [FUNCTION DOC] Checks is_tensor_shardable on a shape with an unbacked dim raises
        # GuardOnDataDependentSymNode without a hint, and returns True once a hint of 16 is
        # overridden.
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
        mesh = DeviceMesh("cpu", torch.arange(4), _init_backend=False, _rank=0)
        spec = DTensorSpec(mesh, (Shard(1),))

        with fake_mode:
            dim_size = fake_mode.shape_env.create_unbacked_symint()

            with self.assertRaises(GuardOnDataDependentSymNode):
                is_tensor_shardable((2, dim_size), spec)

            torch._dynamo.override_optimization_hint(dim_size, 16)
            self.assertTrue(is_tensor_shardable((2, dim_size), spec))

    def test_strided_shard_unbacked_local_size_avoids_chunk_guard(self):
        # [FUNCTION DOC] Checks _StridedShard.local_shard_size_and_offset on an unbacked size with
        # two chunks returns a local size still containing unbacked symbols and an empty offset
        # list.
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())

        with fake_mode:
            dim_size = fake_mode.shape_env.create_unbacked_symint()

        local_size, offsets = _StridedShard(
            dim=1, split_factor=9
        ).local_shard_size_and_offset(
            dim_size,
            num_chunks=2,
            rank=0,
            offset_mode=_StridedShardOffsetMode.ALL,
        )

        self.assertTrue(free_unbacked_symbols(local_size.node.expr))
        self.assertEqual(offsets, [])

    def test_strided_shard_split_tensor_uses_unbacked_safe_chunking(self):
        # [FUNCTION DOC] Checks _StridedShard._split_tensor on an unbacked dim with two chunks
        # returns two shards, no pad sizes, and a first shard size with unbacked symbols.
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())

        with fake_mode:
            dim_size = fake_mode.shape_env.create_unbacked_symint()
            tensor = torch.empty(1, dim_size)
            shards, pad_sizes = _StridedShard(dim=1, split_factor=9)._split_tensor(
                tensor,
                num_chunks=2,
                with_padding=False,
                contiguous=False,
            )

        self.assertEqual(len(shards), 2)
        self.assertEqual(pad_sizes, [])
        self.assertTrue(free_unbacked_symbols(shards[0].size(1).node.expr))


if __name__ == "__main__":
    run_tests()
