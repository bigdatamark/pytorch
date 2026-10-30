# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor SIMD range trees, focused on derived reduction roots that
# split a reduction dimension by a group size. If triton is not installed, the module exits
# (when run as main) or raises SkipTest at import. Uses a mock graph handler, no GPU.

import sys
import typing
import unittest

import sympy

from torch._inductor.codegen.simd import DerivedIterationRangesRoot, IterationRangesRoot
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import IndexingOptions, TritonKernel, TritonSymbols
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import MockGraphHandler
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv


try:
    import triton  # noqa: F401
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904


class TestSIMDRangeTrees(TestCase):
    # [CLASS DOC] Test case with builders for a mock graph, a Triton kernel and range-tree
    # roots, plus tests of derived-root geometry and kernel range-tree switching.
    def _make_graph(self):
        # [FUNCTION DOC] HELPER: Returns a new mock graph handler for use as the active graph.
        return MockGraphHandler()

    def _make_kernel(self, *, persistent: bool):
        # [FUNCTION DOC] HELPER: Builds a Triton kernel with an x range of 4 and a reduction
        # range of 512 (no kernel features inputs); the persistent argument is passed as the
        # persistent reduction override, and cooperative reduction is forced off.
        features = SIMDKernelFeatures([], sympy.Integer(4), sympy.Integer(512))
        return TritonKernel(
            {"x": sympy.Integer(4), "r0_": sympy.Integer(512)},
            features=features,
            override_persistent_reduction=persistent,
            override_cooperative_reduction=False,
        )

    def _make_parent_reduction_root(self):
        # [FUNCTION DOC] HELPER: Builds a standalone reduction-range root with prefix r0_,
        # size 512, loop mode on, tensor dim 1, no grid dim, and a placeholder kernel object.
        return IterationRangesRoot(
            "r0_index",
            sympy.Integer(512),
            "r0_",
            0,
            typing.cast(typing.Any, object()),
            is_loop=True,
            tensor_dim=1,
            grid_dim=None,
            has_zdim=False,
        )

    def _make_derived_root(self, r_tree, *, group_size=sympy.Integer(128)):
        # [FUNCTION DOC] HELPER: Builds a derived range root from the given parent root,
        # dividing the parent numel, block size and block offset by group_size (default 128,
        # floor division) and attaching three named constants: group size, reduced block and
        # reduced numel (the last is marked non-constexpr, the first two constexpr).
        reduced_block = FloorDiv(r_tree.block_size(), group_size)
        reduced_numel = FloorDiv(r_tree.numel, group_size)
        return DerivedIterationRangesRoot(
            r_tree,
            numel=reduced_numel,
            block_size=reduced_block,
            block_offset=FloorDiv(r_tree.block_offset(), group_size),
            named_constants=(
                (sympy.Symbol("nested_R0_GROUP_SIZE"), group_size, True),
                (sympy.Symbol("nested_R0_REDUCED_BLOCK"), reduced_block, True),
                (sympy.Symbol("nested_R0_REDUCED_NUMEL"), reduced_numel, False),
            ),
        )

    def test_derived_root_geometry(self):
        # [FUNCTION DOC] Checks a derived root mirrors its parent's loop flag and prefix, has
        # numel, block size and block offset equal to the parent's divided by 128, owns its
        # own mask but not the parent's, has a distinct mask name, and does not support a
        # constant mask.
        r_tree = self._make_parent_reduction_root()
        derived = self._make_derived_root(r_tree)

        self.assertEqual(derived.is_loop, r_tree.is_loop)
        self.assertEqual(derived.prefix, r_tree.prefix)
        self.assertEqual(derived.numel, FloorDiv(r_tree.numel, 128))
        self.assertEqual(derived.block_size(), FloorDiv(r_tree.block_size(), 128))
        self.assertEqual(derived.block_offset(), FloorDiv(r_tree.block_offset(), 128))
        self.assertTrue(derived.owns_mask(derived.mask_name()))
        self.assertFalse(derived.owns_mask(r_tree.mask_name()))
        self.assertNotEqual(derived.mask_name(), r_tree.mask_name())
        self.assertFalse(derived.supports_constant_mask())

    def test_derived_root_uses_active_range_tree_geometry(self):
        # [FUNCTION DOC] Within a mock graph and kernel, checks that a derived root is not
        # treated as a constant mask, that an index mixing x and derived symbols is broadcast
        # under the kernel's default range trees, and that inside the context swapping in the
        # x tree and derived root it is not broadcast, the block shape becomes XBLOCK by
        # R0_BLOCK floor-divided by 128, and an indexing option using the derived mask reports
        # a reduction mask. After leaving the context, range trees are restored and the
        # reduction mask is gone.
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=False)
            x_tree, r_tree = kernel.range_trees
            derived = self._make_derived_root(r_tree)

            self.assertFalse(kernel._has_constant_mask(derived))

            with V.set_kernel_handler(kernel):
                x_entry = x_tree.full_range()
                derived_entry = derived.full_range()
                index = x_entry.symbol() * derived.numel + derived_entry.symbol()
                self.assertTrue(kernel.is_broadcasted(index))

                indexing = IndexingOptions(
                    index_str="",
                    mask_vars=OrderedSet([derived.mask_name()]),
                    expand_str=None,
                    _has_rindex=True,
                    index=index,
                    expand_shape=None,
                )
                self.assertFalse(indexing.has_rmask())

                with kernel.use_range_trees([x_tree, derived]):
                    self.assertFalse(kernel.is_broadcasted(index))
                    self.assertEqual(
                        TritonSymbols.get_block_shape(index),
                        ("XBLOCK", "(R0_BLOCK//128)"),
                    )
                    self.assertTrue(indexing.has_rmask())

                self.assertEqual(kernel.range_trees, [x_tree, r_tree])
                self.assertFalse(indexing.has_rmask())

    def test_use_range_trees_clears_simplify_indexing_cache(self):
        # [FUNCTION DOC] Checks that swapping range trees on a kernel invalidates the cached
        # index simplification: the cache is populated before, empty on entry, repopulated by
        # a new simplify call inside, and empty again on exit, with the original range trees
        # restored.
        graph = self._make_graph()
        with V.set_graph_handler(graph):
            kernel = self._make_kernel(persistent=False)
            x_tree, r_tree = kernel.range_trees
            derived = self._make_derived_root(r_tree)

            with V.set_kernel_handler(kernel):
                expr = x_tree.full_range().symbol() + r_tree.full_range().symbol()
                kernel.simplify_indexing(expr)
                self.assertGreater(kernel.simplify_indexing.cache_info().currsize, 0)

                with kernel.use_range_trees([x_tree, derived]):
                    self.assertEqual(kernel.simplify_indexing.cache_info().currsize, 0)
                    kernel.simplify_indexing(expr)
                    self.assertGreater(
                        kernel.simplify_indexing.cache_info().currsize, 0
                    )

                self.assertEqual(kernel.simplify_indexing.cache_info().currsize, 0)
                self.assertEqual(kernel.range_trees, [x_tree, r_tree])


if __name__ == "__main__":
    run_tests()
