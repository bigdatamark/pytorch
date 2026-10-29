# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor dependency tracking: that reads of buffers used inside
# bucketize (boundaries and optional sorter) and nested extern kernel arguments are collected,
# the offset of a memory dependency, how the C++ CSE variable records which loop variables it
# depends on, and normalization of memory dependencies by stride order. Runs under a graph
# lowering handler installed in setUp.
import contextlib

import torch
from torch._inductor.codegen.cpp_utils import CppCSEVariable
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    Buffer,
    ExternKernel,
    FixedLayout,
    Pointwise,
    ShapeAsConstantBuffer,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import sympy_index_symbol
from torch._inductor.virtualized import ops, V
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.utils._sympy.value_ranges import ValueRanges


class TestDependencies(InductorTestCase):
    # [CLASS DOC] Dependency analysis tests. setUp installs a graph lowering of a trivial traced
    # module as the current graph handler (closed in tearDown) so IR nodes can be built.
    def _create_buffer(self, name, shape, dtype=torch.float32):
        # [FUNCTION DOC] HELPER: Creates a plain buffer with the given name and a fixed layout
        # on the GPU device with the given shape and dtype (default float32).
        return Buffer(
            name=name,
            layout=FixedLayout(torch.device(GPU_TYPE), dtype=dtype, size=shape),
        )

    def setUp(self):
        # [FUNCTION DOC] Traces a doubling module, builds a graph lowering for it, and enters a
        # context stack that installs the lowering as the current graph handler.
        super().setUp()

        class DummyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that doubles its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input multiplied by two.
                return x * 2

        self._gm = torch.fx.symbolic_trace(DummyModule())
        self._graph = GraphLowering(self._gm)

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(V.set_graph_handler(self._graph))

    def tearDown(self):
        # [FUNCTION DOC] Closes the context stack, removing the graph handler, then runs
        # base teardown.
        self._stack.close()
        super().tearDown()

    def test_bucketize_dependencies_no_sorter(self):
        # [FUNCTION DOC] Builds a pointwise node whose body calls bucketize with boundaries
        # described by an integer offsets buffer and no sorter, and asserts the node has exactly
        # one read.
        offsets = self._create_buffer("offsets", (1025,), torch.int32)

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Pointwise body: bucketizes the first index against the
            # offsets buffer boundaries (name, size, stride info) as right-closed int32 buckets.
            idx = index[0]
            return ops.bucketize(
                values=idx,
                boundaries=(
                    offsets.get_name(),
                    offsets.get_size()[-1],
                    offsets.get_size()[0] * offsets.get_stride()[0],
                    offsets.get_stride()[-1],
                ),
                boundary_indices=0,
                indexing_dtype=torch.int32,
                right=True,
            )

        pointwise = Pointwise.create(
            device=torch.device(GPU_TYPE),
            dtype=torch.int32,
            inner_fn=inner_fn,
            ranges=[1024 * 4],
        )

        self.assertEqual(len(pointwise.get_reads()), 1)

    def test_bucketize_dependencies_sorter(self):
        # [FUNCTION DOC] Same as the no-sorter bucketize test but also passes a sorter buffer to
        # bucketize; asserts the pointwise node reports two reads (offsets and sorter).
        offsets = self._create_buffer("offsets", (1025,), torch.int32)
        sorter = self._create_buffer("sorter", (1025,), torch.int32)

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Pointwise body: bucketizes the first index against the
            # offsets buffer with an additional sorter buffer description.
            idx = index[0]
            return ops.bucketize(
                values=idx,
                boundaries=(
                    offsets.get_name(),
                    offsets.get_size()[-1],
                    offsets.get_size()[0] * offsets.get_stride()[0],
                    offsets.get_stride()[-1],
                ),
                boundary_indices=0,
                indexing_dtype=torch.int32,
                right=True,
                sorter=(
                    sorter.get_name(),
                    sorter.get_stride()[-1],
                ),
                sorter_indices=0,
            )

        pointwise = Pointwise.create(
            device=torch.device(GPU_TYPE),
            dtype=torch.int32,
            inner_fn=inner_fn,
            ranges=[1024 * 4],
        )

        self.assertEqual(len(pointwise.get_reads()), 2)

    def test_extern_kernel_nested_ir_args_are_dependencies(self):
        # [FUNCTION DOC] Builds an extern kernel with one regular input, an index buffer nested
        # in a constant args list with a symbolic shape constant, and a value buffer nested in a
        # keyword argument mapping. Asserts the set of read buffer names is the data, index and
        # value buffers.
        data = self._create_buffer("data", (4,))
        index = self._create_buffer("index", (4,), torch.int64)
        value = self._create_buffer("value", (4,))
        out = self._create_buffer("out", (4,))
        shape_arg = ShapeAsConstantBuffer(expr=sympy_index_symbol("nested_size"))

        extern = ExternKernel(
            name="extern",
            layout=out.layout,
            inputs=[data],
            constant_args=([None, index, shape_arg],),
            kwargs={"value": {"nested": (value, shape_arg)}},
        )

        reads = {dep.name for dep in extern.get_read_writes().reads}
        self.assertEqual(reads, {"data", "index", "value"})

    def test_get_offset(self):
        # [FUNCTION DOC] Creates two memory dependencies over two loop variables whose index is
        # the row-major combination, one with an extra constant of 1024, and asserts get_offset
        # returns 0 for the first and 1024 for the second.
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")
        var_ranges = {
            x: 1024,
            y: 2048,
        }
        dep1 = MemoryDep(
            "dep1",
            x * 2048 + y,
            list(var_ranges.keys()),
            list(var_ranges.values()),
        )
        dep2 = MemoryDep(
            "dep2",
            x * 2048 + y + 1024,
            list(var_ranges.keys()),
            list(var_ranges.values()),
        )
        self.assertEqual(dep1.get_offset(), 0)
        self.assertEqual(dep2.get_offset(), 1024)

    def test_cpp_cse_value_expr_tracks_dependent_itervars(self):
        # [FUNCTION DOC] Builds a C++ CSE variable and updates it for a value-expression op
        # whose argument is an index expression using symbol x, with a dummy kernel whose
        # iteration variables contain x. Asserts the variable then reports a dependency on x.
        x = sympy_index_symbol("x")

        class DummyCSE:
            # [CLASS DOC] HELPER: Dummy CSE with an empty variable name map, used by the
            # dummy kernel.
            varname_map = {}

        class DummyKernel:
            # [CLASS DOC] HELPER: Dummy kernel exposing a CSE object and a set of
            # iteration variables.
            cse = DummyCSE()
            itervars = {x}

        var = CppCSEVariable("tmp0", ValueRanges.unknown(), torch.int64)
        with V.set_kernel_handler(DummyKernel()):
            var.update_on_args("value_expr", (x + 1, torch.int64), {})

        self.assertTrue(var.depends_on(x))

    def test_normalize_with_stride_order_equal(self):
        # [FUNCTION DOC] Builds two memory dependencies on the same buffer and index but with
        # loop variables listed in different orders; asserts they compare unequal, then equal
        # after both are normalized by stride order.
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")

        loop_order1 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y,
            [x, y],
            [1024, 2048],
        )
        loop_order2 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y,
            [y, x],
            [2048, 1024],
        )
        self.assertTrue(loop_order1 != loop_order2)
        normalized_loop_order1 = loop_order1.normalize_with_stride_order()
        normalized_loop_order2 = loop_order2.normalize_with_stride_order()
        self.assertTrue(normalized_loop_order1 == normalized_loop_order2)

    def test_normalize_with_stride_order_unequal(self):
        # [FUNCTION DOC] Like the equal-case test but the second dependency has an extra
        # constant offset of five; asserts the dependencies differ before normalization and
        # still differ after normalization by stride order.
        x = sympy_index_symbol("x")
        y = sympy_index_symbol("y")

        loop_order1 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y,
            [x, y],
            [1024, 2048],
        )
        loop_order2 = MemoryDep(
            "access_the_same_buffer",
            x * 2048 + y + 5,
            [y, x],
            [2048, 1024],
        )
        self.assertTrue(loop_order1 != loop_order2)
        normalized_loop_order1 = loop_order1.normalize_with_stride_order()
        normalized_loop_order2 = loop_order2.normalize_with_stride_order()
        # unequal due to different offset
        self.assertTrue(normalized_loop_order1 != normalized_loop_order2)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU and HAS_GPU:
        run_tests("sympy")
