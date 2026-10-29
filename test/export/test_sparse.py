# Owner(s): ["module: sparse"]
# [MODULE DOC] Tests that torch.export (strict mode) propagates sparse tensor metadata (layout,
# indices, values, dtype) into the per-node fake values of the exported graph. Defines the
# dtype and index-type lists, a decorator factory parametrizing over all sparse layouts, several
# tiny network modules, and the TestSparseProp test case. Import time: only constants and
# class definitions, plus parametrized test instantiation at the bottom of the module.
#
# Test to ensure sparsity information propagates properly into traced graph.
#

import unittest

import torch
from torch._dynamo.source import LocalSource
from torch._environment import is_fbcode
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    RelaxedUnspecConstraint,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


# Various data types (preserved over operations).
DTYPES = [
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]

# Various index types.
ITYPES = [torch.int32, torch.int64]


# Constructs a subtest for every sparse layout currently supported in torch.sparse.
def all_sparse_layouts(test_name="layout"):
    # [FUNCTION DOC] Returns a parametrize decorator that runs a test once per sparse layout
    # (COO, CSR, CSC, BSR, BSC), each as a named subtest. The argument names the test parameter
    # that receives the layout; it defaults to layout. No side effects.
    return parametrize(
        test_name,
        [
            subtest(torch.sparse_coo, name="SparseCOO"),
            subtest(torch.sparse_csr, name="SparseCSR"),
            subtest(torch.sparse_csc, name="SparseCSC"),
            subtest(torch.sparse_bsr, name="SparseBSR"),
            subtest(torch.sparse_bsc, name="SparseBSC"),
        ],
    )


#
# Various network examples.
#


class IdNet(torch.nn.Module):
    # [CLASS DOC] HELPER: Trivial module whose forward returns its single input unchanged.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input tensor as is.
        return x


class SumNet(torch.nn.Module):
    # [CLASS DOC] HELPER: Module that reduces its input to a scalar sum.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the sum over all elements of the input.
        return x.sum()


class EltwiseNet(torch.nn.Module):
    # [CLASS DOC] HELPER: Module applying a chain of elementwise ops (negate, absolute value,
    # scale by two, relu); exported graph has one node per op after the placeholder.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Negates the input, takes the absolute value, doubles it and
        # applies relu; returns a tensor with the input's layout.
        return torch.nn.functional.relu(2 * torch.abs(-x))


class ToDenseNet(torch.nn.Module):
    # [CLASS DOC] HELPER: Module that converts a sparse input into a dense strided tensor.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the dense (strided) version of the input tensor.
        return x.to_dense()


class AddNet(torch.nn.Module):
    # [CLASS DOC] HELPER: Module adding two tensors, used with a sparse and a dense operand.
    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Returns the elementwise sum of the two inputs.
        return torch.add(x, y)


class SparseActivationCOO(torch.nn.Module):
    # [CLASS DOC] HELPER: Module that converts each dense tensor of an input list to sparse COO.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns a list holding each element of the input list
        # converted to sparse COO layout.
        return [xi.to_sparse() for xi in x]


class SparseActivationCSR(torch.nn.Module):
    # [CLASS DOC] HELPER: Module that converts each dense tensor of an input list to sparse CSR.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns a list holding each element of the input list
        # converted to sparse CSR layout.
        return [xi.to_sparse_csr() for xi in x]


#
# The test driver.
#


@unittest.skipIf(is_fbcode(), "See torch._dynamo.config")
class TestSparseProp(TestCase):
    # [CLASS DOC] Test case checking that exported graph node metadata (the val entry of each
    # node meta) is a fake tensor whose layout, indices and values match the real tensors for
    # sparse inputs and results. Skipped under fbcode. Tests are parametrized over dtype, index
    # dtype and layout by instantiate_parametrized_tests at module level.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base class setup only; no extra state.
        super().setUp()

    def assertEqualMeta(self, x, y):
        # [FUNCTION DOC] HELPER: Asserts x is a fake tensor and y a real tensor, then compares x
        # against y moved to the meta device with exact layout and coalesced state. Because meta
        # comparison skips values, it also compares the index and value tensors explicitly:
        # COO uses indices and values, compressed row layouts (CSR, BSR) use row pointers and
        # column indices, compressed column layouts (CSC, BSC) use column pointers and row
        # indices, plus values. Strided tensors need nothing more. Raises AssertionError on
        # mismatch or an unexpected layout.
        self.assertIsInstance(x, FakeTensor)
        self.assertIsInstance(y, torch.Tensor)

        # Convert expected value to meta for comparison.
        y = y.to("meta")
        self.assertEqual(x, y, exact_layout=True, exact_is_coalesced=True)

        # When x or y is a meta tensor (say, `x.device == "meta"`), then
        # assertEqual(x, y) compares only x and y attributes but skips
        # comparing their values. In the case of sparse tensors, this means
        # that comparing indices and values attributes are skipped as well,
        # which is why we are doing that explicitly below.
        if x.layout is torch.strided:
            pass
        elif x.layout is torch.sparse_coo:
            self.assertEqual(x._indices(), y._indices(), exact_layout=True)
            self.assertEqual(x._values(), y._values(), exact_layout=True)
        else:
            if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
                x_meta1, y_meta1 = (x.crow_indices(), y.crow_indices())
                x_meta2, y_meta2 = (x.col_indices(), y.col_indices())
            elif x.layout in {torch.sparse_csc, torch.sparse_bsc}:
                x_meta1, y_meta1 = (x.ccol_indices(), y.ccol_indices())
                x_meta2, y_meta2 = (x.row_indices(), y.row_indices())
            else:
                raise AssertionError(f"Unexpected layout: {x.layout}")
            self.assertEqual(x_meta1, y_meta1, exact_layout=True)
            self.assertEqual(x_meta2, y_meta2, exact_layout=True)
            self.assertEqual(x.values(), y.values(), exact_layout=True)

    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_idnet(self, dtype, itype, layout):
        # [FUNCTION DOC] For every simple sparse input of the given layout, dtype and index dtype
        # on CPU, strict-exports the identity network and checks that the first graph node (the
        # input placeholder) carries matching sparse metadata and all later nodes carry none.
        net = IdNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,), strict=True)
            # Test arg/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertEqualMeta(meta, sparse_input)
                else:
                    self.assertEqual(meta, None)

    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_sumnet(self, dtype, itype, layout):
        # [FUNCTION DOC] For every simple sparse input, runs the sum network eagerly, exports it
        # strictly, and checks the placeholder metadata equals the input, the sum node metadata
        # equals the eager result, and remaining nodes have no value metadata.
        net = SumNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            result = net(sparse_input)
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,), strict=True)
            # Test arg/sum/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertEqualMeta(meta, sparse_input)
                elif i == 1:
                    self.assertEqualMeta(meta, result)
                else:
                    self.assertEqual(meta, None)

    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_eltwisenet(self, dtype, itype, layout):
        # [FUNCTION DOC] For every simple sparse input, runs the elementwise network eagerly,
        # exports it strictly, and checks that the first five nodes (placeholder plus four
        # elementwise ops) have metadata matching the eager result, and later nodes have none.
        net = EltwiseNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            result = net(sparse_input)
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,), strict=True)
            # Test arg/neg/abs/mul/relu/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i <= 4:
                    self.assertEqualMeta(meta, result)
                else:
                    self.assertEqual(meta, None)

    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_todensenet(self, dtype, itype, layout):
        # [FUNCTION DOC] For every simple sparse input, exports the to-dense network strictly and
        # checks the placeholder metadata matches the sparse input, the conversion node metadata
        # matches the dense eager result, and later nodes have none.
        net = ToDenseNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            result = net(sparse_input)
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,), strict=True)
            # Test arg/todense/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertEqualMeta(meta, sparse_input)
                elif i == 1:
                    self.assertEqualMeta(meta, result)
                else:
                    self.assertEqual(meta, None)

    @parametrize("strict", (False, True))
    def test_sparse_coo_dynamic_size_to_dense(self, strict):
        # [FUNCTION DOC] Exports the to-dense network with a sparse COO input whose first dimension
        # is marked dynamic, in strict or non-strict mode per the parameter, then runs the
        # exported module on a differently sized sparse tensor and checks it equals eager output.
        net = ToDenseNet()
        x = torch.tensor(
            [
                [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
                [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
            ]
        ).to_sparse()
        y = torch.tensor(
            [
                [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
                [[0.0, 0.0, 1.0], [4.0, 1.0, 1.0]],
                [[0.0, 0.0, 1.0], [2.0, 6.0, 1.0]],
            ]
        ).to_sparse()

        prog = torch.export.export(
            net,
            (x,),
            dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}},
            strict=strict,
        )

        self.assertEqual(prog.module()(y), net(y))

    def test_sparse_coo_fake_to_dense_dtype_error(self):
        # [FUNCTION DOC] Under a fake tensor mode with export enabled, converts a sparse COO tensor
        # to fake and checks that asking for a dense result with an explicit dtype raises a
        # RuntimeError mentioning that the dtype argument is unsupported by sparse to dense.
        x = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse()
        mode = FakeTensorMode(
            shape_env=ShapeEnv(tracked_fakes=[]),
            allow_non_fake_inputs=True,
            export=True,
        )

        with mode:
            fake_x = mode.from_tensor(x)
            with self.assertRaisesRegex(
                RuntimeError, "dtype argument is not supported by sparse_to_dense"
            ):
                fake_x.to_dense(dtype=torch.float64)

    def test_sparse_coo_fake_symbolic_size_transfer(self):
        # [FUNCTION DOC] Builds a fake sparse COO tensor under one shape environment with the first
        # dimension dynamic and the second static, re-fakes it under a second shape environment
        # with the same symbolic context, converts to dense, and checks the result is strided
        # and its second dimension has size two (static size survives the transfer).
        x = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse()
        symbolic_context = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC],
            constraint_sizes=[RelaxedUnspecConstraint(warn_only=False), None],
        )
        shape_env1 = ShapeEnv(tracked_fakes=[])
        mode1 = FakeTensorMode(
            shape_env=shape_env1,
            allow_non_fake_inputs=True,
            export=True,
        )
        with mode1:
            fake_x1 = mode1.from_tensor(
                x,
                source=LocalSource("x"),
                symbolic_context=symbolic_context,
            )

        shape_env2 = ShapeEnv(tracked_fakes=[])
        mode2 = FakeTensorMode(
            shape_env=shape_env2,
            allow_non_fake_inputs=True,
            export=True,
        )
        with mode2:
            fake_x2 = mode2.from_tensor(
                fake_x1,
                source=LocalSource("x"),
                symbolic_context=symbolic_context,
            )
            dense_x2 = fake_x2.to_dense()

        self.assertEqual(dense_x2.layout, torch.strided)
        self.assertEqual(dense_x2.size(1), 2)

    def test_add(self):
        # [FUNCTION DOC] Exports the add network with a sparse CSR 4x4 matrix and a dense matrix
        # strictly, and checks metadata of the two placeholders and the add result node match the
        # real tensors, with no metadata on the output node.
        net = AddNet()
        Y = torch.arange(16, 32, dtype=torch.float32).view(4, 4)
        A = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 1.0],
                [3.0, 0.0, 3.0, 0.0],
            ],
            dtype=torch.float32,
        )
        S = A.to_sparse_csr()
        result = net(S, Y)
        # Build the traced graph.
        prog = torch.export.export(net, (S, Y), strict=True)
        # Test args/add/output.
        for i, node in enumerate(prog.graph.nodes):
            meta = node.meta.get("val", None)
            if i == 0:
                self.assertEqualMeta(meta, S)
            elif i == 1:
                self.assertEqualMeta(meta, Y)
            elif i == 2:
                self.assertEqualMeta(meta, result)
            else:
                self.assertEqual(meta, None)

    def test_activation_coo(self):
        # [FUNCTION DOC] Exports a network converting a list of three dense matrices to sparse COO
        # strictly and checks the three input placeholders and three conversion nodes have
        # metadata matching the real inputs and eager results; later nodes have none.
        net = SparseActivationCOO()
        x = [torch.randn(3, 3) for _ in range(3)]
        result = net(x)
        # Build the traced graph.
        prog = torch.export.export(net, args=(x,), strict=True)
        # Test args/to_sparse/output.
        for i, node in enumerate(prog.graph.nodes):
            meta = node.meta.get("val", None)
            if i <= 2:
                self.assertEqualMeta(meta, x[i])
            elif i <= 5:
                self.assertEqualMeta(meta, result[i - 3])
            else:
                self.assertEqual(meta, None)

    def test_activation_csr(self):
        # [FUNCTION DOC] Same as the COO activation test but with conversion to sparse CSR: checks
        # metadata of three placeholders and three conversion nodes against real tensors, and
        # that later nodes have none.
        net = SparseActivationCSR()
        x = [torch.randn(3, 3) for _ in range(3)]
        result = net(x)
        # Build the traced graph.
        prog = torch.export.export(net, args=(x,), strict=True)
        # Test args/to_sparse/output.
        for i, node in enumerate(prog.graph.nodes):
            meta = node.meta.get("val", None)
            if i <= 2:
                self.assertEqualMeta(meta, x[i])
            elif i <= 5:
                self.assertEqualMeta(meta, result[i - 3])
            else:
                self.assertEqual(meta, None)


instantiate_parametrized_tests(TestSparseProp)

if __name__ == "__main__":
    run_tests()
