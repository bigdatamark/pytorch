# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor subgraph choices produced by the decompose-K matmul template. A
# custom op wraps a matmul, its lowering builds a decompose-K subgraph choice and autotunes it,
# and the op is compiled with max-autotune. The script entry point runs the tests only when both
# CPU and GPU are available.
from unittest import mock
from unittest.mock import MagicMock

import torch
from torch._inductor.ir import Buffer, FixedLayout, FlexibleLayout
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


def decomposeK(a, b, kPartitions):
    # [FUNCTION DOC] HELPER: Reference decomposition of a matmul along the K dimension: splits K
    # into groups of the given partition size, reshapes the first input to a batch of row blocks
    # and the second to matching blocks, does a batched matmul with float32 output, sums over the
    # batch dimension and casts back to the input dtype. Returns the matrix of shape rows by
    # columns. It is not called anywhere in this file.
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    B = k // kPartitions
    a_reshaped = torch.permute(a.reshape(m, B, kPartitions), (1, 0, 2))
    b_reshaped = b.reshape(B, kPartitions, n)
    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    result_fp32 = result.to(torch.float32)
    reduced_buf = torch.sum(result_fp32, 0)
    return reduced_buf.to(a.dtype)


class TestSubgraphChoice(TestCase):
    # [CLASS DOC] GPU tests of the decompose-K subgraph template choice used through a custom op
    # with a registered lowering.
    def setUp(self):
        # [FUNCTION DOC] Only calls the parent setup.
        super().setUp()

    def _create_buffer(self, name, shape, dtype):
        # [FUNCTION DOC] HELPER: Builds an Inductor buffer with the given name and a fixed layout
        # on the first GPU with the given dtype and size. Not used by any test in this file.
        return Buffer(
            name=name,
            layout=FixedLayout(torch.device(f"{GPU_TYPE}:0"), dtype=dtype, size=shape),
        )

    def test_subgraph_decompose_k(self):
        # [FUNCTION DOC] Registers a custom matmul op and a lowering that builds the choices list
        # with the aten matmul and a decompose-K subgraph choice (256 partitions), first autotunes
        # both to exercise benchmarking, then autotunes with only the decompose-K choice so it is
        # the one generated. Compiles with max-autotune and static shapes on 32 by 4096 and 4096
        # by 32 half precision inputs and checks the result is close to a plain matmul with loose
        # tolerances.
        from torch._inductor.kernel.mm import aten_mm
        from torch._inductor.kernel.mm_common import mm_args

        mat1_shape, mat2_shape = (32, 4096), (4096, 32)

        @torch.library.custom_op("mylib::matmul_decompose", mutates_args={})
        def matmul_decompose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation: plain matrix product of the two
            # inputs.
            return a @ b

        @matmul_decompose.register_fake
        def _(a, b):
            # [FUNCTION DOC] HELPER: Fake implementation of the custom op: matrix product of the
            # two inputs, used for shape and dtype.
            return a @ b

        @register_lowering(torch.ops.mylib.matmul_decompose)
        def _(a, b):
            # [FUNCTION DOC] HELPER: Lowering for the custom op: builds matmul arguments, creates
            # the choice list with the aten matmul and appends the decompose-K subgraph choice,
            # autotunes the full list for benchmarking, then autotunes only the decompose-K choice
            # and returns the resulting node.
            _, _, _, layout, mat1, mat2 = mm_args(a, b)

            choices = [aten_mm.bind((mat1, mat2), layout)]

            kPartitions = 256

            decompose_k_subgraph_template = (
                torch._inductor.kernel.mm.DecomposeKSugraphTemplate()
            )

            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                k_split=kPartitions,
                input_nodes=(mat1, mat2),
                layout=layout,
            )

            # Test benchmarking against aten
            autotune_select_algorithm("test_subgraph_choice", choices, [a, b], layout)

            # Only return decomposeK case for codegen
            choices = [choices[1]]
            node, _ = autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )
            return node

        a_in = torch.randn(
            mat1_shape, dtype=torch.float16, device=torch.device(f"{GPU_TYPE}:0")
        )
        b_in = torch.randn(
            mat2_shape, dtype=torch.float16, device=torch.device(f"{GPU_TYPE}:0")
        )

        def func(mat1, mat2):
            # [FUNCTION DOC] HELPER: Calls the custom matmul op on both arguments.
            return torch.ops.mylib.matmul_decompose(mat1, mat2)

        compiled_func = torch.compile(func, mode="max-autotune", dynamic=False)

        res = compiled_func(a_in, b_in)

        # Check same results of compiled result and regular torch.mm
        torch.testing.assert_close(res, a_in @ b_in, atol=1e-1, rtol=1e-1)

    def test_subgraph_freeze_layout(self):
        # [FUNCTION DOC] Checks that building a decompose-K subgraph choice freezes the flexible
        # layout of the first input into a fixed layout with padded strides. Registers a custom op
        # whose lowering raises an error if the layout was not flexible beforehand, or was not
        # fixed afterward, if the stride did not change, or if the example inputs' strides differ
        # from the layout. Compiles with max-autotune on 4 by 14240 and 14240 by 128 bfloat16
        # inputs with the current node patched to not dislike padding.
        from torch._inductor.kernel.mm_common import mm_args

        M, N, K = (4, 128, 14240)
        a_in = torch.randn(
            (M, K), dtype=torch.bfloat16, device=torch.device(f"{GPU_TYPE}:0")
        )
        b_in = torch.randn(
            (K, N), dtype=torch.bfloat16, device=torch.device(f"{GPU_TYPE}:0")
        )

        @torch.library.custom_op("mylib::matmul_decompose_padding", mutates_args={})
        def matmul_decompose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation: plain matrix product of the two
            # inputs.
            return a @ b

        @matmul_decompose.register_fake
        def _(a, b):
            # [FUNCTION DOC] HELPER: Fake implementation of the custom op: matrix product of the
            # two inputs.
            return a @ b

        @register_lowering(torch.ops.mylib.matmul_decompose_padding)
        def _(a, b):
            # [FUNCTION DOC] HELPER: Lowering for the custom op: records the first input's
            # flexible layout stride, appends a two-way decompose-K subgraph choice, raises an
            # error if the first input's layout is not now fixed, still has the old stride, or its
            # example input strides differ, then autotunes and returns the node.
            _, _, _, layout, mat1, mat2 = mm_args(a, b)
            mat1_layout = mat1.layout
            if not isinstance(mat1_layout, FlexibleLayout):
                raise AssertionError
            mat1_stride = mat1_layout.stride

            choices = []

            kPartitions = 2

            decompose_k_subgraph_template = (
                torch._inductor.kernel.mm.DecomposeKSugraphTemplate()
            )

            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                k_split=kPartitions,
                input_nodes=(mat1, mat2),
                layout=layout,
            )

            choice = choices[0]
            if not isinstance(mat1.layout, FixedLayout):
                raise AssertionError

            # Creating the subgraph choice should have frozen the layout
            # We ensure padding so the stride should differ
            if mat1.layout.stride == mat1_stride:
                raise AssertionError

            for example_stride, layout_stride in zip(
                choice.example_inputs[0].stride(), mat1.layout.stride
            ):
                # Example inputs should have same stride as current layout
                if example_stride != layout_stride:
                    raise AssertionError

            node, _ = autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )
            return node

        def func(mat1, mat2):
            # [FUNCTION DOC] HELPER: Adds one to the first argument and calls the padding custom
            # matmul op.
            return torch.ops.mylib.matmul_decompose_padding((mat1 + 1.0), mat2)

        with mock.patch("torch._inductor.ir.V.get_current_node") as get_node_mock:
            node_mock = MagicMock()
            node_mock.meta = {"dislike_padding": False}
            get_node_mock.return_value = node_mock

            compiled_func = torch.compile(func, mode="max-autotune", dynamic=False)

            compiled_func(a_in, b_in)


if __name__ == "__main__":
    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU:
        run_tests()
