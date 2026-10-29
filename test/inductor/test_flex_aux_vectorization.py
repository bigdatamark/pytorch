# Owner(s): ["module: inductor"]

# [MODULE DOC] Tests for flex attention auxiliary-tensor vectorization selection. Covers the pure
# selector helpers in the inductor flex aux_vectorization module, using fake aux buffers and hand-
# built FX graphs under a mock graph handler: per-load classification (contiguous, lane uniform,
# gather), mask_mod and score_mod vector size choice, and the FlexFlash forward config lists as a
# function of CUDA compute capability (CUDA calls are patched, no GPU is needed).
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
import torch._inductor.kernel.flex.flex_flash_attention as flex_flash_attention_module
from torch._inductor.kernel.flex.aux_vectorization import (
    AuxLoadVecInfo,
    direct_aux_load_vec_size_and_kind,
    select_mask_mod_vec_size,
    select_score_mod_vec_size,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.inductor_utils import MockGraphHandler


@dataclass
class FakeAuxBuffer:
    # [CLASS DOC] HELPER: Minimal stand-in for an inductor buffer with only size, stride and
    # offset. Provides the three accessors the selectors read; the layout offset is returned as a
    # sympy integer.
    size: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int = 0

    def get_size(self):
        # [FUNCTION DOC] HELPER: Return the stored size tuple.
        return self.size

    def get_stride(self):
        # [FUNCTION DOC] HELPER: Return the stored stride tuple.
        return self.stride

    def get_layout(self):
        # [FUNCTION DOC] HELPER: Return a simple namespace whose offset field is the stored offset
        # as a sympy integer, mimicking a buffer layout.
        return SimpleNamespace(offset=sympy.Integer(self.offset))


def _aux_index_graph():
    # [FUNCTION DOC] HELPER: Create an empty FX graph with q_idx and kv_idx placeholders. Returns
    # a tuple of the graph and the two placeholder nodes.
    graph = torch.fx.Graph()
    return graph, graph.placeholder("q_idx"), graph.placeholder("kv_idx")


@instantiate_parametrized_tests
class TestFlexAuxVectorization(InductorTestCase):
    # [CLASS DOC] Tests for the aux vectorization selectors and the flash forward config builder.
    # Parametrized variants are expanded by the instantiate decorator; no shared state between
    # tests.
    def test_direct_aux_load_vec_size_selector(self):
        # [FUNCTION DOC] Check the per-load classification for a single auxiliary load. Builds
        # index expressions on kv_idx (remainder by 4, scaled, floor-divided) and expects: plain
        # kv_idx on a contiguous 1-D buffer gives contiguous width 8; kv remainder 4 gives width
        # 4; q_idx only gives lane uniform; mixed strides, scaling and floor division give gather;
        # a 2-D buffer indexed with both gives gather when the order is swapped relative to the
        # layout.
        graph, q_idx, kv_idx = _aux_index_graph()
        kv_mod_4 = graph.call_function(torch.ops.aten.remainder.Tensor, (kv_idx, 4))
        kv_stride_mix = graph.call_function(
            torch.ops.aten.sub.Tensor,
            (
                graph.call_function(torch.ops.aten.mul.Tensor, (kv_idx, 2)),
                kv_mod_4,
            ),
        )
        kv_times_2 = graph.call_function(torch.ops.aten.mul.Tensor, (kv_idx, 2))
        kv_floor_div_2 = graph.call_function(
            torch.ops.aten.div.Tensor_mode,
            (kv_idx, 2),
            {"rounding_mode": "floor"},
        )

        buffer = FakeAuxBuffer((128,), (1,))
        with V.set_graph_handler(MockGraphHandler()):
            self.assertEqual(
                direct_aux_load_vec_size_and_kind([kv_idx], buffer, q_idx, kv_idx),
                AuxLoadVecInfo.contiguous(8),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind([kv_mod_4], buffer, q_idx, kv_idx),
                AuxLoadVecInfo.contiguous(4),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind([q_idx], buffer, q_idx, kv_idx),
                AuxLoadVecInfo.lane_uniform(),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [kv_stride_mix], buffer, q_idx, kv_idx
                ),
                AuxLoadVecInfo.gather(),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind([kv_times_2], buffer, q_idx, kv_idx),
                AuxLoadVecInfo.gather(),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [kv_floor_div_2], buffer, q_idx, kv_idx
                ),
                AuxLoadVecInfo.gather(),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [kv_idx, q_idx],
                    FakeAuxBuffer((128, 128), (128, 1)),
                    q_idx,
                    kv_idx,
                ),
                AuxLoadVecInfo.gather(),
            )

    def test_direct_aux_load_vec_size_requires_contiguous_aligned_vector_dim(self):
        # [FUNCTION DOC] Check how buffer layout limits the contiguous vector width. Expects width
        # 8 for an aligned row-major 2-D buffer, width 16 when the max width is raised to 32 and
        # the inner dim is 48, and gather for a transposed layout, an unaligned row stride, or a
        # nonzero layout offset.
        graph, q_idx, kv_idx = _aux_index_graph()
        with V.set_graph_handler(MockGraphHandler()):
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [q_idx, kv_idx],
                    FakeAuxBuffer((128, 128), (128, 1)),
                    q_idx,
                    kv_idx,
                ),
                AuxLoadVecInfo.contiguous(8),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [q_idx, kv_idx],
                    FakeAuxBuffer((128, 48), (48, 1)),
                    q_idx,
                    kv_idx,
                    max_vec_size=32,
                ),
                AuxLoadVecInfo.contiguous(16),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [q_idx, kv_idx],
                    FakeAuxBuffer((128, 128), (1, 128)),
                    q_idx,
                    kv_idx,
                ),
                AuxLoadVecInfo.gather(),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [q_idx, kv_idx],
                    FakeAuxBuffer((128, 128), (9, 1)),
                    q_idx,
                    kv_idx,
                ),
                AuxLoadVecInfo.gather(),
            )
            self.assertEqual(
                direct_aux_load_vec_size_and_kind(
                    [kv_idx], FakeAuxBuffer((128,), (1,), offset=1), q_idx, kv_idx
                ),
                AuxLoadVecInfo.gather(),
            )

    @parametrize(
        "case",
        [
            "direct_qkv",
            "chained_rank4_batch_head_qkv",
            "kv_in_prefix",
            "rank1_kv",
            "lane_uniform_aux",
            "mixed_gather",
            "gather_only",
        ],
        name_fn=lambda case: case,
    )
    def test_mask_mod_vec_size_selector_invariants(self, case):
        # [FUNCTION DOC] Parametrized check of the mask_mod vector size selector on hand-built FX
        # graphs. The case name picks the graph and fake buffers: direct, chained and lane-uniform
        # loads expect width 32; kv in a prefix position, a rank-1 kv load, and gather-only loads
        # expect no vectorization (none); a mixed direct plus gather graph expects 32. Raises an
        # assertion error for an unknown case name.
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        h = graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")

        def load(buffer, indices):
            # [FUNCTION DOC] HELPER: Add an index-tensor call node loading from a buffer node with
            # a list of index nodes; returns the new node.
            return graph.call_function(torch.ops.aten.index.Tensor, (buffer, indices))

        def floordiv(index, divisor):
            # [FUNCTION DOC] HELPER: Add a floor-mode division node of an index node by a divisor;
            # returns it.
            return graph.call_function(
                torch.ops.aten.div.Tensor_mode,
                (index, divisor),
                {"rounding_mode": "floor"},
            )

        expected = 32
        match case:
            case "direct_qkv":
                mask_bias = graph.placeholder("mask_bias")
                output = load(mask_bias, [q_idx, kv_idx])
                buffers = [FakeAuxBuffer((128, 128), (128, 1))]
            case "chained_rank4_batch_head_qkv":
                bias = graph.placeholder("bias")
                batch_slice = load(bias, [b])
                head_slice = load(batch_slice, [h])
                row = load(head_slice, [q_idx])
                output = load(row, [kv_idx])
                buffers = [FakeAuxBuffer((2, 4, 128, 128), (65536, 16384, 128, 1))]
            case "kv_in_prefix":
                bias = graph.placeholder("bias")
                direct_load = load(bias, [b, kv_idx, q_idx])
                kv_slice = load(bias, [b, kv_idx])
                output = (direct_load, load(kv_slice, [q_idx]))
                buffers = [FakeAuxBuffer((2, 128, 128), (16384, 128, 1))]
                expected = None
            case "rank1_kv":
                mask_bias = graph.placeholder("mask_bias")
                output = load(mask_bias, [kv_idx])
                buffers = [FakeAuxBuffer((128,), (1,))]
                expected = None
            case "lane_uniform_aux":
                mask_bias = graph.placeholder("mask_bias")
                output = load(mask_bias, [h])
                buffers = [FakeAuxBuffer((4,), (1,))]
            case "mixed_gather":
                mask_bias = graph.placeholder("mask_bias")
                block_keep = graph.placeholder("block_keep")
                output = (
                    load(mask_bias, [q_idx, kv_idx]),
                    load(block_keep, [floordiv(q_idx, 128), floordiv(kv_idx, 128)]),
                )
                buffers = [
                    FakeAuxBuffer((128, 128), (128, 1)),
                    FakeAuxBuffer((1, 1), (1, 1)),
                ]
            case "gather_only":
                block_keep = graph.placeholder("block_keep")
                output = load(block_keep, [floordiv(q_idx, 128), floordiv(kv_idx, 128)])
                buffers = [FakeAuxBuffer((1, 1), (1, 1))]
                expected = None
            case _:
                raise AssertionError(case)

        graph.output(output)
        graph_module = torch.fx.GraphModule({}, graph)
        with V.set_graph_handler(MockGraphHandler()):
            self.assertEqual(
                select_mask_mod_vec_size(
                    has_mask_mod=True,
                    has_mask_aux_tensors=True,
                    supports_mask_mod_vec=True,
                    graph_module=graph_module,
                    other_buffers=buffers,
                ),
                expected,
            )

    def test_score_mod_vec_size_selector_allows_mixed_gather(self):
        # [FUNCTION DOC] Check that the score_mod selector accepts a graph with one direct 2-D
        # load and one gather load derived from kv_idx divided by 128, on an SM100-or-later
        # target, and returns width 8.
        graph = torch.fx.Graph()
        graph.placeholder("score")
        graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        bias = graph.placeholder("bias")
        block_keep = graph.placeholder("block_keep")
        direct_load = graph.call_function(
            torch.ops.aten.index.Tensor, (bias, [q_idx, kv_idx])
        )
        block_idx = graph.call_function(
            torch.ops.aten.div.Tensor_mode,
            (kv_idx, 128),
            {"rounding_mode": "floor"},
        )
        gather_load = graph.call_function(
            torch.ops.aten.index.Tensor, (block_keep, [block_idx])
        )
        graph.output((direct_load, gather_load))
        graph_module = torch.fx.GraphModule({}, graph)
        with V.set_graph_handler(MockGraphHandler()):
            self.assertEqual(
                select_score_mod_vec_size(
                    has_score_mod=True,
                    has_aux_tensors=True,
                    is_sm100_or_later=True,
                    graph_module=graph_module,
                    other_buffers=[
                        FakeAuxBuffer((128, 128), (128, 1)),
                        FakeAuxBuffer((1,), (1,)),
                    ],
                ),
                8,
            )

    @parametrize("chained", [False, True], name_fn=lambda chained: str(chained))
    def test_score_mod_vec_size_selector_rejects_score_placeholder_index(self, chained):
        # [FUNCTION DOC] Parametrized check (chained or single multi-index load) that indexing a
        # buffer with the score placeholder makes the score_mod selector fall back to width 1 (no
        # vectorization).
        graph = torch.fx.Graph()
        score = graph.placeholder("score")
        graph.placeholder("b")
        graph.placeholder("h")
        graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        bias = graph.placeholder("bias")
        if chained:
            row = graph.call_function(torch.ops.aten.index.Tensor, (bias, [score]))
            load = graph.call_function(torch.ops.aten.index.Tensor, (row, [kv_idx]))
        else:
            load = graph.call_function(
                torch.ops.aten.index.Tensor, (bias, [score, kv_idx])
            )
        graph.output(load)
        graph_module = torch.fx.GraphModule({}, graph)
        with V.set_graph_handler(MockGraphHandler()):
            self.assertEqual(
                select_score_mod_vec_size(
                    has_score_mod=True,
                    has_aux_tensors=True,
                    is_sm100_or_later=True,
                    graph_module=graph_module,
                    other_buffers=[FakeAuxBuffer((128, 128), (128, 1))],
                ),
                1,
            )

    @parametrize(
        "cuda_major", [9, 10, 11, 12], name_fn=lambda cuda_major: str(cuda_major)
    )
    def test_mask_mod_vec_config_supports_only_sm100_path(self, cuda_major):
        # [FUNCTION DOC] Parametrized over CUDA major version: with CUDA availability and compute
        # capability patched, the forward config builder with a mask_mod yields a single config
        # with mask vector size 32 for majors 10 and 11, and a default config otherwise (9 and
        # 12).
        expected_config = (
            flex_flash_attention_module.FlexFlashConfig(mask_mod_vec_size=32)
            if cuda_major in (10, 11)
            else flex_flash_attention_module.FlexFlashConfig()
        )
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(cuda_major, 0)
            ),
        ):
            self.assertEqual(
                flex_flash_attention_module.get_flex_flash_fwd_configs(
                    False, False, has_mask_mod=True
                ),
                [expected_config],
            )

    @torch._inductor.config.patch(
        {"max_autotune": True, "test_configs.max_flex_configs": None}
    )
    def test_mask_mod_vec_config_combines_with_max_autotune_score_sizes(self):
        # [FUNCTION DOC] With max_autotune on and no flex config limit, and capability patched to
        # 10, the forward config builder yields one config per score vector size from 1 to 128 in
        # powers of two, each with mask vector size 32.
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 0)
            ),
        ):
            self.assertEqual(
                flex_flash_attention_module.get_flex_flash_fwd_configs(
                    True, False, has_mask_mod=True
                ),
                [
                    flex_flash_attention_module.FlexFlashConfig(
                        score_mod_vec_size=vec_size,
                        mask_mod_vec_size=32,
                    )
                    for vec_size in (1, 2, 4, 8, 16, 32, 64, 128)
                ],
            )


if __name__ == "__main__":
    run_tests()
