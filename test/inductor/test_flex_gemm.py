# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the flex_gemm higher-order op and its Inductor/CuTe DSL (Quack)
#   backend: GEMM epilogue analysis, local-reduce planning, config selection, generated-code
#   checks, and CUDA runtime comparisons against eager references.
# Contents: scale-encoding helpers for MX/NVFP4 epilogues, optional cute.jit epilogue fixtures,
#   CPU-only helper/plan tests, a shared FlexGemmTestCase base, SM100+ runtime tests, and
#   HOP-level tests.
# Import-time side effects: tries to import cutlass and cutlass.cute; when missing, the name
#   cute is None and the jit fixtures are not defined. CUDA tests are skipped without CUDA/CuTe
#   DSL/SM100.

import contextlib
import dataclasses
import importlib
import logging
import math
import struct
import sys
import unittest
from types import SimpleNamespace
from typing import get_args
from unittest import mock

import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch._higher_order_ops import flex_gemm
from torch._higher_order_ops.flex_gemm import _SUPPORTED_FLEX_GEMM_OP_NAMES, nvfp4_pack
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor.exc import InductorError
from torch._inductor.ops_handler import ReductionType
from torch._inductor.utils import run_and_get_code
from torch._subclasses.fake_tensor import is_fake
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM100OrLater, SM120OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_quantized import _f32_to_floatx_unpacked, pack_uint4
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


def mx_e8m0_scale(
    amax: torch.Tensor,
    max_value: float = 448.0,
    rounding: str | None = None,
    pack: int = 1,
) -> torch.Tensor:
    # [FUNCTION DOC] Test helper that encodes per-block absolute maxima into E8M0 (power-of-two
    #   exponent byte) scales. Arguments: amax tensor, max_value (positive finite float, default
    #   448), rounding (None, floor, or rceil; None means rceil), pack (1 or 2, number of values
    #   per inline-asm lane).
    # Raises ValueError for a non-float, non-finite or non-positive max_value, an unknown
    #   rounding mode, or a pack value other than 1 or 2. Returns a float8_e8m0fnu tensor shaped
    #   like amax.
    # When compiling or on fake tensors it emits inline PTX through inline_asm_elementwise
    #   (round-up conversion for rceil, round-to-zero conversion for floor with infinity clamped
    #   to a finite sentinel); in eager mode it computes the same exponent byte with integer bit
    #   manipulation on the float32 representation, mapping NaN to byte 255.
    """Encode E8M0 scales using only public tensor ops and inline assembly."""
    if (
        not isinstance(max_value, float)
        or not math.isfinite(max_value)
        or max_value <= 0
    ):
        raise ValueError(
            "mx_e8m0_scale max_value must be a finite positive float, "
            f"got {max_value!r}"
        )
    if rounding not in (None, "floor", "rceil"):
        raise ValueError(
            f"mx_e8m0_scale rounding must be 'floor', 'rceil', or None, got {rounding!r}"
        )
    rounding = "rceil" if rounding is None else rounding
    if pack not in (1, 2):
        raise ValueError(f"mx_e8m0_scale pack must be 1 or 2, got {pack}")

    max_abs = amax.float()
    use_inline_asm_path = torch.compiler.is_compiling() or is_fake(max_abs)
    if use_inline_asm_path:
        if rounding == "rceil":
            conversion_input = max_abs / max_value
            conversion_instruction = "cvt.rp.satfinite.ue8m0x2.f32"
            floor_infinity_input_bits = None
        else:  # rounding == "floor"
            max_power = math.floor(math.log2(max_value))
            conversion_input = max_abs * 2.0**-max_power
            conversion_instruction = "cvt.rz.ue8m0x2.f32"
            floor_infinity_input_bits = (
                struct.unpack("<I", struct.pack("<f", 2.0 ** (128 - max_power)))[0]
                if max_power > 0
                else None
            )

        if pack == 1:
            asm_str = f"{conversion_instruction} $0, 0.0, $1;"
            if floor_infinity_input_bits is not None:
                asm_str = (
                    "{ .reg .pred is_inf; .reg .f32 finite; "
                    "testp.infinite.f32 is_inf, $1; "
                    f"selp.f32 finite, 0f{floor_infinity_input_bits:08x}, $1, is_inf; "
                    f"{conversion_instruction} $0, 0.0, finite; }}"
                )
            constraints = "=h,r"
        else:  # pack == 2
            unpack_pair_asm = (
                "cvt.u32.u16 widened, packed; "
                "and.b32 $0, widened, 0xff; "
                "shr.u32 $1, widened, 8; }"
            )
            asm_str = (
                "{ .reg .b16 packed; .reg .u32 widened; "
                f"{conversion_instruction} packed, $3, $2; " + unpack_pair_asm
            )
            if floor_infinity_input_bits is not None:
                asm_str = (
                    "{ .reg .pred is_inf0, is_inf1; "
                    ".reg .f32 finite0, finite1; "
                    ".reg .b16 packed; .reg .u32 widened; "
                    "testp.infinite.f32 is_inf0, $2; "
                    "testp.infinite.f32 is_inf1, $3; "
                    f"selp.f32 finite0, 0f{floor_infinity_input_bits:08x}, $2, is_inf0; "
                    f"selp.f32 finite1, 0f{floor_infinity_input_bits:08x}, $3, is_inf1; "
                    f"{conversion_instruction} packed, finite1, finite0; "
                    + unpack_pair_asm
                )
            constraints = "=r,=r,r,r"
        return inline_asm_elementwise(
            conversion_input,
            asm_str=asm_str,
            constraints=constraints,
            dtype=torch.float8_e8m0fnu,
            pack=pack,
        )

    if rounding == "floor":
        max_power = math.floor(math.log2(max_value))
        max_abs_bits = max_abs.view(torch.int32)
        unbiased_exponent = ((max_abs_bits >> 23) & 0xFF) - 127
        unbiased_scale_exponent = torch.clamp(
            unbiased_exponent - max_power, min=-127, max=128
        )
        encoded_scale = (unbiased_scale_exponent + 127).to(torch.uint8)
    else:  # rounding == "rceil"
        conversion_input_bits = (max_abs / max_value).view(torch.int32)
        biased_exponent = (conversion_input_bits >> 23) & 0xFF
        mantissa_bits = conversion_input_bits & 0x7FFFFF
        encoded_scale = biased_exponent + (mantissa_bits != 0).to(torch.int32)
        encoded_scale = torch.where(
            (biased_exponent == 0) & (mantissa_bits <= 0x400000),
            torch.zeros_like(encoded_scale),
            encoded_scale,
        )
        encoded_scale = torch.clamp(encoded_scale, max=254).to(torch.uint8)

    encoded_scale = torch.where(
        torch.isnan(max_abs),
        torch.full_like(encoded_scale, 255),
        encoded_scale,
    )
    return encoded_scale.view(torch.float8_e8m0fnu)


def nvfp4_e4m3_scale(amax: torch.Tensor, max_value: float = 6.0) -> torch.Tensor:
    # [FUNCTION DOC] Test helper computing the ordinary E4M3 scale for NVFP4 epilogues: divides
    #   amax (as float32) by max_value (default 6.0), clamps to the smallest normal and largest
    #   finite float8_e4m3fn values, and returns a float8_e4m3fn tensor. No side effects.
    """Compute the ordinary E4M3 scale used by NVFP4 test epilogues."""
    return torch.clamp(
        amax.float() / max_value,
        min=torch.finfo(torch.float8_e4m3fn).tiny,
        max=torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)


try:
    import cutlass
    import cutlass.cute as cute
except ImportError:
    cute = None

if cute is not None:

    @cute.jit
    def relu_epilogue(acc):
        # [FUNCTION DOC] HELPER: cute.jit epilogue fixture applying ReLU elementwise to the
        #   accumulator fragment (keeps positive values, zeros the rest). Returns the new
        #   fragment.
        return cute.where(acc > cute.full_like(acc, 0), acc, cute.full_like(acc, 0))

    @cute.jit
    def captured_affine_epilogue(acc, col_bias, row_scale, tile_bias):
        # [FUNCTION DOC] HELPER: cute.jit epilogue fixture with three captured arguments: adds a
        #   per-column bias to the accumulator, multiplies by a per-row scale, adds a per-tile
        #   bias, then applies ReLU. Returns the fragment.
        value = (acc + col_bias) * row_scale + tile_bias
        return cute.where(
            value > cute.full_like(value, 0), value, cute.full_like(value, 0)
        )

    @cute.jit
    def row_scale_epilogue(acc, row_scale):
        # [FUNCTION DOC] HELPER: cute.jit epilogue fixture that multiplies the accumulator by a
        #   captured per-row scale and returns the product.
        return acc * row_scale

    @cute.jit
    def scalar_scale_epilogue(acc, scale):
        # [FUNCTION DOC] HELPER: cute.jit epilogue fixture that multiplies the accumulator by a
        #   captured scalar scale and returns the product.
        return acc * scale

    @cute.jit
    def captured_affine_scalar_epilogue(acc, col_bias, row_scale, tile_bias, scale):
        # [FUNCTION DOC] HELPER: cute.jit epilogue fixture combining the captured-affine
        #   computation (column bias, row scale, tile bias) with an extra scalar multiplication,
        #   followed by ReLU. Returns the fragment.
        value = ((acc + col_bias) * row_scale + tile_bias) * scale
        return cute.where(
            value > cute.full_like(value, 0), value, cute.full_like(value, 0)
        )

    @cute.jit
    def tuple_aux_epilogue(acc):
        # [FUNCTION DOC] HELPER: cute.jit epilogue fixture returning a pair for the main output
        #   and an auxiliary output: main is half of the accumulator plus one, aux is the
        #   accumulator squared plus two.
        main = (acc + cute.full_like(acc, 1.0)) * cute.full_like(acc, 0.5)
        aux = acc * acc + cute.full_like(acc, 2.0)
        return main, aux

    @cute.jit
    def captured_tuple_aux_epilogue(acc, col_bias, row_scale, tile_bias):
        # [FUNCTION DOC] HELPER: cute.jit epilogue fixture returning a main/aux pair with
        #   captured arguments. Main is the ReLU of the affine value (column bias added, row scale
        #   applied, tile bias added); aux is the accumulator times the row scale plus the tile
        #   bias.
        biased = (acc + col_bias) * row_scale + tile_bias
        main = cute.where(
            biased > cute.full_like(biased, 0), biased, cute.full_like(biased, 0)
        )
        aux = acc * row_scale + tile_bias
        return main, aux

    @cute.jit
    def fragment_group32_sum_feed_epilogue(acc):
        # [FUNCTION DOC] HELPER: cute.jit fixture mirroring the generated axis-1 group-of-32 sum
        #   feed epilogue. It converts the accumulator to float32, reshapes the fragment so groups
        #   of up to 32 elements are contiguous, sums each group, broadcasts the sums back and
        #   returns the fragment times the reciprocal of its group sum (a normalization by group
        #   sum).
        """Mirror the generated axis-1 group-32 TensorSSA feed epilogue.

        Uses the same symbolic min()/repeat fragment expressions as generated
        code, so a config whose per-thread fragment extent is smaller than the
        group would silently reduce partial groups and fail the reference check.
        """
        tmp0 = acc.to(cutlass.Float32)
        tmp1 = tmp0.reshape(
            (
                (
                    1,
                    cutlass.const_expr(min(32, cute.size(tmp0.shape, mode=[0]))),
                    cutlass.const_expr(
                        cute.size(tmp0.shape, mode=[0])
                        // min(32, cute.size(tmp0.shape, mode=[0]))
                    ),
                ),
                1,
                1,
            )
        )
        tmp2 = tmp1.reduce(
            cute.ReductionOp.ADD,
            init_val=0.0,
            reduction_profile=((None, 1, None), 1, 1),
        )
        tmp3 = tmp2.reshape(
            (
                (
                    1,
                    1,
                    cutlass.const_expr(
                        cute.size(tmp1.shape, mode=[0])
                        // min(32, cute.size(tmp1.shape, mode=[0]))
                    ),
                ),
                1,
                1,
            )
        )
        tmp4 = tmp3.broadcast_to(tmp1.shape)
        return tmp1 * (cute.full_like(tmp4, 1.0) / tmp4)

    @cute.jit
    def physical_group_sum_feed_epilogue(acc, local_reduce0):
        # [FUNCTION DOC] HELPER: cute.jit fixture for a physical group-sum feed: divides the
        #   accumulator by the captured local_reduce0 value (the precomputed group sum).
        return acc / local_reduce0

    @cute.jit
    def physical_group_sum_combine(lhs, rhs):
        # [FUNCTION DOC] HELPER: cute.jit combine callback for a physical group sum; returns the
        #   elementwise sum of its two partial values.
        return lhs + rhs

    @cute.jit
    def physical_group_sum_finalize(value):
        # [FUNCTION DOC] HELPER: cute.jit finalize callback for a physical group sum; returns
        #   the reduced value unchanged.
        return value


class TestFlexGemmRuntimeImport(TestCase):
    # [CLASS DOC] Checks that the flex_gemm runtime module can be imported without pulling in an
    #   external quack package. Stateless; one test method.
    def test_import_does_not_load_external_quack(self):
        # [FUNCTION DOC] Removes quack and all quack submodules from the module table and the
        #   cached flex_gemm runtime module, re-imports the runtime module, and asserts that no
        #   quack entry is present afterwards (the vendored copy must be used). Side effect:
        #   mutates the module table.
        for name in list(sys.modules):
            if name == "quack" or name.startswith("quack."):
                del sys.modules[name]
        sys.modules.pop("torch._inductor.kernel.flex_gemm.runtime", None)
        importlib.import_module("torch._inductor.kernel.flex_gemm.runtime")
        self.assertNotIn("quack", sys.modules)


@instantiate_parametrized_tests
class TestFlexGemmRuntimeHelpers(TestCase):
    # [CLASS DOC] CPU-friendly tests of helper layers behind flex_gemm: reduction table,
    #   fast-math decompositions, post-grad pass guards, epilogue graph normalization,
    #   local-reduce analysis, and dense config selection. Parametrized tests are expanded by the
    #   class decorator.
    @parametrize(
        "reduction_type",
        get_args(ReductionType),
        name_fn=lambda reduction_type: reduction_type,
    )
    def test_tensorssa_reduction_table_covers_inductor_vocabulary(self, reduction_type):
        # [FUNCTION DOC] For each Inductor reduction type, checks the TensorSSA reduction table:
        #   its key set must equal sum, prod, max, min with the expected CuTe reduction op, init
        #   value and combine expression. Unsupported reduction types must raise
        #   NotImplementedError saying they do not map to a CuTe TensorSSA reduction; supported
        #   ones must return the exact table entry.
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            tensorssa_reduction,
            TENSORSSA_REDUCTIONS,
        )

        expected = {
            "sum": ("cute.ReductionOp.ADD", "0.0", "lhs + rhs"),
            "prod": ("cute.ReductionOp.MUL", "1.0", "lhs * rhs"),
            "max": (
                "cute.ReductionOp.MAX",
                'float("-inf")',
                "cutlass.max(lhs, rhs)",
            ),
            "min": (
                "cute.ReductionOp.MIN",
                'float("inf")',
                "cutlass.min(lhs, rhs)",
            ),
        }
        self.assertEqual(set(TENSORSSA_REDUCTIONS), set(expected))
        if reduction_type not in expected:
            with self.assertRaisesRegex(
                NotImplementedError,
                f"{reduction_type} does not map to a CuTe TensorSSA reduction",
            ):
                tensorssa_reduction(reduction_type)
            return

        actual = tensorssa_reduction(reduction_type)
        self.assertIs(actual, TENSORSSA_REDUCTIONS[reduction_type])
        self.assertEqual(
            (actual.cute_op, actual.init_val, actual.combine_expr),
            expected[reduction_type],
        )

    @parametrize(
        "case",
        (
            (
                "sigmoid_fp16",
                torch.ops.aten.sigmoid.default,
                torch.float16,
                (torch.float32, torch.float16),
                torch.float16,
            ),
            (
                "sigmoid_bf16",
                torch.ops.aten.sigmoid.default,
                torch.bfloat16,
                (torch.float32, torch.bfloat16),
                torch.bfloat16,
            ),
            (
                "sigmoid_int",
                torch.ops.aten.sigmoid.default,
                torch.int32,
                (torch.float32,),
                torch.float32,
            ),
            (
                "sigmoid_bool",
                torch.ops.aten.sigmoid.default,
                torch.bool,
                (torch.float32,),
                torch.float32,
            ),
            (
                "silu_fp16",
                torch.ops.aten.silu.default,
                torch.float16,
                (torch.float32, torch.float16),
                torch.float16,
            ),
            (
                "silu_bf16",
                torch.ops.aten.silu.default,
                torch.bfloat16,
                (torch.float32, torch.bfloat16),
                torch.bfloat16,
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_fast_math_decompositions_preserve_type_promotion(self, case):
        # [FUNCTION DOC] Parametrized by case (name, aten op, input dtype, expected convert
        #   dtypes, result dtype) for sigmoid and silu. Traces the op with the fast-math Quack
        #   body decomposition table and verifies the dtype-conversion nodes inserted match the
        #   expected sequence, that the output dtype matches the input promotion rule, and that a
        #   tanh node is present.
        from torch._higher_order_ops.flex_gemm import flex_gemm_body_decomposition_table
        from torch._inductor.decomposition import decompositions
        from torch.fx.experimental.proxy_tensor import make_fx

        _, op, input_dtype, conversion_dtypes, result_dtype = case
        decomposition_table = flex_gemm_body_decomposition_table(
            {"backend": "QUACK", "fast_math": True}, decompositions
        )
        graph_module = make_fx(
            lambda x: op(x), decomposition_table=decomposition_table
        )(torch.ones(4, dtype=input_dtype))

        self.assertEqual(
            tuple(
                node.args[1]
                for node in graph_module.graph.nodes
                if node.target is torch.ops.prims.convert_element_type.default
            ),
            conversion_dtypes,
        )
        self.assertEqual(
            graph_module(torch.ones(4, dtype=input_dtype)).dtype, result_dtype
        )
        self.assertTrue(
            any(
                node.target is torch.ops.aten.tanh.default
                for node in graph_module.graph.nodes
            )
        )

    def test_post_grad_addmm_fusion_preserves_flex_gemm_body_mm(self):
        # [FUNCTION DOC] Builds a graph for a matmul plus bias and checks the post-grad addmm
        #   fusion validity predicate accepts it; after the matmul node is marked as a flex_gemm
        #   body gemm, the predicate must reject the fusion.
        from torch._higher_order_ops.flex_gemm import mark_flex_gemm_body_gemm_node
        from torch._inductor.fx_passes.post_grad import is_valid_addmm_fusion
        from torch.fx.experimental.proxy_tensor import make_fx

        graph_module = make_fx(lambda a, b, bias: torch.mm(a, b) + bias)(
            torch.randn(4, 8), torch.randn(8, 16), torch.randn(16)
        )
        placeholders = [
            node for node in graph_module.graph.nodes if node.op == "placeholder"
        ]
        match = SimpleNamespace(
            args=tuple(placeholders[:2]),
            kwargs={"inp": placeholders[2]},
            nodes=list(graph_module.graph.nodes),
        )
        self.assertTrue(is_valid_addmm_fusion(match))

        mark_flex_gemm_body_gemm_node(graph_module, torch.ops.aten.mm.default)
        self.assertFalse(is_valid_addmm_fusion(match))

    @parametrize(
        "case",
        (
            (
                "addmm",
                torch.ops.aten.addmm.default,
                ((4, 16), (4, 8), (8, 16)),
            ),
            (
                "baddbmm",
                torch.ops.aten.baddbmm.default,
                ((2, 4, 16), (2, 4, 8), (2, 8, 16)),
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_post_grad_unfuse_preserves_flex_gemm_body_gemm(self, case):
        # [FUNCTION DOC] Parametrized over addmm and baddbmm. Traces the gemm followed by relu
        #   under fake tensors, checks the should-prefer-unfused predicate for that gemm is true,
        #   then marks the gemm node as a flex_gemm body gemm and checks the predicate becomes
        #   false. Uses fake CUDA-device tensors only.
        from torch._higher_order_ops.flex_gemm import mark_flex_gemm_body_gemm_node
        from torch._inductor.fx_passes.post_grad import (
            should_prefer_unfused_addmm,
            should_prefer_unfused_baddbmm,
        )
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx

        _, gemm_op, input_shapes = case
        check = {
            torch.ops.aten.addmm.default: should_prefer_unfused_addmm,
            torch.ops.aten.baddbmm.default: should_prefer_unfused_baddbmm,
        }[gemm_op]
        with FakeTensorMode():
            input_values = [torch.empty(shape, device="cuda") for shape in input_shapes]
        graph_module = make_fx(
            lambda inp, mat1, mat2: gemm_op(inp, mat1, mat2).relu(),
            tracing_mode="fake",
        )(*input_values)
        inputs = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
        gemm = next(node for node in graph_module.graph.nodes if node.target is gemm_op)
        match = SimpleNamespace(
            args=tuple(inputs[1:]),
            kwargs={"inp": inputs[0]},
            output_node=lambda: gemm,
        )
        self.assertTrue(check(match))

        mark_flex_gemm_body_gemm_node(graph_module, gemm_op)
        self.assertFalse(check(match))

    def test_epilogue_analysis_matches_prepare_softmax(self):
        # [FUNCTION DOC] Traces a body that views its input as grouped and applies the
        #   online-softmax preparation primitive, runs the local-reduce analysis, and checks the
        #   prepare-softmax node is matched with no reduction type.
        from torch._inductor import inductor_prims
        from torch._inductor.kernel.gemm_epilogue_analysis import (
            GemmLocalReduceAnalysis,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(x):
            # [FUNCTION DOC] HELPER: traced body that views the input as a 3-D grouped tensor
            #   and returns the first result of the online softmax preparation primitive over the
            #   last dim.
            grouped = x.view(4, 4, 2)
            return inductor_prims.prepare_softmax_online(grouped, -1)[0]

        graph_module = make_fx(body)(torch.randn(4, 8))
        analysis = GemmLocalReduceAnalysis.from_graph_module(graph_module)
        prepare_softmax = next(
            node
            for node in graph_module.graph.nodes
            if node.target is inductor_prims.prepare_softmax_online
        )
        self.assertIn(prepare_softmax, analysis.matches)
        self.assertIsNone(analysis.matches[prepare_softmax].reduction_type)

    def test_epilogue_graph_normalizes_selected_fx_nodes(self):
        # [FUNCTION DOC] Traces a body with a keepdim sum, squeeze, max with indices, variance
        #   and a dtype view, builds the epilogue graph, and asserts the normalized node records:
        #   view, reduction (dims, keepdim, reduction name), squeeze, getitem, dtype view, and an
        #   unsupported-reduction record for variance.
        import operator

        from torch._inductor.kernel.gemm_epilogue import (
            GemmEpilogueGraph,
            NormalizedDtypeView,
            NormalizedGetItem,
            NormalizedReduction,
            NormalizedSqueeze,
            NormalizedUnsupportedReduction,
            NormalizedView,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(x):
            # [FUNCTION DOC] HELPER: traced body returning a squeezed grouped sum, the grouped
            #   max values, the grouped variance and an int32 reinterpretation of the input, to
            #   exercise node normalization.
            grouped = x.view(4, 2, 4)
            reduced = grouped.sum(dim=-1, keepdim=True)
            maximum = torch.max(grouped, dim=-1).values
            return (
                reduced.squeeze(-1),
                maximum,
                grouped.var(dim=-1),
                x.view(torch.int32),
            )

        graph_module = make_fx(body)(torch.randn(4, 8))
        normalized_nodes = GemmEpilogueGraph.from_nodes(
            tuple(graph_module.graph.nodes)
        ).normalized_nodes
        nodes = {
            node.target: node
            for node in graph_module.graph.nodes
            if node.target is not operator.getitem
        }
        placeholder = next(
            node for node in graph_module.graph.nodes if node.op == "placeholder"
        )
        view = nodes[torch.ops.aten.view.default]
        reduction = nodes[torch.ops.aten.sum.dim_IntList]
        maximum = nodes[torch.ops.aten.max.dim]
        getitem = next(
            node
            for node in graph_module.graph.nodes
            if node.target is operator.getitem and node.args == (maximum, 0)
        )
        squeeze = nodes[torch.ops.aten.squeeze.dim]
        unsupported = nodes[torch.ops.aten.var.correction]
        dtype_view = nodes[torch.ops.aten.view.dtype]
        self.assertEqual(normalized_nodes[view], NormalizedView(placeholder, (4, 2, 4)))
        self.assertEqual(
            normalized_nodes[reduction],
            NormalizedReduction(view, [-1], True, None, "sum"),
        )
        self.assertEqual(normalized_nodes[squeeze], NormalizedSqueeze(reduction))
        self.assertEqual(normalized_nodes[getitem], NormalizedGetItem(maximum, 0))
        self.assertEqual(
            normalized_nodes[dtype_view], NormalizedDtypeView(placeholder, torch.int32)
        )
        self.assertEqual(
            normalized_nodes[unsupported],
            NormalizedUnsupportedReduction(
                view,
                str(torch.ops.aten.var.correction),
            ),
        )

    def test_local_reduce_propagates_before_grouped_view_matching(self):
        # [FUNCTION DOC] Checks the FlexGemm local-reduce analysis propagates a match through a
        #   squeeze and a later reshape: the final output node must be a match whose value node is
        #   the grouped sum.
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            FlexGemmLocalReduceAnalysis,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(x):
            # [FUNCTION DOC] HELPER: traced body that sums a grouped view over the last dim with
            #   keepdim, squeezes it, then reshapes the result to a different rank-3 layout.
            grouped = x.view(4, 6, 2)
            reduced = grouped.sum(dim=-1, keepdim=True).squeeze(-1)
            return reduced.view(-1, 4, 3)

        graph_module = make_fx(body)(torch.randn(4, 12))
        analysis = FlexGemmLocalReduceAnalysis.from_graph_module(graph_module)
        output = next(
            node for node in graph_module.graph.nodes if node.op == "output"
        ).args[0]
        self.assertIsInstance(output, torch.fx.Node)
        self.assertIn(output, analysis.matches)
        self.assertEqual(
            analysis.matches[output].value_node.target,
            torch.ops.aten.sum.dim_IntList,
        )

    def test_dense_config_selection_is_explicit_and_sm110_reuses_sm100(self):
        # [FUNCTION DOC] Tests dense GEMM config selection in the flex_gemm template heuristics
        #   with patched device capability and patched Quack config enumeration. On capability 11
        #   the candidates (SM100 configs reused) are filtered and ordered with the default first,
        #   TMA-gather configs dropped; the default key depends on M and N (skinny, large, large
        #   rectangular picks), keys round-trip back to a GemmConfig, and a fake graph handler
        #   provides the size-var guard.
        # On capability 12 only the SM120 pingpong config is a candidate; on capability 9 with
        #   no matching configs a RuntimeError mentioning no QuACK configs is raised. Mocks
        #   restore global state on exit.
        from torch._inductor.heuristics.template import (
            flex_gemm as flex_gemm_heuristics,
        )
        from torch._vendor.quack.gemm_config import GemmConfig

        def config(tile_m, tile_n, cluster_m, cluster_n, dynamic, **kwargs):
            # [FUNCTION DOC] HELPER: builds a GemmConfig from tile and cluster sizes and the
            #   dynamic-persistent flag, filling remaining fields with fixed defaults (cluster_k
            #   one, no swap, device capacity ten, swizzle size eight) and allowing keyword
            #   overrides.
            values = {
                "tile_m": tile_m,
                "tile_n": tile_n,
                "cluster_m": cluster_m,
                "cluster_n": cluster_n,
                "cluster_k": 1,
                "is_dynamic_persistent": dynamic,
                "swap_ab": False,
                "use_tma_gather": False,
                "device_capacity": 10,
                "tile_k": None,
                "num_warps": None,
                "pingpong": False,
                "max_swizzle_size": 8,
            }
            values.update(kwargs)
            return GemmConfig(**values)

        default = config(128, 256, 2, 1, True)
        skinny = config(128, 192, 2, 1, True)
        large_rect = config(256, 256, 2, 1, True)
        large = config(256, 256, 2, 2, True)
        swap_variant = config(128, 128, 1, 1, False, swap_ab=True)
        gather_rejected = config(128, 128, 1, 1, False, use_tma_gather=True)

        fake_graph = SimpleNamespace(
            sizevars=SimpleNamespace(guard_or_false=lambda expr: bool(expr))
        )
        from torch._inductor.virtualized import V

        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(11, 0)),
            mock.patch(
                "torch._vendor.quack.gemm_config.get_all_configs",
                return_value=[
                    gather_rejected,
                    swap_variant,
                    large_rect,
                    default,
                    skinny,
                    large,
                ],
            ),
            V.set_graph_handler(fake_graph),
        ):
            self.assertEqual(
                flex_gemm_heuristics.candidate_gemm_configs_for_device(
                    torch.device("cuda")
                ),
                [default, skinny, large_rect, large, swap_variant],
            )
            self.assertEqual(
                flex_gemm_heuristics.default_gemm_config_key(
                    torch.device("cuda"), 256, 4096
                ),
                flex_gemm_heuristics.gemm_config_key(skinny),
            )
            self.assertEqual(
                flex_gemm_heuristics.default_gemm_config_key(
                    torch.device("cuda"), 768, 4096
                ),
                flex_gemm_heuristics.gemm_config_key(large),
            )
            self.assertEqual(
                flex_gemm_heuristics.default_gemm_config_key(
                    torch.device("cuda"), 1024, 4096
                ),
                flex_gemm_heuristics.gemm_config_key(large_rect),
            )
            self.assertEqual(
                flex_gemm_heuristics.default_gemm_config_key(
                    torch.device("cuda"), 1024, 1024
                ),
                flex_gemm_heuristics.gemm_config_key(skinny),
            )
            self.assertEqual(
                flex_gemm_heuristics.candidate_gemm_configs_for_device(
                    torch.device("cuda")
                ),
                [default, skinny, large_rect, large, swap_variant],
            )
            self.assertEqual(
                GemmConfig(**dict(flex_gemm_heuristics.gemm_config_key(large))), large
            )

        sm120_pingpong = config(
            128,
            128,
            1,
            1,
            True,
            device_capacity=12,
            pingpong=True,
        )
        self.assertNotEqual(
            flex_gemm_heuristics.gemm_config_key(default),
            flex_gemm_heuristics.gemm_config_key(sm120_pingpong),
        )
        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(12, 0)),
            mock.patch(
                "torch._vendor.quack.gemm_config.get_all_configs",
                return_value=[default, sm120_pingpong],
            ),
        ):
            self.assertEqual(
                flex_gemm_heuristics.candidate_gemm_configs_for_device(
                    torch.device("cuda")
                ),
                [sm120_pingpong],
            )
        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            mock.patch(
                "torch._vendor.quack.gemm_config.get_all_configs",
                return_value=[default],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "no QuACK configs"):
                flex_gemm_heuristics.candidate_gemm_configs_for_device(
                    torch.device("cuda")
                )

    def test_output_contraction_device_validation(self):
        # [FUNCTION DOC] Checks FlexGemmOutputContraction.validate_quack by device capability:
        #   group 2 passes on capabilities 10 and 11, group 4 passes on 10; a chunked group 4
        #   raises NotImplementedError (interleaved group 4 only); group 4 on 11, any group on 12,
        #   and any group on 9 raise NotImplementedError with the matching message.
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmOutputContraction,
        )

        group_2 = FlexGemmOutputContraction(2)
        group_4 = FlexGemmOutputContraction(4)
        for capability in (10, 11):
            group_2.validate_quack(capability)
        group_4.validate_quack(10)
        with self.assertRaisesRegex(NotImplementedError, "interleaved group 4"):
            FlexGemmOutputContraction(4, chunked=True).validate_quack(10)
        with self.assertRaisesRegex(NotImplementedError, "group 4 on SM100"):
            group_4.validate_quack(11)
        with self.assertRaisesRegex(NotImplementedError, "not yet supported on SM120"):
            group_2.validate_quack(12)
        with self.assertRaisesRegex(NotImplementedError, "only on SM100 and SM110"):
            group_2.validate_quack(9)
        with self.assertRaisesRegex(NotImplementedError, "only on SM100 and SM110"):
            group_4.validate_quack(9)

    @skipIfNoCuteDSL
    def test_quack_grouped_n_contract_device_validation(self):
        # [FUNCTION DOC] Same device rules as the contraction validation test, but against the
        #   vendored Quack grouped-N contract validator taking a group size and a capability
        #   tuple: groups 2 and 4 accepted on SM100, group 2 on SM110, group 4 on SM110 rejected,
        #   SM120 and SM90 rejected, all via NotImplementedError. Requires the CuTe DSL.
        from torch._vendor.quack.gemm_act import validate_grouped_n_contract_device

        validate_grouped_n_contract_device(2, (10, 0))
        validate_grouped_n_contract_device(2, (11, 0))
        validate_grouped_n_contract_device(4, (10, 0))
        with self.assertRaisesRegex(NotImplementedError, "group 2.*group 4"):
            validate_grouped_n_contract_device(4, (11, 0))
        with self.assertRaisesRegex(NotImplementedError, "not yet supported on SM120"):
            validate_grouped_n_contract_device(2, (12, 0))
        with self.assertRaisesRegex(NotImplementedError, "only on SM100 and SM110"):
            validate_grouped_n_contract_device(2, (9, 0))

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    def test_runtime_output_contraction_rejects_unsafe_explicit_config(self):
        # [FUNCTION DOC] Runs gemm_epilogue with an output-contraction plan and an explicit
        #   config key selecting a swap_ab config (patched candidate list and key decoder), with a
        #   device capacity override of SM100. Expects NotImplementedError saying the config is
        #   incompatible with output contraction. Needs CUDA (tensors on the GPU) but does not
        #   launch a kernel.
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmOutputContraction,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeOutputPlan,
            gemm_epilogue,
        )

        config = SimpleNamespace(
            swap_ab=True,
            cluster_m=1,
            tile_m=128,
            cluster_n=1,
            tile_n=16,
        )
        with (
            mock.patch(
                "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
                return_value=[config],
            ),
            mock.patch(
                "torch._inductor.heuristics.template.flex_gemm.gemm_config_from_key",
                return_value=config,
            ),
            self.assertRaisesRegex(
                NotImplementedError, "incompatible with output contraction"
            ),
        ):
            gemm_epilogue(
                torch.randn(4, 8, device="cuda"),
                torch.randn(8, 16, device="cuda"),
                lambda acc: acc,
                "unsafe_output_contraction_config",
                output_plan=FlexGemmRuntimeOutputPlan(
                    output_contraction=FlexGemmOutputContraction(2)
                ),
                config_key=(("swap_ab", True),),
                expected_ndim=2,
                device_capacity_override=(10, 0),
            )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    def test_runtime_output_contraction_selects_safe_default_config(self):
        # [FUNCTION DOC] With two patched candidate configs (an unsafe swap_ab one first, a safe
        #   non-swap one second) and the dispatch function mocked, runs gemm_epilogue with an
        #   output-contraction plan and no explicit config; asserts the config passed to the
        #   dispatch call is the safe candidate, i.e. default selection skips configs that are
        #   unsafe for contraction.
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmOutputContraction,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeOutputPlan,
            gemm_epilogue,
        )

        unsafe = SimpleNamespace(
            swap_ab=True,
            cluster_m=1,
            tile_m=128,
            cluster_n=1,
            tile_n=16,
        )
        safe = SimpleNamespace(
            swap_ab=False,
            cluster_m=1,
            tile_m=128,
            cluster_n=1,
            tile_n=16,
        )
        with (
            mock.patch(
                "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
                return_value=[unsafe, safe],
            ),
            mock.patch(
                "torch._inductor.kernel.flex_gemm.runtime.dispatch_gemm_act"
            ) as dispatch,
        ):
            gemm_epilogue(
                torch.randn(4, 8, device="cuda"),
                torch.randn(8, 16, device="cuda"),
                lambda acc: acc,
                "safe_output_contraction_config",
                output_plan=FlexGemmRuntimeOutputPlan(
                    output_contraction=FlexGemmOutputContraction(2)
                ),
                device_capacity_override=(10, 0),
            )
        self.assertIs(dispatch.call_args.kwargs["config"], safe)

    def test_explicit_dense_config_supports_partial_constraints(self):
        # [FUNCTION DOC] Checks explicit config handling in the heuristics and lowering config
        #   key helpers under patched capability 10. A fully specified key yields exactly that
        #   config; a partial constraint (tile_m only) yields every matching config, used as-is
        #   when not tuned and as the whole tuning set when tuned. Rejections
        #   (NotImplementedError): unknown field (stages), unsupported field values (boolean
        #   cluster_m, TMA gather), and a device capacity mismatch. A swap_ab explicit config
        #   combined with a local-reduce geometry must raise NotImplementedError about
        #   incompatibility with local reduction.
        from torch._inductor.heuristics.template import (
            flex_gemm as flex_gemm_heuristics,
        )
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.lowering import flex_gemm_config_keys
        from torch._inductor.virtualized import V
        from torch._vendor.quack.gemm_config import GemmConfig

        supported = GemmConfig(
            tile_m=128,
            tile_n=192,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        )
        alternative = dataclasses.replace(supported, tile_n=256, cluster_m=1)
        config = dict(flex_gemm_heuristics.gemm_config_key(supported))
        device = torch.device("cuda")
        fake_graph = SimpleNamespace(
            sizevars=SimpleNamespace(guard_or_false=lambda expr: bool(expr))
        )
        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch(
                "torch._vendor.quack.gemm_config.get_all_configs",
                return_value=[supported, alternative],
            ),
            V.set_graph_handler(fake_graph),
        ):
            self.assertEqual(
                flex_gemm_heuristics.explicit_gemm_configs_for_device(config, device),
                (supported,),
            )
            self.assertEqual(
                flex_gemm_heuristics.explicit_gemm_configs_for_device(
                    {"tile_m": 128}, device
                ),
                (supported, alternative),
            )
            self.assertEqual(
                flex_gemm_config_keys(
                    device,
                    128,
                    128,
                    (),
                    tuned=False,
                    explicit_config={"tile_m": 128},
                ),
                (flex_gemm_heuristics.gemm_config_key(supported),),
            )
            self.assertEqual(
                flex_gemm_config_keys(
                    device,
                    128,
                    128,
                    (),
                    tuned=True,
                    explicit_config={"tile_m": 128},
                ),
                tuple(
                    flex_gemm_heuristics.gemm_config_key(candidate)
                    for candidate in (supported, alternative)
                ),
            )
            with self.assertRaisesRegex(NotImplementedError, "unexpected.*stages"):
                flex_gemm_heuristics.explicit_gemm_configs_for_device(
                    {**config, "stages": 4}, device
                )
            with self.assertRaisesRegex(NotImplementedError, "not supported"):
                flex_gemm_heuristics.explicit_gemm_configs_for_device(
                    {"cluster_m": True}, device
                )
            with self.assertRaisesRegex(NotImplementedError, "not supported"):
                flex_gemm_heuristics.explicit_gemm_configs_for_device(
                    {**config, "use_tma_gather": True}, device
                )
            with self.assertRaisesRegex(
                NotImplementedError, "targets SM90.*uses SM100"
            ):
                flex_gemm_heuristics.explicit_gemm_configs_for_device(
                    {**config, "device_capacity": 9}, device
                )

        swap_config = dict(
            flex_gemm_heuristics.gemm_config_key(
                dataclasses.replace(supported, swap_ab=True)
            )
        )
        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch(
                "torch._vendor.quack.gemm_config.get_all_configs",
                return_value=[dataclasses.replace(supported, swap_ab=True)],
            ),
            V.set_graph_handler(fake_graph),
            self.assertRaisesRegex(
                NotImplementedError, "incompatible with local reduction"
            ),
        ):
            flex_gemm_config_keys(
                device,
                128,
                128,
                (FlexGemmLocalReduceGeometry(group=16, axis=1),),
                tuned=False,
                explicit_config=swap_config,
            )

    def test_multi_cta_local_reduce_full_tile_follows_n_warp_topology(self):
        # [FUNCTION DOC] Checks local-reduce config validation for multi-CTA configs: a group
        #   spanning the full tile N is allowed only when a single epilogue N-warp partition owns
        #   N (the 256-wide-N config rejects full-N, the 512-wide one accepts it), groups along
        #   axis 0 sized by tile M are rejected for the single-N-warp case, and the maximum
        #   allowed group over both configs on axis 1 is 512.
        """Allow full-N groups only when one epilogue warp partition owns N."""
        from torch._inductor.kernel.flex_gemm.constraints import (
            max_flex_gemm_local_reduce_group_for_configs,
            validate_flex_gemm_local_reduce_config,
        )
        from torch._vendor.quack.gemm_config import GemmConfig

        configs = (
            GemmConfig(
                tile_m=128,
                tile_n=256,
                pingpong=False,
                cluster_m=2,
                device_capacity=10,
            ),
            GemmConfig(
                tile_m=256,
                tile_n=512,
                pingpong=False,
                cluster_m=2,
                device_capacity=10,
            ),
        )
        split_n_warp, single_n_warp = configs
        self.assertTrue(validate_flex_gemm_local_reduce_config(split_n_warp, 128, 1))
        self.assertFalse(
            validate_flex_gemm_local_reduce_config(split_n_warp, split_n_warp.tile_n, 1)
        )
        self.assertTrue(validate_flex_gemm_local_reduce_config(single_n_warp, 128, 1))
        self.assertTrue(
            validate_flex_gemm_local_reduce_config(
                single_n_warp, single_n_warp.tile_n, 1
            )
        )
        self.assertFalse(
            validate_flex_gemm_local_reduce_config(
                single_n_warp, single_n_warp.tile_m, 0
            )
        )
        self.assertEqual(max_flex_gemm_local_reduce_group_for_configs(configs, 1), 512)

    def test_precompile_metadata_counts_symbolic_skip(self):
        # [FUNCTION DOC] Builds a FlexGemmEpilogueCaller without running its constructor and
        #   gives it fake benchmark request metadata containing a symbolic input size. Asserts
        #   precompile_metadata returns None (precompilation is skipped for dynamic shapes) and
        #   that the Inductor counter for skipped dynamic precompiles is incremented to one.
        #   Clears the global dynamo counters first.
        import sympy

        from torch._dynamo.utils import counters
        from torch._inductor.kernel.flex_gemm.template import FlexGemmEpilogueCaller

        counters.clear()
        caller = FlexGemmEpilogueCaller.__new__(FlexGemmEpilogueCaller)
        caller.bmreq = SimpleNamespace(
            input_tensor_meta=[
                SimpleNamespace(
                    sizes=(sympy.Symbol("s0"), 64),
                    strides=(64, 1),
                    dtype=torch.float32,
                    device=torch.device("cuda", 0),
                )
            ],
            output_tensor_meta=SimpleNamespace(
                sizes=(128, 128),
                strides=(128, 1),
                dtype=torch.float32,
                device=torch.device("cuda", 0),
            ),
        )

        self.assertIsNone(caller.precompile_metadata())
        self.assertEqual(
            counters["inductor"]["flex_gemm_precompile_skipped_dynamic"], 1
        )


class FlexGemmTestCase(TestCase):
    # [CLASS DOC] Shared base class for flex_gemm tests: provides tensor factories,
    #   generated-code checks (via FileCheck), config-key helpers, and numerical comparison
    #   helpers that bound error relative to a low-precision eager reference. Stateless.
    def makeTensor(self, *shape, device="cuda", dtype=torch.bfloat16):
        # [FUNCTION DOC] HELPER: builds a random tensor of the given shape on device (default
        #   CUDA) and dtype (default bfloat16) with values drawn uniformly from minus 0.1 to 0.1.
        return torch.testing.make_tensor(
            *shape, device=device, dtype=dtype, low=-0.1, high=0.1
        )

    def assertFlexGemmGeneratedCode(self, code, *checks):
        # [FUNCTION DOC] HELPER: asserts generated Inductor code calls the flex_gemm runtime:
        #   the import of the runtime plan and gemm_epilogue, the epilogue call, each extra
        #   pattern in checks in order, then the stream and config_key arguments. The code must
        #   not contain a tuned argument, an epilogue_source argument, or any quack import.
        file_check = (
            FileCheck()
            .check("from torch._inductor.kernel.flex_gemm.runtime import (")
            .check("FlexGemmRuntimeLocalReducePlan")
            .check("gemm_epilogue as flex_gemm_epilogue")
            .check("flex_gemm_epilogue(")
        )
        for check in checks:
            file_check = file_check.check(check)
        file_check = (
            file_check.check("stream=stream").check("config_key=").check_not("tuned=")
        )
        file_check = file_check.check_not("epilogue_source=")
        file_check.check_not("from quack").check_not("import quack").run(code)

    def swapAndNonSwapConfigKeys(self, device):
        # [FUNCTION DOC] HELPER: builds config keys for all candidate GEMM configs on the device
        #   and returns the first swap_ab key and first non-swap key. Asserts that both kinds
        #   exist.
        """Return one swap_ab and one non-swap candidate config key for ``device``."""
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )

        keys = [
            gemm_config_key(config)
            for config in candidate_gemm_configs_for_device(device)
        ]
        swap_keys = [key for key in keys if dict(key)["swap_ab"]]
        non_swap_keys = [key for key in keys if not dict(key)["swap_ab"]]
        self.assertTrue(swap_keys and non_swap_keys)
        return swap_keys[0], non_swap_keys[0]

    def assertMatchesLowPrecisionEager(
        self,
        actual,
        low_precision_expected,
        high_precision_expected,
        reduction_size,
    ):
        # [FUNCTION DOC] HELPER: asserts the mean absolute error of actual (against a float64
        #   reference) does not exceed the mean error of the low-precision eager result plus an
        #   allowance. The allowance is the sum of a float32 accumulation epsilon scaled by the
        #   square root of the reduction size and the output dtype epsilon, times the mean
        #   magnitude of the reference. Failure message reports the error terms.
        actual_error = (actual.double() - high_precision_expected).abs().mean()
        eager_error = (
            (low_precision_expected.double() - high_precision_expected).abs().mean()
        )
        # Model the extra slack as fp32 accumulator rounding across K plus final output rounding.
        fp32_accumulation_eps = (
            math.sqrt(reduction_size) * torch.finfo(torch.float32).eps
        )
        result_rounding_eps = torch.finfo(actual.dtype).eps
        output_scale = high_precision_expected.abs().mean().item()
        rounding_atol = (fp32_accumulation_eps + result_rounding_eps) * output_scale
        self.assertLessEqual(
            actual_error.item(),
            eager_error.item() + rounding_atol,
            msg=(
                lambda msg: f"{msg}\nactual error {actual_error.item()} exceeded low precision eager "
                f"error {eager_error.item()} with fp32_accumulation_eps="
                f"{fp32_accumulation_eps}, result_rounding_eps="
                f"{result_rounding_eps}, output_scale={output_scale}, "
                f"and atol={rounding_atol}"
            ),
        )

    def assertTupleAuxMatchesReference(self, actual, aux, a, b, epilogue_fn):
        # [FUNCTION DOC] HELPER: checks a main/aux pair from a tuple-aux epilogue against eager
        #   references. The main output is compared with half of the float64 product plus one
        #   half, the aux output with the squared float64 product plus two, using the K dimension
        #   as reduction size.
        """Validate tuple-aux epilogues against low/high precision references."""
        expected, expected_aux = epilogue_fn(a @ b)
        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            (high_precision_acc + 1.0) * 0.5,
            a.shape[-1],
        )
        self.assertMatchesLowPrecisionEager(
            aux,
            expected_aux,
            high_precision_acc.square() + 2.0,
            a.shape[-1],
        )

    def assertCapturedTupleAuxMatchesReference(
        self, actual, aux, a, b, col_bias, row_scale, tile_bias
    ):
        # [FUNCTION DOC] HELPER: checks main and aux outputs of the captured-argument tuple
        #   epilogue. The main output is the ReLU of the biased and scaled float32 accumulator,
        #   the aux is the row-scaled accumulator plus tile bias; each compared with a float64
        #   reference via the low-precision comparison helper.
        """Validate composed captured-load and tuple-aux epilogues."""
        acc = a @ b
        acc_float = acc.float()
        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            actual,
            ((acc_float + col_bias) * row_scale + tile_bias).relu(),
            (
                (high_precision_acc + col_bias.double()) * row_scale.double()
                + tile_bias.double()
            ).relu(),
            a.shape[-1],
        )
        self.assertMatchesLowPrecisionEager(
            aux,
            acc_float * row_scale + tile_bias,
            high_precision_acc * row_scale.double() + tile_bias.double(),
            a.shape[-1],
        )

    def localReduceGeometryPattern(self, group, axis):
        # [FUNCTION DOC] HELPER: returns the textual form of a local-reduce geometry
        #   construction with the given group and axis, as it should appear in generated code.
        """Return the generated structural local-reduce geometry pattern."""
        return f"FlexGemmLocalReduceGeometry(group={group}, axis={axis})"

    def assertLocalReduceAuxCode(self, code, group, axis=1, callbacks=False):
        # [FUNCTION DOC] HELPER: asserts generated code builds a runtime local-reduce plan with
        #   the expected geometry text and an out argument, includes the callbacks class only when
        #   callbacks is true (and excludes it otherwise), and does not contain the legacy
        #   local_reduce_out, local_reduce_group, local_reduce_axis or local_reduce_op arguments.
        """Check generated code passes a structural compressed-aux plan."""
        file_check = (
            FileCheck()
            .check("local_reduce=FlexGemmRuntimeLocalReducePlan")
            .check(self.localReduceGeometryPattern(group, axis))
            .check("out=")
        )
        if callbacks:
            file_check = file_check.check("callbacks=FlexGemmLocalReduceCallbacks")
        else:
            file_check = file_check.check_not("callbacks=FlexGemmLocalReduceCallbacks")
        file_check.check_not("local_reduce_out=").check_not(
            "local_reduce_group="
        ).check_not("local_reduce_axis=").check_not("local_reduce_op").run(code)

    def runtimeLocalReducePlan(
        self,
        out=None,
        group=8,
        axis=0,
        feeds_main=False,
    ):
        # [FUNCTION DOC] HELPER: builds a FlexGemmRuntimeLocalReducePlan from group, axis, out
        #   and feeds_main. Callbacks (identity-like combine and finalize functions) are attached
        #   when the plan feeds main, the axis is 0, or the group exceeds the fragment width
        #   constant; otherwise callbacks are None.
        """Build the structural local-reduce runtime plan used by generated code."""
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceCallbacks,
            FlexGemmLocalReduceGeometry,
            LOCAL_REDUCE_FRAGMENT_WIDTH,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeLocalReducePlan,
        )

        callbacks = None
        if feeds_main or axis == 0 or group > LOCAL_REDUCE_FRAGMENT_WIDTH:
            callbacks = FlexGemmLocalReduceCallbacks(
                combine_fn=lambda lhs, rhs: lhs,
                finalize_fn=lambda value: value,
            )
        return FlexGemmRuntimeLocalReducePlan(
            FlexGemmLocalReduceGeometry(group, axis),
            out=out,
            callbacks=callbacks,
            feeds_main=feeds_main,
        )

    def assertMatchesEpilogue(
        self, actual, expected, high_precision_expected, reduction_size
    ):
        # [FUNCTION DOC] HELPER: compares actual outputs with eager and high-precision
        #   references. For a tuple of expected outputs, checks lengths match and compares
        #   elementwise with the low-precision comparison helper; for a single tensor compares
        #   directly.
        """Compare one or multiple epilogue outputs against eager references."""
        if isinstance(expected, tuple):
            self.assertEqual(len(actual), len(expected))
            self.assertEqual(len(expected), len(high_precision_expected))
            for actual_item, expected_item, high_precision_item in zip(
                actual, expected, high_precision_expected
            ):
                self.assertMatchesLowPrecisionEager(
                    actual_item,
                    expected_item,
                    high_precision_item,
                    reduction_size,
                )
            return
        self.assertMatchesLowPrecisionEager(
            actual, expected, high_precision_expected, reduction_size
        )

    def assertLocalReduceAuxMatches(self, actual, aux, a, b, epilogue_fn):
        # [FUNCTION DOC] HELPER: computes eager and float64 versions of the epilogue on the GEMM
        #   result. The main output is compared with the low-precision helper (K from the second
        #   dimension of a); the compressed aux output is compared with the float64 aux using
        #   absolute and relative tolerance of one thousandth.
        """Validate compressed local-reduce aux output against high precision GEMM."""
        expected, _ = epilogue_fn(a @ b)
        high_precision_expected, high_precision_aux = epilogue_fn(
            a.double() @ b.double()
        )
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            high_precision_expected,
            a.shape[1],
        )
        torch.testing.assert_close(
            aux,
            high_precision_aux.float(),
            atol=1e-3,
            rtol=1e-3,
        )

    def assertPhysicalFeedMainCode(self, code, group=None):
        # [FUNCTION DOC] HELPER: asserts generated code passes a runtime local-reduce plan with
        #   a feeds-main style ABI: axis 0 geometry (with the given group when specified), a
        #   callbacks object, and none of the legacy local_reduce_out, local_reduce_feeds_main or
        #   local_reduce_op arguments.
        """Check generated code uses the current physical feed-main ABI."""
        file_check = FileCheck().check("local_reduce=FlexGemmRuntimeLocalReducePlan")
        if group is None:
            file_check = file_check.check("FlexGemmLocalReduceGeometry(group=").check(
                "axis=0)"
            )
        else:
            file_check = file_check.check(self.localReduceGeometryPattern(group, 0))
        file_check.check("callbacks=FlexGemmLocalReduceCallbacks").check_not(
            "local_reduce_out="
        ).check_not("local_reduce_feeds_main=True").check_not("local_reduce_op").run(
            code
        )


@skipIfNoCuteDSL
@unittest.skipIf(not TEST_CUDA, "CUDA required")
@instantiate_parametrized_tests
class TestFlexGemmRuntime(FlexGemmTestCase):
    # [CLASS DOC] CUDA runtime tests of gemm_epilogue and the generated epilogue paths. Skipped
    #   without the CuTe DSL or CUDA; many tests additionally require SM100 or later. Class
    #   attributes hold the module-level cute.jit epilogue fixtures as static methods.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: after the base setup, stores the module-level cute.jit
        #   epilogue fixtures (relu, captured affine, row scale, scalar scale, affine with scalar,
        #   tuple aux, captured tuple aux) as static attributes so tests can pass them as
        #   epilogues. Mutates class attributes only.
        super().setUpClass()
        cls.relu_epilogue = staticmethod(relu_epilogue)
        cls.captured_affine_epilogue = staticmethod(captured_affine_epilogue)
        cls.row_scale_epilogue = staticmethod(row_scale_epilogue)
        cls.scalar_scale_epilogue = staticmethod(scalar_scale_epilogue)
        cls.captured_affine_scalar_epilogue = staticmethod(
            captured_affine_scalar_epilogue
        )
        cls.captured_tuple_aux_epilogue = staticmethod(captured_tuple_aux_epilogue)
        cls.tuple_aux_epilogue = staticmethod(tuple_aux_epilogue)

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_with_c_alpha_beta_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: runs gemm_epilogue with a ReLU epilogue, a
        #   column-major C input, alpha 0.5, beta 1.25 and a preallocated column-major output
        #   buffer. Asserts the returned tensor is that same buffer and that it matches an eager
        #   float32 reference (within the low-precision comparison tolerance) of the ReLU of alpha
        #   times the product plus beta times C. Seeds the RNG.
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(0)
        m, n, k = 128, 128, 64
        a = self.makeTensor(k, m).t()
        b = self.makeTensor(k, n)
        c = self.makeTensor(n, m).t()

        out_buffer = torch.empty_strided((m, n), (1, m), device="cuda", dtype=a.dtype)
        out = gemm_epilogue(
            a,
            b,
            self.relu_epilogue,
            "test_flex_gemm_relu_c",
            C=c,
            alpha=0.5,
            beta=1.25,
            out=out_buffer,
        )
        self.assertIs(out, out_buffer)
        low_precision_expected = (
            (0.5 * (a @ b).float() + 1.25 * c.float()).relu().to(out.dtype)
        )
        high_precision_expected = (
            0.5 * (a.double() @ b.double()) + 1.25 * c.double()
        ).relu()
        self.assertMatchesLowPrecisionEager(
            out, low_precision_expected, high_precision_expected, k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_batched_epilogue_beta_zero_ignores_nan_c(self):
        # [FUNCTION DOC] SM100+ CUDA test: batched gemm_epilogue with beta zero and a C filled
        #   with NaN. Output must contain no NaN (beta zero must not read C), and must match the
        #   eager baddbmm with beta zero, alpha 1.5 followed by ReLU.
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(7)
        batch, m, n, k = 2, 128, 128, 64
        a = self.makeTensor(batch, m, k)
        b = self.makeTensor(batch, k, n)
        c = torch.full((m, n), float("nan"), device="cuda", dtype=a.dtype)

        out = gemm_epilogue(
            a,
            b,
            self.relu_epilogue,
            "test_flex_gemm_batched_beta_zero",
            C=c,
            alpha=1.5,
            beta=0,
        )

        self.assertFalse(torch.isnan(out).any())
        self.assertMatchesLowPrecisionEager(
            out,
            torch.baddbmm(c, a, b, beta=0, alpha=1.5).relu(),
            torch.baddbmm(c.double(), a.double(), b.double(), beta=0, alpha=1.5).relu(),
            k,
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_explicit_arg_kind_disambiguates_row_arg(self):
        # [FUNCTION DOC] SM100+ CUDA test: with an M of 1, passes a row-shaped float32 captured
        #   argument together with the explicit kind row so the ambiguous shape is resolved;
        #   checks the float32 output equals the product times the row scale.
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(2)
        m, n, k = 1, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)

        out = gemm_epilogue(
            a,
            b,
            self.row_scale_epilogue,
            "test_flex_gemm_row_scale",
            out_dtype=torch.float32,
            epilogue_args=(row_scale,),
            epilogue_arg_kinds=("row",),
        )
        self.assertMatchesLowPrecisionEager(
            out,
            (a @ b).float() * row_scale,
            (a.double() @ b.double()) * row_scale.double(),
            k,
        )

    def test_explicit_arg_kind_disambiguates_col_arg_shape(self):
        # [FUNCTION DOC] CPU test: with a 128 by 64 A, a 64 by 1 B and a 128 by 1 captured
        #   argument, an explicit kind col is accepted by resolve_epilogue_arg_kinds and returned
        #   unchanged.
        from torch._inductor.kernel.flex_gemm.runtime import resolve_epilogue_arg_kinds

        a = torch.empty(128, 64)
        b = torch.empty(64, 1)
        col_bias = torch.empty(128, 1)

        self.assertEqual(
            resolve_epilogue_arg_kinds(a, b, (col_bias,), ("col",)),
            ("col",),
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_infers_captured_arg_kinds(self):
        # [FUNCTION DOC] SM100+ CUDA test: captured arguments of shapes column vector, row
        #   vector and full tile are passed without explicit kinds; the runtime must infer their
        #   kinds and the float32 output must match the affine-plus-ReLU reference.
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(4)
        m, n, k = 128, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        col_bias = self.makeTensor(m, 1, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, dtype=torch.float32)

        out = gemm_epilogue(
            a,
            b,
            self.captured_affine_epilogue,
            "test_flex_gemm_infer_captured_args",
            out_dtype=torch.float32,
            epilogue_args=(col_bias, row_scale, tile_bias),
        )
        low_precision_expected = (
            ((a @ b).float() + col_bias) * row_scale + tile_bias
        ).relu()
        high_precision_expected = (
            (a.double() @ b.double() + col_bias.double()) * row_scale.double()
            + tile_bias.double()
        ).relu()
        self.assertMatchesLowPrecisionEager(
            out, low_precision_expected, high_precision_expected, k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_infers_scalar_captured_arg_kind(self):
        # [FUNCTION DOC] SM100+ CUDA test: asserts a one by one captured argument resolves to
        #   kind scalar even when M or N equals 1 (which would also fit row or col), then runs
        #   gemm_epilogue with the scalar scale epilogue and compares with the product times the
        #   scalar.
        from torch._inductor.kernel.flex_gemm.runtime import (
            gemm_epilogue,
            resolve_epilogue_arg_kinds,
        )

        torch.manual_seed(5)
        m, n, k = 128, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        scale = self.makeTensor(1, 1, dtype=torch.float32)

        # A [1, 1] arg is a scalar even when M == 1 or N == 1 would also
        # make it a valid row/col broadcast.
        self.assertEqual(
            resolve_epilogue_arg_kinds(
                torch.empty(1, k), torch.empty(k, 1), (torch.empty(1, 1),), ()
            ),
            ("scalar",),
        )

        out = gemm_epilogue(
            a,
            b,
            self.scalar_scale_epilogue,
            "test_flex_gemm_infer_scalar_arg",
            out_dtype=torch.float32,
            epilogue_args=(scale,),
        )
        self.assertMatchesLowPrecisionEager(
            out,
            (a @ b).float() * scale,
            (a.double() @ b.double()) * scale.double(),
            k,
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_validation_rejects_unsupported_epilogue_arg_combinations(self):
        # [FUNCTION DOC] SM100+ test of argument validation in gemm_epilogue. Each case must
        #   raise: captured args together with C (NotImplementedError), non-default alpha with
        #   captured args (NotImplementedError), an unknown arg kind (NotImplementedError), a row
        #   arg with a transposed shape (RuntimeError), and a strided A or output that is neither
        #   row- nor column-major (NotImplementedError).
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        a = self.makeTensor(128, 64)
        b = self.makeTensor(64, 128)
        c = self.makeTensor(128, 128)
        row_scale = self.makeTensor(1, 128, dtype=torch.float32)

        with self.assertRaisesRegex(NotImplementedError, "cannot be combined with C"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_c_args",
                C=c,
                epilogue_args=(row_scale,),
                epilogue_arg_kinds=("row",),
            )
        with self.assertRaisesRegex(NotImplementedError, "non-default alpha/beta"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_alpha_args",
                alpha=0.5,
                epilogue_args=(row_scale,),
                epilogue_arg_kinds=("row",),
            )
        with self.assertRaisesRegex(NotImplementedError, "tile/row/col"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_bad_kind",
                epilogue_args=(row_scale,),
                epilogue_arg_kinds=("diag",),
            )
        with self.assertRaisesRegex(RuntimeError, "row epilogue arg shape"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_bad_shape",
                epilogue_args=(row_scale.t(),),
                epilogue_arg_kinds=("row",),
            )
        bad_layout = self.makeTensor(256, 128)[::2, ::2]
        with self.assertRaisesRegex(NotImplementedError, "row- or column-major"):
            gemm_epilogue(
                bad_layout,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_bad_layout",
            )
        bad_out_layout = self.makeTensor(256, 128)[::2, ::2]
        with self.assertRaisesRegex(NotImplementedError, "row- or column-major"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_bad_out_layout",
                out=bad_out_layout,
            )

    @parametrize("group", (33, 64))
    def test_runtime_plan_rejects_cross_warp_feed_main_group(self, group):
        # [FUNCTION DOC] Parametrized by group size (33 or 64). Building a runtime local-reduce
        #   plan with axis 0 and feeds_main true must raise NotImplementedError because feed-main
        #   supports only same-warp axis-0 groups of at most 32.
        with self.assertRaisesRegex(
            NotImplementedError, "same-warp axis-0 groups <= 32"
        ):
            self.runtimeLocalReducePlan(
                group=group,
                axis=0,
                feeds_main=True,
            )

    def test_runtime_plan_rejects_axis_one_feed_main(self):
        # [FUNCTION DOC] Building a runtime local-reduce plan with axis 1 and feeds_main true
        #   must raise NotImplementedError (feed-main supports only axis 0).
        with self.assertRaisesRegex(NotImplementedError, "supports only axis 0"):
            self.runtimeLocalReducePlan(
                group=8,
                axis=1,
                feeds_main=True,
            )

    def test_runtime_validation_rejects_non_divisible_local_reduce_group(self):
        # [FUNCTION DOC] validate_runtime_local_reduce with a feed-main axis-0 group of 8 on a
        #   tensor with 130 rows must raise RuntimeError since the group must divide the row
        #   count.
        from torch._inductor.kernel.flex_gemm.runtime import (
            validate_runtime_local_reduce,
        )

        plan = self.runtimeLocalReducePlan(
            group=8,
            axis=0,
            feeds_main=True,
        )
        with self.assertRaisesRegex(RuntimeError, "must divide"):
            validate_runtime_local_reduce(
                plan,
                torch.empty(130, 64),
                (130, 64),
                None,
                1.0,
                1.0,
            )

    def test_runtime_validation_rejects_local_reduce_rank_and_output_shape(self):
        # [FUNCTION DOC] Checks validate_runtime_local_reduce and the output-shape validator
        #   reject: a 3-D (batched) input for a compressed plan (NotImplementedError, only 2-D
        #   mm), a compressed output whose shape differs from the expected group-compressed shape
        #   (RuntimeError), and a column-major local-reduce output tensor (NotImplementedError,
        #   must be row-major).
        from torch._inductor.kernel.flex_gemm.constraints import (
            validate_local_reduce_out_shape,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            validate_runtime_local_reduce,
        )

        compressed_plan = self.runtimeLocalReducePlan(
            out=torch.empty(128, 8), group=8, axis=1
        )
        with self.assertRaisesRegex(NotImplementedError, "2-D aten.mm"):
            validate_runtime_local_reduce(
                compressed_plan,
                torch.empty(1, 128, 64),
                (1, 128, 64),
                None,
                1.0,
                1.0,
            )
        with self.assertRaisesRegex(RuntimeError, "local_reduce_out shape"):
            validate_local_reduce_out_shape((128, 7), (128, 8))
        column_major_plan = self.runtimeLocalReducePlan(
            out=torch.empty_strided((128, 8), (1, 128), device="cuda"),
            group=8,
            axis=1,
        )
        with self.assertRaisesRegex(NotImplementedError, "row-major"):
            validate_runtime_local_reduce(
                column_major_plan,
                torch.empty(128, 64, device="cuda"),
                (128, 64),
                None,
                1.0,
                1.0,
            )

    def test_runtime_validation_rejects_local_reduce_with_c_alpha_beta(self):
        # [FUNCTION DOC] For both a compressed-output plan and a feed-main plan,
        #   validate_runtime_local_reduce must raise NotImplementedError when a C tensor is
        #   supplied or alpha or beta differs from one.
        from torch._inductor.kernel.flex_gemm.runtime import (
            validate_runtime_local_reduce,
        )

        compressed_plan = self.runtimeLocalReducePlan(
            out=torch.empty(128, 8), group=8, axis=1
        )
        feed_main_plan = self.runtimeLocalReducePlan(
            group=8,
            axis=0,
            feeds_main=True,
        )
        for plan in (compressed_plan, feed_main_plan):
            for effective_C, alpha, beta in (
                (torch.empty(128, 64), 1.0, 1.0),
                (None, 0.5, 1.0),
                (None, 1.0, 0.5),
            ):
                with self.assertRaisesRegex(NotImplementedError, "C/alpha/beta"):
                    validate_runtime_local_reduce(
                        plan,
                        torch.empty(128, 64),
                        (128, 64),
                        effective_C,
                        alpha,
                        beta,
                    )

    @skipIfNoCuteDSL
    def test_quack_feed_main_host_guards_match_runtime_contract(self):
        # [FUNCTION DOC] Calls the vendored Quack gemm_act on small CPU bf16 tensors with a
        #   feed-main local-reduce setup and a patched device capacity of SM100, checking host
        #   guards raise before any kernel work: C, alpha or beta given (NotImplementedError); a
        #   group that does not divide the TensorSSA fragment width (RuntimeError); a group that
        #   does not divide the output extent (RuntimeError); and missing tensor epilogue function
        #   and key (RuntimeError). Requires CuTe DSL.
        from torch._vendor.quack.gemm_act import gemm_act

        a = torch.empty(1, 4, 8, dtype=torch.bfloat16)
        b = torch.empty(1, 8, 8, dtype=torch.bfloat16)
        out = torch.empty(1, 4, 8, dtype=torch.bfloat16)
        c = torch.empty(1, 4, 8, dtype=torch.bfloat16)

        def tensor_epilogue(acc, local_reduce0):
            # [FUNCTION DOC] HELPER: epilogue stub taking the accumulator and local_reduce0 and
            #   returning the accumulator unchanged.
            return acc

        def call_gemm(
            C=None,
            mat1=a,
            output=out,
            local_reduce_group=8,
            **kwargs,
        ):
            # [FUNCTION DOC] HELPER: invokes gemm_act with the shared tensors, a feed-main
            #   axis-0 local reduce setup and tile sizes; C, mat1, output, local_reduce_group and
            #   extra keyword arguments (alpha, beta) are overridable. Returns whatever gemm_act
            #   returns.
            return gemm_act(
                mat1,
                b,
                None,
                C,
                output,
                None,
                None,
                128,
                128,
                1,
                1,
                tensor_epilogue_fn=tensor_epilogue,
                local_reduce_feeds_main=True,
                local_reduce_group=local_reduce_group,
                local_reduce_axis=0,
                local_reduce_combine_key="combine",
                local_reduce_finalize_key="finalize",
                device_capacity_override=(10, 0),
                **kwargs,
            )

        for kwargs in ({"C": c}, {"alpha": 0.5}, {"beta": 0.5}):
            with self.assertRaisesRegex(NotImplementedError, "C/alpha/beta"):
                call_gemm(**kwargs)

        with self.assertRaisesRegex(RuntimeError, "divide TensorSSA fragment width"):
            call_gemm(local_reduce_group=3)
        with self.assertRaisesRegex(RuntimeError, "divide the selected GEMM output"):
            call_gemm(
                mat1=torch.empty(1, 6, 8, dtype=torch.bfloat16),
                output=torch.empty(1, 6, 8, dtype=torch.bfloat16),
                local_reduce_group=4,
            )

        with self.assertRaisesRegex(
            RuntimeError, "requires tensor_epilogue_fn or tensor_epilogue_key"
        ):
            gemm_act(
                a,
                b,
                None,
                None,
                out,
                None,
                None,
                128,
                128,
                1,
                1,
                local_reduce_feeds_main=True,
                local_reduce_group=8,
                local_reduce_axis=0,
                local_reduce_combine_key="combine",
                local_reduce_finalize_key="finalize",
                device_capacity_override=(10, 0),
            )

    def test_local_reduce_callbacks_reject_missing_functions(self):
        # [FUNCTION DOC] Constructing FlexGemmLocalReduceCallbacks with a missing combine or
        #   finalize function must raise RuntimeError about generated local-reduce callbacks.
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceCallbacks,
        )

        with self.assertRaisesRegex(RuntimeError, "generated local-reduce callbacks"):
            FlexGemmLocalReduceCallbacks(None, lambda value: value)
        with self.assertRaisesRegex(RuntimeError, "generated local-reduce callbacks"):
            FlexGemmLocalReduceCallbacks(lambda lhs, rhs: lhs, None)

    def test_local_reduce_callbacks_use_generated_cache_keys(self):
        # [FUNCTION DOC] Checks local_reduce_callback_key: a function carrying a cache-key hook
        #   returns that hook's value; a plain lambda without the hook returns the fallback key.
        from torch._inductor.kernel.flex_gemm.runtime import local_reduce_callback_key

        def combine(lhs, rhs):
            # [FUNCTION DOC] HELPER: dummy combine callback returning its left operand; the test
            #   attaches a cache-key attribute to it.
            return lhs

        combine.__cache_key__ = lambda: "combine-cache-key"
        self.assertEqual(
            local_reduce_callback_key(combine, "fallback"), "combine-cache-key"
        )
        self.assertEqual(
            local_reduce_callback_key(lambda value: value, "fallback"), "fallback"
        )

    def test_local_reduce_plan_rejects_invalid_group_axis(self):
        # [FUNCTION DOC] FlexGemmLocalReduceGeometry must raise RuntimeError for a non-positive
        #   group and for an axis other than 0 or 1.
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )

        with self.assertRaisesRegex(
            RuntimeError, "local_reduce_group must be positive"
        ):
            FlexGemmLocalReduceGeometry(0, 0)
        with self.assertRaisesRegex(RuntimeError, "local_reduce_axis must be 0 or 1"):
            FlexGemmLocalReduceGeometry(8, 2)

    def test_runtime_local_reduce_plan_rejects_missing_callbacks(self):
        # [FUNCTION DOC] FlexGemmRuntimeLocalReducePlan must raise RuntimeError when it has
        #   neither an output nor feeds_main, when an output is given without callbacks, and when
        #   feeds_main is set without callbacks, each with the stated messages.
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeLocalReducePlan,
        )

        with self.assertRaisesRegex(RuntimeError, "local_reduce_out"):
            FlexGemmRuntimeLocalReducePlan(FlexGemmLocalReduceGeometry(8, 0))
        with self.assertRaisesRegex(RuntimeError, "generated local-reduce callbacks"):
            FlexGemmRuntimeLocalReducePlan(
                FlexGemmLocalReduceGeometry(8, 0), out=torch.empty(1)
            )
        with self.assertRaisesRegex(RuntimeError, "generated local-reduce callbacks"):
            FlexGemmRuntimeLocalReducePlan(
                FlexGemmLocalReduceGeometry(8, 0), feeds_main=True
            )

    def test_local_reduce_plan_uses_explicit_consumers(self):
        # [FUNCTION DOC] Builds runtime plans for feed-only, shared (output plus feed) and
        #   store-only cases and checks the feeds_main flag on each; checks the template-level
        #   local-reduce config feeds_main flag with and without output index; and checks
        #   local_reduce_gemm_act_kwargs sets the feed-main flag and a returns-local-reduce flag
        #   that is true only when the plan has an output as well as feeding main.
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceCallbacks,
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeLocalReducePlan,
            local_reduce_gemm_act_kwargs,
        )
        from torch._inductor.kernel.flex_gemm.template import (
            FlexGemmEpilogueLocalReduceConfig,
        )

        callbacks = FlexGemmLocalReduceCallbacks(
            lambda lhs, rhs: lhs, lambda value: value
        )
        geometry = FlexGemmLocalReduceGeometry(8, 0)
        out = torch.empty(1)
        feed_plan = FlexGemmRuntimeLocalReducePlan(
            geometry, callbacks=callbacks, feeds_main=True
        )
        shared_plan = FlexGemmRuntimeLocalReducePlan(
            geometry,
            out=out,
            callbacks=callbacks,
            feeds_main=True,
        )
        store_plan = FlexGemmRuntimeLocalReducePlan(
            geometry, out=out, callbacks=callbacks
        )
        self.assertTrue(feed_plan.feeds_main)
        self.assertTrue(shared_plan.feeds_main)
        self.assertFalse(store_plan.feeds_main)
        self.assertTrue(
            FlexGemmEpilogueLocalReduceConfig(geometry, feeds_main=True).feeds_main
        )
        self.assertTrue(
            FlexGemmEpilogueLocalReduceConfig(
                geometry, out_index=0, feeds_main=True
            ).feeds_main
        )
        self.assertFalse(
            FlexGemmEpilogueLocalReduceConfig(geometry, out_index=0).feeds_main
        )

        feed_kwargs = local_reduce_gemm_act_kwargs(
            feed_plan, None, ("combine", "finalize")
        )
        self.assertTrue(feed_kwargs["local_reduce_feeds_main"])
        self.assertFalse(feed_kwargs["tensor_epilogue_returns_local_reduce"])
        shared_kwargs = local_reduce_gemm_act_kwargs(
            shared_plan, out, ("combine", "finalize")
        )
        self.assertTrue(shared_kwargs["local_reduce_feeds_main"])
        self.assertTrue(shared_kwargs["tensor_epilogue_returns_local_reduce"])

    def test_output_plan_rejects_invalid_state(self):
        # [FUNCTION DOC] Checks validation in output plan classes using FX placeholder nodes:
        #   output plans reject non-node outputs, local-reduce matches reject non-node sources,
        #   output local-reduce plans require a match and store with a valid node and non-negative
        #   index (RuntimeError), and tuple_output_plan rejects non-node outputs
        #   (NotImplementedError). Then confirms valid plans with a store or with feeds_main
        #   construct without error.
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            FlexGemmLocalReduceAnalysis,
            FlexGemmLocalReduceMatch,
            FlexGemmLocalReduceStore,
            FlexGemmOutputLocalReducePlan,
            FlexGemmOutputPlan,
            tuple_output_plan,
        )
        from torch._inductor.kernel.gemm_epilogue import GemmEpilogueGraph

        graph = torch.fx.Graph()
        node = graph.placeholder("x")
        aux = graph.placeholder("aux")
        geometry = FlexGemmLocalReduceGeometry(8, 0)
        match = FlexGemmLocalReduceMatch(aux, geometry)
        analysis = FlexGemmLocalReduceAnalysis(
            GemmEpilogueGraph(dependencies={}, normalized_nodes={})
        )
        with self.assertRaisesRegex(RuntimeError, "output nodes"):
            FlexGemmOutputPlan(object())
        with self.assertRaisesRegex(RuntimeError, "output nodes"):
            FlexGemmOutputPlan(node, (object(),))
        with self.assertRaisesRegex(RuntimeError, "tensor nodes"):
            FlexGemmLocalReduceMatch(object(), geometry)
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmOutputLocalReducePlan(object())
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmOutputLocalReducePlan(match)
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmLocalReduceStore(object(), 0)
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmLocalReduceStore(aux, -1)
        with self.assertRaisesRegex(NotImplementedError, "tensor outputs"):
            tuple_output_plan(object(), (), analysis)
        with self.assertRaisesRegex(NotImplementedError, "tensor outputs"):
            tuple_output_plan(node, (object(),), analysis)
        FlexGemmOutputPlan(
            node,
            (aux,),
            FlexGemmOutputLocalReducePlan(
                match, store=FlexGemmLocalReduceStore(aux, 0)
            ),
        )
        FlexGemmOutputPlan(
            node,
            (aux,),
            FlexGemmOutputLocalReducePlan(match, feeds_main=True),
        )

    def test_output_plan_uses_shared_reduction_plan(self):
        # [FUNCTION DOC] Builds an output plan whose local reduce has a max reduction, a store
        #   at index zero and feeds_main, and asserts its reduction_plan equals the shared
        #   GemmReductionPlan with the node names, group 8, axis 0, max, identity, the output name
        #   and feed output set.
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            FlexGemmLocalReduceMatch,
            FlexGemmLocalReduceStore,
            FlexGemmOutputLocalReducePlan,
            FlexGemmOutputPlan,
        )
        from torch._inductor.kernel.gemm_epilogue import GemmReductionPlan

        graph = torch.fx.Graph()
        output = graph.placeholder("output")
        reduced = graph.placeholder("reduced")
        match = FlexGemmLocalReduceMatch(
            reduced,
            FlexGemmLocalReduceGeometry(8, 0),
            reduction_type="max",
        )
        outputs = FlexGemmOutputPlan(
            output,
            local_reduce=FlexGemmOutputLocalReducePlan(
                match,
                store=FlexGemmLocalReduceStore(reduced, 0),
                feeds_main=True,
            ),
        )

        self.assertEqual(
            outputs.reduction_plan,
            GemmReductionPlan(
                "reduced",
                8,
                0,
                "max",
                "identity",
                "output",
                feeds_main=True,
                feed_output="output",
            ),
        )

    def test_ordered_outputs_restore_local_reduce_position(self):
        # [FUNCTION DOC] Checks flex_gemm_ordered_outputs places the local-reduce output at the
        #   requested index among main and two aux outputs (positions one, two, three of four).
        from torch._inductor.kernel.flex_gemm.lowering import flex_gemm_ordered_outputs

        expected_outputs = (
            ("main", "local", "aux0", "aux1"),
            ("main", "aux0", "local", "aux1"),
            ("main", "aux0", "aux1", "local"),
        )
        for index, expected in enumerate(expected_outputs):
            self.assertEqual(
                flex_gemm_ordered_outputs("main", ("aux0", "aux1"), ("local",), index),
                expected,
            )

    def test_local_reduce_aux_result_requires_grouped_source(self):
        # [FUNCTION DOC] Checks the emitter aux_result helper: an aux node without a grouped
        #   source mapping raises NotImplementedError, with a mapping it returns the mapped
        #   temporary name, and a None node returns None.
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            FlexGemmEpilogueEmitter,
        )

        graph = torch.fx.Graph()
        aux = graph.placeholder("aux")
        with self.assertRaisesRegex(NotImplementedError, "grouped TensorSSA"):
            FlexGemmEpilogueEmitter.aux_result(aux, {})
        self.assertEqual(FlexGemmEpilogueEmitter.aux_result(aux, {aux: "tmp0"}), "tmp0")
        self.assertIsNone(FlexGemmEpilogueEmitter.aux_result(None, {}))

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_swap_ab_rejects_local_reduce_aux(self):
        # [FUNCTION DOC] SM100+ CUDA test: gemm_epilogue with a compressed local-reduce output
        #   plan and a swap_ab config key must raise NotImplementedError saying swap_ab is not
        #   supported.
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeLocalReducePlan,
            FlexGemmRuntimeOutputPlan,
            gemm_epilogue,
        )

        m, n, k = 128, 128, 64
        group = 16
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        local_reduce_out = torch.empty(
            m, n // group, device="cuda", dtype=torch.float32
        )
        swap_key, _ = self.swapAndNonSwapConfigKeys(a.device)

        with self.assertRaisesRegex(NotImplementedError, "do not support swap_ab"):
            gemm_epilogue(
                a,
                b,
                self.relu_epilogue,
                "test_flex_gemm_swap_ab_local_reduce_rejects",
                out_dtype=torch.float32,
                output_plan=FlexGemmRuntimeOutputPlan(
                    local_reduce=FlexGemmRuntimeLocalReducePlan(
                        FlexGemmLocalReduceGeometry(group, 1),
                        out=local_reduce_out,
                    )
                ),
                config_key=swap_key,
            )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_explicit_config_key_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: takes the first two candidate config keys for the
        #   device and runs gemm_epilogue with the row-scale epilogue and an explicit row argument
        #   under each key; each float32 result must match the eager product times row scale.
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        a = self.makeTensor(128, 64)
        b = self.makeTensor(64, 128)
        row_scale = self.makeTensor(1, 128, dtype=torch.float32)

        config_keys = tuple(
            gemm_config_key(config)
            for config in candidate_gemm_configs_for_device(a.device)
        )
        for index, config_key in enumerate(config_keys[:2]):
            out = gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                f"test_flex_gemm_config_key_{index}",
                out_dtype=torch.float32,
                epilogue_args=(row_scale,),
                epilogue_arg_kinds=("row",),
                config_key=config_key,
            )
            self.assertMatchesLowPrecisionEager(
                out,
                (a @ b).float() * row_scale,
                (a.double() @ b.double()) * row_scale.double(),
                a.shape[1],
            )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_fragment_group32_forced_config_extremes_match_reference(self):
        # [FUNCTION DOC] SM100+ CUDA regression test for the axis-1 group-of-32 fragment
        #   contract. Picks, among candidate configs passing the local-reduce validation for group
        #   32 on axis 1, the ones with the smallest and largest tile N, and runs the group32 sum
        #   feed fixture with each forced config. Results must match the float64 per-group
        #   normalization reference within 1e-4 and must differ by over 1e-2 from the half-group
        #   reference (to catch partial-fragment reductions).
        """Pin the empirically verified g32 axis-1 fragment contract per config.

        Every config passing the axis-1 group-32 gate was verified to expose a
        full 32-lane fragment to the generated TensorSSA reduce; keep the
        extreme tile_n configs as regression sentinels for that geometry.
        """
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )
        from torch._inductor.kernel.flex_gemm.constraints import (
            validate_flex_gemm_local_reduce_config,
        )
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        m, n, k, group = 128, 256, 64, 32
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)
        gated = [
            config
            for config in candidate_gemm_configs_for_device(a.device)
            if validate_flex_gemm_local_reduce_config(config, group, 1)
        ]
        self.assertTrue(gated)
        tile_ns = sorted({config.tile_n for config in gated})
        selected = [
            next(config for config in gated if config.tile_n == tile_n)
            for tile_n in (tile_ns[0], tile_ns[-1])
        ]
        x = (a.double() @ b.double()).view(m, -1, group)
        expected = (x * x.sum(-1, keepdim=True).reciprocal()).view(m, n)
        wrong_group = (a.double() @ b.double()).view(m, -1, group // 2)
        wrong_expected = (
            wrong_group * wrong_group.sum(-1, keepdim=True).reciprocal()
        ).view(m, n)
        for config in selected:
            out = gemm_epilogue(
                a,
                b,
                fragment_group32_sum_feed_epilogue,
                f"test_flex_gemm_fragment_group32_tile_n_{config.tile_n}",
                out_dtype=torch.float32,
                config_key=gemm_config_key(config),
            )
            torch.testing.assert_close(out.double(), expected, atol=1e-4, rtol=1e-4)
            # A partial-fragment reduce would match the half-group reference.
            self.assertGreater((out.double() - wrong_expected).abs().max(), 1e-2)

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_physical_feed_main_forced_config_extremes_match_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test of a physical feed-main local reduce (group 32 along
        #   axis 0, callbacks for sum combine and finalize). Selects configs passing local-reduce
        #   validation with extreme tile M and tile N values (deduplicated by key), runs
        #   gemm_epilogue with the divide-by-group-sum fixture and asserts each result matches the
        #   float64 column-group normalization within 1e-4.
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceCallbacks,
            FlexGemmLocalReduceGeometry,
            validate_flex_gemm_local_reduce_config,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeLocalReducePlan,
            FlexGemmRuntimeOutputPlan,
            gemm_epilogue,
        )

        m, n, k, group = 128, 128, 64, 32
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16) + 0.5
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16) + 0.5
        gated = [
            config
            for config in candidate_gemm_configs_for_device(a.device)
            if validate_flex_gemm_local_reduce_config(config, group, 0)
        ]
        self.assertTrue(gated)
        extremes = (
            min(gated, key=lambda config: (config.tile_m, config.tile_n)),
            max(gated, key=lambda config: (config.tile_m, config.tile_n)),
            min(gated, key=lambda config: (config.tile_n, config.tile_m)),
            max(gated, key=lambda config: (config.tile_n, config.tile_m)),
        )
        selected = {gemm_config_key(config): config for config in extremes}.values()
        accumulator = a.double() @ b.double()
        grouped = accumulator.view(m // group, group, n)
        expected = (grouped / grouped.sum(1, keepdim=True)).view(m, n)
        callbacks = FlexGemmLocalReduceCallbacks(
            physical_group_sum_combine, physical_group_sum_finalize
        )
        local_reduce = FlexGemmRuntimeLocalReducePlan(
            FlexGemmLocalReduceGeometry(group, 0),
            callbacks=callbacks,
            feeds_main=True,
        )
        for config in selected:
            out = gemm_epilogue(
                a,
                b,
                physical_group_sum_feed_epilogue,
                (
                    "test_flex_gemm_physical_feed_main_"
                    f"{config.tile_m}_{config.tile_n}_{config.cluster_m}_{config.cluster_n}"
                ),
                out_dtype=torch.float32,
                output_plan=FlexGemmRuntimeOutputPlan(local_reduce=local_reduce),
                config_key=gemm_config_key(config),
            )
            torch.testing.assert_close(out.double(), expected, atol=1e-4, rtol=1e-4)

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("shape", ((128, 512, 256), (512, 128, 256), (256, 256, 256)))
    def test_swap_ab_matches_non_swap_and_eager(self, shape):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by three (M, N, K) shapes: runs the ReLU
        #   epilogue with a swap_ab config key and with a non-swap key; the two float32 results
        #   must be exactly equal (swap_ab only changes tile orientation) and must match the eager
        #   ReLU of the product within tolerance.
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        m, n, k = shape
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        swap_key, non_swap_key = self.swapAndNonSwapConfigKeys(a.device)

        swapped = gemm_epilogue(
            a,
            b,
            self.relu_epilogue,
            "test_flex_gemm_swap_ab_mm",
            out_dtype=torch.float32,
            config_key=swap_key,
        )
        non_swapped = gemm_epilogue(
            a,
            b,
            self.relu_epilogue,
            "test_flex_gemm_non_swap_ab_mm",
            out_dtype=torch.float32,
            config_key=non_swap_key,
        )
        # swap_ab only reorients tile scheduling, so the result is bit-identical.
        self.assertEqual(swapped, non_swapped)
        self.assertMatchesLowPrecisionEager(
            swapped, (a @ b).float().relu(), (a.double() @ b.double()).relu(), k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_swap_ab_with_c_alpha_beta_matches_non_swap(self):
        # [FUNCTION DOC] SM100+ CUDA test: runs a ReLU epilogue with C, alpha 1.5 and beta 0.5
        #   under a swap_ab and a non-swap config key; results must be equal and match the eager
        #   reference.
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        m, n, k = 256, 384, 192
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        c = self.makeTensor(m, n)
        swap_key, non_swap_key = self.swapAndNonSwapConfigKeys(a.device)

        def run(name, config_key):
            # [FUNCTION DOC] HELPER: runs gemm_epilogue on the enclosing test tensors with C,
            #   alpha and beta and the given kernel name and config key; returns the float32
            #   output.
            return gemm_epilogue(
                a,
                b,
                self.relu_epilogue,
                name,
                C=c,
                alpha=1.5,
                beta=0.5,
                out_dtype=torch.float32,
                config_key=config_key,
            )

        swapped = run("test_flex_gemm_swap_ab_addmm", swap_key)
        non_swapped = run("test_flex_gemm_non_swap_ab_addmm", non_swap_key)
        # The transposed C view must reproduce the non-swapped addmm result.
        self.assertEqual(swapped, non_swapped)
        self.assertMatchesLowPrecisionEager(
            swapped,
            (0.5 * c.float() + 1.5 * (a @ b).float()).relu(),
            (0.5 * c.double() + 1.5 * (a.double() @ b.double())).relu(),
            k,
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_swap_ab_captured_args_matches_non_swap(self):
        # [FUNCTION DOC] SM100+ CUDA test: captured column, row and tile arguments with explicit
        #   kinds must give identical results under swap_ab and non-swap configs (row and column
        #   roles swap internally) and match the affine-ReLU eager reference.
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        m, n, k = 128, 384, 256
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        col_bias = self.makeTensor(m, 1, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, dtype=torch.float32)
        swap_key, non_swap_key = self.swapAndNonSwapConfigKeys(a.device)

        def run(name, config_key):
            # [FUNCTION DOC] HELPER: runs gemm_epilogue with the captured affine epilogue,
            #   explicit kinds col, row, tile, the given name and config key; returns the float32
            #   output.
            return gemm_epilogue(
                a,
                b,
                self.captured_affine_epilogue,
                name,
                out_dtype=torch.float32,
                epilogue_args=(col_bias, row_scale, tile_bias),
                epilogue_arg_kinds=("col", "row", "tile"),
                config_key=config_key,
            )

        swapped = run("test_flex_gemm_swap_ab_captured_args", swap_key)
        non_swapped = run("test_flex_gemm_non_swap_ab_captured_args", non_swap_key)
        # Swapped row/col broadcast roles must reproduce the non-swapped result.
        self.assertEqual(swapped, non_swapped)
        high_precision_expected = (
            (a.double() @ b.double() + col_bias.double()) * row_scale.double()
            + tile_bias.double()
        ).relu()
        low_precision_expected = (
            ((a @ b).float() + col_bias) * row_scale + tile_bias
        ).relu()
        self.assertMatchesLowPrecisionEager(
            swapped, low_precision_expected, high_precision_expected, k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_swap_ab_scalar_captured_arg_matches_non_swap(self):
        # [FUNCTION DOC] SM100+ CUDA test: like the swap_ab captured-args test but with an extra
        #   scalar captured argument; swapped and non-swapped outputs must be exactly equal
        #   (scalars do not depend on orientation) and match the eager reference. Seeds the RNG.
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(11)
        m, n, k = 128, 384, 256
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        col_bias = self.makeTensor(m, 1, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, dtype=torch.float32)
        scale = self.makeTensor(1, 1, dtype=torch.float32)
        swap_key, non_swap_key = self.swapAndNonSwapConfigKeys(a.device)

        def run(name, config_key):
            # [FUNCTION DOC] HELPER: runs gemm_epilogue with the affine-with-scalar epilogue,
            #   explicit kinds col, row, tile, scalar, given name and config key; returns the
            #   float32 output.
            return gemm_epilogue(
                a,
                b,
                self.captured_affine_scalar_epilogue,
                name,
                out_dtype=torch.float32,
                epilogue_args=(col_bias, row_scale, tile_bias, scale),
                epilogue_arg_kinds=("col", "row", "tile", "scalar"),
                config_key=config_key,
            )

        swapped = run("test_flex_gemm_swap_ab_scalar_arg", swap_key)
        non_swapped = run("test_flex_gemm_non_swap_ab_scalar_arg", non_swap_key)
        # Scalars are orientation-invariant, so swap_ab must be bit-identical.
        self.assertEqual(swapped, non_swapped)
        high_precision_expected = (
            (
                (a.double() @ b.double() + col_bias.double()) * row_scale.double()
                + tile_bias.double()
            )
            * scale.double()
        ).relu()
        low_precision_expected = (
            (((a @ b).float() + col_bias) * row_scale + tile_bias) * scale
        ).relu()
        self.assertMatchesLowPrecisionEager(
            swapped, low_precision_expected, high_precision_expected, k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_swap_ab_captured_args_tuple_aux_matches_non_swap(self):
        # [FUNCTION DOC] SM100+ CUDA test: captured arguments combined with a tuple-aux epilogue
        #   and a runtime output plan with one aux output. Outputs and aux from the swap_ab and
        #   non-swap configs must be equal and must match the captured tuple aux reference. Seeds
        #   the RNG.
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeOutputPlan,
            gemm_epilogue,
        )

        torch.manual_seed(10)
        m, n, k = 128, 384, 256
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        col_bias = self.makeTensor(m, 1, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, dtype=torch.float32)
        swap_key, non_swap_key = self.swapAndNonSwapConfigKeys(a.device)

        def run(name, config_key):
            # [FUNCTION DOC] HELPER: allocates fresh bf16 main and float32 aux outputs, runs
            #   gemm_epilogue with the captured tuple aux epilogue (aux provided via the output
            #   plan), explicit kinds and the config key; returns the main and aux tensors.
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            aux = torch.empty(m, n, device="cuda", dtype=torch.float32)
            gemm_epilogue(
                a,
                b,
                self.captured_tuple_aux_epilogue,
                name,
                out=out,
                output_plan=FlexGemmRuntimeOutputPlan(aux_outs=(aux,)),
                epilogue_args=(col_bias, row_scale, tile_bias),
                epilogue_arg_kinds=("col", "row", "tile"),
                config_key=config_key,
            )
            return out, aux

        swapped, swapped_aux = run(
            "test_flex_gemm_swap_ab_captured_tuple_aux", swap_key
        )
        non_swapped, non_swapped_aux = run(
            "test_flex_gemm_non_swap_ab_captured_tuple_aux", non_swap_key
        )

        self.assertEqual(swapped, non_swapped)
        self.assertEqual(swapped_aux, non_swapped_aux)
        self.assertCapturedTupleAuxMatchesReference(
            swapped, swapped_aux, a, b, col_bias, row_scale, tile_bias
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_reads_captured_aux_tensors(self):
        # [FUNCTION DOC] SM100+ CUDA test: default config run of the captured affine epilogue
        #   with column, row and tile arguments given explicit kinds; float32 output must match
        #   the eager and float64 references. Seeds the RNG.
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(1)
        m, n, k = 128, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        col_bias = self.makeTensor(m, 1, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, dtype=torch.float32)

        out = gemm_epilogue(
            a,
            b,
            self.captured_affine_epilogue,
            "test_flex_gemm_affine_aux",
            out_dtype=torch.float32,
            epilogue_args=(col_bias, row_scale, tile_bias),
            epilogue_arg_kinds=("col", "row", "tile"),
        )
        low_precision_expected = (
            ((a @ b).float() + col_bias) * row_scale + tile_bias
        ).relu()
        high_precision_expected = (
            (a.double() @ b.double() + col_bias.double()) * row_scale.double()
            + tile_bias.double()
        ).relu()
        self.assertMatchesLowPrecisionEager(
            out, low_precision_expected, high_precision_expected, k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_writes_tuple_aux_out(self):
        # [FUNCTION DOC] SM100+ CUDA test: runs the tuple-aux epilogue with a preallocated main
        #   output and an aux output in the runtime output plan. Asserts the returned tensor is
        #   the main output buffer and both main and aux match the references. Seeds the RNG.
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeOutputPlan,
            gemm_epilogue,
        )

        torch.manual_seed(8)
        m, n, k = 128, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        out = torch.empty(m, n, device="cuda", dtype=torch.float32)
        aux = torch.empty(m, n, device="cuda", dtype=torch.float32)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: eager reference epilogue returning a main value (half of
            #   accumulator plus one) and an aux value (accumulator squared plus two), both in
            #   float32.
            main = (acc.float() + 1.0) * 0.5
            aux = acc.float().square() + 2.0
            return main, aux

        actual = gemm_epilogue(
            a,
            b,
            self.tuple_aux_epilogue,
            "test_flex_gemm_tuple_aux",
            out_dtype=torch.float32,
            out=out,
            output_plan=FlexGemmRuntimeOutputPlan(aux_outs=(aux,)),
        )

        self.assertIs(actual, out)
        self.assertTupleAuxMatchesReference(out, aux, a, b, epilogue_fn)


@instantiate_parametrized_tests
class TestFlexGemmEpilogueHOP(FlexGemmTestCase):
    # [CLASS DOC] Tests flex_gemm at the higher-order-op level: eager and fake-tensor behavior
    #   of the HOP, rejection of unsupported scopes at codegen, compiled runs on SM100,
    #   local-reduce and output-contraction rejections, debug report output, and grouped-layout
    #   planning. Redefines the shared generated-code check (identical logic).
    def assertFlexGemmGeneratedCode(self, code, *checks):
        # [FUNCTION DOC] HELPER: redefinition of the base class generated-code check with
        #   identical logic: runtime import, plan class, aliased gemm_epilogue and its call,
        #   caller patterns, then stream and config_key present and tuned, epilogue_source and
        #   quack imports absent.
        file_check = (
            FileCheck()
            .check("from torch._inductor.kernel.flex_gemm.runtime import (")
            .check("FlexGemmRuntimeLocalReducePlan")
            .check("gemm_epilogue as flex_gemm_epilogue")
            .check("flex_gemm_epilogue(")
        )
        for check in checks:
            file_check = file_check.check(check)
        file_check = (
            file_check.check("stream=stream").check("config_key=").check_not("tuned=")
        )
        file_check = file_check.check_not("epilogue_source=")
        file_check.check_not("from quack").check_not("import quack").run(code)

    def assertOutputContractionRejected(self, a, b, epilogue, error):
        # [FUNCTION DOC] HELPER: resets dynamo, then compiles (inductor, full graph) a call of
        #   flex_gemm on torch.mm with the given epilogue and the QUACK backend, and asserts
        #   compilation raises InductorError whose message matches the error pattern. Arguments:
        #   a, b, epilogue, error pattern.
        """Assert that an output contraction is rejected during compilation."""
        torch._dynamo.reset()
        with self.assertRaisesRegex(InductorError, error):
            torch.compile(
                lambda a, b: flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue,
                    kernel_options={"backend": "QUACK"},
                ),
                backend="inductor",
                fullgraph=True,
            )(a, b)

    def test_supported_op_names_match_dense_scope(self):
        # [FUNCTION DOC] Asserts the exported supported-op-names string equals the slash-joined
        #   list mm, addmm, bmm, baddbmm.
        self.assertEqual(_SUPPORTED_FLEX_GEMM_OP_NAMES, "mm/addmm/bmm/baddbmm")

    @parametrize(
        "case",
        (
            ("mm", torch.mm, lambda a, b: (a, b), lambda a, b: a @ b),
            ("bmm", torch.bmm, lambda a, b: (a, b), lambda a, b: torch.bmm(a, b)),
        ),
        name_fn=lambda case: case[0],
    )
    def test_default_backend_eager_matches_reference(self, case):
        # [FUNCTION DOC] Parametrized over mm and bmm: with the default (non-Quack) backend,
        #   flex_gemm run eagerly on CPU tensors must return the ReLU of the matmul result. For
        #   mm, the batch dimension is dropped from inputs.
        _, op, args_fn, ref_fn = case
        a = torch.randn(2, 8, 16)
        b = torch.randn(2, 16, 12)
        if op is torch.mm:
            a = a[0]
            b = b[0]

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        actual = flex_gemm(op, args_fn(a, b), epilogue_fn)

        torch.testing.assert_close(actual, epilogue_fn(ref_fn(a, b)))

    def test_default_backend_eager_tuple_aux_matches_reference(self):
        # [FUNCTION DOC] Eager default backend with a two-output epilogue (ReLU, accumulator
        #   plus one): flex_gemm returns a pair equal to applying the epilogue to the plain
        #   matmul.
        a = torch.randn(8, 16)
        b = torch.randn(16, 12)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU of the accumulator and the
            #   accumulator plus one.
            return acc.relu(), acc + 1

        actual, aux = flex_gemm(torch.mm, (a, b), epilogue_fn)
        expected, expected_aux = epilogue_fn(a @ b)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(aux, expected_aux)

    def test_fake_tensor_mode_tuple_aux_returns_fake_tensors(self):
        # [FUNCTION DOC] Under a FakeTensorMode, flex_gemm with a tuple epilogue returns two
        #   fake tensors of shape 8 by 12, float32, both belonging to that fake mode.
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode() as mode:
            a = mode.from_tensor(torch.randn(8, 16))
            b = mode.from_tensor(torch.randn(16, 12))

            def epilogue_fn(acc):
                # [FUNCTION DOC] HELPER: epilogue returning ReLU of the accumulator and the
                #   accumulator plus one.
                return acc.relu(), acc + 1

            actual, aux = flex_gemm(torch.mm, (a, b), epilogue_fn)

        self.assertEqual(actual.shape, torch.Size([8, 12]))
        self.assertEqual(aux.shape, torch.Size([8, 12]))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(aux.dtype, torch.float32)
        self.assertIs(actual.fake_mode, mode)
        self.assertIs(aux.fake_mode, mode)

    def test_autograd_is_not_implemented(self):
        # [FUNCTION DOC] Calling backward on the sum of a flex_gemm eager result with
        #   grad-requiring inputs must raise RuntimeError mentioning flex_gemm (autograd is not
        #   implemented).
        a = torch.randn(8, 16, requires_grad=True)
        b = torch.randn(16, 12, requires_grad=True)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        actual = flex_gemm(torch.mm, (a, b), epilogue_fn)

        with self.assertRaisesRegex(RuntimeError, "flex_gemm"):
            actual.sum().backward()

    @skipIfNoCuteDSL
    def test_generated_captured_arg_rejects_unsupported_shape(self):
        # [FUNCTION DOC] Compiling flex_gemm (QUACK) whose epilogue captures a 1-D scale of
        #   length N must fail with an error saying captured tensor epilogue args must currently
        #   match supported shapes.
        def fn(a, b, scale):
            # [FUNCTION DOC] HELPER: compiled function calling flex_gemm on mm with an epilogue
            #   multiplying the accumulator by the captured scale.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: acc * scale,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, 5)
        scale = torch.randn(5)

        with self.assertRaisesRegex(
            Exception,
            "captured tensor epilogue args currently must match",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b, scale)

    @skipIfNoCuteDSL
    def test_generated_captured_arg_rejects_addmm_scope(self):
        # [FUNCTION DOC] Compiling flex_gemm on addmm with a captured tensor read in the
        #   epilogue must fail with a message that captured tensor reads currently support only
        #   aten.mm.
        def fn(bias, a, b, scale):
            # [FUNCTION DOC] HELPER: compiled function calling flex_gemm on addmm with an
            #   epilogue multiplying the accumulator by the captured scale.
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                lambda acc: acc * scale,
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 5)
        a = torch.randn(4, 8)
        b = torch.randn(8, 5)
        scale = torch.randn(4, 5)

        with self.assertRaisesRegex(
            Exception,
            "captured tensor reads currently support only aten.mm",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(bias, a, b, scale)

    @skipIfNoCuteDSL
    def test_generated_tuple_aux_rejects_unsupported_scope(self):
        # [FUNCTION DOC] Compiling flex_gemm on addmm with a tuple-output epilogue must fail
        #   with a message that support is currently only for aten.mm.
        def addmm_fn(bias, a, b):
            # [FUNCTION DOC] HELPER: compiled function calling flex_gemm on addmm with an
            #   epilogue returning ReLU and the accumulator plus one.
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                lambda acc: (acc.relu(), acc + 1),
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 5)
        a = torch.randn(4, 8)
        b = torch.randn(8, 5)

        with self.assertRaisesRegex(Exception, "currently support only aten.mm"):
            torch.compile(addmm_fn, backend="inductor", fullgraph=True)(bias, a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_compiled_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: compiles flex_gemm (QUACK backend) with a
        #   plus-one-then-ReLU epilogue on bf16 128 by 64 and 64 by 128 inputs and checks it
        #   against low-precision eager and float64 references.
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue adding one to the accumulator and applying ReLU.
            return (acc + 1).relu()

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.mm,
            (a, b),
            epilogue_fn,
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_alpha_clamp_compiled_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: compiled QUACK flex_gemm with an epilogue using an
        #   add with alpha followed by clamp at zero; checked against eager and float64
        #   references.
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue adding two scaled by alpha 0.25 to the accumulator
            #   and clamping the minimum to zero.
            return torch.add(acc, 2.0, alpha=0.25).clamp(min=0.0)

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.mm,
            (a, b),
            epilogue_fn,
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_dynamic_shapes_compiled_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: compiles with dynamic shapes enabled and runs the
        #   plus-one ReLU epilogue with M equal to 128 then 256, checking each result against
        #   references.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue adding one to the accumulator and applying ReLU.
            return (acc + 1).relu()

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the QUACK backend and the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        for m in (128, 256):
            a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
            actual = compiled(a, b)
            self.assertMatchesLowPrecisionEager(
                actual,
                epilogue_fn(a @ b),
                epilogue_fn(a.double() @ b.double()),
                a.shape[1],
            )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_swap_ab_dynamic_shapes_tuned_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: with tuning on and candidate configs patched to a
        #   single swap_ab config, compiles with dynamic shapes and runs two (M, N) sizes,
        #   checking results against references. Asserts that at least one swap_ab candidate
        #   exists.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue adding one to the accumulator and applying ReLU.
            return (acc + 1).relu()

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the QUACK backend, tuned on.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        from torch._inductor.template_heuristics import (
            flex_gemm as flex_gemm_heuristics,
        )

        device = torch.device("cuda")
        swap_configs = [
            config
            for config in flex_gemm_heuristics.candidate_gemm_configs_for_device(device)
            if config.swap_ab
        ]
        self.assertTrue(swap_configs)
        with mock.patch(
            "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
            return_value=swap_configs[:1],
        ):
            compiled = torch.compile(
                fn, backend="inductor", fullgraph=True, dynamic=True
            )
            for m, n in ((128, 128), (256, 192)):
                a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
                actual = compiled(a, b)
                self.assertMatchesLowPrecisionEager(
                    actual,
                    epilogue_fn(a @ b),
                    epilogue_fn(a.double() @ b.double()),
                    a.shape[1],
                )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("tile", lambda m, n: (m, n)),
            ("row", lambda m, n: (1, n)),
            ("col", lambda m, n: (m, 1)),
        ),
        name_fn=lambda case: case[0],
    )
    @parametrize(
        "tuned",
        (False, True),
        name_fn=lambda tuned: "tuned" if tuned else "untuned",
    )
    def test_mm_dynamic_shapes_reads_captured_tensor_epilogue_arg(self, case, tuned):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by captured arg layout (tile, row, col)
        #   and tuned or untuned: compiles with dynamic shapes a flex_gemm whose epilogue
        #   multiplies the float32 accumulator by a captured float32 tensor and applies ReLU,
        #   running two shapes. When tuned, candidate configs are patched to the first two. Resets
        #   dynamo; checks against references.
        torch._dynamo.reset()
        _, shape_fn = case

        def epilogue_fn(acc, scale):
            # [FUNCTION DOC] HELPER: epilogue multiplying the float32 accumulator by the scale
            #   and applying ReLU.
            return (acc.float() * scale).relu()

        def fn(a, b, scale):
            # [FUNCTION DOC] HELPER: function under compilation, wrapping the epilogue with the
            #   captured scale in a flex_gemm call on mm (QUACK, tuned per parameter).
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, scale),
                kernel_options={"backend": "QUACK", "tuned": tuned},
            )

        config_context = contextlib.nullcontext()
        if tuned:
            from torch._inductor.template_heuristics import (
                flex_gemm as flex_gemm_heuristics,
            )

            configs = flex_gemm_heuristics.candidate_gemm_configs_for_device(
                torch.device("cuda")
            )[:2]
            config_context = mock.patch(
                "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
                return_value=configs,
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        with config_context:
            for m, k, n in ((128, 64, 128), (256, 64, 192)):
                a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
                scale = torch.randn(*shape_fn(m, n), device="cuda", dtype=torch.float32)
                actual = compiled(a, b, scale)
                self.assertMatchesLowPrecisionEager(
                    actual,
                    epilogue_fn(a @ b, scale),
                    epilogue_fn(a.double() @ b.double(), scale.double()),
                    a.shape[1],
                )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("tile", lambda m, n: (m, n)),
            ("row", lambda m, n: (1, n)),
            ("col", lambda m, n: (m, 1)),
            ("scalar", lambda m, n: (1, 1)),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_reads_bool_mask_captured_tensor_epilogue_arg(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by mask layout (tile, row, col, scalar):
        #   a captured boolean mask selects between the accumulator and its negation; the compiled
        #   result must match eager and float64 references.
        _, shape_fn = case

        def epilogue_fn(acc, mask):
            # [FUNCTION DOC] HELPER: epilogue returning the float32 accumulator where the mask
            #   is true and its negation elsewhere.
            acc_float = acc.float()
            return torch.where(mask, acc_float, -acc_float)

        def fn(a, b, mask):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the mask-capturing epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, mask),
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        mask = torch.randint(0, 2, shape_fn(m, n), device="cuda", dtype=torch.bool)

        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b, mask)

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b, mask),
            epilogue_fn(a.double() @ b.double(), mask),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_preserves_integer_scalar_captured_tensor_epilogue_arg(self):
        # [FUNCTION DOC] SM100+ CUDA test: a captured int64 scalar (value above the float32
        #   integer precision limit, odd) must keep integer semantics: the epilogue tests its low
        #   bit to choose the sign. Result matches references and generated code must show a
        #   scalar argument kind.
        def epilogue_fn(acc, selector):
            # [FUNCTION DOC] HELPER: epilogue choosing the accumulator or its negation by the
            #   low bit of an integer selector.
            acc_float = acc.float()
            return torch.where(selector.bitwise_and(1).bool(), acc_float, -acc_float)

        def fn(a, b, selector):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the selector-capturing epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, selector),
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        selector = torch.tensor([[2**24 + 1]], device="cuda", dtype=torch.int64)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, selector
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b, selector),
            epilogue_fn(a.double() @ b.double(), selector),
            a.shape[1],
        )
        FileCheck().check("epilogue_arg_kinds=('scalar',)").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("tile", lambda m, n: (m, n)),
            ("row", lambda m, n: (1, n)),
            ("col", lambda m, n: (m, 1)),
            ("scalar", lambda m, n: (1, 1)),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_promotes_low_precision_captured_tensor_epilogue_arg(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by scale layout (tile, row, col, scalar):
        #   a captured bfloat16 scale is promoted against the float32 accumulator inside the
        #   epilogue; the compiled result is compared with eager and float64 references.
        _, shape_fn = case

        def epilogue_fn(acc, scale):
            # [FUNCTION DOC] HELPER: epilogue multiplying the captured scale by the float32
            #   accumulator.
            return scale * acc.float()

        def fn(a, b, scale):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the scale-capturing epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, scale),
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(*shape_fn(m, n), device="cuda", dtype=torch.bfloat16)

        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b, scale)

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b, scale),
            epilogue_fn(a.double() @ b.double(), scale.double()),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "tuned",
        (False, True),
        name_fn=lambda tuned: "tuned" if tuned else "untuned",
    )
    def test_mm_scalar_captured_arg_fp8_quant_matches_reference(self, tuned):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by tuned or untuned: epilogue multiplies
        #   by the absolute value of a captured 1 by 1 float32 scale and casts to float8 e4m3.
        #   Output dtype must be that fp8 type, results match references, and generated code must
        #   record a scalar epilogue argument kind. For the tuned case the first two candidate
        #   configs are patched. Resets dynamo.
        torch._dynamo.reset()
        m, k, n = 128, 64, 128

        def epilogue_fn(acc, scale):
            # [FUNCTION DOC] HELPER: epilogue multiplying the accumulator by the absolute value
            #   of scale.
            return acc * scale.abs()

        def fn(a, b, scale):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the fp8-casting epilogue capturing the scale.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc.float(), scale).to(torch.float8_e4m3fn),
                kernel_options={"backend": "QUACK", "tuned": tuned},
            )

        config_context = contextlib.nullcontext()
        if tuned:
            from torch._inductor.template_heuristics import (
                flex_gemm as flex_gemm_heuristics,
            )

            configs = flex_gemm_heuristics.candidate_gemm_configs_for_device(
                torch.device("cuda")
            )[:2]
            config_context = mock.patch(
                "torch._inductor.template_heuristics.flex_gemm.candidate_gemm_configs_for_device",
                return_value=configs,
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([[-0.25]], device="cuda", dtype=torch.float32)
        with config_context:
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b, scale
            )

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn((a @ b).float(), scale).to(torch.float8_e4m3fn),
            epilogue_fn(a.double() @ b.double(), scale.double()),
            a.shape[1],
        )
        FileCheck().check("epilogue_arg_kinds=('scalar',)").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_scalar_and_row_captured_args_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: captured scalar (1 by 1) and row (1 by N) float32
        #   arguments in one epilogue, which multiplies and applies ReLU. Result matches
        #   references and generated code must record the scalar-then-row argument kinds.
        m, k, n = 128, 64, 128

        def epilogue_fn(acc, scale, row_scale):
            # [FUNCTION DOC] HELPER: epilogue multiplying the float32 accumulator by scale and
            #   row scale and applying ReLU.
            return (acc.float() * scale * row_scale).relu()

        def fn(a, b, scale, row_scale):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the scale- and row-scale-capturing epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, scale, row_scale),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([[0.5]], device="cuda", dtype=torch.float32)
        row_scale = torch.randn(1, n, device="cuda", dtype=torch.float32)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            scale,
            row_scale,
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b, scale, row_scale),
            epilogue_fn(a.double() @ b.double(), scale.double(), row_scale.double()),
            a.shape[1],
        )
        FileCheck().check("epilogue_arg_kinds=('scalar', 'row')").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_scalar_captured_arg_composes_with_compressed_local_reduce(self):
        # [FUNCTION DOC] SM100+ CUDA test: epilogue scales the accumulator by a captured scalar
        #   and also returns per-group (size 32) absolute maxima as a compressed local-reduce aux
        #   output. Checks the main output against the low-precision reference, aux against
        #   float64 within 1e-3, and that generated code has a scalar argument kind and a
        #   compressed local-reduce plan for group 32.
        m, k, n = 128, 64, 128
        group = 32

        def epilogue_fn(acc, scale):
            # [FUNCTION DOC] HELPER: epilogue returning the scaled float32 accumulator and the
            #   absolute maximum over groups of 32 along the last dimension.
            x = acc.float().view(m, -1, group)
            return acc.float() * scale, x.abs().amax(-1)

        def fn(a, b, scale):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue capturing the scalar scale.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, scale),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([[0.25]], device="cuda", dtype=torch.float32)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, scale
        )

        expected, _ = epilogue_fn(a @ b, scale)
        high_precision_expected, high_precision_aux = epilogue_fn(
            a.double() @ b.double(), scale.double()
        )
        self.assertMatchesLowPrecisionEager(
            actual, expected, high_precision_expected, a.shape[1]
        )
        torch.testing.assert_close(
            aux, high_precision_aux.float(), atol=1e-3, rtol=1e-3
        )
        FileCheck().check("epilogue_arg_kinds=('scalar',)").run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @parametrize(
        "case",
        (
            ("reduce_n_keepdim", lambda acc: acc.sum(dim=1, keepdim=True)),
            ("reduce_m_keepdim", lambda acc: acc.sum(dim=0, keepdim=True)),
            ("reduce_n", lambda acc: acc.sum(dim=1)),
            ("reduce_m", lambda acc: acc.sum(dim=0)),
            ("reduce_all", lambda acc: acc.sum()),
            ("mean_n_keepdim", lambda acc: acc.mean(dim=1, keepdim=True)),
            ("logsumexp_n_keepdim", lambda acc: acc.logsumexp(dim=1, keepdim=True)),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_tuple_aux_rejects_partial_reduction_without_contract(self, case):
        # [FUNCTION DOC] Parametrized over partial reductions of the accumulator used as a
        #   second output (sum or mean along either dim, full sum, logsumexp). Compiling flex_gemm
        #   with QUACK must fail with an error mentioning the partial-output contract, since these
        #   reductions are not local-reduce shapes.
        _, aux_fn = case

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with an
            #   epilogue returning ReLU and the parametrized reduction of the accumulator.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: (acc.relu(), aux_fn(acc)),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, 5)

        with self.assertRaisesRegex(Exception, "partial-output contract"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    def test_generated_tuple_aux_rejects_dbias_reduction_without_contract(self):
        # [FUNCTION DOC] Compiling flex_gemm whose second output is a column sum (a
        #   bias-gradient style reduction) must fail with a partial-output contract error.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with an
            #   epilogue returning ReLU and a sum over dim 0.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: (acc.relu(), acc.float().sum(dim=0)),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, 5)

        with self.assertRaisesRegex(Exception, "partial-output contract"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    def test_generated_local_reduce_aux_rejects_addmm_scope(self):
        # [FUNCTION DOC] Compiling flex_gemm on addmm with a grouped-sum aux output must fail
        #   with a message that such support is limited to aten.mm.
        def fn(bias, a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on addmm with
            #   the grouped epilogue.
            def epilogue(acc):
                # [FUNCTION DOC] HELPER: epilogue returning ReLU and the sum over groups of four
                #   along the last dimension.
                x = acc.float().view(4, -1, 4)
                return acc.relu(), x.sum(-1)

            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 8)
        a = torch.randn(4, 8)
        b = torch.randn(8, 8)

        with self.assertRaisesRegex(Exception, "currently support only aten.mm"):
            torch.compile(fn, backend="inductor", fullgraph=True)(bias, a, b)

    @skipIfNoCuteDSL
    def test_generated_local_reduce_rejects_empty_dim_list(self):
        # [FUNCTION DOC] An aten sum with an empty dimension list used as an aux output must be
        #   rejected at compile time with an error about the innermost grouped dimension.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with an
            #   epilogue built on a sum with an empty dim list.
            def epilogue(acc):
                # [FUNCTION DOC] HELPER: epilogue returning ReLU and a sum with an empty
                #   dimension list over a grouped view.
                x = acc.float().view(4, -1, 4)
                return acc.relu(), torch.ops.aten.sum.dim_IntList(x, [], False)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, 8)

        with self.assertRaisesRegex(Exception, "innermost grouped dimension"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    def test_generated_local_reduce_feed_main_rejects_addmm_scope(self):
        # [FUNCTION DOC] Compiling flex_gemm on addmm with a feed-main normalization by the sum
        #   over axis-0 groups must fail with an aten.mm-only error.
        def fn(bias, a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on addmm with
            #   the feed-main epilogue.
            def epilogue(acc):
                # [FUNCTION DOC] HELPER: epilogue grouping the accumulator rows, dividing by the
                #   group sum (via reciprocal), and reshaping back.
                x = acc.float().view(-1, 4, 5)
                scale = x.sum(1, keepdim=True)
                return (x * scale.reciprocal()).view(4, 5)

            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 5)
        a = torch.randn(4, 8)
        b = torch.randn(8, 5)

        with self.assertRaisesRegex(Exception, "currently support only aten.mm"):
            torch.compile(fn, backend="inductor", fullgraph=True)(bias, a, b)

    @skipIfNoCuteDSL
    def test_generated_local_reduce_fragment_feed_main_rejects_addmm_scope(self):
        # [FUNCTION DOC] Compiling flex_gemm on addmm with a fragment-level (axis-1) feed-main
        #   normalization must fail with an aten.mm-only error.
        def fn(bias, a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on addmm with
            #   the fragment feed-main epilogue.
            def epilogue(acc):
                # [FUNCTION DOC] HELPER: epilogue grouping the last dimension in fours, scaling
                #   by the reciprocal of the group sum, and reshaping back.
                x = acc.float().view(4, -1, 4)
                scale = x.sum(-1, keepdim=True)
                return (x * scale.reciprocal()).view(4, 8)

            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 8)
        a = torch.randn(4, 8)
        b = torch.randn(8, 8)

        with self.assertRaisesRegex(Exception, "currently support only aten.mm"):
            torch.compile(fn, backend="inductor", fullgraph=True)(bias, a, b)

    @skipIfNoCuteDSL
    def test_generated_local_reduce_rejects_bmm_scope(self):
        # [FUNCTION DOC] Compiling flex_gemm on bmm with a grouped-sum aux output must fail with
        #   an aten.mm-only error.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on bmm with
            #   the grouped epilogue.
            def epilogue(acc):
                # [FUNCTION DOC] HELPER: epilogue returning ReLU and a grouped sum over a rank-4
                #   view.
                x = acc.float().view(2, 4, -1, 4)
                return acc.relu(), x.sum(-1)

            return flex_gemm(
                torch.bmm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(2, 4, 8)
        b = torch.randn(2, 8, 8)

        with self.assertRaisesRegex(Exception, "currently support only aten.mm"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @parametrize(
        "case",
        (
            (
                "non_innermost",
                lambda acc: (acc.relu(), acc.float().view(4, -1, 4).sum(1)),
                8,
                "innermost grouped dimension",
            ),
            (
                "fragment_unsupported",
                lambda acc: (acc.relu(), acc.float().view(4, -1, 7).sum(-1)),
                14,
                "fragment width 32",
            ),
            (
                "fragment_not_dividing",
                lambda acc: (acc.relu(), acc.float().view(4, -1, 24).sum(-1)),
                48,
                "fragment width 32",
            ),
            (
                "large_group",
                lambda acc: (acc.relu(), acc.float().view(4, -1, 48).sum(-1)),
                96,
                "fragment width 32",
            ),
            (
                "degenerate_group",
                lambda acc: (acc.relu(), acc.float().view(4, -1, 1).sum(-1)),
                8,
                "group size greater than 1",
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_local_reduce_rejects_invalid_group(self, case):
        # [FUNCTION DOC] Parametrized by case (name, epilogue, N, error pattern) with invalid
        #   group shapes for a compressed local-reduce aux: non-innermost grouping, group sizes
        #   not supported by the fragment width of 32 (7, 24 not dividing, 48 too large) and a
        #   degenerate group of one. Compiling flex_gemm (QUACK) on CPU inputs must fail with an
        #   error matching the case's pattern.
        _, epilogue, n, error = case

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the parametrized epilogue and the QUACK backend.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, n)

        with self.assertRaisesRegex(Exception, error):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @parametrize(
        "case",
        (
            ("kwarg", lambda x: x.sum(-1, dtype=torch.float64)),
            (
                "aten_kwarg",
                lambda x: torch.ops.aten.sum.dim_IntList(
                    x, [-1], False, dtype=torch.float64
                ),
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_local_reduce_rejects_explicit_reduction_dtype(self, case):
        # [FUNCTION DOC] Parametrized over sum with a dtype keyword on the tensor method and on
        #   the aten overload: a grouped sum aux with an explicit float64 reduction dtype must be
        #   rejected at compile time with an error about explicit reduction dtype.
        _, reduce_fn = case

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with an
            #   epilogue returning ReLU and the parametrized reduction of the grouped accumulator.
            def epilogue(acc):
                # [FUNCTION DOC] HELPER: epilogue returning ReLU and the parametrized reduction
                #   over groups of four.
                x = acc.float().view(4, -1, 4)
                return acc.relu(), reduce_fn(x)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, 8)

        with self.assertRaisesRegex(Exception, "explicit reduction dtype"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @parametrize(
        "case",
        (
            ("any", lambda x: (x > 0).any(-1)),
            ("all", lambda x: (x > 0).all(-1)),
            ("argmax", lambda x: x.argmax(-1)),
            ("argmin", lambda x: x.argmin(-1)),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_local_reduce_rejects_non_value_reductions(self, case):
        # [FUNCTION DOC] Parametrized over any, all, argmax, argmin as grouped aux outputs:
        #   compiling must fail with an error stating the reduction does not map to a CuTe
        #   TensorSSA reduction.
        _, reduce_fn = case

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with an
            #   epilogue returning ReLU and the parametrized non-value reduction.
            def epilogue(acc):
                # [FUNCTION DOC] HELPER: epilogue returning ReLU and the parametrized reduction
                #   over groups of four.
                x = acc.float().view(4, -1, 4)
                return acc.relu(), reduce_fn(x)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, 8)

        with self.assertRaisesRegex(Exception, "does not map to a CuTe TensorSSA"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @parametrize(
        "case",
        (
            (
                "m_reduce_feeds_main",
                lambda acc: (
                    acc.float().view(-1, 4, 8)
                    * (acc.float().view(-1, 4, 8).sum(1, keepdim=True) + 1.0)
                ).view(4, 8),
                (4, 8),
                "one generated physical reduction",
            ),
            (
                "m_reduce_feeds_same_shape_aux",
                lambda acc: (
                    acc.relu(),
                    (
                        acc.float().view(-1, 4, 8)
                        * (acc.float().view(-1, 4, 8).mean(1, keepdim=True) + 1.0)
                    ).view(4, 8),
                ),
                (4, 8),
                "one generated physical reduction",
            ),
            (
                "large_n_reduce_feeds_main",
                lambda acc: (
                    acc.float().view(4, -1, 64)
                    * (acc.float().view(4, -1, 64).sum(-1, keepdim=True) + 1.0)
                ).view(4, 128),
                (4, 128),
                "post-reduction pointwise transforms",
            ),
            (
                "large_n_reduce_feeds_same_shape_aux",
                lambda acc: (
                    acc.relu(),
                    (
                        acc.float().view(4, -1, 64)
                        * (acc.float().view(4, -1, 64).mean(-1, keepdim=True) + 1.0)
                    ).view(4, 128),
                ),
                (4, 128),
                "post-reduction pointwise transforms",
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_local_reduce_rejects_physical_result_feeding_pointwise(
        self, case
    ):
        # [FUNCTION DOC] Parametrized by cases (name, epilogue, input shape, error pattern)
        #   where a physical group reduction result (over rows, or over a wide group of 64 along
        #   N) feeds further pointwise math in the main output or in a same-shape aux output.
        #   Compiling must fail with the case's message, either about one generated physical
        #   reduction or about post-reduction pointwise transforms.
        _, epilogue_fn, shape, error = case

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the parametrized epilogue and the QUACK backend.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        m, n = shape
        a = torch.randn(m, 8)
        b = torch.randn(8, n)

        with self.assertRaisesRegex(Exception, error):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @parametrize(
        "case",
        (
            (
                "m_then_n",
                lambda acc: (
                    acc.relu(),
                    acc.float().view(-1, 4, 8).sum(1).view(1, -1, 4).sum(-1),
                ),
                (4, 8),
                "local-reduce output contract",
            ),
            (
                "n_then_m",
                lambda acc: (
                    acc.relu(),
                    acc.float().view(4, -1, 4).sum(-1).view(-1, 4, 2).sum(1),
                ),
                (4, 8),
                "local-reduce output contract",
            ),
            (
                "direct_block",
                lambda acc: (
                    acc.relu(),
                    acc.float().view(-1, 4, 2, 4).sum((1, 3)),
                ),
                (4, 8),
                "local-reduce output contract",
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_local_reduce_rejects_block_reductions(self, case):
        # [FUNCTION DOC] Parametrized over block reductions (row-group sum then column-group
        #   sum, the reverse order, and a direct two-dimension block sum) used as aux output.
        #   Compiling must fail with an error mentioning the local-reduce output contract.
        _, epilogue_fn, shape, error = case

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the parametrized epilogue and the QUACK backend.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        m, n = shape
        a = torch.randn(m, 8)
        b = torch.randn(8, n)

        with self.assertRaisesRegex(Exception, error):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    def test_generated_local_reduce_rejects_mixed_row_column_grouping(self):
        # [FUNCTION DOC] Compiling an epilogue that normalizes by a group sum where the
        #   accumulator is viewed with a row group of 4 and an inner width of 32 that does not
        #   split N of 64 exactly must fail with an error saying the grouped reshape must split
        #   exactly.
        m = 8
        n = 64
        group = 4
        bad_inner_n = 32

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            def epilogue(acc):
                # [FUNCTION DOC] HELPER: epilogue grouping rows in fours with an inner width of
                #   32, dividing by the group sum via reciprocal, and reshaping to the original
                #   shape.
                x = acc.float().view(-1, group, bad_inner_n)
                scale = x.sum(1, keepdim=True)
                return (x * scale.reciprocal()).view(m, n)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 8)
        b = torch.randn(8, n)

        with self.assertRaisesRegex(Exception, "grouped reshape must split exactly"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_flex_gemm_debug_report(self):
        # [FUNCTION DOC] Builds an analysis of a traced body (ReLU main output plus a grouped
        #   sum over groups of 32) and checks the debug formatters: the analysis report must
        #   contain main output, no auxiliary, no output contraction, the grouped sum value,
        #   dataflow, geometry (N axis, group 32), consumers, layout and config constraints
        #   sections but not the verbose sections; the details report adds normalized nodes,
        #   grouped layouts and local-reduce matches; the config-key formatter prints tile and
        #   cluster values with auto for a missing tile_k. Also checks log_flex_gemm_artifact: at
        #   INFO only the non-verbose artifact is emitted (one record with the lowering banner),
        #   and at DEBUG the verbose details are emitted.
        from torch._inductor.kernel.flex_gemm.debug import (
            flex_gemm_log,
            format_flex_gemm_analysis,
            format_flex_gemm_analysis_details,
            format_flex_gemm_config_key,
            log_flex_gemm_artifact,
        )
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b):
            # [FUNCTION DOC] HELPER: traced body running the matmul and returning ReLU of it
            #   plus the sum over groups of 32 of its squares.
            acc = torch.mm(a, b)
            return torch.relu(acc), (acc * acc).view(4, -1, 32).sum(-1)

        graph_module = make_fx(body)(torch.randn(4, 8), torch.randn(8, 64))
        analysis = analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )
        analysis_report = format_flex_gemm_analysis(analysis)
        analysis_details = format_flex_gemm_analysis_details(analysis)
        config_report = format_flex_gemm_config_key(
            (
                ("tile_m", 256),
                ("tile_n", 128),
                ("tile_k", None),
                ("cluster_m", 2),
                ("cluster_n", 1),
                ("cluster_k", 1),
            )
        )

        self.assertIn("outputs:\n  main: relu: shape=(4, 64)", analysis_report)
        self.assertIn("auxiliary: (none)", analysis_report)
        self.assertIn("output_contraction: none", analysis_report)
        self.assertIn("value: sum_1", analysis_report)
        self.assertIn("dataflow: view -> sum(dim=[-1], keepdim=False)", analysis_report)
        self.assertIn("geometry: axis=N, group=32", analysis_report)
        self.assertIn("consumers: returned", analysis_report)
        self.assertIn("output_layout: dense", analysis_report)
        self.assertIn("config_constraints:\n  axis=N, group=32", analysis_report)
        self.assertNotIn("recognized_dataflow:", analysis_report)
        self.assertNotIn("normalized_nodes:", analysis_report)
        self.assertIn("normalized_nodes:\n  view: NormalizedView", analysis_details)
        self.assertIn("sum_1: NormalizedReduction", analysis_details)
        self.assertIn("grouped_layouts:\n  view:", analysis_details)
        self.assertIn("local_reduce_matches:\n  sum_1:", analysis_details)
        self.assertIn("output_contraction_select_indices:\n  (none)", analysis_details)
        self.assertIn("tile_m: 256", config_report)
        self.assertIn("tile_k: auto", config_report)
        self.assertIn("cluster_m: 2", config_report)
        with self.assertLogs(flex_gemm_log, level="INFO") as records:
            log_flex_gemm_artifact("analysis", lambda: analysis_report)
            log_flex_gemm_artifact(
                "analysis_details", lambda: analysis_details, verbose=True
            )
        self.assertEqual(len(records.output), 1)
        self.assertIn(
            "FLEXGEMM LOWERING\n ===== ANALYSIS =====\noutputs:\n  main: relu",
            records.output[0],
        )

        with self.assertLogs(flex_gemm_log, level="DEBUG") as records:
            log_flex_gemm_artifact(
                "analysis_details", lambda: analysis_details, verbose=True
            )
        self.assertIn(" ===== ANALYSIS DETAILS =====", records.output[0])

    def test_output_contraction_debug_report(self):
        # [FUNCTION DOC] Analyzes a traced body that selects the two interleaved elements of
        #   each N pair and subtracts them; the analysis report must show an N-axis, group 2,
        #   interleaved output contraction, and the details report must list normalized select
        #   nodes and non-empty output-contraction select indices.
        from torch._inductor.kernel.flex_gemm.debug import (
            format_flex_gemm_analysis,
            format_flex_gemm_analysis_details,
        )
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b):
            # [FUNCTION DOC] HELPER: traced body viewing the matmul result as pairs along N and
            #   returning the difference of the two selected pair elements.
            grouped = torch.mm(a, b).view(4, 8, 2)
            return grouped.select(-1, 0) - grouped.select(-1, 1)

        graph_module = make_fx(body)(torch.randn(4, 8), torch.randn(8, 16))
        analysis = analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )

        self.assertIn(
            "output_contraction: N-axis, group=2, layout=interleaved",
            format_flex_gemm_analysis(analysis),
        )
        details = format_flex_gemm_analysis_details(analysis)
        self.assertIn("NormalizedSelect", details)
        self.assertIn("output_contraction_select_indices:", details)
        self.assertNotIn("output_contraction_select_indices:\n  (none)", details)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_emits_flex_gemm_debug_report(self):
        # [FUNCTION DOC] SM100+ CUDA test: compiles a QUACK flex_gemm with ReLU while capturing
        #   the flex_gemm logger at DEBUG. The result must equal eager ReLU; INFO records must
        #   contain the problem, analysis, lowering plan and selection phases in that order, fixed
        #   mode, one approved candidate, the template name and the log hint, with no ANSI
        #   escapes, normalized nodes, generated epilogue or candidates; DEBUG records must
        #   contain analysis details, generated epilogue (with cute.jit) and config candidates in
        #   order.
        from torch._inductor.kernel.flex_gemm.debug import flex_gemm_log

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   ReLU as the epilogue and the QUACK backend.
            return flex_gemm(
                torch.mm,
                (a, b),
                torch.relu,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        with self.assertLogs(flex_gemm_log, level="DEBUG") as records:
            actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        self.assertEqual(actual, torch.relu(a @ b))
        report = "\n".join(
            record.getMessage()
            for record in records.records
            if record.levelno == logging.INFO
        )
        phases = (
            " ===== PROBLEM =====",
            " ===== ANALYSIS =====",
            " ===== LOWERING PLAN =====",
            " ===== SELECTION =====",
        )
        positions = tuple(report.index(phase) for phase in phases)
        self.assertEqual(positions, tuple(sorted(positions)))
        self.assertIn("gemm_op: aten.mm.default", report)
        self.assertIn("outputs:\n  main: relu", report)
        self.assertNotIn("normalized_nodes:", report)
        self.assertNotIn("GENERATED EPILOGUE", report)
        self.assertNotIn("CONFIG CANDIDATES", report)
        self.assertNotIn("\x1b[", report)
        self.assertIn("FLEXGEMM LOWERING [flex_gemm_body_graph_", report)
        self.assertIn("mode: fixed", report)
        self.assertIn("lowering_approved_candidates: 1", report)
        self.assertIn("template: cutedsl_flex_gemm_epilogue_", report)
        self.assertIn('analysis/codegen: TORCH_LOGS="+flex_gemm"', report)

        verbose_report = "\n".join(
            record.getMessage()
            for record in records.records
            if record.levelno == logging.DEBUG
        )
        verbose_phases = (
            " ===== ANALYSIS DETAILS =====",
            " ===== GENERATED EPILOGUE =====",
            " ===== CONFIG CANDIDATES =====",
        )
        verbose_positions = tuple(
            verbose_report.index(phase) for phase in verbose_phases
        )
        self.assertEqual(verbose_positions, tuple(sorted(verbose_positions)))
        self.assertIn("normalized_nodes:", verbose_report)
        self.assertIn("@cute.jit", verbose_report)
        self.assertIn("candidate 0:", verbose_report)

    def test_grouped_layout_rejects_inexact_inferred_preserved_dimension(self):
        # [FUNCTION DOC] grouped_tensor_layout called with an inferred leading dimension and
        #   group shape that cannot split the 4 by 5 tensor exactly must raise NotImplementedError
        #   saying the grouped reshape must split exactly.
        from torch._inductor.kernel.gemm_epilogue_analysis import grouped_tensor_layout

        with self.assertRaisesRegex(
            NotImplementedError, "grouped reshape must split exactly"
        ):
            grouped_tensor_layout((-1, 2, 2), (4, 5))

    @parametrize("unbacked_dim", (1, 2))
    def test_grouped_layout_rejects_unbacked_structural_dimensions(self, unbacked_dim):
        # [FUNCTION DOC] Parametrized by which of the two trailing dims is an unbacked symbolic
        #   int: grouped_tensor_layout must return None (not a layout) when a structural dimension
        #   is unbacked.
        from torch._inductor.kernel.gemm_epilogue_analysis import grouped_tensor_layout
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape = [4, -1, 2]
        shape[unbacked_dim] = ShapeEnv().create_unbacked_symint()
        self.assertIsNone(grouped_tensor_layout(tuple(shape), (4, 8)))

    def test_grouped_layout_guards_backed_group_before_rejection(self):
        # [FUNCTION DOC] With a backed symbolic group size of 3 that does not split the 4 by 8
        #   tensor, grouped_tensor_layout returns None and exactly one guard is recorded on the
        #   shape environment (the group is guarded before rejection).
        from torch._dynamo.source import ConstantSource
        from torch._inductor.kernel.gemm_epilogue_analysis import grouped_tensor_layout
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(3, ConstantSource("group"))
        group = shape_env.create_symintnode(symbol, hint=3)

        self.assertIsNone(grouped_tensor_layout((4, 5, group), (4, 8)))
        self.assertEqual(len(shape_env.guards), 1)

    def test_output_contraction_plan_only_mutates_analysis_on_match(self):
        # [FUNCTION DOC] For two traced bodies with chunk-based halves (one swapping halves, one
        #   a SiLU gate times up) and a symbolic split size, runs the epilogue analysis in
        #   subtests. The swapped body must have no output contraction and register no split node
        #   in the grouped tensors; the SiLU-gate body must detect a chunked group-2 contraction
        #   and register all split nodes. Exactly one shape guard must be installed in each case.
        """Rejected recognitions must not leak grouped layouts."""
        from torch._dynamo.source import ConstantSource
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmOutputContraction,
        )
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def swap_halves_plus_acc(a, b):
            # [FUNCTION DOC] HELPER: traced body splitting the matmul result into halves along
            #   N, concatenating them swapped, and adding the original result.
            acc = torch.mm(a, b)
            halves = acc.chunk(2, dim=-1)
            return torch.cat((halves[1], halves[0]), dim=-1) + acc

        def silu_mul_halves(a, b):
            # [FUNCTION DOC] HELPER: traced body splitting the matmul result into halves and
            #   returning SiLU of the first multiplied by the second.
            acc = torch.mm(a, b)
            halves = acc.chunk(2, dim=-1)
            return torch.nn.functional.silu(halves[0]) * halves[1]

        chunked_contraction = FlexGemmOutputContraction(group=2, chunked=True)
        for body, expected_contraction in (
            (swap_halves_plus_acc, None),
            (silu_mul_halves, chunked_contraction),
        ):
            with self.subTest(body=body.__name__):
                graph_module = make_fx(body)(torch.randn(4, 8), torch.randn(8, 16))
                shape_env = ShapeEnv()
                symbol = shape_env.create_symbol(
                    8, ConstantSource(f"{body.__name__}_split_size")
                )
                split_size = shape_env.create_symintnode(symbol, hint=8)
                for node in graph_module.graph.nodes:
                    if node.target is torch.ops.aten.split.Tensor:
                        node.args = (node.args[0], split_size, node.args[2])
                analysis = analyze_flex_gemm_epilogue(
                    graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
                )
                split_nodes = [
                    node
                    for node in graph_module.graph.nodes
                    if node.target is torch.ops.aten.split.Tensor
                ]
                self.assertTrue(split_nodes)
                self.assertEqual(
                    analysis.outputs.output_contraction, expected_contraction
                )
                registered = [
                    node
                    for node in split_nodes
                    if node in analysis.local_reduce.grouped_tensors
                ]
                self.assertEqual(
                    registered, split_nodes if expected_contraction else []
                )
                self.assertEqual(len(shape_env.guards), 1)

    def test_incomplete_grouped_selects_are_rejected(self):
        # [FUNCTION DOC] An epilogue concatenating four selects of a group of four where one
        #   index is repeated must make analyze_flex_gemm_epilogue raise NotImplementedError
        #   mentioning a complete output contraction.
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def duplicate_group_value(a, b):
            # [FUNCTION DOC] HELPER: traced body concatenating selects of indices 0, 1, 2, 2 of
            #   a 4-wide group of the matmul result (duplicate index, so incomplete).
            grouped = torch.mm(a, b).view(4, 4, 4)
            return torch.cat(
                (
                    grouped.select(-1, 0),
                    grouped.select(-1, 1),
                    grouped.select(-1, 2),
                    grouped.select(-1, 2),
                ),
                dim=-1,
            )

        graph_module = make_fx(duplicate_group_value)(
            torch.randn(4, 8), torch.randn(8, 16)
        )
        with self.assertRaisesRegex(NotImplementedError, "complete output contraction"):
            analyze_flex_gemm_epilogue(
                graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
            )

    @parametrize("pointwise", (False, True))
    def test_accepted_grouped_reduction_installs_group_guard(self, pointwise):
        # [FUNCTION DOC] Parametrized by whether a pointwise add precedes the reduction: traces
        #   a normalization by a grouped sum, replaces the group size in the view with a backed
        #   symbolic int, runs the analysis and asserts exactly one shape guard is installed on
        #   the shape environment.
        from torch._dynamo.source import ConstantSource
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def grouped_reduction(a, b):
            # [FUNCTION DOC] HELPER: traced body viewing the matmul result in groups of two,
            #   optionally adding one, summing the last dim, dividing the grouped values by the
            #   sum, and reshaping back.
            grouped = torch.mm(a, b).view(4, 8, 2)
            source = grouped + 1 if pointwise else grouped
            reduced = source.sum(-1, keepdim=True)
            return (grouped / reduced).view(4, 16)

        graph_module = make_fx(grouped_reduction)(torch.randn(4, 8), torch.randn(8, 16))
        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(2, ConstantSource("view_group"))
        group = shape_env.create_symintnode(symbol, hint=2)
        self.assertIsInstance(group, torch.SymInt)
        view = next(
            node
            for node in graph_module.graph.nodes
            if node.target is torch.ops.aten.view.default
            and tuple(node.meta["val"].shape) == (4, 8, 2)
        )
        view.args = (view.args[0], (4, 8, group))

        analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )
        self.assertEqual(len(shape_env.guards), 1)

    def test_output_contraction_does_not_contract_other_axes(self):
        # [FUNCTION DOC] Two traced bodies that pair rows (via a row-axis view and selects) or
        #   split along a middle axis must be analyzed without detecting an output contraction
        #   (output_contraction is None), checked in subtests.
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def reassemble_row_pairs(a, b):
            # [FUNCTION DOC] HELPER: traced body that views the result as row pairs and
            #   concatenates the two selected rows along the first dimension.
            acc = torch.mm(a, b)
            rows = acc.view(2, 2, 16)
            return torch.cat((rows.select(1, 0), rows.select(1, 1)), dim=0)

        def split_grouped_view(a, b):
            # [FUNCTION DOC] HELPER: traced body that splits a grouped view into chunks along
            #   the middle axis, subtracts the first two, and repeats the result.
            acc = torch.mm(a, b)
            chunks = acc.view(4, 8, 2).split(1, dim=1)
            pair = (chunks[0] - chunks[1]).reshape(4, 2)
            return pair.repeat(1, 8)

        for body in (reassemble_row_pairs, split_grouped_view):
            with self.subTest(body=body.__name__):
                graph_module = make_fx(body)(torch.randn(4, 8), torch.randn(8, 16))
                analysis = analyze_flex_gemm_epilogue(
                    graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
                )
                self.assertIsNone(analysis.outputs.output_contraction)

    @skipIfNoCuteDSL
    def test_output_contraction_validates_output_layout_and_shape(self):
        # [FUNCTION DOC] Calls the vendored Quack gemm_act on small CPU tensors with a
        #   main-output transform group and checks host validation: a non-N-major post-activation
        #   output raises NotImplementedError; a post-activation with the wrong width (for groups
        #   2 and 4) raises RuntimeError about the required shape; and a physical N of 10 that is
        #   not divisible by group 4 raises RuntimeError. Requires CuTe DSL.
        from torch._vendor.quack.gemm_act import gemm_act

        def run(b, out, group):
            # [FUNCTION DOC] HELPER: calls gemm_act with a fixed 4 by 8 A, given B, given output
            #   and given transform group, identity tensor epilogue, and tile and cluster sizes of
            #   128, 128, 1, 1.
            gemm_act(
                torch.empty(4, 8),
                b,
                None,
                None,
                out,
                None,
                None,
                128,
                128,
                1,
                1,
                tensor_epilogue_fn=lambda acc: acc,
                main_output_transform_group=group,
            )

        with self.assertRaisesRegex(NotImplementedError, "PostAct to be n-major"):
            run(torch.empty(8, 16), torch.empty_strided((4, 8), (1, 4)), 2)
        for group in (2, 4):
            with self.subTest(group=group):
                with self.assertRaisesRegex(RuntimeError, "requires PostAct shape"):
                    run(torch.empty(16, 8), torch.empty(4, 16 // group + 1), group)
        with self.assertRaisesRegex(RuntimeError, "physical N=10 must be divisible"):
            run(torch.empty(10, 8), torch.empty(4, 2), 4)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_dynamic_shapes_compiled_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: compiles with dynamic shapes a tuple epilogue (scaled
        #   accumulator plus one, squared accumulator plus two) for two sets of (M, K, N), and
        #   compares main and aux with the tuple-aux reference helper.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning half of the accumulator plus one and the
            #   accumulator squared plus two in float32.
            main = (acc.float() + 1.0) * 0.5
            aux = acc.float().square() + 2.0
            return main, aux

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the tuple epilogue and QUACK backend.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        for m, k, n in ((128, 64, 128), (256, 64, 192)):
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
            actual, aux = compiled(a, b)
            self.assertTupleAuxMatchesReference(actual, aux, a, b, epilogue_fn)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_temporary_grouped_reshape_without_reduction(self):
        # [FUNCTION DOC] SM100+ CUDA test: an epilogue that views N in groups of three, adds one
        #   and views back (a temporary grouped reshape with no reduction) compiles and matches
        #   the references for M 128, N 96, K 64.
        m, n, k, group = 128, 96, 64, 3

        def epilogue(acc):
            # [FUNCTION DOC] HELPER: epilogue viewing the float32 accumulator as groups of
            #   three, adding one, and reshaping to the original shape.
            grouped = acc.float().view(m, n // group, group)
            return (grouped + 1.0).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the grouped-reshape epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        self.assertMatchesLowPrecisionEager(
            actual,
            (a @ b).float() + 1.0,
            (a.double() @ b.double()) + 1.0,
            k,
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    @parametrize(
        "operation,tuned",
        (
            ("silu_mul", False),
            ("sub", False),
            ("uint8_sub", False),
            ("silu_mul", True),
        ),
    )
    def test_mm_output_contraction_compiled_matches_reference(self, operation, tuned):
        # [FUNCTION DOC] SM100+ (not SM120) CUDA test parametrized by operation (SiLU-mul,
        #   subtraction, uint8 subtraction, and a tuned SiLU-mul): pair-contracting epilogue
        #   halves N (group 2) from float16 inputs. Compares the compiled output with the eager
        #   flex_gemm result (exactly for uint8, loose tolerance otherwise), checks the output
        #   shape is M by N and that generated code uses a group-2 output contraction with no
        #   activation argument and no extern mm.
        m, k, n, group = 64, 64, 64, 2

        def epilogue(acc):
            # [FUNCTION DOC] HELPER: epilogue viewing the accumulator as pairs along N, picking
            #   gate and up elements (via select or indexing), and returning SiLU(gate) times up,
            #   the difference, or the difference cast to uint8, depending on operation.
            grouped_values = acc.float().view(m, n, group)
            if operation == "sub":
                gate, up = grouped_values.select(-1, 0), grouped_values.select(-1, 1)
            else:
                gate, up = grouped_values[..., 0], grouped_values[..., 1]
            match operation:
                case "silu_mul":
                    return (torch.nn.functional.silu(gate) * up).to(acc.dtype)
                case "sub":
                    return (gate - up).to(acc.dtype)
                case "uint8_sub":
                    return (gate - up).to(torch.uint8)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the pair epilogue, QUACK backend, and the tuned option.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK", "tuned": tuned},
            )

        if operation == "uint8_sub":
            a = torch.eye(m, k, device="cuda", dtype=torch.float16)
            columns = torch.arange(group * n, device="cuda")
            pair = (columns // group) % 8
            b = torch.where(columns % group == 0, 3 * pair + 2, pair)
            b = b.expand(k, -1).contiguous().to(torch.float16)
        else:
            a = torch.randn(m, k, device="cuda", dtype=torch.float16)
            b = torch.randn(k, group * n, device="cuda", dtype=torch.float16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected = fn(a, b)
        if operation == "uint8_sub":
            self.assertEqual(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)
        self.assertEqual(actual.shape, (m, n))
        FileCheck().check("FlexGemmOutputContraction(group=2").check_not(
            "activation="
        ).check_not("extern_kernels.mm").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_mm_output_contraction_non_aligned_logical_n(self):
        # [FUNCTION DOC] SM100+ (not SM120) CUDA test: output contraction where the logical N is
        #   68 (not aligned to the tile); compiled result matches the eager pair difference with
        #   loose tolerance, has shape M by N, and generated code has a group-2 contraction and no
        #   extern mm.
        m, k, n = 64, 64, 68

        def epilogue(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the difference of the two interleaved
            #   halves of each pair, cast to the input dtype.
            grouped_values = acc.float().view(m, n, 2)
            return (grouped_values[..., 0] - grouped_values[..., 1]).to(acc.dtype)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the pair-difference epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, 2 * n, device="cuda", dtype=torch.float16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, epilogue(a @ b), atol=1e-1, rtol=1e-1)
        self.assertEqual(actual.shape, (m, n))
        FileCheck().check("FlexGemmOutputContraction(group=2").check_not(
            "extern_kernels.mm"
        ).run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    @parametrize(
        "case",
        (
            ("scalar_expr", "scalar", "scalar_expr", False),
            ("col_pre", "col", "pre", False),
            ("col_expr_chunked", "col", "col_expr", True),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_output_contraction_with_n_invariant_capture(self, case):
        # [FUNCTION DOC] SM100+ (not SM120) CUDA test parametrized by capture case (scalar used
        #   in an expression, column used before contraction, column used in an expression on the
        #   chunked layout): output contraction combined with an N-invariant captured tensor.
        #   Compares with eager and checks generated code records the expected arg kind and no
        #   extern mm.
        _, capture_kind, capture_use, chunked = case
        m = k = n = 64

        def epilogue(acc, capture):
            # [FUNCTION DOC] HELPER: epilogue optionally pre-multiplying the accumulator by the
            #   capture, splitting into pair halves (chunked or interleaved), and returning the
            #   difference optionally scaled by a capture expression, cast to the accumulator
            #   dtype.
            value = acc.float()
            if capture_use == "pre":
                value = value * capture
            if chunked:
                lhs, rhs = value.chunk(2, dim=-1)
            else:
                grouped_values = value.view(m, n, 2)
                lhs, rhs = grouped_values[..., 0], grouped_values[..., 1]
            if capture_use == "scalar_expr":
                return ((lhs - rhs) * (capture + 1.0)).to(acc.dtype)
            if capture_use == "col_expr":
                scale = torch.where(capture > 0, -capture, capture + 1.0)
                return ((lhs - rhs) * scale).to(acc.dtype)
            return (lhs - rhs).to(acc.dtype)

        def fn(a, b, capture):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue capturing the tensor.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue(acc, capture),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = (
            torch.randn(2 * n, k, device="cuda", dtype=torch.float16).t()
            if chunked
            else torch.randn(k, 2 * n, device="cuda", dtype=torch.float16)
        )
        capture_shape = (1, 1) if capture_kind == "scalar" else (m, 1)
        capture = torch.randn(*capture_shape, device="cuda", dtype=torch.float32)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, capture
        )

        torch.testing.assert_close(
            actual, epilogue(a @ b, capture), atol=0.2, rtol=0.05
        )
        FileCheck().check(f"epilogue_arg_kinds=('{capture_kind}',)").check_not(
            "extern_kernels.mm"
        ).run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
        "interleaved group 4 is currently validated only on SM100",
    )
    def test_mm_output_contraction_group4(self):
        # [FUNCTION DOC] SM100-only CUDA test (skipped otherwise): interleaved group-4 output
        #   contraction combining a scalar scale and a column captured argument. Result matches
        #   eager, has shape M by N, and generated code records the scalar and col arg kinds and
        #   an interleaved group-4 contraction without extern mm.
        m, k, n, group = 64, 64, 64, 4

        def epilogue(acc, scale, col):
            # [FUNCTION DOC] HELPER: epilogue combining the four elements of each group of four
            #   with a scalar scale on the first difference, a column multiplier on the third and
            #   subtracting the fourth, cast to the accumulator dtype.
            grouped_values = acc.float().view(m, n, group)
            return (
                (grouped_values[..., 0] - grouped_values[..., 1]) * scale
                + grouped_values[..., 2] * col
                - grouped_values[..., 3]
            ).to(acc.dtype)

        def fn(a, b, scale, col):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the group-4 epilogue capturing scale and col.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue(acc, scale, col),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, group * n, device="cuda", dtype=torch.float16)
        scale = torch.randn(1, 1, device="cuda", dtype=torch.float32)
        col = torch.randn(m, 1, device="cuda", dtype=torch.float32)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, scale, col
        )

        torch.testing.assert_close(
            actual, epilogue(a @ b, scale, col), atol=2e-1, rtol=5e-2
        )
        self.assertEqual(actual.shape, (m, n))
        FileCheck().check("epilogue_arg_kinds=('scalar', 'col')").check(
            "FlexGemmOutputContraction(group=4, chunked=False)"
        ).check_not("extern_kernels.mm").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    @parametrize("chunked", (False, True))
    def test_mm_output_contraction_dynamic_m(self, chunked):
        # [FUNCTION DOC] SM100+ (not SM120) CUDA test parametrized by chunked or interleaved
        #   layout: pair contraction with SiLU gating and a column multiplier, run with M 64 (the
        #   A and column inputs marked dynamic along M) and then M 128 on the same compiled
        #   function; both outputs are compared with eager.
        k = n = 64

        def epilogue(acc, col):
            # [FUNCTION DOC] HELPER: epilogue splitting into gate and up halves (chunked or
            #   interleaved), returning SiLU(gate) times up times the column capture, cast to the
            #   accumulator dtype.
            value = acc.float()
            if chunked:
                gate, up = value.chunk(2, dim=-1)
            else:
                grouped_values = value.view(-1, n, 2)
                gate, up = grouped_values[..., 0], grouped_values[..., 1]
            return (torch.nn.functional.silu(gate) * up * col).to(acc.dtype)

        def fn(a, b, col):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue capturing col.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue(acc, col),
                kernel_options={"backend": "QUACK"},
            )

        b = torch.randn(2 * n, k, device="cuda", dtype=torch.float16)
        b = b.t() if chunked else b.t().contiguous()
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        for index, m in enumerate((64, 128)):
            a = torch.randn(m, k, device="cuda", dtype=torch.float16)
            col = torch.randn(m, 1, device="cuda", dtype=torch.float32)
            if index == 0:
                torch._dynamo.mark_dynamic(a, 0)
                torch._dynamo.mark_dynamic(col, 0)
            torch.testing.assert_close(
                compiled(a, b, col), epilogue(a @ b, col), atol=1e-1, rtol=1e-1
            )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    @parametrize("chunked", (False, True))
    def test_mm_output_contraction_partial_n_tile(self, chunked):
        # [FUNCTION DOC] SM100+ (not SM120) CUDA test parametrized by chunked or interleaved
        #   layout: logical N of 80 with an explicit config of tile N 128 (a partial N tile),
        #   contraction of pairs. Result matches eager with loose tolerance; generated code has a
        #   group-2 contraction with the tile N key and no extern mm.
        m = k = 64
        n = 80

        def epilogue(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the difference of the two halves of each
            #   pair (chunked or interleaved), cast to the accumulator dtype.
            value = acc.float()
            if chunked:
                lhs, rhs = value.chunk(2, dim=-1)
            else:
                grouped_values = value.view(m, n, 2)
                lhs, rhs = grouped_values[..., 0], grouped_values[..., 1]
            return (lhs - rhs).to(acc.dtype)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the pair-difference epilogue and an explicit config with tile N 128, cluster N 1
            #   and no swap.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={
                    "backend": "QUACK",
                    "config": {"tile_n": 128, "cluster_n": 1, "swap_ab": False},
                },
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = (
            torch.randn(2 * n, k, device="cuda", dtype=torch.float16).t()
            if chunked
            else torch.randn(k, 2 * n, device="cuda", dtype=torch.float16)
        )
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        torch.testing.assert_close(actual, epilogue(a @ b), atol=0.2, rtol=0.05)
        FileCheck().check("FlexGemmOutputContraction(group=2").check(
            "('tile_n', 128)"
        ).check_not("extern_kernels.mm").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_mm_output_contraction_concat_layout(self):
        # [FUNCTION DOC] SM100+ (not SM120) CUDA test: concat-layout output contraction where
        #   the accumulator is viewed as two halves along N and the last two slices are selected
        #   (SiLU of one times the other) with a transposed B. Compiled result matches eager with
        #   loose tolerance; generated code has a group-2 chunked contraction, no activation
        #   argument and no extern mm.
        m, k, n = 64, 64, 64

        def epilogue(acc):
            # [FUNCTION DOC] HELPER: epilogue viewing the accumulator as a pair of N-wide
            #   halves, selecting them, and returning SiLU of one times the other, cast to the
            #   accumulator dtype.
            grouped_values = acc.float().view(m, 2, n)
            gate, up = grouped_values.select(-2, -1), grouped_values.select(-2, -2)
            return (torch.nn.functional.silu(gate) * up).to(acc.dtype)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the concat-layout epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(2 * n, k, device="cuda", dtype=torch.float16).t()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=2e-1, rtol=2e-2)
        FileCheck().check("FlexGemmOutputContraction(group=2").check(
            "chunked=True"
        ).check_not("activation=").check_not("extern_kernels.mm").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_mm_output_contraction_specializes_split_size(self):
        # [FUNCTION DOC] SM100+ (not SM120) CUDA test: an epilogue splitting N by a static size
        #   of 64 and subtracting the two pieces, with B's N dimension marked static. Compiled
        #   output matches eager; generated code has a chunked group-2 contraction and no extern
        #   mm.
        m = k = split_size = 64

        def epilogue(acc):
            # [FUNCTION DOC] HELPER: epilogue splitting the accumulator along N into two halves
            #   and returning their difference cast to the accumulator dtype.
            grouped_values = acc.float().split(split_size, dim=-1)
            return (grouped_values[0] - grouped_values[1]).to(acc.dtype)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the split-difference epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(2 * split_size, k, device="cuda", dtype=torch.float16).t()
        torch._dynamo.mark_static(b, 1)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        torch.testing.assert_close(actual, epilogue(a @ b), atol=0.2, rtol=0.05)
        FileCheck().check("FlexGemmOutputContraction(group=2").check(
            "chunked=True"
        ).check_not("extern_kernels.mm").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_mm_output_contraction_supports_clustered_m_configs(self):
        # [FUNCTION DOC] SM100+ (not SM120) CUDA test: group-2 contraction with an explicit bf16
        #   config using cluster M of 2 and tile M of 256 (tile N 256, SM100). Output matches
        #   low-precision and float64 references; generated code records the col arg kind, the
        #   contraction, and the tile M and cluster M keys.
        from torch._vendor.quack.gemm_config import GemmConfig

        m, k, n, group = 256, 64, 128, 2
        tile_m = 256

        def epilogue(acc, col):
            # [FUNCTION DOC] HELPER: epilogue returning SiLU of the first pair element times the
            #   second times the column capture, cast to the accumulator dtype.
            grouped_values = acc.float().view(m, n, group)
            return (
                torch.nn.functional.silu(grouped_values[..., 0])
                * grouped_values[..., 1]
                * col
            ).to(acc.dtype)

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=tile_m,
                tile_n=256,
                pingpong=False,
                cluster_m=2,
                cluster_n=1,
                device_capacity=10,
            )
        )

        def fn(a, b, col):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the explicit config and the column-capturing epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue(acc, col),
                kernel_options={"backend": "QUACK", "config": config},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, group * n, device="cuda", dtype=torch.bfloat16)
        col = torch.randn(m, 1, device="cuda", dtype=torch.float32)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, col
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue(a @ b, col),
            epilogue(a.double() @ b.double(), col.double()),
            k,
        )
        FileCheck().check("epilogue_arg_kinds=('col',)").check(
            "FlexGemmOutputContraction(group=2"
        ).check(f"('tile_m', {tile_m})").check("('cluster_m', 2)").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_output_contraction_filters_unsafe_configs(self):
        # [FUNCTION DOC] SM100+ (not SM120) CUDA test of config filtering for group-2 output
        #   contraction: tuned config keys from flex_gemm_config_keys are non-empty and all
        #   satisfy output_contraction_config_supported for N 128; and an explicit key for a
        #   non-swap candidate that fails that check must raise NotImplementedError about
        #   incompatibility with output contraction.
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_from_key,
            gemm_config_key,
        )
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmOutputContraction,
            output_contraction_config_supported,
        )
        from torch._inductor.kernel.flex_gemm.lowering import flex_gemm_config_keys

        device = torch.device("cuda")
        candidates = candidate_gemm_configs_for_device(device)
        contraction = FlexGemmOutputContraction(group=2)
        configs = tuple(
            gemm_config_from_key(key)
            for key in flex_gemm_config_keys(
                device, 64, 128, (), tuned=True, output_contraction=contraction
            )
        )
        self.assertTrue(configs)
        self.assertTrue(
            all(output_contraction_config_supported(config, 128) for config in configs)
        )

        unsafe = next(
            config
            for config in candidates
            if not config.swap_ab
            and not output_contraction_config_supported(config, 128)
        )
        with self.assertRaisesRegex(
            NotImplementedError, "incompatible with output contraction"
        ):
            flex_gemm_config_keys(
                device,
                64,
                128,
                (),
                tuned=True,
                output_contraction=contraction,
                explicit_config=dict(gemm_config_key(unsafe)),
            )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_mm_output_contraction_rejects_unsupported_composition(self):
        # [FUNCTION DOC] SM100+ (not SM120) CUDA test that unsupported output contraction
        #   compositions are rejected at compile time (via the class helper) with matching
        #   messages: a group of eight; an N-varying captured tensor; a boolean captured tensor;
        #   chunked halves in a non-contiguous layout; dynamic physical N; and an auxiliary output
        #   next to the contraction.
        m = k = n = 64
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)

        def group_eight(acc):
            # [FUNCTION DOC] HELPER: epilogue summing eight consecutive elements of each group
            #   of eight along N.
            grouped_values = acc.view(m, n, 8)
            return sum(grouped_values[..., index] for index in range(8))

        self.assertOutputContractionRejected(
            a,
            torch.randn(k, 8 * n, device="cuda", dtype=torch.float16),
            group_eight,
            "group 2.*group 4",
        )

        scale = torch.randn(1, 2 * n, device="cuda", dtype=torch.float16)

        def n_varying_capture(acc):
            # [FUNCTION DOC] HELPER: epilogue multiplying the accumulator by a captured row of
            #   per-N scales, then taking the difference of the pair elements.
            grouped_values = (acc * scale).view(m, n, 2)
            return grouped_values[..., 0] - grouped_values[..., 1]

        b = torch.randn(k, 2 * n, device="cuda", dtype=torch.float16)
        self.assertOutputContractionRejected(
            a, b, n_varying_capture, "numeric.*captured tensors"
        )

        boolean_scale = torch.ones(1, 1, device="cuda", dtype=torch.bool)

        def boolean_capture(acc):
            # [FUNCTION DOC] HELPER: epilogue multiplying the accumulator by a captured boolean
            #   scalar, then taking the difference of the pair elements.
            grouped_values = (acc * boolean_scale).view(m, n, 2)
            return grouped_values[..., 0] - grouped_values[..., 1]

        self.assertOutputContractionRejected(
            a, b, boolean_capture, "numeric.*captured tensors"
        )

        def chunked(acc):
            # [FUNCTION DOC] HELPER: epilogue splitting the accumulator into two halves along N
            #   and returning their difference.
            gate, up = acc.chunk(2, dim=-1)
            return gate - up

        self.assertOutputContractionRejected(a, b, chunked, "non-contiguous")

        def dynamic_n(acc):
            # [FUNCTION DOC] HELPER: epilogue viewing N with an inferred pair size and returning
            #   the difference of the two pair elements.
            grouped_values = acc.view(m, -1, 2)
            return grouped_values[..., 0] - grouped_values[..., 1]

        dynamic_b = torch.randn(k, 2 * n, device="cuda", dtype=torch.float16)
        torch._dynamo.mark_dynamic(dynamic_b, 1)
        self.assertOutputContractionRejected(
            a, dynamic_b, dynamic_n, "statically known physical N"
        )

        def auxiliary(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the difference of pair elements together
            #   with the raw accumulator as an extra output.
            grouped_values = acc.view(m, n, 2)
            return grouped_values[..., 0] - grouped_values[..., 1], acc

        self.assertOutputContractionRejected(a, b, auxiliary, "do not compose")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_mm_chunked_output_contraction_rejects_grouped_reduction(self):
        # [FUNCTION DOC] SM100+ (not SM120) CUDA test: an epilogue that normalizes by per-group
        #   (32) absolute maxima and then contracts the chunked halves must fail to compile with
        #   InductorError saying concat-layout output contractions do not compose with grouped
        #   reductions.
        m, k, physical_n, group = 128, 64, 256, 32

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue normalizing the float32 accumulator by group-of-32
            #   absolute maxima, splitting into halves and returning their difference cast to the
            #   accumulator dtype.
            value = acc.float()
            grouped = value.view(m, -1, group)
            scale = grouped.abs().amax(-1, keepdim=True)
            normalized = (grouped / scale).view(m, physical_n)
            gate, up = normalized.chunk(2, dim=-1)
            return (gate - up).to(acc.dtype)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the normalization-then-chunk epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(physical_n, k, device="cuda", dtype=torch.bfloat16).t()
        with self.assertRaisesRegex(
            InductorError,
            "concat-layout output contractions do not compose with grouped reductions",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("sum_method", lambda x: x.sum(-1), "cute.ReductionOp.ADD"),
            ("sum_function", lambda x: torch.sum(x, dim=-1), "cute.ReductionOp.ADD"),
            ("mean_method", lambda x: x.mean(-1), "cute.ReductionOp.ADD"),
            ("mean_function", lambda x: torch.mean(x, dim=-1), "cute.ReductionOp.ADD"),
            ("prod_method", lambda x: (x * 0.05).prod(-1), "cute.ReductionOp.MUL"),
            (
                "prod_function",
                lambda x: torch.prod(x * 0.05, dim=-1),
                "cute.ReductionOp.MUL",
            ),
            ("amax_method", lambda x: x.amax(-1), "cute.ReductionOp.MAX"),
            ("amax_function", lambda x: torch.amax(x, dim=-1), "cute.ReductionOp.MAX"),
            ("amin_method", lambda x: x.amin(-1), "cute.ReductionOp.MIN"),
            ("amin_function", lambda x: torch.amin(x, dim=-1), "cute.ReductionOp.MIN"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_n_reduce_compiled_matches_reference(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by reduction (sum, mean, prod, amax, amin
        #   as method or function): for N equal to 128 and then 96, compiles an epilogue returning
        #   ReLU plus the reduction over groups of 16 along N as a compressed aux output. Checks
        #   main and aux against references, that generated code contains the expected CuTe
        #   reduction op, and the compressed local-reduce plan for group 16.
        _, reduce_fn, cute_op = case
        m = 128
        group = 16

        for n in (128, 96):

            def epilogue_fn(acc):
                # [FUNCTION DOC] HELPER: epilogue returning ReLU of the accumulator and the
                #   parametrized reduction over groups of 16 on the last axis of a grouped float32
                #   view (closes over the loop variables and the reduction).
                x = acc.float().view(m, -1, group)
                return acc.relu(), reduce_fn(x)

            def fn(a, b):
                # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm
                #   with the reduction epilogue and QUACK backend.
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK"},
                )

            a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
            (actual, aux), (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

            self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
            FileCheck().check(cute_op).run(code)
            self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("axis", (0, 1))
    def test_mm_tuple_aux_local_reduce_supports_explicit_group_count(self, axis):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by axis (0 or 1): epilogue returns ReLU
        #   and a group sum with an explicit group count along rows (axis 0) or along N (axis 1),
        #   group 16. Output matches references; generated code has a local-reduce plan with the
        #   geometry and callbacks only for axis 0.
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the sum over groups of 16,
            #   along N for axis 1 or along rows for axis 0, using the static shapes.
            match axis:
                case 1:
                    x = acc.float().view(m, n // group, group)
                    return acc.relu(), x.sum(-1)
                case 0:
                    x = acc.float().view(m // group, group, n)
                    return acc.relu(), x.sum(1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the grouped-sum epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        self.assertLocalReduceAuxCode(code, group, axis=axis, callbacks=axis == 0)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("axis", (0, 1))
    def test_mm_tuple_aux_local_reduce_dynamic_explicit_group_count(self, axis):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by axis: same grouped-sum aux as the
        #   explicit group count test but with dynamic shapes, where group counts come from the
        #   symbolic shape. Runs two shape sets (varying N for axis 0, varying M for axis 1) on
        #   one compiled function and checks main and aux against references.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue reading the accumulator shape at trace time and
            #   returning ReLU plus the sum over groups of 16 along N (axis 1) or along rows (axis
            #   0).
            m, n = acc.shape
            match axis:
                case 1:
                    x = acc.float().view(m, n // 16, 16)
                    return acc.relu(), x.sum(-1)
                case 0:
                    x = acc.float().view(m // 16, 16, n)
                    return acc.relu(), x.sum(1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the shape-reading epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        cases = (
            ((128, 64, 128), (128, 64, 192))
            if axis == 0
            else ((128, 64, 128), (256, 64, 128))
        )
        for m, k, n in cases:
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
            actual, aux = compiled(a, b)

            self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (2, 16, 32))
    @parametrize(
        "case",
        (
            ("sum", lambda x: x.sum(1), "local_reduce_combine_fn"),
            ("mean", lambda x: x.mean(1), " / {group}.0"),
            ("prod", lambda x: (x * 0.05).prod(1), "lhs * rhs"),
            ("amax", lambda x: x.amax(1), "cutlass.max"),
            ("amin", lambda x: x.amin(1), "cutlass.min"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_m_reduce_compiled_matches_reference(self, case, group):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by group size (2, 16, 32) and reduction
        #   (sum, mean, prod, amax, amin): axis-0 grouped reduction as aux output. Checks
        #   references, that generated code contains a reduction-specific marker (combine
        #   callback, divide by group, product, or max/min), and a local-reduce plan with axis 0
        #   and callbacks.
        _, reduce_fn, code_check = case
        m = 128
        n = 128
        code_check = code_check.format(group=group)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the parametrized reduction over
            #   row groups of the float32 accumulator.
            x = acc.float().view(-1, group, n)
            return acc.relu(), reduce_fn(x)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the row-group reduction epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        FileCheck().check(code_check).run(code)
        self.assertLocalReduceAuxCode(code, group, axis=0, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_supports_multiple_same_shape_outputs(self):
        # [FUNCTION DOC] SM100+ CUDA test: epilogue returns three same-shape outputs (ReLU,
        #   accumulator plus one, accumulator times two). Compiled results match eager and float64
        #   references and generated code passes an aux output tuple.
        m = 128
        n = 128

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU of the accumulator and two float32
            #   pointwise variants (plus one, times two).
            acc_f = acc.float()
            return acc.relu(), acc_f + 1, acc_f * 2

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the multi-output epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux0, aux1), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            (actual, aux0, aux1),
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        FileCheck().check("aux_outs=(").check(",)").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_combines_same_shape_and_compressed_local_reduce(self):
        # [FUNCTION DOC] SM100+ CUDA test: outputs are the main ReLU, a compressed axis-0
        #   group-sum aux (group 16) and a same-shape scaled aux. Checks each against references
        #   (compressed aux with 1e-3 tolerance) and that generated code has aux outputs and the
        #   axis-0 local-reduce plan with callbacks.
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU, the group sum over rows of the
            #   grouped float32 view and the half-scaled grouped view reshaped to the original
            #   shape.
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.sum(1), (x * 0.5).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the three-output epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, local_reduce_aux, same_shape_aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected_actual, _, expected_aux = epilogue_fn(a @ b)
        (
            high_precision_actual,
            high_precision_local_reduce_aux,
            high_precision_aux,
        ) = epilogue_fn(a.double() @ b.double())

        self.assertMatchesLowPrecisionEager(
            actual, expected_actual, high_precision_actual, a.shape[1]
        )
        torch.testing.assert_close(
            local_reduce_aux,
            high_precision_local_reduce_aux.float(),
            atol=1e-3,
            rtol=1e-3,
        )
        self.assertMatchesLowPrecisionEager(
            same_shape_aux, expected_aux, high_precision_aux, a.shape[1]
        )
        FileCheck().check("aux_outs=").run(code)
        self.assertLocalReduceAuxCode(code, group, axis=0, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_physical_local_reduce_supports_finalize_expression(self):
        # [FUNCTION DOC] SM100+ CUDA test: a physical (cross-warp) group-64 axis-0 maximum
        #   followed by an add of one and a square root as the aux output. Verifies results and
        #   that generated code reduces with a max, takes a square root, and routes through the
        #   finalize callback.
        m = 128
        n = 128
        group = 64

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the square root of one plus the
            #   absolute maximum over row groups of 64.
            x = acc.float().view(-1, group, n)
            return acc.relu(), (x.abs().amax(1) + 1.0).sqrt()

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the finalize-expression epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        FileCheck().check("cutlass.max").check("cute.math.sqrt").check(
            "local_reduce_finalize_fn"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group, axis=0, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("wrapper", ("squeeze", "view"))
    def test_mm_tuple_aux_physical_local_reduce_supports_wrapped_finalize(
        self, wrapper
    ):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by wrapper (squeeze or view): a group-64
        #   sum over N, reshaped via the wrapper to the compressed shape and incremented by one,
        #   is accepted as a physical local reduction with finalize. Checks references and that
        #   combine and finalize callbacks appear in the generated code.
        m = 128
        n = 128
        group = 64

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the keepdim group sum over N
            #   (groups of 64) squeezed or viewed to the compressed shape, plus one.
            x = acc.float().view(m, -1, group)
            reduced = x.sum(-1, keepdim=True)
            reduced = (
                reduced.squeeze(-1)
                if wrapper == "squeeze"
                else reduced.view(m, n // group)
            )
            return acc.relu(), reduced + 1.0

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the wrapped-finalize epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        FileCheck().check("local_reduce_combine_fn").check(
            "local_reduce_finalize_fn"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_rejects_composite_physical_local_reductions(self):
        # [FUNCTION DOC] SM100+ CUDA test: an aux output built by adding a sum and an amax over
        #   the same row groups of 64 (two reductions combined) must fail to compile with an error
        #   saying only a single physical local reduction is supported.
        m = 128
        n = 128
        group = 64

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the sum plus the maximum over
            #   row groups of 64.
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.sum(1) + x.amax(1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the composite-reduction epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "single physical local reduction"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_nvfp4_pack_matches_reference(self):
        # [FUNCTION DOC] CPU test of nvfp4_pack: builds float32 inputs at and around every FP4
        #   E2M1 rounding boundary (just below, at, just above, both signs), special values
        #   (extremes, tiny, signed zeros, infinities) and 4096 seeded random values, shaped as
        #   pairs along the last dim, plus a non-contiguous permuted view. For each, the packed
        #   result must be a contiguous float4 e2m1 pair dtype whose bytes equal the reference
        #   conversion via the test quantization helpers; under a fake tensor mode, the result
        #   must have the same dtype, shape and strides.
        from torch._subclasses.fake_tensor import FakeTensorMode

        boundaries = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
        )
        lower = torch.nextafter(boundaries, torch.full_like(boundaries, -torch.inf))
        upper = torch.nextafter(boundaries, torch.full_like(boundaries, torch.inf))
        smallest = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
        special = torch.tensor(
            [
                -torch.finfo(torch.float32).max,
                -torch.finfo(torch.float32).tiny,
                -smallest,
                -0.0,
                0.0,
                smallest,
                torch.finfo(torch.float32).tiny,
                torch.finfo(torch.float32).max,
                -torch.inf,
                torch.inf,
            ]
        )
        random = torch.empty(4096).uniform_(
            -8.0, 8.0, generator=torch.Generator().manual_seed(0)
        )
        values = torch.cat(
            (-upper, -boundaries, -lower, special, lower, boundaries, upper, random)
        ).view(-1, 2)
        inputs = (values, values[:8].view(2, 4, 2).permute(1, 0, 2))
        for input in inputs:
            with self.subTest(stride=input.stride()):
                expected = pack_uint4(_f32_to_floatx_unpacked(input, 2, 1)).squeeze(-1)
                actual = nvfp4_pack(input)
                self.assertIs(actual.dtype, torch.float4_e2m1fn_x2)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.view(torch.uint8), expected)

                with FakeTensorMode() as mode:
                    fake = nvfp4_pack(mode.from_tensor(input))
                self.assertIs(fake.dtype, torch.float4_e2m1fn_x2)
                self.assertEqual(fake.shape, actual.shape)
                self.assertEqual(fake.stride(), actual.stride())

    def test_quant_scale_fake_strides_match_eager(self):
        # [FUNCTION DOC] Checks mx_e8m0_scale on a fake tensor created from an expanded (stride
        #   zero) amax tensor yields the same output strides as the eager call on the real tensor.
        from torch._subclasses.fake_tensor import FakeTensorMode

        amax = torch.rand(2, 1).expand(2, 4)
        eager_stride = mx_e8m0_scale(amax).stride()
        with FakeTensorMode() as mode:
            fake_amax = mode.from_tensor(amax)
        self.assertEqual(mx_e8m0_scale(fake_amax).stride(), eager_stride)

    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    def test_quant_scale_rounding_args(self):
        # [FUNCTION DOC] CUDA test of mx_e8m0_scale eager results on a fixed amax vector (zero,
        #   448 and neighbours, 512, infinity, NaN): floor and rceil rounding give the listed byte
        #   codes, and the default equals rceil. Also checks rceil on special float bit patterns
        #   scaled by 448 (denormals, powers of two, max finite, infinity, NaN), floor and rceil
        #   for maximum values of 6 and 57344 (exact boundary versus next float), and that an
        #   unknown rounding mode or a pack value of 4 raises ValueError.
        amax = torch.tensor(
            [0.0, 448.0, 449.0, 500.0, 511.0, 512.0, float("inf"), float("nan")],
            device="cuda",
        )
        floor = mx_e8m0_scale(amax, rounding="floor").view(torch.uint8)
        rceil = mx_e8m0_scale(amax, rounding="rceil").view(torch.uint8)
        default = mx_e8m0_scale(amax).view(torch.uint8)

        self.assertEqual(
            floor,
            torch.tensor(
                [0, 127, 127, 127, 127, 128, 247, 255],
                device="cuda",
                dtype=torch.uint8,
            ),
        )
        self.assertEqual(
            rceil,
            torch.tensor(
                [0, 127, 128, 128, 128, 128, 254, 255],
                device="cuda",
                dtype=torch.uint8,
            ),
        )
        self.assertEqual(default, rceil)

        descale_bits = torch.tensor(
            [
                0x00000000,
                0x00000001,
                0x003FFFFF,
                0x00400000,
                0x00400001,
                0x007FFFFF,
                0x00800000,
                0x3F000000,
                0x3F800000,
                0x3F800001,
                0x7F7FFFFF,
                0x7F800000,
                0x7FC00000,
            ],
            device="cuda",
            dtype=torch.int32,
        )
        self.assertEqual(
            mx_e8m0_scale(
                descale_bits.view(torch.float32) * 448.0, rounding="rceil"
            ).view(torch.uint8),
            torch.tensor(
                [0, 0, 0, 0, 1, 1, 1, 126, 127, 128, 254, 254, 255],
                device="cuda",
                dtype=torch.uint8,
            ),
        )

        for max_value in (6.0, 57344.0):
            format_amax = torch.tensor(
                [
                    max_value,
                    torch.nextafter(
                        torch.tensor(max_value), torch.tensor(float("inf"))
                    ),
                ],
                device="cuda",
            )
            self.assertEqual(
                mx_e8m0_scale(format_amax, max_value=max_value, rounding="floor").view(
                    torch.uint8
                ),
                torch.tensor([127, 127], device="cuda", dtype=torch.uint8),
            )
            self.assertEqual(
                mx_e8m0_scale(format_amax, max_value=max_value, rounding="rceil").view(
                    torch.uint8
                ),
                torch.tensor([127, 128], device="cuda", dtype=torch.uint8),
            )

        with self.assertRaisesRegex(ValueError, "rounding must be 'floor', 'rceil'"):
            mx_e8m0_scale(amax, rounding="nearest")
        with self.assertRaisesRegex(ValueError, "pack must be 1 or 2"):
            mx_e8m0_scale(amax, pack=4)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("rceil", 448.0, 450.0, 128),
            ("floor", 448.0, 450.0, 127),
            ("rceil", 448.0, 2.0**-140, 0),
            ("rceil", 448.0, float("inf"), 254),
            ("floor", 448.0, float("inf"), 247),
            ("floor", 2.0**-122, 64.0, 255),
        ),
        name_fn=lambda c: f"{c[0]}_max{c[1]:g}_code{c[3]}".replace(".", "_"),
    )
    def test_mm_tuple_aux_mx_scale_rounding(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by (rounding, max_value, injected value,
        #   expected byte code): the epilogue adds a constant to a zero accumulator, takes
        #   group-of-32 absolute maxima and encodes them with mx_e8m0_scale as a compressed aux
        #   output. All aux bytes must equal the expected code; generated code must contain the
        #   matching PTX conversion instruction (round-to-zero for floor, round-up saturating for
        #   rceil) with no finalize callback and the group-32 local-reduce plan.
        rounding, max_value, value, expected_code = case
        code_check = (
            "cvt.rz.ue8m0x2.f32"
            if rounding == "floor"
            else "cvt.rp.satfinite.ue8m0x2.f32"
        )
        m = 64
        n = 128
        k = 64
        group = 32

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the accumulator and the E8M0 scale of
            #   the per-group (32) absolute maxima of the accumulator plus the injected constant.
            x = (acc.float() + value).view(m, -1, group)
            return acc, mx_e8m0_scale(
                x.abs().amax(-1), max_value=max_value, rounding=rounding
            )

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the scale epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        (_, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertEqual(
            aux.view(torch.uint8),
            torch.full_like(aux.view(torch.uint8), expected_code),
        )
        FileCheck().check(code_check).check_not("local_reduce_finalize_fn").run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_inline_asm_uses_compact_reduction(self):
        # [FUNCTION DOC] SM100+ CUDA test: an inline asm elementwise op (float to unsigned
        #   conversion) applied to a group-of-32 maximum must be lowered as an inline-asm
        #   intrinsic on the compact reduced value rather than via a finalize callback. Inputs are
        #   all ones; the aux must equal the group maximum cast to uint8 and then float; the main
        #   output equals the matmul.
        m, n, k, group = 64, 96, 64, 32

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the accumulator and the float version of
            #   an inline-asm conversion of the per-group maximum.
            reduced = acc.float().view(m, -1, group).amax(-1)
            code = inline_asm_elementwise(
                reduced,
                asm_str="cvt.rzi.u16.f32 $0, $1;",
                constraints="=h,f",
                dtype=torch.uint8,
            )
            return acc, code.float()

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the inline-asm epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.ones(k, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected = a @ b
        expected_aux = (
            expected.float().view(m, -1, group).amax(-1).to(torch.uint8).float()
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(aux, expected_aux)
        FileCheck().check("inline_asm_elementwise_intrinsic(").check_not(
            "local_reduce_finalize_fn"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_inline_asm_pack2_broadcasts_and_repeats_scalar(self):
        # [FUNCTION DOC] SM100+ CUDA test: a pack-2 inline asm op takes row, column and scalar
        #   captured tensors plus the accumulator and sums them per lane; with zero A and B, the
        #   output must equal row plus col plus scalar (broadcasts and scalar repeated across the
        #   packed pair). Generated code must show pack equal to 2.
        m = k = n = 64

        def fn(a, b, row, col, scalar):
            # [FUNCTION DOC] HELPER: function under compilation, defining a pack-2 inline-asm
            #   epilogue over row, col, scalar and accumulator and calling flex_gemm on mm.
            def epilogue_fn(acc):
                # [FUNCTION DOC] HELPER: epilogue applying a pack-2 inline asm that adds the
                #   three captured values to the accumulator for each of the two packed lanes.
                return inline_asm_elementwise(
                    row,
                    col,
                    scalar,
                    acc.float(),
                    asm_str=(
                        "add.f32 $0, $2, $4; add.f32 $0, $0, $6; "
                        "add.f32 $0, $0, $8; add.f32 $1, $3, $5; "
                        "add.f32 $1, $1, $7; add.f32 $1, $1, $9;"
                    ),
                    constraints="=f,=f,f,f,f,f,f,f,f,f",
                    dtype=torch.float32,
                    pack=2,
                )

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        row = torch.randn(1, n, device="cuda", dtype=torch.float32)
        col = torch.randn(m, 1, device="cuda", dtype=torch.float32)
        scalar = torch.randn(1, 1, device="cuda", dtype=torch.float32)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            row,
            col,
            scalar,
        )

        torch.testing.assert_close(actual, row + col + scalar)
        FileCheck().check("pack=2").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        ((torch.float16, "Float16"), (torch.bfloat16, "BFloat16")),
        name_fn=lambda case: str(case[0]).removeprefix("torch."),
    )
    def test_mm_inline_asm_pack2_restores_16bit_inputs(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by 16-bit dtype (float16 or bfloat16):
        #   pack-2 inline asm moves 16-bit captured inputs (a row and a constant scalar) through
        #   unchanged; output must equal the row broadcast over M, and generated code must convert
        #   to the matching CuTe 16-bit type and use pack 2.
        dtype, cute_type = case
        m = k = n = 64

        def fn(a, b, row):
            # [FUNCTION DOC] HELPER: function under compilation, defining the 16-bit pack-2
            #   inline-asm epilogue and calling flex_gemm on mm.
            def epilogue_fn(acc):
                # [FUNCTION DOC] HELPER: epilogue creating a constant 16-bit scalar and applying
                #   a pack-2 inline asm that copies the row values to the output lanes.
                scalar = torch.full((), 0.5, device="cuda", dtype=dtype)
                return inline_asm_elementwise(
                    row,
                    scalar,
                    acc.float(),
                    asm_str="mov.b16 $0, $2; mov.b16 $1, $3;",
                    constraints="=h,=h,h,h,h,h,f,f",
                    dtype=dtype,
                    pack=2,
                )

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        row = torch.randn(1, n, device="cuda", dtype=dtype)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, row
        )

        self.assertEqual(actual, row.expand(m, n))
        FileCheck().check(f".to(cutlass.{cute_type})").check("pack=2").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("nonpositive_pack", 0, "=f,f", "requires pack >= 1"),
            ("output_count", 2, "=f,f,f", "requires 2 output constraints"),
            ("input_count", 2, "=f,=f,f", "requires 2 input constraints"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_inline_asm_rejects_invalid_pack_contract(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by invalid inline-asm pack contracts
        #   (pack zero, wrong output constraint count, wrong input constraint count): compiling
        #   must raise ValueError or InductorError whose message matches the case (pack must be at
        #   least one, or the required number of output or input constraints).
        _, pack, constraints, error = case

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, defining an inline-asm epilogue
            #   with the parametrized pack and constraints and calling flex_gemm on mm.
            def epilogue_fn(acc):
                # [FUNCTION DOC] HELPER: epilogue applying a simple inline-asm move with the
                #   parametrized pack and constraint string.
                return inline_asm_elementwise(
                    acc.float(),
                    asm_str="mov.f32 $0, $1;",
                    constraints=constraints,
                    dtype=torch.float32,
                    pack=pack,
                )

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.zeros(64, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(64, 64, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex((ValueError, InductorError), error):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("mx", mx_e8m0_scale, "cvt.rp.satfinite.ue8m0x2.f32"),
            ("nvfp4", nvfp4_e4m3_scale, " / 6.0"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_large_n_local_reduce_supports_scale_finalizer(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by scale function (MX E8M0 or NVFP4
        #   E4M3): a physical group-64 absolute maximum along N is converted to a scale and
        #   returned as compressed aux. With all-ones inputs main and aux equal the eager
        #   epilogue; generated code must contain the expected conversion marker, a finalize
        #   callback, and the callback local-reduce plan for group 64.
        _, scale_fn, code_check = case
        m = 64
        n = 128
        k = 64
        group = 64

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the parametrized scale function
            #   applied to the group-of-64 absolute maxima along N.
            x = acc.float().view(m, -1, group)
            return acc.relu(), scale_fn(x.abs().amax(-1))

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the scale epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.ones(k, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected, expected_aux = epilogue_fn(a @ b)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(aux.float(), expected_aux.float())
        FileCheck().check(code_check).check("local_reduce_finalize_fn").run(code)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        ((16, 128), (32, 96)),
        name_fn=lambda case: f"group{case[0]}_n{case[1]}",
    )
    def test_mm_tuple_aux_mx_scale_pack2(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by (group, N) of (16, 128) and (32, 96):
        #   MX scale with pack 2 over group maxima, where B contains infinity and NaN column
        #   blocks. Non-NaN main values and all scale bytes must match eager; scale bytes 0, 254
        #   and 255 must each occur; generated code must show pack 2 and a broadcast, no finalize
        #   callback, and the group local-reduce plan.
        group, n = case
        m = k = 16

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the pack-2 MX scale of group
            #   absolute maxima.
            x = acc.float().view(m, -1, group)
            return acc.relu(), mx_e8m0_scale(x.abs().amax(-1), pack=2)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the pack-2 scale epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        b[:, group : 2 * group] = float("inf")
        b[:, 2 * group : 3 * group] = float("nan")
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected, expected_aux = epilogue_fn(a @ b)
        non_nan = ~torch.isnan(expected)
        torch.testing.assert_close(actual[non_nan], expected[non_nan])
        self.assertEqual(aux.view(torch.uint8), expected_aux.view(torch.uint8))
        for expected_code in (0, 254, 255):
            self.assertTrue((aux.view(torch.uint8) == expected_code).any())
        FileCheck().check("pack=2").check("broadcast_to").check_not(
            "local_reduce_finalize_fn"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_physical_mx_scale_preserves_nan(self):
        # [FUNCTION DOC] SM100+ CUDA test: physical group-64 MX scale where one B element is
        #   NaN; scale bytes must equal the eager ones, include both the zero code and the NaN
        #   code 255, and generated code uses a finalize callback with the callback plan.
        m = k = 16
        n = 128
        group = 64

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the MX scale of group-of-64
            #   absolute maxima.
            x = acc.float().view(m, -1, group)
            return acc.relu(), mx_e8m0_scale(x.abs().amax(-1))

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the MX scale epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        b[0, 0] = float("nan")
        (_, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        _, expected_aux = epilogue_fn(a @ b)
        self.assertEqual(aux.view(torch.uint8), expected_aux.view(torch.uint8))
        self.assertTrue((aux.view(torch.uint8) == 0).any())
        self.assertTrue((aux.view(torch.uint8) == 255).any())
        FileCheck().check("local_reduce_finalize_fn").run(code)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_physical_mx_scale_zero_preserves_min_value(self):
        # [FUNCTION DOC] SM100+ CUDA test: with zero B, the physical MX scale of a zero maximum
        #   is converted to float, doubled, and returned; every aux element must equal the doubled
        #   smallest scale value (a power of two at minus 126 exponent) and match eager exactly.
        #   Generated code uses a finalize callback.
        m = 16
        n = 128
        k = 16
        group = 64

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the accumulator and twice the float
            #   version of the MX scale of group maxima.
            x = acc.float().view(m, -1, group)
            return acc, mx_e8m0_scale(x.abs().amax(-1)).float() * 2

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the doubled-scale epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        (_, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        _, expected_aux = epilogue_fn(a @ b)
        torch.testing.assert_close(aux, expected_aux, rtol=0, atol=0)
        self.assertEqual(aux, torch.full_like(aux, 2.0**-126))
        FileCheck().check("local_reduce_finalize_fn").run(code)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_physical_nvfp4_scale_rounds_before_pointwise(self):
        # [FUNCTION DOC] SM100+ CUDA test: the NVFP4 E4M3 scale of group maxima is rounded to
        #   float8 before being doubled in float; the aux must equal the eager value exactly and
        #   generated code must show the float8 e4m3 conversion followed by the finalize callback.
        m = 16
        n = 128
        k = 64
        group = 64

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the accumulator and twice the float
            #   version of the E4M3 scale of group-of-64 maxima.
            x = acc.float().view(m, -1, group)
            return acc, nvfp4_e4m3_scale(x.abs().amax(-1)).float() * 2

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the E4M3 scale epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.ones(k, n, device="cuda", dtype=torch.bfloat16)
        (_, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        _, expected_aux = epilogue_fn(a @ b)
        torch.testing.assert_close(aux, expected_aux, rtol=0, atol=0)
        FileCheck().check("cutlass.Float8E4M3FN").check("local_reduce_finalize_fn").run(
            code
        )
        self.assertLocalReduceAuxCode(code, group, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("reduction", ("amax", "amin"))
    def test_mm_tuple_aux_physical_extrema_preserve_nan(self, reduction):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by amax or amin: a group-of-64 extremum
        #   with a NaN input must propagate NaN to the first group only (second group stays zero).
        #   Compares with eager allowing NaN equality, checks the generated code uses the matching
        #   CuTe extremum helper and a finalize callback.
        m = 16
        n = 128
        k = 16
        group = 64

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the accumulator and the parametrized
            #   extremum over group-of-64 views.
            x = acc.float().view(m, -1, group)
            return acc, getattr(x, reduction)(-1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the extremum epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        b[0, 0] = float("nan")
        (_, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        _, expected_aux = epilogue_fn(a @ b)
        torch.testing.assert_close(aux, expected_aux, equal_nan=True)
        self.assertTrue(torch.isnan(aux[:, 0]).all())
        self.assertEqual(aux[:, 1], torch.zeros_like(aux[:, 1]))
        FileCheck().check(f"cutlass.{reduction[1:]}").check(
            "local_reduce_finalize_fn"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("case", ("clamp", "clamp_min", "clamp_max"))
    def test_mm_epilogue_clamp_preserves_nan(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by clamp, clamp_min and clamp_max: the
        #   epilogue adds NaN to the accumulator and clamps; all results must be NaN (clamp must
        #   propagate NaN). Generated code must use the matching math max and min helpers and must
        #   not contain an inequality comparison.
        m = n = k = 16

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue adding NaN to the float32 accumulator and applying
            #   the parametrized clamp.
            x = acc.float() + float("nan")
            match case:
                case "clamp":
                    return x.clamp(-1.0, 1.0)
                case "clamp_min":
                    return x.clamp_min(-1.0)
                case "clamp_max":
                    return x.clamp_max(1.0)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the clamp epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertTrue(torch.isnan(actual).all())
        check = FileCheck().check("cutlass_math.")
        match case:
            case "clamp":
                check = check.check("max").check("cutlass_math.min")
            case "clamp_min":
                check = check.check("max")
            case "clamp_max":
                check = check.check("min")
        check.check_not("operator.ne").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_fragment_group_mx_quant_feed_uses_tensorSSA(self):
        # [FUNCTION DOC] SM100+ CUDA test: MX quantization feed where the per-group (32 along N)
        #   scale is computed from the keepdim maximum and the grouped values are multiplied by
        #   the scale reciprocal, cast to float8 e4m3, with the squeezed scale as aux. With one
        #   NaN in B the float8 output (NaN equal) and scale bytes must match eager; scale bytes
        #   include 0 and 255; generated code uses the PTX conversion and no finalize callback and
        #   the group-32 plan.
        m = 64
        n = 128
        k = 64
        group = 32

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue computing MX scale of group maxima, returning the
            #   float8 e4m3 scaled grouped values reshaped to M by N, and the squeezed scale.
            x = acc.float().view(m, -1, group)
            scale = mx_e8m0_scale(x.abs().amax(-1, keepdim=True))
            return (
                (x * scale.reciprocal()).view(m, n).to(torch.float8_e4m3fn),
                scale.squeeze(-1),
            )

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the quantization epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        b[0, 0] = float("nan")
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        x = (a @ b).float().view(m, -1, group)
        expected_aux = mx_e8m0_scale(x.abs().amax(-1, keepdim=True))
        expected = (
            (x * expected_aux.float().reciprocal()).view(m, n).to(torch.float8_e4m3fn)
        )
        torch.testing.assert_close(actual.float(), expected.float(), equal_nan=True)
        self.assertEqual(
            aux.view(torch.uint8), expected_aux.squeeze(-1).view(torch.uint8)
        )
        self.assertTrue((aux.view(torch.uint8) == 0).any())
        self.assertTrue((aux.view(torch.uint8) == 255).any())
        FileCheck().check("cvt.rp.satfinite.ue8m0x2.f32").check_not(
            "local_reduce_finalize_fn"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (64, 128))
    @parametrize(
        "case",
        (
            ("sum", lambda x: x.sum(1), "local_reduce_combine_fn"),
            ("mean", lambda x: x.mean(1), " / {group}.0"),
            ("amax", lambda x: x.amax(1), "cutlass.max"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_m_reduce_supports_cta_group(self, case, group):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by group (64 or 128) and reduction (sum,
        #   mean, amax): axis-0 grouped reduction across CTA tiles as aux output. Checks
        #   references and that generated code has the reduction marker and the axis-0 plan with
        #   callbacks.
        _, reduce_fn, code_check = case
        m = 128
        n = 128
        code_check = code_check.format(group=group)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the parametrized reduction over
            #   row groups.
            x = acc.float().view(-1, group, n)
            return acc.relu(), reduce_fn(x)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the row-group reduction epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        FileCheck().check(code_check).run(code)
        self.assertLocalReduceAuxCode(code, group, axis=0, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_local_m_reduce_tuned_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: row-group (16) sum aux output with the tuned option
        #   on, so the candidate configs are auto-selected; main and aux are compared with
        #   references.
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the sum over row groups of 16.
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.sum(1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue and tuned on.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        actual, aux = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_local_m_reduce_tuned_internal_transpose(self):
        # [FUNCTION DOC] SM100+ CUDA test: tuned flex_gemm with B passed as a transposed view
        #   and a row-group (16) sum aux. Picks the first two candidate configs valid for axis-0
        #   group 16 local reduce, patches the candidate list to them (asserts two exist),
        #   compiles, and compares with references computed on the transposed B.
        m = 128
        n = 192
        k = 64
        group = 16

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the sum over row groups of 16.
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.sum(1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the transposed B argument and tuned on.
            return flex_gemm(
                torch.mm,
                (a, b.mT),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

        from torch._inductor.kernel.flex_gemm.constraints import (
            validate_flex_gemm_local_reduce_config,
        )
        from torch._inductor.template_heuristics import (
            flex_gemm as flex_gemm_heuristics,
        )

        configs = tuple(
            config
            for config in flex_gemm_heuristics.candidate_gemm_configs_for_device(
                a.device
            )
            if validate_flex_gemm_local_reduce_config(config, group, 0)
        )[:2]
        self.assertEqual(len(configs), 2)
        with mock.patch(
            "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
            return_value=configs,
        ):
            actual, aux = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        self.assertLocalReduceAuxMatches(actual, aux, a, b.mT, epilogue_fn)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_local_m_reduce_supports_tail_m(self):
        # [FUNCTION DOC] SM100+ CUDA test: row-group (16) sum aux with M equal to 96, which
        #   leaves a partial last tile along M; output and aux match references.
        m = 96
        n = 128
        group = 16

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the sum over row groups of 16.
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.sum(1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), _ = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_local_m_reduce_rejects_var_reduction(self):
        # [FUNCTION DOC] SM100+ CUDA test: a row-group variance aux output (correction zero)
        #   must fail to compile with an error stating it does not map to a CuTe TensorSSA
        #   reduction.
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the variance over row groups of
            #   16.
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.var(1, correction=0)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the variance epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "does not map to a CuTe TensorSSA"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (2, 32))
    def test_mm_tuple_aux_local_reduce_supports_group_extremes(self, group):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by group size 2 or 32: sum over N groups
        #   as aux. Checks references, that generated code contains the CuTe add reduction, and
        #   the compressed local-reduce plan.
        m = 128
        n = 128

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the sum over N groups of the
            #   parametrized size.
            x = acc.float().view(m, -1, group)
            return acc.relu(), x.sum(-1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the grouped-sum epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        FileCheck().check("cute.ReductionOp.ADD").run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_local_reduce_supports_tuple_shape_reshape(self):
        # [FUNCTION DOC] SM100+ CUDA test: the grouped view is requested with a reshape taking a
        #   tuple shape instead of positional dimensions; analysis must still recognize the
        #   group-16 sum. Checks references and the compressed plan.
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue reshaping to a tuple shape of groups of 16 and
            #   returning ReLU plus the sum over the last axis.
            x = acc.float().reshape((m, -1, group))
            return acc.relu(), x.sum(-1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("sum_group64", 128, 64, lambda x: x.sum(-1), "lhs + rhs"),
            ("mean_group64", 128, 64, lambda x: x.mean(-1), " / 64.0"),
            ("amax_group64", 128, 64, lambda x: x.amax(-1), "cutlass.max"),
            ("sum_group128", 256, 128, lambda x: x.sum(-1), "lhs + rhs"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_n_reduce_supports_cta_subtile_group(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by (name, N, group, reduction, marker)
        #   for groups of 64 or 128 along N that span CTA subtiles (sum, mean, amax). Checks
        #   references, that the generated code contains the marker, includes callbacks in the
        #   plan, and does not mention a local_reduce_strategy.
        _, n, group, reduce_fn, code_check = case
        m = 128

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the parametrized reduction over
            #   N groups.
            x = acc.float().view(m, -1, group)
            return acc.relu(), reduce_fn(x)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the reduction epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        self.assertIn(code_check, code)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)
        self.assertNotIn("local_reduce_strategy", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_local_n_reduce_cta_subtile_group_tuned(self):
        # [FUNCTION DOC] SM100+ CUDA test: group-128 sum along N with tuned on, N equal to 256.
        #   Main output and aux match references, generated code has the callback plan, and a
        #   tuned cluster N of one appears in the config keys.
        m = 128
        n = 256
        group = 128

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the sum over N groups of 128.
            x = acc.float().view(m, -1, group)
            return acc.relu(), x.sum(-1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue and tuned on.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b)[0],
            high_precision_acc.relu(),
            a.shape[1],
        )
        torch.testing.assert_close(
            aux, epilogue_fn(high_precision_acc)[1].float(), atol=1e-3, rtol=1e-3
        )
        self.assertLocalReduceAuxCode(code, group, callbacks=True)
        self.assertIn("cluster_n', 1", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            (
                "variance_like",
                lambda x: ((x - x.mean(-1, keepdim=True)).square()).mean(-1) * 0.5
                + 1.0,
                " / 4.0",
                False,
            ),
            (
                "sum_keepdim_squeeze",
                lambda x: x.sum(-1, keepdim=True).squeeze(-1),
                "broadcast_to",
                False,
            ),
            (
                "stable_logsumexp",
                lambda x: (
                    (x - x.amax(-1, keepdim=True)).exp().sum(-1, keepdim=True).log()
                    + x.amax(-1, keepdim=True)
                ).view(x.shape[0], -1),
                "cute.math.log",
                True,
            ),
            (
                "logsumexp_method",
                lambda x: x.logsumexp(-1),
                "cute.math.log",
                True,
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_reduce_supports_chained_grouped_expressions(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by chained grouped expressions with group
        #   4: variance-like (mean of squared deviations), a keepdim sum then squeeze, a
        #   numerically stable logsumexp built from exp, sum, log and max, and the logsumexp
        #   method. Checks references and that generated code contains the add reduction (and max
        #   reduction for logsumexp) plus the expected marker, and the compressed plan.
        _, aux_fn, generated_check, checks_max = case
        m = 128
        n = 96
        group = 4

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the parametrized chained
            #   expression over groups of four along N.
            x = acc.float().view(m, -1, group)
            return acc.relu(), aux_fn(x)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the chained-expression epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        file_check = FileCheck()
        if checks_max:
            file_check = file_check.check("cute.ReductionOp.MAX")
        file_check.check("cute.ReductionOp.ADD").check(generated_check)
        file_check.run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_local_n_reduce_group32_filters_half_fragment_configs(self):
        # [FUNCTION DOC] SM100+ CUDA test of config validation: among candidate configs with
        #   tile M 128, tile N in 128, 160 or 224 and cluster M above one that accept group 16
        #   along N, none may accept group 32 (their per-thread fragment covers only half a
        #   group). Asserts such configs exist.
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
        )
        from torch._inductor.kernel.flex_gemm.constraints import (
            validate_flex_gemm_local_reduce_config,
        )

        candidates = candidate_gemm_configs_for_device(torch.device("cuda"))
        half_fragment_configs = [
            config
            for config in candidates
            if config.tile_m == 128
            and config.tile_n in (128, 160, 224)
            and config.cluster_m > 1
            and validate_flex_gemm_local_reduce_config(config, 16, 1)
        ]
        self.assertTrue(half_fragment_configs)
        self.assertTrue(
            all(
                not validate_flex_gemm_local_reduce_config(config, 32, 1)
                for config in half_fragment_configs
            )
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_reduce_result_feeds_main_output(self):
        # [FUNCTION DOC] SM100+ CUDA test: epilogue multiplies grouped values by the group sum
        #   plus one along N (group 16) so the reduction feeds the main output as a fragment-local
        #   TensorSSA computation. Result matches references; generated code has no feeds-main
        #   plan flag, no combine callback and no separate reduction output, but uses the add
        #   reduction.
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the grouped float32 values multiplied by
            #   the group sum plus one, reshaped to M by N.
            x = acc.float().view(m, -1, group)
            scale = x.sum(-1, keepdim=True) + 1.0
            return (x * scale).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the feed-main epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(high_precision_acc),
            a.shape[1],
        )
        self.assertNotIn("local_reduce_feeds_main=True", code)
        self.assertNotIn("local_reduce_combine_fn", code)
        FileCheck().check("cute.ReductionOp.ADD").check_not("local_reduce_out=").run(
            code
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (2, 4, 8, 16, 32))
    def test_mm_local_m_reduce_result_feeds_main_output(self, group):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by row group size 2, 4, 8, 16, 32:
        #   epilogue scales grouped values by the reciprocal of the row-group sum, so the
        #   reduction result feeds the main output. Inputs are uniform random bf16. Output must
        #   match references, and generated code must use the physical feed-main plan for that
        #   group.
        m = 128
        n = 64

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue grouping rows, multiplying values by the
            #   reciprocal of the row-group sum, and reshaping back to M by N.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the feed-main epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected = epilogue_fn(a @ b)
        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            epilogue_fn(high_precision_acc),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feeds_main_and_returns_scale(self):
        # [FUNCTION DOC] SM100+ CUDA test: row-group (8) sum scales the main output by its
        #   reciprocal and is also returned as a squeezed scale aux. Checks both against
        #   references (aux within 1e-3) and that generated code has a runtime local-reduce plan
        #   with axis 0 geometry, an output, the feeds-main flag and callbacks, without legacy
        #   arguments.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the grouped values times the reciprocal
            #   of the row-group sum, reshaped to M by N, and the squeezed row-group sum.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n), scale.squeeze(1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, scale), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected, expected_scale = epilogue_fn(a @ b)
        high_precision_expected, high_precision_scale = epilogue_fn(
            a.double() @ b.double()
        )
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            high_precision_expected,
            a.shape[1],
        )
        torch.testing.assert_close(
            scale,
            high_precision_scale.float(),
            atol=1e-3,
            rtol=1e-3,
        )
        FileCheck().check("local_reduce=FlexGemmRuntimeLocalReducePlan").check(
            self.localReduceGeometryPattern(group, 0)
        ).check("out=").check("feeds_main=True").check(
            "callbacks=FlexGemmLocalReduceCallbacks"
        ).check_not("local_reduce_group=").check_not("local_reduce_op").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_result_divides_main_output(self):
        # [FUNCTION DOC] SM100+ CUDA test: main output is the grouped values divided by the
        #   row-group (8) sum plus one. Matches references and generated code passes the physical
        #   feed-main check.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue dividing grouped values by the row-group sum plus
            #   one and reshaping to M by N.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True) + 1.0
            return (x / scale).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected = epilogue_fn(a @ b)
        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            epilogue_fn(high_precision_acc),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("consumer", ("main", "aux"))
    def test_mm_local_m_reduce_feed_main_supports_reversed_division(self, consumer):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by consumer (main or aux): the reduction
        #   sum plus one divided by the grouped values (reversed division) feeds either the main
        #   output or an aux next to ReLU. Matches references and the physical feed-main code
        #   check passes.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue computing the row-group sum plus one divided by
            #   the grouped values; returns it alone or paired with ReLU depending on consumer.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True) + 1.0
            transformed = (scale / x).view(m, n)
            if consumer == "main":
                return transformed
            return acc.relu(), transformed

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("consumer", ("main", "aux"))
    def test_mm_local_m_reduce_feed_main_supports_post_scale_pointwise(self, consumer):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by consumer (main or aux): the feed-main
        #   value is divided by the row-group sum plus one and then shifted by 0.5 (pointwise work
        #   after the scaling). Matches references and the physical feed-main code check passes.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue dividing grouped values by the row-group sum plus
            #   one, adding 0.5, and returning it alone or paired with ReLU depending on consumer.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True) + 1.0
            normalized = (x / scale + 0.5).view(m, n)
            if consumer == "main":
                return normalized
            return acc.relu(), normalized

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("consumer", ("main", "aux"))
    def test_mm_local_m_reduce_feed_main_supports_centered_mean(self, consumer):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by consumer (main or aux): grouped values
        #   minus the row-group mean plus 0.5 (centering) as the feed-main value. Matches
        #   references and the physical feed-main code check passes.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue subtracting the row-group mean from grouped
            #   values, adding 0.5, and returning it alone or paired with ReLU depending on
            #   consumer.
            x = acc.float().view(-1, group, n)
            mean = x.mean(1, keepdim=True)
            centered = (x - mean + 0.5).view(m, n)
            if consumer == "main":
                return centered
            return acc.relu(), centered

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "op",
        (
            "add",
            "sub",
            "method_add",
            "method_sub",
            "method_add_alpha",
            "method_sub_alpha",
            "torch_add_alpha",
            "torch_sub_alpha",
        ),
    )
    @parametrize("consumer", ("main", "aux"))
    def test_mm_local_m_reduce_feed_main_supports_reversed_add_sub(self, op, consumer):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by eight add/sub spellings (operator,
        #   tensor method with and without alpha, torch function with alpha) and consumer (main or
        #   aux): the row-group mean on the left side of an add or subtract with the grouped
        #   values, plus 0.5, feeds the output. Matches references; physical feed-main code check
        #   passes.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue combining the row-group mean with grouped values
            #   by the parametrized add or subtract form (including alpha variants), adding 0.5,
            #   reshaping, and returning it alone or paired with ReLU depending on consumer.
            x = acc.float().view(-1, group, n)
            mean = x.mean(1, keepdim=True)
            if op == "add":
                transformed = (mean + x + 0.5).view(m, n)
            elif op == "sub":
                transformed = (mean - x + 0.5).view(m, n)
            elif op == "method_add":
                transformed = mean.add(x).add(0.5).view(m, n)
            elif op == "method_sub":
                transformed = mean.sub(x).add(0.5).view(m, n)
            elif op == "method_add_alpha":
                transformed = mean.add(x, alpha=0.25).add(0.5).view(m, n)
            elif op == "method_sub_alpha":
                transformed = mean.sub(x, alpha=0.25).add(0.5).view(m, n)
            elif op == "torch_add_alpha":
                transformed = torch.add(mean, x, alpha=0.25).add(0.5).view(m, n)
            else:
                transformed = torch.sub(mean, x, alpha=0.25).add(0.5).view(m, n)
            if consumer == "main":
                return transformed
            return acc.relu(), transformed

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_pointwise_wrapper_rejects_two_values(self):
        # [FUNCTION DOC] SM100+ CUDA test: an epilogue dividing the grouped values by two
        #   different row-group reductions (sum plus one and amax plus one) and adding the results
        #   must fail to compile with an error that only one generated physical reduction is
        #   supported.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue adding the grouped values divided by the row-group
            #   sum plus one and by the row-group max plus one.
            x = acc.float().view(-1, group, n)
            sum_scale = x.sum(1, keepdim=True) + 1.0
            max_scale = x.amax(1, keepdim=True) + 1.0
            return (x / sum_scale + x / max_scale).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "one generated physical reduction"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("cross_warp", 128, 64, "same-warp axis-0 groups <= 32"),
            ("boundary", 136, 17, "fragment width 32"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_local_m_reduce_feed_main_rejects_unsupported_group(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by (name, M, group, error): row-group
        #   feed-main reductions with a cross-warp group of 64 (error says only same-warp axis-0
        #   groups up to 32 are supported) and an unaligned boundary group of 17 with M of 136
        #   (error mentions fragment width 32) must fail at compile time. Resets dynamo first to
        #   avoid carrying shape state between cases.
        # Preserve each case's static geometry instead of inheriting PGO state
        # from the other parameterization.
        torch._dynamo.reset()
        _, m, group, error = case
        n = 64

        def epilogue_fn(acc, group=group, m=m, n=n):
            # [FUNCTION DOC] HELPER: epilogue dividing grouped values by the row-group sum
            #   (reciprocal multiply) with group, M and N bound as default arguments from the
            #   enclosing test.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, error):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_specializes_explicit_group(self):
        # [FUNCTION DOC] SM100+ CUDA test: an explicit Python integer group size passed as an
        #   argument to the compiled function is specialized, so calling it with 8, 16 and 32
        #   triggers three compiles (frame count three) while each result matches references.
        torch._dynamo.reset()
        m = 128
        n = 64

        def epilogue_fn(acc, group):
            # [FUNCTION DOC] HELPER: epilogue grouping rows by the given group size and scaling
            #   grouped values by the reciprocal of the group sum plus one.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True) + 1.0
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b, group):
            # [FUNCTION DOC] HELPER: function under compilation, taking the group as an extra
            #   argument and calling flex_gemm on mm with an epilogue that closes over it.
            def epilogue(acc):
                # [FUNCTION DOC] HELPER: epilogue forwarding the accumulator and the enclosing
                #   group to the helper epilogue.
                return epilogue_fn(acc, group)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for group in (8, 16, 32):
            actual = compiled(a, b, group)
            self.assertMatchesEpilogue(
                actual,
                epilogue_fn(a @ b, group),
                epilogue_fn(a.double() @ b.double(), group),
                a.shape[1],
            )
        self.assertEqual(counter.frame_count, 3)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_same_shape_uses_broadcast_local_reduce(self):
        # [FUNCTION DOC] SM100+ CUDA test: a same-shape aux output that multiplies grouped
        #   values by the N-group mean plus one (group 16) must compute the mean fragment-locally.
        #   Checks references; generated code has the add reduction, a divide by 16 and no
        #   separate local reduce output.
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the grouped values times the
            #   N-group mean plus one, reshaped to M by N.
            x = acc.float().view(m, -1, group)
            aux = (x * (x.mean(-1, keepdim=True) + 1.0)).view(m, n)
            return acc.relu(), aux

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            (actual, aux),
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        FileCheck().check("cute.ReductionOp.ADD").check(" / 16.0").check_not(
            "local_reduce_out="
        ).run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_uses_physical_m_broadcast_local_reduce(self):
        # [FUNCTION DOC] SM100+ CUDA test: an aux output multiplying grouped values by the
        #   reciprocal of the row-group (8) sum plus one. Checks references and the physical
        #   feed-main code check (no group specified).
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the grouped values times the
            #   reciprocal of the row-group sum plus one.
            x = acc.float().view(-1, group, n)
            aux = (x * (x.sum(1, keepdim=True) + 1.0).reciprocal()).view(m, n)
            return acc.relu(), aux

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            (actual, aux),
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_divides_by_physical_m_broadcast_local_reduce(self):
        # [FUNCTION DOC] SM100+ CUDA test: an aux output dividing grouped values by the
        #   row-group (8) sum plus one. Checks references and the physical feed-main code check.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the grouped values divided by
            #   the row-group sum plus one.
            x = acc.float().view(-1, group, n)
            aux = (x / (x.sum(1, keepdim=True) + 1.0)).view(m, n)
            return acc.relu(), aux

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            (actual, aux),
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_outputs_share_physical_m_feed_main_reduce(self):
        # [FUNCTION DOC] SM100+ CUDA test: both the main output and an aux output use the same
        #   row-group (8) sum (the main by reciprocal scaling, the aux by multiplying with the sum
        #   plus one), so the physical reduction is shared. Checks both outputs against references
        #   and the physical feed-main code check.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the grouped values times the reciprocal
            #   of the row-group sum and the grouped values times the sum plus one.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            actual = (x * scale.reciprocal()).view(m, n)
            aux = (x * (scale + 1.0)).view(m, n)
            return actual, aux

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the two-output epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            (actual, aux),
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_rejects_repeated_equivalent_reductions(self):
        # [FUNCTION DOC] SM100+ CUDA test: the epilogue builds two separate grouped views of the
        #   same accumulator and takes an equivalent row-group sum from each (one per output);
        #   compiling must fail with an error that only one generated physical reduction is
        #   supported.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue producing two outputs, each from its own grouped
            #   view and its own row-group sum.
            x0 = acc.float().view(-1, group, n)
            out = (x0 * x0.sum(1, keepdim=True).reciprocal()).view(m, n)
            x1 = acc.float().view(-1, group, n)
            aux = (x1 * (x1.sum(1, keepdim=True) + 1.0)).view(m, n)
            return out, aux

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "one generated physical reduction"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_supports_regrouped_reduction_reuse(self):
        # [FUNCTION DOC] SM100+ CUDA test: the aux output reuses the row-group sum through a
        #   regrouped (reshaped twice) view of the grouped values. Compiles and both outputs match
        #   references; physical feed-main code check passes.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning grouped values times the reciprocal sum,
            #   and the same computation from a flattened and regrouped view reusing the sum.
            x0 = acc.float().view(-1, group, n)
            scale = x0.sum(1, keepdim=True)
            out = (x0 * scale.reciprocal()).view(m, n)
            x1 = x0.view(m, n).view(-1, group, n)
            aux = (x1 * scale.reciprocal()).view(m, n)
            return out, aux

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected, expected_aux = epilogue_fn(a @ b)
        high_precision_expected, high_precision_aux = epilogue_fn(
            a.double() @ b.double()
        )
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            high_precision_expected,
            a.shape[1],
        )
        self.assertMatchesLowPrecisionEager(
            aux,
            expected_aux,
            high_precision_aux,
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_rejects_mixed_grouped_layouts(self):
        # [FUNCTION DOC] SM100+ CUDA test: two outputs use different row-group sizes (8 and 4);
        #   compiling must fail with an error saying the outputs must share one grouped layout.
        m = 128
        n = 64

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning two reciprocal-sum normalizations, one
            #   over groups of 8 and one over groups of 4 rows.
            x8 = acc.float().view(-1, 8, n)
            out = (x8 * x8.sum(1, keepdim=True).reciprocal()).view(m, n)
            x4 = acc.float().view(-1, 4, n)
            aux = (x4 * x4.sum(1, keepdim=True).reciprocal()).view(m, n)
            return out, aux

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "share one grouped layout"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_rejects_composite_physical_reductions(self):
        # [FUNCTION DOC] SM100+ CUDA test: the scale combines a row-group sum and a row-group
        #   maximum before the reciprocal; compiling must fail saying only one generated physical
        #   reduction is supported.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue scaling grouped values by the reciprocal of the
            #   sum plus the maximum over row groups.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True) + x.amax(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "one generated physical reduction"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_rejects_unselected_physical_reduction(self):
        # [FUNCTION DOC] SM100+ CUDA test: the scale combines the row-group sum and the
        #   row-group sum of squares; compiling must fail with an error about the two-phase
        #   local-reduce source.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue scaling grouped values by the reciprocal of the
            #   sum plus the sum of squares over row groups.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True) + x.square().sum(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "two-phase local-reduce source"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_rejects_hidden_physical_reduction(self):
        # [FUNCTION DOC] SM100+ CUDA test: one reduction acts on the accumulator and another on
        #   the ReLU of the accumulator (a second, hidden physical reduction), and both feed the
        #   main output; compiling must fail saying only one generated physical reduction is
        #   supported.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue adding grouped values divided by one row-group sum
            #   to a row-group sum taken from ReLU of the accumulator.
            x = acc.float().view(-1, group, n)
            y = acc.float().relu().view(-1, group, n)
            scaled = x / (x.sum(1, keepdim=True) + 1.0)
            return (scaled + y.sum(1, keepdim=True)).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "one generated physical reduction"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        ("rms_main", "rms_aux", "centered"),
    )
    def test_mm_local_m_reduce_feed_main_rejects_source_expression(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by case (RMS normalization as main, RMS
        #   as aux next to ReLU, or mean-centered RMS normalization): the reduction source is
        #   itself an expression (squared values) feeding main, which is not supported; compiling
        #   must fail with an error about the two-phase local-reduce source.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue normalizing grouped values by the reverse square
            #   root of the mean of squares (optionally mean-centered), returned as main or as aux
            #   next to ReLU depending on case.
            x = acc.float().view(-1, group, n)
            if case == "centered":
                mean = x.mean(1, keepdim=True)
                rstd = (x.square().mean(1, keepdim=True) + 1e-5).rsqrt()
                return ((x - mean) * rstd).view(m, n)
            rstd = (x.square().mean(1, keepdim=True) + 1e-5).rsqrt()
            normalized = (x * rstd).view(m, n)
            if case == "rms_main":
                return normalized
            return acc.relu(), normalized

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "two-phase local-reduce source"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_supports_trailing_fp8_quant(self):
        # [FUNCTION DOC] SM100+ CUDA test: reciprocal-sum normalization over row groups of 32
        #   followed by a cast to float8 e4m3 as the final pointwise step. Output dtype must be
        #   fp8; matches the low-precision reference and a float64 reference; generated code has
        #   the physical feed-main plan, the float8 conversion and the feeds-main flag.
        m = 128
        n = 64
        group = 32

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue normalizing grouped values by the reciprocal
            #   row-group sum and casting to float8 e4m3.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n).to(torch.float8_e4m3fn)

        def high_precision_fn(acc):
            # [FUNCTION DOC] HELPER: float64 reference of the normalization without the final
            #   cast.
            x = acc.view(-1, group, n)
            return (x * x.sum(1, keepdim=True).reciprocal()).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            high_precision_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)
        FileCheck().check("cutlass.Float8E4M3FN").check("feeds_main=True").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_supports_mx_scale(self):
        # [FUNCTION DOC] SM100+ CUDA test: the row-group (32) scale is the MX E8M0 scale of the
        #   group maximum; grouped values are scaled by its reciprocal and cast to float8 e4m3.
        #   Output dtype is fp8; matches references; generated code has the PTX conversion and the
        #   feeds-main flag with the physical plan.
        m = 128
        n = 64
        group = 32

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue computing the MX scale of row-group maxima,
            #   scaling by its reciprocal and casting to float8 e4m3.
            x = acc.float().view(-1, group, n)
            scale = mx_e8m0_scale(x.amax(1, keepdim=True)).float()
            return (x * scale.reciprocal()).view(m, n).to(torch.float8_e4m3fn)

        def high_precision_fn(acc):
            # [FUNCTION DOC] HELPER: float64 reference computing the same scaled values without
            #   the final cast.
            x = acc.view(-1, group, n)
            scale = mx_e8m0_scale(x.amax(1, keepdim=True)).float()
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            high_precision_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)
        FileCheck().check("cvt.rp.satfinite.ue8m0x2.f32").check("feeds_main=True").run(
            code
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_supports_trailing_pointwise_chain(self):
        # [FUNCTION DOC] SM100+ CUDA test: reciprocal-sum normalization followed by a chain of
        #   trailing pointwise ops (times 0.5, plus one). Matches references; physical feed-main
        #   code check passes and generated code has the feeds-main flag.
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue normalizing grouped values by the reciprocal of
            #   the row-group sum, reshaping, then multiplying by 0.5 and adding one.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n) * 0.5 + 1.0

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)
        FileCheck().check("feeds_main=True").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_trailing_quant_with_scale_store(self):
        # [FUNCTION DOC] SM100+ CUDA test: fp8 quantized main output normalized by the row-group
        #   (32) sum, with the squeezed sum also returned as aux. Checks fp8 dtype, main against
        #   low-precision and float64 references, aux against the float64 group sum, and that
        #   generated code has the plan with output, feeds-main and callbacks.
        m = 128
        n = 64
        group = 32

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the float8 e4m3 normalized values and
            #   the squeezed row-group sum.
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            quant = (x * scale.reciprocal()).view(m, n).to(torch.float8_e4m3fn)
            return quant, scale.squeeze(1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        high_precision_acc = a.double() @ b.double()
        high_precision_x = high_precision_acc.view(-1, group, n)
        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b)[0],
            (
                high_precision_x * high_precision_x.sum(1, keepdim=True).reciprocal()
            ).view(m, n),
            a.shape[1],
        )
        torch.testing.assert_close(
            aux,
            high_precision_x.sum(1).float(),
            atol=1e-3,
            rtol=1e-3,
        )
        FileCheck().check("local_reduce=FlexGemmRuntimeLocalReducePlan").check(
            self.localReduceGeometryPattern(group, 0)
        ).check("out=").check("feeds_main=True").check(
            "callbacks=FlexGemmLocalReduceCallbacks"
        ).run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (4, 8, 16, 32))
    def test_mm_local_n_reduce_feed_main_fragment_group_sum(self, group):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by group size 4, 8, 16, 32: N-group sum
        #   normalization (values times reciprocal group sum) handled fragment-locally. Matches
        #   references; generated code has the add reduction and no runtime local-reduce plan.
        m = 128
        n = 128

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue scaling grouped values along N by the reciprocal
            #   of their group sum and reshaping to M by N.
            x = acc.float().view(m, -1, group)
            scale = x.sum(-1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        FileCheck().check("cute.ReductionOp.ADD").check_not(
            "local_reduce=FlexGemmRuntimeLocalReducePlan"
        ).run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (4, 8, 16, 32))
    def test_mm_local_n_reduce_feed_main_fragment_group_fp8_quant(self, group):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by group size: fragment-local per-group
        #   quantization (scale from the clamped absolute maximum divided by 448, values divided
        #   by the scale, cast to float8 e4m3). Output dtype fp8; matches references; generated
        #   code has the float8 conversion and no runtime local-reduce plan.
        m = 128
        n = 128

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue scaling grouped values by the reciprocal of the
            #   group scale (clamped maximum over 448), reshaping and casting to float8 e4m3.
            x = acc.float().view(m, -1, group)
            scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (x * scale.reciprocal()).view(m, n).to(torch.float8_e4m3fn)

        def high_precision_fn(acc):
            # [FUNCTION DOC] HELPER: float64 reference of the same scaling without the cast.
            x = acc.view(m, -1, group)
            scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            high_precision_fn(a.double() @ b.double()),
            a.shape[1],
        )
        FileCheck().check("cutlass.Float8E4M3FN").check_not(
            "local_reduce=FlexGemmRuntimeLocalReducePlan"
        ).run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (4, 8, 16, 32))
    def test_mm_local_n_reduce_feed_main_fragment_group_scale_store(self, group):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by group size: like fragment-local
        #   quantization but also returns the squeezed scale as a compressed aux output. Checks
        #   the main and aux against references and that generated code has a compressed plan
        #   without the feeds-main flag.
        m = 128
        n = 128

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the scaled grouped values and the
            #   squeezed group scale (clamped maximum over 448).
            x = acc.float().view(m, -1, group)
            scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (x * scale.reciprocal()).view(m, n), scale.squeeze(-1)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected, _ = epilogue_fn(a @ b)
        high_precision_expected, high_precision_aux = epilogue_fn(
            a.double() @ b.double()
        )
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            high_precision_expected,
            a.shape[1],
        )
        torch.testing.assert_close(
            aux,
            high_precision_aux.float(),
            atol=1e-3,
            rtol=1e-3,
        )
        self.assertLocalReduceAuxCode(code, group)
        FileCheck().check_not("feeds_main=True").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (16, 32))
    def test_mm_local_n_reduce_feed_main_fragment_group_tuned(self, group):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by group 16 or 32: fragment-local fp8
        #   quantization with the tuned option. Matches references; generated code has the float8
        #   conversion, no runtime plan, and only non-swap_ab candidate keys (swap_ab configs
        #   would regroup the wrong elements).
        m = 128
        n = 128

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue scaling grouped values by the reciprocal group
            #   scale and casting to float8 e4m3.
            x = acc.float().view(m, -1, group)
            scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (x * scale.reciprocal()).view(m, n).to(torch.float8_e4m3fn)

        def high_precision_fn(acc):
            # [FUNCTION DOC] HELPER: float64 reference of the same scaling without the cast.
            x = acc.view(m, -1, group)
            scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue and tuned on.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            high_precision_fn(a.double() @ b.double()),
            a.shape[1],
        )
        FileCheck().check("cutlass.Float8E4M3FN").check_not(
            "local_reduce=FlexGemmRuntimeLocalReducePlan"
        ).run(code)
        # Plan-less grouped fragments must gate tuned candidates like a plan:
        # swap_ab configs would regroup the wrong accumulator elements.
        self.assertIn("swap_ab', False", code)
        self.assertNotIn("swap_ab', True", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_n_reduce_feed_main_fragment_group_skinny_default_config(self):
        # [FUNCTION DOC] SM100+ CUDA test: N of 4096 (a skinny M, wide N shape) with
        #   fragment-local group-16 scaling and no tuning. The default config must still satisfy
        #   the grouped-fragment gate: matches references, no runtime plan in generated code, and
        #   only non-swap_ab keys appear.
        m = 128
        n = 4096
        group = 16

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue scaling grouped values by the reciprocal group
            #   scale (clamped maximum over 448) and reshaping.
            x = acc.float().view(m, -1, group)
            scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        FileCheck().check_not("local_reduce=FlexGemmRuntimeLocalReducePlan").run(code)
        # The skinny-shape default config must satisfy the same grouped-fragment
        # gate as tuned candidates even without a runtime local-reduce plan.
        self.assertIn("swap_ab', False", code)
        self.assertNotIn("swap_ab', True", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (64, 128))
    def test_mm_local_n_reduce_feed_main_rejects_multi_fragment_group(self, group):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by group 64 or 128 along N: a
        #   reciprocal-sum normalization whose axis-1 group exceeds one TensorSSA fragment must
        #   fail to compile with an error about axis-1 groups larger than one fragment. Resets
        #   dynamo first.
        torch._dynamo.reset()
        m = 128
        n = 256

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue scaling grouped values along N by the reciprocal
            #   group sum.
            x = acc.float().view(m, -1, group)
            scale = x.sum(-1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(
            Exception, "axis-1 groups larger than one TensorSSA fragment"
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("fragment_group_tuned", 128, 32, True),
            ("multi_chunk_large_group", 256, 128, False),
            ("tile_n_group", 256, 256, False),
            ("wide_m_tile_n_group", 512, 512, False),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_coda_rmsnorm_rewrite_e2e(self, case):
        # [FUNCTION DOC] SM100+ CUDA end-to-end test parametrized by (name, N, group, tuned):
        #   two chained flex_gemm calls implementing an RMS norm rewrite. The first returns a
        #   gamma-scaled bf16 result plus per-group mean of squares as compressed aux; the second
        #   multiplies its accumulator by the reciprocal root of the mean computed outside. Result
        #   compared with eager within 1e-2; generated code must contain two epilogue calls, the
        #   group geometry plan (with combine callback if the group exceeds 32), row and col arg
        #   kinds, and no legacy arguments.
        _, n, group, tuned = case
        m, k, p = 64, 32, 48
        eps = 1e-5

        def fn(a, b1, gamma, b2):
            # [FUNCTION DOC] HELPER: function under compilation: first flex_gemm computes the
            #   gamma-scaled bf16 output and the per-group mean of squares, then the reciprocal
            #   root of the mean is formed, then a second flex_gemm multiplies its accumulator by
            #   it.
            def first_epilogue(acc):
                # [FUNCTION DOC] HELPER: first epilogue returning the gamma-scaled accumulator
                #   cast to bf16 and the per-group mean of squares along N.
                x = acc.float().view(m, -1, group)
                h2 = (acc.float() * gamma).to(torch.bfloat16)
                partial_mean_square = x.square().mean(-1)
                return h2, partial_mean_square

            h2, partial_mean_square = flex_gemm(
                torch.mm,
                (a, b1),
                first_epilogue,
                kernel_options={"backend": "QUACK", "tuned": tuned},
            )
            rstd = (partial_mean_square.mean(-1, keepdim=True) + eps).rsqrt()

            def second_epilogue(acc):
                # [FUNCTION DOC] HELPER: second epilogue multiplying the float32 accumulator by
                #   the computed reciprocal-root scale.
                return acc.float() * rstd

            return flex_gemm(
                torch.mm,
                (h2, b2),
                second_epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = self.makeTensor(m, k)
        b1 = self.makeTensor(k, n)
        gamma = self.makeTensor(1, n)
        b2 = self.makeTensor(n, p)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b1, gamma, b2
        )

        acc1 = a @ b1
        h2 = (acc1.float() * gamma).to(torch.bfloat16)
        partial_mean_square = acc1.float().view(m, -1, group).square().mean(-1)
        rstd = (partial_mean_square.mean(-1, keepdim=True) + eps).rsqrt()
        expected = (h2 @ b2).float() * rstd

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        self.assertEqual(code.count("flex_gemm_epilogue("), 2)
        self.assertIn("local_reduce=FlexGemmRuntimeLocalReducePlan", code)
        self.assertIn(
            f"FlexGemmLocalReduceGeometry(group={group}, axis=1)",
            code,
        )
        if group > 32:
            self.assertIn("combine_fn=", code)
        self.assertIn("epilogue_arg_kinds=('row',)", code)
        self.assertIn("epilogue_arg_kinds=('col',)", code)
        self.assertNotIn("local_reduce_feeds_main=True", code)
        self.assertNotIn("local_reduce_op", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_coda_rmsnorm_rewrite_rejects_group_above_config_limit(self):
        # [FUNCTION DOC] SM100+ CUDA test: the RMS norm rewrite with a group of 1024 along N
        #   exceeds the maximum supported group of 512, so compile must fail with an error stating
        #   requested group 1024 versus max supported group 512 for axis 1.
        m, k, n, p = 64, 32, 1024, 48
        group = 1024
        eps = 1e-5

        def fn(a, b1, gamma, b2):
            # [FUNCTION DOC] HELPER: function under compilation chaining two flex_gemm calls as
            #   in the RMS norm rewrite test.
            def first_epilogue(acc):
                # [FUNCTION DOC] HELPER: first epilogue returning the gamma-scaled bf16
                #   accumulator and the per-group mean of squares.
                x = acc.float().view(m, -1, group)
                h2 = (acc.float() * gamma).to(torch.bfloat16)
                return h2, x.square().mean(-1)

            h2, partial_mean_square = flex_gemm(
                torch.mm,
                (a, b1),
                first_epilogue,
                kernel_options={"backend": "QUACK"},
            )
            rstd = (partial_mean_square.mean(-1, keepdim=True) + eps).rsqrt()

            def second_epilogue(acc):
                # [FUNCTION DOC] HELPER: second epilogue multiplying the float32 accumulator by
                #   the computed reciprocal-root scale.
                return acc.float() * rstd

            return flex_gemm(
                torch.mm,
                (h2, b2),
                second_epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = self.makeTensor(m, k)
        b1 = self.makeTensor(k, n)
        gamma = self.makeTensor(1, n)
        b2 = self.makeTensor(n, p)

        with self.assertRaisesRegex(
            Exception,
            "requested group=1024, max supported group=512 for axis=1",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b1, gamma, b2)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_supports_distinct_output_dtypes(self):
        # [FUNCTION DOC] SM100+ CUDA test: a tuple epilogue with different output dtypes (bf16
        #   ReLU and float32 squared plus two). The outputs keep distinct dtypes and each matches
        #   its reference.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU of the accumulator and its float32
            #   square plus two.
            return acc.relu(), acc.float().square() + 2.0

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        actual, aux = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        expected, expected_aux = epilogue_fn(a @ b)
        high_precision_acc = a.double() @ b.double()
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(aux.dtype, torch.float32)
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            high_precision_acc.relu(),
            a.shape[1],
        )
        self.assertMatchesLowPrecisionEager(
            aux,
            expected_aux,
            high_precision_acc.square() + 2.0,
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_supports_bool_mask_output(self):
        # [FUNCTION DOC] SM100+ CUDA test: a tuple epilogue returning ReLU and a boolean mask of
        #   positive accumulator values, compiled. The aux must be a bool tensor equal to the
        #   eager mask, the main output matches references, and generated code passes aux outputs
        #   (shared generated-code check).
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU of the accumulator and its
            #   positivity mask.
            return acc.relu(), acc > 0

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected, expected_aux = epilogue_fn(a @ b)
        self.assertEqual(aux.dtype, torch.bool)
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            (a.double() @ b.double()).relu(),
            a.shape[1],
        )
        torch.testing.assert_close(aux, expected_aux)
        self.assertFlexGemmGeneratedCode(code, "aux_outs=")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_imports_generated_dependencies(self):
        # [FUNCTION DOC] SM100+ CUDA test: compiles flex_gemm with an epilogue that uses
        #   absolute value and a conditional select (so generated code needs extra imports for
        #   those ops); result must match low-precision and float64 references.
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning the accumulator where its absolute value
            #   exceeds 0.1 and its negation elsewhere.
            abs_acc = torch.abs(acc)
            return torch.where(abs_acc > 0.1, acc, -acc)

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.mm,
            (a, b),
            epilogue_fn,
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_addmm_compiled_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: compiles flex_gemm over addmm with beta 0.5, alpha
        #   1.5 passed through the gemm keyword arguments and a ReLU epilogue; compares with eager
        #   and float64 addmm followed by ReLU.
        bias = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.addmm,
            (bias, a, b),
            epilogue_fn,
            gemm_kwargs={"beta": 0.5, "alpha": 1.5},
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.addmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.addmm(bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5)
            ),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_generated_code_calls_flex_gemm_adapter(self):
        # [FUNCTION DOC] SM100+ CUDA test: with tuning off, the compiled plus-one ReLU epilogue
        #   matches references, generated code passes the shared generated-code check and contains
        #   no output plan argument.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue adding one to the accumulator and applying ReLU.
            return (acc + 1).relu()

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue and tuned off.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": False},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(code)
        self.assertNotIn("output_plan=", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("tile", lambda m, n: (m, n)),
            ("row", lambda m, n: (1, n)),
            ("col", lambda m, n: (m, 1)),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_generated_code_reads_captured_tensor_epilogue_arg(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by captured scale layout (tile, row,
        #   col): the compiled epilogue multiplies by the captured float32 tensor and applies
        #   ReLU. Matches references; generated code passes the shared check with epilogue args
        #   and the expected single arg kind.
        kind, shape_fn = case

        def epilogue_fn(acc, scale):
            # [FUNCTION DOC] HELPER: epilogue multiplying the float32 accumulator by the scale
            #   and applying ReLU.
            return (acc.float() * scale).relu()

        def fn(a, b, scale):
            # [FUNCTION DOC] HELPER: function under compilation, wrapping the epilogue with the
            #   captured scale in a flex_gemm call on mm.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, scale),
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(*shape_fn(m, n), device="cuda", dtype=torch.float32)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, scale
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b, scale),
            epilogue_fn(a.double() @ b.double(), scale.double()),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(
            code, "epilogue_args=", f"epilogue_arg_kinds=('{kind}',)"
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_generated_code_reads_multiple_captured_tensor_epilogue_args(self):
        # [FUNCTION DOC] SM100+ CUDA test: three captured tensors (column bias, row scale, tile
        #   bias) in one affine plus ReLU epilogue. Matches references; generated code records
        #   epilogue args with kinds col, row, tile in that order.
        def fn(a, b, col_bias, row_scale, tile_bias):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with an
            #   inline epilogue applying column bias, row scale and tile bias, then ReLU; also
            #   used as the low-precision eager reference.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: ((acc.float() + col_bias) * row_scale + tile_bias).relu(),
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        col_bias = torch.randn(m, 1, device="cuda", dtype=torch.float32)
        row_scale = torch.randn(1, n, device="cuda", dtype=torch.float32)
        tile_bias = torch.randn(m, n, device="cuda", dtype=torch.float32)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            col_bias,
            row_scale,
            tile_bias,
        )

        low_precision_expected = fn(a, b, col_bias, row_scale, tile_bias)
        high_precision_expected = (
            ((a.double() @ b.double()) + col_bias.double()) * row_scale.double()
            + tile_bias.double()
        ).relu()
        self.assertMatchesLowPrecisionEager(
            actual, low_precision_expected, high_precision_expected, a.shape[1]
        )
        self.assertFlexGemmGeneratedCode(
            code,
            "epilogue_args=",
            "epilogue_arg_kinds=('col', 'row', 'tile')",
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_generated_code_reads_captured_args_and_writes_tuple_aux(self):
        # [FUNCTION DOC] SM100+ CUDA test: three captured tensors combined with a tuple epilogue
        #   (main and aux). Checks both against references with the captured-tuple helper;
        #   generated code records epilogue args, kinds col, row, tile, and aux outputs.
        def fn(a, b, col_bias, row_scale, tile_bias):
            # [FUNCTION DOC] HELPER: function under compilation, defining the tuple epilogue
            #   over captured column bias, row scale and tile bias and calling flex_gemm on mm.
            def epilogue_fn(acc):
                # [FUNCTION DOC] HELPER: epilogue returning the ReLU of the biased and scaled
                #   accumulator and a row-scaled accumulator plus tile bias.
                biased = (acc.float() + col_bias) * row_scale + tile_bias
                return biased.relu(), acc.float() * row_scale + tile_bias

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        col_bias = torch.randn(m, 1, device="cuda", dtype=torch.float32)
        row_scale = torch.randn(1, n, device="cuda", dtype=torch.float32)
        tile_bias = torch.randn(m, n, device="cuda", dtype=torch.float32)

        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            col_bias,
            row_scale,
            tile_bias,
        )

        self.assertCapturedTupleAuxMatchesReference(
            actual, aux, a, b, col_bias, row_scale, tile_bias
        )
        self.assertFlexGemmGeneratedCode(
            code,
            "epilogue_args=",
            "epilogue_arg_kinds=('col', 'row', 'tile')",
            "aux_outs=",
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_generated_code_tuned_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: tuned flex_gemm with candidate configs patched to the
        #   first two candidates; the plus-one ReLU result matches references and generated code
        #   passes the shared check.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue adding one to the accumulator and applying ReLU.
            return (acc + 1).relu()

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the epilogue and tuned on.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        from torch._inductor.template_heuristics import (
            flex_gemm as flex_gemm_heuristics,
        )

        configs = flex_gemm_heuristics.candidate_gemm_configs_for_device(a.device)[:2]
        with mock.patch(
            "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
            return_value=configs,
        ):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_generated_code_tuned_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: tuned flex_gemm with a tuple epilogue and candidate
        #   configs patched to the first two; both outputs match the tuple-aux reference and
        #   generated code passes aux outputs in the shared check.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning half of the accumulator plus one and the
            #   accumulator squared plus two in float32.
            main = (acc.float() + 1.0) * 0.5
            aux = acc.float().square() + 2.0
            return main, aux

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the tuple epilogue and tuned on.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        from torch._inductor.template_heuristics import (
            flex_gemm as flex_gemm_heuristics,
        )

        configs = flex_gemm_heuristics.candidate_gemm_configs_for_device(a.device)[:2]
        with mock.patch(
            "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
            return_value=configs,
        ):
            (actual, aux), (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        self.assertTupleAuxMatchesReference(actual, aux, a, b, epilogue_fn)
        self.assertFlexGemmGeneratedCode(code, "aux_outs=")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_bmm_compiled_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: compiles flex_gemm over bmm (batch 2) with a ReLU
        #   epilogue; the result is compared with eager and float64 bmm followed by ReLU.
        a = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.bmm,
            (a, b),
            epilogue_fn,
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.bmm(a, b)),
            epilogue_fn(torch.bmm(a.double(), b.double())),
            a.shape[-1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_baddbmm_compiled_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: compiles flex_gemm over baddbmm with a full-shape
        #   batch bias, beta 0.5 and alpha 1.5, plus a ReLU epilogue; compared with eager and
        #   float64 baddbmm.
        bias = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.baddbmm,
            (bias, a, b),
            epilogue_fn,
            gemm_kwargs={"beta": 0.5, "alpha": 1.5},
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.baddbmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.baddbmm(
                    bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5
                )
            ),
            a.shape[-1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_baddbmm_broadcast_bias_compiled_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: baddbmm with a two-dimensional bias broadcast over
        #   the batch (beta 0.5, alpha 1.5) and ReLU epilogue; compared with eager and float64
        #   references.
        bias = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.baddbmm,
            (bias, a, b),
            epilogue_fn,
            gemm_kwargs={"beta": 0.5, "alpha": 1.5},
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.baddbmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.baddbmm(
                    bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5
                )
            ),
            a.shape[-1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_baddbmm_matrix_dim_broadcast_bias_compiled_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: baddbmm where the bias broadcasts along a matrix
        #   dimension (a 1-D row of length N, a 1 by N row, or an M by 1 column), run in subtests
        #   with beta 0.5 and alpha 1.5 and a ReLU epilogue; each compared with eager and float64
        #   references.
        batch, m, n, k = 2, 128, 192, 64
        a = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(batch, k, n, device="cuda", dtype=torch.bfloat16)
        bias_cases = (
            ("row_1d", torch.randn(n, device="cuda", dtype=torch.bfloat16)),
            ("row_2d", torch.randn(1, n, device="cuda", dtype=torch.bfloat16)),
            ("col_2d", torch.randn(m, 1, device="cuda", dtype=torch.bfloat16)),
        )

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        for name, bias in bias_cases:
            with self.subTest(name=name):
                actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
                    torch.baddbmm,
                    (bias, a, b),
                    epilogue_fn,
                    gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                    kernel_options={"backend": "QUACK"},
                )

                self.assertMatchesLowPrecisionEager(
                    actual,
                    epilogue_fn(torch.baddbmm(bias, a, b, beta=0.5, alpha=1.5)),
                    epilogue_fn(
                        torch.baddbmm(
                            bias.double(),
                            a.double(),
                            b.double(),
                            beta=0.5,
                            alpha=1.5,
                        )
                    ),
                    k,
                )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_bmm_generated_code_calls_flex_gemm_adapter(self):
        # [FUNCTION DOC] SM100+ CUDA test: compiled bmm with batch 2 and ReLU; matches
        #   references and generated code passes the shared check including an expected rank of
        #   three.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on bmm with
            #   the epilogue.
            return flex_gemm(
                torch.bmm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.bmm(a, b)),
            epilogue_fn(torch.bmm(a.double(), b.double())),
            a.shape[-1],
        )
        self.assertFlexGemmGeneratedCode(code, "expected_ndim=3")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_bmm_batch_one_generated_code_calls_flex_gemm_adapter(self):
        # [FUNCTION DOC] SM100+ CUDA test: same as the bmm adapter test but with batch size one;
        #   generated code must still show an expected rank of three.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on bmm with
            #   the epilogue.
            return flex_gemm(
                torch.bmm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(1, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.bmm(a, b)),
            epilogue_fn(torch.bmm(a.double(), b.double())),
            a.shape[-1],
        )
        self.assertFlexGemmGeneratedCode(code, "expected_ndim=3")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_bmm_generated_code_tuned_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: tuned bmm with candidate configs patched to the first
        #   two; matches references and generated code shows an expected rank of three.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on bmm with
            #   the epilogue and tuned on.
            return flex_gemm(
                torch.bmm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        from torch._inductor.template_heuristics import (
            flex_gemm as flex_gemm_heuristics,
        )

        configs = flex_gemm_heuristics.candidate_gemm_configs_for_device(a.device)[:2]
        with mock.patch(
            "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
            return_value=configs,
        ):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.bmm(a, b)),
            epilogue_fn(torch.bmm(a.double(), b.double())),
            a.shape[-1],
        )
        self.assertFlexGemmGeneratedCode(code, "expected_ndim=3")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_baddbmm_generated_code_calls_flex_gemm_adapter(self):
        # [FUNCTION DOC] SM100+ CUDA test: compiled baddbmm with beta 0.5 and alpha 1.5; matches
        #   references and generated code passes C and an expected rank of three in the shared
        #   check.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        def fn(bias, a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on baddbmm
            #   with beta and alpha keyword arguments.
            return flex_gemm(
                torch.baddbmm,
                (bias, a, b),
                epilogue_fn,
                gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.baddbmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.baddbmm(
                    bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5
                )
            ),
            a.shape[-1],
        )
        self.assertFlexGemmGeneratedCode(code, "C=", "expected_ndim=3")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_addmm_generated_code_calls_flex_gemm_adapter(self):
        # [FUNCTION DOC] SM100+ CUDA test: compiled addmm with beta 0.5 and alpha 1.5 and ReLU;
        #   matches references and generated code passes a C argument.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        def fn(bias, a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on addmm with
            #   beta and alpha keyword arguments.
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                epilogue_fn,
                gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.addmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.addmm(bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5)
            ),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(code, "C=")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_addmm_generated_code_tuned_matches_reference(self):
        # [FUNCTION DOC] SM100+ CUDA test: tuned addmm with candidate configs patched to the
        #   first two; matches references and generated code passes a C argument.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying ReLU to the accumulator.
            return acc.relu()

        def fn(bias, a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on addmm with
            #   beta and alpha and tuned on.
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                epilogue_fn,
                gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        bias = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        from torch._inductor.template_heuristics import (
            flex_gemm as flex_gemm_heuristics,
        )

        configs = flex_gemm_heuristics.candidate_gemm_configs_for_device(a.device)[:2]
        with mock.patch(
            "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
            return_value=configs,
        ):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
            )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.addmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.addmm(bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5)
            ),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(code, "C=")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            (
                "unsupported_epilogue",
                lambda acc: acc.sum(dim=1, keepdim=True),
                {"backend": "QUACK"},
                "unsupported FlexGEMM epilogue",
            ),
            (
                "unknown_kernel_option",
                lambda acc: acc.relu(),
                {"backend": "QUACK", "split_k": 2},
                "unsupported FlexGEMM kernel options",
            ),
            (
                "invalid_fast_math_option",
                lambda acc: acc.relu(),
                {"backend": "QUACK", "fast_math": 1},
                "fast_math kernel option must be bool",
            ),
            (
                "invalid_config_option",
                lambda acc: acc.relu(),
                {"backend": "QUACK", "config": None},
                "config kernel option must be a dict",
            ),
            (
                "unknown_config_field",
                lambda acc: acc.relu(),
                {"backend": "QUACK", "config": {"stages": 4}},
                "unexpected GemmConfig fields",
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_code_rejects_unsupported_cases(self, case):
        # [FUNCTION DOC] SM100+ CUDA test parametrized by (name, epilogue, kernel options, error
        #   pattern): compile-time rejection of an unsupported epilogue (a keepdim row sum),
        #   unknown kernel option, non-bool fast_math, non-dict config, and an unknown config
        #   field. Each must raise an exception matching the case message.
        _, epilogue_fn, kernel_options, error = case

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the parametrized epilogue and kernel options.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options=kernel_options,
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        with self.assertRaisesRegex(Exception, error):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_rejects_unsupported_quack_op(self):
        # [FUNCTION DOC] flex_gemm on CPU tensors with a non-GEMM op (matmul) and the QUACK
        #   backend must raise RuntimeError about an unsupported GEMM op.
        a = torch.randn(8, 16)
        b = torch.randn(16, 12)

        with self.assertRaisesRegex(RuntimeError, "unsupported GEMM op"):
            flex_gemm(
                torch.ops.aten.matmul.default,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK"},
            )

    def test_rejects_unknown_backend(self):
        # [FUNCTION DOC] flex_gemm on CPU tensors with a backend named CUTLASS must raise
        #   RuntimeError about an unsupported FlexGEMM backend.
        a = torch.randn(8, 16)
        b = torch.randn(16, 12)

        with self.assertRaisesRegex(RuntimeError, "unsupported FlexGEMM backend"):
            flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "CUTLASS"},
            )


@skipIfNoCuteDSL
@unittest.skipIf(not SM100OrLater, "SM100+ required")
class TestFlexGemmNVFP4Device(FlexGemmTestCase):
    # [CLASS DOC] Device-generic tests (CUDA only, via device-type instantiation at module
    #   level) of NVFP4 packing in compiled flex_gemm. Skipped without the CuTe DSL or SM100.
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    @parametrize("tuned", (False, True))
    def test_mm_nvfp4_pack_matches_reference(self, device, tuned):
        # [FUNCTION DOC] Parametrized by tuned: A selects column zero so the accumulator equals
        #   a crafted B row of bf16 values at and around every FP4 rounding boundary (both signs,
        #   infinities and signed zeros). The epilogue packs the float32 accumulator pairs with
        #   nvfp4_pack; result dtype must be the float4 e2m1 pair type and its bytes equal the
        #   eager pack; generated code must show an inline-asm intrinsic and a dtype-view to the
        #   float4 type. Not run on SM120 or later.
        m = n = 128
        k = 64
        boundaries = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            device=device,
            dtype=torch.bfloat16,
        )
        lower = torch.nextafter(boundaries, torch.full_like(boundaries, float("-inf")))
        upper = torch.nextafter(boundaries, torch.full_like(boundaries, float("inf")))
        values = torch.cat(
            (
                -upper,
                -boundaries,
                -lower,
                torch.tensor(
                    [float("-inf"), -0.0, 0.0, float("inf")],
                    device=device,
                    dtype=torch.bfloat16,
                ),
                lower,
                boundaries,
                upper,
            )
        ).repeat(4)[:n]
        a = torch.zeros(m, k, device=device, dtype=torch.bfloat16)
        b = torch.zeros(k, n, device=device, dtype=torch.bfloat16)
        a[:, 0] = 1
        b[0] = values

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with an
            #   epilogue that packs accumulator pairs to NVFP4, with the tuned option.
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: nvfp4_pack(acc.float().view(m, -1, 2)),
                kernel_options={"backend": "QUACK", "tuned": tuned},
            )

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = nvfp4_pack((a @ b).float().view(m, -1, 2))
        self.assertIs(actual.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(actual.view(torch.uint8), expected.view(torch.uint8))
        FileCheck().check("inline_asm_elementwise_intrinsic").check(
            "torch.uint8"
        ).check("aten.view.dtype").check("torch.float4_e2m1fn_x2").run(code)


instantiate_device_type_tests(TestFlexGemmNVFP4Device, globals(), only_for="cuda")


@skipIfNoCuteDSL
@unittest.skipIf(not SM100OrLater, "SM100+ required")
class TestFlexGemmFastMathDevice(FlexGemmTestCase):
    # [CLASS DOC] Device-generic tests (CUDA only) of the fast_math kernel option and its effect
    #   on generated CuTe math calls for the epilogue. Skipped without the CuTe DSL or SM100.
    def test_mm_fast_math_kernel_option_controls_cutedsl_math(self, device):
        # [FUNCTION DOC] Compiles a transcendental-heavy epilogue (tanh, sigmoid, exponential,
        #   log1p, square root, reverse square root) with fast_math on and off, resetting dynamo
        #   each time. Fast results must be within 0.2 absolute of a float64 reference, precise
        #   results within 0.02. Fast code must use the CuTe math helpers for the listed ops and
        #   the fastmath flag; precise code must not carry the flag.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue summing tanh, sigmoid, exp of negative absolute
            #   value, log1p of absolute value, square root and reverse square root of the
            #   absolute value plus one.
            magnitude = acc.abs() + 1.0
            return (
                torch.tanh(acc)
                + torch.sigmoid(acc)
                + torch.exp(-acc.abs())
                + torch.log1p(acc.abs())
                + torch.sqrt(magnitude)
                + torch.rsqrt(magnitude)
            )

        def compile_with_fast_math(a, b, fast_math):
            # [FUNCTION DOC] HELPER: resets dynamo and compiles a flex_gemm on mm with the
            #   epilogue and the given fast_math option; returns the output and generated code.
            def fn(a, b):
                # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm
                #   with the fast_math kernel option.
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK", "fast_math": fast_math},
                )

            torch._dynamo.reset()
            return run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        a = self.makeTensor(128, 64, device=device)
        b = self.makeTensor(64, 128, device=device)
        fast_result, (fast_code,) = compile_with_fast_math(a, b, True)
        precise_result, (precise_code,) = compile_with_fast_math(a, b, False)
        expected = epilogue_fn(a.double() @ b.double())

        torch.testing.assert_close(fast_result.double(), expected, atol=0.2, rtol=0.02)
        torch.testing.assert_close(
            precise_result.double(), expected, atol=0.02, rtol=0.002
        )
        for op_name in ("tanh", "exp2", "log", "sqrt", "rsqrt"):
            self.assertIn(f"cute.math.{op_name}", fast_code)
        self.assertIn("fastmath=True", fast_code)
        self.assertNotIn("fastmath=True", precise_code)

    def test_mm_fast_math_sigmoid_uses_tanh_decomposition(self, device):
        # [FUNCTION DOC] Compiles a sigmoid epilogue with fast_math on and off (resetting dynamo
        #   each time). Fast result within 0.2 absolute and precise within 0.02 of float64; fast
        #   code must use the CuTe tanh helper and fastmath flag with no exp2, while precise code
        #   uses exp2 and no tanh (sigmoid is decomposed through tanh in fast mode).
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying sigmoid to the accumulator.
            return torch.sigmoid(acc)

        def compile_with_fast_math(a, b, fast_math):
            # [FUNCTION DOC] HELPER: resets dynamo and compiles a flex_gemm on mm with the
            #   epilogue and the given fast_math option; returns the output and generated code.
            def fn(a, b):
                # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm
                #   with the fast_math kernel option.
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK", "fast_math": fast_math},
                )

            torch._dynamo.reset()
            return run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        a = self.makeTensor(128, 64, device=device)
        b = self.makeTensor(64, 128, device=device)
        fast_result, (fast_code,) = compile_with_fast_math(a, b, True)
        precise_result, (precise_code,) = compile_with_fast_math(a, b, False)
        expected = epilogue_fn(a.double() @ b.double())

        torch.testing.assert_close(fast_result.double(), expected, atol=0.2, rtol=0.02)
        torch.testing.assert_close(
            precise_result.double(), expected, atol=0.02, rtol=0.002
        )
        self.assertIn("cute.math.tanh", fast_code)
        self.assertIn("fastmath=True", fast_code)
        self.assertNotIn("cute.math.exp2", fast_code)
        self.assertIn("cute.math.exp2", precise_code)
        self.assertNotIn("cute.math.tanh", precise_code)

    def test_mm_fast_math_silu_uses_tanh_decomposition(self, device):
        # [FUNCTION DOC] Same structure as the sigmoid fast-math test but for SiLU: fast and
        #   precise compiled results are compared with float64 and the generated code is checked
        #   for the tanh-based decomposition under fast math versus the exponential-based one
        #   otherwise.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying SiLU to the accumulator.
            return torch.nn.functional.silu(acc)

        def compile_with_fast_math(a, b, fast_math):
            # [FUNCTION DOC] HELPER: resets dynamo and compiles a flex_gemm on mm with the
            #   epilogue and the given fast_math option; returns the output and generated code.
            def fn(a, b):
                # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm
                #   with the fast_math kernel option.
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK", "fast_math": fast_math},
                )

            torch._dynamo.reset()
            return run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        a = self.makeTensor(128, 64, device=device)
        b = self.makeTensor(64, 128, device=device)
        fast_result, (fast_code,) = compile_with_fast_math(a, b, True)
        precise_result, (precise_code,) = compile_with_fast_math(a, b, False)
        expected = epilogue_fn(a.double() @ b.double())

        torch.testing.assert_close(fast_result.double(), expected, atol=0.2, rtol=0.02)
        torch.testing.assert_close(
            precise_result.double(), expected, atol=0.02, rtol=0.002
        )
        self.assertIn("cute.math.tanh", fast_code)
        self.assertIn("fastmath=True", fast_code)
        self.assertNotIn("cute.math.exp2", fast_code)
        self.assertIn("cute.math.exp2", precise_code)
        self.assertNotIn("cute.math.tanh", precise_code)

    def test_mm_fast_math_gelu_none_uses_tanh_decomposition(self, device):
        # [FUNCTION DOC] Exact GELU epilogue compiled with fast_math on and off; both results
        #   within 0.02 absolute of float64. Fast code uses the CuTe tanh helper and fastmath flag
        #   without erf; precise code uses erf and no tanh.
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying exact (non-approximate) GELU to the
            #   accumulator.
            return torch.nn.functional.gelu(acc, approximate="none")

        def compile_with_fast_math(a, b, fast_math):
            # [FUNCTION DOC] HELPER: resets dynamo and compiles a flex_gemm on mm with the
            #   epilogue and the given fast_math option; returns the output and generated code.
            def fn(a, b):
                # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm
                #   with the fast_math kernel option.
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK", "fast_math": fast_math},
                )

            torch._dynamo.reset()
            return run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        a = self.makeTensor(128, 64, device=device)
        b = self.makeTensor(64, 128, device=device)
        fast_result, (fast_code,) = compile_with_fast_math(a, b, True)
        precise_result, (precise_code,) = compile_with_fast_math(a, b, False)
        expected = epilogue_fn(a.double() @ b.double())

        torch.testing.assert_close(
            fast_result.double(), expected, atol=0.02, rtol=0.002
        )
        torch.testing.assert_close(
            precise_result.double(), expected, atol=0.02, rtol=0.002
        )
        self.assertIn("cute.math.tanh", fast_code)
        self.assertIn("fastmath=True", fast_code)
        self.assertNotIn("cute.math.erf", fast_code)
        self.assertIn("cute.math.erf", precise_code)
        self.assertNotIn("cute.math.tanh", precise_code)

    def test_mm_fast_math_gelu_tanh_preserves_requested_approximation(self, device):
        # [FUNCTION DOC] Tanh-approximate GELU epilogue compiled with fast_math on: result
        #   within 0.02 of the float64 exact reference; code uses the CuTe tanh helper and
        #   fastmath flag and no erf (the requested approximation is preserved).
        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue applying tanh-approximate GELU to the accumulator.
            return torch.nn.functional.gelu(acc, approximate="tanh")

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the fast_math kernel option on.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "fast_math": True},
            )

        a = self.makeTensor(128, 64, device=device)
        b = self.makeTensor(64, 128, device=device)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = epilogue_fn(a.double() @ b.double())

        torch.testing.assert_close(actual.double(), expected, atol=0.02, rtol=0.002)
        self.assertIn("cute.math.tanh", code)
        self.assertIn("fastmath=True", code)
        self.assertNotIn("cute.math.erf", code)


instantiate_device_type_tests(TestFlexGemmFastMathDevice, globals(), only_for="cuda")


@skipIfNoCuteDSL
@unittest.skipIf(not SM100OrLater, "SM100+ required")
class TestFlexGemmExplicitConfigDevice(FlexGemmTestCase):
    # [CLASS DOC] Device-generic tests (CUDA only) of explicit and partial config kernel options
    #   and expanded local-reduce config coverage. Skipped without the CuTe DSL or SM100.
    def test_mm_explicit_config_matches_reference(self, device):
        # [FUNCTION DOC] Takes the first candidate config with swap_ab for the device and passes
        #   its full key as the explicit config kernel option; compiled plus-one ReLU matches
        #   references, generated code passes the shared check and embeds the exact config key.
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue adding one to the accumulator and applying ReLU.
            return (acc + 1).relu()

        config = next(
            config
            for config in candidate_gemm_configs_for_device(device)
            if config.swap_ab
        )
        config_key = gemm_config_key(config)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the explicit config kernel option.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": dict(config_key)},
            )

        a = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(code)
        self.assertIn(f"config_key={config_key!r}", code)

    @parametrize("tuned", (False, True))
    def test_mm_partial_config_matches_reference(self, device, tuned):
        # [FUNCTION DOC] Parametrized by tuned: passes only tile M, tile N, cluster M, cluster N
        #   and swap_ab from a swap_ab candidate as a partial config. Result matches references,
        #   generated code passes the shared check, and every pinned item appears in the generated
        #   code.
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
        )

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue adding one to the accumulator and applying ReLU.
            return (acc + 1).relu()

        config = next(
            config
            for config in candidate_gemm_configs_for_device(device)
            if config.swap_ab
        )
        pinned = {
            name: getattr(config, name)
            for name in ("tile_m", "tile_n", "cluster_m", "cluster_n", "swap_ab")
        }

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the pinned partial config and tuned option.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={
                    "backend": "QUACK",
                    "config": pinned,
                    "tuned": tuned,
                },
            )

        a = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(code)
        for item in pinned.items():
            self.assertIn(repr(item), code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    @parametrize("group", (64, 128))
    def test_mm_tuple_aux_local_n_reduce_supports_clustered_tile_m256(
        self, device, group
    ):
        # [FUNCTION DOC] Parametrized by group 64 or 128 (skipped on SM120 or later): explicit
        #   config with tile 256 by 256 and cluster M 2 on a 256 by 256 problem; N-group sum aux.
        #   Main and aux match references, generated code has the callback plan and the exact
        #   config key.
        from torch._inductor.heuristics.template.flex_gemm import gemm_config_key
        from torch._vendor.quack.gemm_config import GemmConfig

        m = n = 256

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the sum over N groups of the
            #   parametrized size.
            x = acc.float().view(m, -1, group)
            return acc.relu(), x.sum(-1)

        expected_config = GemmConfig(
            tile_m=256,
            tile_n=256,
            pingpong=False,
            cluster_m=2,
            cluster_n=1,
            device_capacity=10,
        )
        config_key = gemm_config_key(expected_config)
        config = dict(config_key)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the explicit config option.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": config},
            )

        a = torch.randn(m, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, n, device=device, dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)
        self.assertIn(f"config_key={config_key!r}", code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    def test_mm_tuned_local_reduce_supports_max_autotune(self, device):
        # [FUNCTION DOC] With max_autotune enabled, tuned flex_gemm with a group-512 sum aux
        #   output (reduced afterwards in the outer function) must keep mutable local-reduce
        #   outputs out of deferred template selection: main and reduced results match references
        #   and generated code carries tile M 256 and tile N 512 keys.
        """Keep mutable local-reduce outputs out of deferred template selection."""
        from torch._inductor import config as inductor_config

        m, n, group = 256, 512, 512

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the sum over N groups of 512.
            partials = acc.float().view(m, -1, group).sum(-1)
            return acc.relu(), partials

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   tuned on and then summing the partials over the last dim.
            actual, partials = flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )
            return actual, partials.sum(-1)

        a = torch.randn(m, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, n, device=device, dtype=torch.bfloat16)
        with inductor_config.patch(max_autotune=True):
            (actual, reduced), (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        expected, expected_partials = epilogue_fn(a @ b)
        high_precision_expected, high_precision_partials = epilogue_fn(
            a.double() @ b.double()
        )
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            high_precision_expected,
            a.shape[1],
        )
        torch.testing.assert_close(
            reduced,
            high_precision_partials.float().sum(-1),
            atol=1e-3,
            rtol=1e-3,
        )
        self.assertFlexGemmGeneratedCode(code)
        self.assertIn("('tile_m', 256)", code)
        self.assertIn("('tile_n', 512)", code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    def test_mm_full_tile_local_reduce_checks_actual_n_warp_layout(self, device):
        # [FUNCTION DOC] Forces the host-side local-reduce config check to approve (mock) a
        #   full-N group of 256 with a config whose layout splits N across warps; compile must
        #   fail with an error saying a full-N grouped local reduce requires one epilogue N-warp
        #   partition (the layout is checked again).
        """Reject a host-approved full-N group when the kernel layout splits N."""
        from torch._vendor.quack.gemm_config import GemmConfig

        m = n = group = 256

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the full-row sum over N groups
            #   of 256.
            partials = acc.float().view(m, -1, group).sum(-1)
            return acc.relu(), partials

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=128,
                tile_n=256,
                pingpong=False,
                is_dynamic_persistent=True,
                cluster_m=2,
                cluster_n=1,
                device_capacity=10,
            )
        )

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the explicit config option.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": config},
            )

        a = torch.randn(m, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, n, device=device, dtype=torch.bfloat16)
        with (
            mock.patch(
                "torch._inductor.kernel.flex_gemm.lowering."
                "validate_flex_gemm_local_reduce_config",
                return_value=True,
            ),
            self.assertRaisesRegex(
                Exception,
                "full-N GroupedLocalReduce requires one epilogue N-warp partition",
            ),
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    @parametrize(
        "case",
        (
            ("local_n_g32_tile64", 1, 32, 128, 64, 2, 2),
            ("local_n_g16_tile160", 1, 16, 128, 160, 2, 2),
            ("local_n_g32_tile192", 1, 32, 128, 192, 2, 1),
            ("local_n_g32_tile224", 1, 32, 256, 224, 2, 2),
            ("local_n_g32_tile256", 1, 32, 128, 256, 2, 2),
            ("local_n_g64_tile_m128", 1, 64, 128, 256, 2, 1),
            ("local_n_g128_tile_m128", 1, 128, 128, 256, 2, 1),
            ("local_n_full_tile256_tile_m256", 1, 256, 256, 256, 2, 1),
            ("local_n_full_tile512_tile_m256", 1, 512, 256, 512, 2, 1),
            ("local_m_g128_tile160", 0, 128, 128, 160, 1, 1),
            ("local_m_g64_tile_m256", 0, 64, 256, 256, 2, 1),
            ("local_m_g128_tile_m256", 0, 128, 256, 256, 2, 1),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_reduce_supports_expanded_configs(self, device, case):
        # [FUNCTION DOC] Parametrized by cases (name, axis, group, tile M, tile N, cluster M,
        #   cluster N) covering expanded local-reduce configs on axis 1 and axis 0 including
        #   full-tile groups up to 512. Builds the explicit config, sizes the problem to fit,
        #   compiles an epilogue returning ReLU and a group sum, checks references, and checks
        #   generated code (callbacks when group exceeds 32 on axis 1; geometry and callbacks for
        #   axis 0) plus the exact config key.
        from torch._inductor.heuristics.template.flex_gemm import gemm_config_key
        from torch._vendor.quack.gemm_config import GemmConfig

        _, axis, group, tile_m, tile_n, cluster_m, cluster_n = case
        m = max(tile_m, group, 256)
        n = max(tile_n, group if axis == 1 else 256)

        def epilogue_fn(acc):
            # [FUNCTION DOC] HELPER: epilogue returning ReLU and the group sum along N (axis 1)
            #   or along rows (axis 0).
            if axis == 1:
                partial = acc.float().view(m, -1, group).sum(-1)
            else:
                partial = acc.float().view(-1, group, n).sum(1)
            return acc.relu(), partial

        expected_config = GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            pingpong=False,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            device_capacity=10,
        )
        config_key = gemm_config_key(expected_config)
        config = dict(config_key)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compilation, calling flex_gemm on mm with
            #   the explicit config option.
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": config},
            )

        a = torch.randn(m, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, n, device=device, dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        if axis == 1:
            self.assertLocalReduceAuxCode(code, group, callbacks=group > 32)
        else:
            self.assertIn(f"FlexGemmLocalReduceGeometry(group={group}, axis=0)", code)
            self.assertIn("FlexGemmLocalReduceCallbacks(", code)
        self.assertIn(f"config_key={config_key!r}", code)


instantiate_device_type_tests(
    TestFlexGemmExplicitConfigDevice, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
