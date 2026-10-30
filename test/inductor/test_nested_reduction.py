# Owner(s): ["module: inductor"]
# [MODULE DOC] End-to-end tests for inductor's nested reduction fusion, which fuses a row-wise
# reduction (norm statistics) with a dependent reduction over small groups along the same data
# (block amax, weighted sums) into a single triton kernel. The first part holds behavior tests
# (numerics, fusion counts, rejected patterns, dynamic shapes) run with persistent and non-
# persistent outer reductions; the second part captures generated wrapper and kernel source and
# asserts kernel form (loads, stores, metadata, block-size contracts). Defines norm helpers, a
# choices handler factory, mixin test bases and four concrete test classes. The internals classes
# are skipped without a GPU, and the main block only runs the tests with a GPU.

"""End-to-end nested-reduction behavior and kernel-form tests."""

import re

import torch
import torch._inductor.config as inductor_config
from torch._inductor import metrics
from torch._inductor.choices import InductorChoices
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    get_func_call,
    get_kernel_launch,
    GPU_TYPE,
    HAS_GPU,
)


def _choices_context(force_persistent: bool | None):
    # [FUNCTION DOC] HELPER: Returns a context manager. For None it is a do-nothing context.
    # Otherwise it installs an inductor choices handler that always declines cooperative
    # reductions and answers the persistent-reduction question with the given flag; the returned
    # context is entered by the caller to apply it.
    import contextlib

    if force_persistent is None:
        return contextlib.nullcontext()

    class _Choices(InductorChoices):
        # [CLASS DOC] HELPER: Inductor choices subclass with two static overrides: cooperative
        # reduction is never used and persistent reduction is forced to the enclosing flag value.
        @staticmethod
        def should_use_cooperative_reduction(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Ignores its arguments and always returns false.
            return False

        @staticmethod
        def should_use_persistent_reduction(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Ignores its arguments and returns the
            # enclosing force_persistent flag.
            return force_persistent

    return V.set_choices_handler(_Choices())


class TestBase(TestCase):
    # [CLASS DOC] Base test case. Class attribute force_persistent_outer_reduction (default None)
    # selects whether outer reductions are forced persistent, forced looped, or left to inductor.
    # Every test runs with nested reduction and loop ordering after fusion enabled via config; the
    # two context managers are stored on the instance and exited in tearDown. Provides numeric and
    # fusion-count assertion helpers based on inductor metrics.
    force_persistent_outer_reduction: bool | None = None

    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp, resets inductor metrics and dynamo compilation
        # metrics, then enters the config patch enabling nested reduction and loop ordering after
        # fusion and the choices context, keeping both on the instance.
        super().setUp()
        metrics.reset()
        torch._dynamo.utils.clear_compilation_metrics()
        self._nested_reduction_ctx = inductor_config.patch(
            {
                "triton.nested_reduction": True,
                "loop_ordering_after_fusion": True,
            }
        )
        self._nested_reduction_ctx.__enter__()
        self._choices_ctx = _choices_context(self.force_persistent_outer_reduction)
        self._choices_ctx.__enter__()

    def tearDown(self):
        # [FUNCTION DOC] Exits the choices context and then the config patch in reverse order,
        # then calls the parent tearDown.
        self._choices_ctx.__exit__(None, None, None)
        self._nested_reduction_ctx.__exit__(None, None, None)
        super().tearDown()

    def check_numeric(self, f, args, tol=1e-2):
        # [FUNCTION DOC] HELPER: Runs f eagerly and compiled on the args and asserts the results
        # are close with the given tolerance used as both absolute and relative.
        ref = f(*args)
        act = torch.compile(f)(*args)
        self.assertEqual(act, ref, atol=tol, rtol=tol)

    def check_nested_matches_unnested(self, f, args, tol=1e-2):
        # [FUNCTION DOC] HELPER: Compiles f with nested reduction disabled to get a
        # reference, resets metrics and dynamo, compiles again with the test settings and
        # asserts the two results are close with the given tolerance. Used where eager would
        # differ in precision handling.
        with inductor_config.patch("triton.nested_reduction", False):
            ref = torch.compile(f)(*args)

        metrics.reset()
        torch._dynamo.reset()
        act = torch.compile(f)(*args)
        self.assertEqual(act, ref, atol=tol, rtol=tol)

    def check_fusion(self, expected_kernels=1):
        # [FUNCTION DOC] HELPER: Asserts exactly one nested reduction was code
        # generated and, unless expected_kernels is None, that the generated kernel
        # count equals expected_kernels.
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        if expected_kernels is not None:
            self.assertEqual(metrics.generated_kernel_count, expected_kernels)

    def check_no_fusion(self):
        # [FUNCTION DOC] HELPER: Asserts that no nested reduction was code generated.
        self.assertEqual(metrics.codegen_nested_reduction, 0)


def _rmsnorm(x_flat):
    # [FUNCTION DOC] HELPER: Root-mean-square normalization over the last dim with a small epsilon
    # inside the square root.
    return x_flat / torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)


def _layernorm(x_flat):
    # [FUNCTION DOC] HELPER: Layer normalization over the last dim without affine parameters,
    # using population variance (no correction) and a small epsilon.
    mean = x_flat.mean(dim=-1, keepdim=True)
    var = x_flat.var(dim=-1, keepdim=True, correction=0)
    return (x_flat - mean) / torch.sqrt(var + 1e-6)


def _swizzle_scale(scale):
    # [FUNCTION DOC] HELPER: Rearranges a 2D scale matrix into a blocked swizzled layout: rows in
    # groups of 128 and columns in groups of four are permuted and re-flattened, returning a
    # matrix of the original shape.
    rows, cols = scale.shape
    blocks = scale.view(rows // 128, 128, cols // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(rows, cols)


def _rmsnorm_block_scale_swizzle(x, weight, G):
    # [FUNCTION DOC] HELPER: Applies rms norm with weight, splits features into groups of size G,
    # computes the per-group max absolute value, derives a scale (that value over 448, floored at
    # a tiny number), divides, casts the payload to half and returns the payload as float with
    # shape batch by features plus the swizzled scale.
    import torch.nn.functional as F

    B, D = x.shape
    x = F.rms_norm(x, (D,), weight)
    x_groups = x.view(B, D // G, G)
    amax = x_groups.abs().amax(dim=-1)
    scale = (amax / 448.0).clamp(min=1e-12)
    payload = (x_groups / scale.unsqueeze(-1)).to(torch.float16)
    return payload.view(B, D).float(), _swizzle_scale(scale)


def _rmsnorm_mxfp8_scale_swizzle(x, weight, G):
    # [FUNCTION DOC] HELPER: Like the block scale variant but computes group amax in float32,
    # converts the scale to an 8-bit exponent form with an inductor primitive, rebuilds a float
    # scale from it, quantizes to float8 (clamped to plus or minus 448) and returns the payload
    # plus the swizzled exponent scale.
    import torch.nn.functional as F
    from torch._inductor import inductor_prims

    B, D = x.shape
    x = F.rms_norm(x, (D,), weight)
    x_groups = x.view(B, D // G, G)
    amax = x_groups.abs().float().amax(dim=-1)
    scale = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    scale_u8 = inductor_prims.cvt_e8m0_rceil(scale)
    scale_f32 = torch.ldexp(
        torch.ones_like(scale, dtype=torch.float32),
        scale_u8.to(torch.int32) - 127,
    )
    payload = (
        (x_groups.float() / scale_f32.unsqueeze(-1))
        .clamp(min=-448.0, max=448.0)
        .to(torch.float8_e4m3fn)
    )
    return payload.view(B, D), _swizzle_scale(scale_u8)


@instantiate_parametrized_tests
class _NestedReductionBase:
    # [CLASS DOC] HELPER: Mixin (not a test case itself) holding the behavior tests, parametrized
    # and instantiated for concrete subclasses combined with the test base. Helpers build the
    # weighted-reduction and block-reduction scenarios; tests cover small dims along X or R,
    # epilogue and prologue pointwise fusion, dynamic shapes, producer-consumer cases and patterns
    # that must be rejected.
    """Tests for fusing dependent cross-axis reductions into a single kernel."""

    # ---- Small dim in X: norm + weighted reduce ----

    def _weighted_norm_reduce_k(self, norm, reduce_fn, B, K, D):
        # [FUNCTION DOC] HELPER: Builds a normalized-then-weighted reduction scenario with a small
        # dim in X: normalizes (B*K, D) rows with the given norm, multiplies by per-(B,K) weights
        # and reduces over the K axis with the chosen reduction (sum, amax, amin or prod). Asserts
        # numerics against eager and that nested reduction fused once.
        rfn = {
            "sum": torch.Tensor.sum,
            "amax": torch.Tensor.amax,
            "amin": torch.Tensor.amin,
            "prod": torch.Tensor.prod,
        }[reduce_fn]

        def f(x, w):
            # [FUNCTION DOC] HELPER: Reshapes x to rows, applies the norm, reshapes back and
            # reduces the weighted result over dim 1 with the selected reduction.
            x_normed = norm(x.reshape(x.shape[0] * K, D)).reshape(x.shape)
            return rfn(w[:, :, None] * x_normed, dim=1)

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    @parametrize("B", [32, 256])
    @parametrize("K", [16, 32])
    def test_rmsnorm_weighted_sum(self, B, K):
        # [FUNCTION DOC] Weighted rms norm sum for the parametrized batch and K sizes (D 4096).
        self._weighted_norm_reduce_k(_rmsnorm, "sum", B, K, 4096)

    @parametrize("K", [16, 32])
    def test_rmsnorm_weighted_max(self, K):
        # [FUNCTION DOC] Weighted rms norm max reduction over K with batch 64 and D 4096.
        self._weighted_norm_reduce_k(_rmsnorm, "amax", 64, K, 4096)

    @parametrize("reduce_fn", ["sum", "amax", "amin"])
    def test_rmsnorm_weighted_reduce_B1(self, reduce_fn):
        # [FUNCTION DOC] Weighted rms norm with batch one for sum, amax and amin reductions; the
        # flattened small dim in X must still fuse.
        """B=1 flattened small_dim_in_x still fuses."""
        self._weighted_norm_reduce_k(_rmsnorm, reduce_fn, 1, 16, 1024)

    def test_layernorm_weighted_sum(self):
        # [FUNCTION DOC] Weighted layer norm sum, batch 64, K 16, D 4096.
        self._weighted_norm_reduce_k(_layernorm, "sum", 64, 16, 4096)

    def test_layernorm_weighted_sum_B1(self):
        # [FUNCTION DOC] Weighted layer norm sum with batch one, K 16, D 1024.
        self._weighted_norm_reduce_k(_layernorm, "sum", 1, 16, 1024)

    def test_fullres_prologue_small_dim_in_x_loop_order(self):
        # [FUNCTION DOC] Checks fusion and numerics when a full-resolution prologue (relu of
        # normalized plus a bias, forced to be realized) is also an output alongside the
        # weighted sum, so the prologue indexing must be remapped from physical rows to logical
        # batch, K, D.
        """Remap full-res prologue from physical [B*K, D] to logical [B, K, D]."""

        B, K, D = 16, 16, 1024

        def f(x, w, bias):
            # [FUNCTION DOC] HELPER: Rms normalizes the rows, adds a per-batch bias, applies relu,
            # forces realization of that tensor and returns it with its weighted sum over K.
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
            y = torch.ops._inductor_test.realize(
                torch.relu((x_flat / rms).reshape(B, K, D) + bias[:, None, :])
            )
            return y, (w[:, :, None] * y).sum(dim=1)

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        bias = torch.randn(B, D, device=GPU_TYPE)
        self.check_numeric(f, (x, w, bias))
        self.check_fusion()

    # ---- Small dim in R: norm + block reduce ----

    def _norm_block_reduce(self, norm, reduce_fn, B, D, G):
        # [FUNCTION DOC] HELPER: Builds a scenario with a small dim in R: normalizes a batch by D
        # input, groups the features in blocks of G and reduces each block with the chosen
        # reduction (amax uses the absolute value). Checks numerics and a single nested fusion.
        rfn = {
            "sum": torch.Tensor.sum,
            "amax": torch.Tensor.amax,
            "amin": torch.Tensor.amin,
            "prod": torch.Tensor.prod,
        }[reduce_fn]

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the norm, groups the last dim into blocks of G and
            # reduces each group; the amax case takes the max of absolute values.
            x_normed = norm(x)
            grouped = x_normed.reshape(x.shape[0], x.shape[1] // G, G)
            if reduce_fn == "amax":
                # Block scale amax is max(abs(x)); min/max tests cover signed variants.
                return grouped.abs().amax(dim=-1)
            return rfn(grouped, dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_fusion()

    @parametrize(
        "B,D,G",
        [
            (32, 4096, 16),
            (256, 4096, 32),
            (4, 384, 128),
        ],
    )
    def test_layernorm_block_amax(self, B, D, G):
        # [FUNCTION DOC] Layer norm followed by block amax for several (batch, D, group) triples.
        self._norm_block_reduce(_layernorm, "amax", B, D, G)

    @parametrize("G", [8, 16])
    def test_rmsnorm_block_amax(self, G):
        # [FUNCTION DOC] Rms norm followed by block amax with parametrized group size, batch
        # 128, D 8192.
        self._norm_block_reduce(_rmsnorm, "amax", 128, 8192, G)

    @parametrize("reduce_fn", ["sum", "amin"])
    def test_layernorm_block_reduce(self, reduce_fn):
        # [FUNCTION DOC] Layer norm followed by block sum or amin, batch 64, D 4096, group 16.
        self._norm_block_reduce(_layernorm, reduce_fn, 64, 4096, 16)

    def test_layernorm_block_prod(self):
        # [FUNCTION DOC] Layer norm, tanh and block product; compares the compiled GPU result with
        # the eager CPU result at tolerance 1e-3 and checks a nested fusion.
        B, D, G = 64, 4096, 8

        def f(x):
            # [FUNCTION DOC] HELPER: Tanh of the layer norm, grouped by G, product
            # over each group.
            x_normed = torch.tanh(_layernorm(x))
            return x_normed.reshape(B, D // G, G).prod(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        ref = f(x.cpu()).to(GPU_TYPE)
        act = torch.compile(f)(x)
        self.assertEqual(act, ref, atol=1e-3, rtol=1e-3)
        self.check_fusion()

    def test_layernorm_block_amax_group_size_512(self):
        # [FUNCTION DOC] Layer norm and block amax with group size 512.
        self._norm_block_reduce(_layernorm, "amax", 32, 4096, 512)

    def test_layernorm_block_amax_non_power_of_2_groups(self):
        # [FUNCTION DOC] Layer norm and block amax where the number of groups is not a
        # power of two.
        """D/G need not be a power of 2."""
        self._norm_block_reduce(_layernorm, "amax", 16, 6144, 128)

    # ---- Epilogue dtype conversion ----

    def test_weighted_rmsnorm_reduce_k_bf16_epilogue(self):
        # [FUNCTION DOC] Weighted rms norm reduction over K with the result cast to bfloat16;
        # checks numerics and fusion.
        def f(x, w):
            # [FUNCTION DOC] HELPER: Weighted rms norm sum over K cast to bfloat16.
            x_normed = _rmsnorm(x.reshape(x.shape[0] * 16, 4096)).reshape(x.shape)
            return (w[:, :, None] * x_normed).sum(dim=1).to(torch.bfloat16)

        x = torch.randn(64, 16, 4096, device=GPU_TYPE)
        w = torch.randn(64, 16, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    def test_layernorm_block_amax_bf16_epilogue(self):
        # [FUNCTION DOC] Layer norm block amax with a bfloat16 conversion epilogue; checks
        # numerics and fusion.
        def f(x):
            # [FUNCTION DOC] HELPER: Layer norm, groups of 16, absolute max, cast to bfloat16.
            return (
                _layernorm(x)
                .reshape(x.shape[0], -1, 16)
                .abs()
                .amax(dim=-1)
                .to(torch.bfloat16)
            )

        x = torch.randn(64, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_fusion()

    # ---- Downstream pointwise fusion ----

    def test_weighted_rmsnorm_reduce_k_pointwise_epilogue(self):
        # [FUNCTION DOC] Weighted rms norm reduction over K followed by multiplication by a scale
        # and add of a bias, to be fused as a pointwise epilogue.
        """Fuse reduced-output pointwise after weighted small-dim-in-X reduction."""

        def f(x, w, scale, bias):
            # [FUNCTION DOC] HELPER: Weighted rms norm sum over K, then scale and bias applied.
            x_normed = _rmsnorm(x.reshape(x.shape[0] * 16, 4096)).reshape(x.shape)
            out = (w[:, :, None] * x_normed).sum(dim=1)
            return out * scale + bias

        x = torch.randn(64, 16, 4096, device=GPU_TYPE)
        w = torch.randn(64, 16, device=GPU_TYPE)
        scale = torch.randn(64, 4096, device=GPU_TYPE)
        bias = torch.randn(64, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x, w, scale, bias))
        self.check_fusion()

    def test_layernorm_block_amax_reduced_pointwise_epilogue(self):
        # [FUNCTION DOC] Layer norm block amax followed by scale and bias at reduced resolution;
        # checks numerics and fusion.
        """Fuse out * scale + bias after reduced-output block amax."""

        def f(x, scale, bias):
            # [FUNCTION DOC] HELPER: Block amax of layer norm, groups of 16, then times
            # scale plus bias.
            out = (
                _layernorm(x)
                .reshape(x.shape[0], x.shape[1] // 16, 16)
                .abs()
                .amax(dim=-1)
            )
            return out * scale + bias

        x = torch.randn(64, 4096, device=GPU_TYPE)
        scale = torch.randn(64, 256, device=GPU_TYPE)
        bias = torch.randn(64, 256, device=GPU_TYPE)
        self.check_numeric(f, (x, scale, bias))
        self.check_fusion()

    def test_rmsnorm_block_scale_swizzle(self):
        # [FUNCTION DOC] Compares the compiled rms norm block-scale-and-swizzle function
        # (payload and swizzled scale) to eager at loose tolerances (1e-2 payload, 1e-6 scale)
        # and checks fusion.
        B, D, G = 128, 4096, 32
        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)

        def f(x, weight):
            # [FUNCTION DOC] HELPER: Calls the block scale swizzle helper with the
            # outer group size.
            return _rmsnorm_block_scale_swizzle(x, weight, G)

        ref_payload, ref_scale = f(x, weight)
        payload, scale = torch.compile(f)(x, weight)
        self.assertEqual(payload, ref_payload, atol=1e-2, rtol=1e-2)
        self.assertEqual(scale, ref_scale, atol=1e-6, rtol=1e-6)
        self.check_fusion()

    # ---- Edge cases ----

    @parametrize(
        "B,D,G",
        [(256, 4096, 16), (128, 4096, 32), (256, 8192, 32)],
    )
    def test_edge_B_equals_D_over_G(self, B, D, G):
        # [FUNCTION DOC] Edge case where batch equals D over group size, which makes matching axes
        # by size ambiguous; runs the layer norm block amax scenario for such shapes.
        """When B == D/G, size-based matching is ambiguous."""
        self._norm_block_reduce(_layernorm, "amax", B, D, G)

    @parametrize("BK", [16, 32])
    def test_edge_B_equals_K(self, BK):
        # [FUNCTION DOC] Edge case where batch equals K, making size-based axis matching
        # ambiguous; runs the weighted rms norm sum scenario with equal batch and K.
        """When B == K, size-based matching is ambiguous."""
        self._weighted_norm_reduce_k(_rmsnorm, "sum", BK, BK, 4096)

    # ---- Dynamic shapes ----

    @parametrize("dynamic", [False, True])
    def test_shapes_weighted_rmsnorm_reduce_k(self, dynamic):
        # [FUNCTION DOC] Weighted rms norm reduction over K, static or with dynamic shapes. When
        # dynamic, runs three batch and D sizes with the K axis marked static and compares each to
        # eager; checks fusion afterwards.
        """Dynamic small-dim-in-x fuses when the local group size is static."""
        K = 16

        def f(x, w):
            # [FUNCTION DOC] HELPER: Rms normalizes (B*K, D) rows from the input shape, then sums
            # the weighted rows over K.
            B, D = x.shape[0], x.shape[2]
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
            x_normed = (x_flat / rms).reshape(B, K, D)
            return (w[:, :, None] * x_normed).sum(dim=1)

        compiled = torch.compile(f, dynamic=dynamic)
        for B, D in [(32, 1024), (64, 2048), (128, 4096)] if dynamic else [(32, 4096)]:
            x = torch.randn(B, K, D, device=GPU_TYPE)
            w = torch.randn(B, K, device=GPU_TYPE)
            if dynamic:
                torch._dynamo.mark_static(x, 1)
                torch._dynamo.mark_static(w, 1)
            ref = f(x, w)
            act = compiled(x, w)
            self.assertEqual(act, ref, atol=1e-2, rtol=1e-2)
        self.check_fusion()

    @parametrize("dynamic", [False, True])
    def test_shapes_layernorm_block_amax(self, dynamic):
        # [FUNCTION DOC] Layer norm block amax, static or dynamic over three batch sizes,
        # comparing with eager and checking fusion.
        def f(x):
            # [FUNCTION DOC] HELPER: Layer norm, groups of 16, absolute max.
            return _layernorm(x).reshape(x.shape[0], -1, 16).abs().amax(dim=-1)

        compiled = torch.compile(f, dynamic=dynamic)
        for B in [32, 64, 256] if dynamic else [32]:
            x = torch.randn(B, 4096, device=GPU_TYPE)
            self.assertEqual(compiled(x), f(x), atol=1e-2, rtol=1e-2)
        self.check_fusion()

    def test_dynamic_shapes_varying_batch_and_dim(self):
        # [FUNCTION DOC] Dynamic-shape compile where both batch and D vary at runtime across three
        # sizes; compares rms norm plus group amax with eager and checks fusion.
        """Dynamic shapes: vary both B and D at runtime."""
        import torch.nn.functional as F

        def f(x, weight):
            # [FUNCTION DOC] HELPER: Rms norm with the given weight, grouped by 128, absolute max.
            x = F.rms_norm(x, (x.shape[-1],), weight)
            B, D = x.shape
            return x.view(B, D // 128, 128).abs().amax(dim=-1)

        compiled = torch.compile(f, dynamic=True)
        for B, D in [(4, 512), (8, 1024), (16, 2048)]:
            x = torch.randn(B, D, device=GPU_TYPE)
            w = torch.randn(D, device=GPU_TYPE)
            ref = f(x, w)
            act = compiled(x, w)
            self.assertEqual(act, ref, atol=1e-2, rtol=1e-2)
        self.check_fusion()

    # ---- Producer-consumer: node2 reads node1's materialized output ----
    # Instead of node1 and node2 sharing a common input, node2 reads
    # node1's output. This triggers the producer-consumer path in
    # NestedReduction.can_fuse.

    @parametrize("B", [1, 128])
    def test_producer_consumer_rmsnorm_amax(self, B):
        # [FUNCTION DOC] Producer-consumer case: rms norm output is materialized and the group
        # amax reads it rather than sharing an input. Parametrized batch one or 128; checks
        # numerics and fusion.
        """RMS norm materializes output, amax reads it."""
        import torch.nn.functional as F

        D, G = 4096, 16

        def f(x, weight):
            # [FUNCTION DOC] HELPER: Rms norm with weight, grouped by 16, absolute max.
            x = F.rms_norm(x, (D,), weight)
            return x.view(B, D // G, G).abs().amax(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    def test_grouped_reduction_input_broadcast_parent_axis(self):
        # [FUNCTION DOC] Group amax whose input is a realized broadcast of a row sum along the
        # parent axis; checks numerics and fusion.
        B, D, G = 16, 1024, 16

        def f(x):
            # [FUNCTION DOC] HELPER: Row sum, expanded back to the input shape and realized,
            # then grouped amax.
            s = x.sum(dim=-1, keepdim=True)
            y = torch.ops._inductor_test.realize(s.expand_as(x))
            return y.reshape(B, D // G, G).amax(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_fusion()

    @parametrize("pointwise_kind", ["full", "row_broadcast", "col_broadcast"])
    @parametrize("epilogue_resolution", ["reduced", "full"])
    def test_reduction_fusion_pointwise_prologue_epilogue(
        self,
        pointwise_kind,
        epilogue_resolution,
    ):
        # [FUNCTION DOC] Rms norm, a realized pointwise prologue (full, row-broadcast or column-
        # broadcast extra) and a reduced-resolution or full-resolution epilogue using the group
        # amax plus an extra tensor. Parametrized by prologue kind and epilogue resolution; checks
        # numerics and fusion.
        import torch.nn.functional as F

        B, D, G = 128, 4096, 128

        def f(x, weight, prologue_extra, epilogue_extra):
            # [FUNCTION DOC] HELPER: Applies rms norm, adds the reshaped prologue extra and
            # realizes it, takes group amax, adds the epilogue extra and returns it, or the input
            # divided by the amax (full resolution) depending on the parametrization.
            x = F.rms_norm(x, (D,), weight)
            x = x.view(B, D // G, G)
            if pointwise_kind == "full":
                prologue_extra = prologue_extra.view(B, D // G, G)
            elif pointwise_kind == "row_broadcast":
                prologue_extra = prologue_extra[:, :, None]
            else:
                prologue_extra = prologue_extra.view(D // G, G)
            x = torch.ops._inductor_test.realize(x + prologue_extra)
            out = x.abs().amax(dim=-1)
            out = out + epilogue_extra
            if epilogue_resolution == "reduced":
                return out
            return (x / (out.abs() + 1e-6)[:, :, None]).view(B, D)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        prologue_extra_shape = {
            "full": (B, D),
            "row_broadcast": (B, 1),
            "col_broadcast": (D,),
        }[pointwise_kind]
        epilogue_extra_shape = {
            "full": (B, D // G),
            "row_broadcast": (B, 1),
            "col_broadcast": (D // G,),
        }[pointwise_kind]
        prologue_extra = torch.randn(prologue_extra_shape, device=GPU_TYPE)
        epilogue_extra = torch.randn(epilogue_extra_shape, device=GPU_TYPE)
        self.check_numeric(f, (x, w, prologue_extra, epilogue_extra))
        self.check_fusion()

    def test_reduced_resolution_pointwise_prologue(self):
        # [FUNCTION DOC] Checks, through a post-fusion custom pass, that exactly one fused nested
        # reduction node exists whose second part has one reduction and only prologue or epilogue
        # nodes, and that at least one prologue works at the reduced resolution. Disables the FX
        # graph cache; then verifies numerics and fusion.
        from torch._inductor.scheduler import FusedNestedReductions

        B, D, G = 128, 4096, 128

        def f(x, group_extra, epilogue_extra):
            # [FUNCTION DOC] HELPER: Row sum of squares, inverse root scale, a realized group-
            # level extra added to the sums, group amax of the scaled input plus the extra, and an
            # epilogue extra added.
            sums = (x * x).sum(dim=-1, keepdim=True)
            inv = torch.rsqrt(sums / D + 1e-6)
            group_extra = torch.ops._inductor_test.realize(group_extra + sums)
            x = (x * inv).view(B, D // G, G)
            out = (x + group_extra[:, :, None]).abs().amax(dim=-1)
            return out + epilogue_extra

        x = torch.randn(B, D, device=GPU_TYPE)
        group_extra = torch.randn(B, D // G, device=GPU_TYPE)
        epilogue_extra = torch.randn(B, D // G, device=GPU_TYPE)
        saw_reduced_prologue = False

        def check_reduction_fusion(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass hook. Asserts one fused nested reduction
            # node, whose second part holds one reduction and every other node is its prologue or
            # epilogue, and records if a prologue has the same element count as the reduced
            # output. Returns the nodes unchanged.
            nonlocal saw_reduced_prologue
            fused_nodes = [n for n in nodes if isinstance(n, FusedNestedReductions)]
            self.assertEqual(len(fused_nodes), 1)
            node2_nodes = list(fused_nodes[0].node2.get_nodes())
            reductions = [sn for sn in node2_nodes if sn.is_reduction()]
            self.assertEqual(len(reductions), 1)
            reduction = reductions[0]
            reduction_names = reduction.get_operation_names()
            _, (reduced_numel, _) = reduction.group
            for sn in node2_nodes:
                if sn.is_reduction():
                    continue
                is_prologue = bool(sn.get_operation_names() & reduction.ancestors)
                is_epilogue = bool(reduction_names & sn.ancestors)
                self.assertTrue(is_prologue or is_epilogue)
                if is_prologue:
                    _, (sn_numel, _) = sn.group
                    saw_reduced_prologue |= sn_numel == reduced_numel
            return nodes

        with inductor_config.patch(
            _post_fusion_custom_pass=check_reduction_fusion,
            fx_graph_cache=False,
        ):
            self.check_numeric(f, (x, group_extra, epilogue_extra))
        self.assertTrue(saw_reduced_prologue)
        self.check_fusion()

    # ---- Exotic indexing ----

    def test_transposed_input(self):
        # [FUNCTION DOC] Numerics only: rms norm and group amax on a
        # transposed (non-contiguous) input.
        """Non-contiguous (transposed) input - numerics must be correct."""

        def f(x):
            # [FUNCTION DOC] HELPER: Transposes x, rms normalizes rows and takes group-
            # of-16 absolute max.
            x = x.t()
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_norm = x / rms
            return x_norm.reshape(x.shape[0], -1, 16).abs().amax(dim=-1)

        x = torch.randn(4096, 64, device=GPU_TYPE)
        self.check_numeric(f, (x,))

    def test_strided_slice_input(self):
        # [FUNCTION DOC] Numerics only: rms norm and group amax on a stride-two slice
        # of the input.
        """Stride-2 slice input - numerics must be correct."""

        def f(x):
            # [FUNCTION DOC] HELPER: Takes every other column, rms normalizes rows, groups of
            # 16 absolute max.
            x = x[:, ::2]
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_norm = x / rms
            return x_norm.reshape(x.shape[0], -1, 16).abs().amax(dim=-1)

        x = torch.randn(32, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x,))

    def test_multi_op_prologue_and_epilogue(self):
        # [FUNCTION DOC] Prologue with multiply, add and relu (realized) and epilogue with log1p
        # and clamp around a group amax; checks numerics and fusion.
        """Prologue does mul+add+relu, epilogue does log1p+clamp."""
        import torch.nn.functional as F

        B, D, G = 64, 4096, 128

        def f(x, weight, bias, scale):
            # [FUNCTION DOC] HELPER: Rms norm, scale and bias then relu realized, group amax, then
            # clamp of log1p of the result between zero and ten.
            x = F.rms_norm(x, (D,), weight)
            x_scaled = torch.ops._inductor_test.realize(torch.relu(x * scale + bias))
            amax = x_scaled.view(B, D // G, G).abs().amax(dim=-1)
            return torch.clamp(torch.log1p(amax), min=0.0, max=10.0)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        bias = torch.randn(D, device=GPU_TYPE)
        scale = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w, bias, scale))
        self.check_fusion()

    @inductor_config.patch(emulate_precision_casts=True)
    def test_fullres_epilogue_with_multiple_outputs(self):
        # [FUNCTION DOC] With precision-cast emulation enabled, group quantization with a full-
        # resolution epilogue returning the half precision payload as float and the scale.
        # Compared against the unnested compiled result; checks fusion.
        """Full-res epilogue producing both converted output and scale."""
        import torch.nn.functional as F

        B, D, G = 64, 4096, 128
        qmax = 448.0

        def f(x, weight):
            # [FUNCTION DOC] HELPER: Rms norm, group amax, scale, quantize to half and return
            # payload and scale.
            x = F.rms_norm(x, (D,), weight)
            x_groups = x.view(B, D // G, G)
            amax = x_groups.abs().amax(dim=-1)
            scale = (amax / qmax).clamp(min=1e-12)
            x_quant = (x_groups / scale.unsqueeze(-1)).to(torch.float16)
            return x_quant.view(B, D).float(), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, w))
        self.check_fusion()

    def test_grouped_reduction_with_weight_mul(self):
        # [FUNCTION DOC] Group amax of rms norm output multiplied elementwise by an extra per-
        # feature weight; checks numerics and fusion.
        """Grouped reduction input involves element-wise weight multiply."""
        import torch.nn.functional as F

        B, D, G = 128, 4096, 32

        def f(x, weight, group_weight):
            # [FUNCTION DOC] HELPER: Rms norm, multiply by the group weight, group-
            # of-32 absolute max.
            x = F.rms_norm(x, (D,), weight)
            # Weight multiply before grouped reduction
            weighted = x * group_weight
            return weighted.view(B, D // G, G).abs().amax(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        gw = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w, gw))
        self.check_fusion()

    # ---- Producer-consumer ----

    @inductor_config.patch(emulate_precision_casts=True)
    def test_producer_consumer_rmsnorm_scale(self):
        # [FUNCTION DOC] With precision-cast emulation, rms norm, group amax and a scale converted
        # to half then float; checks numerics at tolerance 0.01 and fusion.
        """RMS norm + amax + converted scale epilogue."""
        import torch.nn.functional as F

        B, D, G = 128, 4096, 16

        def f(x, weight):
            # [FUNCTION DOC] HELPER: Rms norm, group amax, scale clamp, half conversion,
            # back to float.
            x = F.rms_norm(x, (D,), weight)
            x = x.view(B, D // G, G)
            amax = x.abs().amax(dim=-1)
            scale = (amax / 448.0).clamp(min=1e-12).to(torch.float16)
            return scale.float()

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w), tol=0.01)
        self.check_fusion()

    @inductor_config.patch(emulate_precision_casts=True)
    @parametrize("B", [128, 1])
    def test_producer_consumer_rmsnorm_quant(self, B):
        # [FUNCTION DOC] With precision-cast emulation, rms norm, group amax, scale and a full-
        # resolution quantization epilogue for batch 128 or one; compared with the unnested result
        # and fusion checked.
        """RMS norm + amax + scale + full-res convert epilogue."""
        import torch.nn.functional as F

        D, G = 4096, 128
        qmax = 448.0

        def f(x, weight):
            # [FUNCTION DOC] HELPER: Rms norm, group amax, scale, quantize to half, return payload
            # as float and the scale.
            x = F.rms_norm(x, (D,), weight)
            x_groups = x.view(B, D // G, G)
            amax = x_groups.abs().amax(dim=-1)
            scale = (amax / qmax).clamp(min=1e-12)
            x_quant = (x_groups / scale.unsqueeze(-1)).to(torch.float16)
            return x_quant.view(B, D).float(), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, w))
        self.check_fusion()

    @inductor_config.patch(emulate_precision_casts=True)
    @parametrize(
        "weight_layout",
        [
            "scalar",
            "scalar_1d",
            "feature",
            "singleton",
            "singleton_batch",
            "batch",
            "full",
            "leading_singleton",
            "leading_singleton_feature",
            "leading_singleton_batch",
            "leading_singleton_full",
        ],
    )
    def test_producer_consumer_residual_rmsnorm_quant(self, weight_layout):
        # [FUNCTION DOC] With precision-cast emulation, bfloat16 residual add, rms norm style
        # normalization, multiplication by a weight whose shape varies over eleven layouts
        # (scalars, features, batch, full, with leading singleton dims), group amax and
        # quantization. Compared with the unnested result; checks fusion.
        B, D, G = 128, 2048, 128
        qmax = 448.0
        min_scale = 1.0 / (qmax * 512.0)

        def f(x, residual, weight):
            # [FUNCTION DOC] HELPER: Adds residual in float, normalizes by inverse root of the
            # variance, converts to bfloat16 and multiplies by weight, groups, computes absmax
            # scales and returns the quantized payload as float with the scales.
            h = x.float() + residual.float()
            variance = h.pow(2).mean(dim=-1, keepdim=True)
            normed = h * torch.rsqrt(variance + 1e-6)
            normed_bf16 = normed.to(torch.bfloat16) * weight
            grouped = normed_bf16.view(B, D // G, G)
            absmax = grouped.abs().amax(dim=-1, keepdim=True).float()
            scales = (absmax / qmax).clamp(min=min_scale)
            x_scaled = (grouped / scales).clamp(-qmax, qmax)
            x_quant = x_scaled.to(torch.float16).view(B, D)
            return x_quant.float(), scales.squeeze(-1)

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        residual = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        weight_shapes = {
            "scalar": (),
            "scalar_1d": (1,),
            "feature": (D,),
            "singleton": (1, 1),
            "singleton_batch": (1, D),
            "batch": (B, 1),
            "full": (B, D),
            "leading_singleton": (1, 1, 1),
            "leading_singleton_feature": (1, 1, D),
            "leading_singleton_batch": (1, B, 1),
            "leading_singleton_full": (1, B, D),
        }
        w = torch.randn(
            weight_shapes[weight_layout], device=GPU_TYPE, dtype=torch.bfloat16
        )

        weighted = x.to(torch.bfloat16) * w
        self.assertEqual(weighted.dtype, torch.bfloat16)
        self.assertEqual(weighted.numel(), x.numel())
        self.check_nested_matches_unnested(f, (x, residual, w))
        self.check_fusion()

    @parametrize(
        "B,K,D,expect_fullres_consumer",
        [(64, 16, 4096, True), (1, 16, 1024, True)],
    )
    def test_fullres_epilogue_small_dim_in_x(self, B, K, D, expect_fullres_consumer):
        # [FUNCTION DOC] Weighted K reduction with a full-resolution consumer reading the
        # normalized tensor and the sum. A post-fusion pass checks there is one fused nested
        # reduction, not grouped along R, and whether its second part has non-reduction nodes as
        # expected by the parameter. Checks numerics and fusion, expecting one kernel when a full-
        # resolution consumer fuses.
        """Small-dim-in-X full-res consumer remaps through the parent tile."""
        from torch._inductor.scheduler import FusedNestedReductions

        def f(x, w):
            # [FUNCTION DOC] HELPER: Weighted rms norm sum over K, then adds the sum back to
            # the normalized tensor.
            x_normed = _rmsnorm(x.reshape(B * K, D)).reshape(B, K, D)
            s = (w[:, :, None] * x_normed).sum(dim=1)
            return x_normed + s[:, None, :]

        def check_reduction_fusion(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass hook asserting exactly one fused nested
            # reduction node with no R grouping and a presence of non-reduction nodes in its
            # second part matching the expected flag. Returns nodes unchanged.
            fused_nodes = [n for n in nodes if isinstance(n, FusedNestedReductions)]
            self.assertEqual(len(fused_nodes), 1)
            self.assertFalse(fused_nodes[0].group_size_in_r)
            self.assertEqual(
                any(not sn.is_reduction() for sn in fused_nodes[0].node2.get_nodes()),
                expect_fullres_consumer,
            )
            return nodes

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        with inductor_config.patch(
            _post_fusion_custom_pass=check_reduction_fusion,
            fx_graph_cache=False,
        ):
            self.check_numeric(f, (x, w))
        self.check_fusion(1 if expect_fullres_consumer else None)

    def test_fullres_x_epilogue_rejects_intermediate_dependency(self):
        # [FUNCTION DOC] A full-resolution consumer must not be fused ahead of an extra realized
        # producer it depends on. A post-fusion pass checks the nested node's second part holds
        # only reductions; numerics are checked and fusion is checked without a kernel count.
        """Do not fuse a full-res consumer before its extra producer."""
        from torch._inductor.scheduler import FusedNestedReductions

        B, K, D = 16, 16, 1024

        def f(x, w):
            # [FUNCTION DOC] HELPER: Weighted rms norm sum over K, a realized row sum of it, then
            # combines the normalized tensor, the sum and the row sum.
            x_normed = _rmsnorm(x.reshape(B * K, D)).reshape(B, K, D)
            s = (w[:, :, None] * x_normed).sum(dim=1)
            row_sum = torch.ops._inductor_test.realize(s.sum(dim=-1, keepdim=True))
            return x_normed + s[:, None, :] + row_sum[:, None, :]

        def check_reduction_fusion(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass hook asserting one fused nested
            # reduction node with no R grouping, whose second part nodes are all reductions.
            # Returns the nodes unchanged.
            fused_nodes = [n for n in nodes if isinstance(n, FusedNestedReductions)]
            self.assertEqual(len(fused_nodes), 1)
            self.assertFalse(fused_nodes[0].group_size_in_r)
            self.assertTrue(
                all(sn.is_reduction() for sn in fused_nodes[0].node2.get_nodes())
            )
            return nodes

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        with inductor_config.patch(
            _post_fusion_custom_pass=check_reduction_fusion,
            fx_graph_cache=False,
        ):
            self.check_numeric(f, (x, w))
        self.check_fusion(expected_kernels=None)

    def test_epilogue_rejects_intermediate_dependency(self):
        # [FUNCTION DOC] A pointwise epilogue must not be fused ahead of another node that
        # depends on the reduction. A post-fusion pass records that a nested reduction was seen
        # and asserts its second part has no pointwise nodes; then numerics, the seen flag and
        # fusion are checked.
        """Do not fuse a pointwise epilogue before another dependent node."""
        import torch.nn.functional as F
        from torch._inductor.scheduler import FusedNestedReductions

        B, D, G = 64, 4096, 128

        def f(x, weight):
            # [FUNCTION DOC] HELPER: Rms norm, group amax, a realized sum of the amax and the amax
            # added to that sum.
            x = F.rms_norm(x, (D,), weight)
            amax = x.view(B, D // G, G).abs().amax(dim=-1)
            row_sum = torch.ops._inductor_test.realize(amax.sum(dim=-1, keepdim=True))
            return amax + row_sum

        saw_nested_reduction = False

        def check_reduction_fusion(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass hook asserting one fused nested reduction
            # node, setting the enclosing seen flag and asserting no pointwise nodes in its second
            # part. Returns nodes.
            nonlocal saw_nested_reduction
            fused_nodes = [n for n in nodes if isinstance(n, FusedNestedReductions)]
            self.assertEqual(len(fused_nodes), 1)
            saw_nested_reduction = True
            node2_pointwise = [
                sn for sn in fused_nodes[0].node2.get_nodes() if not sn.is_reduction()
            ]
            self.assertEqual(node2_pointwise, [])
            return nodes

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        with inductor_config.patch(
            _post_fusion_custom_pass=check_reduction_fusion,
            fx_graph_cache=False,
        ):
            self.check_numeric(f, (x, w))
        self.assertTrue(saw_nested_reduction)
        self.check_fusion(expected_kernels=None)

    # ---- Fusion rejection: patterns that must NOT use nested reduction ----

    def _check_rejected(self, f, args):
        # [FUNCTION DOC] HELPER: Checks numerics and asserts that no nested
        # reduction was generated.
        """Verify numerics are correct but nested reduction did not fire."""
        self.check_numeric(f, args)
        self.assertEqual(metrics.codegen_nested_reduction, 0)

    @parametrize("G", [17, 2048])
    def test_reject_bad_group_size(self, G):
        # [FUNCTION DOC] Group sizes that are not a power of two or are too large must not fuse.
        """Non-power-of-2 or too-large group_size must not fuse."""
        D = G * 4

        def f(x):
            # [FUNCTION DOC] HELPER: Rms norm, groups of G, absolute max.
            return _rmsnorm(x).reshape(4, -1, G).abs().amax(dim=-1)

        self._check_rejected(f, (torch.randn(4, D, device=GPU_TYPE),))

    @parametrize("K", [17, 256])
    def test_reject_bad_x_group_size(self, K):
        # [FUNCTION DOC] X-axis group sizes that are not a power of two or exceed the cap
        # must not fuse.
        """X-axis group_size has the same static power-of-2/block cap checks."""
        B, D = 4, 512

        def f(x, w):
            # [FUNCTION DOC] HELPER: Rms norm of (B*K, D) rows, weighted sum over K.
            x_normed = _rmsnorm(x.reshape(B * K, D)).reshape(B, K, D)
            return (w[:, :, None] * x_normed).sum(dim=1)

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        self._check_rejected(f, (x, w))

    def test_small_outer_reduction_fuses(self):
        # [FUNCTION DOC] Small outer reduction (batch 4, D 128, group 16) still fuses; reuses the
        # block reduce scenario.
        self._norm_block_reduce(_rmsnorm, "amax", 4, 128, 16)

    @parametrize("reduce_fn,G", [("argmax", 128), ("var", 128)])
    def test_reject_unsupported_reduction_type(self, reduce_fn, G):
        # [FUNCTION DOC] Argmax and variance reductions need special accumulator handling and must
        # not fuse; looks up the tensor method by name and passes correction zero for variance.
        """argmax/var need special accumulator handling."""
        rfn = getattr(torch.Tensor, reduce_fn)
        kw = {"correction": 0} if reduce_fn == "var" else {}

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the chosen reduction to groups of
            # rms normalized values.
            return rfn(_rmsnorm(x).reshape(4, -1, G), dim=-1, **kw)

        self._check_rejected(f, (torch.randn(4, 4096, device=GPU_TYPE),))

    def test_reject_three_iter_dims(self):
        # [FUNCTION DOC] A group structure with three iteration dims must not fuse.
        """[B, H, groups, G] needs explicit 3D mapping."""

        def f(x):
            # [FUNCTION DOC] HELPER: Rms norm of a reshaped input, reshaped to four
            # dims, absolute max.
            return _rmsnorm(x.reshape(8, 1024)).reshape(4, 2, 8, 128).abs().amax(dim=-1)

        self._check_rejected(f, (torch.randn(4, 2, 1024, device=GPU_TYPE),))

    def test_reject_x_grouped_reduction_with_three_iter_dims(self):
        # [FUNCTION DOC] An X-grouped reduction over a four-dim tensor with three iteration dims
        # must not fuse.
        """[B, H, K, D].sum(dim=2) needs explicit 3D X-axis mapping."""
        B, H, K, D = 2, 3, 16, 512

        def f(x, w):
            # [FUNCTION DOC] HELPER: Weighted rms norm over four dims, sum over dim 2.
            x_normed = _rmsnorm(x.reshape(B * H * K, D)).reshape(B, H, K, D)
            return (w[:, :, :, None] * x_normed).sum(dim=2)

        x = torch.randn(B, H, K, D, device=GPU_TYPE)
        w = torch.randn(B, H, K, device=GPU_TYPE)
        self._check_rejected(f, (x, w))

    def test_reject_multiple_reduce_dims(self):
        # [FUNCTION DOC] Reducing over two local axes at once must not fuse.
        """[B, groups, G1, G2] needs one local reduce axis."""

        def f(x):
            # [FUNCTION DOC] HELPER: Rms norm, reshape to four dims, absolute max over the
            # last two dims.
            return _rmsnorm(x).reshape(4, 32, 16, 8).abs().amax(dim=(-1, -2))

        self._check_rejected(f, (torch.randn(4, 4096, device=GPU_TYPE),))

    def test_reject_split_reduction(self):
        # [FUNCTION DOC] A true split reduction, which changes total element count, must not fuse.
        """True split reduction changes total numel."""

        def f(x):
            # [FUNCTION DOC] HELPER: Sums over the last dim of a reshaped input twice in a row.
            return x.reshape(4, 4, 512).sum(dim=-1).sum(dim=-1)

        self._check_rejected(f, (torch.randn(4, 2048, device=GPU_TYPE),))


class NestedReductionTest(_NestedReductionBase, TestBase):
    # [CLASS DOC] Concrete test class: behavior tests with persistent outer reductions forced.
    force_persistent_outer_reduction = True


class NestedReductionNonPersistentTest(_NestedReductionBase, TestBase):
    # [CLASS DOC] Concrete test class: behavior tests with non-persistent (looped)
    # outer reductions forced.
    force_persistent_outer_reduction = False


TRITON_KERNEL_RE = re.compile(
    r"(?ms)^@triton_heuristics.*?(?=^@triton_heuristics|^async_compile\.wait|\Z)"
)


def _kernel_name(kernel_code: str) -> str:
    # [FUNCTION DOC] HELPER: Extracts the generated triton kernel function name from kernel
    # source; raises AssertionError if no definition is found.
    match = re.search(r"^def (triton_[^(]+)\(", kernel_code, re.MULTILINE)
    if match is None:
        raise AssertionError("could not find Triton kernel name")
    return match.group(1)


def _nested_kernel_signature(force_persistent_outer_reduction: bool | None) -> str:
    # [FUNCTION DOC] HELPER: Returns the kernel name prefix expected for nested kernels:
    # the looped reduction prefix when persistence is forced off, otherwise the
    # persistent reduction prefix.
    return (
        "triton_red_fused"
        if force_persistent_outer_reduction is False
        else "triton_per_fused"
    )


def _is_wrapper_launched_kernel(wrapper_code: str, kernel_code: str) -> bool:
    # [FUNCTION DOC] HELPER: Returns true if the kernel's name appears as a whole word in the
    # wrapper source, i.e. the kernel is actually launched.
    return (
        re.search(rf"\b{re.escape(_kernel_name(kernel_code))}\b", wrapper_code)
        is not None
    )


def _run_and_capture_source_bundle(
    f,
    args,
    kernel_signature: str,
    *,
    dynamic: bool = False,
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, list[str]]:
    # [FUNCTION DOC] HELPER: Compiles and runs f on args with nested reduction enabled and the
    # given choices context (optionally dynamic), under a fresh inductor cache, capturing
    # generated source codes. Resets metrics and dynamo afterwards. Returns the wrapper code (the
    # source containing the function call marker) and the list of triton kernels in the combined
    # source whose text contains the kernel signature and which the wrapper launches.
    def capture():
        # [FUNCTION DOC] HELPER: Enters the nested reduction config patch and the choices context,
        # compiles f and returns its result on the args.
        with (
            inductor_config.patch("triton.nested_reduction", True),
            _choices_context(force_persistent_outer_reduction),
        ):
            compiled = torch.compile(f, dynamic=dynamic)
            return compiled(*args)

    with fresh_inductor_cache():
        _, source_codes = run_and_get_code(capture)
    metrics.reset()
    torch._dynamo.reset()

    combined_code = "\n\n".join(source_codes)
    wrapper_code = next(code for code in source_codes if get_func_call() in code)
    kernel_codes = [
        kernel_code
        for kernel_code in TRITON_KERNEL_RE.findall(combined_code)
        if kernel_signature in kernel_code
        and _is_wrapper_launched_kernel(wrapper_code, kernel_code)
    ]
    return wrapper_code, kernel_codes


def _run_and_capture_sources(
    f,
    args,
    kernel_signature: str,
    *,
    dynamic: bool = False,
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Captures wrapper and kernels via the bundle helper. If not
    # exactly one kernel matches, narrows to kernels carrying minimum block metadata;
    # raises AssertionError unless exactly one kernel remains. Returns the wrapper code and
    # that kernel's source.
    wrapper_code, kernel_codes = _run_and_capture_source_bundle(
        f,
        args,
        kernel_signature,
        dynamic=dynamic,
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )
    if len(kernel_codes) != 1:
        nested_kernel_codes = [
            code
            for code in kernel_codes
            if "'min_xblock':" in code or "'min_rblock':" in code
        ]
        if len(nested_kernel_codes) == 1:
            kernel_codes = nested_kernel_codes
    if len(kernel_codes) != 1:
        raise AssertionError(
            f"expected exactly one fused kernel matching {kernel_signature!r}, "
            f"got {len(kernel_codes)}: "
            f"{[_kernel_name(kernel_code) for kernel_code in kernel_codes]}"
        )
    return wrapper_code, kernel_codes[0]


def _capture_weighted_rmsnorm_reduce_k_kernel_sources(
    batch_size: int,
    K: int,
    D: int,
    *,
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Builds random batch by K by D input and weights, captures the sources
    # of the weighted rms norm reduction over K and returns wrapper and kernel code.
    def f(x, w):
        # [FUNCTION DOC] HELPER: Rms norm of flattened rows, weighted sum over the K axis.
        B, K, D = x.shape
        x_flat = x.reshape(B * K, D)
        x_normed = _rmsnorm(x_flat).reshape(B, K, D)
        return (w[:, :, None] * x_normed).sum(dim=1)

    x = torch.randn(batch_size, K, D, device=GPU_TYPE)
    w = torch.randn(batch_size, K, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_layernorm_block_amax_kernel_sources(
    batch_size: int,
    D: int,
    G: int,
    *,
    norm_kind: str = "layernorm",
    reduction: str = "amax",
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Captures wrapper and kernel code for group reductions after a layer
    # norm or rms norm, with chosen norm kind and reduction (amax, sum or amin), group size G
    # passed as an argument.
    def f(x, G):
        # [FUNCTION DOC] HELPER: Normalizes x with the outer norm kind, groups by G and applies
        # the outer reduction; raises AssertionError for an unsupported reduction name.
        if norm_kind == "layernorm":
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, correction=0)
            x_normed = (x - mean) / torch.sqrt(var + 1e-6)
        else:
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_normed = x / rms

        grouped = x_normed.reshape(x.shape[0], x.shape[1] // G, G)
        if reduction == "amax":
            return grouped.abs().amax(dim=-1)
        if reduction == "sum":
            return grouped.sum(dim=-1)
        if reduction == "amin":
            return grouped.amin(dim=-1)
        raise AssertionError(f"unsupported reduction: {reduction}")

    x = torch.randn(batch_size, D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, G),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_dynamic_layernorm_block_amax_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Captures sources for a dynamic-shape layer norm group amax with width
    # 4096, using the looped reduction kernel prefix.
    def f(x):
        # [FUNCTION DOC] HELPER: Layer norm, groups of 16, absolute max.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        x_normed = (x - mean) / torch.sqrt(var + 1e-6)
        return x_normed.reshape(x.shape[0], x.shape[1] // 16, 16).abs().amax(dim=-1)

    x = torch.randn(batch_size, 4096, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x,),
        "triton_red_fused",
        dynamic=True,
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_amax_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Captures sources for rms norm followed by group-of-16 amax
    # with D 4096.
    B, D, G = batch_size, 4096, 16
    import torch.nn.functional as F

    def f(x, weight):
        # [FUNCTION DOC] HELPER: Rms norm with weight, group amax.
        x = F.rms_norm(x, (D,), weight)
        return x.view(B, D // G, G).abs().amax(dim=-1)

    x = torch.randn(B, D, device=GPU_TYPE)
    w = torch.randn(D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_producer_scale_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Captures sources for rms norm, group amax, scale converted to half
    # and back to float.
    B, D, G = batch_size, 4096, 16
    import torch.nn.functional as F

    def f(x, weight):
        # [FUNCTION DOC] HELPER: Rms norm, group amax, scale, half and float conversions.
        x = F.rms_norm(x, (D,), weight)
        x = x.view(B, D // G, G)
        amax = x.abs().amax(dim=-1)
        scale = (amax / 448.0).clamp(min=1e-12).to(torch.float16)
        return scale.float()

    x = torch.randn(B, D, device=GPU_TYPE)
    w = torch.randn(D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_fullres_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Captures sources for rms norm, group amax and a full-resolution
    # quantization epilogue, group size 128.
    B, D, G = batch_size, 4096, 128
    qmax = 448.0
    import torch.nn.functional as F

    def f(x, weight):
        # [FUNCTION DOC] HELPER: Rms norm, group amax, scale, quantized payload
        # and scale returned.
        x = F.rms_norm(x, (D,), weight)
        x_groups = x.view(B, D // G, G)
        amax = x_groups.abs().amax(dim=-1)
        scale = (amax / qmax).clamp(min=1e-12)
        x_quant = (x_groups / scale.unsqueeze(-1)).to(torch.float16)
        return x_quant.view(B, D).float(), scale

    x = torch.randn(B, D, device=GPU_TYPE)
    w = torch.randn(D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_rmsnorm_block_scale_swizzle_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Captures sources of the block scale swizzle function with D 4096 and
    # group size 32.
    B, D, G = batch_size, 4096, 32

    def f(x, weight):
        # [FUNCTION DOC] HELPER: Calls the block scale swizzle helper.
        return _rmsnorm_block_scale_swizzle(x, weight, G)

    x = torch.randn(B, D, device=GPU_TYPE)
    weight = torch.randn(D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, weight),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_rmsnorm_mxfp8_scale_swizzle_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Captures sources of the mxfp8 scale swizzle function with bfloat16
    # inputs, D 4096 and unit weights, group size 32.
    B, D, G = batch_size, 4096, 32

    def f(x, weight):
        # [FUNCTION DOC] HELPER: Calls the mxfp8 scale swizzle helper.
        return _rmsnorm_mxfp8_scale_swizzle(x, weight, G)

    x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
    weight = torch.ones(D, device=GPU_TYPE, dtype=torch.bfloat16)
    return _run_and_capture_sources(
        f,
        (x, weight),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_bf16_layernorm_block_amax_epilogue_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Captures sources for a layer norm group amax cast to
    # bfloat16, D 4096.
    def f(x):
        # [FUNCTION DOC] HELPER: Layer norm, groups of 16, absolute max, cast to bfloat16.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        x_normed = (x - mean) / torch.sqrt(var + 1e-6)
        return (
            x_normed.reshape(x.shape[0], x.shape[1] // 16, 16)
            .abs()
            .amax(dim=-1)
            .to(torch.bfloat16)
        )

    x = torch.randn(batch_size, 4096, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x,),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_layernorm_block_amax_pointwise_epilogue_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Captures sources for a layer norm group amax followed by scale and
    # bias, with D 4096 and 256 groups.
    def f(x, scale, bias):
        # [FUNCTION DOC] HELPER: Layer norm, group amax, times scale plus bias.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        x_normed = (x - mean) / torch.sqrt(var + 1e-6)
        out = x_normed.reshape(x.shape[0], x.shape[1] // 16, 16).abs().amax(dim=-1)
        return out * scale + bias

    x = torch.randn(batch_size, 4096, device=GPU_TYPE)
    scale = torch.randn(batch_size, 256, device=GPU_TYPE)
    bias = torch.randn(batch_size, 256, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, scale, bias),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


class _InternalsBase:
    # [CLASS DOC] HELPER: Mixin (not a test case) with kernel-form tests: each captures the
    # wrapper and generated kernel for a scenario and asserts load and store counts, kernel
    # metadata, allocation counts and block-size contracts. The class attribute selects forced
    # persistent or looped outer reductions (default none).
    force_persistent_outer_reduction: bool | None = None

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the parent setUp, then resets inductor metrics and
        # dynamo compilation metrics.
        super().setUp()
        metrics.reset()
        torch._dynamo.utils.clear_compilation_metrics()

    def looped_or_persistent(self, looped, persistent):
        # [FUNCTION DOC] HELPER: Returns the first argument when outer reductions are forced
        # looped, otherwise the second.
        return looped if self.force_persistent_outer_reduction is False else persistent

    def check_code(
        self,
        code_str,
        num_kernels,
        num_allocs: int | None = None,
        num_deallocs: int | None = None,
    ):
        # [FUNCTION DOC] HELPER: Asserts the wrapper code calls the main function and launches
        # exactly num_kernels kernels; optionally asserts the count of strided allocations and,
        # unless the cpp wrapper is on, of explicit deletions.
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(),
            num_kernels,
            exactly=True,
        ).run(code_str)
        if num_allocs is not None:
            FileCheck().check(get_func_call()).check_count(
                "empty_strided", num_allocs, exactly=True
            ).run(code_str)
        if num_deallocs is not None and not inductor_config.cpp_wrapper:
            FileCheck().check(get_func_call()).check_count(
                "del ", num_deallocs, exactly=True
            ).run(code_str)

    def check_kernel_io_counts(
        self,
        kernel_code: str,
        *,
        input_counts: dict[int, int],
        num_outputs: int,
    ) -> None:
        # [FUNCTION DOC] HELPER: Parses kernel source for loads from input pointers and loads and
        # stores for output pointers. Asserts input load counts per pointer index match
        # input_counts, there are no loads from outputs, and exactly num_outputs distinct output
        # pointers are stored.
        load_ids = [int(i) for i in re.findall(r"tl\.load\(in_ptr(\d+)\b", kernel_code)]
        output_load_ids = re.findall(r"tl\.load\(out_ptr(\d+)\b", kernel_code)
        store_ids = re.findall(r"tl\.store\(out_ptr(\d+)\b", kernel_code)
        actual_input_counts = {
            idx: load_ids.count(idx) for idx in sorted(set(load_ids))
        }
        self.assertEqual(actual_input_counts, input_counts)
        self.assertEqual(len(output_load_ids), 0)
        self.assertEqual(len(store_ids), num_outputs)
        self.assertEqual(len(set(store_ids)), num_outputs)

    def check_kernel_meta(
        self, kernel_code: str, *, num_inputs: int, num_outputs: int
    ) -> None:
        # [FUNCTION DOC] HELPER: Asserts the kernel metadata lists exactly one entry for the load
        # count equal to num_inputs and one for the store count equal to num_outputs.
        FileCheck().check_count(
            f"'num_load': {num_inputs}", 1, exactly=True
        ).check_count(f"'num_store': {num_outputs}", 1, exactly=True).run(kernel_code)

    def check_axis_classification_contract(
        self,
        kernel_code: str,
        *,
        min_xblock: int | None = None,
        min_rblock: int | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Asserts the minimum x block and minimum r block metadata: absent
        # when the argument is none, else appearing exactly once with the given value.
        if min_xblock is None:
            FileCheck().check_not("'min_xblock':").run(kernel_code)
        else:
            FileCheck().check_count(f"'min_xblock': {min_xblock}", 1, exactly=True).run(
                kernel_code
            )
        if min_rblock is None:
            FileCheck().check_not("'min_rblock':").run(kernel_code)
        else:
            FileCheck().check_count(f"'min_rblock': {min_rblock}", 1, exactly=True).run(
                kernel_code
            )

    def assert_single_kernel_form(
        self,
        capture,
        *capture_args,
        input_counts: dict[int, int],
        num_outputs: int,
        meta_num_load: int | None = None,
        num_allocs: int | None = None,
        num_deallocs: int | None = None,
        min_xblock: int | None = None,
        min_rblock: int | None = None,
        extra_checks: FileCheck | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Runs a capture function with the class persistence setting, then
        # checks IO counts, metadata (load count defaults to the sum of input counts), wrapper
        # code with one kernel (allocs default to the output count, deallocs to the number of
        # inputs), the block contract, and optional extra pattern checks on the kernel source.
        wrapper_code, kernel_code = capture(
            *capture_args,
            force_persistent_outer_reduction=self.force_persistent_outer_reduction,
        )
        if num_deallocs is None:
            num_deallocs = len(input_counts)
        self.check_kernel_io_counts(
            kernel_code, input_counts=input_counts, num_outputs=num_outputs
        )
        meta_load = (
            meta_num_load if meta_num_load is not None else sum(input_counts.values())
        )
        self.check_kernel_meta(
            kernel_code,
            num_inputs=meta_load,
            num_outputs=num_outputs,
        )
        if num_allocs is None:
            num_allocs = num_outputs
        self.check_code(
            wrapper_code,
            num_kernels=1,
            num_allocs=num_allocs,
            num_deallocs=num_deallocs,
        )
        self.check_axis_classification_contract(
            kernel_code,
            min_xblock=min_xblock,
            min_rblock=min_rblock,
        )
        if extra_checks is not None:
            extra_checks.run(kernel_code)

    def test_layernorm_block_amax_kernel_form(self):
        # [FUNCTION DOC] Kernel form for layer norm block amax at batch 32, D 4096, group 16: the
        # input is loaded twice in looped form, once in persistent form; minimum r block 16.
        self.assert_single_kernel_form(
            _capture_layernorm_block_amax_kernel_sources,
            32,
            4096,
            16,
            input_counts=self.looped_or_persistent({0: 2}, {0: 1}),
            num_outputs=1,
            meta_num_load=self.looped_or_persistent(2, 1),
            min_rblock=16,
        )

    def test_weighted_rmsnorm_reduce_k_kernel_form(self):
        # [FUNCTION DOC] Kernel form for the weighted rms norm reduction over K (batch 32, K 16, D
        # 4096), with minimum x block 16 and source checks for reduced block shapes, reshape, sum
        # and no local R reduction size.
        self.assert_single_kernel_form(
            _capture_weighted_rmsnorm_reduce_k_kernel_sources,
            32,
            16,
            4096,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=1,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_xblock=16,
            extra_checks=FileCheck()
            .check("nested_X_REDUCED_BLOCK")
            .check("tl.reshape")
            .check("nested_X_REDUCED_BLOCK, nested_X_LOCAL_REDUCTION_SIZE, R0_BLOCK")
            .check("tl.sum")
            .check_not("nested_R0_LOCAL_REDUCTION_SIZE"),
        )

    def test_weighted_rmsnorm_reduce_k_B1_kernel_form(self):
        # [FUNCTION DOC] Kernel form for the weighted rms norm reduction with batch one, K
        # 16, D 1024.
        self.assert_single_kernel_form(
            _capture_weighted_rmsnorm_reduce_k_kernel_sources,
            1,
            16,
            1024,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=1,
            min_xblock=16,
        )

    def test_dynamic_layernorm_block_amax_kernel_form(self):
        # [FUNCTION DOC] Kernel form for the dynamic layer norm block amax at batch 32.
        self.assert_single_kernel_form(
            _capture_dynamic_layernorm_block_amax_kernel_sources,
            32,
            input_counts={0: 2},
            num_outputs=1,
            min_rblock=16,
        )

    def test_nested_kernel_disables_cooperative_reduction(self):
        # [FUNCTION DOC] Returns immediately for the looped setting. Otherwise compiles the
        # layer norm block amax with a choices handler that requests cooperative and
        # persistent reduction, and asserts the kernel has no split-reduction markers and
        # keeps minimum r block 16.
        if self.force_persistent_outer_reduction is False:
            return

        class _CooperativeChoices(InductorChoices):
            # [CLASS DOC] HELPER: Choices subclass requesting cooperative
            # and persistent reduction.
            @staticmethod
            def should_use_cooperative_reduction(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Ignores its arguments and returns true.
                return True

            @staticmethod
            def should_use_persistent_reduction(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Ignores its arguments and returns true.
                return True

        with V.set_choices_handler(_CooperativeChoices()):
            _wrapper_code, kernel_code = _capture_layernorm_block_amax_kernel_sources(
                32,
                4096,
                16,
                force_persistent_outer_reduction=None,
            )

        FileCheck().check_not("rsplit").check_not("RSPLIT").run(kernel_code)
        self.check_axis_classification_contract(kernel_code, min_rblock=16)

    def test_producer_consumer_amax_kernel_form(self):
        # [FUNCTION DOC] Kernel form for rms norm followed by group amax (batch 128); checks no
        # tile split in the kernel.
        self.assert_single_kernel_form(
            _capture_amax_kernel_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=1,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=16,
            extra_checks=FileCheck().check_not("tl.split("),
        )

    def test_producer_consumer_scale_kernel_form(self):
        # [FUNCTION DOC] Kernel form for rms norm, amax and half-precision scale at batch 128.
        self.assert_single_kernel_form(
            _capture_producer_scale_kernel_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=1,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=16,
        )

    def test_fullres_kernel_form(self):
        # [FUNCTION DOC] Kernel form for the full-resolution quantization case (two outputs, batch
        # 128, min r block 128); checks no split and a broadcast.
        self.assert_single_kernel_form(
            _capture_fullres_kernel_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=2,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=128,
            extra_checks=FileCheck().check_not("tl.split(").check("tl.broadcast_to"),
        )

    def test_rmsnorm_block_scale_swizzle_kernel_form(self):
        # [FUNCTION DOC] Kernel form for the block scale swizzle with two outputs; checks the
        # swizzled store index expression appears in the kernel.
        self.assert_single_kernel_form(
            _capture_rmsnorm_block_scale_swizzle_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=2,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=32,
            extra_checks=FileCheck().check(
                "tl.store(out_ptr3 + (4*(x0 // 32) + 16*((x0 % 32))"
            ),
        )

    def test_rmsnorm_mxfp8_scale_swizzle_kernel_form(self):
        # [FUNCTION DOC] Skipped unless the CUDA device capability is at least 10.0. Kernel form
        # for the mxfp8 swizzle case with two outputs; checks the hardware exponent conversion
        # instruction appears exactly once.
        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("cvt_e8m0_rceil lowering requires SM100+")

        self.assert_single_kernel_form(
            _capture_rmsnorm_mxfp8_scale_swizzle_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=2,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=32,
            extra_checks=FileCheck().check_count(
                "cvt.rp.satfinite.ue8m0x2.f32", 1, exactly=True
            ),
        )

    def test_bf16_layernorm_block_amax_epilogue_kernel_form(self):
        # [FUNCTION DOC] Kernel form for layer norm block amax with bfloat16 epilogue at batch 64.
        self.assert_single_kernel_form(
            _capture_bf16_layernorm_block_amax_epilogue_sources,
            64,
            input_counts=self.looped_or_persistent({0: 2}, {0: 1}),
            num_outputs=1,
            meta_num_load=self.looped_or_persistent(2, 1),
            min_rblock=16,
        )

    def test_layernorm_block_amax_pointwise_epilogue_kernel_form(self):
        # [FUNCTION DOC] Kernel form for layer norm block amax with scale and bias epilogue at
        # batch 64, three inputs.
        self.assert_single_kernel_form(
            _capture_layernorm_block_amax_pointwise_epilogue_sources,
            64,
            input_counts=self.looped_or_persistent(
                {0: 2, 1: 1, 2: 1},
                {0: 1, 1: 1, 2: 1},
            ),
            num_outputs=1,
            meta_num_load=self.looped_or_persistent(4, 3),
            min_rblock=16,
        )


class NestedReductionInternalsPersistentTest(_InternalsBase, TestCase):
    # [CLASS DOC] Concrete internals test class with persistent outer reductions; skipped
    # without a GPU.
    __unittest_skip__ = not HAS_GPU
    force_persistent_outer_reduction = True


class NestedReductionInternalsNonPersistentTest(_InternalsBase, TestCase):
    # [CLASS DOC] Concrete internals test class with looped outer reductions; skipped
    # without a GPU.
    __unittest_skip__ = not HAS_GPU
    force_persistent_outer_reduction = False


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
