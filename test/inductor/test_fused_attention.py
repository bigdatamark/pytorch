# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the fused attention (scaled dot product attention) pattern rewriter in
# inductor. A template TestCase defines many private test bodies, each covering a hand-written
# attention variant (scaling, masks, dropout, dtype casts, permutes) that the pattern matcher
# should replace with the SDPA aten op. Concrete GPU and CPU (static and dynamic shape) test
# classes at the bottom bind these bodies to public test methods. Also tests pattern registration
# for training. Concrete classes are defined at import only if the matching backend (GPU with
# Triton and fused attention support, or CPU) is available; main runs tests only on Linux.
import functools
import itertools
import math
import os
from unittest import mock

import torch
import torch._inductor.config
import torch.utils.checkpoint
from torch._C import FileCheck
from torch._dynamo.debug_utils import aot_graph_input_parser
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    SM80OrLater,
)
from torch.testing._internal.common_utils import IS_LINUX, skipIfXpu, TEST_WITH_ROCM
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
    HAS_XPU_AND_TRITON,
)


def checkpoint_wrapper(fn):
    # [FUNCTION DOC] HELPER: Wraps a function so it runs under activation checkpointing with the
    # reentrant implementation. Returns a new function taking positional tensors only, which
    # forwards them to the checkpoint call of the original function and returns its result.
    def inner(*args):
        # [FUNCTION DOC] HELPER: Calls the checkpoint utility with the captured function, reentrant
        # mode on, and the positional arguments; returns the function output.
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

    return inner


class TestSDPAPatternRewriterTemplate(TestCase):
    # [CLASS DOC] Shared template for SDPA pattern rewriter tests. Subclasses provide the device
    # attribute and bind the private test bodies as public tests. The class flag
    # use_static_shapes (true here) controls whether inputs get a dynamic batch dimension in the
    # common checker. setUp/tearDown temporarily switch the CUDA matmul fp32 precision setting to
    # tf32 and restore it.
    use_static_shapes = True

    def setUp(self):
        # [FUNCTION DOC] HELPER: Saves the current CUDA matmul fp32 precision, sets it to tf32
        # (global state mutation), then runs the base setUp.
        self.prev_tf32 = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs the base tearDown first, then restores the saved CUDA matmul
        # fp32 precision setting.
        super().tearDown()
        torch.backends.cuda.matmul.fp32_precision = self.prev_tf32

    def _clone_inputs(self, inputs):
        # [FUNCTION DOC] HELPER: Returns a new list with every tensor input cloned and every
        # non- tensor input passed through unchanged, so eager and compiled runs get
        # independent tensors.
        def clone(x):
            # [FUNCTION DOC] HELPER: Returns the argument itself if it is not a tensor, else a
            # clone of it.
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return [clone(x) for x in inputs]

    def _check_common(
        self,
        dot_prod_attention,
        args1=None,
        contains=True,
        atol=1e-3,
        has_fuse_pattern=True,
        has_dropout=False,
        check_train=True,
        override_check_equal=False,
        dtype=torch.float,
        rtol=0.2,
        expected_fused_attention_patterns=None,
    ):
        # [FUNCTION DOC] HELPER: Common checker. Compares eager execution of an attention function
        # with a compiled (full graph) run and verifies the fused attention pattern was applied.
        # When args1 is none it makes three random tensors of shape 4 by 2 by 16 by 32 on the test
        # device in the given dtype; the inputs are cloned for the compiled run. It loops over
        # inference and (if check_train) training; training is skipped on XPU. In each mode
        # floating inputs get requires_grad set, and, for non-static-shape classes, the leading
        # dimension of the first three compiled inputs is marked dynamic. A training flag argument
        # is appended when has_dropout is set. Both runs use the same seed. The compiled run
        # records generated code; if expected_fused_attention_patterns has an entry for the mode,
        # the run happens with the pattern match debug env var patched. Asserts: the fuse_attention
        # counter is at least one (if has_fuse_pattern), the per-pattern counter for the expected
        # pattern is at least one, the code mentions the scaled dot product aten op (if contains),
        # outputs match within atol and rtol (skipped for dropout unless override_check_equal), and
        # in training the gradients of floating inputs match likewise.
        if args1 is None:
            tensor_shape = (4, 2, 16, 32)
            args1 = [
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
            ]
        else:
            args1 = list(args1)
        args2 = self._clone_inputs(args1)

        for training in [False, True] if check_train else [False]:
            if training and self.device == "xpu":
                # Intel GPU have not implemented sdpa backward yet mode.
                # TODO: remove this when sdpa backward is implemented for XPU.
                continue
            for x in itertools.chain(args1[:], args2[:]):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    x.requires_grad = training

            if not self.use_static_shapes:
                for i in range(min(3, len(args2))):
                    if isinstance(args2[i], torch.Tensor):
                        torch._dynamo.mark_dynamic(args2[i], 0)

            dropout_arg = [training] if has_dropout else []
            torch.manual_seed(1234)
            result1 = dot_prod_attention(*(args1 + dropout_arg))

            counters.clear()
            torch.manual_seed(1234)
            expected_fused_attention_pattern = (
                expected_fused_attention_patterns.get(training)
                if expected_fused_attention_patterns is not None
                else None
            )
            if expected_fused_attention_pattern is not None:
                with mock.patch.dict(
                    os.environ, {"TORCHINDUCTOR_PATTERN_MATCH_DEBUG": "__none__"}
                ):
                    result2, source_code = run_and_get_code(
                        torch.compile(dot_prod_attention, fullgraph=True),
                        *(args2 + dropout_arg),
                    )
            else:
                result2, source_code = run_and_get_code(
                    torch.compile(dot_prod_attention, fullgraph=True),
                    *(args2 + dropout_arg),
                )
            source_code = "\n".join(source_code)
            if has_fuse_pattern:
                self.assertGreaterEqual(counters["inductor"]["fuse_attention"], 1)
            if expected_fused_attention_pattern is not None:
                self.assertGreaterEqual(
                    counters["inductor_pattern_matcher_per_pattern"][
                        expected_fused_attention_pattern
                    ],
                    1,
                )
            if contains:
                # many of the patterns get re-expanded in dispatcher
                self.assertIn(
                    "aten._scaled_dot_product",
                    source_code,
                )

            # some tests configured with very low dropout where we still want to check equality
            if not has_dropout or override_check_equal:
                self.assertEqual(result1, result2, atol=atol, rtol=rtol)

            if training:
                result1.sum().backward()
                result2.sum().backward()
                for arg1, arg2 in zip(args1, args2):
                    if (
                        isinstance(arg1, torch.Tensor)
                        and arg1.is_floating_point()
                        and (not has_dropout or override_check_equal)
                    ):
                        self.assertEqual(arg1.grad, arg2.grad, atol=atol, rtol=rtol)

    def _test_sdpa_rewriter_1(self):
        # [FUNCTION DOC] HELPER: Test body for the plain scaled (division) attention pattern. For
        # float and half dtypes it picks tolerances (looser for half on CPU/XPU and for float on
        # ROCm) and runs the common checker on the function and on its checkpoint-wrapped version.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reference attention: matmul of query with transposed key,
            # divided by the square root of the key embedding size, softmax over the last
            # dimension, matmul with value.
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        for dtype in [torch.float, torch.half]:
            atol = 0.0015
            rtol = 1.3e-6 if dtype == torch.float else 0.7
            if self.device in ["cpu", "xpu"] and dtype == torch.half:
                atol = 2e-3
                rtol = 1e-2
            if TEST_WITH_ROCM and dtype == torch.float:
                atol = 3e-3
                rtol = 1e-2
            self._check_common(dot_prod_attention, dtype=dtype, atol=atol, rtol=rtol)
            self._check_common(
                checkpoint_wrapper(dot_prod_attention),
                dtype=dtype,
                atol=atol,
                rtol=rtol,
            )

    @torch._inductor.config.patch("freezing", True)
    def _test_sdpa_rewriter_1_freezing(self):
        # [FUNCTION DOC] HELPER: Same pattern as the first rewriter test but with the freezing
        # inductor config enabled by decorator. Uses half dtype only, no gradient mode, inference
        # only, with tolerances adjusted for CPU/XPU, and runs the common checker once.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reference attention with division scaling, softmax, and
            # value matmul.
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        for dtype in [torch.half]:
            atol = 0.001
            rtol = 1.3e-6 if dtype == torch.float else 0.7
            if self.device in ["cpu", "xpu"] and dtype == torch.half:
                atol = 2e-3
                rtol = 1e-2
            with torch.no_grad():
                self._check_common(
                    dot_prod_attention,
                    dtype=dtype,
                    atol=atol,
                    rtol=rtol,
                    check_train=False,
                )

    def _test_insignificant_strides(self):
        # [FUNCTION DOC] HELPER: Regression test for a stride issue (github issue 124289). Builds
        # the inputs for a captured aot graph function with the input parser on the GPU device,
        # runs the function eagerly and compiled, and requires the first two outputs to be close
        # (the random generator seed and offset outputs are ignored). Passes if compilation
        # succeeds and agrees.
        f32 = torch.float32

        # repro taken from https://github.com/pytorch/pytorch/issues/124289
        # constant_pad_nd is a single element tensor that gets expanded

        def forward(
            permute_3: "f32[1, 32, 1, 128]",
            permute_4: "f32[1, 32, 1, 128]",
            permute_5: "f32[1, 32, 1, 128]",
            permute_6: "f32[1, 1, 64]",
            mul_2: "f32[1, 1, 1, 1]",
        ):
            # [FUNCTION DOC] HELPER: Captured graph function. Computes cos and sin of a doubled
            # position tensor, applies a rotary style rotation to the first two tensors (query and
            # key) by slicing, negating and concatenating halves, builds a one-element padded mask
            # expanded to batch 1, 32 heads, 1 by 1, then calls the efficient scaled dot product
            # attention aten op with it as bias and returns the op output tuple.
            cat = torch.ops.aten.cat.default([permute_6, permute_6], 2)
            permute_6 = None
            cos = torch.ops.aten.cos.default(cat)
            sin = torch.ops.aten.sin.default(cat)
            unsqueeze_10 = torch.ops.aten.unsqueeze.default(cos, 1)
            cos = None
            unsqueeze_11 = torch.ops.aten.unsqueeze.default(sin, 1)
            sin = None
            mul_5 = torch.ops.aten.mul.Tensor(permute_3, unsqueeze_10)
            slice_10 = torch.ops.aten.slice.Tensor(permute_3, 3, 0, 64)
            slice_11 = torch.ops.aten.slice.Tensor(
                permute_3, 3, 64, 9223372036854775807
            )
            permute_3 = None
            neg = torch.ops.aten.neg.default(slice_11)
            slice_11 = None
            cat_1 = torch.ops.aten.cat.default([neg, slice_10], 3)
            neg = slice_10 = None
            mul_6 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_11)
            cat_1 = None
            add_1 = torch.ops.aten.add.Tensor(mul_5, mul_6)
            mul_5 = mul_6 = None
            mul_7 = torch.ops.aten.mul.Tensor(permute_4, unsqueeze_10)
            unsqueeze_10 = None
            slice_12 = torch.ops.aten.slice.Tensor(permute_4, 3, 0, 64)
            slice_13 = torch.ops.aten.slice.Tensor(
                permute_4, 3, 64, 9223372036854775807
            )
            permute_4 = None
            neg_1 = torch.ops.aten.neg.default(slice_13)
            slice_13 = None
            cat_2 = torch.ops.aten.cat.default([neg_1, slice_12], 3)
            neg_1 = slice_12 = None
            mul_8 = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_11)
            cat_2 = unsqueeze_11 = None
            add_2 = torch.ops.aten.add.Tensor(mul_7, mul_8)
            mul_7 = mul_8 = None
            slice_14 = torch.ops.aten.slice.Tensor(mul_2, 0, 0, 9223372036854775807)
            mul_2 = None
            slice_15 = torch.ops.aten.slice.Tensor(slice_14, 1, 0, 9223372036854775807)
            slice_14 = None
            slice_16 = torch.ops.aten.slice.Tensor(slice_15, 2, 0, 9223372036854775807)
            slice_15 = None
            constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                slice_16, [0, 7], 0.0
            )
            slice_16 = None
            slice_17 = torch.ops.aten.slice.Tensor(constant_pad_nd, -1, 0, 1)
            constant_pad_nd = None
            expand_5 = torch.ops.aten.expand.default(slice_17, [1, 32, 1, 1])
            _scaled_dot_product_efficient_attention = (
                torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    add_1, add_2, permute_5, expand_5, True
                )
            )
            return _scaled_dot_product_efficient_attention

        kwargs = aot_graph_input_parser(forward, device=GPU_TYPE)
        # runs successfully
        out_eager = forward(**kwargs)
        out_c = torch.compile(forward)(**kwargs)
        # dont compare philox_seed/offset
        torch.testing.assert_close(out_eager[0:2], out_c[0:2])

    def _test_pattern_fails_with_reuse(self):
        # [FUNCTION DOC] HELPER: Test body checking the rewrite is not applied when the softmax
        # weights are also returned. Compiles the function, runs it on random inputs of shape 2 by
        # 4 by 8 by 16, gets the single generated code string and asserts it lacks the efficient
        # attention aten op.
        """
        This test checks that the replacement is not done
        when an intermediate result is being used / returned downstream
        """

        @torch.compile(fullgraph=True)
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Compiled (full graph) function returning the attention output
            # and the intermediate softmax weights, so the weights have a second user.
            attn_weights = (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
            )
            return attn_weights.matmul(value), attn_weights

        tensor_shape = (2, 4, 8, 16)
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
        ]
        _, (source_code,) = run_and_get_code(dot_prod_attention, *args)
        self.assertNotIn("aten._scaled_dot_product_efficient_attention", source_code)

    def _test_sdpa_rewriter_2(self):
        # [FUNCTION DOC] HELPER: Test body for attention scaled by multiplication with a
        # reciprocal of the square root of the key size; runs the common checker plainly and
        # checkpoint- wrapped.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reference attention: scores multiplied by the reciprocal
            # square root of the key embedding size, softmax over the last dimension, matmul
            # with value.
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .mul(1.0 / math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        self._check_common(dot_prod_attention)
        self._check_common(checkpoint_wrapper(dot_prod_attention))

    def _test_sdpa_rewriter_3(self):
        # [FUNCTION DOC] HELPER: Test body for attention with dropout after softmax and a
        # fixed divisor of 3.0; runs the common checker (dropout on, no code string check)
        # plain and checkpointed.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training: bool
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reference attention: scores divided by 3.0, softmax, dropout
            # with probability 0.4 controlled by the training flag (not in place), then matmul
            # with value.
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1)).div(3.0).softmax(dim=-1),
                p=0.4,
                training=training,
                inplace=False,
            ).matmul(value)

        self._check_common(dot_prod_attention, contains=False, has_dropout=True)
        self._check_common(
            checkpoint_wrapper(dot_prod_attention), contains=False, has_dropout=True
        )

    def _test_sdpa_rewriter_4(self):
        # [FUNCTION DOC] HELPER: Test body like the previous one but with multiplicative scaling
        # of 0.4 and dropout probability 0.2; plain and checkpointed runs, dropout on, no code
        # string check.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reference attention: scores multiplied by 0.4, softmax,
            # dropout with probability 0.2 (training flag), then matmul with value.
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1)).mul(0.4).softmax(dim=-1),
                p=0.2,
                inplace=False,
                training=training,
            ).matmul(value)

        self._check_common(dot_prod_attention, contains=False, has_dropout=True)
        self._check_common(
            checkpoint_wrapper(dot_prod_attention), contains=False, has_dropout=True
        )

    def _test_sdpa_rewriter_5(self):
        # [FUNCTION DOC] HELPER: Test body for causal-mask attention variants. Runs the common
        # checker, without code string check, on three functions and each checkpoint-wrapped.
        def sfdp_pattern_5_v1(query, key, value):
            # [FUNCTION DOC] HELPER: Causal attention: lower triangular boolean mask turned into
            # zero and negative infinity values, added to scores scaled by the square root of the
            # head size, softmax, matmul with value.
            attn_mask = torch.ones(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            return attn_weight @ value

        def sfdp_pattern_5_v2(query, key, value):
            # [FUNCTION DOC] HELPER: Attention with an all-false boolean mask added to the
            # scaled scores, softmax, then matmul with value (see issue 100318 below).
            # https://github.com/pytorch/pytorch/issues/100318.
            attn_mask = torch.zeros(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).bool()
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            return attn_weight @ value

        def sfdp_pattern_5_v3(query, key, value):
            # [FUNCTION DOC] HELPER: Causal attention as the first variant but dividing by the
            # square root of the head size plus 0.1 (see issue 174049 below).
            # https://github.com/pytorch/pytorch/issues/174049.
            attn_mask = torch.ones(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / (math.sqrt(query.size(-1)) + 0.1))
                + attn_mask,
                dim=-1,
            )
            return attn_weight @ value

        self._check_common(sfdp_pattern_5_v1, contains=False)
        self._check_common(checkpoint_wrapper(sfdp_pattern_5_v1), contains=False)
        self._check_common(sfdp_pattern_5_v2, contains=False)
        self._check_common(checkpoint_wrapper(sfdp_pattern_5_v2), contains=False)
        self._check_common(sfdp_pattern_5_v3, contains=False)
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_5_v3),
            contains=False,
        )

    def _test_sdpa_rewriter_6(self):
        # [FUNCTION DOC] HELPER: Test body for causal-mask attention with dropout of 0.5. Runs the
        # common checker with dropout on and no code string check for two scale variants, each
        # plain and wrapped.
        def sfdp_pattern_6(query, key, value, training):
            # [FUNCTION DOC] HELPER: Causal attention with dropout 0.5 (training flag) after
            # softmax and a standard square root scale.
            attn_mask = torch.ones(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            attn_weight = torch.nn.functional.dropout(attn_weight, 0.5, training)
            return attn_weight @ value

        def sfdp_pattern_6_v2(query, key, value, training):
            # [FUNCTION DOC] HELPER: Same as the first variant but the divisor is the square root
            # plus 0.1.
            attn_mask = torch.ones(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / (math.sqrt(query.size(-1)) + 0.1))
                + attn_mask,
                dim=-1,
            )
            attn_weight = torch.nn.functional.dropout(attn_weight, 0.5, training)
            return attn_weight @ value

        self._check_common(sfdp_pattern_6, contains=False, has_dropout=True)
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_6), contains=False, has_dropout=True
        )
        self._check_common(sfdp_pattern_6_v2, contains=False, has_dropout=True)
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_6_v2),
            contains=False,
            has_dropout=True,
        )

    def _test_sdpa_rewriter_7(self):
        # [FUNCTION DOC] HELPER: Test body for permuted half attention with a near zero dropout.
        # Builds half tensors of shape 2 by 8 by 4 by 16 on the test device, then runs the common
        # checker for the first variant (code check requires SM80 or later) and second variant (no
        # code check), then on the GPU device for both checkpoint-wrapped versions. Dropout checks
        # use override_check_equal with atol 2e-3.
        def sfdp_pattern_7(query, key, value, training):
            # [FUNCTION DOC] HELPER: Permutes inputs to sequence-first head layout, scales scores
            # by dividing by the square root, casts to float32, softmax, dropout of 1e-11, casts to
            # half, matmul with v.
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            # Set to False
            attn_weight = torch.dropout(attn_weight, 0.00000000001, training)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        def sfdp_pattern_7_v2(query, key, value, training):
            # [FUNCTION DOC] HELPER: Same as the first variant but the divisor is the square root
            # plus 0.1.
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            div = q @ k.transpose(-2, -1) / (math.sqrt(q.size(-1)) + 0.1)
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            # Set to False
            attn_weight = torch.dropout(attn_weight, 0.00000000001, training)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        self._check_common(
            sfdp_pattern_7,
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )
        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        self._check_common(
            sfdp_pattern_7_v2,
            args,
            contains=False,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )

        args = (
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
        )
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_7),
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )
        args = (
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
        )
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_7_v2),
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )

    def _test_sdpa_rewriter_8(self):
        # [FUNCTION DOC] HELPER: Test body for permuted half attention without dropout. Checks the
        # first variant with the code check and the second with no code check, on the test device
        # and then on the GPU device with checkpoint-wrapped versions, atol 2e-3.
        def sfdp_pattern_8(query, key, value):
            # [FUNCTION DOC] HELPER: Permutes inputs, divides scores by the square root, casts to
            # float32, softmax, casts to half, matmul with the permuted value.
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        def sfdp_pattern_8_v2(query, key, value):
            # [FUNCTION DOC] HELPER: Same as the first variant but the divisor is the square root
            # plus 0.1.
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            div = q @ k.transpose(-2, -1) / (math.sqrt(q.size(-1)) + 0.1)
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        self._check_common(sfdp_pattern_8, args, atol=2e-3)
        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        self._check_common(sfdp_pattern_8_v2, args, atol=2e-3, contains=False)

        args = (
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
        )
        self._check_common(checkpoint_wrapper(sfdp_pattern_8), args, atol=2e-3)
        args = (
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
        )
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_8_v2),
            args,
            atol=2e-3,
            contains=False,
        )

    def _test_sdpa_rewriter_9(self):
        # [FUNCTION DOC] HELPER: Test body for permuted half attention where the query is pre-
        # scaled, with near zero dropout. Four runs (two variants, plain on the test device,
        # wrapped on the GPU device) with the code check requiring SM80 or later, dropout on,
        # and override_check_equal.
        def sfdp_pattern_9(query, key, value, training):
            # [FUNCTION DOC] HELPER: Permutes inputs, divides the query by the square root,
            # computes scores, casts to float32, softmax, near zero dropout, casts to half, matmul
            # with the permuted value.
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            q = q / math.sqrt(q.size(-1))
            div = q @ k.transpose(-2, -1)
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            # very low dropout to make test pass
            attn_weight = torch.dropout(attn_weight, 0.00000000001, training)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        def sfdp_pattern_9_v2(query, key, value, training):
            # [FUNCTION DOC] HELPER: Same as the first variant but the query divisor is the square
            # root plus 0.1.
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            q = q / (math.sqrt(q.size(-1)) + 0.1)
            div = q @ k.transpose(-2, -1)
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            # very low dropout to make test pass
            attn_weight = torch.dropout(attn_weight, 0.00000000001, training)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        self._check_common(
            sfdp_pattern_9,
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )
        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        self._check_common(
            sfdp_pattern_9_v2,
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )
        args = (
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
        )
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_9),
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )
        args = (
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
        )
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_9_v2),
            args,
            contains=SM80OrLater,
            has_dropout=True,
            override_check_equal=True,
            atol=2e-3,
        )

    def _test_sdpa_rewriter_10(self):
        # [FUNCTION DOC] HELPER: Test body for permuted half attention with a pre-scaled query and
        # no dropout. First variant is checked with the code check, second without, plain on the
        # test device and wrapped on the GPU device, with atol 2e-3.
        def sfdp_pattern_10(query, key, value):
            # [FUNCTION DOC] HELPER: Permutes inputs, divides the query by the square root,
            # computes scores, casts to float32, softmax, casts to half, matmul with the
            # permuted value.
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            q = q / math.sqrt(q.size(-1))
            div = q @ k.transpose(-2, -1)
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        def sfdp_pattern_10_v2(query, key, value):
            # [FUNCTION DOC] HELPER: Same as the first variant but the query divisor is the square
            # root plus 0.1.
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            q = q / (math.sqrt(q.size(-1)) + 0.1)
            div = q @ k.transpose(-2, -1)
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        self._check_common(sfdp_pattern_10, args, atol=2e-3)
        args = (
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=self.device, dtype=torch.half),
        )
        self._check_common(
            sfdp_pattern_10_v2,
            args,
            atol=2e-3,
            contains=False,
        )

        args = (
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
        )
        self._check_common(checkpoint_wrapper(sfdp_pattern_10), args, atol=2e-3)
        args = (
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
            torch.randn((2, 8, 4, 16), device=GPU_TYPE, dtype=torch.half),
        )
        self._check_common(
            checkpoint_wrapper(sfdp_pattern_10_v2),
            args,
            atol=2e-3,
            contains=False,
        )

    def _test_pattern_fails_with_tensor_factor(self):
        # [FUNCTION DOC] HELPER: Test body (issue 99124) for attention whose scale is a tensor
        # argument, not a constant. For both division and multiplication by the tensor, builds
        # random query, key and value of shape 2 by 4 by 4 by 4 and a scale tensor of shape 4 by 1
        # by 1, uses an eval-mode model, and runs the common checker with no code string check,
        # atol 1e-3 and without requiring the fusion counter to be positive (the rewrite is not
        # expected to apply).
        # https://github.com/pytorch/pytorch/issues/99124
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a boolean flag choosing whether the scale tensor
            # divides or multiplies the attention scores.
            def __init__(self, is_inv_factor):
                # [FUNCTION DOC] HELPER: Stores the flag selecting division (true) or
                # multiplication (false).
                super().__init__()
                self.is_inv_factor = is_inv_factor

            def forward(self, query, key, value, scale_factor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Detaches the scale tensor, computes scores from query and
                # transposed key, divides or multiplies by the scale per the flag, softmax over the
                # last dimension, and matmul with value.
                # Dividing by scale_factor makes scale_factor gradients very
                # unstable
                scale_factor = scale_factor.detach()
                y = torch.matmul(query, key.transpose(-2, -1))
                if self.is_inv_factor:
                    y = y.div(scale_factor)
                else:
                    y = y.mul(scale_factor)
                return y.softmax(dim=-1).matmul(value)

        tensor_shape = (2, 4, 4, 4)
        for is_inv_factor in [True, False]:
            args = [
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn((4, 1, 1), device=self.device),
            ]
            model = Model(is_inv_factor).eval()
            # The training path has an accuracy gap compared with eager mode.
            self._check_common(
                model, args1=args, contains=False, atol=1e-3, has_fuse_pattern=False
            )

    def _test_pattern_fails_with_unsupported_mask(self):
        # [FUNCTION DOC] HELPER: Test body (issue 100315) for attention masks the rewriter
        # cannot handle. Skipped for dynamic shape classes. Runs an eval-mode model with random
        # 2 by 4 by 4 by 4 inputs and each of two masks (an integer tensor and a Python float)
        # through the common checker with no code string check, atol 1e-4, and no requirement
        # that fusion happens.
        if not self.use_static_shapes:
            self.skipTest("Causes shape specialization. TODO: investigate")

        # https://github.com/pytorch/pytorch/issues/100315
        class Model(torch.nn.Module):
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Calls the base module constructor; no state.
                super().__init__()

            def forward(self, query, key, value, attn_mask) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Scores are query times transposed key divided by the
                # square root of the head size plus the mask, softmax over the last dimension, then
                # matmul with value.
                attn_weight = torch.softmax(
                    query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
                    + attn_mask,
                    dim=-1,
                )
                return attn_weight @ value

        tensor_shape = (2, 4, 4, 4)

        upsupported_masks = [
            torch.randn((2, 4, 4, 4), device=self.device).to(dtype=torch.int),
            2.0,
        ]
        for atte_mask in upsupported_masks:
            args = [
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                atte_mask,
            ]
            model = Model().eval()
            # The training path has an accuracy gap compared with eager mode.
            self._check_common(
                model, args1=args, contains=False, atol=1e-4, has_fuse_pattern=False
            )

    def _test_sdpa_rewriter_11(self):
        # [FUNCTION DOC] HELPER: Test body for division-scaled attention on inputs laid out as
        # batch, sequence, heads, embedding; runs the common checker with defaults.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Swaps dimensions 1 and 2 of q, k, v, then scaled dot product
            # attention with division by the square root of the key size, softmax, matmul.
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            return (
                torch.matmul(q, k.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(v)
            )

        self._check_common(dot_prod_attention)

    def _test_sdpa_rewriter_12(self):
        # [FUNCTION DOC] HELPER: Test body for the same layout with dropout 0.4; runs the common
        # checker with dropout on, no code string check, and expects the training-mode pattern
        # named for the twelfth pattern to be matched at least once in training mode.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Transposes q, k, v, computes division-scaled scores, softmax,
            # dropout with probability 0.4 controlled by the training flag, then matmul with the
            # transposed value.
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            attn_weight = torch.nn.functional.dropout(
                torch.matmul(q, k.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1),
                p=0.4,
                training=training,
                inplace=False,
            )
            return attn_weight.matmul(v)

        self._check_common(
            dot_prod_attention,
            contains=False,
            has_dropout=True,
            expected_fused_attention_patterns={True: "_sfdp_pattern_12_training"},
        )

    def _test_sdpa_prev_13(self):
        # [FUNCTION DOC] HELPER: Test body for scaled attention with an extra clone after softmax.
        # Runs the common checker in inference only, plain and checkpoint-wrapped.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Division-scaled attention with a clone of the softmax output
            # before the value matmul.
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .clone()
                .matmul(value)
            )

        self._check_common(dot_prod_attention, check_train=False)
        self._check_common(checkpoint_wrapper(dot_prod_attention), check_train=False)

    def _test_sdpa_prev_14(self):
        # [FUNCTION DOC] HELPER: Test body for multiplication-scaled attention with a clone after
        # softmax; inference only, plain and checkpoint-wrapped.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reciprocal-scale multiplication, softmax, clone, then
            # value matmul.
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .mul(1.0 / math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .clone()
                .matmul(value)
            )

        self._check_common(dot_prod_attention, check_train=False)
        self._check_common(checkpoint_wrapper(dot_prod_attention), check_train=False)

    def _test_sdpa_prev_15(self):
        # [FUNCTION DOC] HELPER: Test body for the transposed layout with a clone after softmax;
        # inference only, plain run only.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Transposes q, k, v, divides scores by the square root,
            # softmax, clone, then matmul with the transposed value.
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            return (
                torch.matmul(q, k.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .clone()
                .matmul(v)
            )

        self._check_common(dot_prod_attention, check_train=False)

    def _test_sdpa_rewriter_13(self, dtype):
        # [FUNCTION DOC] HELPER: Test body for batched matmul attention with dropout 0.5 in the
        # given dtype. Uses random 3-dimensional inputs of shape 4 by 8 by 16 and runs the
        # common checker in inference only with dropout on, forced equality check, and atol and
        # rtol of 1e-2.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Batched matmul of query with transposed key, softmax over the
            # last dimension, dropout 0.5 (training flag), batched matmul with value.
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
            attn_weight = torch.nn.functional.dropout(
                attn_weight, p=0.5, training=training
            )
            return torch.bmm(attn_weight, value)

        tensor_shape = (4, 8, 16)
        args = [
            torch.randn(tensor_shape, device=self.device, dtype=dtype),
            torch.randn(tensor_shape, device=self.device, dtype=dtype),
            torch.randn(tensor_shape, device=self.device, dtype=dtype),
        ]

        self._check_common(
            dot_prod_attention,
            check_train=False,
            args1=args,
            has_dropout=True,
            override_check_equal=True,
            atol=1e-2,
            rtol=1e-2,
        )

    def _test_sdpa_rewriter_13_non_transpose_permute(self, dtype):
        # [FUNCTION DOC] HELPER: Test body where the key is permuted to a non-transpose layout, so
        # the pattern should not match. Uses inputs of shape 8 by 3 by 4, 3 by 8 by 4 and 8 by 3 by
        # 4 in the given dtype, runs the checker (dropout on, no fusion requirement, inference,
        # forced equality) and then asserts the fuse_attention counter is zero.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Batched matmul of query with a permuted key, softmax, dropout
            # 0.5 (training flag), batched matmul with value.
            attn_weight = torch.bmm(query, key.permute(1, 2, 0)).softmax(dim=-1)
            attn_weight = torch.nn.functional.dropout(
                attn_weight, p=0.5, training=training
            )
            return torch.bmm(attn_weight, value)

        args = [
            torch.randn((8, 3, 4), device=self.device, dtype=dtype),
            torch.randn((3, 8, 4), device=self.device, dtype=dtype),
            torch.randn((8, 3, 4), device=self.device, dtype=dtype),
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_fuse_pattern=False,
            has_dropout=True,
            check_train=False,
            override_check_equal=True,
        )
        self.assertEqual(counters["inductor"]["fuse_attention"], 0)

    def _test_sdpa_rewriter_14(self):
        # [FUNCTION DOC] HELPER: Test body for causal-mask attention on a batch, sequence, heads
        # layout with divisor 3.0; runs the common checker with defaults.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Builds a lower triangular mask from the sequence lengths (zero
            # or negative infinity), permutes q, k, v, adds the mask to scores divided by 3.0,
            # softmax, matmul.
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            attn_mask = torch.ones(
                query.size(1), key.size(1), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            return (
                (torch.matmul(q, k.transpose(-2, -1)).div(3.0) + attn_mask)
                .softmax(dim=-1)
                .matmul(v)
            )

        self._check_common(dot_prod_attention)

    def _test_sdpa_rewriter_15(self):
        # [FUNCTION DOC] HELPER: Test body for key-padding style mask attention; inference only.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Transposes q, k, v, builds a lower triangular boolean mask of
            # batch by key length, scores divided by 3.0, inverts the mask, reshapes and expands
            # it to the scores shape, fills masked scores with negative infinity, softmax, matmul
            # with value.
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            bs = q.size(0)
            k_len = k.size(-2)
            attn_mask = torch.ones(
                bs, k_len, dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            scores = torch.matmul(q, k.transpose(-2, -1)) / 3.0
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
            scores = scores.masked_fill(attn_mask, -float("inf"))
            weights = torch.nn.functional.softmax(scores, dim=-1)
            return torch.matmul(weights, v)

        self._check_common(dot_prod_attention, check_train=False)

    def _test_sdpa_rewriter_16(
        self,
        check_train=True,
        override_check_equal=False,
    ):
        # [FUNCTION DOC] HELPER: Test body for causal-mask attention with dropout 0.4 and a cast
        # back to the query dtype. The check_train and override_check_equal arguments are forwarded
        # to the checker. Runs the default shape and a batch size one input of shape 1 by 2 by 16
        # by 32 (the graph differs), each with dropout on and no code string check.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Causal mask (zero or negative infinity) added to scores
            # divided by 3.0 after permuting q, k, v, softmax, dropout 0.4 (training flag), cast to
            # query dtype, matmul.
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            attn_mask = torch.ones(
                query.size(1), key.size(1), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            return (
                torch.nn.functional.dropout(
                    (torch.matmul(q, k.transpose(-2, -1)).div(3.0) + attn_mask).softmax(
                        dim=-1
                    ),
                    p=0.4,
                    training=training,
                    inplace=False,
                )
                .to(dtype=query.dtype)
                .matmul(v)
            )

        self._check_common(
            dot_prod_attention,
            contains=False,
            has_dropout=True,
            check_train=check_train,
            override_check_equal=override_check_equal,
        )

        # also check batch_size=1 because the graph is slightly different
        tensor_shape = (1, 2, 16, 32)
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_dropout=True,
            check_train=check_train,
            override_check_equal=override_check_equal,
        )

    def _test_sdpa_rewriter_16_fp32_mask(
        self,
        check_train=True,
        override_check_equal=False,
    ):
        # [FUNCTION DOC] HELPER: Same as the previous test body but the mask is a lower
        # triangular part of a random float tensor rather than a boolean-derived infinity mask.
        # Arguments are forwarded.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Same attention as the previous variant but with the random
            # float mask.
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            attn_mask = torch.randn(
                query.size(1), key.size(1), dtype=torch.float, device=query.device
            ).tril(diagonal=0)
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            return (
                torch.nn.functional.dropout(
                    (torch.matmul(q, k.transpose(-2, -1)).div(3.0) + attn_mask).softmax(
                        dim=-1
                    ),
                    p=0.4,
                    training=training,
                    inplace=False,
                )
                .to(dtype=query.dtype)
                .matmul(v)
            )

        self._check_common(
            dot_prod_attention,
            contains=False,
            has_dropout=True,
            check_train=check_train,
            override_check_equal=override_check_equal,
        )

        # also check batch_size=1 because the graph is slightly different
        tensor_shape = (1, 2, 16, 32)
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_dropout=True,
            check_train=check_train,
            override_check_equal=override_check_equal,
        )

    def _test_sdpa_rewriter_17(self):
        # [FUNCTION DOC] HELPER: Test body for key-padding mask attention with dropout 0.4;
        # inference only, dropout on.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Transposed layout, padding mask expanded to scores shape, fill
            # with negative infinity, softmax, dropout 0.4 (training flag), matmul with value.
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            bs = q.size(0)
            k_len = k.size(-2)
            attn_mask = torch.ones(
                bs, k_len, dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            scores = torch.matmul(q, k.transpose(-2, -1)) / 3.0
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
            scores = scores.masked_fill(attn_mask, -float("inf"))
            weights = torch.nn.functional.softmax(scores, dim=-1)
            weights = torch.nn.functional.dropout(
                weights,
                p=0.4,
                training=training,
                inplace=False,
            )
            return torch.matmul(weights, v)

        self._check_common(dot_prod_attention, check_train=False, has_dropout=True)

    def _test_sdpa_rewriter_18(self):
        # [FUNCTION DOC] HELPER: Test body for GPT2 style attention with a precomputed causal mask.
        # Runs both variants for batch sizes 4 and 1 (inference, no code string check); the second
        # variant uses atol 2e-3.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            causal_mask: torch.Tensor,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Permutes q, k, v, divides scores by a zero-dimensional tensor
            # holding the square root of the embedding size, replaces masked positions with the
            # dtype minimum, softmax, a dropout with probability 0.0, matmul; returns the output
            # with permuted key and value.
            # for hf_GPT2 with dropout
            query = query.permute([0, 2, 1, 3])
            key = key.permute([0, 2, 1, 3])
            value = value.permute([0, 2, 1, 3])
            attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
            inv_scale = torch.full(
                (), math.sqrt(value.size(-1)), dtype=query.dtype, device=query.device
            )
            attn_weights = attn_weights.div(inv_scale)
            causal_mask_value = torch.full(
                (), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device
            )
            attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
            return (
                (
                    torch.nn.functional.dropout(
                        attn_weights.softmax(dim=-1), 0.0
                    ).matmul(value)
                ),
                key.permute([0, 2, 1, 3]),
                value.permute([0, 2, 1, 3]),
            )

        def dot_prod_attention_v2(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            causal_mask: torch.Tensor,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Same as the previous variant but the scale tensor holds the
            # square root plus 0.1.
            query = query.permute([0, 2, 1, 3])
            key = key.permute([0, 2, 1, 3])
            value = value.permute([0, 2, 1, 3])
            attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
            inv_scale = torch.full(
                (),
                math.sqrt(value.size(-1)) + 0.1,
                dtype=query.dtype,
                device=query.device,
            )
            attn_weights = attn_weights.div(inv_scale)
            causal_mask_value = torch.full(
                (), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device
            )
            attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
            return (
                (
                    torch.nn.functional.dropout(
                        attn_weights.softmax(dim=-1), 0.0
                    ).matmul(value)
                ),
                key.permute([0, 2, 1, 3]),
                value.permute([0, 2, 1, 3]),
            )

        tensor_shape = (4, 2, 16, 32)
        causal_mask = torch.ones(2, 2, dtype=torch.bool, device=self.device).tril(
            diagonal=0
        )
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            causal_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_dropout=False,
            check_train=False,
        )

        self._check_common(
            dot_prod_attention_v2,
            args1=args,
            atol=2e-3,
            contains=False,
            has_dropout=False,
            check_train=False,
        )

        # also check batch_size=1 because the graph is slightly different
        tensor_shape = (1, 2, 16, 32)
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            causal_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_dropout=False,
            check_train=False,
        )
        self._check_common(
            dot_prod_attention_v2,
            args1=args,
            contains=False,
            atol=2e-3,
            has_dropout=False,
            check_train=False,
        )

    def _test_sdpa_rewriter_19(self):
        # [FUNCTION DOC] HELPER: Test body for attention with both a causal mask and an additive
        # attention mask, followed by dropout 0.4. Uses 16 by 16 boolean causal mask and random
        # float mask; checks both variants with dropout on, inference, no code string check.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            causal_mask: torch.Tensor,
            attn_mask: torch.Tensor,
            training,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Scores divided by a tensor scale, masked via where with the
            # dtype minimum, added attention mask, softmax cast to value dtype, dropout 0.4, matmul
            # with value.
            attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
            inv_scale = torch.full(
                (),
                math.sqrt(value.size(-1)),
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
            attn_weights = attn_weights.div(inv_scale)
            causal_mask_value = torch.full(
                (), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device
            )
            attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
            attn_weights = attn_weights + attn_mask
            attn_weights = attn_weights.softmax(dim=-1).type(value.dtype)
            return torch.nn.functional.dropout(
                attn_weights,
                p=0.4,
                training=training,
                inplace=False,
            ).matmul(value)

        def dot_prod_attention_v2(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            causal_mask: torch.Tensor,
            attn_mask: torch.Tensor,
            training,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Same as the previous variant but the scale is the square root
            # plus 0.1.
            attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
            inv_scale = torch.full(
                (),
                math.sqrt(value.size(-1)) + 0.1,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
            attn_weights = attn_weights.div(inv_scale)
            causal_mask_value = torch.full(
                (), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device
            )
            attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
            attn_weights = attn_weights + attn_mask
            attn_weights = attn_weights.softmax(dim=-1).type(value.dtype)
            return torch.nn.functional.dropout(
                attn_weights,
                p=0.4,
                training=training,
                inplace=False,
            ).matmul(value)

        tensor_shape = (4, 2, 16, 32)
        causal_mask = torch.ones(16, 16, dtype=torch.bool, device=self.device).tril(
            diagonal=0
        )
        attn_mask = torch.randn((16, 16), dtype=torch.float, device=self.device)
        args = [
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            torch.randn(tensor_shape, device=self.device),
            causal_mask,
            attn_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_dropout=True,
            check_train=False,
        )
        self._check_common(
            dot_prod_attention_v2,
            args1=args,
            contains=False,
            has_dropout=True,
            check_train=False,
        )

    def _test_sdpa_rewriter_20(self):
        # [FUNCTION DOC] HELPER: Test body for key-padding mask attention with a pre-scaled query
        # and dropout 0.4; inference only, dropout on.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, training
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Transposes q, k, v, divides the query by the square root,
            # computes scores, applies a padding mask expanded to the scores shape, softmax,
            # dropout 0.4, matmul.
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            bs = q.size(0)
            k_len = k.size(-2)
            q = q / math.sqrt(q.size(-1))
            scores = torch.matmul(q, k.transpose(-2, -1))
            attn_mask = torch.ones(
                bs, k_len, dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
            scores = scores.masked_fill(attn_mask, -float("inf"))
            weights = torch.nn.functional.softmax(scores, dim=-1)
            weights = torch.nn.functional.dropout(
                weights,
                p=0.4,
                training=training,
                inplace=False,
            )
            return torch.matmul(weights, v)

        self._check_common(dot_prod_attention, check_train=False, has_dropout=True)

    def _test_sdpa_rewriter_21(self):
        # [FUNCTION DOC] HELPER: Test body for additive mask attention (in-place add of a mask,
        # softmax in float). Loops over batch sizes 4 and 1 with a float mask of shape 1 by 1 by 1
        # by 2; inference only; tolerances are atol 1e-3 and rtol 0.2, or 3e-3 and 1e-2 on ROCm.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Transposes q, k, v, adds the mask in place to scores, softmax
            # in float cast back to the value dtype, matmul with value.
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))
            scores += attn_mask
            attn_weights = scores.float().softmax(dim=-1).type(value.dtype)
            return attn_weights.matmul(value)

        atol = 1e-3
        rtol = 0.2
        if TEST_WITH_ROCM:
            atol = 3e-3
            rtol = 1e-2

        tensor_shapes = [(4, 2, 16, 32), (1, 2, 16, 32)]
        for tensor_shape in tensor_shapes:
            attn_mask = torch.randn((1, 1, 1, 2), dtype=torch.float, device=self.device)
            args = [
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                attn_mask,
            ]
            self._check_common(
                dot_prod_attention,
                args1=args,
                has_dropout=False,
                check_train=False,
                atol=atol,
                rtol=rtol,
            )

    def _test_sdpa_rewriter_22(self):
        # [FUNCTION DOC] HELPER: Test body like the previous one but the function also returns key
        # and value, and the mask has shape 1 by 1 by 2 by 2. For each batch size it checks the
        # mask as given and then the mask with its last two dimensions swapped, where the code
        # string check applies only on CPU.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Transposes q, k, v, adds the mask in place to scores,
            # softmax in float cast to the value dtype, returns the attention output together
            # with key and value.
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))
            scores += attn_mask
            attn_weights = scores.float().softmax(dim=-1).type(value.dtype)
            return attn_weights.matmul(value), key, value

        tensor_shapes = [(4, 2, 16, 32), (1, 2, 16, 32)]
        for tensor_shape in tensor_shapes:
            attn_mask = torch.randn((1, 1, 2, 2), dtype=torch.float, device=self.device)
            args = [
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                attn_mask,
            ]
            self._check_common(
                dot_prod_attention,
                args1=args,
                has_dropout=False,
                check_train=False,
            )
            # test attn_mask with stride of last dim != 1
            attn_mask_ = attn_mask.transpose(2, 3)
            args[3] = attn_mask_
            self._check_common(
                dot_prod_attention,
                args1=args,
                has_dropout=False,
                check_train=False,
                contains=self.device == "cpu",
            )

    def _test_sdpa_rewriter_23(self):
        # [FUNCTION DOC] HELPER: Test body where the mask is a constant created inside the
        # function; for batch sizes 4 and 1, runs inference only with default checks.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Builds a zero mask constant, transposes q, k, v, adds it to
            # scores, softmax in float cast to the value dtype, returns output with key and value.
            attn_mask = torch.full((1, 1, 1, 2), 0.0, device=query.device)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))
            scores += attn_mask
            attn_weights = scores.float().softmax(dim=-1).type(value.dtype)
            return attn_weights.matmul(value), key, value

        tensor_shapes = [(4, 2, 16, 32), (1, 2, 16, 32)]
        for tensor_shape in tensor_shapes:
            args = [
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
                torch.randn(tensor_shape, device=self.device),
            ]
            self._check_common(
                dot_prod_attention,
                args1=args,
                has_dropout=False,
                check_train=False,
            )

    def _test_sdpa_rewriter_24(self):
        # [FUNCTION DOC] HELPER: Test body for attention in a merged batch and heads form using
        # batched matmul. Uses float inputs of shape 4 by 2 by 16 by 32 and a mask of shape 1 by 1
        # by 16 by 16; inference only.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reshapes q, k, v to a batch-heads by sequence by embedding
            # layout, batched matmul for scores, restores the four-dimensional view to add the
            # mask, merges again, softmax, batched matmul with value and reshapes the output back.
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            bs = query.size(0)
            n_head = query.size(1)
            seq_len = query.size(2)
            embed_dim = query.size(3)
            q = query.view(bs * n_head, seq_len, embed_dim)
            k = key.reshape(bs * n_head, seq_len, embed_dim)
            v = value.reshape(bs * n_head, seq_len, embed_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            attn_weights = attn_weights.view(bs, n_head, seq_len, seq_len) + attn_mask
            attn_weights = attn_weights.view(bs * n_head, seq_len, seq_len)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.view(bs, n_head, seq_len, embed_dim)
            return attn_output

        tensor_shape = (4, 2, 16, 32)
        attn_mask = torch.randn((1, 1, 16, 16), dtype=torch.float, device=self.device)
        args = [
            torch.randn(tensor_shape, device=self.device, dtype=torch.float),
            torch.randn(tensor_shape, device=self.device, dtype=torch.float),
            torch.randn(tensor_shape, device=self.device, dtype=torch.float),
            attn_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            has_dropout=False,
            check_train=False,
        )

    def _test_sdpa_rewriter_25(self):
        # [FUNCTION DOC] HELPER: Test body for half precision mask attention with dropout 0.1. Uses
        # half tensors of shape 4 by 2 by 16 by 32 and a half mask, running the checker with
        # training on.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Transposes q, k, v, adds mask in place, softmax in float cast
            # to the value dtype, dropout 0.1 (training flag), matmul with value.
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))
            scores += attn_mask
            attn_weights = scores.float().softmax(dim=-1).type(value.dtype)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=0.1, training=training
            )
            return attn_weights.matmul(value)

        tensor_shape = (4, 2, 16, 32)
        attn_mask = torch.randn((1, 1, 1, 2), dtype=torch.half, device=self.device)
        args = [
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            attn_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            has_dropout=True,
            check_train=True,
        )

    @torch._inductor.config.patch("cache_sdpa_constraint", True)
    def _test_cache_sdpa_constraint_shared_kv_enabled(self):
        # [FUNCTION DOC] HELPER: Runs with the cache_sdpa_constraint config on. Compiles an
        # attention call where the same tensor is both key and value, after clearing counters and
        # resetting dynamo, compares with eager (atol 1e-3, rtol 0.2) and returns how many times
        # the substring copy_ appears in the generated code.
        """When cache_sdpa_constraint is True and the same tensor feeds both key
        and value in SDPA (PMA pattern), the constraint cache should deduplicate
        the buffer copy, producing fewer copy operations than when disabled."""

        def pma_attention(query, kv):
            # [FUNCTION DOC] HELPER: Calls the built-in scaled dot product attention with a shared
            # key and value tensor.
            return torch.nn.functional.scaled_dot_product_attention(query, kv, kv)

        tensor_shape = (4, 2, 16, 32)
        dtype = torch.float
        query = torch.randn(tensor_shape, device=self.device, dtype=dtype)
        kv = torch.randn(tensor_shape, device=self.device, dtype=dtype)

        counters.clear()
        torch._dynamo.reset()
        result_eager = pma_attention(query, kv)
        result_compiled, (source_code,) = run_and_get_code(
            torch.compile(pma_attention, fullgraph=True),
            query.clone(),
            kv.clone(),
        )
        self.assertEqual(result_eager, result_compiled, atol=1e-3, rtol=0.2)
        copy_count_cached = source_code.count("copy_")
        return copy_count_cached

    @torch._inductor.config.patch("cache_sdpa_constraint", False)
    def _test_cache_sdpa_constraint_shared_kv_disabled(self):
        # [FUNCTION DOC] HELPER: Same as the enabled variant but with the cache_sdpa_constraint
        # config off; returns the occurrence count of copy_ in the generated code.
        """When cache_sdpa_constraint is False and the same tensor feeds both key
        and value in SDPA (PMA pattern), the constraint should create separate
        buffer copies, producing more copy operations than when enabled."""

        def pma_attention(query, kv):
            # [FUNCTION DOC] HELPER: Calls the built-in scaled dot product attention with a shared
            # key and value tensor.
            return torch.nn.functional.scaled_dot_product_attention(query, kv, kv)

        tensor_shape = (4, 2, 16, 32)
        dtype = torch.float
        query = torch.randn(tensor_shape, device=self.device, dtype=dtype)
        kv = torch.randn(tensor_shape, device=self.device, dtype=dtype)

        counters.clear()
        torch._dynamo.reset()
        result_eager = pma_attention(query, kv)
        result_compiled, (source_code,) = run_and_get_code(
            torch.compile(pma_attention, fullgraph=True),
            query.clone(),
            kv.clone(),
        )
        self.assertEqual(result_eager, result_compiled, atol=1e-3, rtol=0.2)
        copy_count_uncached = source_code.count("copy_")
        return copy_count_uncached

    def _test_cache_sdpa_constraint_shared_kv(self):
        # [FUNCTION DOC] HELPER: Runs both cache variants and asserts the count with caching is
        # less than or equal to the count without it, with a custom failure message.
        """Verify that cache_sdpa_constraint=True produces no more copy_
        operations than cache_sdpa_constraint=False when the same tensor feeds
        both key and value in SDPA."""
        copy_count_cached = self._test_cache_sdpa_constraint_shared_kv_enabled()
        copy_count_uncached = self._test_cache_sdpa_constraint_shared_kv_disabled()
        self.assertLessEqual(
            copy_count_cached,
            copy_count_uncached,
            lambda msg: f"{msg}\nExpected caching to produce <= copy_ ops (got {copy_count_cached}) "
            f"vs no caching ({copy_count_uncached})",
        )

    def _test_sdpa_rewriter_26(self):
        # [FUNCTION DOC] HELPER: Test body for half mask attention with dropout 0.1 where the
        # function also returns key and value; training enabled.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Transposes q, k, v, in-place mask add, float softmax cast to
            # value dtype, dropout 0.1 (training flag), returns output with key and value.
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))
            scores += attn_mask
            attn_weights = scores.float().softmax(dim=-1).type(value.dtype)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=0.1, training=training
            )

            return attn_weights.matmul(value), key, value

        tensor_shape = (4, 2, 16, 32)
        attn_mask = torch.randn((1, 1, 1, 2), dtype=torch.half, device=self.device)
        args = [
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            attn_mask,
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            has_dropout=True,
            check_train=True,
        )

    def _test_sdpa_rewriter_27(self):
        # [FUNCTION DOC] HELPER: Test body for half attention whose mask is a constant built inside
        # the function, with dropout 0.1, key and value returned, and training enabled.
        def dot_prod_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Builds a half zero mask, transposes q, k, v, adds the mask
            # to scores, float softmax cast to value dtype, dropout 0.1, returns output with key
            # and value.
            attn_mask = torch.full(
                (1, 1, 1, 2), 0.0, dtype=torch.half, device=query.device
            )
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scores = torch.matmul(query, key.permute(0, 1, 3, 2))
            scores += attn_mask
            attn_weights = scores.float().softmax(dim=-1).type(value.dtype)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=0.1, training=training
            )
            return attn_weights.matmul(value), key, value

        tensor_shape = (4, 2, 16, 32)
        args = [
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            has_dropout=True,
            check_train=True,
        )

    def _test_sdpa_rewriter_29(self):
        # [FUNCTION DOC] HELPER: Test body for attention using aten scalar multiplies and the safe
        # softmax op with a zero additive mask. For half and float, checks batch size 2 (with
        # training) then batch size 1 (inference only).
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Permutes q, k, v, scales q and transposed k each by the
            # reciprocal square root of the head size, adds a zero mask, safe softmax, matmul
            # with value.
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            scale = 1.0 / query.size(-1) ** 0.5
            q = torch.ops.aten.mul.Scalar(q, scale)
            k_t = torch.ops.aten.mul.Scalar(k.transpose(-2, -1), scale)
            attn_mask = torch.zeros(
                1, 1, query.size(1), key.size(1), device=query.device, dtype=query.dtype
            )
            attn_weight = (q @ k_t) + attn_mask
            attn_weight = torch.ops.aten._safe_softmax(attn_weight, -1)
            return attn_weight @ v

        for dtype in [torch.half, torch.float]:
            # batch_size > 1 (generates clone nodes), also exercise training.
            tensor_shape = (2, 8, 4, 16)
            args = [
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
            ]
            self._check_common(
                dot_prod_attention, args1=args, dtype=dtype, check_train=True
            )

            # batch_size = 1 (no clone)
            tensor_shape_bs1 = (1, 8, 4, 16)
            args_bs1 = [
                torch.randn(tensor_shape_bs1, device=self.device, dtype=dtype),
                torch.randn(tensor_shape_bs1, device=self.device, dtype=dtype),
                torch.randn(tensor_shape_bs1, device=self.device, dtype=dtype),
            ]
            self._check_common(
                dot_prod_attention, args1=args_bs1, dtype=dtype, check_train=False
            )

    def _test_sdpa_rewriter_30(self):
        # [FUNCTION DOC] HELPER: Same as the previous test body but without the additive mask.
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Permutes q, k, v, scales q and transposed k each by the
            # reciprocal square root of the head size, safe softmax, matmul with value.
            """Input tensors assumed to have shape (batch_size, seq_len, n_head, embed_dim)"""
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            scale = 1.0 / query.size(-1) ** 0.5
            q = torch.ops.aten.mul.Scalar(q, scale)
            k_t = torch.ops.aten.mul.Scalar(k.transpose(-2, -1), scale)
            attn_weight = q @ k_t
            attn_weight = torch.ops.aten._safe_softmax(attn_weight, -1)
            return attn_weight @ v

        for dtype in [torch.half, torch.float]:
            # batch_size > 1 (generates clone nodes), also exercise training.
            tensor_shape = (2, 8, 4, 16)
            args = [
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
                torch.randn(tensor_shape, device=self.device, dtype=dtype),
            ]
            self._check_common(
                dot_prod_attention, args1=args, dtype=dtype, check_train=True
            )

            # batch_size = 1 (no clone)
            tensor_shape_bs1 = (1, 8, 4, 16)
            args_bs1 = [
                torch.randn(tensor_shape_bs1, device=self.device, dtype=dtype),
                torch.randn(tensor_shape_bs1, device=self.device, dtype=dtype),
                torch.randn(tensor_shape_bs1, device=self.device, dtype=dtype),
            ]
            self._check_common(
                dot_prod_attention, args1=args_bs1, dtype=dtype, check_train=False
            )

    def _test_sdpa_rewriter_28(self):
        # [FUNCTION DOC] HELPER: Test body for attention on a packed qkv tensor of half type
        # with shape 2 by 3 by 4 by 16 by 8, with dropout 0.1 and training enabled, no code
        # string check.
        def dot_prod_attention(
            qkv: torch.Tensor,
            training: bool,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Permutes the packed tensor and unbinds it into q, k, v, scores
            # scaled by 0.2, softmax, dropout 0.1 (training flag), matmul with v.
            q, k, v = qkv.permute(1, 0, 2, 4, 3).unbind(0)
            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores.mul(0.2)
            attn_weights = scores.softmax(dim=-1)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=0.1, training=training
            )
            return attn_weights.matmul(v)

        tensor_shape = (2, 3, 4, 16, 8)
        args = [
            torch.randn(tensor_shape, dtype=torch.half, device=self.device),
        ]
        self._check_common(
            dot_prod_attention,
            args1=args,
            contains=False,
            has_dropout=True,
            check_train=True,
        )


class TestSDPAPatternRegistration(TestCase):
    # [CLASS DOC] Tests that the registered SDPA patterns are consistent with training mode.
    def test_inference_only_patterns_not_registered_for_training(self):
        # [FUNCTION DOC] Checks the set of inference-only SDPA patterns in the fuse_attention
        # module equals a fixed list of ten names, that no registered CPU pattern with a training
        # suffix has one of those names as prefix, and that each of them has a registered
        # inference variant.
        from torch._inductor.fx_passes import fuse_attention

        inference_only_patterns = [
            "_sfdp_pattern_13",
            "_sfdp_pattern_15",
            "_sfdp_pattern_17",
            "_sfdp_pattern_18",
            "_sfdp_pattern_19",
            "_sfdp_pattern_20",
            "_sfdp_pattern_21",
            "_sfdp_pattern_22",
            "_sfdp_pattern_23",
            "_sfdp_pattern_24",
        ]
        self.assertEqual(
            inference_only_patterns,
            sorted(fuse_attention._INFERENCE_ONLY_SFDP_PATTERNS),
        )

        registered_names = [
            name for name, _ in fuse_attention._get_sfdp_patterns(torch.device("cpu"))
        ]
        unexpected_training_names = sorted(
            name
            for name in registered_names
            if name.endswith("_training")
            and any(
                name.startswith(f"{pattern_name}_")
                for pattern_name in inference_only_patterns
            )
        )
        self.assertEqual([], unexpected_training_names)

        missing_inference_names = sorted(
            pattern_name
            for pattern_name in inference_only_patterns
            if not any(
                name.endswith("_inference") and name.startswith(f"{pattern_name}_")
                for name in registered_names
            )
        )
        self.assertEqual([], missing_inference_names)


if HAS_XPU_AND_TRITON or (HAS_CUDA_AND_TRITON and PLATFORM_SUPPORTS_FUSED_ATTENTION):

    class SDPAPatternRewriterGpuTests(TestSDPAPatternRewriterTemplate):
        # [CLASS DOC] GPU test class (defined only when XPU or CUDA with Triton and fused attention
        # support exists). Binds template bodies to public tests on the GPU device, with ROCm, CUDA
        # and XPU specific selections of the 16 and 25 to 27 variants.
        device = GPU_TYPE
        test_sdpa_rewriter_1_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_1
        test_sdpa_rewriter_1_freezing = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_1_freezing
        )
        test_insignificant_strides = (
            TestSDPAPatternRewriterTemplate._test_insignificant_strides
        )
        test_pattern_fails_with_reuse_gpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_reuse
        )
        test_sdpa_rewriter_2_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_2
        test_sdpa_rewriter_3_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_3
        test_sdpa_rewriter_4_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_4
        test_sdpa_rewriter_5_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_5
        test_sdpa_rewriter_6_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_6
        test_sdpa_rewriter_7_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_7
        test_sdpa_rewriter_8_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_8
        test_sdpa_rewriter_9_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_9
        test_sdpa_rewriter_10_gpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_10
        )
        test_pattern_fails_with_tensor_factor_gpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_tensor_factor
        )
        test_pattern_fails_with_unsupported_mask_gpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_unsupported_mask
        )
        test_sdpa_rewriter_11_gpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_11
        )
        test_sdpa_rewriter_12_gpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_12
        )
        test_sdpa_prev_13_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_13
        test_sdpa_prev_14_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_14
        test_sdpa_prev_15_gpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_15
        test_sdpa_rewriter_13_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_13, dtype=torch.half
        )
        test_sdpa_rewriter_14_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_14
        )
        test_sdpa_rewriter_15_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_15
        )
        if TEST_WITH_ROCM:
            test_sdpa_rewriter_16_gpu = functools.partialmethod(
                TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_16
            )
            test_sdpa_rewriter_16_fp32_mask_gpu = functools.partialmethod(
                TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_16_fp32_mask
            )
        elif GPU_TYPE == "cuda":
            test_sdpa_rewriter_16_inference_gpu = functools.partialmethod(
                TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_16,
                check_train=False,
                override_check_equal=True,
            )
        test_sdpa_rewriter_17_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_17
        )
        test_sdpa_rewriter_18_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_18
        )
        test_sdpa_rewriter_19_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_19
        )
        test_sdpa_rewriter_20_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_20
        )
        test_sdpa_rewriter_21_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_21
        )
        test_sdpa_rewriter_22_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_22
        )
        test_sdpa_rewriter_23_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_23
        )
        test_sdpa_rewriter_24_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_24
        )
        test_cache_sdpa_constraint_shared_kv_gpu = (
            TestSDPAPatternRewriterTemplate._test_cache_sdpa_constraint_shared_kv
        )
        test_sdpa_rewriter_28_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_28
        )
        test_sdpa_rewriter_29_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_29
        )
        test_sdpa_rewriter_30_gpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_30
        )
        if HAS_XPU_AND_TRITON:
            test_sdpa_rewriter_25_gpu = functools.partialmethod(
                TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_25
            )
            test_sdpa_rewriter_26_gpu = functools.partialmethod(
                TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_26
            )
            test_sdpa_rewriter_27_gpu = functools.partialmethod(
                TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_27
            )

        @skipIfXpu(msg="FIXME: ENable for XPU")
        def test_skip_non_tf32(self):
            # [FUNCTION DOC] Skipped on XPU. With the CUDA matmul fp32 precision set to ieee,
            # compiles a model of three linear layers feeding attention (division by a stored
            # factor, mask add, softmax) on CUDA and asserts the generated code has no scaled dot
            # product use and the result matches eager. Restores the precision setting afterwards.
            try:
                orig = torch.backends.cuda.matmul.fp32_precision
                torch.backends.cuda.matmul.fp32_precision = "ieee"

                class Model(torch.nn.Module):
                    # [CLASS DOC] HELPER: Module with query, key and value linear layers of width
                    # 64 and a stored scale factor.
                    def __init__(self):
                        # [FUNCTION DOC] HELPER: Creates the three linear layers and stores
                        # the factor.
                        super().__init__()
                        self.inv_scale = 1.0 / 8**0.5
                        self.query = torch.nn.Linear(64, 64)
                        self.key = torch.nn.Linear(64, 64)
                        self.value = torch.nn.Linear(64, 64)

                    def forward(self, x1, attn_mask):
                        # [FUNCTION DOC] HELPER: Projects the input, permutes, computes scores,
                        # divides by the stored factor, adds the mask, softmax and multiplies by
                        # the value projection.
                        q = self.query(x1).permute([0, 2, 1, 3])
                        k = self.key(x1).permute([0, 2, 1, 3])
                        v = self.value(x1).permute([0, 2, 1, 3])
                        t1 = torch.matmul(q, k.transpose(-2, -1))
                        t2 = t1.div(self.inv_scale)
                        t3 = t2 + attn_mask
                        t4 = t3.softmax(dim=-1)
                        t5 = t4.matmul(v)
                        return t5

                func = Model().to("cuda")
                x1 = torch.randn(1, 16, 64, 64, device="cuda")
                attn_mask = torch.zeros(1, 1, 16, 16, device="cuda")
                test_inputs = [x1, attn_mask]

                out, code = run_and_get_code(torch.compile(func), *test_inputs)
                FileCheck().check_not("scaled_dot_product").run(code[0])
                self.assertEqual(out, func(*test_inputs))

            finally:
                torch.backends.cuda.matmul.fp32_precision = orig

    class SDPAPatternRewriterGpuDynamicTests(SDPAPatternRewriterGpuTests):
        # [CLASS DOC] GPU test class variant with dynamic shapes: sets the flag so the common
        # checker marks the leading dimension of inputs dynamic.
        use_static_shapes = False


if HAS_CPU:

    class SDPAPatternRewriterCpuTests(TestSDPAPatternRewriterTemplate):
        # [CLASS DOC] CPU test class (defined when CPU support exists). Binds template bodies to
        # public tests on the CPU device, using float32 for the dtype-parameterized test.
        device = "cpu"
        test_sdpa_rewriter_1_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_1
        test_pattern_fails_with_reuse_cpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_reuse
        )
        test_sdpa_rewriter_2_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_2
        test_sdpa_rewriter_5_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_5
        test_pattern_fails_with_tensor_factor_cpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_tensor_factor
        )
        test_pattern_fails_with_unsupported_mask_cpu = (
            TestSDPAPatternRewriterTemplate._test_pattern_fails_with_unsupported_mask
        )
        test_sdpa_rewriter_11_cpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_11
        )
        test_sdpa_rewriter_12_cpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_12
        )
        test_sdpa_prev_13_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_13
        test_sdpa_prev_14_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_14
        test_sdpa_prev_15_cpu = TestSDPAPatternRewriterTemplate._test_sdpa_prev_15
        test_sdpa_rewriter_13_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_13, dtype=torch.float32
        )
        test_sdpa_rewriter_14_cpu = (
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_14
        )
        test_sdpa_rewriter_15_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_15
        )
        test_sdpa_rewriter_16_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_16
        )
        test_sdpa_rewriter_16_fp32_mask_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_16_fp32_mask
        )
        test_sdpa_rewriter_17_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_17
        )
        test_sdpa_rewriter_18_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_18
        )
        test_sdpa_rewriter_19_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_19
        )
        test_sdpa_rewriter_20_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_20
        )
        test_sdpa_rewriter_21_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_21
        )
        test_sdpa_rewriter_22_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_22
        )
        test_sdpa_rewriter_23_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_23
        )
        test_sdpa_rewriter_24_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_24
        )
        test_cache_sdpa_constraint_shared_kv_cpu = (
            TestSDPAPatternRewriterTemplate._test_cache_sdpa_constraint_shared_kv
        )
        test_sdpa_rewriter_28_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_28
        )
        test_sdpa_rewriter_29_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_29
        )
        test_sdpa_rewriter_30_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_30
        )

    class SDPAPatternRewriterCpuDynamicTests(SDPAPatternRewriterCpuTests):
        # [CLASS DOC] CPU test class variant with dynamic shapes.
        use_static_shapes = False

    class SDPAPatternRewriterPatternGuardCpuTests(TestSDPAPatternRewriterTemplate):
        # [CLASS DOC] CPU test class for the non-transpose permute pattern guard check only,
        # using float32.
        device = "cpu"
        test_sdpa_rewriter_13_non_transpose_permute_cpu = functools.partialmethod(
            TestSDPAPatternRewriterTemplate._test_sdpa_rewriter_13_non_transpose_permute,
            dtype=torch.float32,
        )


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
