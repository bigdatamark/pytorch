# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for fx_traceback.annotate: custom metadata attached to FX
# nodes is verified on the dynamo graph and on AOT forward/backward graphs (recorded by
# AotEagerAndRecordGraphs) via expected-inline strings, covering context-manager and
# decorator use, activation checkpointing, graph breaks, runtime asserts and flex
# attention. The device-generic class is instantiated for cuda and xpu at import time.

import warnings

import torch
import torch._dynamo.test_case
import torch.fx.traceback as fx_traceback
import torch.utils.checkpoint
from torch._dynamo.test_case import run_tests
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch.nn.attention.flex_attention import (
    _dense_to_ordered,
    create_block_mask,
    flex_attention,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED,
    IS_FLEX_ATTENTION_XPU_PLATFORM_SUPPORTED,
    skipCUDAIf,
    skipXPUIf,
)


def checkpoint_wrapper(fn):
# [FUNCTION DOC] HELPER: Decorator that returns a wrapper calling the checkpoint utility
# on the given function with reentrant checkpointing, forwarding positional args only.
    def inner(*args):
    # [FUNCTION DOC] HELPER: Runs the wrapped function under reentrant activation
    # checkpointing and returns its result.
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

    return inner


class AnnotateTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that annotate metadata propagates to compiled forward and backward
    # graphs and survives checkpointing, decorators and runtime assertions.
    def test_annotations(self):
        # [FUNCTION DOC] Compiles a module with nested annotate contexts using the recording aot
        # backend, runs forward and backward, and checks one forward and one backward graph were
        # captured and that the custom metadata per node in the dynamo, forward and backward
        # graphs matches the expected strings (inner annotations merged with outer ones).
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying sin, subtract, multiply and divide under nested
            # annotate contexts.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies sin, subtract and multiply inside annotate scopes
                # (outer stage plus inner bucket or stream labels), then divides outside them.
                with fx_traceback.annotate({"pp_stage": 0}):
                    with fx_traceback.annotate({"fdsp_bucket": 0}):
                        sin = torch.sin(x)
                    sub = sin - 2
                    with fx_traceback.annotate({"cuda_stream": 2, "fsdp_bucket": 1}):
                        mul = sub * 2
                div = mul / 3
                return div

        m = Mod()
        backend = AotEagerAndRecordGraphs()
        opt_m = torch.compile(m, backend=backend, fullgraph=True)
        x = torch.randn(10, requires_grad=True)
        opt_m(x).sum().backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)

        dynamo_metadata = fx_traceback._get_custom_metadata(backend.graphs[0])
        fw_metadata = fx_traceback._get_custom_metadata(backend.fw_graphs[0])
        bw_metadata = fx_traceback._get_custom_metadata(backend.bw_graphs[0])
        self.assertExpectedInline(
            str(dynamo_metadata),
            """\
('placeholder', 'l_x_', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'sin', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'sub', {'pp_stage': 0})
('call_function', 'mul', {'pp_stage': 0, 'cuda_stream': 2, 'fsdp_bucket': 1})""",
        )
        self.assertExpectedInline(
            str(fw_metadata),
            """\
('call_function', 'sin', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'sub', {'pp_stage': 0})
('call_function', 'mul', {'pp_stage': 0, 'cuda_stream': 2, 'fsdp_bucket': 1})""",
        )
        self.assertExpectedInline(
            str(bw_metadata),
            """\
('call_function', 'mul_1', {'pp_stage': 0, 'cuda_stream': 2, 'fsdp_bucket': 1})
('call_function', 'cos', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'mul_2', {'pp_stage': 0, 'fdsp_bucket': 0})""",
        )

    def test_flex_attention_block_mask_fallback_to_eager(self):
        # [FUNCTION DOC] Compiles a function calling the block-mask ordering helper on a boolean
        # tensor and asserts the dynamo graph has exactly one argsort node whose custom metadata
        # marks fallback_to_eager as true.
        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(
            lambda x: _dense_to_ordered(x)[1], backend=backend, fullgraph=True
        )
        opt_fn(torch.randint(0, 2, (2, 3), dtype=torch.bool))

        annotated_sorts = [
            node
            for node in backend.graphs[0].graph.nodes
            if node.op == "call_function" and node.target == torch.argsort
        ]
        self.assertEqual(len(annotated_sorts), 1)
        self.assertTrue(annotated_sorts[0].meta["custom"]["fallback_to_eager"])

    def test_activation_checkpointing(self):
        # [FUNCTION DOC] Annotates a call of a checkpointed sin function, compiles with the aot
        # recording backend, runs backward, and checks the custom metadata (the checkpoint tag
        # and wrapped body included) in the dynamo, forward and backward graphs.
        @checkpoint_wrapper
            # [FUNCTION DOC] HELPER: Checkpointed function computing sin.
        def gn(x):
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the checkpointed function inside an annotate scope and
            # returns the sigmoid of the result.
            with fx_traceback.annotate({"ac_sin": 0}):
                ac = gn(x)
            return torch.sigmoid(ac)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.randn(10, requires_grad=True)
        opt_fn(x).sum().backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)

        dynamo_metadata = fx_traceback._get_custom_metadata(backend.graphs[0])
        fw_metadata = fx_traceback._get_custom_metadata(backend.fw_graphs[0])
        bw_metadata = fx_traceback._get_custom_metadata(backend.bw_graphs[0])
        self.assertExpectedInline(
            str(dynamo_metadata),
            """\
('placeholder', 'l_x_', {'ac_sin': 0})
('get_attr', 'wrap_body_0', {'ac_sin': 0})
[('placeholder', 'l_x_', {'ac_sin': 0}), ('call_function', 'sin', {'ac_sin': 0}), ('output', 'output', {'ac_sin': 0})]
('call_function', 'tag_activation_checkpoint', {'ac_sin': 0})
('call_function', 'getitem', {'ac_sin': 0})""",
        )
        self.assertExpectedInline(
            str(fw_metadata),
            """('call_function', 'sin', {'ac_sin': 0})""",
        )
        self.assertExpectedInline(
            str(bw_metadata),
            """\
('call_function', 'cos', {'ac_sin': 0})
('call_function', 'mul', {'ac_sin': 0})""",
        )

    def test_activation_checkpointing_annotation_inside(self):
        # [FUNCTION DOC] Like the checkpointing test but the annotate scope is inside the
        # checkpointed function, so only the sin node carries metadata; checks dynamo, forward
        # and backward custom metadata strings.
        @checkpoint_wrapper
            # [FUNCTION DOC] HELPER: Checkpointed function: add one, sin under annotate, add one
            # again.
        def gn(x):
            x = x + 1
            with fx_traceback.annotate({"stage": 0}):
                p = torch.sin(x)
            return p + 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies the checkpointed function and returns the sigmoid.
            ac = gn(x)
            return torch.sigmoid(ac)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.randn(10, requires_grad=True)
        opt_fn(x).sum().backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)

        dynamo_metadata = fx_traceback._get_custom_metadata(backend.graphs[0])
        fw_metadata = fx_traceback._get_custom_metadata(backend.fw_graphs[0])
        bw_metadata = fx_traceback._get_custom_metadata(backend.bw_graphs[0])
        self.assertExpectedInline(
            str(dynamo_metadata),
            """[('call_function', 'p', {'stage': 0})]""",
        )
        self.assertExpectedInline(
            str(fw_metadata),
            """('call_function', 'sin', {'stage': 0})""",
        )
        self.assertExpectedInline(
            str(bw_metadata),
            """\
('call_function', 'cos', {'stage': 0})
('call_function', 'mul', {'stage': 0})""",
        )

    def test_as_decorator(self):
        # [FUNCTION DOC] Checks annotate used as a method decorator combined with an outer
        # annotate context: sin nodes get both labels while others get only the outer one, in
        # the dynamo, forward and backward graphs.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose sin method is decorated with annotate.
            @fx_traceback.annotate({"fdsp_bucket": 0})
            def sin(self, x):
                # [FUNCTION DOC] HELPER: Returns the sine of the input.
                return torch.sin(x)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the decorated sin, subtract and multiply under an
                # outer annotate scope, then divides by three outside it.
                with fx_traceback.annotate({"pp_stage": 0}):
                    sin = self.sin(x)
                    sub = sin - 2
                    mul = sub * 2
                div = mul / 3
                return div

        m = Mod()
        backend = AotEagerAndRecordGraphs()
        opt_m = torch.compile(m, backend=backend, fullgraph=True)
        x = torch.randn(10, requires_grad=True)
        m(x)
        opt_m(x).sum().backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)

        dynamo_metadata = fx_traceback._get_custom_metadata(backend.graphs[0])
        fw_metadata = fx_traceback._get_custom_metadata(backend.fw_graphs[0])
        bw_metadata = fx_traceback._get_custom_metadata(backend.bw_graphs[0])
        self.assertExpectedInline(
            str(dynamo_metadata),
            """\
('placeholder', 'l_x_', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'sin', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'sub', {'pp_stage': 0})
('call_function', 'mul', {'pp_stage': 0})""",
        )
        self.assertExpectedInline(
            str(fw_metadata),
            """\
('call_function', 'sin', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'sub', {'pp_stage': 0})
('call_function', 'mul', {'pp_stage': 0})""",
        )
        self.assertExpectedInline(
            str(bw_metadata),
            """\
('call_function', 'mul_1', {'pp_stage': 0})
('call_function', 'cos', {'pp_stage': 0, 'fdsp_bucket': 0})
('call_function', 'mul_2', {'pp_stage': 0, 'fdsp_bucket': 0})""",
        )

    def test_graph_break(self):
        # [FUNCTION DOC] Checks a graph break inside an annotate scope does not break eager
        # compilation: compiled output equals eager output.
        def fn(x):
            # [FUNCTION DOC] HELPER: Applies sin, forces a graph break, then cos, inside an
            # annotate scope.
            with torch.fx.traceback.annotate({"pp_stage": 0}):
                x = torch.sin(x)
                torch._dynamo.graph_break()
                x = torch.cos(x)
            return x

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(10, requires_grad=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_annotation_on_runtime_asserts(self):
        # [FUNCTION DOC] Compiles a model with dynamic input size, an item call and a check on
        # its value under annotate, and verifies all dynamo graph nodes, including the generated
        # runtime assertion, carry the annotation.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with data-dependent runtime assertion under annotate.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Under annotate, concatenates x with itself, converts y to
                # a scalar, checks it is at least the new length, and returns the product.
                with torch.fx.traceback.annotate({"moo": 0}):
                    x = torch.cat([x, x])
                    b = y.item()
                    torch._check(b >= x.shape[0])
                    return x * b

        backend = AotEagerAndRecordGraphs()

        inputs = (torch.randn(3), torch.tensor(6))
        model = M()
        torch._dynamo.mark_dynamic(inputs[0], 0)
        torch.compile(model, fullgraph=True, backend=backend)(*inputs)
        dynamo_metadata = fx_traceback._get_custom_metadata(backend.graphs[0])
        self.assertExpectedInline(
            str(dynamo_metadata),
            """\
('placeholder', 'l_x_', {'moo': 0})
('placeholder', 's77', {'moo': 0})
('placeholder', 'l_y_', {'moo': 0})
('call_function', 'cat', {'moo': 0})
('call_method', 'item', {'moo': 0})
('call_function', 'mul', {'moo': 0})
('call_function', 'ge', {'moo': 0})
('call_function', '_assert_scalar_default', {'moo': 0})
('call_function', 'mul_1', {'moo': 0})""",
        )


class AnnotateTestsDevice(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Device-generic annotate tests for flex attention (instantiated for cuda and
    # xpu only).
    @skipCUDAIf(
        not IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED,
        "requires CUDA SM>=8.0 and Triton",
    )
    @skipXPUIf(not IS_FLEX_ATTENTION_XPU_PLATFORM_SUPPORTED, "requires XPU and Triton")
    def test_ac_flex_attention(self, device):
        # [FUNCTION DOC] Compiles a function using flex attention with a causal-style block mask
        # and score_mod under annotate, runs backward, and verifies the annotation appears on the
        # dynamo, forward and backward graphs including subgraphs. Skipped without platform
        # support.
        def _squared(score, b, h, m, n):
            # [FUNCTION DOC] HELPER: Score modification that squares the score.
            return score * score

        def mask_mod(b, h, q, k):
            # [FUNCTION DOC] HELPER: Mask function: true when the query index is nonnegative.
            return q >= 0

        a = 12
        b = 64
        block_mask = create_block_mask(
            mask_mod, None, None, a * b, a * b, device=device
        )

        def gn(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Runs flex attention with x as query, key and value in an
            # annotate scope using the captured block mask and squared score.
            with fx_traceback.annotate({"compile_inductor": 0}):
                return flex_attention(
                    x, x, x, block_mask=block_mask, score_mod=_squared
                )

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies sin, the annotated attention and cos.
            x = torch.sin(x)
            x = gn(x)
            return torch.cos(x)

        x = torch.randn(
            1,
            1,
            a * b,
            b,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        opt_fn(x).sum().backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)

        dynamo_metadata = fx_traceback._get_custom_metadata(backend.graphs[0])
        fw_metadata = fx_traceback._get_custom_metadata(backend.fw_graphs[0])
        bw_metadata = fx_traceback._get_custom_metadata(backend.bw_graphs[0])
        self.assertExpectedInline(
            str(dynamo_metadata),
            """\
('placeholder', 'l_gn_closure_1_cell_contents_full_kv_indices', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_full_kv_num_blocks', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_full_q_indices', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_full_q_num_blocks', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_kv_indices', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_kv_num_blocks', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_q_indices', {'compile_inductor': 0})
('placeholder', 'l_gn_closure_1_cell_contents_q_num_blocks', {'compile_inductor': 0})
('get_attr', 'mask_fn_0', {'compile_inductor': 0})
[('placeholder', 'child', {'compile_inductor': 0}), ('placeholder', 'child_1', {'compile_inductor': 0}), ('placeholder', 'child_2', {'compile_inductor': 0}), ('placeholder', 'child_3', {'compile_inductor': 0}), ('call_function', 'ge', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('get_attr', 'score_mod_0', {'compile_inductor': 0})
[('placeholder', 'child', {'compile_inductor': 0}), ('placeholder', 'child_1', {'compile_inductor': 0}), ('placeholder', 'child_2', {'compile_inductor': 0}), ('placeholder', 'child_3', {'compile_inductor': 0}), ('placeholder', 'child_4', {'compile_inductor': 0}), ('call_function', 'mul', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('call_function', 'flex_attention', {'compile_inductor': 0})
('call_function', 'getitem', {'compile_inductor': 0})""",
        )
        self.assertExpectedInline(
            str(fw_metadata),
            """\
('get_attr', 'sdpa_score0', {'compile_inductor': 0})
[('placeholder', 'arg0_1', {'compile_inductor': 0}), ('placeholder', 'arg1_1', {'compile_inductor': 0}), ('placeholder', 'arg2_1', {'compile_inductor': 0}), ('placeholder', 'arg3_1', {'compile_inductor': 0}), ('placeholder', 'arg4_1', {'compile_inductor': 0}), ('call_function', 'mul', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('get_attr', 'sdpa_mask0', {'compile_inductor': 0})
[('placeholder', 'arg0_1', {'compile_inductor': 0}), ('placeholder', 'arg1_1', {'compile_inductor': 0}), ('placeholder', 'arg2_1', {'compile_inductor': 0}), ('placeholder', 'arg3_1', {'compile_inductor': 0}), ('call_function', 'ge', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('call_function', 'flex_attention', {'compile_inductor': 0})
('call_function', 'getitem', {'compile_inductor': 0})
('call_function', 'getitem_1', {'compile_inductor': 0})
('call_function', 'detach_1', {'compile_inductor': 0})
('call_function', 'detach_3', {'compile_inductor': 0})""",
        )
        self.assertExpectedInline(
            str(bw_metadata),
            """\
('placeholder', 'getitem', {'compile_inductor': 0})
('placeholder', 'detach_3', {'compile_inductor': 0})
('call_function', 'detach', {'compile_inductor': 0})
('call_function', 'detach_2', {'compile_inductor': 0})
('get_attr', 'fw_graph0', {'compile_inductor': 0})
[('placeholder', 'arg0_1', {'compile_inductor': 0}), ('placeholder', 'arg1_1', {'compile_inductor': 0}), ('placeholder', 'arg2_1', {'compile_inductor': 0}), ('placeholder', 'arg3_1', {'compile_inductor': 0}), ('placeholder', 'arg4_1', {'compile_inductor': 0}), ('call_function', 'mul', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('get_attr', 'joint_graph0', {'compile_inductor': 0})
[('placeholder', 'arg0_1', {'compile_inductor': 0}), ('placeholder', 'arg1_1', {'compile_inductor': 0}), ('placeholder', 'arg2_1', {'compile_inductor': 0}), ('placeholder', 'arg3_1', {'compile_inductor': 0}), ('placeholder', 'arg4_1', {'compile_inductor': 0}), ('placeholder', 'arg5_1', {'compile_inductor': 0}), ('call_function', 'mul_1', {'compile_inductor': 0}), ('call_function', 'mul_2', {'compile_inductor': 0}), ('call_function', 'add', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('get_attr', 'mask_graph0', {'compile_inductor': 0})
[('placeholder', 'arg0_1', {'compile_inductor': 0}), ('placeholder', 'arg1_1', {'compile_inductor': 0}), ('placeholder', 'arg2_1', {'compile_inductor': 0}), ('placeholder', 'arg3_1', {'compile_inductor': 0}), ('call_function', 'ge', {'compile_inductor': 0}), ('output', 'output', {'compile_inductor': 0})]
('call_function', 'flex_attention_backward', {'compile_inductor': 0})
('call_function', 'getitem_3', {'compile_inductor': 0})
('call_function', 'getitem_4', {'compile_inductor': 0})
('call_function', 'getitem_5', {'compile_inductor': 0})""",
        )

    @skipCUDAIf(
        not IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED,
        "requires CUDA SM>=8.0 and Triton",
    )
    @skipXPUIf(not IS_FLEX_ATTENTION_XPU_PLATFORM_SUPPORTED, "requires XPU and Triton")
    def test_flex_attention_backward_tag_does_not_leak(self, device):
        # [FUNCTION DOC] Traces a function that calls flex attention and takes autograd grad
        # inside an annotate scope using make_fx with node metadata preserved, and asserts that
        # backward-tagged nodes exist but the flex attention forward node lacks the backward
        # tag and keeps only the annotation. Skipped without platform support.
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.traceback import preserve_node_meta

        def causal_mask(batch, head, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: Causal mask: true when the query index is at least the key
            # index.
            del batch, head
            return q_idx >= kv_idx

        q = torch.randn(
            1, 2, 128, 32, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            1, 2, 128, 32, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn(
            1, 2, 128, 32, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        block_mask = create_block_mask(causal_mask, 1, 2, 128, 128, device=device)

        def fn(q, k, v, block_mask):
            # [FUNCTION DOC] HELPER: In an annotate scope, runs flex attention, computes gradients
            # of the output with respect to the inputs (discarded), and returns the output cosine.
            with fx_traceback.annotate({"ac_region_id": 0}):
                y = flex_attention(q, k, v, block_mask=block_mask)
                torch.autograd.grad(y, (q, k, v), torch.ones_like(y))
                return y.cos()

        warnings.filterwarnings(
            "ignore",
            message="flex_attention called without torch.compile",
        )
        with (
            torch._dynamo.config.patch(error_on_nested_fx_trace=False),
            torch.compiler._non_strict_tracing_context(),
            torch.compiler._patch_autograd_grad(),
            preserve_node_meta(),
        ):
            gm = make_fx(fn, record_stack_traces=True)(q, k, v, block_mask)

        backward_nodes = [
            node for node in gm.graph.nodes if node.meta.get("autograd_backward", False)
        ]
        self.assertTrue(backward_nodes)

        flex_nodes = gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.flex_attention
        )
        self.assertEqual(len(flex_nodes), 1)
        self.assertNotIn("autograd_backward", flex_nodes[0].meta)
        self.assertEqual(flex_nodes[0].meta.get("custom", {}), {"ac_region_id": 0})


instantiate_device_type_tests(
    AnnotateTestsDevice,
    globals(),
    only_for=("cuda", "xpu"),
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests()
