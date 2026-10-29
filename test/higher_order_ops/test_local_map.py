# Owner(s): ["module: higher order operators"]


# [MODULE DOC] Tests for the local_map higher order op (DTensor local_map wrapped by Dynamo).
# Builds a small attention transformer block whose attention is local_map'd with context parallel
# placements on a fake 2048-rank process group, then checks Dynamo graph capture, selective
# activation checkpointing (SAC), placement validation errors, local shapes, gradient filtering,
# symbolic-int activations and FX annotation / metadata propagation. Import-time effects: imports
# distributed local_map names only when distributed is available. Most tests are skipped when
# distributed is unavailable or when already compiled by harness.
import functools
import unittest
from collections.abc import Callable
from contextlib import contextmanager, ExitStack
from typing import Any

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
import torch.fx.traceback as fx_traceback
import torch.nn.functional as F
from torch import nn
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
from torch._dynamo.variables.higher_order_ops import LocalMapWrappedHigherOrderVariable
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import create_selective_checkpoint_contexts


if torch.distributed.is_available():
    from torch.distributed._tensor.experimental import local_map
    from torch.distributed.tensor.placement_types import Replicate, Shard

from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TEST_WITH_TORCHINDUCTOR,
    TestCase,
)


nested_compile_region = torch.compiler.nested_compile_region


@contextmanager
def enable_local_map_wrapping():
    # [FUNCTION DOC] HELPER: Context manager that turns on deferred local_map handling: it enables
    # the Dynamo variable class for wrapped local_map and defers inlining of the local_map HOP
    # until later compilation stages. Yields nothing; both settings are undone on exit.
    from torch._dynamo.variables.higher_order_ops import (
        LocalMapWrappedHigherOrderVariable as vt_cls,
    )
    from torch._higher_order_ops import local_map as local_map_module

    with vt_cls.enable(), local_map_module.defer_inlining():
        yield


def ap_style_initial_capture(
    model: torch.nn.Module, inputs_fn: Callable
) -> torch.nn.Module:
    # [FUNCTION DOC] HELPER: Runs a simplified version of an auto-parallel style capture. Builds
    # inputs with the given factory under a fake tensor mode with a dynamic shape env, requires a
    # tuple result (raises AssertionError otherwise), traces the model with Dynamo export capture
    # under deferred local_map wrapping, then exports the joint forward-backward graph with
    # AOTAutograd using the inductor decomposition table. Returns the joint graph module.
    """
    Similar to AP's initial capture, but:
    - no dtype casting
    - no AP decomps
    - no inductor
    """
    fake_mode = FakeTensorMode()
    fake_mode.shape_env = ShapeEnv()
    fake_mode.static_shapes = False

    with fake_mode:
        inputs = inputs_fn()
    if not isinstance(inputs, tuple):
        raise AssertionError(f"Expected inputs to be tuple, got {type(inputs)}")

    with (
        enable_local_map_wrapping(),
        torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing(),
    ):
        torch_ir_with_fqn = dynamo_graph_capture_for_export(model)(*inputs)
        unused = ExitStack()
        joint_with_descriptors = aot_export_joint_with_descriptors(
            unused,
            torch_ir_with_fqn,
            inputs,
            decompositions=torch._inductor.decomposition.select_decomp_table(),
        )
        unused.close()
    return joint_with_descriptors.graph_module


def get_skip_reasons():
    # [FUNCTION DOC] HELPER: Computes skip arguments for unittest skipIf. Returns a pair of a
    # boolean (true when the test should be skipped) and a reason string, built from whether
    # distributed is missing and whether the run is already under Dynamo or Inductor test modes.
    msg = ""
    if not torch.distributed.is_available():
        msg += "Torch distributed not available. "
    if TEST_WITH_TORCHINDUCTOR or TEST_WITH_TORCHDYNAMO:
        msg += "Already manually torch.compile'd. "

    return msg != "", msg


class MyTransform(torch.autograd.Function):
    # [CLASS DOC] HELPER: Custom autograd function that adds 100 to the input in forward and adds
    # 100 to the incoming gradient in backward (a deliberately non-standard gradient). Not used in
    # this file's tests.
    @staticmethod
    def forward(ctx, x):
        # [FUNCTION DOC] HELPER: Returns the input plus 100; the context argument is unused.
        return x + 100

    @staticmethod
    def backward(ctx, grad):
        # [FUNCTION DOC] HELPER: Returns the incoming gradient plus 100; the context is unused.
        return grad + 100


def context_parallel_attention(query, key, value):
    # [FUNCTION DOC] HELPER: Non-causal scaled dot product attention over the given query,
    # key and value tensors; returns the attention output. This is the function that tests
    # wrap in local_map.
    out = F.scaled_dot_product_attention(
        query=query, key=key, value=value, is_causal=False
    )
    return out


# NOTE: we use this function directly in the node checks
def save_scalar_muls(ctx, op, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Selective-checkpoint policy: returns the MUST_SAVE policy for the
    # scalar multiply op and MUST_RECOMPUTE for every other op. Called directly by tests to compute
    # expected node metadata.
    if op == torch.ops.aten.mul.Scalar:
        return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
    return torch.utils.checkpoint.CheckpointPolicy.MUST_RECOMPUTE


def save_mm(ctx, op, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Selective-checkpoint policy: returns MUST_SAVE for the default matrix
    # multiply op and MUST_RECOMPUTE for every other op.
    if op == torch.ops.aten.mm.default:
        return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
    return torch.utils.checkpoint.CheckpointPolicy.MUST_RECOMPUTE


def create_model(attention_fn, nheads, dim1, dim2, sac_policy=None):
    # [FUNCTION DOC] HELPER: Builds a transformer block module for tests. The block projects the
    # input to query/key/value (no bias), splits heads, applies the given attention function,
    # projects out with a residual, then runs a ReLU feed-forward with a second residual. If
    # sac_policy is given, the forward pass runs under non-reentrant checkpointing with a selective
    # checkpoint context built from that policy. Returns the module instance.
    class LocalMapTransformerBlock(nn.Module):
        # [CLASS DOC] HELPER: Attention plus feed-forward block with six bias-free linear layers
        # (wq, wk, wv, wo, w1, w2). Holds nheads and an optional sac_context_fn (a partial of the
        # selective checkpoint context factory, or None when no policy is configured).
        def __init__(self, nheads, dim1, dim2):
            # [FUNCTION DOC] HELPER: Creates the linear layers (model dim dim1, hidden dim dim2)
            # and sets the checkpoint context factory from the enclosing sac_policy, or None if
            # there is none.
            super().__init__()
            self.nheads = nheads
            bias = False
            self.wq = nn.Linear(dim1, dim1, bias=bias)
            self.wk = nn.Linear(dim1, dim1, bias=bias)
            self.wv = nn.Linear(dim1, dim1, bias=bias)
            self.wo = nn.Linear(dim1, dim1, bias=bias)
            self.w1 = nn.Linear(dim1, dim2, bias=bias)
            self.w2 = nn.Linear(dim2, dim1, bias=bias)
            if sac_policy:
                self.sac_context_fn = functools.partial(
                    create_selective_checkpoint_contexts, sac_policy
                )
            else:
                self.sac_context_fn = None

        def _forward(self, x):
            # [FUNCTION DOC] HELPER: Plain forward computation: q/k/v projections, reshape to
            # heads and permute, attention via the enclosing attention function, merge heads,
            # output projection plus residual, then ReLU feed-forward plus second residual.
            # Returns the result tensor.
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)

            q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
            k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
            v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

            o = attention_fn(q, k, v)
            o = o.permute(0, 2, 1, 3).flatten(-2)

            o = self.wo(o)

            o0 = o + x

            o = self.w1(o0)
            o = torch.nn.functional.relu(o)
            o = self.w2(o)

            o = o0 + o
            return o

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Runs the block, wrapped in non-reentrant checkpointing
            # with the stored context factory when one is configured; otherwise calls the
            # plain forward directly.
            if self.sac_context_fn is not None:
                return torch.utils.checkpoint.checkpoint(
                    self._forward,
                    x,
                    use_reentrant=False,
                    context_fn=self.sac_context_fn,
                )
            return self._forward(x)

    return LocalMapTransformerBlock(nheads, dim1, dim2)


def get_local_mapped_functions(mesh):
    # [FUNCTION DOC] HELPER: Creates two equivalent local_map versions of the context-parallel
    # attention for the given mesh: one built with decorator syntax and one by wrapping the
    # function directly. Both shard query over three mesh dims, shard key/value over two and
    # replicate on the third, shard the output, and redistribute inputs. Raises AssertionError if
    # distributed is unavailable. Returns the pair (decorated version, wrapped version).
    if not torch.distributed.is_available():
        raise AssertionError("torch.distributed is not available")

    @local_map(
        out_placements=((Shard(0), Shard(1), Shard(2)),),
        in_placements=(
            (Shard(0), Shard(1), Shard(2)),  # query
            (Shard(0), Shard(1), Replicate()),  # key
            (Shard(0), Shard(1), Replicate()),  # value
        ),
        redistribute_inputs=True,
        in_grad_placements=None,
        device_mesh=mesh,
    )
    def cp_decorated(query, key, value):
        # [FUNCTION DOC] HELPER: Local function that simply calls the context parallel attention.
        return context_parallel_attention(query, key, value)

    cp_function = local_map(
        context_parallel_attention,
        out_placements=((Shard(0), Shard(1), Shard(2)),),
        in_placements=(
            (Shard(0), Shard(1), Shard(2)),  # query
            (Shard(0), Shard(1), Replicate()),  # key
            (Shard(0), Shard(1), Replicate()),  # value
        ),
        redistribute_inputs=True,
        in_grad_placements=None,
        device_mesh=mesh,
    )

    return cp_decorated, cp_function


class TestLocalMap(TestCase):
    # [CLASS DOC] Test case for local_map under Dynamo, AOTAutograd and export. setUp creates a
    # fake process group of 2048 ranks and a 4-D CPU device mesh (dp, tp, cp, ep) stored as mesh,
    # and forces the math SDPA backend; tearDown undoes both. Tests are skipped when distributed is
    # unavailable or the run is already compiled.
    def setUp(self):
        # [FUNCTION DOC] Resets Dynamo, restricts scaled dot product attention to the math
        # backend via an exit stack, and (if distributed is available) initializes a fake process
        # group with a fake store (rank 0, world size 2048) and builds an 8x8x4x8 CPU mesh named
        # dp, tp, cp, ep.
        super().setUp()
        torch._dynamo.reset()
        self.exit_stack = ExitStack()
        self.exit_stack.enter_context(sdpa_kernel(backends=[SDPBackend.MATH]))
        if torch.distributed.is_available():
            from torch.testing._internal.distributed.fake_pg import FakeStore

            self.fake_store = FakeStore()
            self.world_size = 2048
            torch.distributed.init_process_group(
                "fake", store=self.fake_store, rank=0, world_size=self.world_size
            )
            self.mesh = torch.distributed.device_mesh.init_device_mesh(
                "cpu",
                (8, 8, 4, 8),
                mesh_dim_names=(
                    "dp",
                    "tp",
                    "cp",
                    "ep",
                ),
            )

    def tearDown(self):
        # [FUNCTION DOC] Closes the exit stack (restoring the SDPA backend) and destroys the
        # process group when distributed is available.
        self.exit_stack.close()
        if torch.distributed.is_available():
            torch.distributed.destroy_process_group()

    @unittest.skipIf(*get_skip_reasons())
    def test_simple(self):
        # [FUNCTION DOC] Compiles the model with both local_map variants under the wrapped
        # local_map mode with an eager recording backend and runs backward on each. Unless
        # running under crossref, checks two graphs were captured, that they print identically,
        # and that the first matches an expected graph with a single local_map HOP call over a
        # subgraph with attention.
        cp_decorated, cp_function = get_local_mapped_functions(self.mesh)
        bs = 8 * 1
        dim1 = 96
        dim2 = dim1 * 4
        nheads = 16
        seq_len = 16

        from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm

        backend = EagerAndRecordGraphs()

        model = create_model(cp_decorated, nheads, dim1, dim2)
        inputs = (torch.randn(bs, seq_len, dim1, requires_grad=True),)
        with LocalMapWrappedHigherOrderVariable.enable():
            out = torch.compile(model, backend=backend)(*inputs)
        out.sum().backward()

        model = create_model(cp_function, nheads, dim1, dim2)
        inputs = (torch.randn(bs, seq_len, dim1, requires_grad=True),)
        with LocalMapWrappedHigherOrderVariable.enable():
            out = torch.compile(model, backend=backend)(*inputs)
        out.sum().backward()

        if not TEST_WITH_CROSSREF:
            self.assertEqual(len(backend.graphs), 2)
            self.assertEqual(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                normalize_gm(backend.graphs[1].print_readable(print_output=False)),
            )
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_self_modules_wq_parameters_weight_: "f32[96, 96]", L_x_: "f32[8, 16, 96]", L_self_modules_wk_parameters_weight_: "f32[96, 96]", L_self_modules_wv_parameters_weight_: "f32[96, 96]", L_self_modules_wo_parameters_weight_: "f32[96, 96]", L_self_modules_w1_parameters_weight_: "f32[384, 96]", L_self_modules_w2_parameters_weight_: "f32[96, 384]"):
        l_self_modules_wq_parameters_weight_ = L_self_modules_wq_parameters_weight_
        l_x_ = L_x_
        l_self_modules_wk_parameters_weight_ = L_self_modules_wk_parameters_weight_
        l_self_modules_wv_parameters_weight_ = L_self_modules_wv_parameters_weight_
        l_self_modules_wo_parameters_weight_ = L_self_modules_wo_parameters_weight_
        l_self_modules_w1_parameters_weight_ = L_self_modules_w1_parameters_weight_
        l_self_modules_w2_parameters_weight_ = L_self_modules_w2_parameters_weight_
        q: "f32[8, 16, 96]" = torch._C._nn.linear(l_x_, l_self_modules_wq_parameters_weight_, None);  l_self_modules_wq_parameters_weight_ = None
        k: "f32[8, 16, 96]" = torch._C._nn.linear(l_x_, l_self_modules_wk_parameters_weight_, None);  l_self_modules_wk_parameters_weight_ = None
        v: "f32[8, 16, 96]" = torch._C._nn.linear(l_x_, l_self_modules_wv_parameters_weight_, None);  l_self_modules_wv_parameters_weight_ = None
        unflatten: "f32[8, 16, 16, 6]" = q.unflatten(-1, (16, -1));  q = None
        q_1: "f32[8, 16, 16, 6]" = unflatten.permute(0, 2, 1, 3);  unflatten = None
        unflatten_1: "f32[8, 16, 16, 6]" = k.unflatten(-1, (16, -1));  k = None
        k_1: "f32[8, 16, 16, 6]" = unflatten_1.permute(0, 2, 1, 3);  unflatten_1 = None
        unflatten_2: "f32[8, 16, 16, 6]" = v.unflatten(-1, (16, -1));  v = None
        v_1: "f32[8, 16, 16, 6]" = unflatten_2.permute(0, 2, 1, 3);  unflatten_2 = None
        subgraph_0 = self.subgraph_0
        local_map_hop = torch.ops.higher_order.local_map_hop(subgraph_0, q_1, k_1, v_1);  subgraph_0 = q_1 = k_1 = v_1 = None
        getitem: "f32[8, 16, 16, 6]" = local_map_hop[0];  local_map_hop = None
        permute_3: "f32[8, 16, 16, 6]" = getitem.permute(0, 2, 1, 3);  getitem = None
        o: "f32[8, 16, 96]" = permute_3.flatten(-2);  permute_3 = None
        o_1: "f32[8, 16, 96]" = torch._C._nn.linear(o, l_self_modules_wo_parameters_weight_, None);  o = l_self_modules_wo_parameters_weight_ = None
        o0: "f32[8, 16, 96]" = o_1 + l_x_;  o_1 = l_x_ = None
        o_2: "f32[8, 16, 384]" = torch._C._nn.linear(o0, l_self_modules_w1_parameters_weight_, None);  l_self_modules_w1_parameters_weight_ = None
        o_3: "f32[8, 16, 384]" = torch.nn.functional.relu(o_2);  o_2 = None
        o_4: "f32[8, 16, 96]" = torch._C._nn.linear(o_3, l_self_modules_w2_parameters_weight_, None);  o_3 = l_self_modules_w2_parameters_weight_ = None
        o_5: "f32[8, 16, 96]" = o0 + o_4;  o0 = o_4 = None
        return (o_5,)
    class subgraph_0(torch.nn.Module):
        def forward(self, query: "f32[1, 2, 4, 6]", key: "f32[1, 2, 16, 6]", value: "f32[1, 2, 16, 6]"):
            out: "f32[1, 2, 4, 6]" = torch._C._nn.scaled_dot_product_attention(query = query, key = key, value = value, is_causal = False);  query = key = value = None
            return (out,)""",
                ignore_empty_lines=True,
            )

    @unittest.skipIf(*get_skip_reasons())
    def test_sac(self):
        # [FUNCTION DOC] Same flow as the simple test but with a save-scalar-muls SAC policy and
        # the AOT eager recording backend. Unless under crossref, checks that both variants give
        # identical joint, forward and backward graphs, exactly one activation checkpoint call, and
        # that each forward node with recompute metadata matches the policy: saved nodes appear in
        # forward outputs and backward inputs, recomputed nodes are not backward inputs.
        cp_decorated, cp_function = get_local_mapped_functions(self.mesh)
        bs = 8 * 1
        dim1 = 96
        dim2 = dim1 * 4
        nheads = 16
        seq_len = 16

        from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm

        backend = AotEagerAndRecordGraphs()

        model = create_model(
            cp_decorated, nheads, dim1, dim2, sac_policy=save_scalar_muls
        )
        inputs = (torch.randn(bs, seq_len, dim1, requires_grad=True),)
        with LocalMapWrappedHigherOrderVariable.enable():
            out = torch.compile(model, backend=backend)(*inputs)
        out.sum().backward()

        model = create_model(
            cp_function, nheads, dim1, dim2, sac_policy=save_scalar_muls
        )
        inputs = (torch.randn(bs, seq_len, dim1, requires_grad=True),)
        with LocalMapWrappedHigherOrderVariable.enable():
            out = torch.compile(model, backend=backend)(*inputs)
        out.sum().backward()

        if not TEST_WITH_CROSSREF:
            self.assertEqual(len(backend.graphs), 2)
            self.assertEqual(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                normalize_gm(backend.graphs[1].print_readable(print_output=False)),
            )
            self.assertEqual(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                normalize_gm(backend.fw_graphs[1].print_readable(print_output=False)),
            )
            self.assertEqual(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                normalize_gm(backend.bw_graphs[1].print_readable(print_output=False)),
            )
            self.assertEqual(
                len(
                    backend.graphs[0].graph.find_nodes(
                        op="call_function",
                        target=torch._higher_order_ops.wrap.tag_activation_checkpoint,
                    )
                ),
                1,
            )
            # TODO: add joint to the testing compile backend
            fw_outs = {
                n.name
                for n in backend.fw_graphs[0].graph.find_nodes(op="output")[0].args[0]
            }
            bw_ins = {
                n.name for n in backend.bw_graphs[0].graph.find_nodes(op="placeholder")
            }
            for node in backend.fw_graphs[0].graph.nodes:
                if "recompute" in node.meta:
                    expected = save_scalar_muls(None, node.target, None, None)
                    actual = node.meta["recompute"]
                    self.assertEqual(expected, actual)
                    if actual == torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE:
                        self.assertTrue(node.name in fw_outs and node.name in bw_ins)
                    elif (
                        actual == torch.utils.checkpoint.CheckpointPolicy.MUST_RECOMPUTE
                    ):
                        # can still be in fw_outs for post-graph bytecode
                        self.assertFalse(node.name in bw_ins)

    @unittest.skipIf(*get_skip_reasons())
    def test_sac_deferred(self):
        # [FUNCTION DOC] Like the SAC test but with bfloat16 and deferred local_map inlining. A
        # known limitation makes the compile or backward raise AssertionError whose message
        # mentions expected_fw_outputs; that is recorded and tolerated. Graph and metadata
        # checks (equal graphs, one checkpoint call, saved nodes in forward outputs) run only if
        # no such failure occurred.
        # This test is in a bit of a weird state, it needs compositional compile API
        # so that we can defer inlining for up until AOTAutograd stage 1.
        # Then we should be inlined by stage 2. But we can't do that today.

        cp_decorated, cp_function = get_local_mapped_functions(self.mesh)
        bs = 8 * 1
        dim1 = 128
        dim2 = dim1 * 4
        nheads = 16
        seq_len = 16

        from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm

        backend = AotEagerAndRecordGraphs()
        saw_deferred_sac_failure = False

        def check_deferred_sac_failure(e):
            # [FUNCTION DOC] HELPER: Marks the enclosing flag as seen and asserts the given error
            # text mentions expected_fw_outputs.
            nonlocal saw_deferred_sac_failure
            saw_deferred_sac_failure = True
            self.assertIn("expected_fw_outputs", str(e))

        model = create_model(
            cp_decorated, nheads, dim1, dim2, sac_policy=save_scalar_muls
        ).to(torch.bfloat16)
        inputs = (
            torch.randn(bs, seq_len, dim1, requires_grad=True, dtype=torch.bfloat16),
        )
        try:
            with enable_local_map_wrapping():
                out = torch.compile(model, backend=backend)(*inputs)
            out.sum().backward()
        except AssertionError as e:
            # TODO: deferred local_map + SAC still needs local_map installed
            # as a subgraph throughout the runtime path.
            check_deferred_sac_failure(e)

        model = create_model(
            cp_function, nheads, dim1, dim2, sac_policy=save_scalar_muls
        ).to(torch.bfloat16)
        inputs = (
            torch.randn(bs, seq_len, dim1, requires_grad=True, dtype=torch.bfloat16),
        )
        try:
            with enable_local_map_wrapping():
                out = torch.compile(model, backend=backend)(*inputs)
            out.sum().backward()
        except AssertionError as e:
            # TODO: deferred local_map + SAC still needs local_map installed
            # as a subgraph throughout the runtime path.
            check_deferred_sac_failure(e)

        # TODO: re-enable tests on backward when we can install as a subgraph
        if not TEST_WITH_CROSSREF and not saw_deferred_sac_failure:
            self.assertEqual(len(backend.graphs), 2)
            self.assertEqual(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                normalize_gm(backend.graphs[1].print_readable(print_output=False)),
            )
            self.assertEqual(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                normalize_gm(backend.fw_graphs[1].print_readable(print_output=False)),
            )
            # self.assertEqual(
            #     normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
            #     normalize_gm(backend.bw_graphs[1].print_readable(print_output=False)),
            # )
            self.assertEqual(
                len(
                    backend.graphs[0].graph.find_nodes(
                        op="call_function",
                        target=torch._higher_order_ops.wrap.tag_activation_checkpoint,
                    )
                ),
                1,
            )
            # TODO: add joint to the testing compile backend
            fw_outs = {
                n.name
                for n in backend.fw_graphs[0].graph.find_nodes(op="output")[0].args[0]
            }
            # bw_ins = {
            #     n.name for n in backend.bw_graphs[0].graph.find_nodes(op="placeholder")
            # }
            for node in backend.fw_graphs[0].graph.nodes:
                if "recompute" in node.meta:
                    expected = save_scalar_muls(None, node.target, None, None)
                    actual = node.meta["recompute"]
                    self.assertEqual(expected, actual)
                    if actual == torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE:
                        self.assertTrue(node.name in fw_outs)
                    #     self.assertTrue(node.name in fw_outs and node.name in bw_ins)
                    # elif (
                    #     actual == torch.utils.checkpoint.CheckpointPolicy.MUST_RECOMPUTE
                    # ):
                    #     # can still be in fw_outs for post-graph bytecode
                    #     self.assertFalse(node.name in bw_ins)

    @unittest.skipIf(*get_skip_reasons())
    def test_local_map_dynamo_mismatch_placements(self):
        # [FUNCTION DOC] Checks that Dynamo validates placements against the call: a local_map with
        # one input placement called with two tensors raises AssertionError saying it expected 1
        # inputs but found 2, and one with one output placement returning two values raises
        # AssertionError saying it expected 1 outputs but found 2.
        @local_map(
            out_placements=((Shard(0), Shard(1), Shard(2)),),
            in_placements=((Shard(0), Shard(1), Shard(2)),),
            redistribute_inputs=True,
            in_grad_placements=None,
            device_mesh=self.mesh,
        )
        def mismatch_input(x, y):
            # [FUNCTION DOC] HELPER: Adds its two inputs; declared with only one input placement so
            # the call is intentionally mismatched.
            return x + y

        x = torch.randn(64, 64, 64, requires_grad=True)
        y = torch.randn(64, 64, 64, requires_grad=True)
        with (
            LocalMapWrappedHigherOrderVariable.enable(),
            self.assertRaisesRegex(
                AssertionError,
                "Expecting 1 inputs to local_map function based on placements, but found 2.",
            ),
        ):
            torch.compile(mismatch_input, backend="eager", fullgraph=True)(x, y)

        @local_map(
            out_placements=(
                (Shard(0), Shard(1), Shard(2)),
                # purposefully mismatched outputs
            ),
            in_placements=((Shard(0), Shard(1), Shard(2)),),
            redistribute_inputs=True,
            in_grad_placements=None,
            device_mesh=self.mesh,
        )
        def mismatch_outputs(x):
            # [FUNCTION DOC] HELPER: Returns the input plus 11 and plus 12 as two outputs; declared
            # with only one output placement so the call is intentionally mismatched.
            return x + 11, x + 12

        x = torch.randn(64, 64, 64, requires_grad=True)
        with (
            LocalMapWrappedHigherOrderVariable.enable(),
            self.assertRaisesRegex(
                AssertionError,
                "Expecting 1 outputs to local_map function based on placements, but found 2.",
            ),
        ):
            torch.compile(mismatch_outputs, backend="eager", fullgraph=True)(x)

    @unittest.skipIf(*get_skip_reasons())
    def test_local_map_dynamo_reordered_inputs(self):
        # [FUNCTION DOC] Verifies Dynamo keeps local_map subgraph placeholders in the user's
        # call-site order even when the body uses the second input first. Captures the compiled
        # graph with a recording backend and asserts one graph, subgraph placeholders named
        # first_input then second_input, and local_map HOP call arguments of subgraph, first
        # arg, second arg.
        # Dynamo lifts subgraph freevars in first-use order inside the body,
        # which can differ from the user's call-site order (especially under
        # dynamic shapes where `.shape` accesses force early lifts). The HOP
        # reorders subgraph placeholders back to call-site order so that
        # `in_placements` always lines up with the user-written arg order.
        @local_map(
            out_placements=((Shard(0), Shard(0)),),
            in_placements=(
                (Shard(0), Shard(0)),
                (Replicate(), Shard(0)),
            ),
            redistribute_inputs=True,
            in_grad_placements=None,
            device_mesh=self.mesh,
        )
        def reorder_inputs(first_input, second_input):
            # [FUNCTION DOC] HELPER: Returns the sum of the second input times ten plus the first
            # input; it uses the second input first, which tests placeholder reordering.
            return second_input.sum() * 10 + first_input  # first-use: second, first

        x = torch.randn(64, 64, 64, requires_grad=True)
        y = torch.randn(8, 64, 64, requires_grad=True)

        captured_gms: list[torch.fx.GraphModule] = []

        def capture(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Dynamo backend that appends the graph module to the captured
            # list and returns it unchanged.
            captured_gms.append(gm)
            return gm

        with LocalMapWrappedHigherOrderVariable.enable():
            torch.compile(reorder_inputs, backend=capture, fullgraph=True)(x, y)

        self.assertEqual(len(captured_gms), 1)
        root = captured_gms[0]
        subgraph = root.subgraph_0
        ph_names = [n.name for n in subgraph.graph.find_nodes(op="placeholder")]
        self.assertEqual(ph_names, ["first_input", "second_input"])
        # The HOP call in the parent graph should pass the args in the user's
        # call-site order: (subgraph, first_input, second_input).
        hop_call = next(
            n
            for n in root.graph.nodes
            if n.op == "call_function" and "local_map_hop" in str(n.target)
        )
        arg_names = [a.name if hasattr(a, "name") else a for a in hop_call.args]
        self.assertEqual(arg_names, ["subgraph_0", "l_args_0_", "l_args_1_"])

    @unittest.skipIf(*get_skip_reasons())
    def test_local_map_with_local_shapes_hop_tracing(self):
        # [FUNCTION DOC] Traces a function that requires local shape (10 by 80) with make_fx,
        # attaches local_map metadata (placements on three mesh dims and the mesh) to the graph,
        # then calls the local_map HOP on a fake global tensor of shape 80 by 80 with deferred
        # inlining and runs backward. Asserts the global tensor keeps its global shape.
        def fn(x):
            # [FUNCTION DOC] HELPER: Raises AssertionError unless the input has local shape 10
            # by 80, then adds ten via a flattened view and returns the result viewed back as
            # a one-element tuple.
            if x.shape != (10, 80):
                raise AssertionError("expected local shapes")
            # force view specialization ops
            out = x.view(-1) + 10
            return (out.view(x.shape),)

        x = torch.randn(10, 80)
        gm = make_fx(fn)(x)

        gm.meta = {
            "local_map_kwargs": {
                "in_placements": ((Shard(0), Replicate(), Replicate()),),
                "out_placements": ((Shard(0), Replicate(), Replicate()),),
                "device_mesh": self.mesh,
            }
        }

        with FakeTensorMode():
            global_tensor = torch.randn(80, 80, requires_grad=True)
        with torch._higher_order_ops.local_map.defer_inlining():
            out = torch._higher_order_ops.local_map_hop(gm, global_tensor)
            out[0].sum().backward()
        self.assertEqual(global_tensor.shape, (80, 80))

    @unittest.skipIf(*get_skip_reasons())
    def test_local_map_with_local_shapes_dynamo_tracing(self):
        # [FUNCTION DOC] Captures a model that calls a local_map function using view operations
        # through the auto-parallel style capture. Finds the forward and backward local_map HOP
        # nodes in the joint graph and asserts (raising AssertionError on mismatch) that each has
        # exactly one input/output and that recorded shapes are the global shape 80 by 80.
        @local_map(
            out_placements=((Shard(0), Replicate(), Replicate()),),
            in_placements=((Shard(0), Replicate(), Replicate()),),
            redistribute_inputs=True,
            in_grad_placements=None,
            device_mesh=self.mesh,
        )
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds ten using a flattened view and returns the result
            # reshaped to the local input shape as a one-element tuple.
            out = x.view(-1) + 10
            return (out.view(x.shape),)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward just calls the local_map function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the local_map function applied to the input.
                return fn(x)

        model = MyModule()

        def inputs_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with an 80 by 80 random tensor
            # that requires grad.
            return (torch.randn(80, 80, requires_grad=True),)

        gm = ap_style_initial_capture(model, inputs_fn)
        fw_node, bw_node = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.higher_order.local_map_hop
        ]

        # Graph should not be aware that Fake key used local shapes
        fw_inputs = fw_node.args[1:]
        if len(fw_inputs) != 1:
            raise AssertionError(f"Expected len(fw_inputs) == 1, got {len(fw_inputs)}")
        self.assertEqual(fw_inputs[0].meta["val"].shape, (80, 80))

        fw_outputs = fw_node.args[1:]
        if len(fw_outputs) != 1:
            raise AssertionError(
                f"Expected len(fw_outputs) == 1, got {len(fw_outputs)}"
            )
        self.assertEqual(fw_outputs[0].meta["val"].shape, (80, 80))

        bw_inputs = bw_node.args[1:]
        if len(bw_inputs) != 1:
            raise AssertionError(f"Expected len(bw_inputs) == 1, got {len(bw_inputs)}")
        self.assertEqual(bw_inputs[0].meta["val"].shape, (80, 80))

        bw_outputs = bw_node.meta["val"]
        if len(bw_outputs) != 1:
            raise AssertionError(
                f"Expected len(bw_outputs) == 1, got {len(bw_outputs)}"
            )
        self.assertEqual(bw_outputs[0].shape, (80, 80))

    @unittest.skipIf(*get_skip_reasons())
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_deferred_local_map_inside_nested_compile_region_export(self):
        # [FUNCTION DOC] Checks export of a model with a local_map inside a nested compile
        # region. Runs the export helper on the bare module (expects no invoke_subgraph node),
        # resets Dynamo, then on the wrapped module (expects an invoke_subgraph node); each
        # export also runs the joint AOTAutograd export. Single-use invoke subgraph inlining is
        # disabled by config.
        placement = (Shard(0), Shard(1), Replicate(), Replicate())

        class Inner(nn.Module):
            # [CLASS DOC] HELPER: Module with a bias-free linear layer and a stored mesh;
            # forward applies the layer and then a local_map-wrapped clone with the
            # shared four-dim placement.
            def __init__(self, dim, mesh):
                # [FUNCTION DOC] HELPER: Creates a bias-free linear layer of the given size and
                # stores the mesh.
                super().__init__()
                self.lin = nn.Linear(dim, dim, bias=False)
                self.mesh = mesh

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer, then a local_map-wrapped clone
                # function with the shared placement for input and output on the stored mesh, and
                # returns the result.
                x = self.lin(x)

                def ident(t):
                    # [FUNCTION DOC] HELPER: Returns a clone of the input tensor.
                    return t.clone()

                return local_map(
                    ident,
                    out_placements=(placement,),
                    in_placements=(placement,),
                    device_mesh=self.mesh,
                    redistribute_inputs=True,
                )(x)

        class NestedRegion(nn.Module):
            # [CLASS DOC] HELPER: Wraps a layer module; its forward is marked as a nested compile
            # region so export produces an invoke_subgraph call.
            def __init__(self, layer):
                # [FUNCTION DOC] HELPER: Stores the wrapped layer.
                super().__init__()
                self.layer = layer

            @nested_compile_region
            def forward(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the wrapped layer and
                # returns its output.
                return self.layer(*args, **kwargs)

        def run_export(model, x, expect_invoke_subgraph):
            # [FUNCTION DOC] HELPER: Exports the model with deferred local_map under Dynamo
            # capture, asserts that the presence of an invoke_subgraph node in the graph
            # equals expect_invoke_subgraph, and then runs the AOTAutograd joint export on
            # the captured graph.
            with enable_local_map_wrapping():
                gm = dynamo_graph_capture_for_export(model)(x)
                has_invoke_subgraph = any(
                    n.op == "call_function"
                    and n.target is torch.ops.higher_order.invoke_subgraph
                    for n in gm.graph.nodes
                )
                self.assertEqual(has_invoke_subgraph, expect_invoke_subgraph)
                with ExitStack() as stack:
                    aot_export_joint_with_descriptors(stack, gm, (x,))

        x = torch.randn(8, 32, 16, requires_grad=True)
        inner = Inner(16, self.mesh)
        run_export(inner, x, expect_invoke_subgraph=False)

        torch._dynamo.reset()
        run_export(NestedRegion(inner), x, expect_invoke_subgraph=True)

    @unittest.skipIf(*get_skip_reasons())
    def test_none_gradients(self):
        # [FUNCTION DOC] Captures a fully replicated local_map linear where the second input does
        # not require grad (so its gradient is None) through the auto-parallel style capture;
        # passes if the capture completes without error.
        @local_map(
            out_placements=((Replicate(), Replicate(), Replicate()),),
            in_placements=(
                (Replicate(), Replicate(), Replicate()),
                (Replicate(), Replicate(), Replicate()),
            ),
            redistribute_inputs=True,
            in_grad_placements=None,
            device_mesh=self.mesh,
        )
        def replicate_linear(w, x):
            # [FUNCTION DOC] HELPER: Matrix-multiplies the second input by the
            # transposed first input.
            # x does not requires_grad, so it will have None gradients
            return torch.matmul(x, w.t())

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding an 80 by 80 linear layer, whose forward applies
            # the local_map linear to its weight and the input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.w = nn.Linear(80, 80)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the local_map linear with the layer weight
                # and the input.
                return replicate_linear(self.w.weight, x)

        model = MyModule()

        def inputs_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple holding an 80 by 80 random
            # tensor without grad.
            return (torch.randn(80, 80),)

        ap_style_initial_capture(model, inputs_fn)

    @unittest.skipIf(*get_skip_reasons())
    def test_none_placements(self):
        # [FUNCTION DOC] Checks local_map with None placements for non-tensor arguments (an int and
        # a module). Captures a model calling it with a scalar and a holder module; passes if
        # capture completes without error.
        class ScalarHolder(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a scalar and adding it to its input in forward.
            def __init__(self, scalar):
                # [FUNCTION DOC] HELPER: Stores the scalar value.
                super().__init__()
                self.scalar = scalar

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the stored scalar.
                return x + self.scalar

        @local_map(
            out_placements=((Replicate(), Replicate(), Replicate()),),
            in_placements=(
                (Replicate(), Replicate(), Replicate()),
                None,
                None,
            ),
            redistribute_inputs=True,
            in_grad_placements=None,
            device_mesh=self.mesh,
        )
        def fn_with_non_tensors(x, scalar, module):
            # [FUNCTION DOC] HELPER: Returns the input plus ten, plus the scalar argument, plus the
            # scalar held by the module argument.
            return x + 10 + scalar + module.scalar

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that owns a scalar holder with value 10 and calls the
            # local_map function with the input, the integer 10 and that holder.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the scalar holder submodule.
                super().__init__()
                self.module = ScalarHolder(10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the local_map function with the input, a constant
                # and the module.
                return fn_with_non_tensors(x, 10, self.module)

        def inputs_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with a 10 by 10 random
            # tensor requiring grad.
            return (torch.randn(10, 10, requires_grad=True),)

        model = MyModule()
        ap_style_initial_capture(model, inputs_fn)

    @unittest.skipIf(*get_skip_reasons())
    def test_filtered_gradients(self):
        # [FUNCTION DOC] Checks gradient filtering when a local_map output is an input that does
        # not require grad: the tangent for it is dropped. Captures a model using two outputs;
        # passes if capture completes without error.
        @local_map(
            out_placements=(
                (Replicate(), Replicate(), Replicate()),
                (Replicate(), Replicate(), Replicate()),
            ),
            in_placements=(
                (Replicate(), Replicate(), Replicate()),
                (Replicate(), Replicate(), Replicate()),
            ),
            redistribute_inputs=True,
            in_grad_placements=None,
            device_mesh=self.mesh,
        )
        def returns_non_param(w, x):
            # [FUNCTION DOC] HELPER: Returns two outputs: the matrix product of the second input
            # with the transposed first, and the second input plus 20.
            # x does not requires_grad, and it is an output, so its corresponding tangent is filtered out
            return torch.matmul(x, w.t()), x + 20

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an 80 by 80 linear layer; forward sums the two
            # outputs of the local_map function applied to its weight and input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.w = nn.Linear(80, 80)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the local_map function and returns the sum of the
                # totals of both outputs.
                a, b = returns_non_param(self.w.weight, x)
                return a.sum() + b.sum()

        model = MyModule()

        def inputs_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple holding an 80 by 80 random
            # tensor without grad.
            return (torch.randn(80, 80),)

        ap_style_initial_capture(model, inputs_fn)

    @unittest.skipIf(*get_skip_reasons())
    def test_symint_activations(self):
        # [FUNCTION DOC] Captures a local_map body that performs all-to-all collectives with data
        # dependent split sizes (converted to Python lists on CPU), so activations contain symbolic
        # integers. Defines helpers for group name, axis size and the all-to-all, and an autograd
        # function whose backward swaps the splits. Passes if capture completes without error.
        import torch.distributed.distributed_c10d as c10d

        def _get_group_name_from_axis_name(mesh_name):
            # [FUNCTION DOC] HELPER: Returns the group name of the process group for the
            # named mesh dim.
            mesh = self.mesh
            group = mesh.get_group(mesh_name)
            return group.group_name

        def axis_size(axis_name):
            # [FUNCTION DOC] HELPER: Returns the size of the mesh dimension with the given name.
            mesh = self.mesh
            # assert axis_name in mesh.mesh_dim_names
            axis_dim = mesh.mesh_dim_names.index(axis_name)
            return mesh.size(axis_dim)

        def _all_to_all(
            self: torch.Tensor,
            output_split_sizes: list[int] | None,
            input_split_sizes: list[int] | None,
            group_name: str,
        ):
            # [FUNCTION DOC] HELPER: Functional all-to-all on a tensor over the named group. Both
            # split size lists must be given together or both be None, else AssertionError is
            # raised; if None, equal splits of the first dimension per group member are used.
            # Returns the waited result.
            group_size = c10d._get_group_size_by_name(group_name)
            if output_split_sizes is None or input_split_sizes is None:
                if not (output_split_sizes is None and input_split_sizes is None):
                    raise AssertionError(
                        "output_split_sizes and input_split_sizes must either be "
                        "specified together or both set to None"
                    )
                output_split_sizes = [self.shape[0] // group_size] * group_size
                input_split_sizes = output_split_sizes

            tensor = torch.ops._c10d_functional.all_to_all_single(
                self, output_split_sizes, input_split_sizes, group_name
            )
            res = torch.ops._c10d_functional.wait_tensor(tensor)
            return res

        class _AllToAll(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function wrapping the all-to-all by axis name. Forward
            # stores the group name and split sizes on the context; backward runs the all-to-all
            # with input and output splits swapped.
            @staticmethod
            def forward(
                ctx: Any,
                x: torch.Tensor,
                output_split_sizes: list[int] | None,
                input_split_sizes: list[int] | None,
                axis_name: str,
            ):
                # [FUNCTION DOC] HELPER: Resolves the group name from the axis name, saves it and
                # the split sizes on the context, and returns the all-to-all of the input.
                group_name = _get_group_name_from_axis_name(axis_name)
                ctx.group_name = group_name
                ctx.output_split_sizes = output_split_sizes
                ctx.input_split_sizes = input_split_sizes
                return _all_to_all(x, output_split_sizes, input_split_sizes, group_name)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
                # [FUNCTION DOC] HELPER: Returns the all-to-all of the gradient using the saved
                # group name with the saved input and output split sizes swapped.
                return _all_to_all(
                    grad_output,
                    ctx.input_split_sizes,
                    ctx.output_split_sizes,
                    ctx.group_name,
                )

        all_to_all = _AllToAll.apply

        @local_map(
            out_placements=((Replicate(), Replicate(), Replicate()),),
            in_placements=(
                (Replicate(), Replicate(), Replicate()),
                (Replicate(), Replicate(), Replicate()),
                None,
            ),
            redistribute_inputs=True,
            in_grad_placements=None,
            device_mesh=self.mesh,
        )
        def _forward(num_tokens_per_expert, x, axis_name):
            # [FUNCTION DOC] HELPER: Without gradient tracking, exchanges per-expert token counts
            # by all-to-all, computes input/output split lists on CPU (output side blocks on a
            # device sync), then all-to-alls the data. Returns a clone of the input with the routed
            # output added in place.
            ep_size = axis_size(axis_name)

            # generate the input splits and output splits for all-to-all
            with torch.no_grad():
                num_tokens_per_expert_group = all_to_all(
                    num_tokens_per_expert,
                    None,
                    None,
                    axis_name,
                )
                input_splits = (
                    num_tokens_per_expert.view(ep_size, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=True)
                )
                # NOTE: this would incur a device-to-host sync
                output_splits = (
                    num_tokens_per_expert_group.view(ep_size, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=False)
                )
                input_splits = input_splits.tolist()
                output_splits = output_splits.tolist()

            # perform all-to-all
            routed_inputs = all_to_all(
                x,
                output_splits,
                input_splits,
                axis_name,
            )

            # noop routed experts
            routed_outputs = routed_inputs
            # noop shared experts
            out = x.clone()

            return out.add_(routed_outputs)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward calls the local_map function with the expert
            # counts, the routed input and the axis name "ep".
            def __init__(self):
                # [FUNCTION DOC] HELPER: Only calls the parent constructor.
                super().__init__()

            def forward(self, num_tokens_per_expert, routed_input):
                # [FUNCTION DOC] HELPER: Calls the local_map function with the two inputs and the
                # axis name ep.
                return _forward(num_tokens_per_expert, routed_input, "ep")

        model = MyModule()

        def inputs_fn():
            # [FUNCTION DOC] HELPER: Returns a pair: an integer tensor of ones (1 by 8) and an 8 by
            # 80 float random tensor requiring grad.
            x = torch.randn(8, 80, requires_grad=True)
            num_tokens_per_expert = torch.ones(1, 8, dtype=torch.int32)
            return (
                num_tokens_per_expert,
                x,
            )

        ap_style_initial_capture(model, inputs_fn)

    @unittest.skipIf(*get_skip_reasons())
    def test_fx_annotations(self):
        # [FUNCTION DOC] Checks that FX annotations inside local_map are kept and that autograd
        # metadata is right after inlining. A local_map function annotates its matmul with an id;
        # it is called once directly and once under selective checkpointing. After capture and
        # inlining with make_fx, asserts four matrix multiplies tagged forward, forward, backward,
        # backward with the expected ids, and that backward nodes carry the backward metadata flag.
        @local_map(
            out_placements=((Replicate(), Replicate(), Replicate()),),
            in_placements=(
                (Replicate(), Replicate(), Replicate()),
                (Replicate(), Replicate(), Replicate()),
                None,
            ),
            redistribute_inputs=True,
            in_grad_placements=None,
            device_mesh=self.mesh,
        )
        def fn(w, x, id):
            # [FUNCTION DOC] HELPER: Matrix-multiplies the second input by the transposed first
            # inside an FX annotation scope that records the given id under a custom key.
            with fx_traceback.annotate({"inside_local_map": id}):
                return torch.matmul(x, w.t())

        context_fn = functools.partial(create_selective_checkpoint_contexts, save_mm)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an 80 by 80 linear layer that sums the results of
            # the annotated function called directly (id 0) and via checkpointing with a save-mm
            # policy (id 1).
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.w = nn.Linear(80, 80)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes one result by calling the annotated function and
                # another via non-reentrant checkpointing, and returns the sum of their totals.
                a = fn(self.w.weight, x, 0)
                b = torch.utils.checkpoint.checkpoint(
                    fn, self.w.weight, x, 1, use_reentrant=False, context_fn=context_fn
                )
                return a.sum() + b.sum()

        model = MyModule()

        def inputs_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple holding an 80 by 80 random
            # tensor without grad.
            return (torch.randn(80, 80),)

        with fx_traceback.preserve_node_meta():
            joint_gm_deferred = ap_style_initial_capture(model, inputs_fn)
            joint_inputs = [
                n.meta["val"]
                for n in joint_gm_deferred.graph.nodes
                if n.op == "placeholder"
            ]
            # TODO: need a local shape interpreter for cases where the graph specializes on shapes
            interp = torch.fx.Interpreter(joint_gm_deferred)
            joint_gm_inlined = make_fx(interp.run)(*joint_inputs)

        mm_nodes = joint_gm_inlined.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )
        self.assertEqual(len(mm_nodes), 4)
        self.assertEqual(mm_nodes[0].meta["partitioner_tag"], "is_forward")
        self.assertEqual(mm_nodes[1].meta["partitioner_tag"], "is_forward")
        self.assertEqual(mm_nodes[2].meta["partitioner_tag"], "is_backward")
        self.assertEqual(mm_nodes[3].meta["partitioner_tag"], "is_backward")
        self.assertEqual(mm_nodes[0].meta["custom"]["inside_local_map"], 0)
        self.assertEqual(mm_nodes[1].meta["custom"]["inside_local_map"], 1)
        self.assertEqual(mm_nodes[2].meta["custom"]["inside_local_map"], 1)
        self.assertEqual(mm_nodes[3].meta["custom"]["inside_local_map"], 0)
        for node in joint_gm_inlined.graph.nodes:
            if node.meta.get("partitioner_tag") == "is_backward":
                self.assertTrue(node.meta.get("autograd_backward", False))

    @unittest.skipIf(*get_skip_reasons())
    def test_no_autograd_backward_metadata_on_inlined_forward_nodes(self):
        # [FUNCTION DOC] Verifies that after capturing and inlining a joint graph, no node
        # tagged as forward carries the autograd backward metadata flag. Uses a local_map body
        # with view and permute operations on fully replicated placements and asserts the list
        # of bad nodes is empty.
        placements = (Replicate(), Replicate(), Replicate(), Replicate())

        @local_map(
            out_placements=(placements,),
            in_placements=(placements, placements),
            redistribute_inputs=True,
            in_grad_placements=None,
            device_mesh=self.mesh,
        )
        def fn(x, w):
            # [FUNCTION DOC] HELPER: Matrix-multiplies the two inputs, applies view, permute,
            # permute and reshape ops, then adds the first input.
            y = x @ w
            y = y.view(2, 4, 16).permute(1, 0, 2).permute(1, 0, 2).reshape(8, 16)
            return y + x

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 16 by 16 parameter; forward returns the total of
            # the local_map function applied to the input and the parameter.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the random 16 by 16 parameter.
                super().__init__()
                self.w = nn.Parameter(torch.randn(16, 16))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sum of the local_map result for the input
                # and the parameter.
                return fn(x, self.w).sum()

        def inputs_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple holding an 8 by 16 random tensor.
            return (torch.randn(8, 16),)

        with fx_traceback.preserve_node_meta():
            joint_gm_deferred = ap_style_initial_capture(MyModule(), inputs_fn)
            joint_inputs = [
                n.meta["val"]
                for n in joint_gm_deferred.graph.nodes
                if n.op == "placeholder"
            ]
            joint_gm_inlined = make_fx(torch.fx.Interpreter(joint_gm_deferred).run)(
                *joint_inputs
            )

        bad_nodes = [
            n
            for n in joint_gm_inlined.graph.nodes
            if n.meta.get("partitioner_tag") == "is_forward"
            and n.meta.get("autograd_backward", False)
        ]
        self.assertEqual(bad_nodes, [])

    @unittest.skipIf(*get_skip_reasons())
    def test_local_map_make_contiguous_strides_for(self):
        # [FUNCTION DOC] Checks a local_map body that builds strides with the contiguous
        # strides helper and calls as_strided compiles with dynamic shapes. Compiles the model
        # with the eager recording backend under deferred wrapping and asserts the first output
        # has the input's shape.
        # make_contiguous_strides_for inside local_map must compile with dynamic shapes.
        from torch._prims_common import make_contiguous_strides_for

        @local_map(
            out_placements=((Shard(0), Replicate()),),
            in_placements=((Shard(0), Replicate()),),
            redistribute_inputs=True,
            device_mesh=self.mesh,
        )
        def fn(x):
            # [FUNCTION DOC] HELPER: Computes contiguous strides for the input shape and returns a
            # one-tuple with the input as_strided to its own shape with those strides.
            strides = make_contiguous_strides_for(x.shape)
            return (x.as_strided(x.shape, strides),)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward calls the local_map function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the local_map function applied to the input.
                return fn(x)

        model = MyModule()
        from torch._dynamo.testing import EagerAndRecordGraphs

        backend = EagerAndRecordGraphs()
        x = torch.randn(80, 80)
        with enable_local_map_wrapping():
            out = torch.compile(model, backend=backend)(x)
        self.assertEqual(out[0].shape, x.shape)


if __name__ == "__main__":
    run_tests()
