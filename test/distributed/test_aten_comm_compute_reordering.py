# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the aten-level compute/communication overlap scheduling pass
# (schedule_overlap_bucketing) in Inductor, run as multi-process tests with functional
# collectives. Provides helpers that install a post-grad custom pass to capture the
# rewritten aten graph, config patch dictionaries, and three test classes: basic
# reordering, automatic bucketing and manual (module-path based) bucketing. Defines the
# module-level device_type from get_devtype at import time.
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case

# for some reason importing functional collectives after dynamo breaks collectives handling!
import torch.distributed._functional_collectives as _functional_collectives
import torch.fx as fx
from torch._C import FileCheck
from torch._dynamo.utils import counters, detect_fake_mode, same
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch._inductor import compile_fx
from torch._inductor.compile_fx import _recursive_post_grad_passes
from torch._inductor.fx_utils import get_fake
from torch._inductor.utils import run_and_get_code, run_and_get_triton_code
from torch._inductor.virtualized import V
from torch.nn.attention.flex_attention import flex_attention
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    at_least_x_gpu,
    DynamoDistributedMultiProcTestCase,
    requires_accelerator_dist_backend,
)
from torch.utils import _pytree as pytree


aten = torch.ops.aten
import functools

from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import skipIfRocm
from torch.testing._internal.inductor_utils import HAS_GPU


def estimate_aten_runtime(fx_node, override_size=None, compute_multiplier=1.0):
    # [FUNCTION DOC] HELPER: Test runtime estimator: any node whose target string contains c10
    # (a collective) costs 1.0, aten.mm costs compute_multiplier, everything else returns None
    # (no estimate). override_size is accepted but unused.
    # for tests, assume a matmul can hide a single collective
    if "c10" in str(fx_node.target):
        return 1.0
    elif fx_node.target == aten.mm.default:
        return compute_multiplier
    else:
        return None


device_type = get_devtype().type


def apply_reordering_and_get_graph(graph, out_li) -> None:
    # [FUNCTION DOC] HELPER: Post-grad custom pass for tests. Reads the distributed optimization
    # config keys (bucketing, prefetch, estimator, overlap deps, bucket mode, FSDP pre-bucketing
    # and so on), forwards only non-None values to schedule_overlap_bucketing on the graph's
    # module, lints the graph, and appends its string form to out_li. Returns nothing.
    gm = graph.owning_module
    from torch._inductor.config import aten_distributed_optimizations as dist_opts
    from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing

    # Read config values, only pass non-None values to use function defaults
    kwargs: dict[str, object] = {}
    config_keys = (
        "collective_bucketing",
        "max_compute_pre_fetch",
        "custom_runtime_estimation",
        "insert_overlap_deps",
        "collective_estimator",
        "bucket_exposed_first",
        "bucket_only_internode_comms",
        "bucket_mode",
        "pre_bucketing_fsdp_collectives",
        "pre_bucketing_fsdp_collectives_bucket_cap_mb",
    )
    for key in config_keys:
        if (val := getattr(dist_opts, key)) is not None:
            kwargs[key] = val

    schedule_overlap_bucketing(gm, **kwargs)
    gm.graph.lint()
    out_li.append(str(gm.graph))


def run_and_get_aten_graph(fn, *inputs):
    # [FUNCTION DOC] HELPER: Call fn on inputs with the reordering pass installed as
    # post_grad_custom_post_pass and return (output, string of the rewritten aten graph).
    # Only the first graph captured is returned; raises IndexError if no graph was compiled.
    li = []
    apply = functools.partial(apply_reordering_and_get_graph, out_li=li)
    with torch._inductor.config.patch(post_grad_custom_post_pass=apply):
        out = fn(*inputs)

    return out, li[0]


def get_patches():
    # [FUNCTION DOC] HELPER: Inductor config patch dict for these tests: uses the simple test
    # runtime estimator, disables locality reordering, native matmul, legacy overlap passes,
    # caches and overlap deps insertion, uses one compile thread, and turns off the latency
    # assumption for bucketing, so graphs are deterministic.
    return {
        "aten_distributed_optimizations.custom_runtime_estimation": estimate_aten_runtime,
        "reorder_for_locality": False,
        "triton.native_matmul": False,
        "reorder_for_compute_comm_overlap_passes": [],
        "compile_threads": 1,
        "force_disable_caches": True,
        # Messes up existing test strings
        "aten_distributed_optimizations.insert_overlap_deps": False,
        # interferes with testing, / custom estimation
        "test_configs.assume_bucketing_reduces_latency": False,
    }


@requires_accelerator_dist_backend()
# TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestComputeCommReorderingMultiProc(DynamoDistributedMultiProcTestCase):
    # [CLASS DOC] Base multi-process test case (world size forced to 2) running compiled
    # functions under a per-rank distributed init (fake process group if fewer than 2 GPUs).
    # Each test compiles a function with functional collectives, captures the post-grad aten
    # graph and checks the ordering of collectives, waits and matmuls plus overlap counters.
    # Skipped without accelerator distributed backend or GPU triton support.
    """
    Run correctness checks in multi-proc runner, mark with minimum # GPUs to run under

    Note: these tests are a fork of test/distributed/test_compute_comm_reordering.py

    """

    def setUp(self):
        # [FUNCTION DOC] Reset dynamo state and clear the dynamo counters before each test.
        super().setUp()
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()

    def get_world_trs(self):
        # [FUNCTION DOC] HELPER: Return the process-group kwargs (empty tag, rank list, group size)
        # for functional collectives.
        return {
            "tag": "",
            "ranks": list(range(self.world_size)),
            "group_size": self.world_size,
        }

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Property: always 2 regardless of available GPUs.
        # hack: no matter whether we have 2 or 3 or 4 gpus, just run on 2
        # works around issue with skipif<2 and workers with unpredictable #s gpu
        return 2

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_patches())
    def test_sink_waits(self):
        # [FUNCTION DOC] Compiled all_reduce followed by independent matmul then use of the result:
        # checks the wait_tensor is sunk below the first matmul and above the second, output
        # matches eager, and no exposed collectives are counted.
        def func(a):
            # [FUNCTION DOC] HELPER: all_reduce a, matmul a with itself, then matmul the
            # reduced tensor with that product.
            ar = _functional_collectives.all_reduce(a, "sum", "0")
            b = torch.matmul(a, a)
            return torch.matmul(ar, b)

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank

            out, aten_graph_str = run_and_get_aten_graph(torch.compile(func), inputs)

            # Verify that the wait_tensor is sinked below the 1st matmul but
            # above the 2nd matmul.
            (
                FileCheck()
                .check("all_reduce.default")
                .check("aten.mm.default")
                .check("wait_tensor.default")
                .check("aten.mm.default")
                .run(aten_graph_str)
            )
            correct = func(inputs)
            self.assertTrue(same(out, correct))
            self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 0)

    @torch._inductor.config.patch(get_patches())
    def test_raise_comms(self):
        # [FUNCTION DOC] Checks the all_reduce is raised after the first matmul (and relu) but
        # before the second, with the wait before the last matmul; result matches eager and no
        # exposed collectives.
        def func(a):
            # [FUNCTION DOC] HELPER: Chain of matmul, relu, matmul plus an all_reduce of the first
            # matmul output plus one, combined by a final matmul.
            b = torch.matmul(a, a)
            c = torch.relu(b)
            d = torch.matmul(c, c)
            e = _functional_collectives.all_reduce((b + 1), "sum", "0")
            return torch.matmul(d, e)

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            compiled = torch.compile(func)
            out, aten_graph_str = run_and_get_aten_graph(torch.compile(func), inputs)
            # Verify that the all_reduce_ has been raised above the 2nd matmul
            # but below the 1st matmul. Note that the all_reduce_ directly
            # writes to the output buffer of the 1st matmul, which is an input
            # to the first relu. Therefore, the all_reduce_ should be scheduled
            # after the first relu.
            (
                FileCheck()
                .check("aten.mm")
                .check("all_reduce.default")
                .check("aten.mm")
                .check("wait_tensor.default")
                .check("aten.mm")
                .run(aten_graph_str)
            )
            out = compiled(inputs)
            correct = func(inputs)
            self.assertTrue(same(out, correct))
            self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 0)

    @torch._inductor.config.patch(get_patches())
    def test_sink_waits_raise_comms(self):
        # [FUNCTION DOC] Checks, with an exact expected aten graph string, that the all_reduce is
        # raised just after its producer matmul and the wait sunk below the third matmul; the
        # result matches eager and no exposed collectives.
        def func(a, *, tag, ranks, group_size):
            # [FUNCTION DOC] HELPER: Matmul chain with an all_reduce of the first matmul consumed
            # by the final mm. tag, ranks, group_size are accepted but unused.
            b = torch.matmul(a, a)
            c = torch.relu(b)
            d = torch.matmul(c, c)
            e = _functional_collectives.all_reduce(b, "sum", "0")
            f = torch.relu(d)
            g = torch.matmul(f, f)
            return torch.mm(e, g)

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(
                4, 4, dtype=torch.float, device=device_type
            )  # + self.rank
            kwargs = self.get_world_trs()
            func = functools.partial(func, **kwargs)
            compiled = torch.compile(func)
            out, aten_graph_str = run_and_get_aten_graph(compiled, inputs)
            # Things to verify:
            # - The all_reduce_ and its prologue should be raised above the 2nd
            # matmul but below the 1st matmul.
            # - The wait_tensor should be sinked below the 3rd matmul but above
            # the 4th matmul.

            self.assertExpectedInline(
                aten_graph_str,
                """\
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %mm : [num_users=2] = call_function[target=torch.ops.aten.mm.default](args = (%arg0_1, %arg0_1), kwargs = {})
    %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%mm,), kwargs = {})
    %all_reduce : [num_users=1] = call_function[target=torch.ops._c10d_functional.all_reduce.default](args = (%mm, sum, 0), kwargs = {})
    %mm_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%relu, %relu), kwargs = {})
    %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%mm_1,), kwargs = {})
    %mm_2 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%relu_1, %relu_1), kwargs = {})
    %wait_tensor : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor.default](args = (%all_reduce,), kwargs = {})
    %mm_3 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%wait_tensor, %mm_2), kwargs = {})
    return (mm_3,)""",
            )

            # Note: this triggered an all_reduce_ bug
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))
            self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 0)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_patches())
    def test_schedulable_wait(self):
        # [FUNCTION DOC] Builds a graph with chained all_gather and wait nodes and checks that
        # _schedulable_wait_node is true only for the first two waits (names wait_tensor_default
        # and wait_tensor_default_1); raises AssertionError otherwise.
        """Test that if a wait node is scheduable or not."""
        from torch._inductor.fx_passes.bucketing import _schedulable_wait_node

        def test_graph():
            # [FUNCTION DOC] HELPER: Build an fx graph with placeholders, two all_gather ops each
            # followed by a wait, and an extra wait on the second wait, and return it.
            graph = fx.Graph()

            inp = graph.placeholder("inp")
            group_size = graph.placeholder("group_size")
            group_name = graph.placeholder("group_name")

            ag_0_out = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(inp, group_size, group_name),
            )
            ag_0_wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(ag_0_out,),
            )
            ag_1_out = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(ag_0_wait, group_size, group_name),
            )
            ag_1_wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(ag_1_out,),
            )
            ag_2_wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(ag_1_wait,),
            )

            graph.output(ag_2_wait)
            return graph

        graph = test_graph()
        schedulable = {"wait_tensor_default", "wait_tensor_default_1"}
        for node in list(graph.nodes):
            expected = node.name in schedulable
            if _schedulable_wait_node(node) is not expected:
                raise AssertionError(
                    f"Expected _schedulable_wait_node({node.name}) is {expected}"
                )

    @torch._inductor.config.patch(get_patches())
    def test_reorder_compute_for_overlap_mul(self):
        # [FUNCTION DOC] Two dependent all_reduces with equal cost for collectives and mm: checks
        # each collective overlaps one matmul, the schedule order, exactly one exposed collective
        # and that output matches eager.
        def func(a, *, tag, ranks, group_size):
            # [FUNCTION DOC] HELPER: all_reduce a, compute matmuls and elementwise ops, a second
            # all_reduce of an elementwise result, then a final matmul; returns a tuple.
            ar = _functional_collectives.all_reduce(a, "sum", ranks, tag)
            g = torch.matmul(a, a)
            c = torch.relu(a)
            d = torch.matmul(c, c)
            f = d * c * ar
            fr = _functional_collectives.all_reduce(f, "sum", ranks, tag)
            e = torch.matmul(d + ar + fr, g)
            return (e,)

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            func_c = functools.partial(func, **self.get_world_trs())
            compiled = torch.compile(func_c)
            out_c, aten_graph_str = run_and_get_aten_graph(compiled, inputs)
            # Note: because we have given collectives and mms equal estimation,
            # we overlap each collective with a single mm.
            # Same schedule as in test_reorder_compute_for_overlap_custom_runtime_estimation
            # although there is an exposed collective
            (
                FileCheck()
                .check("all_reduce.default")
                .check("aten.mm")
                .check("aten.mm")
                .check("wait_tensor.default")
                .check("aten.mul")
                .check("all_reduce.default")
                .check("wait_tensor.default")
                .check("aten.mm")
                .run(aten_graph_str)
            )
            correct = func(inputs, **self.get_world_trs())
            self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 1)
            self.assertTrue(same(out_c, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skipIfRocm
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @unittest.skipIf(True, "Logic not yet implemented")
    @torch._inductor.config.patch(get_patches())
    def test_grouped_scheduler_node(self):
        # [FUNCTION DOC] Unconditionally skipped (logic not implemented). Intended to check that
        # ops in a grouped scheduler node with an all_reduce fuse with each other but not outside
        # ops, using the generated triton code.
        def func(a, *, tag, ranks, group_size):
            # [FUNCTION DOC] HELPER: add and divide feeding an all_reduce, a separate mul, and a
            # matmul of mul with the reduced tensor.
            add = a + a
            div = add / a
            ar = _functional_collectives.all_reduce(div, "sum", ranks, tag)
            # Normally, we would fuse `add = a + a`, `div = add / a` and `mul = a * a` together into a single fused op,
            # but here in this unit test, we intentionally put `add`, `div` and `ar` computation
            # into a GroupedSchedulerNode, which prevents them from being fused with any other ops.
            mul = a * a
            mm = torch.matmul(mul, ar)
            return (mm,)

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # Expectations:
            # 1. `add = a + a` and `div = add / a` are still fused, which means fusion
            #    still happens among nodes within a GroupedSchedulerNode.
            # 2. `mul = a * a` is not fused with `add` or `div`, because the latter two are within
            #    GroupedSchedulerNode and thus are prevented from being fused with any outside ops.
            FileCheck().check("triton_poi_fused_add_all_reduce_div_0.").check(
                "_c10d_functional.all_reduce_."
            ).check("triton_poi_fused_mul_1.").run(code)
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_patches())
    def test_inductor_default_comms_ordering(self):
        # [FUNCTION DOC] Three independent all_reduces waited in reverse order, on a fake process
        # group: checks the collectives stay in program order (by argument names), three exposed
        # collectives and no bad exposed ones.
        pg_info = self.get_world_trs()
        tag = pg_info["tag"]
        ranks = pg_info["ranks"]
        group_size = pg_info["group_size"]

        g1 = torch.ones(10, 10, device=device_type)
        g2 = torch.ones(11, 11, device=device_type)
        g3 = torch.ones(12, 12, device=device_type)

        @torch.compile
        def fn(g1, g2, g3):
            # [FUNCTION DOC] HELPER: Launch three avg all_reduces then wait in reverse order;
            # return the three waited tensors.
            handle1 = torch.ops.c10d_functional.all_reduce(
                g1, "avg", tag, ranks, group_size
            )
            handle2 = torch.ops.c10d_functional.all_reduce(
                g2, "avg", tag, ranks, group_size
            )
            handle3 = torch.ops.c10d_functional.all_reduce(
                g3, "avg", tag, ranks, group_size
            )

            # wait on them in a different order
            grad3 = torch.ops._c10d_functional.wait_tensor.default(handle3)
            grad2 = torch.ops._c10d_functional.wait_tensor.default(handle2)
            grad1 = torch.ops._c10d_functional.wait_tensor.default(handle1)
            return grad3, grad2, grad1

        with _dynamo_dist_per_rank_init(
            self.rank, self.world_size, self.backend(device_type), fake_pg=True
        ):
            # all_reduces remain in order!
            # note: this isn't actually invariant of pass currently..
            # but we should keep collectives stable without reordering opportunities

            _, code = run_and_get_aten_graph(fn, g1, g2, g3)

            FileCheck().check("all_reduce").check_same("arg0_1").check(
                "all_reduce"
            ).check_same("arg1_1").check("all_reduce").check_same("arg2_1").run(code)
            self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 3)
            # these have no overlap opportunities
            self.assertEqual(counters["inductor"]["overlap_scheduling_bad_exposed"], 0)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_overlap_scheduling_via_config(self):
        # [FUNCTION DOC] Enables overlap scheduling only through the config flag, compiles with
        # normal inductor codegen and checks the generated code has the wait after a matmul;
        # output matches eager and no exposed collectives.
        """Test overlap scheduling enabled via config in post_grad pass."""

        def func(a):
            # [FUNCTION DOC] HELPER: all_reduce a, matmul a with itself, matmul the two.
            ar = _functional_collectives.all_reduce(a, "sum", "0")
            b = torch.matmul(a, a)
            return torch.matmul(ar, b)

        patches = {
            **get_patches(),
            "aten_distributed_optimizations.enable_overlap_scheduling": True,
        }

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank

            with torch._inductor.config.patch(patches):
                compiled_func = torch.compile(func)
                out, code = run_and_get_code(compiled_func, inputs)

                # Verify that wait_tensor is sinked below matmul
                FileCheck().check("all_reduce").check("mm").check("wait_tensor").check(
                    "mm"
                ).run(code[0])

                correct = func(inputs)
                self.assertTrue(same(out, correct))
                self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 0)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_overlap_scheduling_flex_attention_backward(self):
        # [FUNCTION DOC] Compiles forward and backward of flex attention whose score_mod uses an
        # all_reduce result, with a custom inner compile that only runs post-grad passes and
        # returns zero outputs. Checks two graphs were produced, one contains flex attention
        # backward, one contains control_deps, and all input gradients exist.
        def func(q, k, v, bias):
            # [FUNCTION DOC] HELPER: all_reduce bias and run flex attention with a score_mod adding
            # the reduced bias; returns the sum of the output.
            ar = _functional_collectives.all_reduce(bias, "sum", "0")

            def score_mod(score, b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: score_mod adding the all_reduced bias at (q_idx, kv_idx).
                return score + ar[q_idx, kv_idx]

            return flex_attention(q, k, v, score_mod=score_mod).sum()

        patches = {
            **get_patches(),
            "aten_distributed_optimizations.enable_overlap_scheduling": True,
            "aten_distributed_optimizations.insert_overlap_deps": True,
            "aten_distributed_optimizations.pre_bucketing_fsdp_collectives": False,
        }
        compiled_graphs = []

        def materialize_output(gm):
            # [FUNCTION DOC] HELPER: Build concrete zero tensors matching the fake output values of
            # the graph's output node (non-tensors are passed through).
            output = gm.graph.output_node().args[0]
            fake_output = pytree.tree_map(lambda x: get_fake(x, gm), output)

            def materialize(x):
                # [FUNCTION DOC] HELPER: Zero tensor with same shape, device and dtype for tensors,
                # other values unchanged.
                if isinstance(x, torch.Tensor):
                    return torch.zeros(tuple(x.shape), device=x.device, dtype=x.dtype)
                return x

            return pytree.tree_map(materialize, fake_output)

        def compile_fx_inner(gm, example_inputs, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Fake inner_compile for the backend: detects the fake mode from
            # example inputs (AssertionError if none), runs the recursive post-grad passes under
            # it, records the graph module in compiled_graphs and returns a boxed function
            # returning pre-built zero outputs.
            fake_mode = detect_fake_mode(example_inputs)
            if fake_mode is None:
                raise AssertionError("expected fake mode from example inputs")
            with V.set_fake_mode(fake_mode):
                _recursive_post_grad_passes(
                    gm, is_inference=kwargs.get("is_inference", False)
                )
            compiled_graphs.append(gm)
            output = materialize_output(gm)
            return make_boxed_func(lambda *args: output)

        backend = functools.partial(
            compile_fx.compile_fx, inner_compile=compile_fx_inner
        )

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            q = torch.randn(1, 1, 128, 16, device=device_type, requires_grad=True)
            k = torch.randn(1, 1, 128, 16, device=device_type, requires_grad=True)
            v = torch.randn(1, 1, 128, 16, device=device_type, requires_grad=True)
            bias = torch.randn(128, 128, device=device_type, requires_grad=True)

            with torch._inductor.config.patch(patches):
                loss = torch.compile(func, backend=backend, fullgraph=True)(
                    q, k, v, bias
                )
                loss.backward()

            self.assertEqual(len(compiled_graphs), 2)
            self.assertTrue(
                any("flex_attention_backward" in gm.code for gm in compiled_graphs)
            )
            self.assertTrue(any("control_deps" in gm.code for gm in compiled_graphs))
            self.assertIsNotNone(q.grad)
            self.assertIsNotNone(k.grad)
            self.assertIsNotNone(v.grad)
            self.assertIsNotNone(bias.grad)

    @torch._inductor.config.patch(get_patches())
    def test_custom_estimator_for_non_compute_nodes(self):
        # [FUNCTION DOC] With a custom estimator giving relu a cost equal to a collective, checks
        # an independent all_reduce is scheduled before relu and waited after, with no exposed
        # collectives and eager-matching output.
        """Test that non-compute nodes with custom runtime estimates can trigger collective prefetching."""

        def custom_estimator_with_relu(fx_node, override_size=None):
            # [FUNCTION DOC] HELPER: Estimator returning 1.0 for collectives and relu,
            # and None for any other node.
            """Custom estimator that provides runtime for relu."""
            # Collective ops
            if "c10" in str(fx_node.target):
                return 1.0
            # Non-compute ops that we want to overlap
            elif fx_node.target == aten.relu.default:
                return 1.0  # relu has same time as collective
            else:
                return None

        def func(a, b):
            # [FUNCTION DOC] HELPER: relu and matmul of a, an independent all_reduce of b, product.
            c = torch.relu(a)
            d = torch.mm(c, c)

            # Collective that is independent and should be prefetched during relu
            ar = _functional_collectives.all_reduce(b, "sum", "0")

            # Use both results
            return d * ar

        patches = {
            **get_patches(),
            "aten_distributed_optimizations.custom_runtime_estimation": custom_estimator_with_relu,
        }

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs_a = (
                torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            )
            inputs_b = torch.ones(4, 4, dtype=torch.float, device=device_type) * 2

            with torch._inductor.config.patch(patches):
                out, aten_graph_str = run_and_get_aten_graph(
                    torch.compile(func), inputs_a, inputs_b
                )

                # Verify that all_reduce is prefetched to run concurrently with relu
                # The collective should start before relu completes to enable perfect overlap
                FileCheck().check("all_reduce").check("relu").check("wait_tensor").run(
                    aten_graph_str
                )

                correct = func(inputs_a, inputs_b)
                self.assertTrue(same(out, correct))
                self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 0)


def get_bucket_patches(compute_multiplier=1.0):
    # [FUNCTION DOC] HELPER: Config patch dict for bucketing tests: custom estimator with the
    # given compute multiplier, bucketing on, bucket-exposed-first and internode-only off,
    # no locality or legacy overlap passes, one compile thread, no caches, no overlap deps,
    # and FSDP pre-bucketing disabled.
    estimate_aten_runtime_part = functools.partial(
        estimate_aten_runtime, compute_multiplier=compute_multiplier
    )
    return {
        "aten_distributed_optimizations.custom_runtime_estimation": estimate_aten_runtime_part,
        "aten_distributed_optimizations.collective_bucketing": True,
        "aten_distributed_optimizations.bucket_exposed_first": False,
        "aten_distributed_optimizations.bucket_only_internode_comms": False,
        "reorder_for_locality": False,
        "triton.native_matmul": False,
        "reorder_for_compute_comm_overlap_passes": [],
        "compile_threads": 1,
        "force_disable_caches": True,
        # messes up test strings
        "aten_distributed_optimizations.insert_overlap_deps": False,
        # interferes with testing, / custom estimation
        "test_configs.assume_bucketing_reduces_latency": False,
        # these tests verify overlap scheduling bucketing, not pre-bucketing
        "aten_distributed_optimizations.pre_bucketing_fsdp_collectives": False,
    }


class TestComputeCommReorderingBucketing(TestComputeCommReorderingMultiProc):
    # [CLASS DOC] Tests that independent collectives are (or are not) merged into a bucket by
    # the overlap scheduling pass. Inherits the 2-rank setup from the reordering test class
    # and so also re-runs its inherited tests.
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_basic_all_gather_bucketing(self):
        # [FUNCTION DOC] Three independent all_gathers must become a single bucketed all_gather
        # in the aten graph; result matches eager.
        """Test that independent all_gather operations get bucketed together."""

        def func(a, b, c, *, ranks):
            # [FUNCTION DOC] HELPER: Three all_gather_singles each plus a constant, summed.
            # Three independent all_gathers that should be bucketed
            ag1 = _functional_collectives.all_gather_single(a, 0, ranks) + 3
            ag2 = _functional_collectives.all_gather_single(b, 0, ranks) + 4
            ag3 = _functional_collectives.all_gather_single(c, 0, ranks) + 5
            return ag1 + ag2 + ag3

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs_a = (
                torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            )
            inputs_b = torch.ones(4, 4, dtype=torch.float, device=device_type) * 2
            inputs_c = torch.ones(4, 4, dtype=torch.float, device=device_type) * 3
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(
                compiled, inputs_a, inputs_b, inputs_c
            )

            # Should see a single bucketed all_gather
            FileCheck().check_count(
                "torch.ops._c10d_functional.all_gather_into_tensor", 1, exactly=True
            ).run(aten_graph_str)

            correct = func(inputs_a, inputs_b, inputs_c, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_reduce_scatter_bucketing(self):
        # [FUNCTION DOC] Three reduce_scatters must become one reduce_scatter_tensor in the graph.
        # The eager comparison is commented out (known to fail on CI).
        """Test bucketing of reduce_scatter operations."""

        def func(a, b, c):
            # [FUNCTION DOC] HELPER: Three reduce_scatter_singles, concatenated.
            rs1 = _functional_collectives.reduce_scatter_single(a, "sum", 0, "0")
            rs2 = _functional_collectives.reduce_scatter_single(b, "sum", 0, "0")
            rs3 = _functional_collectives.reduce_scatter_single(c, "sum", 0, "0")
            return torch.cat([rs1, rs2, rs3])

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs_a = torch.ones(8, 4, dtype=torch.float, device=device_type)
            inputs_b = torch.ones(8, 4, dtype=torch.float, device=device_type) * 2
            inputs_c = torch.ones(8, 4, dtype=torch.float, device=device_type) * 3

            out, aten_graph_str = run_and_get_aten_graph(
                torch.compile(func), inputs_a, inputs_b, inputs_c
            )

            # Should bucket reduce_scatter ops
            FileCheck().check_count(
                "torch.ops._c10d_functional.reduce_scatter_tensor", 1, exactly=True
            ).run(aten_graph_str)

            # TODO: debug - on ci this fails.
            # correct = func(inputs_a, inputs_b, inputs_c)
            # self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_no_bucketing_with_dependent_hiding_nodes(self):
        # [FUNCTION DOC] When the matmul hiding the second all_gather depends on the first
        # all_gather, the two must stay separate (two all_gathers); output matches eager.
        """Test that collectives with dependent hiding nodes don't get bucketed."""

        def func(a, b, *, ranks):
            # [FUNCTION DOC] HELPER: all_gather a, matmul a, matmul on a slice of the first gather,
            # all_gather b, combine all results.
            # ag1 could be hidden by mm1
            ag1 = _functional_collectives.all_gather_single(a, 0, ranks)
            mm1 = torch.matmul(a, a)

            # ag2 can be hidden by mm2, but mm2 depends on ag1's result
            # ag2 start
            mm2 = torch.matmul(ag1[:4], b)
            # ag2 end
            ag2 = _functional_collectives.all_gather_single(b, 0, ranks)

            return ag1.sum() * ag2.sum() * mm1 * mm2

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs_a = torch.ones(4, 4, dtype=torch.float, device=device_type)
            inputs_b = torch.ones(4, 4, dtype=torch.float, device=device_type)
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, inputs_a, inputs_b)

            # mm2 depends on ag1, so if mm2 is to hide ag2, we can't bucket ag1 and ag2
            # because that would create a dependency issue, even though we could bucket them
            FileCheck().check_count(
                "torch.ops._c10d_functional.all_gather_into_tensor", 2, exactly=True
            ).run(aten_graph_str)

            correct = func(inputs_a, inputs_b, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_no_bucketing_when_collective_depends_on_hiding_node(self):
        # [FUNCTION DOC] When the second all_gather input depends on the matmul hiding the first,
        # they must not be bucketed (two all_gathers remain); output matches eager.
        """Test that collectives don't get bucketed when one depends on another's hiding node."""

        def func(a, *, ranks):
            # [FUNCTION DOC] HELPER: all_gather a, matmul a, all_gather of twice the
            # matmul, combine.
            # ag1 hidden by mm1
            ag1 = _functional_collectives.all_gather_single(a, 0, ranks)
            mm1 = torch.matmul(a, a)

            # ag2 depends on mm1 (which hides ag1)
            b = mm1 * 2
            ag2 = _functional_collectives.all_gather_single(b, 0, ranks)

            return ag1.sum() * ag2.sum() * mm1

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type)
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, inputs)

            # ag2 depends on mm1 (ag1's hiding node), so they can't be bucketed
            FileCheck().check_count(
                "_c10d_functional.all_gather_into_tensor", 2, exactly=True
            ).run(aten_graph_str)

            correct = func(inputs, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches(2.0))
    def test_bucketing_wait_sink(self):
        # [FUNCTION DOC] Four independent all_gathers with two matmuls (compute multiplier 2):
        # expects one bucketed all_gather, both matmuls, then the wait; output matches eager.
        """Test that 4 independent all-gathers split bucketed."""

        def func(a, b, c, d, *, ranks):
            # [FUNCTION DOC] HELPER: Four all_gathers of a, b and slices of c and d, two
            # independent matmuls, and a weighted sum of every result.
            # All 4 all-gathers are independent - COULD be bucketed together
            ag1 = _functional_collectives.all_gather_single(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_single(b, 0, ranks)
            ag3 = _functional_collectives.all_gather_single(c[:4], 0, ranks)
            ag4 = _functional_collectives.all_gather_single(d[:4], 0, ranks)

            # First compute - can hide ag1 and ag2
            e = a * 5
            mm1 = torch.matmul(e, e.T)

            # Second compute - can hide ag3 and ag4
            f = b * 6
            mm2 = torch.matmul(f, f.T)

            # Use all collective results
            result = (
                ag1.sum() * 1.1
                + ag2.sum() * 1.2
                + ag3.sum() * 1.3
                + ag4.sum() * 1.4
                + mm1.sum()
                + mm2.sum()
            )

            return result

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            d = torch.ones(8, 8, dtype=torch.float, device=device_type) * 4
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c, d)

            # The 4 all gathers can be bucketed, and their waits should be sunk below the mms
            FileCheck().check_count(
                "_c10d_functional.all_gather_into_tensor", 1, exactly=True
            ).check_count("ops.aten.mm", 2, exactly=True).check(
                "_c10d_functional.wait_tensor"
            ).run(aten_graph_str)

            correct = func(a, b, c, d, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches(2.0))
    def test_bucketing_split_for_overlap(self):
        # [FUNCTION DOC] Four independent all_gathers where the second matmul depends on the first
        # two gathers: expects two bucketed all_gathers (out variant) ordered bucket, mm, bucket,
        # mm; output matches eager.
        """Test that 4 independent all-gathers split into 2+2 buckets for better overlap with compute."""

        def func(a, b, c, d, *, ranks):
            # [FUNCTION DOC] HELPER: Four all_gathers, a matmul, a padded small slice combining the
            # first two gathers into the second matmul, and a weighted sum of all results.
            # All 4 all-gathers are independent - COULD be bucketed together
            ag1 = _functional_collectives.all_gather_single(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_single(b, 0, ranks)
            ag3 = _functional_collectives.all_gather_single(c[:4], 0, ranks)
            ag4 = _functional_collectives.all_gather_single(d[:4], 0, ranks)

            # First compute - can hide ag1 and ag2
            e = a * 5  # Use a to avoid fusion
            mm1 = torch.matmul(e, e.T)

            # Force ag1/ag2 to complete before mm2 (but ag3/ag4 can still be deferred)
            intermediate = ag1[:2, :2] + ag2[:2, :2]  # Small slice to minimize compute

            # Second compute - depends on ag1/ag2 through intermediate, can hide ag3/ag4
            f = b * 6
            # Expand intermediate to match mm1's shape for broadcasting
            intermediate_expanded = torch.nn.functional.pad(intermediate, (0, 6, 0, 6))
            mm2 = torch.matmul(mm1 + intermediate_expanded, f.T)

            # Use all results
            result = (
                ag1.sum() * 1.1
                + ag2.sum() * 1.2
                + ag3.sum() * 1.3
                + ag4.sum() * 1.4
                + mm1.sum()
                + mm2.sum()
            )

            return result

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            d = torch.ones(8, 8, dtype=torch.float, device=device_type) * 4
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c, d)

            # Should have 2 bucketed all-gathers (one for ag1+ag2, one for ag3+ag4)
            FileCheck().check_count(
                "_c10d_functional.all_gather_into_tensor_out", 2, exactly=True
            ).run(aten_graph_str)

            # Verify the ordering - first bucket, then mm1, then second bucket, then mm2
            FileCheck().check("_c10d_functional.all_gather_into_tensor_out").check(
                "ops.aten.mm"
            ).check("_c10d_functional.all_gather_into_tensor_out").check(
                "ops.aten.mm"
            ).run(aten_graph_str)

            # Verify correctness
            correct = func(a, b, c, d, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_bucket_exposed_with_hidden_single_overlap(self):
        # [FUNCTION DOC] One hidden and two exposed all_gathers should bucket into one (single
        # wait) that still overlaps the matmul; output matches eager.
        """Test that exposed and hidden collectives bucket together when overlap is preserved."""

        def func(a, b, c, *, ranks):
            # [FUNCTION DOC] HELPER: Three all_gathers and one small matmul that can hide only one.
            # ag1 will be hidden by mm1
            ag1 = _functional_collectives.all_gather_single(a, 0, ranks)

            # ag2 and ag3 are exposed (no compute to hide them)
            ag2 = _functional_collectives.all_gather_single(b, 0, ranks)
            ag3 = _functional_collectives.all_gather_single(c, 0, ranks)

            # can only hide one collective
            mm1 = torch.matmul(a[:2], a[:2].T)  # 2x2 matmul, hides only ag1

            # All three can bucket together because:
            # bucketing ag1, ag2, ag3 together does not prevent ag1 being hidden by mm1.

            return ag1.sum() + ag2.sum() + ag3.sum() + mm1.sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c)

            # Should have 1 bucketed operation containing all 3 all-gathers
            FileCheck().check_count("wait_tensor.default", 1, exactly=True).run(
                aten_graph_str
            )

            # Verify bucketed collective overlaps with mm1
            FileCheck().check("functional.all_gather_into_tensor").check(
                "aten.mm"
            ).check("wait_tensor").run(aten_graph_str)

            # Verify correctness
            correct = func(a, b, c, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches(2.0))
    def test_bucketing_split_for_overlap_blocking_deps_inductor(self):
        # [FUNCTION DOC] Same split-bucket scenario as the split-for-overlap test but with overlap
        # deps insertion on: checks control_deps nodes in the aten graph and, in the generated
        # code, the order of all_gather out, mm and wait twice; output matches eager.
        """Test that 4 independent all-gathers split into 2+2 buckets for better overlap with compute."""

        # check that ordering is preserved in inductor

        def func(a, b, c, d, *, ranks):
            # [FUNCTION DOC] HELPER: Four all_gathers, two dependent matmuls and weighted
            # sum of results.
            # All 4 all-gathers are independent - COULD be bucketed together
            ag1 = _functional_collectives.all_gather_single(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_single(b, 0, ranks)
            ag3 = _functional_collectives.all_gather_single(c[:4], 0, ranks)
            ag4 = _functional_collectives.all_gather_single(d[:4], 0, ranks)

            # First compute - can hide ag1 and ag2
            e = a * 5  # Use a to avoid fusion
            mm1 = torch.matmul(e, e.T)

            # Force ag1/ag2 to complete before mm2 (but ag3/ag4 can still be deferred)
            # Use first 8x8 elements to match mm1's shape
            intermediate = ag1[:8, :8] + ag2[:8, :8]

            # Second compute - depends on ag1/ag2 through intermediate, can hide ag3/ag4
            mm2 = torch.matmul(mm1 + intermediate, c[:8])

            # Use all results
            result = (
                ag1.sum() * 1.1
                + ag2.sum() * 1.2
                + ag3.sum() * 1.3
                + ag4.sum() * 1.4
                + mm1.sum()
                + mm2.sum()
            )
            return result

        li = []
        apply = functools.partial(apply_reordering_and_get_graph, out_li=li)
        with (
            _dynamo_dist_per_rank_init(
                self.rank,
                self.world_size,
                self.backend(device_type),
                fake_pg=not at_least_x_gpu(2),
            ),
            torch._inductor.config.patch(
                "aten_distributed_optimizations.insert_overlap_deps", True
            ),
            torch._inductor.config.patch(post_grad_custom_post_pass=apply),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            d = torch.ones(8, 8, dtype=torch.float, device=device_type) * 4
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            test_out, (code,) = run_and_get_code(compiled, a, b, c, d)

            # Check that right deps are added
            f = FileCheck()
            for _ in range(2):
                f.check("control_deps").check_same("all_gather").check_same(
                    "subgraph_mm"
                )
                f.check("control_deps").check_same("mm").check_same("subgraph_wait")
            f.run(li[0])

            f = FileCheck()
            for _ in range(2):
                f.check_count("all_gather_into_tensor_out.default(", 1, exactly=True)
                f.check_count("extern_kernels.mm(", 1, exactly=True)
                f.check_count("wait_tensor.default(", 1, exactly=True)
            f.run(code)

            correct = func(a, b, c, d, ranks=ranks)
            self.assertTrue(same(test_out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_custom_estimation_with_fake_tensor_mode(self):
        # [FUNCTION DOC] A custom estimator creating a FakeTensorMode inside must still be called
        # (counted via a nonlocal counter); output matches eager.
        """Test that custom estimation can use FakeTensorMode for analysis."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        estimation_calls = 0

        def estimate_with_fake_mode(fx_node, compute_multiplier=1.0):
            # [FUNCTION DOC] HELPER: Estimator counting calls, asserting that a tensor created
            # under a fresh FakeTensorMode is fake (AssertionError otherwise); returns 1.0.
            with FakeTensorMode():
                nonlocal estimation_calls
                estimation_calls += 1
                if not isinstance(torch.rand([20]), torch._subclasses.FakeTensor):
                    raise AssertionError("Expected FakeTensor")

            return 1.0

        patches = get_bucket_patches()
        patches["aten_distributed_optimizations.custom_runtime_estimation"] = (
            estimate_with_fake_mode
        )

        def func(a, b, *, ranks):
            # [FUNCTION DOC] HELPER: Two all_gathers and a matmul, summed.
            # Two independent all_gathers that should be bucketed
            ag1 = _functional_collectives.all_gather_single(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_single(b, 0, ranks)

            # Matmul that can hide the collectives
            mm1 = torch.matmul(a, a)

            return ag1.sum() + ag2.sum() + mm1.sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs_a = torch.ones(4, 4, dtype=torch.float, device=device_type)
            inputs_b = torch.ones(4, 4, dtype=torch.float, device=device_type) * 2
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            with torch._inductor.config.patch(patches):
                compiled = torch.compile(func_c)
                out, aten_graph_str = run_and_get_aten_graph(
                    compiled, inputs_a, inputs_b
                )

            # Verify the custom estimation was called
            self.assertTrue(
                estimation_calls > 0, "Custom estimation should have been called"
            )

            correct = func(inputs_a, inputs_b, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_collective_benchmarking_with_real_pg(self):
        # [FUNCTION DOC] Compiles all_reduce, all_gather and reduce_scatter with the benchmark
        # collective estimator and no custom estimator; checks all three appear in the graph and
        # output matches eager.
        """Test collective benchmarking with real process group (falls back on fake)."""

        def func(a):
            # [FUNCTION DOC] HELPER: all_reduce, all_gather, reduce_scatter of a plus two
            # matmuls, with all results summed.
            # Test all three collective types with 8x8 (power of 2 size = 256 elements = 1024 bytes for fp32)
            ar = _functional_collectives.all_reduce(a, "sum", "0")
            ag = _functional_collectives.all_gather_single(
                a, 0, list(range(self.world_size))
            )
            rs = _functional_collectives.reduce_scatter_single(a, "sum", 0, "0")

            b = torch.matmul(a, a)
            c = torch.matmul(ar, b)
            return c.sum() + ag.sum() + rs.sum()

        patches = {
            **get_patches(),
            "aten_distributed_optimizations.collective_estimator": "benchmark",
            "aten_distributed_optimizations.custom_runtime_estimation": None,  # Remove custom estimation so benchmarking happens
        }

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(8, 8, dtype=torch.float, device=device_type) + self.rank

            with torch._inductor.config.patch(patches):
                compiled = torch.compile(func)
                out, aten_graph_str = run_and_get_aten_graph(compiled, inputs)

                # Verify all three collective types are present
                FileCheck().check_dag("all_reduce").check_dag("all_gather").check_dag(
                    "reduce_scatter"
                ).run(aten_graph_str)

                # Test passes if compilation succeeded with benchmarking enabled
                # Cache verification is tricky due to multiprocess test setup
                correct = func(inputs)
                self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {
            **get_bucket_patches(),
            "aten_distributed_optimizations.bucket_mode": "custom_ops_multidtype",
        }
    )
    def test_multidtype_bucketing(self):
        # [FUNCTION DOC] With multidtype bucket mode, float32 and float16 all_gathers must share
        # a single wait (one bucket).
        """Test that all_gathers with different dtypes get bucketed together."""

        def func(a, b, c, *, ranks):
            # [FUNCTION DOC] HELPER: Three all_gathers of mixed dtypes, summed.
            # Three all_gathers with different dtypes
            ag1 = _functional_collectives.all_gather_single(a, 0, ranks)  # float32
            ag2 = _functional_collectives.all_gather_single(b, 0, ranks)  # float16
            ag3 = _functional_collectives.all_gather_single(c, 0, ranks)  # float16

            # Use all results
            return ag1.sum() + ag2.sum() + ag3.sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(4, 4, dtype=torch.float32, device=device_type)
            b = torch.ones(4, 4, dtype=torch.float16, device=device_type) * 2
            c = torch.ones(4, 4, dtype=torch.float16, device=device_type) * 3
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c)

            # Should have 1 bucketed all_gather despite different dtypes
            FileCheck().check_count(
                "torch.ops._c10d_functional.wait_tensor.default", 1, exactly=True
            ).run(aten_graph_str)

            # Verify correctness
            correct = func(a, b, c, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_basic_all_reduce_bucketing(self):
        # [FUNCTION DOC] Three independent all_reduces must produce a single wait (bucketed).
        """Test that independent all_reduce operations get bucketed together."""

        def func(a, b, c):
            # [FUNCTION DOC] HELPER: Three all_reduces of a, b, c, summed.
            # Three independent all_reduces that should be bucketed
            ar1 = _functional_collectives.all_reduce(a, "sum", "0")
            ar2 = _functional_collectives.all_reduce(b, "sum", "0")
            ar3 = _functional_collectives.all_reduce(c, "sum", "0")

            return ar1.sum() + ar2.sum() + ar3.sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            b = torch.ones(4, 4, dtype=torch.float, device=device_type) * 2
            c = torch.ones(4, 4, dtype=torch.float, device=device_type) * 3

            compiled = torch.compile(func)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c)

            # Should see a single bucketed all_reduce
            FileCheck().check_count(
                "torch.ops._c10d_functional.wait_tensor.default", 1, exactly=True
            ).run(aten_graph_str)

            # Verify correctness
            correct = func(a, b, c)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(get_bucket_patches())
    def test_multiple_hiding_nodes_bucketing(self):
        # [FUNCTION DOC] With a half compute multiplier estimator, two all_gathers each hidden by
        # multiple matmuls bucket into one wait, scheduled before all three matmuls.
        """Test that collectives hidden by multiple compute ops can bucket together."""

        # Use 0.5 compute multiplier so each collective needs 2 matmuls to be fully hidden
        def estimate_with_half_compute(fx_node, override_size=None):
            # [FUNCTION DOC] HELPER: Estimator using the base estimate with multiplier 0.5.
            return estimate_aten_runtime(fx_node, override_size, compute_multiplier=0.5)

        def func(a, b, *, ranks):
            # [FUNCTION DOC] HELPER: Two all_gathers and three matmuls, summed.
            # Two all_gathers that will be hidden by multiple compute operations
            ag1 = _functional_collectives.all_gather_single(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_single(b, 0, ranks)

            # Multiple compute operations that can hide the collectives
            # With 0.5 multiplier: mm1 and mm2 together hide ag1, mm2 and mm3 together hide ag2
            mm1 = torch.matmul(a, a.T)
            mm2 = torch.matmul(b, b.T)
            mm3 = torch.matmul(a + b, (a + b).T)

            return ag1.sum() + ag2.sum() + mm1.sum() + mm2.sum() + mm3.sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)

            # Patch with custom estimation that uses 0.5 multiplier
            with torch._inductor.config.patch(
                {
                    "aten_distributed_optimizations.custom_runtime_estimation": estimate_with_half_compute
                }
            ):
                compiled = torch.compile(func_c)
                out, aten_graph_str = run_and_get_aten_graph(compiled, a, b)

            # Should have 1 bucketed all_gather (both ag1 and ag2 bucketed together)
            FileCheck().check_count(
                "torch.ops._c10d_functional.wait_tensor.default", 1, exactly=True
            ).run(aten_graph_str)

            # Verify bucketed collective is scheduled before all matmuls
            FileCheck().check("functional.all_gather_into_tensor").check(
                "aten.mm"
            ).check("aten.mm").check("aten.mm").check("wait_tensor").run(aten_graph_str)

            # Verify correctness
            correct = func(a, b, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {
            **get_bucket_patches(),
            "aten_distributed_optimizations.bucket_mode": "custom_ops_multidtype",
        }
    )
    def test_bucketing_with_convert_dtype(self):
        # [FUNCTION DOC] Multidtype bucketing with dtype conversions before all_gather: one wait,
        # no convert_element_type ops remain in the graph, and outputs match eager.
        """Test that all_gathers with dtype conversion get bucketed and produce correct results."""

        def func(a, b, c, d, *, ranks):
            # [FUNCTION DOC] HELPER: Convert a, b, d to float16, all_gather four tensors,
            # return all.
            # Convert inputs to float16 before all_gather
            a_fp16 = a.to(torch.float16)
            b_fp16 = b.to(torch.float16)
            d_fp16 = d.to(torch.float16)

            # Two all_gathers with converted dtypes
            ag1 = _functional_collectives.all_gather_single(a_fp16, 0, ranks)
            ag2 = _functional_collectives.all_gather_single(b_fp16, 0, ranks)

            # same dtype
            ag3 = _functional_collectives.all_gather_single(c, 0, ranks)
            ag4 = _functional_collectives.all_gather_single(d_fp16, 0, ranks)

            return ag1, ag2, ag3, ag4

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(4, 4, dtype=torch.float32, device=device_type)
            b = torch.ones(4, 4, dtype=torch.float64, device=device_type) * 2
            c = torch.ones(4, 4, dtype=torch.float16, device=device_type) * 3
            d = torch.ones(4, 4, dtype=torch.float64, device=device_type) * 4
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a, b, c, d)

            # Should have 1 bucketed all_gather (both ag1 and ag2 bucketed together)
            FileCheck().check_count(
                "torch.ops._c10d_functional.wait_tensor.default", 1, exactly=True
            ).run(aten_graph_str)

            # Verify convert_element_type ops are removed (dtype conversion handled by _pre_bucket_all_gather)
            FileCheck().check_not("torch.ops.prims.convert_element_type").run(
                aten_graph_str
            )

            # Verify correctness - this tests that dtype conversion is handled correctly
            correct = func(a, b, c, d, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {
            **get_bucket_patches(),
            "aten_distributed_optimizations.enable_overlap_scheduling": True,
        }
    )
    def test_uneven_sharding_spmd_graphs(self):
        # [FUNCTION DOC] With uneven DTensor-like shards (7 rows over 2 ranks), compiles pad,
        # all_gather and narrow, then gathers each rank's sorted list of call_function targets with
        # all_gather_object and checks they are identical, i.e. no-op pad was not removed on one
        # rank only.
        """Test that uneven DTensor sharding produces SPMD graphs across ranks.

        When a tensor dimension is not divisible by world_size, DTensor pads
        before all_gather and unpads after. The no-op pad/unpad on ranks with
        full-size shards must not be eliminated by remove_noop_ops, so all
        ranks produce identical FX graphs with matching op counts.
        """

        def func(a, *, ranks):
            # [FUNCTION DOC] HELPER: Pad a to the full chunk, all_gather, narrow to 7 rows,
            # add one.
            # Simulate DTensor's pad-before-all_gather for uneven shards.
            # rank 0: a is (4, 8), pad_size=0 → no-op pad
            # rank 1: a is (3, 8), pad_size=1 → real pad to (4, 8)
            full_chunk = (7 + len(ranks) - 1) // len(ranks)
            pad_size = full_chunk - a.size(0)
            a_padded = torch.nn.functional.pad(a, [0, 0, 0, pad_size])
            ag = _functional_collectives.all_gather_single(a_padded, 0, ranks)
            # Unpad after all_gather: narrow to original logical size
            result = ag.narrow(0, 0, 7)
            return result + 1

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            import torch.distributed as dist
            from torch._subclasses.fake_tensor import unset_fake_temporarily

            world_size = self.world_size
            # 7 is not divisible by 2: rank 0 gets 4 rows, rank 1 gets 3
            full_chunk = (7 + world_size - 1) // world_size
            local_size = full_chunk if self.rank == 0 else 7 - full_chunk
            a = torch.randn(local_size, 8, device=device_type)
            ranks = list(range(world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph_str = run_and_get_aten_graph(compiled, a)

            # Build structural fingerprint: sorted list of call_function targets.
            # Node names differ across ranks, but targets and op counts must match.
            targets_r = sorted(
                str(n_line.split("target=")[1].split("]")[0])
                for n_line in aten_graph_str.split("\n")
                if "call_function" in n_line and "target=" in n_line
            )

            with unset_fake_temporarily():
                all_targets: list[list[str] | None] = [None] * world_size
                dist.all_gather_object(all_targets, targets_r)

            self.assertEqual(
                all_targets[0],
                all_targets[1],
                "FX graph op targets differ across ranks — not SPMD. "
                "No-op pad/slice may have been eliminated by remove_noop_ops.",
            )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {
            **get_bucket_patches(),
            "aten_distributed_optimizations.enable_overlap_scheduling": True,
            "aten_distributed_optimizations.spmd_check": True,
            "aten_distributed_optimizations.spmd_mismatch": "error",
        }
    )
    def test_spmd_verify_crashes_on_mismatch(self):
        # [FUNCTION DOC] With spmd_check enabled and mismatch mode error, compiling a function that
        # pads on only one rank must raise a RuntimeError during the call.
        """Test that spmd_mismatch="error" raises on non-SPMD graphs."""

        def func(a, *, ranks):
            # [FUNCTION DOC] HELPER: Pad a only if the rank's shard is short, all_gather, add one.
            # rank 0 gets (4, 8), rank 1 gets (3, 8) — different node counts
            # if pad is eliminated on rank 0 but not rank 1
            full_chunk = (7 + len(ranks) - 1) // len(ranks)
            pad_size = full_chunk - a.size(0)
            # Only rank 1 will have a real pad op here
            if pad_size > 0:
                a = torch.nn.functional.pad(a, [0, 0, 0, pad_size])
            ag = _functional_collectives.all_gather_single(a, 0, ranks)
            return ag + 1

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            world_size = self.world_size
            full_chunk = (7 + world_size - 1) // world_size
            local_size = full_chunk if self.rank == 0 else 7 - full_chunk
            a = torch.randn(local_size, 8, device=device_type)
            ranks = list(range(world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)

            # The graph will differ: rank 0 has no pad, rank 1 has pad.
            # With spmd_check_crash_on_mismatch=True, this should raise.
            with self.assertRaises(RuntimeError, msg="SPMD graph verification"):
                compiled(a)


def get_toy_model(device_type: str):
    # [FUNCTION DOC] HELPER: Build a small test model on device_type: two blocks of three
    # 4x4 linear layers (query, key, projection) with relu on the sum of query and key outputs,
    # followed by a LayerNorm. Defines the block and model classes locally and returns the model.
    """
    Helper to construct a small multi-layer ToyModel
    """

    class ToyBlock(torch.nn.Module):
        # [CLASS DOC] HELPER: Toy block with three Linear(4, 4) layers named wq, wk and proj;
        # forward computes proj of relu of wq(x) plus wk(x).
        def __init__(self):
            # [FUNCTION DOC] HELPER: Create the three linear submodules.
            super().__init__()
            self.wq = torch.nn.Linear(4, 4)
            self.wk = torch.nn.Linear(4, 4)
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Apply wq and wk, add, relu, then proj.
            attn = self.wq(x) + self.wk(x)
            return self.proj(torch.nn.functional.relu(attn))

    class ToyModel(torch.nn.Module):
        # [CLASS DOC] HELPER: Toy model holding a ModuleList of two ToyBlock layers and a
        # LayerNorm(4); forward applies the blocks in order, then the norm.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Create the two blocks and the norm.
            super().__init__()
            self.layers = torch.nn.ModuleList([ToyBlock() for _ in range(2)])
            self.norm = torch.nn.LayerNorm(4)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Run all blocks sequentially and apply the norm.
            for blk in self.layers:
                x = blk(x)
            return self.norm(x)

    model = ToyModel().to(device_type)
    return model


def apply_manual_reordering_and_get_graph(
    graph, module_bucket_plans, out_li, custom_module_stack_fn=None
) -> None:
    # [FUNCTION DOC] HELPER: Post-grad pass for manual bucketing tests. Tags the first four
    # gather, scatter, all_reduce and wait nodes (by name) with module_1 and the next four with
    # module_2 in node metadata (side effect, overwriting nn_module_stack), runs
    # ManualOverlapScheduler with module_bucket_plans (no overlap deps, optional module stack
    # function, configured bucket mode), lints the result and appends its graph to out_li.
    gm = graph.owning_module
    from torch._inductor.config import aten_distributed_optimizations as dist_opts
    from torch._inductor.fx_passes.overlap_manual_scheduling import (
        ManualOverlapScheduler,
    )

    for node in list(gm.graph.nodes):
        # Handle all-gather, reduce-scatter, and all-reduce nodes for module_1
        if node.name in (
            "all_gather_into_tensor",
            "all_gather_into_tensor_1",
            "reduce_scatter_tensor",
            "reduce_scatter_tensor_1",
            "all_reduce",
            "all_reduce_1",
            "wait_tensor",
            "wait_tensor_1",
        ):
            node.meta["nn_module_stack"] = {"test": ["module_1", ""]}
        # Handle all-gather, reduce-scatter, and all-reduce nodes for module_2
        if node.name in (
            "all_gather_into_tensor_2",
            "all_gather_into_tensor_3",
            "reduce_scatter_tensor_2",
            "reduce_scatter_tensor_3",
            "all_reduce_2",
            "all_reduce_3",
            "wait_tensor_2",
            "wait_tensor_3",
        ):
            node.meta["nn_module_stack"] = {"test": ["module_2", ""]}

    overlapped_gm = ManualOverlapScheduler(
        gm,
        module_bucket_plans,
        insert_overlap_deps=False,
        module_stack_fn=custom_module_stack_fn,
        bucket_mode=dist_opts.bucket_mode,
    ).run()
    overlapped_gm.graph.lint()
    out_li.append(overlapped_gm.graph)


def run_and_get_manual_aten_graph(
    fn, module_bucket_plans, *inputs, custom_module_stack_fn=None
):
    # [FUNCTION DOC] HELPER: Call fn on inputs with the manual reordering pass installed as
    # post_grad_custom_post_pass and return (output, captured rewritten graph). The first
    # captured graph is used; raises IndexError if none was captured.
    li = []
    apply = functools.partial(
        apply_manual_reordering_and_get_graph,
        module_bucket_plans=module_bucket_plans,
        out_li=li,
        custom_module_stack_fn=custom_module_stack_fn,
    )
    with torch._inductor.config.patch(post_grad_custom_post_pass=apply):
        out = fn(*inputs)

    return out, li[0]


class TestManualOverlapBucketing(TestComputeCommReorderingMultiProc):
    # [CLASS DOC] Tests for manual (module-path-planned) bucketing and graph-view utilities.
    # Supplies test function builders and shared runners for gather, scatter and all_reduce
    # bucketing; inherits the 2-rank setup (and inherited tests) from the base reordering class.
    """
    Tests for manual overlap scheduling and subgraph utilities.
    """

    def _get_all_gather_test_func(self):
        # [FUNCTION DOC] HELPER: Return the function used by gather bucketing tests: four
        # independent all_gathers, two dependent matmuls and a weighted sum of all results.
        """Return the all-gather test function used by bucketing tests."""

        def func(a, b, c, d, *, ranks):
            # [FUNCTION DOC] HELPER: Four all_gathers, two matmuls where the second depends on the
            # first two gathers, and a weighted sum.
            # All 4 all-gathers are independent - COULD be bucketed together
            ag1 = _functional_collectives.all_gather_single(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_single(b, 0, ranks)
            ag3 = _functional_collectives.all_gather_single(c[:4], 0, ranks)
            ag4 = _functional_collectives.all_gather_single(d[:4], 0, ranks)

            # First compute - can hide ag1 and ag2
            e = a * 5  # Use a to avoid fusion
            mm1 = torch.matmul(e, e.T)

            # Force ag1/ag2 to complete before mm2 (but ag3/ag4 can still be deferred)
            # Use first 8x8 elements to match mm1's shape
            intermediate = ag1[:8, :8] + ag2[:8, :8]

            # Second compute - depends on ag1/ag2 through intermediate, can hide ag3/ag4
            mm2 = torch.matmul(mm1 + intermediate, c[:8])

            # Use all results
            result = (
                ag1.sum() * 1.1
                + ag2.sum() * 1.2
                + ag3.sum() * 1.3
                + ag4.sum() * 1.4
                + mm1.sum()
                + mm2.sum()
            )
            return result

        return func

    def _get_reduce_scatter_test_func(self):
        # [FUNCTION DOC] HELPER: Return a function doing four independent reduce_scatters whose
        # results are returned directly (required for the manual scheduler to bucket them).
        """Return the reduce-scatter test function used by bucketing tests.

        For reduce-scatter to be bucketed by ManualOverlapScheduler, the wait nodes
        must be directly returned as output (FSDP gradient pattern).
        """

        def func(a, b, c, d):
            # [FUNCTION DOC] HELPER: Four reduce_scatter_singles of a, b, c, d returned as a tuple.
            # All 4 reduce-scatters are independent - COULD be bucketed together
            rs1 = _functional_collectives.reduce_scatter_single(a, "sum", 0, "0")
            rs2 = _functional_collectives.reduce_scatter_single(b, "sum", 0, "0")
            rs3 = _functional_collectives.reduce_scatter_single(c, "sum", 0, "0")
            rs4 = _functional_collectives.reduce_scatter_single(d, "sum", 0, "0")

            # Return reduce-scatter results directly as outputs (FSDP gradient pattern)
            return rs1, rs2, rs3, rs4

        return func

    def _get_all_reduce_test_func(self):
        # [FUNCTION DOC] HELPER: Return a function doing four independent all_reduces whose results
        # are returned directly so launches can be paired with waits.
        """Return the all-reduce test function used by bucketing tests.

        All-reduce is the HSDP replicate-axis / DDP gradient sync. As with the
        reduce-scatter case the waits are returned directly as outputs so the
        manual bucketer can pair each launch with its wait.
        """

        def func(a, b, c, d):
            # [FUNCTION DOC] HELPER: Four all_reduces of a, b, c, d returned as a tuple.
            # All 4 all-reduces are independent - COULD be bucketed together
            ar1 = _functional_collectives.all_reduce(a, "sum", "0")
            ar2 = _functional_collectives.all_reduce(b, "sum", "0")
            ar3 = _functional_collectives.all_reduce(c, "sum", "0")
            ar4 = _functional_collectives.all_reduce(d, "sum", "0")

            # Return all-reduce results directly as outputs (DDP gradient pattern)
            return ar1, ar2, ar3, ar4

        return func

    def _run_all_reduce_bucketing_test(
        self, module_bucket_plans, expected_num_all_reduce
    ):
        # [FUNCTION DOC] HELPER: Compile the all_reduce function under distributed init, apply
        # manual bucketing with module_bucket_plans, and assert the number of all_reduce nodes
        # (both the functional and in-place forms) equals expected_num_all_reduce and that output
        # matches eager.
        """Bucket independent all-reduces per ``module_bucket_plans`` and assert
        the surviving all-reduce count plus numerical equivalence."""
        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            d = torch.ones(8, 8, dtype=torch.float, device=device_type) * 4

            func = self._get_all_reduce_test_func()
            compiled = torch.compile(func)
            out, aten_graph = run_and_get_manual_aten_graph(
                compiled, module_bucket_plans, a, b, c, d
            )

            # Bucketed all-reduces are emitted as the in-place
            # ``all_reduce_`` op (operating on the cat buffer); unbucketed
            # ones stay as the functional ``all_reduce``. Count both forms.
            all_reduce_targets = (
                torch.ops._c10d_functional.all_reduce.default,
                torch.ops._c10d_functional.all_reduce_.default,
            )
            num_all_reduce = sum(
                1
                for node in aten_graph.nodes
                if node.op == "call_function" and node.target in all_reduce_targets
            )
            self.assertEqual(num_all_reduce, expected_num_all_reduce)
            self.assertTrue(same(out, func(a, b, c, d)))

    def _run_manual_bucketing_test(
        self,
        collective_type,
        module_bucket_plans,
        expected_checks,
    ):
        # [FUNCTION DOC] HELPER: Shared runner. For collective_type all_gather or (anything else,
        # treated as) reduce_scatter, compiles the matching test function, applies manual bucketing
        # with module_bucket_plans, checks the substrings in expected_checks appear in order in the
        # graph string, and that output matches eager.
        """Common test logic for manual bucketing tests.

        Args:
            collective_type: Either "all_gather" or "reduce_scatter"
            module_bucket_plans: The bucket plans to use
            expected_checks: List of strings to check in the aten graph (in order)
        """
        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            d = torch.ones(8, 8, dtype=torch.float, device=device_type) * 4

            if collective_type == "all_gather":
                func = self._get_all_gather_test_func()
                ranks = list(range(self.world_size))
                func_c = functools.partial(func, ranks=ranks)
                compiled = torch.compile(func_c)
                out, aten_graph = run_and_get_manual_aten_graph(
                    compiled, module_bucket_plans, a, b, c, d
                )
                correct = func(a, b, c, d, ranks=ranks)
            else:  # reduce_scatter
                func = self._get_reduce_scatter_test_func()
                compiled = torch.compile(func)
                out, aten_graph = run_and_get_manual_aten_graph(
                    compiled, module_bucket_plans, a, b, c, d
                )
                correct = func(a, b, c, d)

            # Run expected checks in order
            fc = FileCheck()
            for check in expected_checks:
                fc.check(check)
            fc.run(str(aten_graph))

            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_make_graph_view_and_get_subgraph_by_path(self):
        # [FUNCTION DOC] Traces the toy model with symbolic trace, builds a graph view and checks
        # get_subgraph_by_path returns the node list for one path, several paths in order, an
        # empty list for unknown paths, and only known nodes for a mixed list.
        from torch._inductor.fx_passes.graph_view import (
            get_subgraph_by_path,
            make_graph_view,
        )

        model = get_toy_model(device_type)
        gm = torch.fx.symbolic_trace(model)
        graph_view = make_graph_view(gm.graph)
        # Fetch subgraph for first transformer layer
        sub_nodes = get_subgraph_by_path(graph_view, "layers.0.wq")
        self.assertEqual([n.name for n in sub_nodes], ["layers_0_wq"])

        # Fetch multiple paths at once
        multi_nodes = get_subgraph_by_path(graph_view, ["layers.0.wq", "layers.0.proj"])
        self.assertEqual(
            [n.name for n in multi_nodes], ["layers_0_wq", "layers_0_proj"]
        )

        # Fetch non existing paths
        non_exist_nodes = get_subgraph_by_path(graph_view, "nonexistent.module.path")
        self.assertEqual(non_exist_nodes, [])

        # Fetch mixed of existing and non existing paths
        mixed_nodes = get_subgraph_by_path(
            graph_view, ["layers.0.wq", "nonexistent.module.path"]
        )
        self.assertEqual([n.name for n in mixed_nodes], ["layers_0_wq"])

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_make_graph_view_and_get_subgraph_by_path_custom_module_stack_fn(self):
        # [FUNCTION DOC] Annotates each module forward with its module path in node custom
        # metadata, captures an export graph, removes nn_module_stack metadata, then builds a graph
        # view using a custom module stack function and checks get_subgraph_by_path returns the
        # expected node names for single, multiple, missing and mixed paths.
        from torch._dynamo.functional_export import dynamo_graph_capture_for_export
        from torch._inductor.fx_passes.graph_view import (
            get_subgraph_by_path,
            make_graph_view,
        )

        model = get_toy_model(device_type)

        module_path_key = "module_path"
        # Add annotation to node.meta["custom"]
        for name, m in model.named_modules():
            m.forward = torch.fx.traceback.annotate_fn({module_path_key: name})(
                m.forward
            )

        def module_stack_fn(node):
            # [FUNCTION DOC] HELPER: Return a one element module stack list from the node's
            # annotated module path (empty string if absent).
            module_stack = node.meta.get("custom", {}).get(module_path_key, "")
            return [(module_stack, torch.nn.Module)]

        gm = dynamo_graph_capture_for_export(model)(torch.randn(2, 4).to(device_type))

        # delete "nn_module_stack" to make sure the graph view is only constructed from annotation
        for n in gm.graph.nodes:
            if "nn_module_stack" in n.meta:
                del n.meta["nn_module_stack"]

        graph_view = make_graph_view(gm.graph, module_stack_fn=module_stack_fn)
        # Fetch subgraph for first transformer layer
        sub_nodes = get_subgraph_by_path(graph_view, "layers.0.wq")
        self.assertEqual(
            [n.name for n in sub_nodes],
            [
                "l_func_self_modules_layers_modules_0_modules_wq_parameters_weight_",
                "l_func_self_modules_layers_modules_0_modules_wq_parameters_bias_",
                "l_args_0_",
                "linear",
            ],
        )

        # Fetch multiple paths at once
        multi_nodes = get_subgraph_by_path(graph_view, ["layers.0.wq", "layers.0.proj"])
        self.assertEqual(
            [n.name for n in multi_nodes],
            [
                "l_func_self_modules_layers_modules_0_modules_wq_parameters_weight_",
                "l_func_self_modules_layers_modules_0_modules_wq_parameters_bias_",
                "l_args_0_",
                "linear",
                "l_func_self_modules_layers_modules_0_modules_proj_parameters_weight_",
                "l_func_self_modules_layers_modules_0_modules_proj_parameters_bias_",
                "x",
            ],
        )

        # Fetch non existing paths
        non_exist_nodes = get_subgraph_by_path(graph_view, "nonexistent.module.path")
        self.assertEqual(non_exist_nodes, [])

        # Fetch mixed of existing and non existing paths
        mixed_nodes = get_subgraph_by_path(
            graph_view, ["layers.0.wq", "nonexistent.module.path"]
        )
        self.assertEqual(
            [n.name for n in mixed_nodes],
            [
                "l_func_self_modules_layers_modules_0_modules_wq_parameters_weight_",
                "l_func_self_modules_layers_modules_0_modules_wq_parameters_bias_",
                "l_args_0_",
                "linear",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.bucket_mode": "custom_ops_multidtype"}
    )
    def test_manual_reordering_bucketing_pass_all_gather_separate_buckets(self):
        # [FUNCTION DOC] Gather bucketing with plans for module_1 and module_2 as separate buckets
        # (multidtype mode): expects two pre-bucket gather plus out-gather pairs, then two waits.
        self._run_manual_bucketing_test(
            collective_type="all_gather",
            module_bucket_plans=["module_1", "module_2"],
            expected_checks=[
                "_pre_bucket_all_gather",
                "all_gather_into_tensor_out",
                "_pre_bucket_all_gather_1",
                "all_gather_into_tensor_out_1",
                "wait_tensor_4",
                "wait_tensor_5",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_manual_bucketing_reordering_pass_all_gather_no_bucket(self):
        # [FUNCTION DOC] Gather with an empty bucket plan: the four gathers and their
        # waits stay unbucketed.
        self._run_manual_bucketing_test(
            collective_type="all_gather",
            module_bucket_plans=[],
            expected_checks=[
                "all_gather_into_tensor",
                "all_gather_into_tensor_1",
                "all_gather_into_tensor_2",
                "all_gather_into_tensor_3",
                "wait_tensor",
                "wait_tensor_1",
                "wait_tensor_2",
                "wait_tensor_3",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.bucket_mode": "custom_ops_multidtype"}
    )
    def test_manual_bucketing_reordering_pass_all_gather_single_bucket(self):
        # [FUNCTION DOC] Gather bucketing with both modules in one bucket (multidtype mode):
        # expects one pre-bucket gather, one out-gather and one wait.
        self._run_manual_bucketing_test(
            collective_type="all_gather",
            module_bucket_plans=[["module_1", "module_2"]],
            expected_checks=[
                "_pre_bucket_all_gather",
                "all_gather_into_tensor_out",
                "wait_tensor_4",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.bucket_mode": "custom_ops_multidtype"}
    )
    def test_bucketing_reordering_pass_all_gather_single_bucket_custom_module_stack_fn(
        self,
    ):
        # [FUNCTION DOC] Single-bucket gather plan where modules are identified by custom
        # annotation metadata via a custom module stack function. Checks the bucketed graph shape,
        # output match with eager, and that every new bucketed gather-prep and wait node carries
        # the module and custom metadata, a stack trace and bucketing source metadata.
        module_path_key = "module_path"

        def module_stack_fn(node):
            # [FUNCTION DOC] HELPER: Return a one element module stack list from the node's
            # annotated module path (empty string if absent).
            module_stack = node.meta.get("custom", {}).get(module_path_key, "")
            return [(module_stack, torch.nn.Module)]

        def func(a, b, c, d, *, ranks):
            # [FUNCTION DOC] HELPER: Four all_gathers annotated as my_module_1 (first two) and
            # my_module_2 (last two), two dependent matmuls and a weighted sum.
            # All 4 all-gathers are independent - COULD be bucketed together
            with torch.fx.traceback.annotate({module_path_key: "my_module_1"}):
                ag1 = _functional_collectives.all_gather_single(a, 0, ranks)
                ag2 = _functional_collectives.all_gather_single(b, 0, ranks)
            with torch.fx.traceback.annotate({module_path_key: "my_module_2"}):
                ag3 = _functional_collectives.all_gather_single(c[:4], 0, ranks)
                ag4 = _functional_collectives.all_gather_single(d[:4], 0, ranks)

            # First compute - can hide ag1 and ag2
            e = a * 5  # Use a to avoid fusion
            mm1 = torch.matmul(e, e.T)

            # Force ag1/ag2 to complete before mm2 (but ag3/ag4 can still be deferred)
            # Use first 8x8 elements to match mm1's shape
            intermediate = ag1[:8, :8] + ag2[:8, :8]

            # Second compute - depends on ag1/ag2 through intermediate, can hide ag3/ag4
            mm2 = torch.matmul(mm1 + intermediate, c[:8])

            # Use all results
            result = (
                ag1.sum() * 1.1
                + ag2.sum() * 1.2
                + ag3.sum() * 1.3
                + ag4.sum() * 1.4
                + mm1.sum()
                + mm2.sum()
            )
            return result

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            d = torch.ones(8, 8, dtype=torch.float, device=device_type) * 4
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph = run_and_get_manual_aten_graph(
                compiled,
                [["my_module_1", "my_module_2"]],
                a,
                b,
                c,
                d,
                custom_module_stack_fn=module_stack_fn,
            )

            (
                FileCheck()
                .check("_pre_bucket_all_gather")
                .check("all_gather_into_tensor_out")
                .check("wait_tensor_4")
                .run(str(aten_graph))
            )

            correct = func(a, b, c, d, ranks=ranks)
            self.assertTrue(same(out, correct))

            # Add metadata to the collective nodes to test preservation
            test_metadata = {
                "nn_module_stack": {
                    "test": ("module_1", ""),
                },
                "custom": {
                    "module_path": "my_module_1",
                },
            }

            # Verify metadata preservation: new bucketed nodes should have the metadata
            new_ag_nodes = aten_graph.find_nodes(
                op="call_function",
                target=torch.ops.bucketing._pre_bucket_all_gather.default,
            )
            new_wait_nodes = aten_graph.find_nodes(
                op="call_function",
                target=torch.ops._c10d_functional.wait_tensor.default,
            )

            all_new_nodes = list(new_ag_nodes) + list(new_wait_nodes)
            self.assertGreater(len(all_new_nodes), 0, "Should have created new nodes")

            for node in all_new_nodes:
                self.assertEqual(
                    node.meta.get("nn_module_stack"), test_metadata["nn_module_stack"]
                )
                self.assertEqual(node.meta.get("custom"), test_metadata["custom"])
                self.assertTrue(node.meta.get("stack_trace", None) is not None)
                self.assertTrue(
                    node.meta.get("bucketing_stack_trace_sources", None) is not None
                )
                self.assertTrue(
                    node.meta.get("bucketing_custom_sources", None) is not None
                )
                self.assertTrue(
                    node.meta.get("bucketing_nn_module_stack_sources", None) is not None
                )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.bucket_mode": "custom_ops_multidtype"}
    )
    def test_manual_bucketing_reordering_pass_reduce_scatter_separate_buckets(self):
        # [FUNCTION DOC] Scatter bucketing with separate per-module buckets (multidtype mode):
        # expects two pre-bucket scatter plus scatter pairs and two waits.
        self._run_manual_bucketing_test(
            collective_type="reduce_scatter",
            module_bucket_plans=["module_1", "module_2"],
            expected_checks=[
                "_pre_bucket_reduce_scatter",
                "reduce_scatter_tensor_4",
                "_pre_bucket_reduce_scatter_1",
                "reduce_scatter_tensor_5",
                "wait_tensor_4",
                "wait_tensor_5",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.bucket_mode": "custom_ops_multidtype"}
    )
    def test_manual_bucketing_reordering_pass_reduce_scatter_single_bucket(self):
        # [FUNCTION DOC] Scatter bucketing with both modules in one bucket (multidtype mode):
        # expects one pre-bucket scatter, one scatter and one wait.
        self._run_manual_bucketing_test(
            collective_type="reduce_scatter",
            module_bucket_plans=[["module_1", "module_2"]],
            expected_checks=[
                "_pre_bucket_reduce_scatter",
                "reduce_scatter_tensor_4",
                "wait_tensor_4",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_manual_bucketing_reordering_pass_reduce_scatter_no_bucket(self):
        # [FUNCTION DOC] Scatter with an empty plan: four scatters and waits stay unbucketed.
        self._run_manual_bucketing_test(
            collective_type="reduce_scatter",
            module_bucket_plans=[],
            expected_checks=[
                "reduce_scatter_tensor",
                "reduce_scatter_tensor_1",
                "reduce_scatter_tensor_2",
                "reduce_scatter_tensor_3",
                "wait_tensor",
                "wait_tensor_1",
                "wait_tensor_2",
                "wait_tensor_3",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_manual_bucketing_reordering_pass_all_reduce_separate_buckets(self):
        # [FUNCTION DOC] All_reduce bucketing with one bucket per module: four all_reduces
        # become two.
        # One bucket per module: 4 all-reduces collapse to 2 (one per module).
        self._run_all_reduce_bucketing_test(
            module_bucket_plans=["module_1", "module_2"],
            expected_num_all_reduce=2,
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_manual_bucketing_reordering_pass_all_reduce_single_bucket(self):
        # [FUNCTION DOC] All_reduce bucketing with both modules in one bucket: the four
        # all_reduces become one.
        # Both modules in one bucket: 4 all-reduces collapse to 1.
        self._run_all_reduce_bucketing_test(
            module_bucket_plans=[["module_1", "module_2"]],
            expected_num_all_reduce=1,
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_manual_bucketing_reordering_pass_all_reduce_no_bucket(self):
        # [FUNCTION DOC] All_reduce with an empty plan: all four remain.
        # Empty plan: all 4 all-reduces remain unbucketed.
        self._run_all_reduce_bucketing_test(
            module_bucket_plans=[],
            expected_num_all_reduce=4,
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.bucket_mode": "custom_ops"}
    )
    def test_manual_bucketing_ag_convert_inputs_before_consumers(self):
        # [FUNCTION DOC] Manual bucketing (custom_ops mode) of two all_gathers in one annotated
        # module, where the second input is computed late and converted to bfloat16: checks a pre-
        # bucket gather then wait appear and that the output matches eager.
        module_path_key = "module_path"

        def module_stack_fn(node):
            # [FUNCTION DOC] HELPER: Return a one element module stack list from the node's
            # annotated module path (empty string if absent).
            module_stack = node.meta.get("custom", {}).get(module_path_key, "")
            return [(module_stack, torch.nn.Module)]

        def func(a, b, *, ranks):
            # [FUNCTION DOC] HELPER: Gather a bfloat16 copy of a, a late computed bfloat16 copy of
            # b+1, and return the sum of both float results.
            with torch.fx.traceback.annotate({module_path_key: "mod"}):
                ag1 = _functional_collectives.all_gather_tensor(
                    a.to(torch.bfloat16), 0, ranks
                )
            use_ag1 = ag1.float().sum()
            late = b + 1.0
            with torch.fx.traceback.annotate({module_path_key: "mod"}):
                ag2 = _functional_collectives.all_gather_tensor(
                    late.to(torch.bfloat16), 0, ranks
                )
            return use_ag1 + ag2.float().sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            ranks = list(range(self.world_size))

            compiled = torch.compile(functools.partial(func, ranks=ranks))
            out, aten_graph = run_and_get_manual_aten_graph(
                compiled,
                [["mod"]],
                a,
                b,
                custom_module_stack_fn=module_stack_fn,
            )

            FileCheck().check("_pre_bucket_all_gather").check("wait_tensor").run(
                str(aten_graph)
            )
            self.assertTrue(same(out, func(a, b, ranks=ranks)))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.bucket_mode": "custom_ops"}
    )
    def test_manual_bucketing_ag_late_input_repairs_wait_consumer(self):
        # [FUNCTION DOC] Builds an fx graph by hand with fake tensor metadata (two gathers, the
        # second fed by a late add and bfloat16 convert, and consumers of the first wait), buckets
        # both gathers with ManualOverlapPreservingBucketer, and checks the add, pre-bucket gather,
        # wait, then consumers order is repaired. Uses a fake process group when GPUs are scarce.
        from torch._inductor.fx_passes.overlap_manual_scheduling import (
            ManualOverlapPreservingBucketer,
        )
        from torch._inductor.fx_passes.overlap_scheduling import CollectiveInfo
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.utils._ordered_set import OrderedSet

        def set_val(node, val):
            # [FUNCTION DOC] HELPER: Store val as the node's val metadata and return the node.
            node.meta["val"] = val
            return node

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            with FakeTensorMode():
                a_val = torch.empty(8, 8, device=device_type)
                b_val = torch.empty(8, 8, device=device_type)
                ag1_val = torch.empty(self.world_size * 8, 8, device=device_type)
                late_val = torch.empty(8, 8, device=device_type)
                convert_val = torch.empty(
                    8, 8, device=device_type, dtype=torch.bfloat16
                )
                ag2_val = torch.empty(
                    self.world_size * 8,
                    8,
                    device=device_type,
                    dtype=torch.bfloat16,
                )
                use_ag1_val = torch.empty_like(ag1_val)
                use_ag1_2_val = torch.empty_like(ag1_val)

            graph = fx.Graph()
            a = set_val(graph.placeholder("a"), a_val)
            b = set_val(graph.placeholder("b"), b_val)
            ag1 = set_val(
                graph.call_function(
                    torch.ops._c10d_functional.all_gather_into_tensor.default,
                    args=(a, self.world_size, "0"),
                ),
                ag1_val,
            )
            wait1 = set_val(
                graph.call_function(
                    torch.ops._c10d_functional.wait_tensor.default,
                    args=(ag1,),
                ),
                ag1_val,
            )
            use_ag1 = set_val(
                graph.call_function(torch.ops.aten.neg.default, args=(wait1,)),
                use_ag1_val,
            )
            use_ag1_2 = set_val(
                graph.call_function(torch.ops.aten.relu.default, args=(use_ag1,)),
                use_ag1_2_val,
            )
            late = set_val(
                graph.call_function(torch.ops.aten.add.Tensor, args=(b, 1.0)),
                late_val,
            )
            convert = set_val(
                graph.call_function(
                    torch.ops.prims.convert_element_type.default,
                    args=(late, torch.bfloat16),
                ),
                convert_val,
            )
            ag2 = set_val(
                graph.call_function(
                    torch.ops._c10d_functional.all_gather_into_tensor.default,
                    args=(convert, self.world_size, "0"),
                ),
                ag2_val,
            )
            wait2 = set_val(
                graph.call_function(
                    torch.ops._c10d_functional.wait_tensor.default,
                    args=(ag2,),
                ),
                ag2_val,
            )
            graph.output((use_ag1_2, wait2))
            gm = fx.GraphModule(torch.nn.Module(), graph)
            gm.graph.lint()

            bucketer = ManualOverlapPreservingBucketer(
                gm.graph,
                {
                    ag1: CollectiveInfo(ag1, wait1, 0, 0, 0),
                    ag2: CollectiveInfo(ag2, wait2, 0, 0, 0),
                },
                OrderedSet(gm.graph.nodes),
                bucket_mode="custom_ops",
            )
            bucketer._bucket_group([ag1, ag2])
            gm.graph.lint()

            (
                FileCheck()
                .check("add")
                .check("_pre_bucket_all_gather")
                .check("wait_tensor")
                .check("neg")
                .check("relu")
                .run(str(gm.graph))
            )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.bucket_mode": "custom_ops"}
    )
    def test_manual_bucket_insertion_uses_graph_order(self):
        # [FUNCTION DOC] Bucket the two gathers in reversed order via a custom post-grad pass and
        # check the bucket is inserted according to graph order (pre-bucket gather then wait);
        # output matches eager.
        def func(a, b, *, ranks):
            # [FUNCTION DOC] HELPER: Two all_gather_tensors, sum of slices of each.
            ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)
            ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)
            return ag1[:8, :8].sum() + ag2[:8, :8].sum()

        def reverse_bucket_order(graph, out_li):
            # [FUNCTION DOC] HELPER: Post-grad pass: build a ManualOverlapScheduler with no plans,
            # find the two gather nodes (AssertionError unless exactly two), bucket them in
            # reversed order through its bucketer, lint, and append the graph to out_li.
            gm = graph.owning_module
            from torch._inductor.fx_passes.overlap_manual_scheduling import (
                ManualOverlapScheduler,
            )

            scheduler = ManualOverlapScheduler(
                gm,
                module_bucket_plans=[],
                insert_overlap_deps=False,
            )
            ag_nodes = [
                n
                for n in scheduler.nodes
                if n.name in ("all_gather_into_tensor", "all_gather_into_tensor_1")
            ]
            self.assertEqual(len(ag_nodes), 2)
            scheduler.bucketer._bucket_group(list(reversed(ag_nodes)))
            scheduler.graph.lint()
            out_li.append(scheduler.graph)

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            ranks = list(range(self.world_size))

            li = []
            apply = functools.partial(reverse_bucket_order, out_li=li)
            with torch._inductor.config.patch(post_grad_custom_post_pass=apply):
                compiled = torch.compile(functools.partial(func, ranks=ranks))
                out = compiled(a, b)

            FileCheck().check("_pre_bucket_all_gather").check("wait_tensor").run(
                str(li[0])
            )
            self.assertTrue(same(out, func(a, b, ranks=ranks)))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.bucket_mode": "custom_ops"}
    )
    def test_manual_reduce_scatter_bucket_wait_stays_after_inputs(self):
        # [FUNCTION DOC] Buckets two reduce_scatters where the second input is computed by an add:
        # checks the add comes before the pre-bucket scatter, then the wait; output matches eager.
        def func(a, b):
            # [FUNCTION DOC] HELPER: Reduce_scatter a, reduce_scatter b+1, return both.
            rs1 = _functional_collectives.reduce_scatter_tensor(a, "sum", 0, "0")
            b = b + 1
            rs2 = _functional_collectives.reduce_scatter_tensor(b, "sum", 0, "0")
            return rs1, rs2

        def bucket_reduce_scatters(graph, out_li):
            # [FUNCTION DOC] HELPER: Post-grad pass: build a ManualOverlapScheduler with no plans,
            # bucket the two reduce_scatter nodes (AssertionError unless exactly two), lint, and
            # append the graph to out_li.
            gm = graph.owning_module
            from torch._inductor.fx_passes.overlap_manual_scheduling import (
                ManualOverlapScheduler,
            )

            scheduler = ManualOverlapScheduler(
                gm,
                module_bucket_plans=[],
                insert_overlap_deps=False,
            )
            rs_nodes = list(
                scheduler.graph.find_nodes(
                    op="call_function",
                    target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
                )
            )
            self.assertEqual(len(rs_nodes), 2)
            scheduler.bucketer._bucket_group(rs_nodes)
            scheduler.graph.lint()
            out_li.append(scheduler.graph)

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2

            li = []
            apply = functools.partial(bucket_reduce_scatters, out_li=li)
            with torch._inductor.config.patch(post_grad_custom_post_pass=apply):
                compiled = torch.compile(func)
                out = compiled(a, b)

            FileCheck().check("add").check("_pre_bucket_reduce_scatter").check(
                "wait_tensor"
            ).run(str(li[0]))
            self.assertTrue(same(out, func(a, b)))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.bucket_mode": "default"}
    )
    def test_manual_bucketing_ag_with_intermediate_deps(self):
        # [FUNCTION DOC] Default bucket mode, one plan combining annotated modules mod1 and mod2
        # with intermediate computations between gathers: checks gathers and waits appear and
        # output matches eager.
        module_path_key = "module_path"

        def module_stack_fn(node):
            # [FUNCTION DOC] HELPER: Return a one element module stack list from the node's
            # annotated module path (empty string if absent).
            module_stack = node.meta.get("custom", {}).get(module_path_key, "")
            return [(module_stack, torch.nn.Module)]

        def func(a, b, c, *, ranks):
            # [FUNCTION DOC] HELPER: One gather in mod1, two gathers of prepared inputs in
            # mod2, summed.
            with torch.fx.traceback.annotate({module_path_key: "mod1"}):
                ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)
            b_prepared = b[:4] + 1.0
            c_prepared = c[:4] * 2.0
            with torch.fx.traceback.annotate({module_path_key: "mod2"}):
                ag2 = _functional_collectives.all_gather_tensor(b_prepared, 0, ranks)
                ag3 = _functional_collectives.all_gather_tensor(c_prepared, 0, ranks)
            return ag1.sum() + ag2.sum() + ag3.sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph = run_and_get_manual_aten_graph(
                compiled,
                [["mod1", "mod2"]],
                a,
                b,
                c,
                custom_module_stack_fn=module_stack_fn,
            )

            graph_str = str(aten_graph)
            (
                FileCheck()
                .check("all_gather_into_tensor")
                .check("wait_tensor")
                .run(graph_str)
            )

            correct = func(a, b, c, ranks=ranks)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.bucket_mode": "default"}
    )
    def test_manual_bucketing_rs_with_intermediate_deps(self):
        # [FUNCTION DOC] Default bucket mode, one plan combining mod1 and mod2 reduce_scatters with
        # an intermediate computation on one input: checks scatters and waits appear and output
        # matches eager.
        module_path_key = "module_path"

        def module_stack_fn(node):
            # [FUNCTION DOC] HELPER: Return a one element module stack list from the node's
            # annotated module path (empty string if absent).
            module_stack = node.meta.get("custom", {}).get(module_path_key, "")
            return [(module_stack, torch.nn.Module)]

        def func(a, b, c):
            # [FUNCTION DOC] HELPER: One reduce_scatter in mod1 and two in mod2, one on a computed
            # input; returns all three.
            with torch.fx.traceback.annotate({module_path_key: "mod1"}):
                rs1 = _functional_collectives.reduce_scatter_tensor(a, "sum", 0, "0")
            c_grad = c * 2.0 + 1.0
            with torch.fx.traceback.annotate({module_path_key: "mod2"}):
                rs2 = _functional_collectives.reduce_scatter_tensor(b, "sum", 0, "0")
                rs3 = _functional_collectives.reduce_scatter_tensor(
                    c_grad, "sum", 0, "0"
                )
            return rs1, rs2, rs3

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3

            compiled = torch.compile(func)
            out, aten_graph = run_and_get_manual_aten_graph(
                compiled,
                [["mod1", "mod2"]],
                a,
                b,
                c,
                custom_module_stack_fn=module_stack_fn,
            )

            graph_str = str(aten_graph)
            (
                FileCheck()
                .check("reduce_scatter_tensor")
                .check("wait_tensor")
                .run(graph_str)
            )

            correct = func(a, b, c)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @torch._inductor.config.patch(
        {"aten_distributed_optimizations.bucket_mode": "custom_ops"}
    )
    def test_manual_bucketing_ag_forward_and_backward(self):
        # [FUNCTION DOC] Simulates reshard-after-forward with forward and backward gathers on the
        # same modules; applies manual bucketing (custom_ops) and checks the graph lints (stays
        # valid) and output matches eager.
        """Simulate reshard-after-forward: forward all-gathers and backward
        all-gathers for the same modules. Verifies graph stays topologically valid after
        bucketing and overlap reordering with _move_overlap_nodes."""
        module_path_key = "module_path"

        def module_stack_fn(node):
            # [FUNCTION DOC] HELPER: Return a one element module stack list from the node's
            # annotated module path (empty string if absent).
            module_stack = node.meta.get("custom", {}).get(module_path_key, "")
            return [(module_stack, torch.nn.Module)]

        def func(a, b, c, d, *, ranks):
            # [FUNCTION DOC] HELPER: Forward gathers in mod1 and mod2, a matmul, backward gathers
            # in mod2 and mod1, another matmul, and the sum of both.
            # Forward all-gathers (mod1 then mod2)
            with torch.fx.traceback.annotate({module_path_key: "mod1"}):
                fwd_ag1 = _functional_collectives.all_gather_tensor(a, 0, ranks)
            with torch.fx.traceback.annotate({module_path_key: "mod2"}):
                fwd_ag2 = _functional_collectives.all_gather_tensor(b, 0, ranks)

            # Forward compute
            mm1 = torch.matmul(fwd_ag1[:8, :8], fwd_ag2[:8, :8])

            # Backward all-gathers (mod2 then mod1, reverse order)
            with torch.fx.traceback.annotate({module_path_key: "mod2"}):
                bwd_ag2 = _functional_collectives.all_gather_tensor(c, 0, ranks)
            with torch.fx.traceback.annotate({module_path_key: "mod1"}):
                bwd_ag1 = _functional_collectives.all_gather_tensor(d, 0, ranks)

            # Backward compute
            mm2 = torch.matmul(bwd_ag2[:8, :8], bwd_ag1[:8, :8])

            return mm1.sum() + mm2.sum()

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            a = torch.ones(8, 8, dtype=torch.float, device=device_type)
            b = torch.ones(8, 8, dtype=torch.float, device=device_type) * 2
            c = torch.ones(8, 8, dtype=torch.float, device=device_type) * 3
            d = torch.ones(8, 8, dtype=torch.float, device=device_type) * 4
            ranks = list(range(self.world_size))

            func_c = functools.partial(func, ranks=ranks)
            compiled = torch.compile(func_c)
            out, aten_graph = run_and_get_manual_aten_graph(
                compiled,
                [["mod1", "mod2"]],
                a,
                b,
                c,
                d,
                custom_module_stack_fn=module_stack_fn,
            )

            aten_graph.lint()

            correct = func(a, b, c, d, ranks=ranks)
            self.assertTrue(same(out, correct))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
