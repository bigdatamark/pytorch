# Owner(s): ["module: inductor"]
# [MODULE DOC] Two-rank tests for Inductor's compute/communication overlap scheduling passes
# (sink_waits, raise_comms, reorder_compute_for_overlap, a custom runtime estimator, grouped
# scheduler nodes, default comms ordering) and sanity checks of NCCL cost-model tables. Each
# test compiles a function using functional all_reduce, inspects the generated Triton code
# ordering with FileCheck, and compares results with eager. Defines a custom cost model, a
# pre-fusion grouping pass, and one multi-process test class.
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case

# for some reason importing functional collectives after dynamo breaks collectives handling!
import torch.distributed._functional_collectives as _functional_collectives
from torch._C import FileCheck
from torch._dynamo.utils import same
from torch._inductor import ir, scheduler
from torch._inductor.comm_analysis import (
    baseLat,
    hwLat,
    llMaxBws,
    NCCL_ALGO,
    NCCL_HW,
    NCCL_PROTO,
    NVIDIA_GPU_TYPE,
)
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    at_least_x_gpu,
    DynamoDistributedMultiProcTestCase,
    requires_accelerator_dist_backend,
)
from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_GPU


device_type = get_devtype().type


def get_snode_runtime_for_reorder_compute_test(snode):
    # [FUNCTION DOC] HELPER: Custom Inductor runtime estimator for a scheduler node:
    # collective kernels cost 100, wait kernels 0, other extern kernels (e.g. matmuls) 5, and
    # everything else 1. Returns the number; used to show the reordering responds to the model.
    # NOTE: custom cost model to show that the compute reordering algorithm is working
    # Collective kernels
    if isinstance(snode.node, ir._CollectiveKernel):
        return 100
    elif isinstance(snode.node, ir._WaitKernel):
        return 0
    # High-arithmetic-intensity compute kernels
    elif isinstance(snode.node, ir.ExternKernel):
        return 5
    # All other kernels
    return 1


def create_grouped_node_for_allreduce_and_its_deps(snodes):
    # [FUNCTION DOC] HELPER: Pre-fusion custom pass taking the list of scheduler nodes.
    # Finds the single in-place all_reduce collective node and its single producer node
    # (looked up by buffer name), merges both into one grouped scheduler node, and returns a
    # new order with that group first followed by all remaining nodes in original order.
    # Raises AssertionError unless there is exactly one all_reduce and exactly one input node.
    name_to_snode = {snode.node.name: snode for snode in snodes}
    all_reduce_snodes = [
        snode
        for snode in snodes
        if isinstance(snode.node, ir._CollectiveKernel)
        and snode.node.op_overload == torch.ops._c10d_functional.all_reduce_.default
    ]
    if len(all_reduce_snodes) != 1:
        raise AssertionError(
            f"Expected exactly 1 all_reduce_snode, got {len(all_reduce_snodes)}"
        )
    all_reduce_snode = all_reduce_snodes[0]
    all_reduce_dep_snodes = [
        name_to_snode[node.name] for node in all_reduce_snode.node.inputs
    ]
    if len(all_reduce_dep_snodes) != 1:
        raise AssertionError(
            f"Expected exactly 1 all_reduce_dep_snode, got {len(all_reduce_dep_snodes)}"
        )
    all_reduce_dep_snode = all_reduce_dep_snodes[0]

    grouped_snode = scheduler.GroupedSchedulerNode.create(
        [all_reduce_dep_snode, all_reduce_snode]
    )
    new_snode_order = []
    new_snode_order.append(grouped_snode)
    for snode in snodes:
        if snode in grouped_snode.snodes:
            continue
        new_snode_order.append(snode)
    return new_snode_order


@requires_accelerator_dist_backend()
@unittest.skipIf(
    torch._inductor.config.triton.native_matmul,
    "native matmul is fused with surrounding ops",
)
@instantiate_parametrized_tests
class TestComputeCommReorderingMultiProc(DynamoDistributedMultiProcTestCase):
    # [CLASS DOC] Multi-process (two ranks, fake process group when fewer than two GPUs)
    # Inductor test case requiring an accelerator distributed backend and non-native matmul.
    # Tests patch Inductor config for each reordering pass, compile small functions and
    # check kernel ordering in the generated code.
    """
    Run correctness checks in multi-proc runner, mark with minimum # GPUs to run under
    """

    def get_world_trs(self):
        # [FUNCTION DOC] HELPER: Returns a mapping with an empty tag, the list of all ranks
        # (0 to world size minus one) and the group size, used as keyword arguments that
        # describe the process group to the functional collectives.
        return {
            "tag": "",
            "ranks": list(range(self.world_size)),
            "group_size": self.world_size,
        }

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Property fixing the world size at two regardless of the
        # number of GPUs available.
        # hack: no matter whether we have 2 or 3 or 4 gpus, just run on 2
        # works around issue with skipif<2 and workers with unpredictable #s gpu
        return 2

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_locality", False)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "sink_waits",
        ],
    )
    def test_sink_waits(self):
        # [FUNCTION DOC] With only the sink_waits pass enabled, compiles a function doing an
        # all_reduce followed by two matmuls and checks the generated code places the wait
        # after the first matmul and before the second. Also checks the compiled output
        # matches eager. Runs inside a per-rank dynamo distributed init.
        def func(a):
            # [FUNCTION DOC] HELPER: Model function: all_reduce of a, matmul of a with itself,
            # then matmul of the reduced tensor with that product.
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
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs)
            # Verify that the wait_tensor is sinked below the 1st matmul but
            # above the 2nd matmul.
            (
                FileCheck()
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("extern_kernels.mm")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("extern_kernels.mm")
                .run(code)
            )
            out = compiled(inputs)
            correct = func(inputs)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_locality", False)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "raise_comms",
        ],
    )
    def test_raise_comms(self):
        # [FUNCTION DOC] With only the raise_comms pass enabled, compiles a function whose
        # all_reduce depends on the first matmul, and checks the collective is issued after
        # the first relu (it writes into the first matmul's buffer) but before the second
        # matmul, with the wait before the final matmul. Also compares against eager.
        def func(a):
            # [FUNCTION DOC] HELPER: Model function: matmul, relu, matmul chain, plus an
            # all_reduce of the first matmul result; returns the product of the chain end
            # with the reduced tensor.
            b = torch.matmul(a, a)
            c = torch.relu(b)
            d = torch.matmul(c, c)
            e = _functional_collectives.all_reduce(b, "sum", "0")
            return torch.matmul(d, e)

        with _dynamo_dist_per_rank_init(
            self.rank,
            self.world_size,
            self.backend(device_type),
            fake_pg=not at_least_x_gpu(2),
        ):
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs)
            # Verify that the all_reduce_ has been raised above the 2nd matmul
            # but below the 1st matmul. Note that the all_reduce_ directly
            # writes to the output buffer of the 1st matmul, which is an input
            # to the first relu. Therefore, the all_reduce_ should be scheduled
            # after the first relu.
            (
                FileCheck()
                .check("extern_kernels.mm")
                .check("triton_poi_fused_relu")
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check_same("buf0")
                # mm not use buf prior to wait_tensor
                .check("extern_kernels.mm")
                .check_not("buf0")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("extern_kernels.mm")
                .run(code)
            )
            out = compiled(inputs)
            correct = func(inputs)
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "sink_waits",
            "raise_comms",
        ],
    )
    def test_sink_waits_raise_comms(self):
        # [FUNCTION DOC] Enables both sink_waits and raise_comms and checks that the
        # all_reduce and its clone prologue (not fused with relus) are raised between the
        # first and second matmuls while the wait is sunk past the third matmul and before
        # the fourth. Also compares compiled and eager results.
        def func(a, *, tag, ranks, group_size):
            # [FUNCTION DOC] HELPER: Model function: matmul chain with relus, an all_reduce of
            # the first matmul result, and a final matmul of the reduced tensor with the
            # chain output. The keyword arguments are accepted but unused.
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
            inputs = torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # Things to verify:
            # - The clone prologue of the all_reduce_ should not be fused with
            # any relus.
            # - The all_reduce_ and its prologue should be raised above the 2nd
            # matmul but below the 1st matmul.
            # - The wait_tensor should be sinked below the 3rd matmul but above
            # the 4th matmul.
            (
                FileCheck()
                .check("extern_kernels.mm")
                .check("triton_poi_fused_all_reduce_0")
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("triton_poi_fused_relu")
                .check("extern_kernels.mm")
                .check("triton_poi_fused_relu")
                .check("extern_kernels.mm")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("extern_kernels.mm")
                .run(code)
            )
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "reorder_compute_for_overlap",
        ],
    )
    @patch.object(
        torch._inductor.config,
        "runtime_estimations_mms_benchmark",
        False,
    )
    def test_reorder_compute_for_overlap(self):
        # [FUNCTION DOC] Enables only the reorder_compute_for_overlap pass (matmul runtime
        # benchmarking off) on a function with two dependent all_reduces and checks the
        # emitted order: first collective, compute needed by the second collective, compute
        # independent of both, the wait, the fused mul, second collective and wait, then the
        # final add and matmul. Also compares compiled and eager results.
        def func(a, *, tag, ranks, group_size):
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
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # NOTE: after scheduling the first all_reduce:
            # 1. we first schedule the ops (c and d) that ARE required for second all_reduce but DO NOT depend on first all_reduce.
            # 2. then, we schedule the ops (g) that ARE NOT required for second all_reduce and DO NOT depend on first all_reduce.
            # 3. then, we schedule the ops (f) that ARE required for second all_reduce and DO depend on first all_reduce.
            # and then, we schedule the second all_reduce. And then schedule all ops that depend on second all_reduce.
            (
                FileCheck()
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("triton_poi_fused_relu")
                .check("extern_kernels.mm")
                .check("extern_kernels.mm")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("triton_poi_fused_all_reduce_mul")
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("triton_poi_fused_add")
                .check("extern_kernels.mm")
                .run(code)
            )
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config, "allow_buffer_reuse", True)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", True)
    @patch.object(
        torch._inductor.config,
        "reorder_for_compute_comm_overlap_passes",
        [
            "reorder_compute_for_overlap",
        ],
    )
    @patch.object(
        torch._inductor.config,
        "estimate_op_runtime",
        get_snode_runtime_for_reorder_compute_test,
    )
    def test_reorder_compute_for_overlap_custom_runtime_estimation(self):
        # [FUNCTION DOC] Same function and expected kernel order as the previous test, but
        # the pass uses the custom runtime estimator configured via estimate_op_runtime
        # (collectives cost most). Compares compiled output with eager.
        def func(a, *, tag, ranks, group_size):
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
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inputs, **self.get_world_trs())
            # NOTE: after scheduling the first all_reduce:
            # 1. we first schedule the ops (c and d) that ARE required for second all_reduce but DO NOT depend on first all_reduce.
            # 2. then, we schedule the ops (g) that ARE NOT required for second all_reduce and DO NOT depend on first all_reduce.
            # 3. then, we schedule the ops (f) that ARE required for second all_reduce and DO depend on first all_reduce.
            # and then, we schedule the second all_reduce. And then schedule all ops that depend on second all_reduce.
            (
                FileCheck()
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("triton_poi_fused_relu")
                .check("extern_kernels.mm")
                .check("extern_kernels.mm")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("triton_poi_fused_all_reduce_mul")
                .check("torch.ops._c10d_functional.all_reduce_.default")
                .check("torch.ops._c10d_functional.wait_tensor.default")
                .check("triton_poi_fused_add")
                .check("extern_kernels.mm")
                .run(code)
            )
            out = compiled(inputs, **self.get_world_trs())
            correct = func(inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @unittest.skipIf(
        torch._inductor.config.triton.native_matmul,
        "native matmul is fused with surrounding ops",
    )
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    @patch.object(
        torch._inductor.config,
        "_pre_fusion_custom_pass",
        create_grouped_node_for_allreduce_and_its_deps,
    )
    @parametrize("combo_kernels", (False, True))
    def test_grouped_scheduler_node(self, combo_kernels):
        # [FUNCTION DOC] Parametrized on the combo_kernels flag. Installs the pre-fusion pass
        # that groups the all_reduce with its producer, compiles a function and checks the
        # producer ops (add, div) are still fused with each other, then the collective, and
        # that the independent multiply is a separate kernel (grouped nodes cannot fuse with
        # outside ops). Also compares compiled and eager results.
        def func(a, *, tag, ranks, group_size):
            # [FUNCTION DOC] HELPER: Model function: elementwise add and divide feeding an
            # all_reduce, an independent elementwise multiply, and a matmul of the multiply
            # with the reduced tensor.
            add = a + a
            div = add / a
            ar = _functional_collectives.all_reduce(div, "sum", ranks, tag)
            # Normally, we would fuse `add = a + a`, `div = add / a` and `mul = a * a` together into a single fused op,
            # but here in this unit test, we intentionally put `add`, `div` and `ar` computation
            # into a GroupedSchedulerNode, which prevents them from being fused with any other ops.
            mul = a * a
            mm = torch.matmul(mul, ar)
            return (mm,)

        with torch._inductor.config.patch(combo_kernels=combo_kernels):
            with _dynamo_dist_per_rank_init(
                self.rank,
                self.world_size,
                self.backend(device_type),
                fake_pg=not at_least_x_gpu(2),
            ):
                inputs = (
                    torch.ones(4, 4, dtype=torch.float, device=device_type) + self.rank
                )
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
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_inductor_default_comms_ordering(self):
        # [FUNCTION DOC] Compiles a function that launches three all_reduces and waits on them
        # in reverse order, with a post-grad pass that asserts the graph keeps all_reduces in
        # launch order followed by waits in reverse order. Runs on a fake process group.
        # Side effect: assigns the post_grad_custom_post_pass config globally and never
        # restores it. Only the graph assertion runs; there is no numerical check.
        # KNOWN ISSUE: the global config assignment is not undone, so it can leak into
        # tests that run later in the same process.
        pg_info = self.get_world_trs()
        tag = pg_info["tag"]
        ranks = pg_info["ranks"]
        group_size = pg_info["group_size"]

        g1 = torch.ones(10, 10, device=device_type)
        g2 = torch.ones(11, 11, device=device_type)
        g3 = torch.ones(12, 12, device=device_type)

        def assert_pass(graph):
            # [FUNCTION DOC] HELPER: Post-grad pass callback; compares the printed graph with
            # the expected text via the expect-test helper (all_reduces in order, then waits
            # reversed). Raises on mismatch; returns nothing.
            # all_reduces need to remain in order!
            self.assertExpectedInline(
                graph,
                """\
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %arg2_1 : [num_users=1] = placeholder[target=arg2_1]
    %all_reduce : [num_users=1] = call_function[target=torch.ops._c10d_functional.all_reduce.default](args = (%arg0_1, avg, 0), kwargs = {})
    %all_reduce_1 : [num_users=1] = call_function[target=torch.ops._c10d_functional.all_reduce.default](args = (%arg1_1, avg, 0), kwargs = {})
    %all_reduce_2 : [num_users=1] = call_function[target=torch.ops._c10d_functional.all_reduce.default](args = (%arg2_1, avg, 0), kwargs = {})
    %wait_tensor : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor.default](args = (%all_reduce_2,), kwargs = {})
    %wait_tensor_1 : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor.default](args = (%all_reduce_1,), kwargs = {})
    %wait_tensor_2 : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor.default](args = (%all_reduce,), kwargs = {})
    return (wait_tensor, wait_tensor_1, wait_tensor_2)""",
            )

        torch._inductor.config.post_grad_custom_post_pass = assert_pass

        @torch.compile
        def fn(g1, g2, g3):
            # [FUNCTION DOC] HELPER: Launches an averaging all_reduce on each of three
            # tensors, then waits on them in reverse order; returns the three results
            # ordered third, second, first.
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
            fn(g1, g2, g3)

    def test_nccl_heuristics(self):
        # [FUNCTION DOC] Checks shape consistency of the NCCL cost-model tables: base latency
        # has one row per algorithm with one entry per protocol; hardware latency has one
        # entry per hardware type, each with per-algorithm rows of per-protocol entries; the
        # low-latency max bandwidth table has one entry per GPU type. Raises AssertionError
        # on any mismatch.
        if len(baseLat) != len(NCCL_ALGO):
            raise AssertionError(
                f"Expected len(baseLat) == len(NCCL_ALGO), got {len(baseLat)} vs {len(NCCL_ALGO)}"
            )
        if not all(len(x) == len(NCCL_PROTO) for x in baseLat):
            raise AssertionError(
                "Expected all elements in baseLat to have len(NCCL_PROTO)"
            )

        if len(hwLat) != len(NCCL_HW):
            raise AssertionError(
                f"Expected len(hwLat) == len(NCCL_HW), got {len(hwLat)} vs {len(NCCL_HW)}"
            )
        if not all(len(x) == len(NCCL_ALGO) for x in hwLat):
            raise AssertionError(
                "Expected all elements in hwLat to have len(NCCL_ALGO)"
            )
        if not all(len(y) == len(NCCL_PROTO) for x in hwLat for y in x):
            raise AssertionError(
                "Expected all nested elements in hwLat to have len(NCCL_PROTO)"
            )

        if len(llMaxBws) != len(NVIDIA_GPU_TYPE):
            raise AssertionError(
                f"Expected len(llMaxBws) == len(NVIDIA_GPU_TYPE), got {len(llMaxBws)} vs {len(NVIDIA_GPU_TYPE)}"
            )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
