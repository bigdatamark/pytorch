# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for inductor scheduler memory heuristics: reordering of operators to reduce
# peak memory (default, least-peak-memory-first, breadth-first and depth-first orderings),
# memory-planning size propagation across mutations, fusion limits on accumulated reads,
# reductions fusing, buffer lifecycle tracking with multiple mutations, and stability of
# torch.cond ordering under reordering. Tests run on the GPU device, mostly compiling small
# modules and matching expected buffer-order text in generated Triton code. Import-time: tries
# to import Triton and sets a flag recording availability; no other side effects. The main block
# runs only with a GPU.
import unittest
from unittest import mock

import torch
from torch._C import FileCheck
from torch._dynamo.utils import same
from torch._inductor import config, memory
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.common_utils import serialTest
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


try:
    import triton
    from triton import language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class Foo(torch.nn.Module):
    # [CLASS DOC] HELPER: Small module with four parameter matrices (ones) that computes two
    # chains of matmuls from the same input and returns the sum of their element sums; the
    # docstring shows the expected dataflow graph. Used as the model under test in the reorder
    # tests.
    """
    The default compiled graph is
    graph():
        ...
        %op0 : [num_users=2] = call_function[...](args = (%primals_2, %primals_1), ...)
        %op1 : [num_users=2] = call_function[...](args = (%primals_2, %primals_3), ...)
        %op2 : [num_users=1] = call_function[...](args = (%op0, %primals_4), ...)
        %op3 : [num_users=1] = call_function[...](args = (%op1, %primals_5), ...)
        %op4 : [num_users=1] = call_function[...](args = (%op2,), ...)
        %op5 : [num_users=1] = call_function[...](args = (%op3,), ...)
        %op6_op7 : [num_users=1] = call_function[...](args = (%op5, %op4), ...)
    """

    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates four all-ones parameters of shapes 2x10, 2x2, 10x1 and
        # 2x10.
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.ones(2, 10))
        self.w2 = torch.nn.Parameter(torch.ones(2, 2))
        self.w3 = torch.nn.Parameter(torch.ones(10, 1))
        self.w4 = torch.nn.Parameter(torch.ones(2, 10))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Computes products x with w1 and with w2, then multiplies those
        # with w3 and w4 respectively, and returns the sum of the two results' element sums.
        t1 = torch.matmul(x, self.w1)
        t2 = torch.matmul(x, self.w2)
        t3 = torch.matmul(t1, self.w3)
        t4 = torch.matmul(t2, self.w4)
        return t3.sum() + t4.sum()


# The tests in this class uses very small tensors. The default
# score_fusion_memory threshold will cause different fusion decisions and
# generate a different wrapper. Override the threshold to make these tests
# happy.
@config.patch("score_fusion_memory_threshold", 1)
class TestOperatorReorderForPeakMemory(TestCase):
    # [CLASS DOC] Tests for operator reordering to reduce peak memory, with the fusion memory
    # score threshold lowered to one so tiny tensors still produce a stable wrapper. Setup
    # builds the model and inputs; each test compares generated buffer ordering text. State set
    # in setUp: model, inputs and the original reorder function.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setup, moves the Foo model to the GPU, builds an
        # input of ones with 2048 rows (4096 on HIP) and two columns, and saves the original
        # reorder_for_peak_memory function of the memory module.
        super().setUp()

        self.model = Foo().to(GPU_TYPE)
        M = 4096 if torch.version.hip is not None else 2048
        self.inputs = torch.ones((M, 2), device=GPU_TYPE)
        self.orig_reorder_method = memory.reorder_for_peak_memory

    @mock.patch.object(config, "reorder_for_peak_memory", True)
    def test_reorder_peak_memory(self):
        # [FUNCTION DOC] With reorder-for-peak-memory enabled, compiles the model and checks the
        # generated Triton code defines the call function (class method form under graph
        # partition) and allocates buffers in a specific interleaved order, then checks the
        # compiled output matches eager.
        outp_corr = self.model(self.inputs)
        compiled_model = torch.compile(self.model)
        code = run_and_get_triton_code(compiled_model, self.inputs)

        call_str = (
            "def call(self, args):"
            if torch._inductor.config.graph_partition
            else "def call(args):"
        )

        (
            FileCheck()
            .check(call_str)
            .check("buf1 = ")
            .check("buf0 = ")
            .check("buf2 = ")
            .check("buf4 = ")
            .check("buf3 = ")
            .check("buf5 = ")
            .check("buf7 = ")
            .run(code)
        )
        # check for correctness
        outp = compiled_model(self.inputs)
        self.assertTrue(same(outp, outp_corr))

    @mock.patch.object(config, "reorder_for_peak_memory", True)
    def test_reorder_peak_memory_lpmf(self):
        # [FUNCTION DOC] Same as the default reorder test but patches the reorder function so
        # only the least-peak-memory-first topological sort is used; checks the generated buffer
        # order and numerical agreement with eager.
        outp_corr = self.model(self.inputs)

        def reorder_with_only_lpmf(
            nodes,
            name_to_buf,
            name_to_fused_node,
            graph_inputs,
            graph_outputs,
            methods=None,
        ):
            # [FUNCTION DOC] HELPER: Replacement for the reorder function with the same
            # arguments (nodes, buffer maps, graph inputs and outputs, and an ignored methods
            # argument) that calls the original reorder function restricting it to the
            # least-peak-memory-first sort. Returns the reordered node list.
            return self.orig_reorder_method(
                nodes,
                name_to_buf,
                name_to_fused_node,
                graph_inputs,
                graph_outputs,
                methods=[memory.topological_sort_lpmf],
            )

        call_str = (
            "def call(self, args):"
            if torch._inductor.config.graph_partition
            else "def call(args):"
        )

        with mock.patch.object(
            memory, "reorder_for_peak_memory", reorder_with_only_lpmf
        ):
            compiled_model = torch.compile(self.model)

            code = run_and_get_triton_code(compiled_model, self.inputs)
            (
                FileCheck()
                .check(call_str)
                .check("buf1 = ")
                .check("buf0 = ")
                .check("buf2 = ")
                .check("buf4 = ")
                .check("buf3 = ")
                .check("buf5 = ")
                .check("buf7 = ")
                .run(code)
            )
            # check for correctness
            outp = compiled_model(self.inputs)
            self.assertTrue(same(outp, outp_corr))

    @mock.patch.object(config, "reorder_for_peak_memory", True)
    def test_reorder_peak_memory_bfs(self):
        # [FUNCTION DOC] Same pattern as the default reorder test but restricted to
        # breadth-first topological sort; the expected buffer order is sequential (buf0 to buf5,
        # buf7); also checks equality with eager.
        outp_corr = self.model(self.inputs)

        def reorder_with_only_bfs(
            nodes,
            name_to_buf,
            name_to_fused_node,
            graph_inputs,
            graph_outputs,
            methods=None,
        ):
            # [FUNCTION DOC] HELPER: Replacement reorder function forwarding all arguments to
            # the original but forcing the breadth-first topological sort method. Returns its
            # result.
            return self.orig_reorder_method(
                nodes,
                name_to_buf,
                name_to_fused_node,
                graph_inputs,
                graph_outputs,
                methods=[memory.topological_sort_bfs],
            )

        call_str = (
            "def call(self, args):"
            if torch._inductor.config.graph_partition
            else "def call(args):"
        )

        with mock.patch.object(
            memory, "reorder_for_peak_memory", reorder_with_only_bfs
        ):
            compiled_model = torch.compile(self.model)

            code = run_and_get_triton_code(compiled_model, self.inputs)

            (
                FileCheck()
                .check(call_str)
                .check("buf0 = ")
                .check("buf1 = ")
                .check("buf2 = ")
                .check("buf3 = ")
                .check("buf4 = ")
                .check("buf5 = ")
                .check("buf7 = ")
                .run(code)
            )
            # check for correctness
            outp = compiled_model(self.inputs)
            self.assertTrue(same(outp, outp_corr))

    @mock.patch.object(config, "reorder_for_peak_memory", True)
    def test_reorder_peak_memory_dfs(self):
        # [FUNCTION DOC] Same pattern but restricted to depth-first topological sort; the
        # expected buffer order follows each dependent chain (buf0, buf2, buf4, then buf1, buf3,
        # buf5, buf7); also checks equality with eager.
        outp_corr = self.model(self.inputs)

        def reorder_with_only_dfs(
            nodes,
            name_to_buf,
            name_to_fused_node,
            graph_inputs,
            graph_outputs,
            methods=None,
        ):
            # [FUNCTION DOC] HELPER: Replacement reorder function forwarding all arguments to
            # the original but forcing the depth-first topological sort method. Returns its
            # result.
            return self.orig_reorder_method(
                nodes,
                name_to_buf,
                name_to_fused_node,
                graph_inputs,
                graph_outputs,
                methods=[memory.topological_sort_dfs],
            )

        call_str = (
            "def call(self, args):"
            if torch._inductor.config.graph_partition
            else "def call(args):"
        )

        with mock.patch.object(
            memory, "reorder_for_peak_memory", reorder_with_only_dfs
        ):
            compiled_model = torch.compile(self.model)

            code = run_and_get_triton_code(compiled_model, self.inputs)
            (
                FileCheck()
                .check(call_str)
                .check("buf0 = ")
                .check("buf2 = ")
                .check("buf4 = ")
                .check("buf1 = ")
                .check("buf3 = ")
                .check("buf5 = ")
                .check("buf7 = ")
                .run(code)
            )
            # check for correctness
            outp = compiled_model(self.inputs)
            self.assertTrue(same(outp, outp_corr))

    @mock.patch.object(config, "allow_buffer_reuse", False)
    @unittest.skipUnless(TRITON_AVAILABLE, "Triton is not available")
    @config.patch("test_configs.track_memory_lifecycle", "assert")
    def test_mutation_size_propagation(self):
        # [FUNCTION DOC] With buffer reuse disabled and memory lifecycle tracking in assert
        # mode, compiles a loop of Triton conversions to bfloat16 followed by matmuls, recording
        # memory planning info of every scheduler buffer. For each pre-mutation buffer it checks
        # allocation and free sizes of 2048 bytes, for the matching post-mutation buffer both
        # zero, and that the successor nodes of the mutated buffer are a subset of those of the
        # original. Needs Triton.
        """
        This tests correct size propagation in the case of mutations.
        In this example, buf1 is a mutation of buf0; we should have:
        * buf0: has size_alloc 2048 and size_free 0;
        * buf1: has size_alloc 0 and size_free 2048.
        This is because
        - when buf1 is created, no additional memory is used; and
        - the 2048 bytes of memory can only be released when buf1 is freed.
        Similar arguments for buf2 and buf3, buf4 and buf5, etc.
        """

        # using triton custom kernel to creat small example with mutations
        # [FUNCTION DOC] HELPER: Triton kernel converting a block of input values to bfloat16
        # and storing them to the output pointer, with a mask on the element count. Each program
        # instance handles one contiguous block.
        @triton.jit
        def convert_to_bf16_kernel(
            input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(input_ptr + offsets, mask=mask)
            x_bf16 = x.to(tl.bfloat16)
            tl.store(output_ptr + offsets, x_bf16, mask=mask)

        def convert_to_bf16(x):
            # [FUNCTION DOC] HELPER: Allocates an empty bfloat16 tensor like the input, launches
            # the Triton conversion kernel over flattened views with a one-dimensional grid of
            # 1024-element blocks, and returns the output viewed in the input shape.
            output = torch.empty_like(x, dtype=torch.bfloat16)
            n_elements = x.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            convert_to_bf16_kernel[grid](
                x.flatten(), output.flatten(), n_elements, BLOCK_SIZE
            )
            return output.view(x.shape)

        # create a custom function to record the buffer size information
        buffer_info = {}
        og_method = memory.assign_memory_planning_info_for_scheduler_buffers

        def assign_memory_planning_info_for_scheduler_buffers_with_records(
            nodes, name_to_buf
        ):
            # [FUNCTION DOC] HELPER: Wrapper that calls the original planning-info assignment
            # and then stores, per buffer name, the allocated size, freed size and successor
            # nodes of its planning info in the shared mapping.
            og_method(nodes, name_to_buf)
            for buf_name, buf in name_to_buf.items():
                buffer_info[buf_name] = (
                    buf.mpi_buffer.size_alloc,
                    buf.mpi_buffer.size_free,
                    buf.mpi_buffer.succ_nodes,
                )

        # test example and checks
        def f(a, p):
            # [FUNCTION DOC] HELPER: Loops over a list of tensors, converts each to bfloat16 and
            # multiplies the running matrix by it; returns the final matrix.
            for e in a:
                e = convert_to_bf16(e)
                p = p @ e
            return p

        a = [torch.randn(32, 32, device=GPU_TYPE) for _ in range(4)]
        p = torch.ones(a[0].size(), dtype=torch.bfloat16, device=GPU_TYPE)

        with mock.patch.object(
            memory,
            "assign_memory_planning_info_for_scheduler_buffers",
            assign_memory_planning_info_for_scheduler_buffers_with_records,
        ):
            f_compiled = torch.compile(f)
            f_compiled(a, p)

            pre_mutation = ["buf0", "buf2", "buf4", "buf6"]
            post_mutation = ["buf1", "buf3", "buf5", "buf7"]

            for pre, post in zip(pre_mutation, post_mutation):
                self.assertEqual(buffer_info[pre][0:2], (2048, 2048))
                self.assertEqual(buffer_info[post][0:2], (0, 0))
                # succ nodes should be forwarded to pre mutation buffer
                self.assertTrue(buffer_info[post][2] <= buffer_info[pre][2])

    def test_fusing_reductions_increase_peak_memory(self):
        # [FUNCTION DOC] Runs the compiled function on large GPU tensors with peak memory stats
        # reset, then checks peak allocated memory is below twice the byte size of one full
        # product matrix, i.e. that the two products are not both materialized.
        @torch.compile
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Compiled function summing two matmul results along the last
            # dimension and adding them.
            return (a @ c).sum(dim=-1) + (b @ c).sum(dim=-1)

        a = torch.randn(1024 * 32, 16, device=GPU_TYPE)
        b = torch.randn(1024 * 32, 16, device=GPU_TYPE)
        c = torch.randn(16, 1024 * 32, device=GPU_TYPE)
        torch.get_device_module(GPU_TYPE).reset_peak_memory_stats()
        f(a, b, c)
        peak_mem = torch.get_device_module(GPU_TYPE).max_memory_allocated()

        expected_bound = a.size(0) * c.size(1) * a.dtype.itemsize * 2
        self.assertLess(peak_mem, expected_bound)

    @serialTest()
    def test_fusion_acc_large_reads(self):
        # [FUNCTION DOC] Serial test: replaces the inductor choices handler with a subclass that
        # limits fusion by the total size of accumulated reads against a config threshold, then
        # compiles a loop of four matmul-plus-add steps under three thresholds (infinite, three
        # input sizes, one input size) and checks the generated fused kernel calls differ
        # accordingly. Side effect: installs the custom handler globally and never restores it.
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Starts from zeros shaped like the first row of x, then four
            # times adds a matmul of x and y plus z to the running result; returns the result.
            res = torch.zeros_like(x[0])
            for _ in range(4):
                temp = torch.matmul(x, y) + z
                res = res + temp
            return res

        N = 128
        x = torch.rand(N, N, dtype=torch.float32, device=GPU_TYPE)
        y = torch.rand(N, N, dtype=torch.float32, device=GPU_TYPE)
        # Keep the add as pointwise so this test continues to exercise scheduler
        # fusion choices instead of addmm fusion.
        z = torch.rand(1, N, dtype=torch.float32, device=GPU_TYPE).expand(N, N)

        from torch._inductor.choices import InductorChoices
        from torch._inductor.scheduler import BaseSchedulerNode, Scheduler

        class CustomInductorChoices(InductorChoices):
            # [CLASS DOC] HELPER: Choices subclass overriding only the fusion decision. A pair
            # of nodes fuses only if the default says so and, when a read-size threshold is
            # configured, the combined size of reads not produced by the pair is under the
            # threshold.
            @staticmethod
            def can_fuse(
                scheduler: Scheduler,
                node1: BaseSchedulerNode,
                node2: BaseSchedulerNode,
                shared_data_score: int,
            ) -> bool:
                # [FUNCTION DOC] HELPER: Static method deciding whether two scheduler nodes
                # fuse: defers to the default decision, returns it directly if false or if the
                # accumulation threshold config is zero or unset; otherwise computes the union
                # of the reads of both nodes minus their writes, sums their size hints, and
                # returns whether the sum is below the threshold.
                can_fuse_default = InductorChoices.can_fuse(
                    scheduler, node1, node2, shared_data_score
                )
                if (not can_fuse_default) or (
                    not config.realize_acc_reads_size_threshold
                ):
                    return can_fuse_default

                all_reads = (node1.read_writes.reads | node2.read_writes.reads) - (
                    node1.read_writes.writes | node2.read_writes.writes
                )
                size_of_reads = [scheduler.dep_size_hint(dep) for dep in all_reads]
                return sum(size_of_reads) < config.realize_acc_reads_size_threshold

        torch._inductor.virtualized.V.set_choices_handler(CustomInductorChoices())

        # CASE 1: no restriction on the amount of accumulation
        with config.patch({"realize_acc_reads_size_threshold": float("inf")}):
            f_compiled = torch.compile(f)
            code = run_and_get_triton_code(f_compiled, x, y, z)
            (
                FileCheck()
                .check("triton_poi_fused_add_0.run(buf4, arg2_1, buf1, buf2, buf3")
                .run(code)
            )

        # CASE 2: for tensors with the same size as x (which is 4 * N**2 bytes)
        # at most 12 / 4 = 3 reads can be accumulated during fusion
        with config.patch({"realize_acc_reads_size_threshold": 12 * N**2}):
            f_compiled = torch.compile(f)
            code = run_and_get_triton_code(f_compiled, x, y, z)
            (
                FileCheck()
                .check("triton_poi_fused_add_0.run(buf3, arg2_1, buf1, buf2,")
                .check("triton_poi_fused_add_1.run(buf5, buf4, arg2_1,")
                .run(code)
            )

        # CASE 3: no such fusion allowed
        with config.patch({"realize_acc_reads_size_threshold": N**2}):
            f_compiled = torch.compile(f)
            code = run_and_get_triton_code(f_compiled, x, y, z)
            (
                FileCheck()
                .check("triton_poi_fused_add_0.run(buf2, arg2_1, buf1,")
                .check("triton_poi_fused_add_1.run(buf4, buf3, arg2_1")
                .check("triton_poi_fused_add_1.run(buf6, buf5, arg2_1,")
                .run(code)
            )

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton is not available")
    def test_multiple_mutations_of_buf(self):
        # [FUNCTION DOC] With the post-grad pass swapping a functional foreach add for its
        # in-place variant, buffer reuse and reordering off, and memory lifecycle tracking in
        # assert mode, compiles a function with views of a matmul result mutated by the foreach
        # op, and checks the generated code lists buffer 0 as allocated. Needs Triton.
        @torch.compile()
        def foo(inp, inp2):
            # [FUNCTION DOC] HELPER: Compiled function: squares a matrix, views it as two
            # slices, adds one to both with the foreach op, and returns a sum, a row-wise sum
            # and a separate matmul of the second input.
            inp = inp @ inp
            inp = inp.view(2, -1, 256)
            x = inp[0]
            y = inp[1]
            x, y = torch._foreach_add([x, y], 1.0)
            out = x.sum()
            out2 = y.sum(dim=-1)

            return out, out2, inp2 @ inp2

        inp = torch.rand([256, 256], device=GPU_TYPE)
        inp2 = torch.rand([256, 256], device=GPU_TYPE)

        def replace_foreach(gm):
            # [FUNCTION DOC] HELPER: Post-grad graph pass. Requires exactly one functional
            # foreach-add scalar node, retargets it to the in-place variant, and then for each
            # input and its getitem user rewires users of the output to the original input and
            # erases the output node. Raises an assertion error if the node count is not one.
            # Mutates the graph in place.
            nodes = gm.find_nodes(
                op="call_function", target=torch.ops.aten._foreach_add.Scalar
            )
            if len(nodes) != 1:
                raise AssertionError
            node = nodes[0]
            nodes[0].target = torch.ops.aten._foreach_add_.Scalar
            for inp, out in zip(node.args[0], list(node.users.keys())):
                out.replace_all_uses_with(inp)
                gm.erase_node(out)

        with torch._inductor.config.patch(
            {
                "post_grad_custom_post_pass": replace_foreach,
                "test_configs.track_memory_lifecycle": "assert",
                "allow_buffer_reuse": False,
                # make sure the mm is at the end so
                # the earlier deallocation is not at the last step,
                # which doesn't distinguish between returned tensors
                # and which tensors are deallocated immediately prior
                "reorder_for_peak_memory": False,
            }
        ):
            code = run_and_get_triton_code(foo, inp, inp2)
            FileCheck().check("allocated=['buf0']").run(code)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton is not available")
    def test_torch_cond_ordering_consistency(self):
        # [FUNCTION DOC] Builds a module with three conditionals on a counter parity, one over a
        # large buffer and two over small ones, compiles it with reordering off and then on,
        # extracts the order of conditional branch buffer allocations from generated code and,
        # if both orders are found, checks they are identical (reordering could otherwise hang
        # distributed collectives). Needs Triton.
        small_sz, large_sz = 256, 1024

        class MultiCondModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three constant zero buffers (one large, two small)
            # and a long counter; the forward runs three conditionals and adds the sums of their
            # outputs to the input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers the large buffer, two small buffers and a
                # counter buffer.
                super().__init__()
                self.register_buffer("large_buffer", torch.zeros(large_sz))
                self.register_buffer("small_buffer1", torch.zeros(small_sz))
                self.register_buffer("small_buffer2", torch.zeros(small_sz))
                self.register_buffer("counter", torch.tensor(0, dtype=torch.long))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Computes whether the counter is even, runs three
                # conditionals each cloning one buffer (doubling it in the true branch), and
                # returns the input plus the sums of the three results.
                condition = self.counter % 2 == 0

                def true_fn_large(buf):
                    # [FUNCTION DOC] HELPER: Branch returning the buffer cloned and doubled.
                    return buf.clone() * 2.0

                def false_fn_large(buf):
                    # [FUNCTION DOC] HELPER: Branch returning a clone of the buffer.
                    return buf.clone()

                def true_fn_small(buf):
                    # [FUNCTION DOC] HELPER: Branch returning the buffer cloned and doubled
                    # (same as the large true branch).
                    return buf.clone() * 2.0

                def false_fn_small(buf):
                    # [FUNCTION DOC] HELPER: Branch returning a clone of the buffer (same as the
                    # large false branch).
                    return buf.clone()

                result_large = torch.cond(
                    condition,
                    lambda: true_fn_large(self.large_buffer),
                    lambda: false_fn_large(self.large_buffer),
                )
                result_small1 = torch.cond(
                    condition,
                    lambda: true_fn_small(self.small_buffer1),
                    lambda: false_fn_small(self.small_buffer1),
                )
                result_small2 = torch.cond(
                    condition,
                    lambda: true_fn_small(self.small_buffer2),
                    lambda: false_fn_small(self.small_buffer2),
                )
                return (
                    x + result_large.sum() + result_small1.sum() + result_small2.sum()
                )

        def extract_cond_order(code: str) -> list[tuple[str, int]]:
            # [FUNCTION DOC] HELPER: Scans generated code line by line for assignments to
            # the buffer named after a true branch graph, and returns a list of pairs of
            # conditional name and first size number, in order of appearance.
            """
            Extract the order of torch.cond operations from generated code.
            Returns list of (cond_name, buffer_size) tuples in execution order.
            """
            import re

            cond_order = []
            # Look for patterns like "cond" or "cond_1" in the generated code
            # along with their buffer sizes
            lines = code.split("\n")
            for i, line in enumerate(lines):
                # Match true_graph buffer allocations which indicate cond execution
                match = re.search(r"true_graph_(\d+)_buf0\s*=.*\((\d+),", line)
                if match:
                    cond_idx = int(match.group(1))
                    buf_size = int(match.group(2))
                    cond_order.append((f"cond_{cond_idx}", buf_size))
            return cond_order

        model = MultiCondModel().to(GPU_TYPE)
        x = torch.randn(10, device=GPU_TYPE)

        # Compile with base settings (no reordering)
        torch._dynamo.reset()
        with config.patch({"reorder_for_peak_memory": False}):
            compiled_base = torch.compile(model)
            code_base = run_and_get_triton_code(compiled_base, x)

        base_order = extract_cond_order(code_base)

        # Compile with reorder_for_peak_memory=True
        torch._dynamo.reset()
        with config.patch({"reorder_for_peak_memory": True}):
            compiled_peak_mem = torch.compile(model)
            code_peak_mem = run_and_get_triton_code(compiled_peak_mem, x)

        peak_mem_order = extract_cond_order(code_peak_mem)

        if base_order and peak_mem_order:
            self.assertEqual(
                base_order,
                peak_mem_order,
                msg=(
                    lambda msg: f"{msg}\ntorch.cond operations were reordered by reorder_for_peak_memory!\n"
                    f"Base order: {base_order}\n"
                    f"Peak memory order: {peak_mem_order}\n"
                    f"This can cause NCCL hangs when torch.cond contains collective operations "
                    f"because different ranks may execute collectives in different orders."
                ),
            )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
