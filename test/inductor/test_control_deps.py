# Owner(s): ["module: inductor"]

# [MODULE DOC] Tests for inductor control dependency support: the preserve_node_ordering pass
# that wraps nodes in control_deps higher-order-op calls, interaction with nested args, auto-
# functionalized ops and triton kernels, stream record/wait event handling, the reinplace helper
# for ordering-only uses, and OrderingBarrier IR nodes. Most tests need a GPU. Defines a skip
# decorator for CUDA with triton. The main block runs the tests with the filelock requirement
# only on Linux with a GPU and triton.
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
    HAS_GPU_AND_TRITON,
    requires_gpu,
)


requires_cuda_triton = unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA")


class TestControlDeps(InductorTestCase):
    # [CLASS DOC] Inductor test case grouping control_deps tests: custom post-grad passes that
    # add ordering dependencies, then checks on generated code, scheduler behavior and IR nodes.
    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_control_deps_prevents_fusion(self):
        # [FUNCTION DOC] Checks that a user post-grad pass adding control dependencies (mm after
        # add, mul after mm) keeps those ops from being fused: generated code shows a kernel
        # run, then an extern mm call, then another kernel run, and results match eager.
        # Requires a GPU; locality reordering is disabled.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compile; computes an add on a, a matmul of b
            # with itself, and a multiply, returning the matmul and multiply results.
            c = a + 1
            d = b @ b
            e = c * 2
            return d, e

        def add_control_deps(graph):
            # [FUNCTION DOC] HELPER: post-grad pass. Expects exactly three call_function nodes
            # (add, mm, mul) and raises AssertionError otherwise, makes mm depend on add and mul
            # depend on mm via preserve_node_ordering, then verifies two control_deps nodes exist
            # and that all fake values (including inside the subgraphs) have the 256 by 256
            # shape. Returns the graph.
            nodes = [n for n in graph.nodes if n.op == "call_function"]
            if len(nodes) != 3:
                raise AssertionError(f"Expected 3 nodes, got {len(nodes)}")
            c_node = nodes[0]
            d_node = nodes[1]
            e_node = nodes[2]

            if d_node.target != torch.ops.aten.mm.default:
                raise AssertionError(f"Expected mm.default, got {d_node.target}")

            from torch.utils._ordered_set import OrderedSet

            deps_map = {d_node: OrderedSet([c_node]), e_node: OrderedSet([d_node])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )
            sub_g = graph.find_nodes(
                op="call_function", target=torch.ops.higher_order.control_deps
            )
            if len(sub_g) != 2:
                raise AssertionError(f"Expected 2 control_deps nodes, got {len(sub_g)}")

            if list(sub_g[0].meta["val"].shape) != [256, 256]:
                raise AssertionError(
                    f"Expected shape [256, 256], got {list(sub_g[0].meta['val'].shape)}"
                )
            if list(sub_g[1].meta["val"].shape) != [256, 256]:
                raise AssertionError(
                    f"Expected shape [256, 256], got {list(sub_g[1].meta['val'].shape)}"
                )

            for attr in graph.find_nodes(op="get_attr"):
                for n in getattr(graph.owning_module, attr.target).graph.nodes:
                    if list(n.meta["val"].shape) != [256, 256]:
                        raise AssertionError(
                            f"Expected shape [256, 256], got {list(n.meta['val'].shape)}"
                        )

            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            compiled_fn = torch.compile(fn)
            a = torch.rand([256, 256], device=GPU_TYPE)
            b = torch.rand([256, 256], device=GPU_TYPE)

            _, code = run_and_get_code(torch.compile(fn), a, b)
            result = compiled_fn(a, b)

            FileCheck().check(".run(").check("extern_kernels.mm(").check(".run(").run(
                code[0]
            )

            expected = fn(a, b)
            torch.testing.assert_close(result, expected)

    @config.patch(allow_buffer_reuse=False)
    @requires_gpu()
    def test_control_deps_do_not_extend_buffer_lifetime(self):
        # [FUNCTION DOC] Checks that a control dependency does not extend a buffer lifetime: with
        # buffer reuse disabled, a chain of four matmuls where the last depends on the first
        # still deletes the first buffer right after its single use, as seen in the generated
        # code. Results match eager. Requires a GPU.
        """
        Control deps should not extend buffer lifetimes - buf0/buf1 should be
        deleted before the 4th matmul, not kept alive by the control dependency.
        """

        def fn(a, b):
            # [FUNCTION DOC] HELPER: function under compile; a chain of four matmuls, each of the
            # previous result with b.
            # Chain of 4 matmuls: mm0 -> mm1 -> mm2 -> mm3
            mm0 = a @ b
            mm1 = mm0 @ b
            mm2 = mm1 @ b
            mm3 = mm2 @ b
            return mm3

        def add_control_deps(graph):
            # [FUNCTION DOC] HELPER: post-grad pass. Requires exactly four mm nodes (raises
            # AssertionError otherwise) and adds a control dependency of the last mm on the first
            # through preserve_node_ordering. Returns the graph.
            from torch.utils._ordered_set import OrderedSet

            mm_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mm.default
            )
            if len(mm_nodes) != 4:
                raise AssertionError(f"Expected 4 mm nodes, got {len(mm_nodes)}")

            # Add control dep: mm3 depends on mm0's output
            # This should NOT extend mm0's buffer lifetime
            deps_map = {mm_nodes[3]: OrderedSet([mm_nodes[0]])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            a = torch.rand([256, 256], device=GPU_TYPE)
            b = torch.rand([256, 256], device=GPU_TYPE)

            result, code = run_and_get_code(torch.compile(fn), a, b)
            torch.testing.assert_close(result, fn(a, b))

            # buf0 should be allocated, passed in out=, used once, then del
            FileCheck().check("buf0 = ").check_count(
                "extern_kernels.mm", 2, exactly=True
            ).check("del buf0").run(code[0])

    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_control_deps_with_nested_args(self):
        # [FUNCTION DOC] Checks control_deps on a node with nested args (a concatenation taking a
        # list of tensors): the pass makes the cat depend on an add and a single control_deps
        # node results; the compiled output matches eager. Requires a GPU.
        """Test control_deps with operations that have nested args (e.g., torch.cat)."""

        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: function under compile; adds one to a, doubles b,
            # concatenates along dim 0, then adds c.
            x = a + 1
            y = b * 2
            # torch.cat has nested args: (List[Tensor], dim)
            cat_result = torch.cat([x, y], dim=0)
            z = cat_result + c
            return z

        def add_control_deps(graph):
            # [FUNCTION DOC] HELPER: post-grad pass. Requires exactly one cat node whose first
            # arg is a list or tuple, makes it depend on the first add node, and verifies exactly
            # one control_deps node exists; raises AssertionError otherwise. Returns the graph.
            from torch.utils._ordered_set import OrderedSet

            # Find the cat node which has nested args
            cat_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.cat.default
            )
            if len(cat_nodes) != 1:
                raise AssertionError(f"Expected 1 cat node, got {len(cat_nodes)}")
            cat_node = cat_nodes[0]

            # Verify it has nested args (list of tensors)
            if not isinstance(cat_node.args[0], (list, tuple)):
                raise AssertionError(
                    f"Expected nested args, got {type(cat_node.args[0])}"
                )

            # Find a node that comes before cat to use as dependency
            add_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.add.Tensor
            )
            # Use the first add node (x = a + 1)
            dep_node = add_nodes[0]

            deps_map = {cat_node: OrderedSet([dep_node])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )

            # Verify control_deps was created
            control_deps_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.higher_order.control_deps
            )
            if len(control_deps_nodes) != 1:
                raise AssertionError(
                    f"Expected 1 control_deps node, got {len(control_deps_nodes)}"
                )
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            a = torch.rand([128, 64], device=GPU_TYPE)
            b = torch.rand([128, 64], device=GPU_TYPE)
            c = torch.rand([256, 64], device=GPU_TYPE)

            compiled_fn = torch.compile(fn)
            result = compiled_fn(a, b, c)

            expected = fn(a, b, c)
            torch.testing.assert_close(result, expected)

    @config.patch(enable_auto_functionalized_v2=True)
    def test_control_deps_with_auto_functionalized_v2(self):
        # [FUNCTION DOC] Checks that control_deps works on an auto_functionalized_v2 node: inside
        # a scoped custom library fragment it defines an in-place add-one op with a fake
        # implementation, compiles a function using it with a pass that makes the op depend on a
        # multiply, and verifies the control_deps subgraph contains the auto-functionalized node.
        # Compiled result and the mutated input match eager. The library is cleaned up on exit.
        with torch.library._scoped_library("control_deps_auto_func", "FRAGMENT") as lib:
            lib.define(
                "add_one_(Tensor(a!) x) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
            )

            def add_one_impl(x):
                # [FUNCTION DOC] HELPER: eager implementation of the in-place op: adds one to the
                # input in place and returns a clone.
                x.add_(1.0)
                return x.clone()

            lib.impl("add_one_", add_one_impl, "CompositeExplicitAutograd")

            @torch.library.register_fake("control_deps_auto_func::add_one_", lib=lib)
            def add_one_fake(x):
                # [FUNCTION DOC] HELPER: fake (meta) implementation of the op, returning a clone
                # of the input.
                return x.clone()

            def fn(x):
                # [FUNCTION DOC] HELPER: function under compile; doubles x, calls the in-place op
                # on x and adds the doubled value to the result.
                y = x * 2
                z = torch.ops.control_deps_auto_func.add_one_(x)
                return z + y

            def add_control_deps(graph):
                # [FUNCTION DOC] HELPER: post-grad pass. Requires exactly one
                # auto_functionalized_v2 node and one mul node, makes the former depend on the
                # latter, then verifies one control_deps node whose subgraph attribute holds
                # exactly one nested auto_functionalized_v2 node. Raises AssertionError on any
                # mismatch. Returns the graph.
                from torch.utils._ordered_set import OrderedSet

                auto_func_nodes = graph.find_nodes(
                    op="call_function",
                    target=torch.ops.higher_order.auto_functionalized_v2,
                )
                if len(auto_func_nodes) != 1:
                    raise AssertionError(
                        f"Expected 1 auto_functionalized_v2 node, got {len(auto_func_nodes)}"
                    )

                mul_nodes = graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mul.Tensor
                )
                if len(mul_nodes) != 1:
                    raise AssertionError(f"Expected 1 mul node, got {len(mul_nodes)}")

                torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                    graph, {auto_func_nodes[0]: OrderedSet([mul_nodes[0]])}
                )
                control_deps_nodes = graph.find_nodes(
                    op="call_function", target=torch.ops.higher_order.control_deps
                )
                if len(control_deps_nodes) != 1:
                    raise AssertionError(
                        f"Expected 1 control_deps node, got {len(control_deps_nodes)}"
                    )

                subgraph_attr = control_deps_nodes[0].args[1]
                if not isinstance(subgraph_attr, torch.fx.Node):
                    raise AssertionError(
                        f"Expected get_attr node for subgraph, got {type(subgraph_attr)}"
                    )
                subgraph = getattr(graph.owning_module, subgraph_attr.target)
                nested_auto_func_nodes = subgraph.graph.find_nodes(
                    op="call_function",
                    target=torch.ops.higher_order.auto_functionalized_v2,
                )
                if len(nested_auto_func_nodes) != 1:
                    raise AssertionError(
                        "Expected control_deps subgraph to contain "
                        f"1 auto_functionalized_v2 node, got {len(nested_auto_func_nodes)}"
                    )
                return graph

            x = torch.zeros(8)
            compiled_x = x.clone()
            eager_x = x.clone()

            with torch._inductor.config.patch(
                post_grad_custom_post_pass=add_control_deps,
            ):
                result = torch.compile(fn, backend="inductor", fullgraph=True)(
                    compiled_x
                )

            expected = fn(eager_x)
            torch.testing.assert_close(result, expected)
            torch.testing.assert_close(compiled_x, eager_x)

    @requires_gpu()
    def test_control_deps_with_triton_kernel(self):
        # [FUNCTION DOC] Checks control_deps around a triton kernel wrapper: a pass makes the
        # first functional triton kernel node depend on the first mul node; the compiled output
        # equals eager. Requires a GPU and triton imports.
        """Test control_deps with triton_kernel_wrapper_mutation."""
        import triton
        import triton.language as tl

        # [FUNCTION DOC] HELPER: triton kernel adding two input vectors elementwise into an
        # output vector, one block per program id, with masking for the tail beyond the element
        # count.
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: function under compile; doubles x, launches the add kernel
            # on x and y into a zero tensor with block size 16, and returns that output plus the
            # doubled value.
            z = x * 2
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output + z

        def add_control_deps(graph):
            # [FUNCTION DOC] HELPER: post-grad pass. Requires at least one functional triton
            # kernel wrapper node and one mul node (raises AssertionError otherwise) and makes
            # the first kernel depend on the first mul via preserve_node_ordering. Returns the
            # graph.
            from torch.utils._ordered_set import OrderedSet

            # Find triton_kernel_wrapper_mutation nodes
            triton_nodes = graph.find_nodes(
                op="call_function",
                target=torch.ops.higher_order.triton_kernel_wrapper_functional,
            )
            if not triton_nodes:
                raise AssertionError("Expected triton_kernel_wrapper_functional nodes")
            # Find mul node (z = x * 2) to use as dependency
            mul_nodes = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mul.Tensor
            )
            if not mul_nodes:
                raise AssertionError("Expected mul.Tensor nodes")
            deps_map = {triton_nodes[0]: OrderedSet([mul_nodes[0]])}
            torch._inductor.fx_passes.control_dependencies.preserve_node_ordering(
                graph, deps_map
            )
            return graph

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=add_control_deps,
        ):
            x = torch.rand([256], device=GPU_TYPE)
            y = torch.rand([256], device=GPU_TYPE)

            compiled_fn = torch.compile(fn)
            result = compiled_fn(x, y)

            expected = fn(x, y)
            torch.testing.assert_close(result, expected)

    def test_wait_event_threads_record_event_passthroughs(self):
        # [FUNCTION DOC] Builds an FX graph by hand: a reload add tagged as backward on side
        # stream 19, a record_event, other compute on stream 0, a wait_event, and a consumer
        # using the reload and the other compute. After wrap_all_sync_nodes_with_control_deps,
        # checks the consumer's first arg is an item extracted from a control_deps node whose
        # subgraph contains the wait_event call and not the record_event call. Covers wait_event
        # threading record_event passthroughs. No GPU needed.
        """wait_event must thread record_event's passthroughs to consumers.

        Regression test for the backward reload pattern:
          reload [side stream] -> record_event -> ... -> wait_event [default] -> consumer

        Without threading, the consumer depends only on record_event's getitem
        (wrong stream), so the inductor scheduler can place the consumer kernel
        before wait_event fires.
        """
        import operator

        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        g = gm.graph
        val = torch.randn(4)

        x = g.placeholder("x")
        x.meta["val"] = val

        # Simulated reloaded activation on side stream 19
        reload = g.call_function(torch.ops.aten.add.Tensor, args=(x, x))
        reload.meta.update(
            {
                "val": val,
                "partitioner_tag": "is_backward",
                "custom": {"stream": 19},
            }
        )

        # record_event on side stream after reload completes
        record = g.call_function(torch.ops.streams.record_event.default, args=(42, 19))
        record.meta["partitioner_tag"] = "must_be_in_backward"

        # Other backward compute on default stream (between record and wait)
        other = g.call_function(torch.ops.aten.mul.Tensor, args=(x, x))
        other.meta.update(
            {
                "val": val,
                "partitioner_tag": "is_backward",
                "custom": {"stream": 0},
            }
        )

        # wait_event on default stream
        wait = g.call_function(torch.ops.streams.wait_event.default, args=(42, 0))
        wait.meta["partitioner_tag"] = "must_be_in_backward"

        # Consumer reads reload result + other compute on default stream
        consumer = g.call_function(torch.ops.aten.add.Tensor, args=(reload, other))
        consumer.meta.update(
            {
                "val": val,
                "partitioner_tag": "is_backward",
                "custom": {"stream": 0},
            }
        )

        g.output((consumer,))
        gm.recompile()

        wrap_all_sync_nodes_with_control_deps(gm)

        # The consumer's reload arg must flow through wait_event's control_deps,
        # NOT record_event's.
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        final_consumer = output_node.args[0][0]
        reload_arg = final_consumer.args[0]

        self.assertEqual(reload_arg.target, operator.getitem)
        ctrl_node = reload_arg.args[0]
        self.assertIs(ctrl_node.target, control_deps)

        # The control_deps wraps wait_event, not record_event
        subgraph = getattr(gm, ctrl_node.args[1].target)
        subgraph_targets = {
            n.target for n in subgraph.graph.nodes if n.op == "call_function"
        }
        self.assertIn(torch.ops.streams.wait_event.default, subgraph_targets)
        self.assertNotIn(torch.ops.streams.record_event.default, subgraph_targets)

    @requires_gpu()
    def test_control_deps_orders_void_op_across_nested_calls(self):
        # [FUNCTION DOC] Regression test: compiles a function with two streams and an event
        # record/wait, capturing void (no layout) op names and the additional buffer deps via a
        # pre-fusion custom pass. Asserts at least one compile, at least one void op, and that
        # some void op name appears among the referenced additional deps. Requires a GPU.
        """record_event's void op must be named as an additional_buffer_dep
        of the subsequent wait_event's operations after Inductor lowering.

        record_event lowers to a NoneLayout (void) op and the overall
        control_deps call around it returns a tuple/None.  When a later
        control_deps call (around wait_event) lists the record's control_deps
        node as an additional dep, the lowered value fails the
        ``isinstance(dep, IRNode)`` check.  Before the fix, the void op was
        silently dropped and never referenced in the wait's
        additional_buffer_deps, so Inductor's cudagraph partitioning and
        other consumers of additional_buffer_deps could reorder the wait
        before the record.
        """
        from torch._inductor import ir
        from torch._inductor.virtualized import V

        def fn(x):
            # [FUNCTION DOC] HELPER: function under compile; computes on stream one then records
            # an event, then on stream two computes, waits on the event and multiplies results.
            s1 = torch.Stream(device=GPU_TYPE)
            s2 = torch.Stream(device=GPU_TYPE)
            e = torch.Event()
            with s1:
                y = x + 1
                e.record()
            with s2:
                a = x * 3
                e.wait()
                z = y * a
            return z

        captured: list[dict] = []

        def capture(nodes):
            # [FUNCTION DOC] HELPER: pre-fusion pass. Appends to the enclosing captured list the
            # names of graph operations that are buffers with no layout, and all names referenced
            # in additional buffer deps. Returns the nodes unchanged.
            void_names = {
                op.get_name()
                for op in V.graph.operations
                if isinstance(op, ir.Buffer) and isinstance(op.layout, ir.NoneLayout)
            }
            referenced: set[str] = set()
            for deps in V.graph.additional_buffer_deps.values():
                referenced.update(deps)
            captured.append({"void_names": void_names, "referenced": referenced})
            return nodes

        torch._dynamo.reset()
        with config.patch(_pre_fusion_custom_pass=capture):
            x = torch.ones(2, 2, device=GPU_TYPE)
            torch.compile(fn)(x)

        self.assertTrue(captured, "expected at least one Inductor compile")

        void_names: set[str] = set()
        referenced: set[str] = set()
        for state in captured:
            void_names |= state["void_names"]
            referenced |= state["referenced"]

        self.assertGreater(
            len(void_names),
            0,
            "expected record_event/wait_event to lower to NoneLayout ops",
        )
        self.assertTrue(
            void_names & referenced,
            lambda msg: f"{msg}\n"
            + (
                "no record_event void op appears as an additional_buffer_dep; "
                f"void_names={void_names}, referenced={referenced}"
            ),
        )

    def test_reinplace_not_blocked_by_control_deps_ordering_dep(self):
        # [FUNCTION DOC] Checks the reinplace helper that decides whether a use is ordering-only:
        # true when the value is only in the first arg group of a control_deps call, false when
        # it is also passed through or only passed through, and false for a non-control_deps
        # node. Graph built by hand; no execution.
        """Views in control_deps' ordering-only deps should not block reinplacing.

        When a tensor appears only in control_deps' additional_deps (args[0])
        and NOT in the pass-through args (args[2:]), it is an ordering-only
        dependency.  The reinplace pass must not treat this as a real data use.
        """
        from torch._inductor.fx_passes.control_dependencies import control_deps
        from torch._inductor.fx_passes.reinplace import (
            _is_control_deps_ordering_only_use,
        )

        g = torch.fx.Graph()
        view = g.placeholder("view")
        other = g.placeholder("other")
        subgraph = g.placeholder("subgraph")

        # view only in ordering deps (args[0]) -> ordering only, not a real use
        ctrl = g.call_function(control_deps, args=((view,), subgraph, other))
        self.assertTrue(_is_control_deps_ordering_only_use(ctrl, view))

        # view in both ordering deps AND pass-through -> real data use
        ctrl2 = g.call_function(control_deps, args=((view,), subgraph, view))
        self.assertFalse(_is_control_deps_ordering_only_use(ctrl2, view))

        # view only in pass-through, not in ordering deps -> real data use
        ctrl3 = g.call_function(control_deps, args=((other,), subgraph, view))
        self.assertFalse(_is_control_deps_ordering_only_use(ctrl3, view))

        # non-control_deps node -> not ordering only
        add = g.call_function(torch.ops.aten.add.Tensor, args=(view, other))
        self.assertFalse(_is_control_deps_ordering_only_use(add, view))

    @requires_gpu()
    def test_control_deps_passthrough_creates_ordering_barrier(self):
        # [FUNCTION DOC] Checks that pass-through values in control_deps lead to OrderingBarrier
        # IR nodes: at least one exists, each has the ordering-only flag set, and their source
        # buffer names are not in the graph's mutated buffers. Compiles a stream and event
        # function, results match eager. Requires a GPU.
        """Pass-through values in control_deps create OrderingBarrier nodes.

        Verifies that OrderingBarrier:
        - is created for pass-through buffers
        - has ordering_only=True (so scheduler uses WeakDep, not StarDep)
        - does not add pass-through buffers to mutated_buffers
        """
        from torch._inductor import ir
        from torch._inductor.virtualized import V

        captured: list[dict] = []

        def capture(nodes):
            # [FUNCTION DOC] HELPER: pre-fusion pass. Collects OrderingBarrier operations and
            # appends to the enclosing list a record with the barriers, their first mutation
            # names, their ordering-only flags and a set of the graph's mutated buffers. Returns
            # the nodes unchanged.
            barriers = [
                op for op in V.graph.operations if isinstance(op, ir.OrderingBarrier)
            ]
            captured.append(
                {
                    "barriers": barriers,
                    "barrier_source_names": [b.mutation_names[0] for b in barriers],
                    "ordering_only_flags": [b.ordering_only for b in barriers],
                    "mutated_buffers": set(V.graph.mutated_buffers),
                }
            )
            return nodes

        def fn(x):
            # [FUNCTION DOC] HELPER: function under compile; adds one on a side stream with an
            # event record, waits for the event on the current stream, then doubles the result.
            s = torch.Stream(device=GPU_TYPE)
            e = torch.Event()
            with s:
                y = x + 1
                e.record()
            e.wait()
            return y * 2

        torch._dynamo.reset()
        with config.patch(_pre_fusion_custom_pass=capture):
            x = torch.ones(4, device=GPU_TYPE)
            result = torch.compile(fn)(x)

        expected = fn(torch.ones(4, device=GPU_TYPE))
        torch.testing.assert_close(result, expected)

        self.assertTrue(captured, "expected at least one Inductor compile")
        for c in captured:
            self.assertGreater(len(c["barriers"]), 0, "expected OrderingBarrier nodes")
            self.assertTrue(
                all(c["ordering_only_flags"]),
                "all OrderingBarrier nodes must have ordering_only=True",
            )
            for source_name in c["barrier_source_names"]:
                self.assertNotIn(
                    source_name,
                    c["mutated_buffers"],
                    f"barrier source {source_name} should not be in mutated_buffers",
                )

    @requires_gpu()
    def test_ordering_barrier_is_nop(self):
        # [FUNCTION DOC] Checks that OrderingBarrier IR nodes are no-ops: for every captured
        # barrier, should_allocate is false and is_no_op is true, and at least one barrier
        # exists. Compiles a stream and event function, results match eager. Requires a GPU.
        """OrderingBarrier must not allocate or generate code.

        Verifies should_allocate() and is_no_op() on the actual IR node.
        """
        from torch._inductor import ir
        from torch._inductor.virtualized import V

        captured: list[dict] = []

        def capture(nodes):
            # [FUNCTION DOC] HELPER: pre-fusion pass. Collects OrderingBarrier operations and
            # appends to the enclosing list the results of their allocation and no-op queries.
            # Returns the nodes unchanged.
            barriers = [
                op for op in V.graph.operations if isinstance(op, ir.OrderingBarrier)
            ]
            captured.append(
                {
                    "barriers_allocate": [b.should_allocate() for b in barriers],
                    "barriers_nop": [b.is_no_op() for b in barriers],
                }
            )
            return nodes

        def fn(x):
            # [FUNCTION DOC] HELPER: function under compile; adds one on a side stream with an
            # event record, waits for the event on the current stream, then doubles the result.
            s = torch.Stream(device=GPU_TYPE)
            e = torch.Event()
            with s:
                y = x + 1
                e.record()
            e.wait()
            return y * 2

        torch._dynamo.reset()
        with config.patch(_pre_fusion_custom_pass=capture):
            x = torch.ones(4, device=GPU_TYPE)
            result = torch.compile(fn)(x)

        expected = fn(torch.ones(4, device=GPU_TYPE))
        torch.testing.assert_close(result, expected)

        self.assertTrue(captured, "expected at least one compile")
        for c in captured:
            self.assertGreater(
                len(c["barriers_allocate"]),
                0,
                "expected OrderingBarrier nodes (precondition)",
            )
            self.assertTrue(
                all(not a for a in c["barriers_allocate"]),
                "OrderingBarrier should_allocate() must be False",
            )
            self.assertTrue(
                all(c["barriers_nop"]),
                "OrderingBarrier is_no_op() must be True",
            )

    @requires_cuda_triton
    def test_bidirectional_stream_sync_correctness(self):
        # [FUNCTION DOC] Regression test for bidirectional stream synchronization using real CUDA
        # streams and events: two streams exchange events while computing multiplies and an add;
        # the compiled result must equal eager. Guards against ordering dependencies being keyed
        # by buffer name instead of operation name, which previously allowed wrong scheduling.
        # Needs CUDA and triton.
        """Regression: passthrough OrderingBarrier must be ordered after all subgraph ops.

        Bidirectional stream sync exposes a bug where additional_buffer_deps was
        keyed by barrier.get_name() (buffer name) instead of
        barrier.get_operation_name() (operation name), so compute_dependencies
        silently lost the ordering dep and the scheduler could place barrier2
        before record_event1, producing wrong results.
        """
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            # [FUNCTION DOC] HELPER: function under compile; stream one doubles x and records an
            # event, stream two waits, adds one and records a second event, stream one waits and
            # doubles again; both streams are synchronized before returning.
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            event_s1 = torch.cuda.Event()
            event_s2 = torch.cuda.Event()
            with torch.cuda.stream(s1):
                a = x * 2
                event_s1.record(s1)
            with torch.cuda.stream(s2):
                event_s1.wait(s2)
                b = a + 1
                event_s2.record(s2)
            with torch.cuda.stream(s1):
                event_s2.wait(s1)
                c = b * 2
            s1.synchronize()
            s2.synchronize()
            return c

        x = torch.randn(1024, device="cuda")
        expected = fn(x)
        result, _ = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU_AND_TRITON:
        run_tests(needs="filelock")
