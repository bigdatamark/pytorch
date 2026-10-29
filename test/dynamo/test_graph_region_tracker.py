# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the graph region tracker, which finds identical regions (repeated
# sequences of nodes) in a traced graph and tracks which node arguments are mutated. Tests
# compare the string form of the identical region groups, mutation tracking map and duplicate
# map with expected strings. Import time has no side effects.
import contextlib

import torch
import torch.fx
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import extract_graph_and_tracker
from torch.testing._internal.common_utils import recover_orig_fp32_precision
from torch.utils._pytree import tree_map


class GraphRegionTrackerTests(TestCase):
    # [CLASS DOC] Tests for region tracking. setUp turns on the config for tracking nodes for
    # deduplication through an ExitStack and tearDown restores it.
    def setUp(self):
        # [FUNCTION DOC] Run base setup, then create an ExitStack and enter a config patch
        # that enables tracking nodes for deduplication.
        super().setUp()
        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(
            torch._dynamo.config.patch("track_nodes_for_deduplication", True)
        )

    def tearDown(self):
        # [FUNCTION DOC] Close the ExitStack (restoring the config) and then run
        # base teardown.
        self.exit_stack.close()
        super().tearDown()

    def get_result(self, fn, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Trace the function, ask the tracker for identical regions in
        # the graph, convert nodes to names and return the string form of the region groups.
        graph, region_tracker = extract_graph_and_tracker(fn, *args, **kwargs)
        region_groups = region_tracker.get_identical_regions(graph)
        region_groups = tree_map(lambda n: n.name, region_groups)
        return str(region_groups)

    def get_mutation_tracking(self, fn, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Trace the function and return the string form of the
        # tracker's map from node to mutated argument positions.
        _, region_tracker = extract_graph_and_tracker(fn, *args, **kwargs)
        return str(region_tracker.node_to_mutated_arg_positions)

    def test_get_regions_single_region_group(self):
        # [FUNCTION DOC] A repeated inner function called three times yields one group of
        # three regions of five nodes each, checked against an expected string.
        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Adds 1 to x and 2 to y and returns
            # the sum of their reductions.
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls the inner function with
            # different second arguments (including sine of y), multiplies and adds results.
            o0 = inner_fn(x, y)
            o1 = torch.sin(y)
            o2 = inner_fn(x, o1)
            o3 = inner_fn(x, y)
            o4 = o3 * o3
            return o2 * o4 + o0

        self.assertExpectedInline(
            self.get_result(
                fn,
                torch.rand(10, 10),
                torch.ones(10, 20),
            ),
            """[[['add', 'add_3', 'sum_1', 'sum_4', 'add_6'], ['add_1', 'add_5', 'sum_2', 'sum_6', 'add_8'], ['add_2', 'add_4', 'sum_3', 'sum_5', 'add_7']]]""",
        )

    def test_get_regions_multiple_region_groups(self):
        # [FUNCTION DOC] Two different repeated functions produce two separate region groups;
        # checks the expected region string.
        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Adds 1 to x and 2 to y and returns
            # the sum of their reductions.
            x1 = x + 1
            y1 = y + 2
            z = x1.sum() + y1.sum()
            return z

        def inner_fn2(a, b):
            # [FUNCTION DOC] HELPER: Function under test. Adds 2 and 3 in place to both inputs
            # and multiplies a by the sum of cosine of b.
            a += 2
            b += 3
            c = a * b.cos().sum()
            return c

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls the two inner functions
            # several times with sine and cosine inputs and combines results.
            x0 = torch.cos(x)
            y0 = torch.sin(y)
            o1 = inner_fn2(x0, y0)
            o0 = inner_fn(x, y)
            o1 = torch.sin(o0)
            o2 = inner_fn(x, y0)
            o2 = inner_fn2(x0, y0)
            o3 = inner_fn(x, y)
            return o1 * o2 + o3

        self.assertExpectedInline(
            self.get_result(
                fn,
                torch.rand(10, 10),
                torch.ones(10, 20),
            ),
            """[[['iadd', 'iadd_1', 'cos_1', 'sum_5', 'mul'], ['iadd_2', 'iadd_3', 'cos_2', 'sum_8', 'mul_1']], [['add', 'add_2', 'sum_1', 'sum_3', 'add_4'], ['add_1', 'add_3', 'sum_2', 'sum_4', 'add_5'], ['add_6', 'add_7', 'sum_6', 'sum_7', 'add_8']]]""",
        )

    def test_no_single_node_regions(self):
        # [FUNCTION DOC] Regions consisting of a single node are not reported; the expected
        # result is an empty list.
        def inner_fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns x plus one.
            return x + 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Calls the inner function three times
            # and adds the results.
            o0 = inner_fn(x)
            o1 = inner_fn(x)
            o2 = inner_fn(x)
            return o0 + o1 + o2

        self.assertExpectedInline(self.get_result(fn, torch.ones(10, 10)), """[]""")

    def test_mismatched_arg_shapes(self):
        # [FUNCTION DOC] Calls whose arguments have mismatching shapes yield shorter regions;
        # checks the expected groups.
        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Adds 1 to x and 2 to y and returns
            # the sum of their reductions.
            x1 = x + 1
            y1 = y + 2
            z = x1.sum() + y1.sum()
            return z

        def inner_fn2(a, b):
            # [FUNCTION DOC] HELPER: Function under test. Adds 2 and 3 in place to both inputs
            # and multiplies a by the sum of cosine of b.
            a += 2
            b += 3
            c = a * b.cos().sum()
            return c

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls the inner functions with
            # differently shaped arguments and combines the results.
            x0 = torch.cos(x)
            y0 = torch.sin(y)
            o1 = inner_fn2(x0, y0)
            o0 = inner_fn(x, o1)
            o1 = torch.sin(o0)
            o2 = inner_fn(x, y0)
            o2 = inner_fn2(o2, y0)
            o3 = inner_fn(x, y)
            return o1 * o2 + o3

        self.assertExpectedInline(
            self.get_result(
                fn,
                torch.rand(10, 10),
                torch.ones(10, 20),
            ),
            """[[['iadd_1', 'cos_1', 'sum_4'], ['iadd_3', 'cos_2', 'sum_8']], [['add', 'sum_1', 'add_5'], ['add_1', 'sum_2', 'add_3'], ['add_6', 'sum_6', 'add_8']], [['add_2', 'sum_3'], ['add_7', 'sum_7']]]""",
        )

    def test_mismatched_dtypes(self):
        # [FUNCTION DOC] Calls with different dtypes (one with bfloat16 inputs) are left out
        # of the groups; four float calls remain in one group.
        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Multiplies x by one, adds one to y
            # and returns x plus the sum of y.
            x1 = x * 1
            y1 = y + 1
            return x1 + y1.sum()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls the inner function four times
            # with floats and once with bfloat16 conversions, then combines results.
            x0 = torch.sin(x)
            y0 = torch.cos(y)
            o0 = inner_fn(x0, y0)
            o2 = inner_fn(x0, y0)
            o4 = inner_fn(x0, y0)
            o5 = inner_fn(x0, y0)
            o1 = inner_fn(x0.to(torch.bfloat16), y0.to(torch.bfloat16))
            o3 = o1 + o2
            return o3 * o0 + o4 + o5

        self.assertExpectedInline(
            self.get_result(
                fn,
                torch.rand(10, 10),
                torch.ones(10, 20),
            ),
            """[[['add', 'mul', 'sum_1', 'add_4'], ['add_1', 'mul_1', 'sum_2', 'add_5'], ['add_2', 'mul_2', 'sum_3', 'add_6'], ['add_3', 'mul_3', 'sum_4', 'add_7']]]""",
        )

    def test_nested_args(self):
        # [FUNCTION DOC] Arguments given as lists of tensors are handled; the float calls form
        # one group while the bfloat16 call is excluded.
        def inner_fn(xs, ys):
            # [FUNCTION DOC] HELPER: Function under test. Adds two lists of tensors
            # elementwise with the foreach add and returns the first output plus the sum of
            # the second.
            out = torch._foreach_add(xs, ys)
            return out[0] + out[1].sum()

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Function under test. Calls the inner function four times
            # with float lists and once with bfloat16 conversions, then combines results.
            x0 = torch.sin(x)
            y0 = torch.cos(y)
            z0 = torch.sin(z)
            o0 = inner_fn([x0, z0], [x0, y0])
            o2 = inner_fn([x0, z0], [x0, y0])
            o4 = inner_fn([x0, z0], [x0, y0])
            o5 = inner_fn([x0, z0], [x0, y0])
            o1 = inner_fn(
                [x0.to(torch.bfloat16), z0.to(torch.bfloat16)],
                [x0.to(torch.bfloat16), y0.to(torch.bfloat16)],
            )
            o3 = o1 + o2
            return o3 * o0 + o4 + o5

        self.assertExpectedInline(
            self.get_result(
                fn,
                torch.rand(10, 10),
                torch.rand(10, 20),
                torch.ones(10, 20),
            ),
            """[[['_foreach_add', 'getitem', 'getitem_1', 'sum_1', 'add'], ['_foreach_add_1', 'getitem_2', 'getitem_3', 'sum_2', 'add_1'], ['_foreach_add_2', 'getitem_4', 'getitem_5', 'sum_3', 'add_2'], ['_foreach_add_3', 'getitem_6', 'getitem_7', 'sum_4', 'add_3']]]""",
        )

    @recover_orig_fp32_precision
    def test_mismatched_global_state(self):
        # [FUNCTION DOC] Calls made under different global state (grad mode, inference mode,
        # saved tensor hooks, default dtype, deterministic algorithms, matmul precision flags)
        # are put in separate groups. Loops over several context managers or toggle function
        # pairs and checks the same expected groups each time. The fp32 precision flags are
        # restored afterwards.
        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Multiplies x by one, adds one to y
            # and returns x plus the sum of y.
            x1 = x * 1
            y1 = y + 1
            return x1 + y1.sum()

        def fn(x, y, c):
            # [FUNCTION DOC] HELPER: Function under test. Calls the inner function twice, then
            # changes global state (a pair of functions in a tuple or a context manager),
            # calls it twice more and sums the four results.
            x0 = torch.sin(x)
            y0 = torch.cos(y)
            o4 = inner_fn(x0, y0)
            o5 = inner_fn(x0, y0)
            if isinstance(c, tuple):
                c[0]()
                o0 = inner_fn(x0, y0)
                o2 = inner_fn(x0, y0)
                c[1]()
            else:
                with c():
                    o0 = inner_fn(x0, y0)
                    o2 = inner_fn(x0, y0)
            return o0 + o2 + o4 + o5

        def create_toggle_fns(property):
            # [FUNCTION DOC] HELPER: Return two functions for the matmul backend property of
            # the given name: one sets it to the opposite of its value at creation, one
            # restores the original.
            old_value = getattr(torch.backends.cuda.matmul, property)

            def toggle_property():
                # [FUNCTION DOC] HELPER: Set the property to the opposite of the
                # original value.
                setattr(torch.backends.cuda.matmul, property, not old_value)

            def reset_property():
                # [FUNCTION DOC] HELPER: Set the property back to the original value.
                setattr(torch.backends.cuda.matmul, property, old_value)

            return toggle_property, reset_property

        old_dtype = torch.get_default_dtype()

        def set_default_dtype_bfloat16():
            # [FUNCTION DOC] HELPER: Set the default dtype to bfloat16.
            torch.set_default_dtype(torch.bfloat16)

        def reset_default_dtype():
            # [FUNCTION DOC] HELPER: Restore the default dtype saved when the test started.
            torch.set_default_dtype(old_dtype)

        for ctx in [
            lambda: torch.set_grad_enabled(False),
            torch.autograd.grad_mode.inference_mode,
            lambda: torch.autograd.graph.disable_saved_tensors_hooks(
                "This is not supported"
            ),
            # lambda: torch.set_num_threads(2), : Unsupported
            (set_default_dtype_bfloat16, reset_default_dtype),
            (
                lambda: torch.use_deterministic_algorithms(True),
                lambda: torch.use_deterministic_algorithms(False),
            ),
            # (lambda: torch.use_deterministic_algorithms(True, warn_only=True),
            # lambda: torch.use_deterministic_algorithms(False)), : Unsupported
            create_toggle_fns("allow_bf16_reduced_precision_reduction"),
            create_toggle_fns("allow_fp16_reduced_precision_reduction"),
            create_toggle_fns("allow_tf32"),
        ]:
            self.assertExpectedInline(
                self.get_result(fn, torch.rand(10, 10), torch.ones(10, 20), ctx),
                """[[['add_4', 'mul_2', 'sum_3', 'add_6'], ['add_5', 'mul_3', 'sum_4', 'add_7']], [['add', 'mul', 'sum_1', 'add_2'], ['add_1', 'mul_1', 'sum_2', 'add_3']]]""",
            )

    def test_mutation_tracking_simple(self):
        # [FUNCTION DOC] In-place operations on inputs and intermediates are recorded in the
        # mutation tracking map with the mutated argument position zero.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Function under test. Applies sin and cos, mutates z
            # with an in-place sine and y0 with an in-place add of z, returning the sum
            # of reductions.
            x0 = torch.sin(x)
            y0 = torch.cos(y)
            z.sin_()
            y0.add_(z)
            return x0.sum() + y0.sum()

        self.assertExpectedInline(
            self.get_mutation_tracking(
                fn,
                torch.rand(10, 10),
                torch.rand(10, 20),
                torch.ones(10, 20),
            ),
            """{sin_: OrderedSet([0]), add_: OrderedSet([0])}""",
        )

    def test_mutation_tracking_setitem(self):
        # [FUNCTION DOC] An item assignment is recorded as mutating argument position zero.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds 1 to x, assigns 3 to element 0
            # of the result and returns it.
            y = x + 1
            y[0] = 3
            return y

        self.assertExpectedInline(
            self.get_mutation_tracking(fn, torch.rand(10, 10)),
            """{setitem: OrderedSet([0])}""",
        )

    def test_mutation_tracking_allow_in_graph(self):
        # [FUNCTION DOC] A mutating allow-in-graph function and an in-place sine are both
        # recorded in the mutation tracking map.
        @torch._dynamo.allow_in_graph
        def fn_mut(x, y):
            # [FUNCTION DOC] HELPER: Allow-in-graph function adding y into x in place and
            # returning the sum of reductions.
            x.add_(y)
            return x.sum() + y.sum()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls the mutating function on the
            # sum of x and y, applies in-place sine and returns x plus the call result.
            z = x + y
            o0 = fn_mut(z, y)
            z.sin_()
            return x + o0

        self.assertExpectedInline(
            self.get_mutation_tracking(
                fn,
                torch.rand(20, 10),
                torch.rand(20, 10),
            ),
            """{fn_mut: OrderedSet([0]), sin_: OrderedSet([0])}""",
        )

    def test_non_tensor_arg_hashing(self):
        # [FUNCTION DOC] Calls whose non-tensor arguments (convolution stride tuples) differ
        # are not grouped together; the matching calls form one group.
        def inner(x, w, t):
            # [FUNCTION DOC] HELPER: Function under test. Doubles x via addition and applies a
            # convolution with the given weight and parameters.
            y = x + x
            return torch.conv2d(y, w, None, *t)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls the inner function three times
            # with one set of convolution parameters and once with another, returning the sum
            # of reductions.
            o1 = inner(x, y, ((1, 1), (0, 0), (1, 1), 1))
            o2 = inner(x, y, ((1, 1), (0, 0), (1, 1), 1))
            o3 = inner(x, y, ((1, 1), (0, 0), (1, 1), 1))
            o4 = inner(x, y, ((2, 2), (0, 0), (1, 1), 1))
            return o1.sum() + o2.sum() + o3.sum() + o4.sum()

        self.assertExpectedInline(
            self.get_result(
                fn,
                torch.rand(32, 256, 56, 56),
                torch.nn.Parameter(torch.rand(512, 256, 1, 1)),
            ),
            """[[['add', 'conv2d'], ['add_1', 'conv2d_1'], ['add_2', 'conv2d_2']]]""",
        )

    def test_region_sorting(self):
        # [FUNCTION DOC] Sorts regions by the order of a reference region and checks the
        # result: every region is permuted consistently with the rank mapping.
        from torch._dynamo.graph_region_tracker import _sort_with_ref_region

        index_to_rank = {0: 0, 2: 1, 1: 2}
        regions = [[0, 1, 2], [1, 2, 0]]
        _sort_with_ref_region(index_to_rank, regions)
        self.assertExpectedInline(regions, """[[0, 2, 1], [1, 0, 2]]""")

    def test_no_duplicate_tracking(self):
        # [FUNCTION DOC] The tracker's duplicate map is checked against an expected string
        # after tracing, then tracking an already tracked node again is done to check it does
        # not fail or add the node twice.
        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Adds 1 to x and 2 to y and returns
            # the sum of their reductions.
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls the inner function with
            # different second arguments, multiplies and adds results.
            o0 = inner_fn(x, y)
            o1 = torch.sin(y)
            o2 = inner_fn(x, o1)
            o3 = inner_fn(x, y)
            o4 = o3 * o3
            return o2 * o4 + o0

        graph, tracker = extract_graph_and_tracker(
            fn, torch.rand(10, 10), torch.ones(10, 20)
        )
        self.assertExpectedInline(
            tracker.node_to_duplicates,
            """{l_x_: [l_x_], add: [add, add_1, add_2], l_y_: [l_y_], add_3: [add_3, add_5, add_4], sum_1: [sum_1, sum_2, sum_3], sum_4: [sum_4, sum_6, sum_5], add_6: [add_6, add_8, add_7], sin: [sin], add_1: [add, add_1, add_2], add_5: [add_3, add_5, add_4], sum_2: [sum_1, sum_2, sum_3], sum_6: [sum_4, sum_6, sum_5], add_8: [add_6, add_8, add_7], add_2: [add, add_1, add_2], add_4: [add_3, add_5, add_4], sum_3: [sum_1, sum_2, sum_3], sum_5: [sum_4, sum_6, sum_5], add_7: [add_6, add_8, add_7], mul: [mul], mul_1: [mul_1], add_9: [add_9]}""",
        )
        key = next(iter(tracker.node_to_duplicates.keys()))
        tracker.track_node(None, key)  # this will fail if the node is added again


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
