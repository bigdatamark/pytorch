# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the automatic dynamic exclusion guard feature of Dynamo.
# When a dimension or integer input becomes dynamic after a recompile, the new graph
# gets an exclusion guard that rejects the previously seen static values, so inputs
# fall back to the older, more specialized graph. Contains a graph-tracking backend
# and one TestCase. No import-time side effects other than imports.
import torch
import torch._dynamo
import torch._dynamo.testing
from torch._dynamo.test_case import run_tests, TestCase


class GraphTracker:
    # [CLASS DOC] Compiler backend callable that wraps each compiled graph and records which
    # graph (numbered by compilation order) ran on each call.
    # State: graphs holds every GraphModule received; call_log holds the graph id per call.
    """Backend that tracks which compiled graph (by compilation order) handles each call."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes empty lists for graphs and call_log.
        self.graphs = []
        self.call_log = []

    def __call__(self, gm, example_inputs):
        # [FUNCTION DOC] Backend entry point. Assigns the next graph id, stores the GraphModule,
        # and returns a wrapper that appends the id to call_log and then runs the graph forward
        # with the same arguments. The example inputs are ignored.
        graph_id = len(self.graphs)
        self.graphs.append(gm)

        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Records the captured graph id in call_log, then returns the
            # result of running the graph forward on the given positional and keyword arguments.
            self.call_log.append(graph_id)
            return gm.forward(*args, **kwargs)

        return wrapper

    @property
    def frame_count(self):
        # [FUNCTION DOC] Property returning the number of graphs compiled so far (the length of
        # the graphs list).
        return len(self.graphs)

    def reset(self):
        # [FUNCTION DOC] Clears both graphs and call_log in place.
        self.graphs.clear()
        self.call_log.clear()


@torch._dynamo.config.patch(automatic_dynamic_exclusion_guard=True)
class TestGuardExclusion(TestCase):
    # [CLASS DOC] Tests that exclusion guards generated under automatic dynamic shapes
    # route inputs to the most specialized compiled graph. The class decorator enables
    # the automatic_dynamic_exclusion_guard config for every test. Each test compiles with
    # a GraphTracker backend and checks frame_count and the last call_log entry.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp and resets Dynamo state so each test starts
        # with an empty cache.
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs the base tearDown and resets Dynamo state.
        super().tearDown()
        torch._dynamo.reset()

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_automatic_dynamic_exclusive_guard_basic(self):
        # [FUNCTION DOC] Static graph for shape 3x4, then dim 0 becomes dynamic at 5x4; checks
        # size 7 reuses the dynamic graph without recompiling and that size 3 goes back to the
        # static graph. Also checks all three outputs equal the eager doubling of the inputs.
        """
        1. [3, 4] -> Graph 0 (static)
        2. [5, 4] -> Graph 1 (dim 0 dynamic), exclusion rejects dim0==3
        3. [7, 4] -> Graph 1 (reuse dynamic graph)
        4. [3, 4] -> Graph 0 (exclusion triggers, reverts to static)
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Function under test; returns the input doubled.
            return x * 2

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        x1 = torch.randn(3, 4)
        result1 = opt(x1)
        self.assertEqual(tracker.frame_count, 1)
        self.assertEqual(tracker.call_log[-1], 0)

        x2 = torch.randn(5, 4)
        result2 = opt(x2)
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[-1], 1)

        # dynamic graph reuse
        opt(torch.randn(7, 4))
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[-1], 1)

        # original shape reverts to Graph 0
        x3 = torch.randn(3, 4)
        result3 = opt(x3)
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[-1], 0)

        self.assertEqual(result1, x1 * 2)
        self.assertEqual(result2, x2 * 2)
        self.assertEqual(result3, x3 * 2)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_accumulated_exclusion_does_not_shadow_intermediate_graph(self):
        # [FUNCTION DOC] Shapes 3x4, 5x4, 3x19 produce three graphs. Verifies that the exclusion
        # of the last graph rejects only dim 1 equal to 4 (independently of dim 0), so 5x4
        # routes to graph 1 and 3x4 still routes to graph 0.
        """
        Tensor accumulation: dims become dynamic one at a time.
        1. func(3, 4)  -> Graph 0: static (3, 4)
        2. func(5, 4)  -> Graph 1: (s0, 4), excluded dim0=3
        3. func(3, 19) -> Graph 2: (s0, s1), excluded dim1=4
                          (dim0's exclusion is cleared since no dim transitioned)
        4. func(5, 4)  -> should use Graph 1, not Graph 2
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Function under test; returns the input doubled.
            return x * 2

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        # Call 1: shape [3, 4] -> compiles Graph 0 (static)
        opt(torch.randn(3, 4))
        self.assertEqual(tracker.frame_count, 1)
        self.assertEqual(tracker.call_log[-1], 0)

        # Call 2: shape [5, 4] -> compiles Graph 1 (s0, 4)
        opt(torch.randn(5, 4))
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[-1], 1)

        # Call 3: shape [3, 19] -> Graph 1 exclusion rejects (size(0)==3),
        # Graph 0 rejects (19!=4), recompiles Graph 2 (s0, s1)
        opt(torch.randn(3, 19))
        self.assertEqual(tracker.frame_count, 3)
        self.assertEqual(tracker.call_log[-1], 2)

        # Call 4: shape [5, 4] -> should still use Graph 1 (s0, 4).
        # Graph 2's exclusion is dim1=4, so (5, 4) is rejected and
        # falls through to Graph 1.
        opt(torch.randn(5, 4))

        self.assertEqual(
            tracker.call_log[-1],
            1,
            "Input [5,4] should use Graph 1 (s0, 4), not Graph 2 (s0, s1). "
            "Graph 2's exclusion must reject size(1)==4 independently, not "
            "require size(0)==3 AND size(1)==4.",
        )

        # Call 5: shape [3, 4] -> should still use Graph 0 (static)
        opt(torch.randn(3, 4))
        self.assertEqual(
            tracker.call_log[-1],
            0,
            "Input [3,4] should use Graph 0 (static)",
        )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_4d_progressive_dynamism_cascading(self):
        # [FUNCTION DOC] Four dimensional tensor whose dims become dynamic one per recompile
        # (dims 0, 1, 2), giving four graphs. Checks each probe shape routes to the most
        # specialized graph, that a fully new shape lands on graph 3, and that no extra
        # recompiles occur.
        """
        4D tensor where dims become dynamic one at a time across recompilations.
        Each new graph is more general, and exclusion guards ensure inputs cascade
        to the most specialized graph.

        Graph 0: (2, 3, 4, 5) static
        Graph 1: (dyn, 3, 4, 5) excluded dim0=2
        Graph 2: (dyn, dyn, 4, 5) excluded dim1=3
        Graph 3: (dyn, dyn, dyn, 5) excluded dim2=4
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Function under test; returns the sum of all input elements.
            return x.sum()

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        # Graph 0: (2, 3, 4, 5) static
        opt(torch.randn(2, 3, 4, 5))
        self.assertEqual(tracker.frame_count, 1)

        # Graph 1: dim0 changes -> (dyn, 3, 4, 5)
        opt(torch.randn(7, 3, 4, 5))
        self.assertEqual(tracker.frame_count, 2)

        # Graph 2: dim1 also changes -> (dyn, dyn, 4, 5)
        # Input (7, 8, 4, 5): Graph 1 rejects dim1=8≠3, Graph 0 rejects dim0=7≠2
        opt(torch.randn(7, 8, 4, 5))
        self.assertEqual(tracker.frame_count, 3)

        # Graph 3: dim2 also changes -> (dyn, dyn, dyn, 5)
        # Input (7, 8, 9, 5): Graph 2 rejects dim2=9≠4
        opt(torch.randn(7, 8, 9, 5))
        self.assertEqual(tracker.frame_count, 4)

        # Now verify cascading: each input routes to the most specialized graph.
        # (2, 3, 4, 5) -> Graph 0 (static, most specialized)
        opt(torch.randn(2, 3, 4, 5))
        self.assertEqual(tracker.call_log[-1], 0, "(2,3,4,5) -> Graph 0 (static)")

        # (7, 3, 4, 5) -> Graph 1 (dyn, 3, 4, 5)
        opt(torch.randn(7, 3, 4, 5))
        self.assertEqual(tracker.call_log[-1], 1, "(7,3,4,5) -> Graph 1 (dyn,3,4,5)")

        # (7, 8, 4, 5) -> Graph 2 (dyn, dyn, 4, 5)
        opt(torch.randn(7, 8, 4, 5))
        self.assertEqual(tracker.call_log[-1], 2, "(7,8,4,5) -> Graph 2 (dyn,dyn,4,5)")

        # (7, 8, 9, 5) -> Graph 3 (dyn, dyn, dyn, 5)
        opt(torch.randn(7, 8, 9, 5))
        self.assertEqual(
            tracker.call_log[-1], 3, "(7,8,9,5) -> Graph 3 (dyn,dyn,dyn,5)"
        )

        # (20, 30, 40, 5) -> Graph 3 (most general, no exclusion hit)
        opt(torch.randn(20, 30, 40, 5))
        self.assertEqual(
            tracker.call_log[-1], 3, "(20,30,40,5) -> Graph 3 (most general)"
        )

        self.assertEqual(tracker.frame_count, 4, "No additional recompilations")

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_5d_two_rounds_of_dynamism(self):
        # [FUNCTION DOC] Five dimensional tensor with two dynamism rounds (dim 0, then dim 2),
        # giving three graphs. Checks routing for original, dim 0 changed, and both changed
        # shapes, and that dim 0 equal to its old static value with dim 2 changed still goes to
        # graph 2 because that graph holds no exclusion on dim 0. No extra recompiles.
        """
        5D tensor with two rounds of automatic_dynamic. Verify inputs route
        to the most specialized graph after each round.

        Graph 0: (2, 3, 4, 5, 6) static
        Graph 1: (dyn, 3, 4, 5, 6) excluded dim0=2
        Graph 2: (dyn, 3, dyn, 5, 6) excluded dim2=4
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Function under test; returns the input doubled.
            return x * 2

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        # Graph 0: static
        opt(torch.randn(2, 3, 4, 5, 6))
        self.assertEqual(tracker.frame_count, 1)

        # Graph 1: dim0 becomes dynamic
        opt(torch.randn(8, 3, 4, 5, 6))
        self.assertEqual(tracker.frame_count, 2)

        # Graph 2: dim2 also becomes dynamic
        opt(torch.randn(8, 3, 10, 5, 6))
        self.assertEqual(tracker.frame_count, 3)

        # Verify routing:
        # Original static shape -> Graph 0
        opt(torch.randn(2, 3, 4, 5, 6))
        self.assertEqual(tracker.call_log[-1], 0, "Original -> Graph 0")

        # dim0 differs, dim2 matches static -> Graph 1
        opt(torch.randn(9, 3, 4, 5, 6))
        self.assertEqual(tracker.call_log[-1], 1, "dim0 changed -> Graph 1")

        # dim0 differs, dim2 differs -> Graph 2
        opt(torch.randn(9, 3, 11, 5, 6))
        self.assertEqual(tracker.call_log[-1], 2, "dim0+dim2 changed -> Graph 2")

        # dim0 is original excluded value, dim2 differs -> Graph 2 should still
        # accept because dim0's exclusion is None (already dynamic when snapshot taken)
        opt(torch.randn(2, 3, 11, 5, 6))
        self.assertEqual(
            tracker.call_log[-1],
            2,
            "dim0=2 with dim2≠4 -> Graph 2 (dim0 exclusion is None)",
        )

        self.assertEqual(tracker.frame_count, 3, "No additional recompilations")

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_many_entries_wrong_graph_selection(self):
        # [FUNCTION DOC] Builds four graphs by making dims 0, 1, 2 of a four dimensional tensor
        # dynamic in turn, then runs a table of shapes and checks each hits the expected graph,
        # so the most general graph does not shadow earlier ones. No extra recompiles.
        """
        Convoluted scenario: 4D tensor, three rounds of dynamism creating 4 graphs.
        Without exclusion guards, the most general graph would shadow all others.
        Test that each input gets the best (most specialized) match.

        Graph 0: (2, 3, 4, 5) static
        Graph 1: (dyn, 3, 4, 5) excluded dim0=2
        After Graph 1, (2, 8, 4, 5) triggers dim1 dynamic:
        Graph 2: (dyn, dyn, 4, 5) excluded dim1=3
        After Graph 2, (2, 8, 9, 5) triggers dim2 dynamic:
        Graph 3: (dyn, dyn, dyn, 5) excluded dim2=4

        Key: Graph 3 should NOT steal inputs that belong to Graph 0, 1, or 2.
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Function under test; returns relu of the input.
            return x.relu()

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        # Build up 4 graphs progressively
        opt(torch.randn(2, 3, 4, 5))  # Graph 0
        opt(torch.randn(7, 3, 4, 5))  # Graph 1: dim0 dynamic
        opt(torch.randn(7, 8, 4, 5))  # Graph 2: dim1 also dynamic
        opt(torch.randn(7, 8, 9, 5))  # Graph 3: dim2 also dynamic
        self.assertEqual(tracker.frame_count, 4)

        # Now stress-test routing with various inputs:
        test_cases = [
            # (shape, expected_graph, description)
            ((2, 3, 4, 5), 0, "exact original -> static Graph 0"),
            ((7, 3, 4, 5), 1, "dim0 differs -> Graph 1"),
            ((99, 3, 4, 5), 1, "dim0 differs (large) -> Graph 1"),
            ((7, 8, 4, 5), 2, "dim0+dim1 differ -> Graph 2"),
            ((7, 99, 4, 5), 2, "dim0+dim1 differ (large) -> Graph 2"),
            ((7, 8, 9, 5), 3, "dim0+dim1+dim2 differ -> Graph 3"),
            ((99, 99, 99, 5), 3, "all non-static dims differ -> Graph 3"),
        ]

        for shape, expected_graph, desc in test_cases:
            opt(torch.randn(*shape))
            self.assertEqual(tracker.call_log[-1], expected_graph, desc)

        self.assertEqual(tracker.frame_count, 4, "No additional recompilations")

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_multi_dim_dynamic_and_semantics(self):
        # [FUNCTION DOC] When dims 0, 2 and 3 turn dynamic together, the exclusion rejects only
        # inputs matching all excluded values at once. Checks the exact original shape goes to
        # graph 0, while partially matching shapes and a new shape stay on graph 1 with no
        # recompile.
        """
        When multiple dims become dynamic at once, AND semantics is critical.
        Graph 0: (4, 3, 234, 5) static
        Graph 1: (s0, 3, s2, s3) dynamic on dims 0,2,3. excluded=(4, _, 234, 5)

        OR semantics (wrong): rejects (4, 3, 100, 20) because dim0==4 matches.
        AND semantics (correct): accepts (4, 3, 100, 20) because not ALL excluded
        dims match (dim2=100≠234 and dim3=20≠5).
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Function under test; returns the input doubled.
            return x * 2

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        opt(torch.randn(4, 3, 234, 5))  # Graph 0: static
        opt(torch.randn(10, 3, 100, 20))  # Graph 1: dims 0,2,3 dynamic
        self.assertEqual(tracker.frame_count, 2)

        # Only the exact original shape should be excluded from Graph 1
        opt(torch.randn(4, 3, 234, 5))
        self.assertEqual(tracker.call_log[-1], 0, "Exact original -> Graph 0")

        # Partial matches should NOT be excluded (AND semantics)
        opt(torch.randn(4, 3, 100, 20))  # dim0=4 matches, dims 2,3 don't
        self.assertEqual(tracker.call_log[-1], 1, "dim0=4 partial match -> Graph 1")

        opt(torch.randn(10, 3, 234, 20))  # dim2=234 matches, dims 0,3 don't
        self.assertEqual(tracker.call_log[-1], 1, "dim2=234 partial match -> Graph 1")

        opt(torch.randn(10, 3, 234, 5))  # dim2=234, dim3=5 match, dim0 doesn't
        self.assertEqual(tracker.call_log[-1], 1, "dim2+dim3 partial match -> Graph 1")

        opt(torch.randn(4, 3, 234, 20))  # dim0=4, dim2=234 match, dim3 doesn't
        self.assertEqual(tracker.call_log[-1], 1, "dim0+dim2 partial match -> Graph 1")

        # Totally new shape, no exclusion hit
        opt(torch.randn(99, 3, 88, 77))
        self.assertEqual(tracker.call_log[-1], 1, "New shape -> Graph 1")

        self.assertEqual(tracker.frame_count, 2, "No additional recompilations")

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_integer_input_exclusion_basic(self):
        # [FUNCTION DOC] An integer argument that turns dynamic gets an exclusion on its old
        # value: 3 uses graph 0, 5 compiles graph 1, and 3 again returns to graph 0.
        """
        Integer inputs that become dynamic should also get exclusion guards.
        1. foo(x, 3) -> Graph 0: static n=3
        2. foo(x, 5) -> Graph 1: dynamic n, excluded should reject n==3
        3. foo(x, 3) -> should use Graph 0 (static), not Graph 1
        """

        def foo(x, n):
            # [FUNCTION DOC] HELPER: Function under test; multiplies the tensor by the integer.
            return x * n

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        x = torch.randn(4)

        opt(x, 3)
        self.assertEqual(tracker.frame_count, 1)
        self.assertEqual(tracker.call_log[-1], 0)

        opt(x, 5)
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[-1], 1)

        opt(x, 3)
        self.assertEqual(
            tracker.call_log[-1],
            0,
            "Input n=3 should use Graph 0 (static), not Graph 1 (dynamic n).",
        )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_integer_input_exclusion_accumulation(self):
        # [FUNCTION DOC] Integer analogue of the accumulation test: two integer arguments turn
        # dynamic one after the other (three graphs). Checks 5,4 goes to graph 1, and 3,4 goes
        # to graph 0, so the second exclusion is independent of the first.
        """
        Same accumulation scenario as the tensor test but with integer inputs.
        1. foo(x, 3, 4)  -> Graph 0: static (3, 4)
        2. foo(x, 5, 4)  -> Graph 1: dynamic (s0, 4), exclusion rejects n0==3
        3. foo(x, 3, 19) -> Graph 2: dynamic (s0, s1), exclusion should reject
                            n1==4 independently, not require n0==3 AND n1==4
        4. foo(x, 5, 4)  -> should use Graph 1, not Graph 2
        """

        def foo(x, n, m):
            # [FUNCTION DOC] HELPER: Function under test; returns tensor times n plus m.
            return x * n + m

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        x = torch.randn(4)

        opt(x, 3, 4)
        self.assertEqual(tracker.frame_count, 1)
        self.assertEqual(tracker.call_log[-1], 0)

        opt(x, 5, 4)
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[-1], 1)

        opt(x, 3, 19)
        self.assertEqual(tracker.frame_count, 3)
        self.assertEqual(tracker.call_log[-1], 2)

        opt(x, 5, 4)
        self.assertEqual(
            tracker.call_log[-1],
            1,
            "Input (5, 4) should use Graph 1 (s0, 4), not Graph 2 (s0, s1).",
        )

        opt(x, 3, 4)
        self.assertEqual(
            tracker.call_log[-1],
            0,
            "Input (3, 4) should use Graph 0 (static)",
        )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_two_tensor_inputs_exclusion(self):
        # [FUNCTION DOC] Two tensor inputs with dim 1 dynamic in both. The exclusion combines all
        # excluded values and rejects only when all match. Checks partial matches stay on graph
        # 1 without recompiling, and both matching goes to graph 0.
        """
        Multi-tensor exclusion: all pairs are flattened and guarded with
        not-all semantics. The guard rejects only when ALL excluded values
        across ALL tensors match simultaneously.

        1. foo(x=[3,4], y=[5,6])   -> Graph 0: all static
        2. foo(x=[3,10], y=[5,11]) -> Graph 1: x.dim1, y.dim1 dynamic
                                      exclusion: Or(x.dim1!=4, y.dim1!=6)
        3. foo(x=[3,10], y=[5,6])  -> Graph 1 (not all match -> passes)
        4. foo(x=[3,4], y=[5,21])  -> Graph 1 (not all match -> passes)
        5. foo(x=[3,4], y=[5,6])   -> Graph 0 (all match -> rejected)
        6. foo(x=[3,10], y=[5,11]) -> Graph 1
        """

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Function under test; returns the sum of both tensors.
            return x.sum() + y.sum()

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        opt(torch.randn(3, 4), torch.randn(5, 6))
        self.assertEqual(tracker.frame_count, 1)
        self.assertEqual(tracker.call_log[-1], 0)

        opt(torch.randn(3, 10), torch.randn(5, 11))
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[-1], 1)

        # Only y.dim1 matches excluded value; combined Or guard passes.
        opt(torch.randn(3, 10), torch.randn(5, 6))
        self.assertEqual(tracker.frame_count, 2, "Should not recompile")
        self.assertEqual(tracker.call_log[-1], 1)

        # Only x.dim1 matches excluded value; combined Or guard passes.
        opt(torch.randn(3, 4), torch.randn(5, 21))
        self.assertEqual(tracker.frame_count, 2, "Should not recompile")
        self.assertEqual(tracker.call_log[-1], 1)

        # Both match excluded values; Or guard fails -> falls to Graph 0.
        opt(torch.randn(3, 4), torch.randn(5, 6))
        self.assertEqual(tracker.call_log[-1], 0)

        # Neither matches; Or guard passes -> Graph 1.
        opt(torch.randn(3, 10), torch.randn(5, 11))
        self.assertEqual(tracker.call_log[-1], 1)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_multi_tensor_and_scalar_accumulation(self):
        # [FUNCTION DOC] Two 3-dim tensors and three integers made dynamic stepwise (5 graphs).
        # Verifies each step only keeps the exclusion for the newest transition, and that
        # probing with older values falls down the chain of graphs 4 to 0, while all-new values
        # use graph 4. No recompiles during verification.
        """
        3-dim tensors + 3 scalar inputs with cascading accumulation.
        Each step transitions one input while the rest stay the same,
        verifying that only the current transition's exclusion is emitted.

        Graph 0: all static
        Graph 1: x.dim2, y.dim2 dynamic    excl: Or(Ne(x.dim2,4), Ne(y.dim2,7))
        Graph 2: + n dynamic                excl: Ne(n, 2)
        Graph 3: + m dynamic                excl: Ne(m, 3)
        Graph 4: + k dynamic                excl: Ne(k, 4)
        """

        def foo(x, y, n, m, k):
            # [FUNCTION DOC] HELPER: Function under test; combines tensor sums with the integers.
            return x.sum() * n + y.sum() * m + k

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        # -- Compilation steps --

        # Graph 0: all static
        opt(torch.randn(2, 3, 4), torch.randn(5, 6, 7), 2, 3, 4)
        self.assertEqual(tracker.frame_count, 1)
        self.assertEqual(tracker.call_log[-1], 0)

        # Graph 1: tensor dim2 changes
        opt(torch.randn(2, 3, 10), torch.randn(5, 6, 11), 2, 3, 4)
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[-1], 1)

        # Graph 2: scalar n changes (tensor exclusions cleared)
        opt(torch.randn(2, 3, 10), torch.randn(5, 6, 11), 8, 3, 4)
        self.assertEqual(tracker.frame_count, 3)
        self.assertEqual(tracker.call_log[-1], 2)

        # Graph 3: scalar m changes (n exclusion cleared)
        opt(torch.randn(2, 3, 10), torch.randn(5, 6, 11), 8, 9, 4)
        self.assertEqual(tracker.frame_count, 4)
        self.assertEqual(tracker.call_log[-1], 3)

        # Graph 4: scalar k changes (m exclusion cleared)
        opt(torch.randn(2, 3, 10), torch.randn(5, 6, 11), 8, 9, 15)
        self.assertEqual(tracker.frame_count, 5)
        self.assertEqual(tracker.call_log[-1], 4)

        # -- Verification: each input routes to the correct graph --

        # k=4 triggers Graph 4 exclusion -> Graph 3
        opt(torch.randn(2, 3, 10), torch.randn(5, 6, 11), 8, 9, 4)
        self.assertEqual(tracker.call_log[-1], 3, "k=4 should fall to Graph 3")

        # m=3 also triggers Graph 3 exclusion -> Graph 2
        opt(torch.randn(2, 3, 10), torch.randn(5, 6, 11), 8, 3, 4)
        self.assertEqual(tracker.call_log[-1], 2, "m=3 should fall to Graph 2")

        # n=2 also triggers Graph 2 exclusion -> Graph 1
        opt(torch.randn(2, 3, 10), torch.randn(5, 6, 11), 2, 3, 4)
        self.assertEqual(tracker.call_log[-1], 1, "n=2 should fall to Graph 1")

        # tensor dims match original -> Graph 1 exclusion triggers -> Graph 0
        opt(torch.randn(2, 3, 4), torch.randn(5, 6, 7), 2, 3, 4)
        self.assertEqual(tracker.call_log[-1], 0, "Original sizes should use Graph 0")

        # mixed: new tensor dims + new scalars -> Graph 4
        opt(torch.randn(2, 3, 20), torch.randn(5, 6, 21), 50, 60, 70)
        self.assertEqual(tracker.frame_count, 5, "Should not recompile")
        self.assertEqual(tracker.call_log[-1], 4, "All-new values should use Graph 4")


if __name__ == "__main__":
    run_tests()
