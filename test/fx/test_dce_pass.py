# Owner(s): ["module: fx"]
# [MODULE DOC] Tests for dead code elimination (DCE) on FX graphs. A shared helper traces a
# module, checks that dead nodes exist (or not), runs the graph's DCE (optionally with a
# custom impurity predicate), checks the placeholder count is preserved and no dead pure
# nodes remain, and compares outputs before and after. Individual tests cover chains, unused
# attributes, placeholders, impure modules, side-effecting ops (asserts, in-place ops, out
# arguments, random ops, collectives) and op overload packets registered as effectful.
# Running the file directly raises an error pointing to the main FX test file.
import copy
import unittest

import torch
import torch.fx
from torch.testing._internal.common_utils import (
    IS_MACOS,
    raise_on_run_directly,
    TestCase,
)


class TestDCE(TestCase):
    # [CLASS DOC] Test case for graph dead code elimination. Helper methods inspect graphs
    # and drive one shared run-and-verify routine; each test defines a small module and
    # states whether DCE is expected to change the graph. No instance state.
    def _custom_is_impure_node(self, node: torch.fx.Node) -> bool:
        # [FUNCTION DOC] HELPER: Custom impurity predicate for DCE tests. Returns True if
        # the node is impure under the default rule or if its target is the aten add
        # operator packet; otherwise False.
        if node.is_impure():
            return True
        # a custom function that defines add operators as impure.
        if node.target == torch.ops.aten.add:
            return True
        return False

    def _has_nodes_without_users(self, m: torch.fx.GraphModule, custom: bool = False):
        # [FUNCTION DOC] HELPER: Returns True if some node in the graph that is considered
        # pure has no users, using either the default impurity rule or, when custom is set,
        # the custom predicate. Impure nodes (placeholders, outputs and others) are skipped.
        # Returns False otherwise.
        for node in m.graph.nodes:
            if (not custom and node.is_impure()) or (
                custom and self._custom_is_impure_node(node)
            ):
                continue
            if len(node.users) == 0:
                return True
        return False

    def _get_num_placeholders(self, m: torch.fx.GraphModule) -> int:
        # [FUNCTION DOC] HELPER: Counts and returns the number of placeholder nodes in the
        # module's graph.
        count = 0
        for node in m.graph.nodes:
            if node.op == "placeholder":
                count += 1
        return count

    def _run_dce_and_test(
        self,
        m: torch.nn.Module,
        expect_dce_changes: bool,
        modules_to_be_leafs: set[type] | None = None,
        custom: bool = False,
    ):
        # [FUNCTION DOC] HELPER: Core routine of the tests. Traces the module with a custom
        # tracer (so chosen module types become leaves), wraps it in a GraphModule, prints
        # the graph, and asserts that nodes without users exist exactly when
        # expect_dce_changes is set. Runs dead code elimination (with the custom impurity
        # predicate if custom is set) and asserts its returned changed flag matches the
        # expectation, that no dead pure nodes remain and that the placeholder count is
        # unchanged. Recompiles and asserts the original module and the traced module give
        # equal results on one element tensors (value 1.5, one per placeholder).
        class TestTracer(torch.fx.Tracer):
            # [CLASS DOC] HELPER: Tracer that treats the module types given in the enclosing
            # call as leaf modules.
            def is_leaf_module(self, m, qualname):
                # [FUNCTION DOC] HELPER: Returns True when the module type is in the leaf
                # set. Otherwise returns the result of the base class trace method called
                # with the module and qualified name.
                # KNOWN ISSUE: the fallback calls the trace method rather than the base
                #     is_leaf_module, which looks like a mistake; it is only reached for
                #     modules not in the leaf set.
                if modules_to_be_leafs and type(m) in modules_to_be_leafs:
                    return True
                return super().trace(m, qualname)

        traced: torch.fx.GraphModule = torch.fx.GraphModule(m, TestTracer().trace(m))
        print(str(traced.graph))

        # Verify there are nodes without users (if expected).
        has_nodes_without_users = self._has_nodes_without_users(traced, custom=custom)
        if expect_dce_changes:
            self.assertTrue(has_nodes_without_users)
        else:
            self.assertFalse(has_nodes_without_users)

        # Get the original number of placeholders to verify it doesn't change
        # during DCE.
        orig_num_phs = self._get_num_placeholders(traced)
        if custom:
            changed = traced.graph.eliminate_dead_code(
                is_impure_node=self._custom_is_impure_node
            )
        else:
            changed = traced.graph.eliminate_dead_code()

        self.assertTrue(changed if expect_dce_changes else not changed)

        # Verify there are no nodes without users after DCE is run.
        self.assertFalse(self._has_nodes_without_users(traced, custom=custom))
        new_num_phs = self._get_num_placeholders(traced)
        self.assertEqual(orig_num_phs, new_num_phs)

        traced.recompile()
        # Make sure we run and get the same results before/after DCE.
        inputs = [torch.tensor([1.5])] * new_num_phs
        inputs_copy = copy.deepcopy(inputs)
        self.assertTrue(torch.equal(m(*inputs), traced(*inputs_copy)))

    def test_simple(self):
        # [FUNCTION DOC] Single dead node (an unused addition) is removed by DCE.
        """
        Tests that a single node in the graph is DCE'd correctly.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one parameter; forward computes an unused sum
            # and returns the input plus the parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers one parameter holding a one element
                # tensor.
                super().__init__()
                self.attr_1 = torch.nn.Parameter(torch.tensor([-0.9]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes an unused value and returns the input plus
                # the parameter.
                a = x + 1  # noqa: F841
                return x + self.attr_1

        self._run_dce_and_test(TestModule(), expect_dce_changes=True)

    def test_dead_chain(self):
        # [FUNCTION DOC] A dead chain of two nodes (an unused addition feeding an unused
        # multiplication) is fully removed.
        """
        Tests that a chain of two nodes in the graph are DCE'd correctly.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one parameter and a forward containing a dead
            # two node chain.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers one parameter holding a one element
                # tensor.
                super().__init__()
                self.attr_1 = torch.nn.Parameter(torch.tensor([-0.9]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a dead chain from the input and returns the
                # input plus the parameter.
                a = x + 1
                b = a * 7  # noqa: F841
                return x + self.attr_1

        self._run_dce_and_test(TestModule(), expect_dce_changes=True)

    def test_dead_getattr(self):
        # [FUNCTION DOC] A dead attribute fetch (a parameter used only by a dead node) is
        # removed with that node.
        """
        Tests that a getatrr in the graph is DCE'd correctly.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward uses its parameter only in a dead
            # computation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers one parameter holding a one element
                # tensor.
                super().__init__()
                self.attr_1 = torch.nn.Parameter(torch.tensor([-0.9]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Dead value uses the parameter; returns the input
                # plus 11.
                a = x + 1
                b = a * self.attr_1  # noqa: F841
                return x + 11

        self._run_dce_and_test(TestModule(), expect_dce_changes=True)

    def test_dead_placeholder(self):
        # [FUNCTION DOC] An unused placeholder must not be removed since it would change the
        # function signature; DCE is expected to make no change.
        """
        Tests that a placeholder in the graph is not DCE'd, as that would change
        the function signature.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward ignores its second argument.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the first input plus 7 and ignores the
                # second.
                return x + 7

        self._run_dce_and_test(TestModule(), expect_dce_changes=False)

    def test_dead_placeholder_with_user(self):
        # [FUNCTION DOC] An unused placeholder is kept while a dead node that uses it is
        # removed.
        """
        Tests that a placeholder in the graph is not DCE'd, as that would change
        the function signature. Also verifies that a dead node that uses the
        placeholder is DCE'd.

        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a forward that has a dead node using the
            # second input.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes an unused value from the second input;
                # returns the first input plus 7.
                a = y + 2  # noqa: F841
                return x + 7

        self._run_dce_and_test(TestModule(), expect_dce_changes=True)

    def test_keep_module_with_side_effects(self):
        # [FUNCTION DOC] A module call whose module class is marked impure is kept by DCE,
        # using a subclass of ReLU as a leaf module in the tracer.
        """
        Test that DCE doesn't remove a module if it's specified as having side effects.
        """

        class ReLUImpure(torch.nn.ReLU):
            # [CLASS DOC] HELPER: ReLU subclass carrying the impure marker attribute set to
            # true.
            _is_impure = True

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module owning an impure ReLU whose result is unused.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the impure ReLU submodule.
                super().__init__()
                self.relu = ReLUImpure()

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Applies the ReLU, discards the result and returns
                # the input times 2.
                r = self.relu(a)  # noqa: F841
                return a * 2

        self._run_dce_and_test(
            TestModule(), expect_dce_changes=False, modules_to_be_leafs={ReLUImpure}
        )

    def test_keep_torch_assert(self):
        # [FUNCTION DOC] The torch assertion call has side effects and survives DCE.
        """
        Test that DCE doesn't remove torch._assert since it has side effects.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward asserts a tensor equals itself.
            def forward(self, a: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Calls the torch assert function (result unused) and
                # returns the input times 2.
                torch._assert(torch.equal(a, a), "a must equal a")
                return a * 2

        # Note: Don't need to specify torch._assert as having side effects
        # because it's known to.
        self._run_dce_and_test(TestModule(), expect_dce_changes=False)

    def test_keep_setitem(self):
        # [FUNCTION DOC] Regression test for a GitHub issue: the item assignment operator
        # has side effects and must not be removed. Compiles a copy of the model with a
        # backend that applies constant folding to the graph, runs eager and compiled
        # versions under inference mode on separate copies of a random 4D input and compares
        # with a small tolerance.
        """
        Fix issue: https://github.com/pytorch/pytorch/issues/145697
        Test that DCE doesn't remove operator.setitem since it has side effects.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that multiplies one element of its input in place
            # by 2.0 and returns the input times 2.
            def forward(self, a: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: In-place scales one element of the input (side
                # effect on the argument) and returns the input times 2.
                a[0, 0, 0, 0] *= 2.0
                return a * 2

        def dce_backend(gm, inputs, **kwargs):
            # [FUNCTION DOC] HELPER: Compile backend for the test; imports the inductor
            # constant folding module, folds constants in the graph module in place and
            # returns it. Extra arguments are ignored.
            import torch._inductor.constant_folding

            torch._inductor.constant_folding.constant_fold(gm)
            return gm

        x = torch.randn(1, 3, 224, 224)
        dce_x = x.detach().clone()
        model = TestModule().eval()
        dce_mod = torch.compile(copy.deepcopy(model), backend=dce_backend)

        with torch.inference_mode():
            eager_out = model(x)
            out = dce_mod(dce_x)
        self.assertEqual(eager_out, out, atol=1e-5, rtol=1e-5)

    def test_impure_nodes_args(self):
        # [FUNCTION DOC] An in-place aten op called as a function node (effect visible
        # through its argument) is kept by DCE.
        """
        Test that DCE doesn't remove call_function nodes with side effects.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling an in-place add op on its input and
            # returning twice the input.
            def forward(self, a: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Adds 1 to the input in place through the op
                # namespace and returns the input times 2.
                torch._ops.ops.aten.add_.Tensor(a, 1)
                return a * 2

        # %add_ node should not be removed because it has side effects.
        self._run_dce_and_test(TestModule(), expect_dce_changes=False)

    def test_impure_random(self):
        # [FUNCTION DOC] Random ops must not be removed by DCE. First traces a module with
        # an unused random tensor and checks nothing is removed. Then compiles a module that
        # makes many unused random tensors (several distribution functions, some using an
        # explicit generator) with a backend that counts random op nodes before and after
        # DCE, asserting the count is unchanged, and compares eager with compiled results
        # under the same seed.
        """
        Test that DCE doesn't remove call_function for torch.rand and other random functions.
        Tests both FX tracing and AOT compilation (issue #151524).
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module creating an unused random tensor and returning the
            # input times 2.
            def forward(self, a: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Makes an unused uniform random tensor and returns
                # the input times 2.
                x = torch.rand([10])  # noqa: F841
                return a * 2

        # Test FX tracing + DCE
        self._run_dce_and_test(TestModule(), expect_dce_changes=False)

        # Test comprehensive random functions in AOT compilation
        class ComprehensiveRandomModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module creating many random tensors, two of which are used
            # in the result.
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Generates unused random tensors from several random
                # functions and generators, plus two used ones (one default and one from a
                # seeded generator), and returns the input plus the used values.
                # Test various random functions that should be preserved
                a = torch.rand(1)  # noqa: F841
                b = torch.randn(1)  # noqa: F841
                c = torch.randint(0, 10, (1,))  # noqa: F841
                d = torch.randperm(5)  # noqa: F841
                e = torch.normal(0, 1, (1,))  # noqa: F841
                f = torch.poisson(torch.tensor([1.0]))  # noqa: F841
                g = torch.rand(1)  # Used

                # Test that random operations with explicit generators are also preserved
                gen = torch.Generator().manual_seed(123)
                h = torch.rand(1, generator=gen)  # noqa: F841
                i = torch.randn(1, generator=gen)  # noqa: F841
                j = torch.rand(1, generator=gen)  # Used
                return x + g + j

        def aot_backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Compile backend that counts nodes calling random
            # functions before and after eliminating dead code, asserts via assertEqual that
            # the count is unchanged, and returns the module's forward callable. Uses the
            # enclosing test case.
            def count_random_ops():
                # [FUNCTION DOC] HELPER: Counts call_function nodes in the graph whose
                # target name contains any of the random function names; returns the count.
                return len(
                    [
                        n
                        for n in gm.graph.nodes
                        if n.op == "call_function"
                        and any(
                            fn in str(n.target)
                            for fn in [
                                "rand",
                                "randn",
                                "randint",
                                "randperm",
                                "normal",
                                "poisson",
                            ]
                        )
                    ]
                )

            rand_count = count_random_ops()
            gm.graph.eliminate_dead_code()
            self.assertEqual(
                count_random_ops(), rand_count, "Random ops should be preserved"
            )
            return gm.forward

        model = ComprehensiveRandomModule()
        torch.manual_seed(42)
        eager_result = model(torch.tensor([1.0]))
        torch.manual_seed(42)
        compiled_result = torch.compile(model, backend=aot_backend)(torch.tensor([1.0]))
        self.assertEqual(eager_result, compiled_result)

    def test_impure_kwargs(self):
        # [FUNCTION DOC] A node whose side effect is expressed through an out keyword
        # argument is kept by DCE.
        """
        Test that DCE doesn't remove call_function nodes with side effects on kwargs.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module writing into its input through an out keyword
            # argument.
            def forward(self, a: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Computes the input plus 1, calls the add out op
                # writing into the input and returns the input.
                b = a + 1
                torch._ops.ops.aten.add.out(b, b, out=a, alpha=2)
                return a

        # %add_out node should not be removed because it has side effects.
        self._run_dce_and_test(TestModule(), expect_dce_changes=False)

    def test_impure_custom(self):
        # [FUNCTION DOC] A node that is pure by default but marked impure by a custom
        # predicate (aten add) is kept when the custom predicate is passed to DCE.
        """
        Test that DCE doesn't remove nodes marked as impure by a custom function.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an unused aten add call.
            def forward(self, a: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Computes the input plus 1, an unused aten add of
                # that value, and returns the input.
                b = a + 1
                c = torch._ops.ops.aten.add(b, b)  # noqa: F841
                return a

        # %add_out node should not be removed because it has side effects.
        self._run_dce_and_test(TestModule(), expect_dce_changes=False, custom=True)

    @unittest.skipIf(IS_MACOS, "Not working on macos")
    def test_keep_collectives(self):
        # [FUNCTION DOC] Collective operations are kept even if results are unused. Skipped
        # on macOS. Initializes a fake process group with world size 2 and rank 0, runs the
        # shared check (default impurity rule), then destroys the process group.
        """
        Test that DCE doesn't remote collective ops even the results are not used.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module issuing an all-reduce collective and wait on a
            # tensor product, then returning another product.
            def forward(
                self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Multiplies the inputs twice, launches an all-reduce
                # on the second product with the sum reduction, waits on it (result unused),
                # and returns the first product.
                d = torch.ops.aten.mul.Tensor(a, b)
                e = torch.ops.aten.mul.Tensor(a, c)
                future = torch.ops._c10d_functional.all_reduce.default(e, "sum", "0")
                torch.ops._c10d_functional.wait_tensor.default(future)
                return d

        torch.distributed.init_process_group(
            backend="fake",
            world_size=2,
            rank=0,
        )
        # collective nodes should not be removed because they have side effects.
        self._run_dce_and_test(TestModule(), expect_dce_changes=False, custom=False)
        torch.distributed.destroy_process_group()

    @unittest.skipIf(IS_MACOS, "Not working on macos")
    def test_keep_collectives_no_overload(self):
        # [FUNCTION DOC] Same as the collectives test but using the ops without explicit
        # overload names. Skipped on macOS. Initializes and destroys a fake process group
        # around the shared check.
        """
        Test that DCE doesn't remote collective ops (no overload version) even the results are not used.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module issuing collectives through overload-less op names.
            def forward(
                self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Same computation as the overload version but using
                # packet-style op names.
                d = torch.ops.aten.mul(a, b)
                e = torch.ops.aten.mul(a, c)
                future = torch.ops._c10d_functional.all_reduce(e, "sum", "0")
                torch.ops._c10d_functional.wait_tensor(future)
                return d

        torch.distributed.init_process_group(
            backend="fake",
            world_size=2,
            rank=0,
        )
        # collective nodes should not be removed because they have side effects.
        self._run_dce_and_test(TestModule(), expect_dce_changes=False, custom=False)
        torch.distributed.destroy_process_group()

    def test_side_effectful_op_overload_packet(self):
        # [FUNCTION DOC] An op overload packet whose default overload is registered as
        # effectful survives DCE. Defines a temporary library with one op, registers the
        # effect, traces a module that calls the packet, asserts the node target is a packet
        # and has no users, runs DCE and asserts the node remains. Always destroys the
        # registration handle and drops the library.
        """Test that OpOverloadPacket targets survive DCE when the default
        overload is registered as effectful via _register_effectful_op.
        is_impure() delegates from OpOverloadPacket to the default overload."""
        from torch._higher_order_ops.effects import _register_effectful_op
        from torch._library.effects import EffectType

        lib = torch.library.Library("dce_test", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("check_op(Tensor x) -> Tensor")
        lib.impl("check_op", lambda x: x.clone(), "CPU")
        lib.impl("check_op", lambda x: x.clone(), "Meta")

        handle = _register_effectful_op(
            torch.ops.dce_test.check_op.default, EffectType.ORDERED
        )
        try:

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module that calls the test op and returns its input.
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Calls the registered effectful op, ignoring the
                    # result, and returns the input.
                    torch.ops.dce_test.check_op(x)
                    return x

            traced = torch.fx.symbolic_trace(M())
            check_nodes = [
                n
                for n in traced.graph.nodes
                if n.op == "call_function" and "check_op" in str(n.target)
            ]
            self.assertTrue(len(check_nodes) > 0)
            self.assertIsInstance(check_nodes[0].target, torch._ops.OpOverloadPacket)
            self.assertEqual(len(check_nodes[0].users), 0)
            traced.graph.eliminate_dead_code()
            check_nodes_after = [
                n
                for n in traced.graph.nodes
                if n.op == "call_function" and "check_op" in str(n.target)
            ]
            self.assertTrue(
                len(check_nodes_after) > 0,
                "OpOverloadPacket node DCE'd despite effectful default overload",
            )
        finally:
            handle.destroy()
            del lib

    @torch._dynamo.config.patch(disable=True)
    def test_side_effectful_packet_only_registration(self):
        # [FUNCTION DOC] An op overload packet added directly to the side-effectful function
        # set is respected by DCE even when no effect is registered on the default overload.
        # Dynamo is disabled for this test. Creates a temporary library op, adds it to the
        # set, traces and runs DCE and asserts the call remains; the set entry is removed in
        # a finally block.
        """Test that OpOverloadPacket in _side_effectful_functions is respected
        even when the default overload is NOT registered via _register_effectful_op."""
        from torch.fx.node import _side_effectful_functions

        lib = torch.library.Library("dce_test", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("packet_only_op(Tensor x) -> Tensor")
        lib.impl("packet_only_op", lambda x: x.clone(), "CPU")
        lib.impl("packet_only_op", lambda x: x.clone(), "Meta")

        _side_effectful_functions.add(torch.ops.dce_test.packet_only_op)
        try:

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module that calls the packet-only op and returns its
                # input.
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Calls the op, ignoring the result, and returns
                    # the input.
                    torch.ops.dce_test.packet_only_op(x)
                    return x

            traced = torch.fx.symbolic_trace(M())
            traced.graph.eliminate_dead_code()
            check_nodes = [
                n
                for n in traced.graph.nodes
                if n.op == "call_function" and "packet_only_op" in str(n.target)
            ]
            self.assertTrue(
                len(check_nodes) > 0,
                "OpOverloadPacket DCE'd despite packet-only registration",
            )
        finally:
            _side_effectful_functions.discard(torch.ops.dce_test.packet_only_op)
            del lib


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
