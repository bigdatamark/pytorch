# Owner(s): ["module: fx"]
# [MODULE DOC] Tests for the FX pass infrastructure in torch.fx.passes.infra: PassManager
# running PassBase and plain callable passes, pass_result_wrapper, constraint validation, check
# functions, topological sorting of passes, and error reporting. Defines several simple passes
# that swap add, mul, div and sub targets.

import torch
import torch.fx as fx
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.infra.pass_manager import (
    _topological_sort_passes,
    pass_result_wrapper,
    PassManager,
    this_before_that_pass_constraint,
)
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


# Pass that uses PassBase and returns a PassResult (best scenario)
class ReplaceAddWithMulPass(PassBase):
    # [CLASS DOC] HELPER: PassBase pass that returns a PassResult; it is the well-behaved form.
    def call(self, gm) -> PassResult:
        # [FUNCTION DOC] HELPER: Rewrites every call_function node whose target is torch.add to
        # target torch.mul in place and returns a PassResult of the graph module, flagged
        # modified only if something was changed.
        modified = False
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.add:
                node.target = torch.mul
                modified = True
        return PassResult(gm, modified)


# Pass that is a callable and returns a PassResult
def replace_mul_with_div_pass(gm) -> PassResult:
    # [FUNCTION DOC] HELPER: Callable pass doing the same for torch.mul nodes, retargeting them
    # to torch.div, and returning a PassResult with the modified flag set when a node changed.
    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.mul:
            node.target = torch.div
            modified = True
    return PassResult(gm, modified)


# Pass that is a PassBase and does not return a PassResult
# Need to wrap with pass_result_wrapper or else it will fail
class ReplaceDivWithSubPass(PassBase):
    # [CLASS DOC] HELPER: PassBase pass whose call method does not return a
    # PassResult (it returns nothing), so it must be wrapped by pass_result_wrapper
    # before use in a PassManager.
    def call(self, gm) -> PassResult:
        # [FUNCTION DOC] HELPER: Retargets call_function nodes with target torch.div to
        # torch.sub in place. Returns nothing.
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.div:
                node.target = torch.sub


# Pass that is a callable and does not return a PassResult
# Need to wrap with pass_result_wrapper or else it will fail
def replace_sub_with_add_pass(gm) -> PassResult:
    # [FUNCTION DOC] HELPER: Callable pass retargeting torch.sub call_function nodes to
    # torch.add in place. Returns nothing, so it needs pass_result_wrapper.
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.sub:
            node.target = torch.add


class AddModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose forward adds the input to itself and then adds the input
    # once more; used as the traced test subject.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Computes the input added to itself, adds the input again and
        # returns the result.
        y = torch.add(x, x)
        z = torch.add(y, x)
        return z


class TestPassManager(TestCase):
    # [CLASS DOC] Test case for PassManager behavior: pass execution, constraints, check
    # functions, ordering and error messages.
    def test_pass_manager(self):
        # [FUNCTION DOC] Builds a PassManager with the add to mul, mul to div, div to sub and
        # sub to add passes (the last two wrapped with pass_result_wrapper) and five steps.
        # Validates constraints, checks four passes are registered, runs it on a traced
        # AddModule, asserts the result is a GraphModule, and asserts every call_function node
        # ends up targeting torch.add after the full cycle. The comment in the code saying nodes
        # are divs is stale.
        """
        Tests that the pass manager runs the passes correctly.
        """

        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        pm = PassManager(
            passes=[
                ReplaceAddWithMulPass(),
                replace_mul_with_div_pass,
                pass_result_wrapper(ReplaceDivWithSubPass()),
                pass_result_wrapper(replace_sub_with_add_pass),
            ],
            steps=5,
        )

        pm.validate_constraints()
        self.assertEqual(len(pm.passes), 4)

        res = pm(traced_m)
        modified_m = res.graph_module
        if not isinstance(modified_m, fx.GraphModule):
            raise AssertionError(
                f"Expected modified_m to be fx.GraphModule, got {type(modified_m)}"
            )

        # Check that all call_function nodes are divs
        for node in modified_m.graph.nodes:
            if node.op == "call_function":
                self.assertEqual(node.target, torch.add)

    def test_this_before_that_pass_constraint(self):
        # [FUNCTION DOC] Adds a constraint demanding the last of ten identical lambda passes run
        # before the first, which cannot be satisfied by the given order, and asserts
        # validate_constraints raises RuntimeError.
        """
        Tests the construction of constraints
        """
        passes = [lambda x: 2 * x for _ in range(10)]
        pm = PassManager(passes)

        # add unfulfillable constraint
        pm.add_constraint(this_before_that_pass_constraint(passes[-1], passes[0]))

        with self.assertRaises(RuntimeError):
            pm.validate_constraints()

    def test_pass_manager_checks(self):
        # [FUNCTION DOC] Registers a check function that raises ValueError when any
        # call_function node target is not torch.div, on a manager with the add to mul and mul
        # to div passes only, and asserts running the manager raises ValueError.
        """
        Tests that users can add in check functions correctly
        """
        m = AddModule()
        traced_m = fx.symbolic_trace(m)
        pm = PassManager(passes=[ReplaceAddWithMulPass(), replace_mul_with_div_pass])

        def check_div_target(graph_module):
            # [FUNCTION DOC] HELPER: Check function raising ValueError if any call_function node
            # in the graph module has a target other than torch.div.
            for node in graph_module.graph.nodes:
                if node.op == "call_function" and node.target != torch.div:
                    raise ValueError("Target should be div!")

        pm.add_checks(check_div_target)

        with self.assertRaises(ValueError):
            pm(traced_m)

    def test_pass_manager_bad_checks(self):
        # [FUNCTION DOC] Asserts add_checks raises TypeError when given a check function with
        # two parameters instead of one.
        """
        Checks that we error if we pass in a check function with the wrong parameters
        """

        def check_bad_args(graph_module, i):
            # [FUNCTION DOC] HELPER: Check function with an invalid two-parameter signature; the
            # body does nothing.
            pass

        pm = PassManager()
        self.assertRaises(TypeError, pm.add_checks, check_bad_args)

    def test_topological_sort(self):
        # [FUNCTION DOC] Tests _topological_sort_passes. With no constraints the original
        # order is kept; with a set of before constraints over six passes the result is the
        # order 4, 5, 0, 2, 3, 1; with circular constraints over three passes it raises
        # RuntimeError whose first argument is a message listing the passes in the cycle.
        # Asserts on the message text.
        """
        Tests that passes are correctly ordered based on constraints.
        """

        def pass0(x):
            # [FUNCTION DOC] HELPER: Identity pass.
            return x

        def pass1(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one.
            return x + 1

        def pass2(x):
            # [FUNCTION DOC] HELPER: Returns the input plus two.
            return x + 2

        def pass3(x):
            # [FUNCTION DOC] HELPER: Returns the input plus three.
            return x + 3

        def pass4(x):
            # [FUNCTION DOC] HELPER: Returns the input plus four.
            return x + 4

        def pass5(x):
            # [FUNCTION DOC] HELPER: Returns the input plus five.
            return x + 5

        # Not passing any constraints should keep the original order
        passes = [pass0, pass1, pass2, pass3, pass4, pass5]
        sorted = _topological_sort_passes(passes, [])
        self.assertEqual(sorted, passes)

        # Graph that we are constructing:
        #     5 ---->  0  <---- 4
        #     |                 |
        #     +-> 2 -> 3 -> 1 <-+
        # Which has a possible topological order of: [4, 5, 0, 2, 3, 1]
        passes = [pass0, pass1, pass2, pass3, pass4, pass5]
        constraints = [
            this_before_that_pass_constraint(pass5, pass0),
            this_before_that_pass_constraint(pass5, pass2),
            this_before_that_pass_constraint(pass4, pass0),
            this_before_that_pass_constraint(pass4, pass1),
            this_before_that_pass_constraint(pass2, pass3),
            this_before_that_pass_constraint(pass3, pass1),
        ]
        sorted = _topological_sort_passes(passes, constraints)
        self.assertEqual(sorted, [pass4, pass5, pass0, pass2, pass3, pass1])

        # Circular dependency should result in the circular_dep flag being set
        passes = [pass0, pass1, pass2]
        constraints = [
            this_before_that_pass_constraint(passes[0], passes[1]),
            this_before_that_pass_constraint(passes[1], passes[2]),
            this_before_that_pass_constraint(passes[2], passes[0]),
        ]
        with self.assertRaises(RuntimeError) as e:
            _topological_sort_passes(passes, constraints)
        expected_error_msg = (
            f"Circular dependency detected within the following passes: {passes}"
        )
        self.assertEqual(e.exception.args[0], expected_error_msg)

    def test_pass_manager_error(self):
        # [FUNCTION DOC] Checks the error message on a failing pass. First, an unwrapped
        # ReplaceDivWithSubPass (returns no result) must raise an exception whose message
        # names the passes run so far in reverse order. Then a manager whose last pass
        # raises RuntimeError must report all earlier pass names in the message. Asserted
        # with regular expressions.
        """
        Tests error catching + debug
        """

        def pass_fail(graph_module):
            # [FUNCTION DOC] HELPER: Always raises RuntimeError with the message bad.
            raise RuntimeError("bad")

        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        pm = PassManager(
            passes=[
                ReplaceAddWithMulPass(),
                replace_mul_with_div_pass,
                ReplaceDivWithSubPass(),
                pass_result_wrapper(replace_sub_with_add_pass),
            ],
        )

        # Comment out this line to see the actual error message
        error_msg = (
            "ReplaceDivWithSubPass.*ReplaceAddWithMulPass.*replace_mul_with_div_pass"
        )
        with self.assertRaisesRegex(Exception, error_msg):
            pm(traced_m)

        pm = PassManager(
            passes=[
                ReplaceAddWithMulPass(),
                replace_mul_with_div_pass,
                pass_result_wrapper(ReplaceDivWithSubPass()),
                pass_result_wrapper(replace_sub_with_add_pass),
                pass_fail,
            ],
        )

        # Comment out this line to see the actual error message
        error_msg = "pass_fail.*ReplaceAddWithMulPass.*replace_mul_with_div_pass.*ReplaceDivWithSubPass.*replace_sub_with_add_pass"
        with self.assertRaisesRegex(Exception, error_msg):
            pm(traced_m)


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
