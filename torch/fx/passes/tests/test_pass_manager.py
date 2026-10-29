# [MODULE DOC] Unit tests for the fx pass manager: building a PassManager, validating ordering
# constraints (raising RuntimeError when unsatisfiable) and running two independent managers.
# No import-time side effects beyond imports.
import unittest

from ..pass_manager import (
    inplace_wrapper,
    PassManager,
    these_before_those_pass_constraint,
    this_before_that_pass_constraint,
)


class TestPassManager(unittest.TestCase):
# [CLASS DOC] unittest test case exercising PassManager construction, constraint validation and execution.
# Stateless; each test builds its own managers and passes.
    def test_pass_manager_builder(self) -> None:
    # [FUNCTION DOC] Builds a manager from ten multiply-by-two passes and checks validate succeeds without error.
        passes = [lambda x: 2 * x for _ in range(10)]
        pm = PassManager(passes)
        pm.validate()

    def test_this_before_that_pass_constraint(self) -> None:
    # [FUNCTION DOC] Adds a constraint requiring the last pass to run before the first (unsatisfiable given the
    # list order) and asserts validate raises RuntimeError.
        passes = [lambda x: 2 * x for _ in range(10)]
        pm = PassManager(passes)

        # add unfulfillable constraint
        pm.add_constraint(this_before_that_pass_constraint(passes[-1], passes[0]))

        self.assertRaises(RuntimeError, pm.validate)

    def test_these_before_those_pass_constraint(self) -> None:
    # [FUNCTION DOC] Wraps passes with inplace_wrapper, adds an unsatisfiable these-before-those constraint and
    # asserts validate raises RuntimeError.
        passes = [lambda x: 2 * x for _ in range(10)]
        constraint = these_before_those_pass_constraint(passes[-1], passes[0])
        pm = PassManager([inplace_wrapper(p) for p in passes])

        # add unfulfillable constraint
        pm.add_constraint(constraint)

        self.assertRaises(RuntimeError, pm.validate)

    def test_two_pass_managers(self) -> None:
    # [FUNCTION DOC] Builds two managers with distinct passes and constraints, runs each on input one and checks
    # the outputs (8 and 27) to show no state is shared between managers.
        """Make sure we can construct the PassManager twice and not share any
        state between them"""

        passes = [lambda x: 2 * x for _ in range(3)]
        constraint = these_before_those_pass_constraint(passes[0], passes[1])
        pm1 = PassManager()
        for p in passes:
            pm1.add_pass(p)
        pm1.add_constraint(constraint)
        output1 = pm1(1)
        self.assertEqual(output1, 2**3)

        passes = [lambda x: 3 * x for _ in range(3)]
        constraint = these_before_those_pass_constraint(passes[0], passes[1])
        pm2 = PassManager()
        for p in passes:
            pm2.add_pass(p)
        pm2.add_constraint(constraint)
        output2 = pm2(1)
        self.assertEqual(output2, 3**3)
