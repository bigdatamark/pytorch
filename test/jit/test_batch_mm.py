# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for the batch_mm graph pass, which fuses trees of matrix multiplications into
# a tree-reduce node and fuses multiplications sharing one side into a batch-side node, unless
# inputs are mutated in a way that makes the rewrite unsafe. Each test scripts a function, runs
# the pass on its graph and uses file checks to count the nodes before and after.

import torch
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestBatchMM(JitTestCase):
    # [CLASS DOC] Test case for the batch_mm pass covering the no-mutation, permitted-mutation and
    # prohibited-mutation cases for both the tree-reduce and the batch-side patterns.
    @staticmethod
    def _get_test_tensors(n: int):
        # [FUNCTION DOC] HELPER: Static method returning a list of n small integer tensors,
        # alternating between two-by-three (even index) and three-by-two (odd index) shapes so
        # adjacent pairs are valid matrix multiplication operands.
        return [
            (
                torch.tensor([[1 + x, 2 + x, 3 + x], [4 + x, 5 + x, 6 + x]])
                if x % 2 == 0
                else torch.tensor([[1 + x, 2 + x], [3 + x, 4 + x], [5 + x, 6 + x]])
            )
            for x in range(n)
        ]

    def test_batch_mm_no_mutation(self):
        # [FUNCTION DOC] Scripts a sum of four matrix products, checks four multiplications
        # before the pass and exactly one tree-reduce node after, and that scripted output
        # equals eager output.
        def test_batch_mm(
            T1: torch.Tensor,
            T2: torch.Tensor,
            T3: torch.Tensor,
            T4: torch.Tensor,
            T5: torch.Tensor,
            T6: torch.Tensor,
            T7: torch.Tensor,
            T8: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Returns the sum of four matrix products of consecutive
            # argument pairs.
            return (
                torch.mm(T1, T2)
                + torch.mm(T3, T4)
                + torch.mm(T5, T6)
                + torch.mm(T7, T8)
            )

        test_batch_mm_scripted = torch.jit.script(test_batch_mm)

        tensors = TestBatchMM._get_test_tensors(8)
        expected = test_batch_mm(*tensors)

        FileCheck().check_count("aten::mm", 4, exactly=True).run(
            test_batch_mm_scripted.graph
        )
        self.run_pass("batch_mm", test_batch_mm_scripted.graph)
        FileCheck().check_count("prim::MMTreeReduce", 1, exactly=True).run(
            test_batch_mm_scripted.graph
        )

        actual = test_batch_mm_scripted(*tensors)
        self.assertEqual(expected, actual, atol=1e-9, rtol=1e-9)

    def test_batch_mm_permitted_mutation(self):
        # [FUNCTION DOC] Same as the no-mutation test but the function stores the product and a
        # constant tensor into a mapping that is returned; mutating a fresh container is
        # permitted, so one tree-reduce node is expected and outputs must match eager.
        def test_batch_mm(
            T1: torch.Tensor,
            T2: torch.Tensor,
            T3: torch.Tensor,
            T4: torch.Tensor,
            T5: torch.Tensor,
            T6: torch.Tensor,
            T7: torch.Tensor,
            T8: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Builds a mapping with the product sum of four matrix products
            # and a constant tensor, and returns it.
            result = {}
            result["product"] = (
                torch.mm(T1, T2)
                + torch.mm(T3, T4)
                + torch.mm(T5, T6)
                + torch.mm(T7, T8)
            )
            result["constant"] = torch.tensor([42.0])
            return result

        test_batch_mm_scripted = torch.jit.script(test_batch_mm)

        tensors = TestBatchMM._get_test_tensors(8)
        expected = test_batch_mm(*tensors)

        FileCheck().check_count("aten::mm", 4, exactly=True).run(
            test_batch_mm_scripted.graph
        )
        self.run_pass("batch_mm", test_batch_mm_scripted.graph)
        FileCheck().check_count("prim::MMTreeReduce", 1, exactly=True).run(
            test_batch_mm_scripted.graph
        )

        actual = test_batch_mm_scripted(*tensors)
        self.assertEqual(expected, actual, atol=1e-9, rtol=1e-9)

    def test_batch_mm_prohibited_mutation(self):
        # [FUNCTION DOC] Mutates the first operand in place with relu before the products; the
        # pass must leave all four multiplications and create no tree-reduce node.
        @torch.jit.script
        def test_batch_mm(n: int):
            # [FUNCTION DOC] HELPER: Scripted function creating eight zero matrices, mutating the
            # first, and returning the sum of four products.
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            torch.relu_(T1)
            result = (
                torch.mm(T1, T2)
                + torch.mm(T3, T4)
                + torch.mm(T5, T6)
                + torch.mm(T7, T8)
            )
            return result

        FileCheck().check_count("aten::mm", 4, exactly=True).run(test_batch_mm.graph)
        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("aten::mm", 4, exactly=True).check_not(
            "prim::MMTreeReduce"
        ).run(test_batch_mm.graph)

    def test_batch_mm_prohibited_mutation_multiple_adds(self):
        # [FUNCTION DOC] With a mutated operand in one of two product sums stored in a mapping,
        # only the unmutated four-product sum is rewritten: one tree-reduce node and five
        # multiplications remain.
        @torch.jit.script
        def test_batch_mm(n: int):
            # [FUNCTION DOC] HELPER: Scripted function creating ten zero matrices, mutating the
            # first, and returning a mapping of a four-product sum and a five-product sum.
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            T9 = torch.zeros((n, n))
            T10 = torch.zeros((n, n))
            torch.relu_(T1)
            result = {}
            result["no_mutated_parameters"] = (
                torch.mm(T2, T3)
                + torch.mm(T4, T5)
                + torch.mm(T6, T7)
                + torch.mm(T8, T9)
            )
            result["all_parameters"] = (
                torch.mm(T1, T2)
                + torch.mm(T3, T4)
                + torch.mm(T5, T6)
                + torch.mm(T7, T8)
                + torch.mm(T9, T10)
            )
            return result

        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("prim::MMTreeReduce", 1, exactly=True).check_count(
            "aten::mm", 5, exactly=True
        ).run(test_batch_mm.graph)

    def test_batch_mm_prohibited_mutation_if_node(self):
        # [FUNCTION DOC] Mutation inside one branch of a conditional blocks the rewrite of that
        # branch only; after the pass there are five multiplications and one tree-reduce node
        # (from the other branch).
        @torch.jit.script
        def test_batch_mm(n: int, use_t1: bool):
            # [FUNCTION DOC] HELPER: Scripted function with a flag; one branch mutates the first
            # matrix and sums five products, the other sums four products.
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            T9 = torch.zeros((n, n))
            T10 = torch.zeros((n, n))
            if use_t1:
                torch.relu_(T1)
                return (
                    torch.mm(T1, T2)
                    + torch.mm(T3, T4)
                    + torch.mm(T5, T6)
                    + torch.mm(T7, T8)
                    + torch.mm(T9, T10)
                )
            else:
                return (
                    torch.mm(T2, T3)
                    + torch.mm(T4, T5)
                    + torch.mm(T6, T7)
                    + torch.mm(T8, T9)
                )

        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("aten::mm", 5, exactly=True).check_count(
            "prim::MMTreeReduce", 1, exactly=True
        ).run(test_batch_mm.graph)

    def test_batch_mm_side_permitted_mutation(self):
        # [FUNCTION DOC] Eight multiplications sharing the left operand with no mutation are fused
        # into one batch-side node and no plain multiplication remains.
        @torch.jit.script
        def test_batch_mm(n: int):
            # [FUNCTION DOC] HELPER: Scripted function multiplying a shared matrix by eight others
            # and returning the results in a mapping.
            result = {}
            A = torch.zeros((n, n))
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            result["T1"] = torch.mm(A, T1)
            result["T2"] = torch.mm(A, T2)
            result["T3"] = torch.mm(A, T3)
            result["T4"] = torch.mm(A, T4)
            result["T5"] = torch.mm(A, T5)
            result["T6"] = torch.mm(A, T6)
            result["T7"] = torch.mm(A, T7)
            result["T8"] = torch.mm(A, T8)
            return result

        FileCheck().check_count("aten::mm", 8, exactly=True).run(test_batch_mm.graph)
        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("prim::MMBatchSide", 1, exactly=True).check_not(
            "aten::mm"
        ).run(test_batch_mm.graph)

    def test_batch_mm_side_prohibited_mutation_uncommon_side(self):
        # [FUNCTION DOC] Mutating one of the non-shared operands: the pass leaves one plain
        # multiplication (for the mutated operand) and one batch-side node covering the rest.
        @torch.jit.script
        def test_batch_mm(n: int):
            # [FUNCTION DOC] HELPER: Scripted function with ten products sharing the left matrix,
            # where the first right operand is mutated in place.
            A = torch.zeros((n, n))
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            T9 = torch.zeros((n, n))
            T10 = torch.zeros((n, n))
            torch.relu_(T1)
            result = {}
            result["T1"] = torch.mm(A, T1)
            result["T2"] = torch.mm(A, T2)
            result["T3"] = torch.mm(A, T3)
            result["T4"] = torch.mm(A, T4)
            result["T5"] = torch.mm(A, T5)
            result["T6"] = torch.mm(A, T6)
            result["T7"] = torch.mm(A, T7)
            result["T8"] = torch.mm(A, T8)
            result["T9"] = torch.mm(A, T9)
            result["T10"] = torch.mm(A, T10)
            return result

        FileCheck().check_count("aten::mm", 10, exactly=True).run(test_batch_mm.graph)
        self.run_pass("batch_mm", test_batch_mm.graph)

        FileCheck().check_count("aten::mm", 1, exactly=True).run(test_batch_mm.graph)
        FileCheck().check_count("prim::MMBatchSide", 1, exactly=True).run(
            test_batch_mm.graph
        )

    def test_batch_mm_side_prohibited_mutation_common_side(self):
        # [FUNCTION DOC] Mutating the shared operand blocks the batch-side rewrite: all ten
        # multiplications remain and no batch-side node is created.
        @torch.jit.script
        def test_batch_mm(n: int):
            # [FUNCTION DOC] HELPER: Scripted function with ten products sharing the left matrix,
            # which is mutated in place first.
            A = torch.zeros((n, n))
            T1 = torch.zeros((n, n))
            T2 = torch.zeros((n, n))
            T3 = torch.zeros((n, n))
            T4 = torch.zeros((n, n))
            T5 = torch.zeros((n, n))
            T6 = torch.zeros((n, n))
            T7 = torch.zeros((n, n))
            T8 = torch.zeros((n, n))
            T9 = torch.zeros((n, n))
            T10 = torch.zeros((n, n))
            torch.relu_(A)
            result = {}
            result["T1"] = torch.mm(A, T1)
            result["T2"] = torch.mm(A, T2)
            result["T3"] = torch.mm(A, T3)
            result["T4"] = torch.mm(A, T4)
            result["T5"] = torch.mm(A, T5)
            result["T6"] = torch.mm(A, T6)
            result["T7"] = torch.mm(A, T7)
            result["T8"] = torch.mm(A, T8)
            result["T9"] = torch.mm(A, T9)
            result["T10"] = torch.mm(A, T10)
            return result

        FileCheck().check_count("aten::mm", 10, exactly=True).run(test_batch_mm.graph)
        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("aten::mm", 10, exactly=True).check_not(
            "prim::MMBatchSide"
        ).run(test_batch_mm.graph)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
