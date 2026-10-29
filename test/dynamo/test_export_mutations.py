# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that torch._dynamo.export handles modules that assign to tensor attributes
# inside forward, plus cases where no such mutation happens. Each test builds a small module and
# compares the exported graph's result with eager execution. Importing defines one TestCase
# class only; the main guard runs the Dynamo test runner.
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.testing._internal.common_utils import IS_FBCODE


class MutationExportTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo TestCase holding export-versus-eager comparisons for modules that do or
    # do not mutate attributes. Stateless apart from two shared check helpers; each test defines
    # its own nested module class.
    def check_failure_on_export(self, mod, *args):
        # [FUNCTION DOC] HELPER: Asserts that exporting the given module with the given
        # positional arguments raises an AssertionError. Not used by any test in this file.
        with self.assertRaises(AssertionError):
            torch._dynamo.export(mod)(*args)

    def check_same_with_export(self, mod, arg):
        # [FUNCTION DOC] HELPER: Runs the module eagerly on arg, exports it with Dynamo export
        # using the same arg, runs the exported graph on arg, and asserts the two results are
        # equal. Fails on mismatch or if export raises.
        real_result = mod(arg)
        graph, _ = torch._dynamo.export(mod)(arg)
        result = graph(arg)
        self.assertEqual(result, real_result)

    def test_module_attribute_mutation_violation_positive_1(self):
        # [FUNCTION DOC] Module whose forward rebinds a tensor attribute to a float64 copy and
        # then sums input and attribute. Checks export succeeds and matches eager.
        # Mutating attribute with a Tensor type
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module holding a random 3x2 tensor attribute named a. Its
            # forward converts a to float64, stores it back on the module, and returns the sum
            # of the input plus the sum of the new attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the base module and sets attribute a to a
                # random 3x2 tensor.
                super().__init__()
                self.a = torch.randn(3, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reassigns attribute a to its float64 version (a side
                # effect on the module) and returns the sum of x plus the sum of a.
                self.a = self.a.to(torch.float64)
                return x.sum() + self.a.sum()

        self.check_same_with_export(Foo(), torch.randn(3, 2))

    def test_module_attribute_mutation_violation_negative_1(self):
        # [FUNCTION DOC] Module that only reads attribute a in forward, converting a copy to
        # float64 without storing it. Checks export succeeds and matches eager.
        # Mutating attribute with a Tensor type inside __init__ but
        # not in forward()
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module with a random 3x2 tensor attribute a; forward does
            # not modify it and sums x plus a float64 copy of a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets attribute a to a random 3x2 tensor.
                super().__init__()
                self.a = torch.randn(3, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sum of x plus the sum of a float64 copy of
                # a; the attribute itself is left unchanged.
                return x.sum() + self.a.to(torch.float64).sum()

        self.check_same_with_export(Foo(), torch.randn(3, 2))

    def test_module_attribute_mutation_violation_negative_2(self):
        # [FUNCTION DOC] Module that rebinds attribute a twice during construction (random
        # tensor, then float64 copy) but not in forward. Checks export matches eager.
        # Mutating attribute with a Tensor type inside __init__ twice
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module whose constructor leaves attribute a as a float64
            # tensor; forward only reads it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets attribute a to a random 3x2 tensor, then rebinds
                # it to its float64 conversion.
                super().__init__()
                self.a = torch.randn(3, 2)
                self.a = self.a.to(torch.float64)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sum of x plus the sum of attribute a.
                return x.sum() + self.a.sum()

        self.check_same_with_export(Foo(), torch.randn(3, 2))

    def test_module_attribute_mutation_violation_negative_3(self):
        # [FUNCTION DOC] Module whose forward only mutates a local Python integer, not any
        # attribute. Checks export succeeds and matches eager.
        # Mutating local variable inside forward()
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module with a random tensor attribute a and a forward
            # that uses a local integer variable.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets attribute a to a random 3x2 tensor.
                super().__init__()
                self.a = torch.randn(3, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Rebinds a local integer to five times itself and
                # returns the sum of x plus the sum of a plus that local value.
                b = 1
                b = b * 5
                return x.sum() + self.a.sum() + b

        self.check_same_with_export(Foo(), torch.randn(3, 2))

    @unittest.skipIf(IS_FBCODE, "Broken in fbcode")
    def test_module_attribute_mutation_violation_negative_4(self):
        # [FUNCTION DOC] Skipped when running in fbcode. Uses the attribute-mutating module
        # without export: runs it eagerly, then runs a compiled copy using the eager backend
        # with fullgraph, and asserts both results are equal.
        # Mutating attribute with a Tensor type
        # But not exporting but using eager mode as well as dynamo optimize mode
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Same attribute-mutating module as in the first test: forward
            # rebinds attribute a to a float64 copy and returns a sum.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets attribute a to a random 3x2 tensor.
                super().__init__()
                self.a = torch.randn(3, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Rebinds attribute a to its float64 copy and returns the
                # sum of x plus the sum of a.
                self.a = self.a.to(torch.float64)
                return x.sum() + self.a.sum()

        mod = Foo()
        arg = torch.randn(3, 2)
        real_result = mod(arg)
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        self.assertEqual(opt_mod(arg), real_result)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
