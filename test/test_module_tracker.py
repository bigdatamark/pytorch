# Owner(s): ["module: unknown"]
# [MODULE DOC] Tests for ModuleTracker from torch.utils.module_tracker: it records the stack
# of currently executing module names (the parents set, always containing Global) and
# whether execution is in backward (the is_bw flag). Covers nested module hierarchies,
# recursive modules, checkpoint usage and backward detection.

from copy import copy

import torch
from torch import nn
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TestCase,
    xfailIfTorchDynamo,
)
from torch.utils.checkpoint import checkpoint
from torch.utils.module_tracker import ModuleTracker


class TestModuleTracker(TestCase):
    # [CLASS DOC] Test case exercising ModuleTracker parent tracking and backward detection.
    # "https://github.com/pytorch/pytorch/issues/127112
    @xfailIfTorchDynamo
    def test_module_hierarchy(self):
        # [FUNCTION DOC] Builds a module with a plain child, a ModuleDict child and a
        # ModuleList child, each a small module that records a copy of the tracker parents
        # set and the is_bw flag during forward and from a gradient hook. Runs forward and
        # backward twice under the tracker and asserts the recorded forward parents follow
        # the call order (c.0, a, b.nest) with is_bw false, and backward parents in reverse
        # order with is_bw true. Skipped as an expected failure under TorchDynamo.
        seen_fw = []
        seen_bw = []

        class Foo(nn.Module):
            # [CLASS DOC] HELPER: Leaf module whose forward takes a dict with key a.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies relu in place to the tensor under key a,
                # appends a copy of the tracker parents and is_bw flag to the forward record
                # list, registers a gradient hook on the tensor that appends the same pair
                # to the backward record list, and returns a dict with key a holding the
                # matrix product of the tensor with itself.
                x = x["a"].relu_()
                seen_fw.append((copy(tracker.parents), tracker.is_bw))
                x.register_hook(
                    lambda grad: seen_bw.append((copy(tracker.parents), tracker.is_bw))
                )
                return {"a": torch.mm(x, x)}

        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Container with submodules a, b (ModuleDict with key nest)
            # and c (ModuleList of one element), all of the leaf type.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor; creates the three leaf submodule
                # holders a, b and c.
                super().__init__()
                self.a = Foo()
                self.b = nn.ModuleDict({"nest": Foo()})
                self.c = nn.ModuleList([Foo()])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the list child, then the plain child, then the
                # dict child on the data in that order and returns the final result.
                x = self.c[0](x)
                return self.b["nest"](self.a(x))

        mod = Mod()

        with ModuleTracker() as tracker:
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()

        self.assertEqual(
            seen_fw,
            [
                ({"Global", "Mod", "Mod.c.0"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b.nest"}, False),
                ({"Global", "Mod", "Mod.c.0"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b.nest"}, False),
            ],
        )

        self.assertEqual(
            seen_bw,
            [
                ({"Global", "Mod", "Mod.b.nest"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.c.0"}, True),
                ({"Global", "Mod", "Mod.b.nest"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.c.0"}, True),
            ],
        )

    @skipIfTorchDynamo("unexplained 3.13+ recursion error")
    def test_confused_hierarchy(self):
        # [FUNCTION DOC] Checks that the tracker does not fail with a module that calls
        # itself recursively once (first call re-enters, second call runs the inner Linear).
        # Runs forward and backward under the tracker both directly and through activation
        # checkpointing. No assertions beyond not raising. Skipped under TorchDynamo.
        class MyMod(nn.Module):
            # [CLASS DOC] HELPER: Module with an inner Linear and a ran flag used to make
            # forward re-enter.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Constructor; creates the inner Linear layer and
                # sets the ran flag false.
                super().__init__()
                self.inner = nn.Linear(2, 2)
                self.ran = False

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: On first call sets the ran flag and calls the
                # module itself again; on the nested call clears the flag and returns the
                # inner Linear output.
                if not self.ran:
                    self.ran = True
                    return self(inp)
                else:
                    self.ran = False
                    return self.inner(inp)

        mod = MyMod()
        inp = torch.rand(1, 2, requires_grad=True)

        # Should not fail
        with ModuleTracker():
            res = mod(inp)
            res.sum().backward()

        # Should not fail
        with ModuleTracker():
            res = checkpoint(lambda inp: mod(inp), inp)
            res.sum().backward()

    def test_bw_detection(self):
        # [FUNCTION DOC] Runs a Linear forward and backward under the tracker and, still
        # inside the tracker scope, asserts is_bw is false and parents equals just Global.
        mod = nn.Linear(2, 2)

        with ModuleTracker() as tracker:
            mod(torch.rand(2, requires_grad=True)).sum().backward()
            self.assertFalse(tracker.is_bw)
            self.assertEqual(tracker.parents, {"Global"})


if __name__ == "__main__":
    run_tests()
