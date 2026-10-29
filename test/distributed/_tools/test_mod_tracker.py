# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for ModTracker (torch.distributed._tools.mod_tracker): the set of parent
# module names and backward flag seen inside forward and backward hooks, backward detection,
# user-registered pre and post hooks for forward and backward (including after the module is
# deleted), and behavior under non-reentrant activation checkpointing. Several tests are expected
# failures under TorchDynamo.

from copy import copy

import torch
from torch.distributed._tools.mod_tracker import ModTracker
from torch.testing._internal.common_utils import run_tests, TestCase, xfailIfTorchDynamo
from torch.utils.checkpoint import checkpoint


class TestModTracker(TestCase):
    # [CLASS DOC] Single-process test case for ModTracker parent tracking and user hooks.
    # "https://github.com/pytorch/pytorch/issues/127112
    @xfailIfTorchDynamo
    def test_module_hierarchy(self):
        # [FUNCTION DOC] Runs a nested module whose children record a copy of the tracker's
        # parents and backward flag during forward, and again from a gradient hook during
        # backward, over two iterations under ModTracker. Asserts forward records show Global,
        # Mod and the active child path in call order (c.0, a, b.nest) with the flag false, and
        # backward records show the reverse order with the flag true. Expected failure under
        # Dynamo.
        seen_fw = []
        seen_bw = []

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Leaf module that applies in-place relu to the input mapping's
            # tensor, records the tracker state in the forward, registers a gradient hook
            # recording the state in backward, and returns a mapping with the matrix product of
            # the tensor with itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Takes a mapping with key a, applies in-place relu,
                # appends the current parents and backward flag to the forward log, adds a hook
                # that appends them to the backward log, and returns a mapping with the matrix
                # product.
                x = x["a"].relu_()
                seen_fw.append((copy(tracker.parents), tracker.is_bw))
                x.register_hook(
                    lambda grad: seen_bw.append((copy(tracker.parents), tracker.is_bw))
                )
                return {"a": torch.mm(x, x)}

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Container with child Foo modules held directly (a), in a
            # ModuleDict (b.nest) and in a ModuleList (c.0); forward calls c.0, then a, then
            # b.nest.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the three Foo children.
                super().__init__()
                self.a = Foo()
                self.b = torch.nn.ModuleDict({"nest": Foo()})
                self.c = torch.nn.ModuleList([Foo()])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Passes the input through c.0, then a, then b.nest and
                # returns the result.
                x = self.c[0](x)
                return self.b["nest"](self.a(x))

        mod = Mod()

        with ModTracker() as tracker:
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

    def test_bw_detection(self):
        # [FUNCTION DOC] Runs a linear layer forward and backward under ModTracker and checks
        # that after the backward call the tracker reports it is not in backward and only Global
        # is in the parents set.
        mod = torch.nn.Linear(2, 2)

        with ModTracker() as tracker:
            mod(torch.rand(2, requires_grad=True)).sum().backward()
            self.assertFalse(tracker.is_bw)
            self.assertEqual(tracker.parents, {"Global"})

    @xfailIfTorchDynamo
    def test_user_hooks(self):
        # [FUNCTION DOC] Registers user hooks (pre and post forward, pre and post backward) that
        # log hook name, the known qualified name of the module, whether it is among the tracker
        # parents, and the backward flag. Checks the exact log for forward and backward of a
        # small module, that registering hooks again raises AssertionError, and that when the
        # module is deleted before backward the backward hooks log a missing name and false
        # parent membership. Expected failure under Dynamo.
        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper holding one linear layer and applying in-place relu to
            # its output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer named foo.
                super().__init__()
                self.foo = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the in-place relu of the linear layer output.
                return self.foo(x).relu_()

        mt = ModTracker()
        test_op = []

        def hook(mod, hook_name):
            # [FUNCTION DOC] HELPER: Appends to the shared log a tuple of hook name, the
            # tracker's known qualified name for the module (None if the module is None), whether
            # that name is among current parents, and the backward flag.
            mfqn = mt.get_known_fqn(mod) if mod is not None else None
            test_op.append((hook_name, mfqn, mfqn in mt.parents, mt.is_bw))

        mod = Bar()

        mt.register_user_hooks(
            lambda m, inp: hook(m, "pre_fw"),
            lambda m, inp, op: hook(m, "post_fw"),
            lambda m, gop: hook(m, "pre_bw"),
            lambda m, ginp: hook(m, "post_bw"),
        )
        with mt:
            mod(torch.rand(10, 10, requires_grad=True)).sum().backward()
        expected_op = [
            ("pre_fw", "Bar", True, False),
            ("pre_fw", "Bar.foo", True, False),
            ("post_fw", "Bar.foo", True, False),
            ("post_fw", "Bar", True, False),
            ("pre_bw", "Bar", True, True),
            ("pre_bw", "Bar.foo", True, True),
            ("post_bw", "Bar", True, True),
            ("post_bw", "Bar.foo", True, True),
        ]
        self.assertEqual(test_op, expected_op)

        with self.assertRaises(AssertionError):
            mt.register_user_hooks(lambda x, y: x, None, None, None)

        test_op.clear()
        with mt:
            loss = mod(torch.rand(10, 10, requires_grad=True)).sum()
            del mod
            loss.backward()
        expected_op = [
            ("pre_fw", "Bar", True, False),
            ("pre_fw", "Bar.foo", True, False),
            ("post_fw", "Bar.foo", True, False),
            ("post_fw", "Bar", True, False),
            ("pre_bw", None, False, True),
            ("pre_bw", None, False, True),
            ("post_bw", None, False, True),
            ("post_bw", None, False, True),
        ]
        self.assertEqual(test_op, expected_op)

    @xfailIfTorchDynamo
    def test_ac(self):
        # [FUNCTION DOC] Checks the hook sequence when blocks after the first are activation
        # checkpointed: forward hooks run in order, and during backward the checkpointed layer's
        # forward hooks fire again (recomputation) while the tracker is already in backward mode.
        # Compares against an exact expected log. Expected failure under Dynamo.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Stack of linear layers with relu after each; when use_ac is
            # set, every layer after the first runs through non-reentrant checkpoint with RNG
            # state preserved.
            def __init__(self, n_layers: int, dim: int, use_ac: bool = False):
                # [FUNCTION DOC] HELPER: Builds the list of linear layers and stores the
                # checkpointing flag.
                super().__init__()
                self.linears = torch.nn.ModuleList()
                self.use_ac = use_ac
                for _ in range(n_layers):
                    self.linears.append(torch.nn.Linear(dim, dim))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each layer (checkpointed after the first when
                # enabled), raises AssertionError if the result is None, applies relu and returns
                # the final tensor.
                for i, block in enumerate(self.linears):
                    if i >= 1 and self.use_ac:
                        x = checkpoint(
                            block, x, preserve_rng_state=True, use_reentrant=False
                        )
                    else:
                        x = block(x)
                    if x is None:
                        raise AssertionError("Expected x to not be None")
                    x = torch.nn.functional.relu(x)
                return x

        bsz = 2
        dim = 8
        n_layers = 2
        test_op = []

        def hook(mod, mt, hook_name):
        # [FUNCTION DOC] HELPER: Appends to the shared log a tuple of hook name, the tracker's
        # known qualified name for the module (None if None), whether the name is among the
        # tracker's parents, and the backward flag.
            mfqn = mt.get_known_fqn(mod) if mod is not None else None
            test_op.append((hook_name, mfqn, mfqn in mt.parents, mt.is_bw))

        mt = ModTracker()
        mt.register_user_hooks(
            lambda m, i: hook(m, mt, "pre_fw"),
            lambda m, i, o: hook(m, mt, "post_fw"),
            lambda m, go: hook(m, mt, "pre_bw"),
            lambda m, gi: hook(m, mt, "post_bw"),
        )
        model = Foo(n_layers, dim, True)
        x = torch.randn(bsz, dim)
        with mt:
            model(x).sum().backward()

        expected_op = [
            ("pre_fw", "Foo", True, False),
            ("pre_fw", "Foo.linears.0", True, False),
            ("post_fw", "Foo.linears.0", True, False),
            ("pre_fw", "Foo.linears.1", True, False),
            ("post_fw", "Foo.linears.1", True, False),
            ("post_fw", "Foo", True, False),
            ("pre_bw", "Foo", True, True),
            ("pre_bw", "Foo.linears.1", True, True),
            ("pre_fw", "Foo.linears.1", True, True),
            ("post_fw", "Foo.linears.1", True, True),
            ("post_bw", "Foo.linears.1", True, True),
            ("pre_bw", "Foo.linears.0", True, True),
            ("post_bw", "Foo.linears.0", True, True),
            ("post_bw", "Foo", True, True),
        ]
        self.assertEqual(test_op, expected_op)


if __name__ == "__main__":
    run_tests()
