# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] Tests for TorchScript module freezing (torch.jit.freeze and the C++ freeze_module
# pass) and the optimizations built on it: attribute and submodule folding, preserved
# attributes, alias and mutation handling, conv/batchnorm and linear/batchnorm folding,
# mkldnn/oneDNN conversion, freezing of quantized modules, and conv-add-relu fusion. Importing
# the file probes for torchvision and defines skipIfNoTorchVision; running the file directly is
# rejected through raise_on_run_directly.

import io
import unittest
from itertools import product
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit._recursive import wrap_cpp_module
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_CUDNN, tf32_on_and_off
from torch.testing._internal.common_quantization import skipIfNoFBGEMM
from torch.testing._internal.common_quantized import override_quantized_engine
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_LINUX,
    raise_on_run_directly,
    set_default_dtype,
    skipCUDAMemoryLeakCheckIf,
    skipIfTorchDynamo,
    TEST_WITH_ROCM,
    xfailIf,
)
from torch.testing._internal.jit_utils import JitTestCase
from torch.utils import mkldnn as mkldnn_utils


try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


def removeExceptions(graph):
    # [FUNCTION DOC] HELPER: Deletes every node of kind prim::RaiseException from the given graph
    # (found with findAllNodes, which also searches nested blocks), mutating the graph in place.
    # Used so tests can compare graphs without exception-raising branches. Returns nothing.
    for n in graph.findAllNodes("prim::RaiseException"):
        n.destroy()


class TestFreezing(JitTestCase):
    # [CLASS DOC] Test suite (a JitTestCase) for torch.jit.freeze and torch._C._freeze_module. Most
    # tests script a small nn.Module, freeze it, and check which attributes or submodules remain
    # (via hasattr), check graph contents with FileCheck, and compare the frozen output against the
    # unfrozen scripted output.
    def test_freeze_module(self):
        # [FUNCTION DOC] Scripts a module with many attribute types (ints, floats, strings, lists,
        # tuples, dicts, tensors), freezes it with the C++ pass, round-trips it through an
        # in-memory save/load, and checks that every attribute has been removed from the loaded
        # module and that the forward output matches the unfrozen one.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module holding attributes of many types; its forward concatenates
            # the string forms of all of them. Comments in the constructor note which are expected
            # to fold.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates attributes a through tt covering scalars, strings
                # (including non-ASCII), lists, tuples, dicts of tensors, and nested tensor lists.
                super().__init__()
                self.a = 1  # folded
                self.b = 1.2  # folded
                self.c = "hello"  # folded
                self.c2 = "hi\xa1"  # not folded
                self.d = [1, 1]  # folded
                self.e = [1.0, 1.1]  # folded
                self.f = ["hello", "world"]  # folded
                self.f2 = [(1, "Over \u0e55\u0e57 57")]
                self.g = (
                    [1, 2],
                    3.2,
                    "4.4",
                    torch.tensor([5.5], requires_grad=True),
                )  # folded
                self.h = {"layer": [torch.tensor([7.7], requires_grad=True)]}
                self.h2 = {"layer\xb1": [torch.tensor([8.8], requires_grad=True)]}
                self.t = torch.tensor([1.2, 2.4], requires_grad=True)  # folded
                self.ts = [
                    torch.tensor([1.0, 2.0], requires_grad=True),
                    torch.tensor([3.0, 4.0], requires_grad=True),
                ]  # folded
                self.tt = [[torch.tensor([3.3, 2.3], requires_grad=True), None]]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns one string made by joining the string form of
                # every attribute.
                return (
                    str(self.a)
                    + str(self.b)
                    + self.c
                    + self.c2
                    + str(self.d)
                    + str(self.e)
                    + str(self.f)
                    + str(self.f2)
                    + str(self.g)
                    + str(self.h)
                    + str(self.h2)
                    + str(self.t)
                    + str(self.ts)
                    + str(self.tt)
                )

        m = torch.jit.script(M())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        m._c = torch._C._freeze_module(m._c)
        buffer = io.BytesIO()
        torch.jit.save(m._c, buffer)
        buffer.seek(0)
        m2 = torch.jit.load(buffer)
        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     tt = ...
        #   }
        #   ...
        # }
        self.assertFalse(m2._c.hasattr("a"))
        self.assertFalse(m2._c.hasattr("b"))
        self.assertFalse(m2._c.hasattr("c"))
        self.assertFalse(m2._c.hasattr("c2"))
        self.assertFalse(m2._c.hasattr("d"))
        self.assertFalse(m2._c.hasattr("e"))
        self.assertFalse(m2._c.hasattr("f"))
        self.assertFalse(m2._c.hasattr("f2"))
        self.assertFalse(m2._c.hasattr("g"))
        self.assertFalse(m2._c.hasattr("h"))
        self.assertFalse(m2._c.hasattr("h2"))
        self.assertFalse(m2._c.hasattr("t"))
        self.assertFalse(m2._c.hasattr("ts"))
        self.assertFalse(m2._c.hasattr("tt"))
        output_f = m2.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_submodule(self):
        # [FUNCTION DOC] Freezes a scripted module that has two submodules and a mutated attribute.
        # Checks that unmutated submodule sub1 and unmutated attribute a are folded away, while the
        # attribute b (assigned in forward) and submodule sub2 (which assigns its own b) are kept,
        # and that frozen output equals scripted output.
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Submodule with constant int attributes a and b; forward returns a
            # + b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets int attributes a=11 and b=2.
                super().__init__()
                self.a = 11
                self.b = 2

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sum of the two attributes without
                # mutating anything.
                return self.a + self.b

        class SubModule2(nn.Module):
            # [CLASS DOC] HELPER: Like the previous submodule, but forward overwrites attribute
            # b first.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets int attributes a=12 and b=2.
                super().__init__()
                self.a = 12
                self.b = 2

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Assigns 30 to attribute b (a mutation that prevents
                # folding b) and returns a + b.
                self.b = 30
                return self.a + self.b

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Top-level module owning sub1 and sub2 plus int attributes a
            # and b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two submodules and sets int attributes a=3
                # and b=4.
                super().__init__()
                self.sub1 = SubModule()
                self.sub2 = SubModule2()
                self.a = 3
                self.b = 4

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Overwrites b with 20, then returns sub1 output plus a plus
                # b plus sub2 output.
                self.b = 20
                return self.sub1(x) + self.a + self.b + self.sub2(x)

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        mf = torch.jit.freeze(m)

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     sub2 = ...
        #      b =
        #   }
        #   ...
        #   submodule {
        #     module m {
        #       attributes {
        #         sub2 = ...
        #         b =
        #       }
        #       ...
        #     }
        #   }
        # }
        mf = mf._c
        self.assertFalse(mf.hasattr("sub1"))
        self.assertFalse(mf.hasattr("a"))
        self.assertTrue(mf.hasattr("b"))
        self.assertTrue(mf.hasattr("sub2"))
        self.assertTrue(mf.sub2.hasattr("b"))  # verify b is preserved in sub2
        self.assertFalse(mf.sub2.hasattr("a"))  # verify a is removed in sub2
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_fork(self):
        # [FUNCTION DOC] Freezes a module that forks a submodule's forward and waits for the
        # result. Checks that after freezing, the folded tensor attributes a and b are not present
        # and that the frozen output equals the scripted output.
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Submodule with two 20x20 ones tensors; forward returns a * b + x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor attributes a and b filled with ones.
                super().__init__()
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a * b + x.
                return self.a * self.b + x

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module that runs its submodule both asynchronously (fork)
            # and directly.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the SubModule instance as attribute sub.
                super().__init__()
                self.sub = SubModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forks the submodule forward on x, calls the submodule
                # directly, waits for the future, and returns the sum of both results.
                fut = torch.jit._fork(self.sub.forward, x)
                y_hat = self.sub(x)
                y = torch.jit._wait(fut)
                return y_hat + y

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(20, 20)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        self.assertFalse(mf.hasattr("a"))
        self.assertFalse(mf.hasattr("b"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_nested_fork(self):
        # [FUNCTION DOC] Freezes a module with forks nested at two levels, where the top module
        # also assigns attribute d. Checks that a, b and c are folded away, the mutated d is kept,
        # and the frozen output equals the scripted output.
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Innermost submodule with ones tensors a and b; forward returns a
            # * b + x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor attributes a and b filled with ones.
                super().__init__()
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a * b + x.
                return self.a * self.b + x

        class SubModule2(nn.Module):
            # [CLASS DOC] HELPER: Middle module wrapping SubModule and its own ones tensor c; forks
            # the inner module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates submodule sub and tensor attribute c.
                super().__init__()
                self.sub = SubModule()
                self.c = torch.ones(20, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forks sub.forward, calls sub directly, waits, and returns
                # the sum of both results plus c.
                fut = torch.jit._fork(self.sub.forward, x)
                y_hat = self.sub(x)
                y = torch.jit._wait(fut)
                return y_hat + y + self.c

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Outer module that forks the middle module and mutates int
            # attribute d.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates submodule sub and int attribute d=1.
                super().__init__()
                self.sub = SubModule2()
                self.d = 1

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forks sub.forward, calls sub, waits, assigns d=2, and
                # returns the product of both results plus d.
                fut = torch.jit._fork(self.sub.forward, x)
                y_hat = self.sub(x)
                y = torch.jit._wait(fut)
                self.d = 2
                return y_hat * y + self.d

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(20, 20)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)
        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        self.assertFalse(mf.hasattr("a"))
        self.assertFalse(mf.hasattr("b"))
        self.assertFalse(mf.hasattr("c"))
        self.assertTrue(mf.hasattr("d"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_fork2(self):
        # [FUNCTION DOC] Freezes a module that forks a scripted free function with a module
        # attribute. Records the current behavior: attribute a (passed to the fork) is kept because
        # alias analysis conservatively treats fork arguments as possibly mutated, while b is
        # folded away.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted free function returning its input times 2.
            return x * 2

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module with ones tensors a and b; forks foo on a and calls foo
            # on b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates 20x20 ones tensor attributes a and b.
                super().__init__()
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forks foo on a, calls foo on b directly, waits, and
                # returns the sum.
                fut = torch.jit._fork(foo, self.a)
                y_hat = foo(self.b)
                y = torch.jit._wait(fut)
                return y_hat + y

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     self.a = ...
        #     self.b = ..
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        # TODO:  Although there are no mutation, the alias analysis
        # conservatively assumes there is a mutation because attributes are
        # passed to fork subgraph. both 'a' and 'b' are preserved.
        self.assertTrue(mf.hasattr("a"))
        self.assertFalse(mf.hasattr("b"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_fork_calling_module_method(self):
        # [FUNCTION DOC] Freezes a module whose forward forks one exported method and calls
        # another. Checks that attribute a is folded away while b is kept (the forked method
        # receives it), and that frozen output equals scripted output.
        @torch.jit.script
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Scripted free function returning the product of its two inputs
            # (not actually used by the module below).
            return x * y

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module with ones tensors a and b and exported methods foo
            # and bar.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates 20x20 ones tensor attributes a and b.
                super().__init__()
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            @torch.jit.export
            def foo(self, x):
                # [FUNCTION DOC] HELPER: Exported method returning x times attribute a.
                return x * self.a

            @torch.jit.export
            def bar(self, x):
                # [FUNCTION DOC] HELPER: Exported method returning x times attribute b.
                return x * self.b

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forks foo on b, calls bar on a, waits, and returns
                # the sum.
                fut = torch.jit._fork(self.foo, self.b)
                y_hat = self.bar(self.a)
                y = torch.jit._wait(fut)
                return y_hat + y

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)
        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     self.b = ..
        #   }
        #   ...
        # TODO:  Although there are no mutation, the alias analysis
        # conservatively assumes there is a mutation because attributes are
        # passed to fork subgraph. 'b' is preserved.
        self.assertFalse(mf.hasattr("a"))
        self.assertTrue(mf.hasattr("b"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_sharedclasstype(self):
        # [FUNCTION DOC] Freezes a module where sub1 and sub2.sub share one class type but are
        # mutated differently. Verifies the per-instance results: sub1 keeps both a and b, sub2
        # keeps only its sub submodule (b folded) whose a and b are kept, top-level a is folded,
        # and outputs match.
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Submodule with two one-element tensors and methods that
            # mutate them.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor attributes a=1.1 and b=2.2.
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a + b.
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                # [FUNCTION DOC] HELPER: Exported; adds 10 in place to the first element of a and
                # returns b.
                self.a[0] += 10
                return self.b

            @torch.jit.export
            def modify_b(self, x):
                # [FUNCTION DOC] HELPER: Exported; adds 20 in place to the first element of b and
                # returns a.
                self.b[0] += 20
                return self.a

        class SubModule2(nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a SubModule and a tensor b; forward
            # uses modify_b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates submodule sub and tensor attribute b=3.3.
                super().__init__()
                self.sub = SubModule()
                self.b = torch.tensor([3.3])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls sub.modify_b and returns the result plus b.
                y = self.sub.modify_b(x)
                return y + self.b

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Top module with sub1 (same type as sub2.sub), sub2 and tensor a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates sub1, sub2 and tensor attribute a=4.4.
                super().__init__()
                self.sub1 = SubModule()  # sub1 and sub2.sub shared same class type.
                self.sub2 = SubModule2()
                self.a = torch.tensor([4.4])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls sub1.modify_a, then returns sub2 output plus that
                # result plus a.
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z + self.a

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)

        # Checking if  Frozen module looks as  below
        # module mf {
        #   attributes {
        #     sub1 = ...
        #     sub2 = ...
        #   }
        #   ...
        #   submodules {
        #     module sub1 {
        #       attributes {
        #         a = ...
        #         b = ...
        #       }
        #       ...
        #     }
        #     module sub2 {
        #       attributes {
        #         sub = ...
        #       }
        #       ...
        #       submodule {
        #         module sub {
        #           attributes {
        #             a = ...
        #             b = ...
        #           }
        #           ...
        #         }
        #       }
        #     }
        #   }
        # }

        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertTrue(mf.sub1.hasattr("b"))
        self.assertFalse(mf.hasattr("a"))
        self.assertTrue(mf.hasattr("sub2"))
        self.assertTrue(mf.sub2.hasattr("sub"))
        self.assertFalse(mf.sub2.hasattr("b"))
        self.assertTrue(mf.sub2.sub.hasattr("a"))
        self.assertTrue(mf.sub2.sub.hasattr("b"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_nestedaliasing(self):
        # [FUNCTION DOC] Freezes a module in which one Sub instance is shared between sub1 and
        # sub2.sub. Checks that alias detection keeps sub1.a and sub2.sub.a, folds b, and that the
        # frozen output equals the scripted output.
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Submodule with tensors a and b and exported methods that
            # overwrite them.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor attributes a=1.1 and b=2.2.
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a + b.
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                # [FUNCTION DOC] HELPER: Exported; writes 10 into the first element of a and
                # returns b.
                self.a[0] = 10
                return self.b

            @torch.jit.export
            def modify_b(self, x):
                # [FUNCTION DOC] HELPER: Exported; writes 20 into the first element of b and
                # returns a.
                self.b[0] = 20
                return self.a

        Sub = SubModule()

        class SubModule2(nn.Module):
            # [CLASS DOC] HELPER: Module holding the shared Sub instance; forward returns its
            # attribute a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the shared Sub instance as sub.
                super().__init__()
                self.sub = Sub  # aliasing

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the tensor attribute a of the shared submodule.
                return self.sub.a

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Top module referencing the shared Sub as sub1 and also
            # owning SubModule2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores Sub as sub1 and creates sub2.
                super().__init__()
                self.sub1 = Sub  # aliasing
                self.sub2 = SubModule2()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls sub1.modify_a and returns the sub2 output plus
                # the result.
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z

        m = torch.jit.script(TestModule())
        m.eval()
        mf = torch._C._freeze_module(m._c)
        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertFalse(mf.sub1.hasattr("b"))
        self.assertTrue(mf.hasattr("sub2"))
        self.assertTrue(mf.sub2.hasattr("sub"))
        self.assertTrue(
            mf.sub2.sub.hasattr("a")
        )  # Freezing detects that self.sub2.sub.a and self.sub1.a are alias
        self.assertFalse(mf.sub2.sub.hasattr("b"))
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    # FIXME: JIT is not honoring aliasing. 'Sub' module is copied. As a result
    # Eager and Script modules produce different output.
    def test_freeze_module_with_nestedaliasingscalar(self):
        # [FUNCTION DOC] Scalar-attribute variant of the aliasing test. Scripting copies the shared
        # submodule, so eager and scripted outputs differ (asserted unequal); sub1 keeps a, sub2 is
        # fully folded, and the frozen output equals the scripted output.
        # KNOWN ISSUE: the FIXME above the test says JIT does not honor module aliasing.
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Submodule with float attributes a and b and methods that
            # reassign them.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets float attributes a=1.1 and b=2.2.
                super().__init__()
                self.a = 1.1
                self.b = 2.2

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a + b.
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                # [FUNCTION DOC] HELPER: Exported; assigns 10.0 to a and returns b.
                self.a = 10.0
                return self.b

            @torch.jit.export
            def modify_b(self, x):
                # [FUNCTION DOC] HELPER: Exported; assigns 20.0 to b and returns a.
                self.b = 20.0
                return self.a

        Sub = SubModule()

        class SubModule2(nn.Module):
            # [CLASS DOC] HELPER: Module holding the shared Sub instance; forward returns its
            # attribute a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the shared Sub instance as sub.
                super().__init__()
                self.sub = Sub  # aliasing

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the float attribute a of the submodule.
                return self.sub.a

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Top module referencing the shared Sub as sub1 and
            # owning SubModule2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores Sub as sub1 and creates sub2.
                super().__init__()
                self.sub1 = Sub  # aliasing
                self.sub2 = SubModule2()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls sub1.modify_a and returns the sub2 output plus
                # the result.
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z

        m = TestModule()
        ms = torch.jit.script(m)
        ms.eval()
        mf = torch._C._freeze_module(ms._c)
        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertFalse(mf.sub1.hasattr("b"))
        # sub2 is fully folded because self.sub1 and self.sub2.sub are not alias (Scripting bug)
        self.assertFalse(mf.hasattr("sub2"))
        input = torch.randn(2, 2)
        output = m.forward(input)
        output_s = ms.forward(input)
        output_f = mf.forward(input)
        # Should be equal
        self.assertNotEqual(output, output_s)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_preserve_sub_module(self):
        # [FUNCTION DOC] Freezes with sub1 listed as a preserved attribute. Checks that sub1 is
        # kept whole (both a and b) while the identical sub2 is folded away entirely, and that
        # outputs match.
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Submodule with a tensor a and float b; forward returns a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets tensor attribute a=1.1 and float attribute b=2.2.
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = 2.2

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the tensor attribute a.
                return self.a

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module with two separate SubModule instances sub1 and sub2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates sub1 and sub2.
                super().__init__()
                self.sub1 = SubModule()  # aliasing
                self.sub2 = SubModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns sub2 output plus sub1 output.
                return self.sub2(x) + self.sub1(x)

        m = TestModule()
        ms = torch.jit.script(m)
        ms.eval()
        mf = torch._C._freeze_module(ms._c, ["sub1"])

        # Test that 'sub1' is preserved entirely and 'sub2' is completely folded
        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertTrue(mf.sub1.hasattr("b"))
        self.assertFalse(mf.hasattr("sub2"))
        input = torch.randn(2, 2)
        output_s = ms.forward(input)
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_preserve_sub_module_and_mutation(self):
        # [FUNCTION DOC] Freezes with sub1 preserved where the submodule forward mutates its
        # tensor. Checks that sub1 keeps a and b (even though b is unused), sub2 is also kept
        # because of the mutation, and outputs match.
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Submodule with tensor a and float b whose forward mutates a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets tensor attribute a=1.1 and float attribute b=2.2.
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = 2.2

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Writes 3.3 into the first element of a and returns a.
                self.a[0] = 3.3
                return self.a

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module with two separate SubModule instances sub1 and sub2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates sub1 and sub2.
                super().__init__()
                self.sub1 = SubModule()  # aliasing
                self.sub2 = SubModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns sub2 output plus sub1 output.
                return self.sub2(x) + self.sub1(x)

        m = TestModule()
        ms = torch.jit.script(m)
        ms.eval()
        mf = torch._C._freeze_module(ms._c, ["sub1"])

        # Test that be both sub1 and sub1 are preserved and 'b' is preserved
        # even if it is not used. To fulfill user request to preserve 'sub1'
        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertTrue(mf.sub1.hasattr("b"))
        self.assertTrue(mf.hasattr("sub2"))
        self.assertTrue(mf.sub2.hasattr("a"))
        self.assertTrue(mf.sub2.hasattr("b"))
        input = torch.randn(2, 2)
        output_s = ms.forward(input)
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_helperfunction(self):
        # [FUNCTION DOC] Freezes a module whose forward calls a private helper method. Checks that
        # sub and a are folded while the reassigned b remains, and that the helper method is no
        # longer present on the frozen module (accessing it raises AttributeError with the
        # expected text).
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Submodule with int attributes a and b; forward returns their sum.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets int attributes a=11 and b=2.
                super().__init__()
                self.a = 11
                self.b = 2

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a + b.
                return self.a + self.b

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module with a submodule, ints a and b, and a helper
            # method _forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates submodule sub and int attributes a=3 and b=4.
                super().__init__()
                self.sub = SubModule()
                self.a = 3
                self.b = 4

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reassigns b to 20 and returns the helper result plus a
                # plus b.
                self.b = 20
                return self._forward(x) + self.a + self.b

            def _forward(self, x):
                # [FUNCTION DOC] HELPER: Helper method that returns the output of the submodule.
                return self.sub(x)

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        mf = torch._C._freeze_module(m._c)
        self.assertFalse(mf.hasattr("sub"))
        self.assertFalse(mf.hasattr("a"))
        self.assertTrue(mf.hasattr("b"))
        with self.assertRaisesRegex(
            AttributeError, "TestModule (.*) does not have a field with name '_forward'"
        ):
            mf._forward(x)  # noqa: F821

    def test_freeze_module_with_inplace_mutable(self):
        # [FUNCTION DOC] Freezes a ScriptModule whose method appends to a list attribute in a loop.
        # Checks that the mutable list attribute a is preserved and that calling the frozen module
        # gives the list extended once by the unfrozen call and once by the frozen call.
        class FreezeMe(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule with list attribute a that its script
            # method extends.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates list attribute a holding two ints.
                super().__init__()
                self.a = [11, 22]

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Appends the values 0, 1, 2 to the list attribute in a loop
                # and returns the list.
                for i in range(3):
                    self.a.append(i)
                return self.a

        m = FreezeMe()
        m.eval()
        m_f = torch._C._freeze_module(m._c)
        self.assertTrue(m_f.hasattr("a"))
        m.forward(torch.tensor([3]))
        out = m_f.forward(torch.tensor([5]))
        expected = [11, 22, 0, 1, 2, 0, 1, 2]
        self.assertEqual(out, expected)

    # Mutable attributes
    def test_freeze_module_with_mutable_list(self):
        # [FUNCTION DOC] Freezes a module with a list attribute modified before and after
        # scripting. Checks that the frozen module has its own copy: appends to the scripted module
        # after freezing do not change the frozen output, which holds only the earlier values.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with a list attribute a that forward returns.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates list attribute a with two ints.
                super().__init__()
                self.a = [1, 2]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the list attribute a.
                return self.a

        m = FreezeMe()
        m.eval()
        m.a.append(3)
        m_s = torch.jit.script(m)
        v = m_s.a
        v.append(4)
        m_s.a = v
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        # Post-freezing mutating m_s.a  does not affect m_f (m_f has its own copy).
        v = m_s.a
        v.append(5)
        m_s.a = v
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(torch.tensor([5]))
        expected = [1, 2, 3, 4]
        self.assertEqual(out, expected)

    def test_freeze_module_with_mutable_dict(self):
        # [FUNCTION DOC] Freezes a module with a dict attribute that an exported method mutates.
        # Checks that the dict attribute is folded away and that the frozen output keeps the
        # contents at freeze time, unaffected by later mutation of the scripted module.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with a string dict attribute a, returned by forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates dict attribute a with one entry.
                super().__init__()
                self.a = {"layer": "4"}

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the dict attribute a.
                return self.a

            @torch.jit.export
            def modify_a(self, x):
                # [FUNCTION DOC] HELPER: Exported; appends a character to the layer entry of the
                # dict and returns the dict.
                self.a["layer"] = self.a["layer"] + "1"
                return self.a

        m = FreezeMe()
        m.eval()
        m.a["layer2"] = "3"
        m_s = torch.jit.script(m)
        t = torch.tensor(5)
        m_s.modify_a(t)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        m.a["layer2"] += "2"
        m_s.modify_a(t)
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(t)
        expected = {"layer": "411", "layer2": "3"}
        self.assertEqual(out, expected)

    def test_freeze_module_with_mutable_tensor(self):
        # [FUNCTION DOC] Freezes a module with a tensor attribute modified in place beforehand.
        # Records that, unlike lists, later in-place edits of the scripted tensor still show up in
        # the frozen module because the tensor is shared (see the FIXME in the body).
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with a three-element float tensor attribute a returned
            # by forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor attribute a.
                super().__init__()
                self.a = torch.tensor([1.0, 2.0, 3.0])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the tensor attribute a.
                return self.a

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.a[1] += 3.0
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        # Post-freezing tensor attribute mutations affect m_f.
        # FIXME: deep copy all folded attributes so that m_f has full ownership.
        m_s.a[0] += 5.0
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(torch.tensor([5]))
        expected = [6.0, 5.0, 3.0]
        self.assertEqual(out, expected)

    def test_freeze_module_with_tuple(self):
        # [FUNCTION DOC] Freezes a module with a tuple attribute holding a tensor and a string,
        # where forward conditionally mutates the tensor. Checks the attribute is folded and the
        # frozen output equals the output recorded before freezing.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with tuple attribute a (tensor, string).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tuple attribute a.
                super().__init__()
                self.a = (torch.tensor([1, 2, 3, 4, 5, 6]), "hi")

            def forward(self, x):
                # [FUNCTION DOC] HELPER: If the first input element is 2.0, writes 10 to the first
                # element of the tensor in the tuple; returns the sum of that tensor.
                if x[0] == 2.0:
                    self.a[0][0] = 10
                return self.a[0].sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([2.0])
        expected = m_s.forward(inp)
        m_s.a[0][0] = 1
        m_f = torch._C._freeze_module(m_s._c)
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_tensor(self):
        # [FUNCTION DOC] Freezes a module whose forward mutates a view of its tensor attribute.
        # Checks the attribute is kept (due to the mutation) and that the frozen output matches the
        # expected result after the test offsets the tensor in place.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with integer tensor attribute a mutated through a view
            # in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor attribute a with six values.
                super().__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 10 in place to the first element via a 2x3 view and
                # returns the sum of the tensor.
                x = self.a.view(2, 3)
                x[0][0] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        m_f.a[0] -= 10
        out = m_f.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_list(self):
        # [FUNCTION DOC] Freezes a module with a list-of-tensor attribute mutated in forward.
        # Checks that the attribute is folded away and that the frozen output matches the earlier
        # output even though forward mutated the tensor and the test restored it before freezing.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with a list attribute holding one integer tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates list attribute a with one tensor.
                super().__init__()
                self.a = [torch.tensor([1, 2, 3, 4, 5, 6])]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 10 in place to the second tensor element and returns
                # the sum.
                self.a[0][1] += 10
                return self.a[0].sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        m_s.a[0][1] -= 10
        m_f = torch._C._freeze_module(m_s._c)
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_tensor_attr(self):
        # [FUNCTION DOC] Freezes a module where attribute b is a view of tensor attribute a and
        # forward mutates the view. Checks that a is kept and the output reflects the in-place
        # update (sum 51).
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with tensor a and attribute b that aliases a through
            # a view.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor a and view attribute b.
                super().__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                self.b = self.a.view(2, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 10 in place to the second row of b and returns the
                # sum of a.
                self.b[1] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        inp = torch.tensor([5])
        out = m_f.forward(inp)
        expected = torch.tensor(51)  # 1+2+3+14+15+16
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_tensor_attr2(self):
        # [FUNCTION DOC] Freezes a module whose dict, list-in-tuple and tuple attributes contain
        # views of attribute a. Expects the freeze call to raise RuntimeError saying that the
        # module contains overlapping attribute values.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with tensor a and attributes b, c, d containing views
            # of a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor a and nested containers b, c, d holding
                # views of a.
                super().__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                self.b = {"layer": ([self.a.view(2, 3), torch.tensor([10])], 20)}
                self.c = ([self.a.view(2, 3), torch.tensor([10])], 20)
                self.d = (self.a.view(2, 3), 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 10 in place to the first row of the view in d and
                # returns the sum of a.
                self.d[0][0] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        with self.assertRaisesRegex(
            RuntimeError, "module contains attributes values that overlaps"
        ):
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_with_aliased_tensor_attr3(self):
        # [FUNCTION DOC] Freezes a module whose list attribute b contains the same tensor object as
        # attribute a. Checks that both a and b are preserved and the frozen output matches the
        # scripted output plus the in-place increment.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with tensor a and list b containing a itself and
            # another tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor a and list b holding a and a second tensor.
                super().__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                self.b = [self.a, torch.tensor([10])]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 10 in place to the second element of a and returns
                # the sum of the tensor stored first in b.
                self.a[1] += 10
                return self.b[0].sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        self.assertTrue(m_f.hasattr("b"))
        out = m_f.forward(inp)
        expected += 10  # account for  self.a += 10.
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_tensor_attr4(self):
        # [FUNCTION DOC] Variant where list b contains tensor a and forward mutates it through b,
        # and the test also changes a before freezing. Expects the freeze call to raise
        # RuntimeError about overlapping attribute values.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with tensor a and list b containing a and another tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor a and list b holding a and a second tensor.
                super().__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                self.b = [self.a, torch.tensor([10])]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 10 in place to the first element via b and returns
                # the sum of a.
                self.b[0][0] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        m_s.a[0] -= 10
        with self.assertRaisesRegex(
            RuntimeError, "module contains attributes values that overlaps"
        ):
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_with_overlapping_attrs(self):
        # [FUNCTION DOC] Builds a module whose list and tuple attributes hold two different views
        # of one shared tensor defined outside the class. Expects the freeze call to raise
        # RuntimeError about overlapping attribute values.
        a = torch.tensor([1, 2, 3, 4, 5, 6])

        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with attributes b and c holding differently shaped views
            # of the same outer tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates list b and tuple c containing views of the
                # shared tensor.
                super().__init__()
                self.b = [a.view(3, 2), torch.tensor([10])]
                self.c = (20, a.view(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 10 in place to an element via b and returns the sum
                # of the view in c.
                self.b[0][0] += 10
                return self.c[1].sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        a[0] -= 10
        with self.assertRaisesRegex(
            RuntimeError, "module contains attributes values that overlaps"
        ):
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_with_aliased_attr(self):
        # [FUNCTION DOC] Freezes a module whose list attribute is aliased by b and by a tuple c.
        # Records that scripting copies the aliased lists, so a and c are folded (asserted absent)
        # and the frozen output matches the scripted output.
        # KNOWN ISSUE: the FIXME in the body says a should be preserved; the assertion is inverted
        # because scripting copies the list when assigning b.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with list a, alias b of a, and tuple c containing a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates list a, alias b, and tuple c holding a and an int.
                super().__init__()
                self.a = [1, 2, 3, 4, 5, 6]
                self.b = self.a
                self.c = (self.a, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 10 to the second element via b and returns the
                # strings of a and c joined.
                self.b[1] += 10
                return str(self.a) + str(self.c)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        # FIXME: It should be assertTrue. Currently scripting is making a copy for setting self.b (see #33034)
        self.assertFalse(m_f.hasattr("a"))
        self.assertFalse(m_f.hasattr("c"))
        inp = torch.tensor([5])
        out = m_f.forward(inp)
        expected = m_s.forward(inp)
        self.assertEqual(out, expected)

    # Check attribute a is preserved. Alias analysis detects that 'a' has output writers.
    # In this example, 'a' is not mutated. However, we do not track which sub
    # values of a composite ivalue is mutated.
    def test_freeze_module_with_aliased_attr2(self):
        # [FUNCTION DOC] Freezes a module where forward stores list a inside a tuple attribute and
        # appends to another list. Checks that a is preserved, because alias analysis cannot tell
        # which sub-value of a composite is written, and that outputs match the eager module.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with list a and tuple b of lists.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates list a and tuple attribute b.
                super().__init__()
                self.a = [1, 2, 3, 4, 5, 6]
                self.b = ([11], [10])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Stores a in the tuple attribute b, appends 7 to the other
                # list of the tuple, and returns the strings of both lists.
                v = self.a
                self.b = (v, [12])
                v2 = self.b[1]
                v2.append(7)
                return str(v) + str(v2)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        inp = torch.tensor([5])
        out = m_f.forward(inp)
        expected = m.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_attr3(self):
        # [FUNCTION DOC] Freezes a module whose forward appends to a list reached through a local
        # tuple that contains attribute a. Checks that a is preserved and outputs match eager.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with list a and tuple b of lists.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates list a and tuple attribute b.
                super().__init__()
                self.a = [1, 2, 3, 4, 5, 6]
                self.b = ([11], [10])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Wraps a in a local tuple, appends 7 to the list via the
                # tuple, and returns the string of a.
                v = self.a
                v2 = (v, [12])
                v3 = v2[0]
                v3.append(7)
                return str(self.a)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        inp = torch.tensor([5])
        out = m_f.forward(inp)
        expected = m.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_return_self(self):
        # [FUNCTION DOC] Scripts a module whose forward returns the module itself and expects the
        # freeze call to raise RuntimeError saying a module that returns itself cannot be frozen.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with tensor attribute a whose forward returns self.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor attribute a.
                super().__init__()
                self.a = torch.tensor([1.0, 2.0, 3.0])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the module itself.
                return self

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        with self.assertRaisesRegex(
            RuntimeError, "attempted to freeze a module that return itself"
        ):
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_inlining(self):
        # [FUNCTION DOC] Freezes a module holding an instance of a scripted class and prints it.
        # Checks that the object constant in the graph is non-holding (inlined), before and after a
        # save/load round trip.
        @torch.jit.script
        class Obj:
            # [CLASS DOC] HELPER: Scripted class with int fields x and y.
            def __init__(self, x: int, y: int):
                # [FUNCTION DOC] HELPER: Stores the two ints as fields.
                self.x = x
                self.y = y

        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Module holding a scripted class instance as attribute obj.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Obj instance with x=2 and y=3.
                super().__init__()
                self.obj = Obj(2, 3)

            def forward(self, i: int):
                # [FUNCTION DOC] HELPER: Prints the object and returns the integer argument.
                print(self.obj)
                return i

        mod = torch.jit.freeze(torch.jit.script(Mod().eval()))
        obj = mod.graph.findNode("prim::Constant")
        self.assertTrue(torch._C._jit_object_is_non_holding(obj))

        buffer = io.BytesIO()
        torch.jit.save(mod, buffer)
        buffer.seek(0)

        loaded = torch.jit.load(buffer)
        obj = mod.graph.findNode("prim::Constant")
        self.assertTrue(torch._C._jit_object_is_non_holding(obj))

    def test_freeze_module_return_sub_module(self):
        # [FUNCTION DOC] Freezes a module whose forward returns a submodule. Checks that the
        # submodule attribute conv1 is retained in the frozen module.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with a Conv2d submodule whose forward returns
            # that submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution submodule conv1.
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the submodule itself.
                return self.conv1

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("conv1"))

    def test_freeze_module_no_forward(self):
        # [FUNCTION DOC] Freezes (C++ pass) a module that has no forward method but an exported
        # foo, passing foo as preserved attribute, and checks the frozen foo equals the
        # scripted foo.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with a Linear submodule and exported method foo only.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear submodule lin.
                super().__init__()
                self.lin = nn.Linear(10, 1)

            @torch.jit.export
            def foo(self, x):
                # [FUNCTION DOC] HELPER: Exported; applies the linear layer to the input.
                return self.lin(x)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c, preservedAttrs=["foo"])
        input = torch.ones(10)
        self.assertEqual(m_s.foo(input), m_f.foo(input))

    def test_freeze_no_forward(self):
        # [FUNCTION DOC] Same as the previous test but through the public torch.jit.freeze API with
        # preserved_attrs naming the exported method foo; checks output equality.
        class FreezeMe(nn.Module):
            # [CLASS DOC] HELPER: Module with a Linear submodule and exported method foo only.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear submodule lin.
                super().__init__()
                self.lin = nn.Linear(10, 1)

            @torch.jit.export
            def foo(self, x):
                # [FUNCTION DOC] HELPER: Exported; applies the linear layer to the input.
                return self.lin(x)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch.jit.freeze(m_s, preserved_attrs=["foo"])
        input = torch.ones(10)
        self.assertEqual(m_s.foo(input), m_f.foo(input))

    def test_freeze_module_in_training_mode(self):
        # [FUNCTION DOC] Freezes a CNN in train mode, checking that training flags are dropped from
        # conv and linear submodules but kept for dropout submodules, with parameters preserved.
        # Then freezes in eval mode and checks everything is folded, state_dict is unavailable
        # (raises AttributeError), a save/load graph has no GetAttr nodes, and the
        # preserveParameters option keeps conv and linear parameters.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Small CNN with two convs, two Dropout2d layers and two
            # linear layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the conv, dropout and linear submodules.
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout2d(0.25)
                self.dropout2 = nn.Dropout2d(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs conv, relu, conv, max pool, dropout, flatten, linear,
                # relu, dropout, linear, then returns log softmax over dimension 1.
                x = self.conv1(x)
                x = nn.functional.relu(x)
                x = self.conv2(x)
                x = nn.functional.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = nn.functional.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                output = nn.functional.log_softmax(x, dim=1)
                return output

        model = torch.jit.script(Net())
        model.train()
        mTrain_freezed = torch._C._freeze_module(model._c)
        # verify mTrain_freezed looks exactly as:
        # module {
        #   attributes {
        #     conv1 = ...
        #     conv2 = ...
        #     dropout1 = ...
        #     dropout2 = ...
        #     fc1 = ...
        #     fc2 = ...
        #   }
        #   ...
        #   submodules {
        #     module conv1 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        #     module conv2 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        #     module dropout1 {
        #       attributes {
        #          training = ...
        #       }
        #       ...
        #     }
        #     module dropout2 {
        #       attributes {
        #          training = ...
        #       }
        #       ...
        #     }
        #     module fc1 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        #     module fc2 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        self.assertFalse(mTrain_freezed.hasattr("training"))
        self.assertTrue(mTrain_freezed.hasattr("conv1"))
        self.assertFalse(mTrain_freezed.conv1.hasattr("training"))
        self.assertTrue(mTrain_freezed.conv1.hasattr("weight"))
        self.assertTrue(mTrain_freezed.conv1.hasattr("bias"))
        self.assertTrue(mTrain_freezed.hasattr("conv2"))
        self.assertFalse(mTrain_freezed.conv2.hasattr("training"))
        self.assertTrue(mTrain_freezed.conv2.hasattr("weight"))
        self.assertTrue(mTrain_freezed.conv2.hasattr("bias"))
        self.assertTrue(mTrain_freezed.hasattr("dropout1"))
        self.assertTrue(mTrain_freezed.dropout1.hasattr("training"))
        self.assertTrue(mTrain_freezed.hasattr("dropout2"))
        self.assertTrue(mTrain_freezed.dropout2.hasattr("training"))
        self.assertTrue(mTrain_freezed.hasattr("fc1"))
        self.assertTrue(mTrain_freezed.fc1.hasattr("weight"))
        self.assertTrue(mTrain_freezed.fc1.hasattr("bias"))
        self.assertTrue(mTrain_freezed.hasattr("fc2"))
        self.assertTrue(mTrain_freezed.fc2.hasattr("weight"))
        self.assertTrue(mTrain_freezed.fc2.hasattr("bias"))
        model.eval()
        mEval_freezed = torch._C._freeze_module(model._c)
        self.assertFalse(mEval_freezed.hasattr("conv1"))
        self.assertFalse(mEval_freezed.hasattr("conv2"))
        self.assertFalse(mEval_freezed.hasattr("dropout1"))
        self.assertFalse(mEval_freezed.hasattr("training"))
        self.assertFalse(mEval_freezed.hasattr("fc1"))
        self.assertFalse(mEval_freezed.hasattr("dropout2"))
        self.assertFalse(mEval_freezed.hasattr("fc2"))
        with self.assertRaisesRegex(
            AttributeError, "does not have a field with name 'state_dict'"
        ):
            print(mEval_freezed.state_dict())
        buffer = io.BytesIO()
        torch.jit.save(mEval_freezed, buffer)
        buffer.seek(0)
        m = torch.jit.load(buffer)
        FileCheck().check_not("GetAttr[name=").run(m._c._get_method("forward").graph)
        m2 = torch._C._freeze_module(model._c, preserveParameters=True)
        self.assertTrue(m2.hasattr("conv1"))
        self.assertTrue(m2.hasattr("conv2"))
        self.assertFalse(m2.hasattr("dropout1"))
        self.assertFalse(m2.hasattr("training"))
        self.assertTrue(m2.hasattr("fc1"))
        self.assertFalse(m2.hasattr("dropout2"))
        self.assertTrue(m2.hasattr("fc2"))

    def test_freeze_module_detach_gradient(self):
        # [FUNCTION DOC] Freezes a scripted Conv2d. Checks that the original and scripted weights
        # keep requires_grad, the frozen module has no weight attribute, and that a later in-place
        # change of the scripted weight changes the scripted output and the frozen output alike.
        # KNOWN ISSUE: the FIXME in the body notes the frozen module is mutated from outside.
        mod = nn.Conv2d(8, 3, 4, 2, 1)
        self.assertTrue(mod.weight.requires_grad)
        smod = torch.jit.script(mod)
        smod.eval()
        fmod = torch._C._freeze_module(smod._c)
        self.assertTrue(mod.weight.requires_grad)
        self.assertTrue(smod.weight.requires_grad)
        self.assertFalse(fmod.hasattr("weight"))
        inp = torch.ones(1, 8, 32, 32)
        out1 = fmod.forward(inp)
        # FIXME: frozen module mutated from outside (original module).
        with torch.no_grad():
            smod.weight[0, 0, 0, 0] += 100.0
        out2 = fmod.forward(inp)
        out3 = smod(inp)
        self.assertNotEqual(out1, out2)
        self.assertEqual(out2, out3)

    def test_freeze_module_with_user_preserved_attr(self):
        # [FUNCTION DOC] Freezes with attribute a named as preserved and checks that a is retained
        # while b is folded.
        class Module(nn.Module):
            # [CLASS DOC] HELPER: Module with tensors a and b; forward returns their sum.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor attributes a=1.1 and b=2.2.
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a + b.
                return self.a + self.b

        m = torch.jit.script(Module())
        m.eval()
        fm = torch._C._freeze_module(m._c, ["a"])
        # Attribute "a" is preserved
        self.assertTrue(fm.hasattr("a"))
        self.assertFalse(fm.hasattr("b"))

    def test_freeze_module_with_user_preserved_method(self):
        # [FUNCTION DOC] Freezes with the exported method modify_a preserved. Checks that a is
        # kept (mutated by that method), b is folded, and the frozen forward output equals the
        # scripted one.
        class Module(nn.Module):
            # [CLASS DOC] HELPER: Module with tensors a and b and exported mutating methods.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor attributes a=1.1 and b=2.2.
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a + b.
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                # [FUNCTION DOC] HELPER: Exported; adds 10 in place to the first element of a and
                # returns b.
                self.a[0] += 10
                return self.b

            @torch.jit.export
            def modify_b(self, x):
                # [FUNCTION DOC] HELPER: Exported; adds 20 in place to the first element of b and
                # returns a.
                self.b[0] += 20
                return self.a

        m = torch.jit.script(Module())
        m.eval()
        fm = torch._C._freeze_module(m._c, ["modify_a"])
        # Both attribute "a" and method "modify_a" are preserved
        self.assertTrue(fm.hasattr("a"))
        self.assertFalse(fm.hasattr("b"))
        input = torch.randn(2, 2)
        expected = m.forward(input)
        out = fm.forward(input)
        self.assertEqual(out, expected)

    def test_freeze_module_with_user_preserved_method2(self):
        # [FUNCTION DOC] Freezes with modify_a preserved where forward and modify_a touch different
        # attributes. Uses FileCheck to confirm the forward graph still reads attribute a and the
        # preserved method graph reads attribute b.
        class Module(nn.Module):
            # [CLASS DOC] HELPER: Module with tensors a and b; forward mutates b, modify_a
            # mutates a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor attributes a=1.1 and b=2.2.
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 10 in place to b and returns a + b.
                self.b += 10
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                # [FUNCTION DOC] HELPER: Exported; adds 10 in place to the first element of a and
                # returns b + a.
                self.a[0] += 10
                return self.b + self.a

        m = torch.jit.script(Module())
        m.eval()
        fm = torch._C._freeze_module(m._c, ["modify_a"])
        FileCheck().check('prim::GetAttr[name="a"]').run(fm.forward.graph)
        FileCheck().check('prim::GetAttr[name="b"]').run(fm.modify_a.graph)

    def test_freeze_module_with_user_preserved_attribute_on_submodule(self):
        # [FUNCTION DOC] Freezes with dotted preserved names sub1.a and sub2.a. Checks that both
        # submodules remain with attribute a kept and b removed, the module returns 6, and after
        # incrementing sub1.a it returns 7.
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Submodule with int attributes a=1 and b=2; forward returns
            # their sum.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets int attributes a and b.
                super().__init__()
                self.a = 1
                self.b = 2

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns a + b.
                return self.a + self.b

        class Module(nn.Module):
            # [CLASS DOC] HELPER: Module with two SubModule instances; forward sums their outputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates sub1 and sub2.
                super().__init__()
                self.sub1 = SubModule()
                self.sub2 = SubModule()

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the sum of the two submodule outputs.
                return self.sub1() + self.sub2()

        m = torch.jit.script(Module())
        m.eval()
        m = torch.jit.freeze(m, preserved_attrs=["sub1.a", "sub2.a"])
        fm = m._c

        self.assertTrue(fm.hasattr("sub1"))
        self.assertTrue(fm.sub1.hasattr("a"))
        self.assertFalse(fm.sub1.hasattr("b"))
        self.assertTrue(fm.hasattr("sub2"))
        self.assertTrue(fm.sub2.hasattr("a"))
        self.assertFalse(fm.sub2.hasattr("b"))
        self.assertEqual(m(), 6)
        m.sub1.a += 1
        self.assertEqual(m(), 7)

    def test_freeze_module_with_user_preserved_attribute_on_unused_submodule(self):
        # [FUNCTION DOC] Freezes a module whose forward never uses its submodule, preserving the
        # attribute sub.a and method sub.method_a. Checks the submodule is kept with a and the
        # method, while b is dropped.
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Submodule with int attributes a and b and an exported method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets int attributes a=1 and b=2.
                super().__init__()
                self.a = 1
                self.b = 2

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns a + b.
                return self.a + self.b

            @torch.jit.export
            def method_a(self):
                # [FUNCTION DOC] HELPER: Exported; returns the constant 42.
                return 42

        class Module(nn.Module):
            # [CLASS DOC] HELPER: Module with a submodule that forward does not use (returns 1).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates submodule sub.
                super().__init__()
                self.sub = SubModule()

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the constant 1.
                return 1

        m = torch.jit.script(Module())
        m.eval()
        fm = torch.jit.freeze(m, preserved_attrs=["sub.a", "sub.method_a"])._c

        self.assertTrue(fm.hasattr("sub"))
        self.assertTrue(fm.sub.hasattr("a"))
        self.assertFalse(fm.sub.hasattr("b"))
        self.assertTrue(fm.sub._has_method("method_a"))

    def test_freeze_module_with_user_preserved_method_on_submodule(self):
        # [FUNCTION DOC] Freezes while preserving only the method sub.method_a. Checks that the
        # submodule has method_a but its other helper method_b is removed.
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Submodule with two helper methods used by its forward.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns method_a output plus method_b output.
                return self.method_a(x) + self.method_b(x)

            def method_a(self, x):
                # [FUNCTION DOC] HELPER: Returns the input squared.
                return x * x

            def method_b(self, x):
                # [FUNCTION DOC] HELPER: Returns the input doubled.
                return x + x

        class Module(nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a SubModule that forward calls.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates submodule sub.
                super().__init__()
                self.sub = SubModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the submodule output.
                return self.sub(x)

        m = torch.jit.script(Module())
        m.eval()
        fm = torch.jit.freeze(m, preserved_attrs=["sub.method_a"])._c

        self.assertTrue(fm.hasattr("sub"))
        self.assertTrue(fm.sub._has_method("method_a"))
        self.assertFalse(fm.sub._has_method("method_b"))

    @xfailIf(IS_ARM64)
    # see https://github.com/pytorch/pytorch/issues/177258
    @skipIfNoFBGEMM
    def test_module_with_shared_type_instances(self):
        # [FUNCTION DOC] Regression test for freezing a statically quantized (fbgemm) module in
        # which two Child submodules share a type. Quantizes, scripts, inlines and freezes, then
        # checks the dump has no packed-params-false marker and that frozen and scripted outputs
        # are equal. Marked xfail on ARM64 and skipped without FBGEMM.
        class Child(nn.Module):
            # [CLASS DOC] HELPER: Module with a float Conv2d; forward applies it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the float32 convolution conv1.
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1).to(dtype=torch.float32)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv1 and returns the result.
                x = self.conv1(x)
                return x

        class Parent(nn.Module):
            # [CLASS DOC] HELPER: Module with quant and dequant stubs, a convolution and two
            # Child modules.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the quant stub, conv1, two Child modules and the
                # dequant stub.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv1 = nn.Conv2d(1, 1, 1).to(dtype=torch.float32)
                self.child = Child()
                self.child2 = Child()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantizes the input, applies conv1 and both children,
                # dequantizes, and returns the result.
                x = self.quant(x)
                x = self.conv1(x)
                x = self.child(x)
                x = self.child2(x)
                x = self.dequant(x)
                return x

        def _static_quant(model):
            # [FUNCTION DOC] HELPER: Wraps model in a quantization wrapper with the default
            # qconfig, prepares it in place, runs one random batch to calibrate, converts it in
            # place, and returns the original model object.
            qModel = torch.ao.quantization.QuantWrapper(model)
            qModel.qconfig = torch.ao.quantization.default_qconfig
            torch.ao.quantization.prepare(qModel, inplace=True)
            qModel(torch.rand(4, 1, 4, 4, dtype=torch.float32))
            torch.ao.quantization.convert(qModel, inplace=True)
            return model

        with override_quantized_engine("fbgemm"):
            data = torch.randn(4, 1, 4, 4, dtype=torch.float32)
            m = Parent().to(torch.float32)
            m = _static_quant(m)
            m = torch.jit.script(m)
            m.eval()
            torch._C._jit_pass_inline(m.graph)
            m_frozen = wrap_cpp_module(torch._C._freeze_module(m._c))
            # Earlier bug resulted in _packed_params set to false.
            FileCheck().check_not("_packed_params = False").run(
                m_frozen._c.dump_to_str(True, True, False)
            )

            m_res = m(data)
            # It used to segfault while running frozen module.
            m_frozen_res = m_frozen(data)
            self.assertEqual(m_res, m_frozen_res)

    def test_module_getattr_indirection(self):
        # [FUNCTION DOC] Freezes a module whose forward picks one of two script-class attributes by
        # a boolean condition and returns its field. Checks that the frozen module returns the same
        # value as an unscripted instance for both condition values.
        @torch.jit.script
        class ValHolder:
            # [CLASS DOC] HELPER: Scripted class with one int field.
            def __init__(self, val: int):
                # [FUNCTION DOC] HELPER: Stores the int argument in the field.
                self.val: int = val

        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Module holding two ValHolder objects (values 1 and 2).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two holder attributes.
                super().__init__()
                self.mod1 = ValHolder(1)
                self.mod2 = ValHolder(2)

            def forward(self, cond: bool):
                # [FUNCTION DOC] HELPER: Chooses mod1 when the flag is true, otherwise mod2, and
                # returns its field.
                if cond:
                    mod = self.mod1
                else:
                    mod = self.mod2
                return mod.val

        mod = Mod()
        mod.eval()
        frozen_mod = torch.jit.freeze(torch.jit.script(mod))
        mod_eager = Mod()
        self.assertEqual(mod_eager(True), frozen_mod(True))
        self.assertEqual(mod_eager(False), frozen_mod(False))

    def test_freeze_module_with_non_static_module_container_index(self):
        # [FUNCTION DOC] Checks that freezing refuses modules that index a module container with a
        # runtime value. Scripts one module that indexes a ModuleDict by a string key and one that
        # indexes a ModuleList by an int, each through an interface type, and expects the freeze
        # call to raise RuntimeError saying ModuleContainerIndex is not supported.
        """
        Test that Modules containing non-static ModuleDict or ModuleList
        indexing cannot be frozen.
        """

        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface type declaring a forward method that takes and returns
            # any value.
            def forward(self, inp: Any) -> Any:
                # [FUNCTION DOC] HELPER: Interface method declaration; the body does nothing.
                pass

        class ImplementsInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Implementation of the interface; returns the max over dim 0
            # for tensors.
            def forward(self, inp: Any) -> Any:
                # [FUNCTION DOC] HELPER: Returns the max over dimension 0 when the input is a
                # tensor, otherwise returns the input unchanged.
                if isinstance(inp, torch.Tensor):
                    return torch.max(inp, dim=0)

                return inp

        class ModWithDict(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ModuleDict holding one implementation, indexed
            # by key.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the module dictionary with the entry named module.
                super().__init__()
                self.d = torch.nn.ModuleDict({"module": ImplementsInterface()})

            def forward(self, x: torch.Tensor, key: str) -> Any:
                # [FUNCTION DOC] HELPER: Looks up the module by string key and calls its forward.
                value: ModuleInterface = self.d[key]
                return value.forward(x)

        m = torch.jit.script(ModWithDict())
        m.eval()
        with self.assertRaisesRegex(
            RuntimeError,
            "Freezing modules containing prim::ModuleContainerIndex is not supported",
        ):
            mf = torch._C._freeze_module(m._c)

        class ModWithList(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ModuleList holding one implementation, indexed
            # by int.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the module list with one implementation.
                super().__init__()
                self.l = torch.nn.ModuleList([ImplementsInterface()])

            def forward(self, x: torch.Tensor, idx: int) -> Any:
                # [FUNCTION DOC] HELPER: Looks up the module by int index and calls its forward.
                value: ModuleInterface = self.l[idx]
                return value.forward(x)

        m = torch.jit.script(ModWithList())
        m.eval()
        with self.assertRaisesRegex(
            RuntimeError,
            "Freezing modules containing prim::ModuleContainerIndex is not supported",
        ):
            mf = torch._C._freeze_module(m._c)

    def test_freeze_with_interface_mutable(self):
        # [FUNCTION DOC] Freezes a module that holds a submodule through an interface-typed
        # attribute, where the submodule accumulates into its own tensor. After one call,
        # checks the frozen submodule sum equals the relu of the input, i.e. the mutable state
        # is preserved.
        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface type requiring a tensor-to-tensor forward.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; the body does nothing.
                pass

        class ImplementsInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Implementation holding a 2x2 zeros tensor sum that it
            # accumulates into.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the zeros tensor attribute sum.
                super().__init__()
                self.sum = torch.zeros((2, 2))

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Adds the relu of the input into sum in place and
                # returns sum.
                self.sum += inp.relu()
                return self.sum

        class WrapperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed attribute impl set to
            # the implementation.
            impl: ModuleInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the implementation as attribute impl.
                super().__init__()
                self.impl = ImplementsInterface()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Delegates to the implementation forward.
                return self.impl.forward(x)

        m = torch.jit.script(WrapperModule())
        m.eval()
        m_frozen = torch.jit.freeze(m)

        x = torch.rand((2, 2))

        m_frozen(x)
        self.assertEqual(m_frozen.impl.sum, x.relu())

    def test_freeze_with_swapping_interfaces(self):
        # [FUNCTION DOC] Builds a module that reassigns its interface-typed attribute on
        # alternating calls and expects torch.jit.freeze to raise RuntimeError saying that freezing
        # does not support attribute assignment on an interface type.
        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface type requiring a tensor-to-tensor forward.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; the body does nothing.
                pass

        class Implementation1(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface implementation that applies relu.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the relu of the input.
                return inp.relu()

        class Implementation2(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface implementation that applies sin.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the sine of the input.
                return inp.sin()

        class WrapperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that swaps its interface-typed impl between two options
            # per call.
            impl: ModuleInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates both implementations, sets impl to the first, and
                # idx to 0.
                super().__init__()
                self.option1 = Implementation1()
                self.option2 = Implementation2()
                self.impl = self.option1
                self.idx = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Increments idx, assigns impl to option1 on odd counts and
                # option2 otherwise, then returns impl applied to the input.
                self.idx += 1
                if self.idx % 2 == 1:
                    self.impl = self.option1
                else:
                    self.impl = self.option2
                return self.impl(x)

        m = torch.jit.script(WrapperModule())
        m.eval()
        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type"
        ):
            m_frozen = torch.jit.freeze(m)

    def test_freeze_recursive_interfaces(self):
        # [FUNCTION DOC] Freezes a module with two levels of interface-typed attributes that are
        # never reassigned. Checks the frozen module output equals the eager output.
        @torch.jit.interface
        class InnerInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface type for the inner level.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; the body does nothing.
                pass

        @torch.jit.interface
        class OuterInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface type for the outer level.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; the body does nothing.
                pass

        class InnerImpl(torch.nn.Module):
            # [CLASS DOC] HELPER: Inner implementation with a ones tensor x; forward is cos(inp)
            # * x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the 2x2 ones tensor attribute x.
                super().__init__()
                self.x = torch.ones((2, 2))

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Returns the cosine of the input times x.
                return inp.cos() * self.x

        class OuterImpl(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer implementation holding an inner implementation via an
            # interface typed attribute.
            inner_impl: InnerInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the inner implementation as inner_impl.
                super().__init__()
                self.inner_impl = InnerImpl()

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Returns relu(inp) plus the inner implementation applied
                # to sin(inp).
                return inp.relu() + self.inner_impl(inp.sin())

        class WrapperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper holding the outer implementation via an
            # interface-typed attribute.
            outer_impl: OuterInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the outer implementation as outer_impl.
                super().__init__()
                self.outer_impl = OuterImpl()

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Returns the outer implementation output plus the input.
                return self.outer_impl(inp) + inp

        m = WrapperModule()
        x = torch.rand((2, 2))
        expected = m(x)

        m_s = torch.jit.script(m)
        m_s.eval()
        m_s = torch.jit.freeze(m_s)
        actual = m_s(x)

        self.assertEqual(expected, actual)

    def test_freeze_recursive_interfaces_with_reassignment(self):
        # [FUNCTION DOC] Like the recursive interface test, but the inner interface attribute is
        # reassigned in forward between two submodules. Expects torch.jit.freeze to raise
        # RuntimeError saying that freezing does not support attribute assignment on an
        # interface type.
        @torch.jit.interface
        class InnerInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface type for the inner level.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; the body does nothing.
                pass

        @torch.jit.interface
        class OuterInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface type for the outer level.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; the body does nothing.
                pass

        class InnerImpl1(torch.nn.Module):
            # [CLASS DOC] HELPER: Inner implementation with ones tensor x; forward is cos(inp) * x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the 2x2 ones tensor attribute x.
                super().__init__()
                self.x = torch.ones((2, 2))

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Returns the cosine of the input times x.
                return inp.cos() * self.x

        class InnerImpl2(torch.nn.Module):
            # [CLASS DOC] HELPER: Second inner implementation; forward is sin(inp) / x with x
            # twice ones.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the tensor attribute x equal to a 2x2 tensor
                # of twos.
                super().__init__()
                self.x = torch.ones((2, 2)) * 2

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Returns the sine of the input divided by x.
                return inp.sin() / self.x

        class OuterImpl(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer implementation that reassigns its interface-typed
            # inner_impl each call between two InnerImpl1 submodules.
            inner_impl: InnerInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates inner_impl, impl1, impl2 (all InnerImpl1) and
                # counter idx.
                super().__init__()
                self.inner_impl = InnerImpl1()
                self.impl1 = InnerImpl1()
                self.impl2 = InnerImpl1()
                self.idx = 0

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Increments idx, assigns inner_impl to impl1 on even counts
                # and impl2 otherwise, and returns relu(inp) plus inner_impl applied to sin(inp).
                self.idx += 1
                if self.idx % 2 == 0:
                    self.inner_impl = self.impl1
                else:
                    self.inner_impl = self.impl2
                return inp.relu() + self.inner_impl(inp.sin())

        class WrapperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper holding the outer implementation via an
            # interface-typed attribute.
            outer_impl: OuterInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the outer implementation as outer_impl.
                super().__init__()
                self.outer_impl = OuterImpl()

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Returns the outer implementation output plus the input.
                return self.outer_impl(inp) + inp

        m = WrapperModule()

        m_s = torch.jit.script(m)
        m_s.eval()
        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type"
        ):
            m_s = torch.jit.freeze(m_s)

    def test_freeze_interface_swapping_two_methods(self):
        # [FUNCTION DOC] Two wrapper modules each reassign an interface-typed attribute, one in
        # forward and one in an exported method. Freezing either (with other_method preserved) must
        # raise RuntimeError saying attribute assignment on an interface type is unsupported.
        @torch.jit.interface
        class MyInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface type requiring a tensor-to-tensor forward.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; the body does nothing.
                pass

        class Impl1(torch.nn.Module):
            # [CLASS DOC] HELPER: Implementation returning the cosine of its input.
            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Returns the cosine of the input.
                return inp.cos()

        class Impl2(torch.nn.Module):
            # [CLASS DOC] HELPER: Implementation returning the sine of its input.
            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Returns the sine of the input.
                return inp.sin()

        class WrapperModule1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose exported method reassigns interface_impl; forward
            # only calls it.
            interface_impl: MyInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates interface_impl, impl1, impl2 and a counter idx.
                super().__init__()
                self.interface_impl = Impl1()
                self.impl1 = Impl1()
                self.impl2 = Impl2()
                self.idx = 0

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies interface_impl to the input.
                return self.interface_impl(x)

            @torch.jit.export
            def other_method(self, x):
                # [FUNCTION DOC] HELPER: Exported; increments idx, assigns interface_impl to impl1
                # on even counts and impl2 otherwise, then applies it to the input.
                self.idx += 1
                if self.idx % 2 == 0:
                    self.interface_impl = self.impl1
                else:
                    self.interface_impl = self.impl2
                return self.interface_impl(x)

        class WrapperModule2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward reassigns interface_impl; an exported method
            # only calls it.
            interface_impl: MyInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates interface_impl, impl1, impl2 and a counter idx.
                super().__init__()
                self.interface_impl = Impl1()
                self.impl1 = Impl1()
                self.impl2 = Impl2()
                self.idx = 0

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Increments idx, assigns interface_impl to impl1 on even
                # counts and impl2 otherwise, then applies it to the input.
                self.idx += 1
                if self.idx % 2 == 0:
                    self.interface_impl = self.impl1
                else:
                    self.interface_impl = self.impl2
                return self.interface_impl(x)

            @torch.jit.export
            def other_method(self, x):
                # [FUNCTION DOC] HELPER: Exported; applies interface_impl to the input.
                return self.interface_impl(x)

        m1 = torch.jit.script(WrapperModule1())
        m2 = torch.jit.script(WrapperModule2())

        m1.eval()
        m2.eval()

        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type"
        ):
            torch.jit.freeze(m1, preserved_attrs=["other_method"])

        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type"
        ):
            torch.jit.freeze(m2, preserved_attrs=["other_method"])

    def test_freeze_recursive_interfaces_same_name(self):
        # [FUNCTION DOC] Freezes nested interface-typed attributes where the outer and inner
        # modules both use the attribute name impl and the outer one has its own tensor x. Checks
        # that the frozen output equals the eager output.
        @torch.jit.interface
        class InnerInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface type for the inner level.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; the body does nothing.
                pass

        @torch.jit.interface
        class OuterInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface type for the outer level.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; the body does nothing.
                pass

        class InnerImpl(torch.nn.Module):
            # [CLASS DOC] HELPER: Inner implementation with ones tensor x; forward is cos(inp) * x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the 2x2 ones tensor attribute x.
                super().__init__()
                self.x = torch.ones((2, 2))

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Returns the cosine of the input times x.
                return inp.cos() * self.x

        class OuterImpl(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer implementation with an inner interface-typed impl and a
            # tensor x equal to five times ones.
            impl: InnerInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates impl (an InnerImpl) and the tensor attribute x.
                super().__init__()
                self.impl = InnerImpl()
                self.x = torch.ones((2, 2)) * 5

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Delegates to other_method.
                return self.other_method(inp)

            def other_method(self, inp):
                # [FUNCTION DOC] HELPER: Returns relu(inp) plus impl applied to sin(inp) plus x.
                return inp.relu() + self.impl(inp.sin()) + self.x

        class WrapperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper with an interface-typed attribute impl set to
            # an OuterImpl.
            impl: OuterInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the OuterImpl as impl.
                super().__init__()
                self.impl = OuterImpl()

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Returns the impl output plus the input.
                return self.impl(inp) + inp

        m = WrapperModule()
        x = torch.rand((2, 2))
        expected = m(x)

        m_s = torch.jit.script(m)
        m_s.eval()
        m_s = torch.jit.freeze(m_s)
        actual = m_s(x)

        self.assertEqual(expected, actual)

    def test_freeze_non_interface_module_swap(self):
        # [FUNCTION DOC] Freezes a module that swaps a plain (non-interface) submodule attribute
        # between two instances on alternating calls. Checks the frozen output equals the output of
        # the unscripted module on the first call.
        class InnerModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a constant tensor x and returning relu(inp) + x.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores the given tensor as x.
                super().__init__()
                self.x = x

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns relu of the input plus x.
                return inp.relu() + self.x

        class WrapperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that reassigns attribute impl between two InnerModule
            # options on alternating calls.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates option1, option2 (random tensors), sets impl to
                # option1 and idx to 0.
                super().__init__()
                self.option1 = InnerModule(torch.rand((2, 2)))
                self.option2 = InnerModule(torch.rand((2, 2)))
                self.impl = self.option1
                self.idx = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Increments idx, assigns impl to option1 on odd counts and
                # option2 otherwise, and applies impl to the input.
                self.idx += 1
                if self.idx % 2 == 1:
                    self.impl = self.option1
                else:
                    self.impl = self.option2
                return self.impl(x)

        unfrozen = WrapperModule()
        m = torch.jit.script(unfrozen)
        m.eval()
        m_frozen = torch.jit.freeze(m)

        x = torch.rand((2, 2))
        expected = unfrozen(x)
        actual = m_frozen(x)
        self.assertEqual(expected, actual)

    @unittest.expectedFailure
    def test_freeze_interface_within_object(self):
        # [FUNCTION DOC] Expected-failure test (marked with expectedFailure) for freezing a module
        # that holds a plain Python object containing a module-typed field. Scripts and freezes it
        # and then compares outputs. Kept as an xfail so that a fix prompts someone to review the
        # freezing behavior.
        # KNOWN ISSUE: the final comparison calls expectEqual, which is not defined on the test
        # class (the usual helper is assertEqual); the test is expected to fail anyway.
        # I don't think there's any way to create a plain python object that
        # contains a torch.nn.Module inside it, but just in case... I'm not
        # sure freezing would handle this case correctly, so marking as xfail
        # so that if this ever _does_ start working someone will need to
        # investigate to make sure this is handled correctly.
        class MyIface(torch.nn.Module):
            # [CLASS DOC] HELPER: Module type used as an annotation for the object field; forward
            # does nothing.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Declaration only; the body does nothing.
                pass

        class MyImpl(torch.nn.Module):
            # [CLASS DOC] HELPER: Module implementation that applies sin.
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the sine of the input.
                return inp.sin()

        class MyObject:
            # [CLASS DOC] HELPER: Plain Python object (not a module) with a module-typed
            # field impl.
            impl: MyIface

            def run(self, x):
                # [FUNCTION DOC] HELPER: Calls the held module on the input.
                return self.impl(x)

        class WrapperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that owns a MyObject whose impl field is set to a MyImpl.
            impl: MyObject

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the object and assigns its inner
                # module implementation.
                super().__init__()
                self.impl = MyObject()
                self.impl.impl = MyImpl()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Calls the held object on the input.
                return self.impl(x)

        unfrozen = WrapperModule()
        m = torch.jit.script(unfrozen)
        m.eval()
        m_frozen = torch.jit.freeze(m)

        x = torch.rand((2, 2))
        expected = unfrozen(x)
        actual = m_frozen(x)
        self.expectEqual(expected, actual)

    def test_freeze_non_module_class_getattr(self):
        # [FUNCTION DOC] Freezes a module holding a plain Python class instance and calls a method
        # on it. Checks scripted-frozen output equals eager output and that the frozen graph has no
        # attribute reads.
        class BoxCoder:
            # [CLASS DOC] HELPER: Plain class with a float field bbox_xform_clip and a
            # decode method.
            def __init__(self, bbox_xform_clip):
                # [FUNCTION DOC] HELPER: Stores the float argument as bbox_xform_clip.
                # type: (float) -> None
                self.bbox_xform_clip = bbox_xform_clip

            def decode(self, input):
                # [FUNCTION DOC] HELPER: Returns the input multiplied by the stored float.
                return input * self.bbox_xform_clip

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with annotation declaring box_coder as a BoxCoder, holding
            # one with value 50.0.
            __annotations__ = {
                "box_coder": BoxCoder,
            }

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the BoxCoder attribute.
                super().__init__()
                self.box_coder = BoxCoder(50.0)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Decodes the input with the box coder.
                return self.box_coder.decode(input)

        model = MyModule()
        model.eval()
        script_model = torch.jit.freeze(torch.jit.script(model))
        inp = torch.randn([4, 4])
        output_eager = model(inp)
        self.assertEqual(model(inp), script_model(inp))
        FileCheck().check_not("GetAttr").run(script_model.graph)

    def test_freeze_module_with_tupleoutput_submodule(self):
        # [FUNCTION DOC] Freezes a module whose submodule returns a tuple that the parent unpacks.
        # Checks that tuple construct and unpack nodes are gone from the frozen graph and that the
        # output matches the scripted output.
        class SubModule(nn.Module):
            # [CLASS DOC] HELPER: Submodule returning a tuple of x + 1 and x + 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a two-element tuple of the input plus 1 and
                # plus 2.
                return (x + 1, x + 2)

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module that unpacks the submodule tuple and sums the two parts.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates submodule sub.
                super().__init__()
                self.sub = SubModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Unpacks the submodule tuple output into two values and
                # returns their sum.
                y1, y2 = self.sub(x)
                return y1 + y2

        m = torch.jit.script(TestModule())
        m = m.eval()
        mf = torch.jit.freeze(m)
        inp = torch.randn(2, 2)
        expected = m.forward(inp)
        output = mf.forward(inp)
        # Check if prim::TupleConstruct and prim::TupleUnpack
        # Don't exist in frozen graph
        FileCheck().check_not("prim::TupleConstruct").run(mf.graph)
        FileCheck().check_not("prim::TupleUnpack").run(mf.graph)
        self.assertEqual(output, expected)

    def test_freeze_module_with_call_method(self):
        # [FUNCTION DOC] Freezes a module while preserving the exported method
        # make_prediction, which calls forward. Checks the frozen forward equals the
        # unscripted forward within tolerance.
        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Module with a Parameter and an exported method that
            # calls forward.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Wraps the given tensor as the parameter param.
                super().__init__()
                self.param = nn.Parameter(val)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus param.
                # this method will change during freezing
                return x + self.param

            @torch.jit.export
            def make_prediction(self, x):
                # [FUNCTION DOC] HELPER: Exported; doubles the input and returns forward of
                # the result.
                y = x + x
                return self.forward(y)

        param = torch.rand([2, 2])
        x = torch.rand([2, 2])

        unscripted_mod = Mod(param)
        mod = torch.jit.script(unscripted_mod)
        mod.eval()
        mod = torch.jit.freeze(mod, preserved_attrs=["make_prediction"])

        self.assertEqual(
            mod.forward(x), unscripted_mod.forward(x), atol=1e-5, rtol=1e-5
        )


@skipIfTorchDynamo("somehow causing hanging during python shutdown")
class TestFrozenOptimizations(JitTestCase):
    # [CLASS DOC] Tests for graph optimizations applied to frozen modules: folding conv or linear
    # with batch norm, folding add/sub/mul/div into conv, mkldnn/oneDNN conversions, dropout
    # removal, concat and linear optimizations, and related passes. Skipped under TorchDynamo. The
    # default dtype is switched to double for each test.
    def setUp(self):
        # [FUNCTION DOC] Saves the current default dtype and sets the default dtype to double.
        super().setUp()
        self.default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.double)

    def tearDown(self):
        # [FUNCTION DOC] Restores the saved default dtype, then runs the base tearDown.
        torch.set_default_dtype(self.default_dtype)
        super().tearDown()

    def test_conv_bn_folding(self):
        # [FUNCTION DOC] Parametrized by loop over conv dimensionality, conv bias, tracing vs
        # scripting, and batch norm running stats. Builds conv + batch norm modules, runs inline,
        # peephole and constant propagation, checks the fold pass is a no-op before freezing, then
        # freezes and checks that batch norm is folded away only when running stats are tracked.
        # Outputs are compared against eager.
        conv_bias = [True, False]
        module_pairs = [
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d),
        ]
        use_tracing = [True, False]
        bn_running_stats = [True, False]

        for use_bias, modules, tracing, track_stats in product(
            conv_bias, module_pairs, use_tracing, bn_running_stats
        ):

            class ConvBN(torch.nn.Module):
                # [CLASS DOC] HELPER: Module containing a convolution followed by batch norm, built
                # from the classes chosen by the enclosing loop.
                def __init__(self, in_channels, out_channels, **kwargs):
                    # [FUNCTION DOC] HELPER: Creates the conv (with or without bias) and the batch
                    # norm (eps 0.001, running stats per the loop variable).
                    super().__init__()
                    self.conv = modules[0](
                        in_channels, out_channels, bias=use_bias, **kwargs
                    )
                    self.bn = modules[1](
                        out_channels, eps=0.001, track_running_stats=track_stats
                    )

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the convolution then batch norm.
                    x = self.conv(x)
                    return self.bn(x)

            mod_eager = ConvBN(3, 32, kernel_size=3, stride=2).eval()
            inps = [4, 3, 4]
            if modules[0] is nn.Conv2d:
                inps.append(inps[-1])
            if modules[0] is nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            inp = torch.rand(inps)

            if tracing:
                scripted_mod = torch.jit.trace(mod_eager, (inp))
            else:
                scripted_mod = torch.jit.script(mod_eager)

            self.run_pass("inline", scripted_mod.graph)
            self.run_pass("peephole", scripted_mod.graph)
            self.run_pass("constant_propagation", scripted_mod.graph)

            FileCheck().check("conv").check("batch").run(scripted_mod.graph)
            # successfully no-ops with non-const inputs
            self.run_pass("fold_frozen_conv_bn", scripted_mod.graph)
            FileCheck().check("conv").check("aten::batch_norm").run(scripted_mod.graph)

            scripted_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("fold_frozen_conv_bn", scripted_mod.graph)
            if track_stats:
                FileCheck().check("conv").check_not("aten::batch_norm").run(
                    scripted_mod.graph
                )
            else:
                FileCheck().check("conv").check("aten::batch_norm").run(
                    scripted_mod.graph
                )

            self.assertEqual(mod_eager(inp), scripted_mod(inp))
            self.assertEqual(mod_eager(inp), scripted_mod(inp))

    def test_conv_bn_folding_not_forward(self):
        # [FUNCTION DOC] Checks that conv-batch norm folding is applied to the exported method
        # make_prediction (not only forward) and that preserving the non-method attribute amt does
        # not break freezing. Verifies via graph checks that batch norm is gone after freezing.
        class ConvBN(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with Conv2d, BatchNorm2d, a float amt and an
            # exported method.
            def __init__(self, in_channels, out_channels, **kwargs):
                # [FUNCTION DOC] HELPER: Creates conv with bias, batch norm with eps 0.001, and
                # float amt.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels, out_channels, bias=True, **kwargs
                )
                self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
                self.amt = 3.2

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv then batch norm.
                x = self.conv(x)
                return self.bn(x)

            @torch.jit.export
            def make_prediction(self, x):
                # [FUNCTION DOC] HELPER: Exported; returns forward of the input plus amt.
                return self.forward(x) + self.amt

        mod_eager = ConvBN(3, 32, kernel_size=3, stride=2).eval()
        scripted_mod = torch.jit.script(mod_eager)
        torch._C._jit_pass_inline(scripted_mod.make_prediction.graph)
        FileCheck().check("conv").check("aten::batch_norm").run(
            scripted_mod.make_prediction.graph
        )

        # _jit_pass_optimize_frozen_graph should not be called on non-method attributes (e.g. "amt")
        scripted_mod = torch.jit.freeze(
            scripted_mod, preserved_attrs=["make_prediction", "amt"]
        )
        FileCheck().check("conv").check_not("aten::batch_norm").run(
            scripted_mod.make_prediction.graph
        )

    # During freezing this creates tensors constants that are attached to the frozen graph,
    # which is then kept alive by the compilation unit (which causes a leak)
    @skipCUDAMemoryLeakCheckIf(True)
    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_conv_bn_folding_autocast_scenario_cuda(self):
        # [FUNCTION DOC] CUDA-only regression for conv-batch norm folding where the conv is
        # half and the batch norm is float. After freezing, checks batch norm is folded, the
        # conv bias is half, and outputs match eager within tolerance. Skipped without CUDA and
        # with memory-leak check disabled because frozen constants are kept alive by the
        # compilation unit.
        # CUDA conv takes input tensors which must all be the same dtype,
        # which can cause issues if folding produces inputs of different dtypes.

        class ConvBN(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a half precision bias-free conv and a float
            # batch norm.
            def __init__(self, in_channels, out_channels, **kwargs):
                # [FUNCTION DOC] HELPER: Creates the half-precision conv and float batch norm.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels, out_channels, bias=False, dtype=torch.half, **kwargs
                )
                self.bn = torch.nn.BatchNorm2d(
                    out_channels, eps=0.001, dtype=torch.float
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv then batch norm.
                return self.bn(self.conv(x))

        mod_eager = ConvBN(3, 32, kernel_size=3, stride=2).cuda().eval()
        scripted_mod = torch.jit.script(mod_eager)
        scripted_mod = torch.jit.freeze(scripted_mod)
        FileCheck().check("conv").check_not("aten::batch_norm").run(scripted_mod.graph)
        conv_node = scripted_mod.graph.findNode("aten::conv2d", True)
        self.assertTrue(conv_node is not None)
        bias_input = conv_node.namedInput("bias")
        self.assertTrue(bias_input is not None)
        self.assertTrue(bias_input.type().dtype() == torch.half)

        x = torch.rand((3, 3, 32, 32), dtype=torch.half).cuda()

        self.assertEqual(mod_eager(x), scripted_mod(x), atol=1e-2, rtol=1e-2)
        self.assertEqual(mod_eager(x), scripted_mod(x), atol=1e-2, rtol=1e-2)

    def test_conv_add_folding(self):
        # [FUNCTION DOC] Tests folding of add, sub, mul and div into a preceding convolution.
        # Defines a helper that builds and freezes a module, runs the fold passes, and checks the
        # op node disappears (or stays) as expected. The test then sweeps conv dimensionality,
        # bias, tracing, operator and scalar operand, and also broadcasting tensor operands.
        @torch.no_grad()
        def test_conv_fusion(
            use_bias, module, tracing, op, scalar, add_tensor, expect_success
        ):
            # [FUNCTION DOC] HELPER: Builds a conv followed by the given operator (scalar or tensor
            # operand), converts to a traced or scripted module, checks the fold passes do nothing
            # before freezing, freezes, and checks that the op is folded away only if
            # expect_success is true. Compares outputs against eager. Runs with gradients disabled.
            class ConvOp(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a conv, an unused second conv, and a configurable
                # operator applied to the conv output.
                __constants__ = ["use_scalar"]

                def __init__(self, in_channels, out_channels, tensor=None, **kwargs):
                    # [FUNCTION DOC] HELPER: Builds convs, the scalar-mode flag, an operand tensor
                    # (given or random with one channel-wide dimension) and stores the operator.
                    super().__init__()
                    self.conv = module(
                        in_channels, out_channels, bias=use_bias, **kwargs
                    )
                    self.conv2 = module(
                        in_channels, out_channels, bias=use_bias, **kwargs
                    )
                    self.use_scalar = scalar
                    tensor_size = [1 for _ in range(self.conv.weight.ndim)]
                    tensor_size[1] = self.conv.weight.size(0)
                    self.tensor = (
                        add_tensor
                        if add_tensor is not None
                        else torch.rand(tensor_size)
                    )
                    self.op = op

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies conv, then the operator with scalar 2.0 or the
                    # stored tensor.
                    x = self.conv(x)
                    if self.use_scalar:
                        return self.op(x, 2.0)
                    else:
                        return self.op(x, self.tensor)

            mod_eager = ConvOp(3, 32, kernel_size=3, stride=2).eval()

            inps = [4, 3, 4]
            if module is nn.Conv2d:
                inps.append(inps[-1])
            if module is nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            inp = torch.rand(inps)

            if tracing:
                scripted_mod = torch.jit.trace(mod_eager, (inp,))
            else:
                scripted_mod = torch.jit.script(mod_eager)

            self.run_pass("inline", scripted_mod.graph)
            op_str = "aten::" + op.__name__

            FileCheck().check("conv").check(op_str).run(scripted_mod.graph)
            # successively no-ops with non-const inputs
            self.run_pass("fold_frozen_conv_mul_or_div", scripted_mod.graph)
            self.run_pass("fold_frozen_conv_add_or_sub", scripted_mod.graph)
            FileCheck().check("conv").check(op_str).run(scripted_mod.graph)
            scripted_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("fold_frozen_conv_mul_or_div", scripted_mod.graph)
            self.run_pass("fold_frozen_conv_add_or_sub", scripted_mod.graph)

            if expect_success:
                FileCheck().check("conv").check_not(op_str).run(scripted_mod.graph)
            else:
                FileCheck().check("conv").check(op_str).run(scripted_mod.graph)

            self.assertEqual(mod_eager(inp), scripted_mod(inp))
            self.assertEqual(mod_eager(inp), scripted_mod(inp))

        conv_bias = [True, False]
        modules = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        use_tracing = [False, True]
        use_scalar = [False, True]
        ops = [torch.add, torch.sub, torch.mul, torch.div]

        for use_bias, module, tracing, pytorch_op, scalar in product(
            conv_bias, modules, use_tracing, ops, use_scalar
        ):
            test_conv_fusion(
                use_bias,
                module,
                tracing,
                pytorch_op,
                scalar,
                add_tensor=None,
                expect_success=True,
            )

        for use_bias, pytorch_op in product(conv_bias, ops):
            # broadcasting add
            test_conv_fusion(
                use_bias,
                nn.Conv2d,
                False,
                pytorch_op,
                False,
                add_tensor=torch.rand(32, 1, 32),
                expect_success=False,
            )

            # broadcasting add
            test_conv_fusion(
                use_bias,
                nn.Conv2d,
                False,
                pytorch_op,
                False,
                add_tensor=torch.rand(1, 1),
                expect_success=True,
            )

            # add with different dtype
            test_conv_fusion(
                use_bias,
                nn.Conv2d,
                False,
                pytorch_op,
                False,
                add_tensor=torch.tensor([2]).to(torch.int),
                expect_success=True,
            )

    def test_conv_mul_add_bn(self):
        # [FUNCTION DOC] Traces a conv, multiply by a tensor, add a tensor, batch norm chain and
        # freezes it with gradients disabled. Checks the result equals eager and that batch norm
        # and add are both folded away in the frozen graph.
        class Conv_Mul_Add_Bn(nn.Module):
            # [CLASS DOC] HELPER: Module applying Conv2d, then multiply by tensor1, add tensor2,
            # batch norm.
            def __init__(self, in_channels, out_channels, **kwargs):
                # [FUNCTION DOC] HELPER: Creates conv, batch norm (eps 0.001) and two scalar
                # tensors 2.2 and 2.
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
                self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
                self.tensor1 = torch.tensor(2.2)
                self.tensor2 = torch.tensor(2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns batch norm of the sum of tensor2 and the product
                # of conv output with tensor1.
                return self.bn(
                    torch.add(torch.mul(self.conv(x), self.tensor1), self.tensor2)
                )

        input = torch.randn(8, 3, 64, 64)
        model = Conv_Mul_Add_Bn(3, 32, kernel_size=3, stride=1).eval()

        with torch.no_grad():
            result = model(input)
            traced_model = torch.jit.trace(model, input).eval()
            traced_model = torch.jit.freeze(traced_model)
            tresult = traced_model(input)
            self.assertEqual(result, tresult)
            FileCheck().check("conv").check_not("aten::batch_norm").run(
                traced_model.graph
            )
            FileCheck().check("conv").check_not("aten::add").run(traced_model.graph)

    def test_linear_bn_folding(self):
        # [FUNCTION DOC] Parametrized by loop over batch norm dimensionality, tracing vs scripting,
        # and running stats. Builds linear + batch norm, checks the linear-bn fold pass is a no-op
        # before freezing, then after freezing checks batch norm disappears only when running stats
        # are tracked. Outputs are compared against eager.
        module_pairs = [
            (nn.Linear, nn.BatchNorm1d),
            (nn.Linear, nn.BatchNorm2d),
            (nn.Linear, nn.BatchNorm3d),
        ]
        use_tracing = [True, False]
        bn_running_stats = [True, False]

        for modules, tracing, track_stats in product(
            module_pairs, use_tracing, bn_running_stats
        ):

            class LinearBN(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a linear layer followed by batch norm of the type
                # chosen by the enclosing loop.
                def __init__(self, in_features, out_features):
                    # [FUNCTION DOC] HELPER: Creates the linear layer and batch norm (eps 0.001,
                    # running stats per the loop variable).
                    super().__init__()
                    self.linear = modules[0](in_features, out_features)
                    self.bn = modules[1](
                        out_features, eps=0.001, track_running_stats=track_stats
                    )

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies linear then batch norm.
                    x = self.linear(x)
                    return self.bn(x)

            mod_eager = LinearBN(32, 32).eval()

            inps = [3, 32]
            if modules[1] is nn.BatchNorm2d:
                inps.append(inps[-1])
                inps.append(inps[-1])
            if modules[1] is nn.BatchNorm3d:
                inps.append(inps[-1])
                inps.append(inps[-1])
                inps.append(inps[-1])

            inp = torch.rand(inps)

            if tracing:
                scripted_mod = torch.jit.trace(mod_eager, (inp))
            else:
                scripted_mod = torch.jit.script(mod_eager)

            self.run_pass("inline", scripted_mod.graph)
            self.run_pass("peephole", scripted_mod.graph)
            self.run_pass("constant_propagation", scripted_mod.graph)

            FileCheck().check("linear").check("batch").run(scripted_mod.graph)
            # successfully no-ops with non-const inputs
            self.run_pass("fold_frozen_linear_bn", scripted_mod.graph)
            FileCheck().check("linear").check("aten::batch_norm").run(
                scripted_mod.graph
            )

            scripted_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("fold_frozen_linear_bn", scripted_mod.graph)
            if track_stats:
                FileCheck().check("linear").check_not("aten::batch_norm").run(
                    scripted_mod.graph
                )
            else:
                FileCheck().check("linear").check("aten::batch_norm").run(
                    scripted_mod.graph
                )

            self.assertEqual(mod_eager(inp), scripted_mod(inp))
            self.assertEqual(mod_eager(inp), scripted_mod(inp))

    def test_bn_not_broadcast_with_linear(self):
        # [FUNCTION DOC] Checks that linear-bn folding is skipped when the linear output size is
        # not compatible with the batch norm feature count (smaller, larger, or one). For each case
        # the frozen graph still contains batch norm, outputs match eager, and the eager fusion
        # helper raises AssertionError with the expected feature-count message.
        module_pairs = [
            (nn.Linear, nn.BatchNorm1d),
            (nn.Linear, nn.BatchNorm2d),
            (nn.Linear, nn.BatchNorm3d),
        ]
        use_tracing = [True, False]
        linear_in = 3
        # (linear_out, bn_in)
        # case 1: linear_out < bn_in
        # case 2: linear_out > bn_in
        # case 3: linear_out != bn_in && linear_out = 1
        dims = [(2, 4), (4, 2), (1, 2)]

        for modules, tracing, dim in product(module_pairs, use_tracing, dims):
            linear_out, bn_in = dim[0], dim[1]

            linear = modules[0](linear_in, linear_out)
            bn = modules[1](bn_in)
            mod_eager = nn.Sequential(linear, bn).eval()

            N, C = 3, bn_in
            input_shape = [N, C]
            if modules[1] is nn.BatchNorm1d:
                H = linear_in
                input_shape.append(H)
            elif modules[1] is nn.BatchNorm2d:
                H, W = 4, linear_in
                input_shape.append(H)
                input_shape.append(W)
            elif modules[1] is nn.BatchNorm3d:
                D, H, W = 4, 4, linear_in
                input_shape.append(D)
                input_shape.append(H)
                input_shape.append(W)

            inp = torch.rand(input_shape)

            if tracing:
                scripted_mod = torch.jit.trace(mod_eager, (inp))
            else:
                scripted_mod = torch.jit.script(mod_eager)

            self.run_pass("inline", scripted_mod.graph)
            self.run_pass("peephole", scripted_mod.graph)
            self.run_pass("constant_propagation", scripted_mod.graph)

            FileCheck().check("linear").check("batch").run(scripted_mod.graph)
            self.run_pass("fold_frozen_linear_bn", scripted_mod.graph)
            FileCheck().check("linear").check("aten::batch_norm").run(
                scripted_mod.graph
            )

            frozen_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("fold_frozen_linear_bn", frozen_mod.graph)
            # successfully skipped folding
            FileCheck().check("linear").check("aten::batch_norm").run(frozen_mod.graph)

            self.assertEqual(mod_eager(inp), frozen_mod(inp))
            self.assertEqual(mod_eager(inp), frozen_mod(inp))

            # successfully failed folding
            with self.assertRaisesRegex(
                AssertionError,
                "To fuse, linear.out_features == bn.num_features or bn.num_features == 1",
            ):
                nn.utils.fusion.fuse_linear_bn_eval(linear, bn)

    @skipCUDAMemoryLeakCheckIf(True)
    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_bn_folding_autocast_scenario_cuda(self):
        # [FUNCTION DOC] CUDA-only: folds a half-precision bias-free linear with a float batch norm
        # under freezing. Checks batch norm is removed, linear weight and bias are half, and
        # outputs match eager within tolerance for each batch norm dimensionality, tracing mode and
        # stats mode. Skipped without CUDA; memory-leak check disabled.
        module_pairs = [
            (nn.Linear, nn.BatchNorm1d),
            (nn.Linear, nn.BatchNorm2d),
            (nn.Linear, nn.BatchNorm3d),
        ]
        use_tracing = [True, False]
        bn_running_stats = [True, False]

        for modules, tracing, track_stats in product(
            module_pairs, use_tracing, bn_running_stats
        ):

            class LinearBN(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a half-precision linear layer and a float
                # batch norm.
                def __init__(self, in_features, out_features):
                    # [FUNCTION DOC] HELPER: Creates the linear layer without bias and the
                    # batch norm.
                    super().__init__()
                    self.linear = modules[0](
                        in_features, out_features, bias=False, dtype=torch.half
                    )
                    self.bn = modules[1](out_features, eps=0.001, dtype=torch.float)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies linear then batch norm.
                    x = self.linear(x)
                    return self.bn(x)

            mod_eager = LinearBN(32, 32).cuda().eval()

            inps = [3, 32]
            if modules[1] is nn.BatchNorm2d:
                inps.append(inps[-1])
                inps.append(inps[-1])
            if modules[1] is nn.BatchNorm3d:
                inps.append(inps[-1])
                inps.append(inps[-1])
                inps.append(inps[-1])

            x = torch.rand(inps, dtype=torch.half).cuda()

            if tracing:
                scripted_mod = torch.jit.trace(mod_eager, (x))
            else:
                scripted_mod = torch.jit.script(mod_eager)
            scripted_mod = torch.jit.freeze(scripted_mod)
            FileCheck().check("linear").check_not("aten::batch_norm").run(
                scripted_mod.graph
            )
            lin_node = scripted_mod.graph.findNode("aten::linear", True)
            self.assertTrue(lin_node is not None)
            weight_input = lin_node.namedInput("weight")
            bias_input = lin_node.namedInput("bias")
            self.assertTrue(bias_input is not None)
            self.assertTrue(weight_input.type().dtype() == torch.half)
            self.assertTrue(bias_input.type().dtype() == torch.half)

            self.assertEqual(mod_eager(x), scripted_mod(x), atol=1e-2, rtol=1e-2)
            self.assertEqual(mod_eager(x), scripted_mod(x), atol=1e-2, rtol=1e-2)

    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_concat(self):
        # [FUNCTION DOC] CUDA-only: for two sets of output sizes, builds a module with two linear
        # layers on the same input and checks via check_linear_optimizations that the pair is
        # merged into one linear on CUDA.
        out_dimms = [[5, 10], [1, 5]]

        for w1_dim, w2_dim in out_dimms:

            class ModMultLinear(nn.Module):
                # [CLASS DOC] HELPER: Module with two parameter sets applied as separate linears to
                # one input.
                def __init__(self, w1_dim, w2_dim):
                    # [FUNCTION DOC] HELPER: Creates weight and bias parameters for both linears
                    # with random data.
                    super().__init__()
                    self.w1 = nn.Parameter(torch.rand([w1_dim, 5]))
                    self.b1 = nn.Parameter(torch.rand([w1_dim]))
                    self.w2 = nn.Parameter(torch.rand([w2_dim, 5]))
                    self.b2 = nn.Parameter(torch.rand([w2_dim]))

                def forward(self, in_tensor1):
                    # [FUNCTION DOC] HELPER: Applies both linears to the input and returns the
                    # two results.
                    res1 = torch._C._nn.linear(in_tensor1, self.w1, self.b1)
                    res2 = torch._C._nn.linear(in_tensor1, self.w2, self.b2)
                    return res1, res2

            mod_eager = ModMultLinear(w1_dim, w2_dim).eval()

            test_val1 = torch.rand([50, 5])
            self.check_linear_optimizations(mod_eager, 2, 1, (test_val1,))

    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_concat_complex(self):
        # [FUNCTION DOC] CUDA-only: checks that the linear concat optimization works on a graph
        # where several linears share inputs or weights, with four linears reduced to two on CUDA.
        """
        Testing that the interleaving of multiple optimizations does not
        cause errors, and gets optimized as expected
        """

        class ModMultLinear(nn.Module):
            # [CLASS DOC] HELPER: Module with two linears applied to the input and two applied to
            # an intermediate result.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two parameter sets (5 and 10 output features) with
                # random data.
                super().__init__()
                w1_dim = 5
                w2_dim = 10
                self.w1 = nn.Parameter(torch.rand([w1_dim, 5]))
                self.b1 = nn.Parameter(torch.rand([w1_dim]))
                self.w2 = nn.Parameter(torch.rand([w2_dim, 5]))
                self.b2 = nn.Parameter(torch.rand([w2_dim]))

            def forward(self, in_tensor1):
                # [FUNCTION DOC] HELPER: Returns the results of three linears: second
                # weights on the input, second weights on an intermediate, and first weights
                # on the intermediate.
                res1 = torch._C._nn.linear(in_tensor1, self.w1, self.b1)
                res3 = torch._C._nn.linear(res1, self.w2, self.b2)
                res2 = torch._C._nn.linear(in_tensor1, self.w2, self.b2)
                res4 = torch._C._nn.linear(res1, self.w1, self.b1)
                return res2, res3, res4

        mod_eager = ModMultLinear().eval()
        test_val1 = torch.rand([50, 5])
        self.check_linear_optimizations(mod_eager, 4, 2, (test_val1,))

    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_concat_different_input(self):
        # [FUNCTION DOC] CUDA-only: two linears applied to different input tensors must not be
        # merged; check_linear_optimizations expects the count to stay at two.
        """
        There should be no change to the graph due to the optimization pass
        due to the two input tensors being different
        """

        # Freezing requires that the graph be a module
        class ModMultLinear(nn.Module):
            # [CLASS DOC] HELPER: Module applying two linears to two separate inputs.
            def __init__(self, w1_dim, w2_dim):
                # [FUNCTION DOC] HELPER: Creates two weight and bias parameter pairs of the
                # given sizes.
                super().__init__()
                self.w1 = nn.Parameter(torch.rand([w1_dim, 5]))
                self.b1 = nn.Parameter(torch.rand([w1_dim]))
                self.w2 = nn.Parameter(torch.rand([w2_dim, 5]))
                self.b2 = nn.Parameter(torch.rand([w2_dim]))

            def forward(self, in_tensor1, in_tensor2):
                # [FUNCTION DOC] HELPER: Applies the first linear to the first input and the second
                # linear to the second input, returning both.
                res1 = torch._C._nn.linear(in_tensor1, self.w1, self.b1)
                res2 = torch._C._nn.linear(in_tensor2, self.w2, self.b2)
                return res1, res2

        mod_eager = ModMultLinear(5, 5).eval()
        test_val1 = torch.rand([50, 5])
        test_val2 = torch.rand([50, 5])
        self.check_linear_optimizations(mod_eager, 2, 2, (test_val1, test_val2))

    @unittest.skipIf(not TEST_CUDA, "Optimization currently only run for GPU")
    def test_linear_multiple_blocks(self):
        # [FUNCTION DOC] CUDA-only: checks the linear concat pass on a graph with a
        # conditional block. Four linears are expected to become three, so only linears
        # sharing a block are merged.
        class ModMultLinear(nn.Module):
            # [CLASS DOC] HELPER: Module with linears inside and outside a conditional branch.
            def __init__(self, w1_dim, w2_dim):
                # [FUNCTION DOC] HELPER: Creates two weight and bias parameter pairs of the
                # given sizes.
                super().__init__()
                self.w1 = nn.Parameter(torch.rand([w1_dim, 5]))
                self.b1 = nn.Parameter(torch.rand([w1_dim]))
                self.w2 = nn.Parameter(torch.rand([w2_dim, 5]))
                self.b2 = nn.Parameter(torch.rand([w2_dim]))

            def forward(self, in_tensor1, in_tensor2, cond: bool):
                # [FUNCTION DOC] HELPER: Computes one linear, then two linears inside the true
                # branch (the false branch raises an error), and another linear on the same input
                # outside it.
                res1 = torch._C._nn.linear(in_tensor1, self.w1, self.b1)
                if cond:
                    res3 = torch._C._nn.linear(in_tensor2, self.w2, self.b2)
                    res4 = torch._C._nn.linear(in_tensor1, self.w2, self.b1)
                else:
                    raise AssertionError
                res2 = torch._C._nn.linear(in_tensor1, self.w2, self.b1)
                return res1, res2, res3, res4

        mod_eager = ModMultLinear(5, 5).eval()
        test_val1 = torch.rand([50, 5])
        test_val2 = torch.rand([50, 5])
        self.check_linear_optimizations(mod_eager, 4, 3, (test_val1, test_val2, True))

    def check_linear_optimizations(
        self, eager_mod, orig_linears, new_linears, test_vals
    ):
        # [FUNCTION DOC] HELPER: Runs on CPU and, in the second round, CUDA. Scripts the module,
        # checks the linear count is the original, runs the concat pass (no-op before freezing),
        # freezes and reruns the pass, and checks the count equals new_linears on CUDA or the
        # original on CPU. Finally compares eager and scripted outputs.
        for is_cuda in [False, True]:
            if is_cuda:
                mod_to_device = eager_mod.cuda()
                test_vals_to_device = [
                    t.cuda() if isinstance(t, torch.Tensor) else t for t in test_vals
                ]
            else:
                mod_to_device = eager_mod
                test_vals_to_device = test_vals

            script_mod = torch.jit.script(mod_to_device)
            op_graph = script_mod.graph

            FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
                op_graph
            )
            # successively no-ops with non-const inputs
            self.run_pass("concat_frozen_linear", op_graph)
            FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
                op_graph
            )

            script_mod = torch.jit.freeze(script_mod)
            op_graph = script_mod.graph
            self.run_pass("concat_frozen_linear", op_graph)
            if is_cuda:
                FileCheck().check_count("aten::linear", new_linears, exactly=True).run(
                    op_graph
                )
            else:
                FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
                    op_graph
                )

            self.assertEqual(
                mod_to_device(*test_vals_to_device), script_mod(*test_vals_to_device)
            )

    def test_optimize_freeze_module(self):
        # [FUNCTION DOC] Checks the optimize_numerics flag of freeze. With the flag off batch norm
        # remains until run_frozen_optimizations is called explicitly; with the default it is
        # already removed after freezing.
        in_channels, out_channels = 3, 32
        conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True
        )
        bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        mod = torch.nn.Sequential(conv, bn)
        # set optimize to False here, by default freezing runs run_frozen_optimizations
        frozen_mod = torch.jit.freeze(
            torch.jit.script(mod.eval()), optimize_numerics=False
        )
        # inspect frozen mod
        FileCheck().check("batch_norm").run(frozen_mod.graph)
        torch.jit.run_frozen_optimizations(frozen_mod)
        FileCheck().check_not("batch_norm").run(frozen_mod.graph)

        # run_frozen_optimizations should be run
        frozen_mod = torch.jit.freeze(torch.jit.script(mod.eval()))
        FileCheck().check_not("batch_norm").run(frozen_mod.graph)

    def test_freeze_remove_dropout(self):
        # [FUNCTION DOC] Checks that freezing a module in eval mode removes dropout and keeps
        # outputs the same as the scripted module.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a Dropout layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a dropout layer with probability 0.5.
                super().__init__()
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the dropout layer.
                return self.dropout(x)

        mod = torch.jit.script(Net())
        # inspect mod
        torch._C._jit_pass_inline(mod.graph)
        FileCheck().check("aten::dropout").run(mod.graph)
        frozen_mod = torch.jit.freeze(mod.eval())
        FileCheck().check_not("aten::dropout").run(frozen_mod.graph)

        input = torch.randn(2)
        output_s = mod.forward(input)
        output_f = frozen_mod.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_remove_feature_dropout(self):
        # [FUNCTION DOC] Same as the dropout removal test, but for the channel-wise dropout
        # (feature dropout) layer: the op disappears after freezing and outputs stay equal.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a Dropout2d layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a Dropout2d layer with probability 0.5.
                super().__init__()
                self.dropout = nn.Dropout2d(0.5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the dropout layer.
                return self.dropout(x)

        mod = torch.jit.script(Net().eval())
        # inspect mod
        torch._C._jit_pass_inline(mod.graph)
        FileCheck().check("aten::feature_dropout").run(mod.graph)
        frozen_mod = torch.jit.freeze(mod)
        FileCheck().check_not("aten::feature_dropout").run(frozen_mod.graph)

        input = torch.randn(2, 2, 1, 1)
        output_s = mod.forward(input)
        output_f = frozen_mod.forward(input)
        self.assertEqual(output_s, output_f)

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_freeze_mkdlnn(self):
        # [FUNCTION DOC] Converts a Conv2d to its mkldnn form, scripts and freezes it, and checks
        # that its output on an mkldnn tensor, converted back to dense, equals the regular conv
        # result. Skipped when MKL-DNN is unavailable.
        conv = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2).eval().float()
        convmkl = mkldnn_utils.to_mkldnn(conv)
        out = torch.jit.freeze(torch.jit.script(convmkl.eval()))
        inp = torch.rand([4, 3, 4, 4]).float()
        self.assertEqual(out(inp.to_mkldnn()).to_dense(), conv(inp))

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_conv_to_mkldnn(self):
        # [FUNCTION DOC] Checks that the convert-to-mkldnn pass is a no-op on an unfrozen graph,
        # and that after freezing a conv gets surrounded by conversion to mkldnn and back to dense
        # with an mkldnn convolution between. Loops over 2d and 3d convs. Compares to eager.
        # Skipped without MKL-DNN.
        # KNOWN ISSUE: the flag named trace selects scripting when true and tracing when false, so
        # the variable name is inverted relative to what it does.
        with set_default_dtype(torch.float):
            for module, trace in product([nn.Conv2d, nn.Conv3d], [False, True]):
                mod = module(3, 32, kernel_size=3, stride=2).eval()
                inps = [4, 3, 4]
                if module is nn.Conv2d:
                    inps.append(inps[-1])
                if module is nn.Conv3d:
                    inps.append(inps[-1])
                    inps.append(inps[-1])

                inp = torch.rand(inps)
                if trace:
                    scripted_mod = torch.jit.script(mod)
                else:
                    scripted_mod = torch.jit.trace(mod, (inp,))

                self.run_pass("inline", scripted_mod.graph)

                FileCheck().check("conv").run(scripted_mod.graph)
                # successfully no-ops with non-const inputs
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
                FileCheck().check_not("to_mkldnn").run(scripted_mod.graph)

                scripted_mod = torch.jit.freeze(scripted_mod)
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
                FileCheck().check("to_mkldnn").check("prim::mkldnn_convolution").check(
                    "to_dense"
                ).run(scripted_mod.graph)

                self.assertEqual(mod(inp), scripted_mod(inp))
                self.assertEqual(mod(inp), scripted_mod(inp))

    def test_linear_transpose(self):
        # [FUNCTION DOC] Checks the transpose pass for linear: with a constant weight
        # the linear is removed from the frozen graph (count goes from one to zero),
        # via check_linear_optimizations_2.
        class ModLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying a linear op with parameters weight and bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates random bias and weight parameters (30 outputs,
                # 20 inputs).
                super().__init__()
                self.bias = torch.nn.Parameter(torch.rand(30))
                self.weight = torch.nn.Parameter(torch.rand([30, 20]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear op to the input with the
                # stored parameters.
                return torch._C._nn.linear(x, self.weight, self.bias)

        mod_eager = ModLinear().eval()
        test_val = torch.rand([50, 20])
        self.check_linear_optimizations_2(
            mod_eager, 1, 0, "transpose_frozen_linear", (test_val,)
        )

    def test_linear_non_constant_weight(self):
        # [FUNCTION DOC] Same pass check as above but the weight is a forward argument, so the
        # linear stays in the graph (count remains one).
        class ModLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying linear with a stored bias and a weight given by
            # the caller.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a random bias parameter.
                super().__init__()
                self.bias = torch.nn.Parameter(torch.rand(30))

            def forward(self, x, weight):
                # [FUNCTION DOC] HELPER: Applies linear with the passed weight and stored bias.
                return torch._C._nn.linear(x, weight, self.bias)

        mod_eager = ModLinear().eval()
        test_val = torch.rand([50, 20])
        test_weight = torch.rand([30, 20])
        self.check_linear_optimizations_2(
            mod_eager, 1, 1, "transpose_frozen_linear", (test_val, test_weight)
        )

    def check_linear_optimizations_2(
        self, eager_mod, orig_linears, new_linears, opt_pass, test_vals
    ):
        # [FUNCTION DOC] HELPER: Scripts the module, checks the linear count equals orig_linears,
        # runs the named pass (no-op before freezing), freezes, reruns the pass and checks the
        # count equals new_linears, then compares eager and scripted outputs. Always runs on the
        # module device. A TODO says to merge it with check_linear_optimizations.
        # TODO: merge with check_linear_optimizations once both diffs land
        mod_to_device = eager_mod
        test_vals_to_device = test_vals

        script_mod = torch.jit.script(mod_to_device)
        op_graph = script_mod.graph

        FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
            op_graph
        )
        # successively no-ops with non-const inputs
        self.run_pass(opt_pass, op_graph)
        FileCheck().check_count("aten::linear", orig_linears, exactly=True).run(
            op_graph
        )

        script_mod = torch.jit.freeze(script_mod)
        op_graph = script_mod.graph
        self.run_pass(opt_pass, op_graph)
        FileCheck().check_count("aten::linear", new_linears, exactly=True).run(op_graph)

        self.assertEqual(
            mod_to_device(*test_vals_to_device), script_mod(*test_vals_to_device)
        )

    @staticmethod
    def conv():
        # [FUNCTION DOC] HELPER: Static method returning a new Conv2d with 8 input channels, 8
        # output channels and a 1x1 kernel, used as a building block.
        # Generic composable conv for testing purposes
        return nn.Conv2d(8, 8, 1)

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_collapse_adjacent_conversions(self):
        # [FUNCTION DOC] Builds two convs in sequence, freezes and converts to mkldnn, and checks
        # that only one conversion to mkldnn exists (adjacent conversions are collapsed) and
        # outputs match eager. Skipped without MKL-DNN.
        with set_default_dtype(torch.float):
            mod = nn.Sequential(self.conv(), self.conv()).eval()
            scripted_mod = torch.jit.script(mod)
            scripted_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
            FileCheck().check("to_mkldnn").check("prim::mkldnn_convolution").check(
                "prim::mkldnn_convolution"
            ).check("to_dense").run(scripted_mod.graph)
            FileCheck().check_count("to_mkldnn", 1, exactly=True).run(
                scripted_mod.graph
            )

            inp = torch.rand([1, 8, 8, 8])
            self.assertEqual(scripted_mod(inp), mod(inp))
            self.assertEqual(scripted_mod(inp), mod(inp))

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_mkldnn_fuser_broadcasting(self):
        # [FUNCTION DOC] Checks mkldnn conversion of conv followed by an add of a broadcast
        # tensor: the graph contains a broadcast-mkldnn op and outputs match eager, and that plain
        # mkldnn tensors cannot be added with broadcasting (a RuntimeError is expected). Skipped
        # without MKL-DNN.
        class Add(nn.Module):
            # [CLASS DOC] HELPER: Module adding a fixed tensor to its input.
            def __init__(self, tensor):
                # [FUNCTION DOC] HELPER: Stores the tensor operand.
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the stored tensor.
                return x + self.tensor

        with set_default_dtype(torch.float):
            for add_inp in [8], [8, 8, 1]:
                mod = nn.Sequential(self.conv(), Add(torch.rand(add_inp))).eval()
                scripted_mod = torch.jit.script(mod)
                scripted_mod = torch.jit.freeze(scripted_mod)
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
                FileCheck().check("prim::BroadcastMKLDNNTensors").run(
                    scripted_mod.graph
                )
                inp = torch.rand([1, 8, 8, 8])
                self.assertEqual(scripted_mod(inp), mod(inp))
                self.assertEqual(scripted_mod(inp), mod(inp))

                # for good measure, check that broadcasting does not work without this op
                # so we can remove the op if it ever gets supported
                with self.assertRaisesRegex(RuntimeError, ""):
                    (
                        torch.rand([1, 8, 8, 8]).to_mkldnn()
                        + torch.rand(add_inp).to_mkldnn()
                    )

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_mkldnn_inplace_removal(self):
        # [FUNCTION DOC] Checks mkldnn conversion with in-place ops: after conversion the graph
        # still has the add and div in-place ops following the conversion, and outputs match eager.
        # Skipped without MKL-DNN.
        class AddMul(nn.Module):
            # [CLASS DOC] HELPER: Module with in-place add and div by a tensor followed by
            # subtracting 4.
            def __init__(self, tensor):
                # [FUNCTION DOC] HELPER: Stores the tensor operand.
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the tensor in place, divides in place by it, and
                # subtracts 4.
                return x.add_(self.tensor).div_(self.tensor) - 4

        with set_default_dtype(torch.float):
            mod = nn.Sequential(self.conv(), AddMul(torch.rand([8]))).eval()
            scripted_mod = torch.jit.script(mod)
            scripted_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
            # add gets uninplaced and reinplaced
            FileCheck().check("aten::to_mkldnn").check("aten::add_").check(
                "aten::div_"
            ).run(scripted_mod.graph)
            inp = torch.rand([1, 8, 8, 8])
            self.assertEqual(scripted_mod(inp), mod(inp))
            self.assertEqual(scripted_mod(inp), mod(inp))

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    @skipIfNoTorchVision
    def test_maxpool_mkldnn(self):
        # [FUNCTION DOC] Takes the conv, batch norm, relu and max pool part of a torchvision
        # resnet, freezes it and converts to mkldnn. Checks that the graph has a max pool
        # followed by exactly one conversion to dense and that outputs match eager. Requires
        # MKL-DNN and torchvision.
        with set_default_dtype(torch.float):
            model = torchvision.models.resnet18()
            sub_model = torch.nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool
            )
            mod = torch.jit.freeze(torch.jit.script(sub_model.eval()))
            (
                N,
                C,
                H,
                W,
            ) = (
                10,
                3,
                224,
                224,
            )
            inp = torch.randn(N, C, H, W)
            self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
            FileCheck().check("max_pool").check("to_dense").run(mod.graph)
            FileCheck().check_count("to_dense", 1, exactly=True).run(mod.graph)
            self.assertEqual(mod(inp), sub_model(inp))

    @unittest.skipIf(torch.backends.mkldnn.is_available(), "Testing no mkldnn")
    def test_conv_to_mkldnn_no_mkldnn(self):
        # [FUNCTION DOC] Runs only when MKL-DNN is unavailable: checks that freezing a Conv2d and
        # running the mkldnn conversion pass does not raise and does not change the output.
        # test no error when mkldnn not available
        with set_default_dtype(torch.float):
            mod = torch.jit.script(nn.Conv2d(3, 32, kernel_size=3, stride=2).eval())
            frozen = torch.jit.freeze(mod)
            self.run_pass("convert_frozen_ops_to_mkldnn", frozen.graph)
            inp = torch.rand([4, 3, 4, 4])
            self.assertEqual(frozen(inp), mod(inp))

    @tf32_on_and_off(0.005)
    @unittest.skipIf(not (TEST_CUDNN or TEST_WITH_ROCM), "requires CUDNN")
    def test_freeze_conv_relu_fusion(self):
        # [FUNCTION DOC] Parametrized by loop over conv bias, conv dimensionality, whether an extra
        # tensor is added, and tracing vs scripting. Begins the conv-relu fusion test with a module
        # of a conv, in-place relu and optional add flag; the rest continues below. Requires CUDNN
        # or ROCm and runs with tf32 both on and off.
        with set_default_dtype(torch.float):
            conv_bias = [True, False]
            conv_ops = [nn.Conv2d, nn.Conv3d]
            use_add_z = [True, False]
            use_tracing = [True, False]
            for use_bias, conv, add_z, tracing in product(
                conv_bias, conv_ops, use_add_z, use_tracing
            ):

                class Net(nn.Module):
                    # [CLASS DOC] HELPER: Module with a conv, an in-place ReLU and an add flag.
                    def __init__(self, in_channels, out_channels, **kwargs):
                        # [FUNCTION DOC] HELPER: Creates the conv, the relu and stores the
                        # add flag.
                        super().__init__()
                        self.conv = conv(
                            in_channels, out_channels, bias=use_bias, **kwargs
                        )
                        self.relu = nn.ReLU(inplace=True)
                        self.add_z = add_z

                    def forward(self, x):
                        # [FUNCTION DOC] HELPER: Computes the conv twice on the input, optionally
                        # adds the first result into the second in place, and returns the relu of
                        # the result.
                        z = self.conv(x)
                        out = self.conv(x)
                        if self.add_z:
                            out += z
                        out = self.relu(out)
                        return out

                mod_eager = Net(3, 6, kernel_size=3, stride=2).eval().cuda()

                inps = [5, 3, 4, 4]
                if conv is nn.Conv3d:
                    inps.append(inps[-1])
                inp = torch.rand(inps).cuda()

                if tracing:
                    scripted_mod = torch.jit.trace(mod_eager, (inp))
                else:
                    scripted_mod = torch.jit.script(mod_eager)

                frozen_mod = torch.jit.optimize_for_inference(scripted_mod)
                if TEST_WITH_ROCM:
                    if add_z:
                        FileCheck().check("aten::miopen_convolution_add_relu").run(
                            frozen_mod.graph
                        )
                    else:
                        FileCheck().check("aten::miopen_convolution_relu").run(
                            frozen_mod.graph
                        )
                else:
                    if add_z:
                        FileCheck().check("aten::cudnn_convolution_add_relu").run(
                            frozen_mod.graph
                        )
                    else:
                        FileCheck().check("aten::cudnn_convolution_relu").run(
                            frozen_mod.graph
                        )

                self.assertEqual(mod_eager(inp), frozen_mod(inp))

    @unittest.skipIf(not (TEST_CUDNN or TEST_WITH_ROCM), "requires CUDNN")
    def test_freeze_conv_relu_fusion_not_forward(self):
        # [FUNCTION DOC] Checks conv-relu fusion for an exported non-forward method. Freezes with
        # make_prediction preserved, runs optimize_for_inference on that method, and checks the
        # graph contains the fused cudnn (or miopen on ROCm) conv-relu op and that outputs match
        # eager. Requires CUDNN or ROCm.
        with set_default_dtype(torch.float):

            class Net(nn.Module):
                # [CLASS DOC] HELPER: Module with a bias-free conv, in-place relu, and an
                # exported method.
                def __init__(self, in_channels, out_channels, **kwargs):
                    # [FUNCTION DOC] HELPER: Creates a bias-free Conv2d and an in-place ReLU.
                    super().__init__()
                    self.conv = nn.Conv2d(
                        in_channels, out_channels, bias=None, **kwargs
                    )
                    self.relu = nn.ReLU(inplace=True)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Computes conv twice, applies relu to the second result
                    # and returns it.
                    z = self.conv(x)
                    out = self.conv(x)
                    out = self.relu(out)
                    return out

                @torch.jit.export
                def make_prediction(self, x):
                    # [FUNCTION DOC] HELPER: Exported; returns forward of the input.
                    return self.forward(x)

            mod_eager = Net(3, 6, kernel_size=3, stride=2).eval().cuda()

            inps = [5, 3, 4, 4]
            inp = torch.rand(inps).cuda()

            scripted_mod = torch.jit.script(mod_eager)

            frozen_mod = torch.jit.freeze(
                scripted_mod, preserved_attrs=["make_prediction"]
            )
            optimized_mod = torch.jit.optimize_for_inference(
                frozen_mod, other_methods=["make_prediction"]
            )
            if TEST_WITH_ROCM:
                FileCheck().check("aten::miopen_convolution_relu").run(
                    optimized_mod.make_prediction.graph
                )
            else:
                FileCheck().check("aten::cudnn_convolution_relu").run(
                    optimized_mod.make_prediction.graph
                )

            self.assertEqual(
                mod_eager.make_prediction(inp), optimized_mod.make_prediction(inp)
            )

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_numel_less_than_size_with_padding(self):
        # [FUNCTION DOC] Regression test for the mkldnn path where a convolution with padding and
        # dilation applies to a very small input. Traces the module, runs optimize_for_inference,
        # and checks that every output matches eager. Skipped without MKL-DNN.
        with set_default_dtype(torch.float):

            class MyModule(nn.Module):
                # [CLASS DOC] HELPER: Module with one padded, dilated convolution returning the
                # elementwise max against the input and the clipped conv output.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the convolution with kernel (2, 4), stride 2,
                    # padding 2 and dilation (2, 1).
                    super().__init__()
                    self.conv1 = nn.Conv2d(
                        1,
                        2,
                        kernel_size=(2, 4),
                        stride=2,
                        padding=2,
                        dilation=(2, 1),
                    )

                def forward(self, i0):
                    # [FUNCTION DOC] HELPER: Returns the max of conv output and the input, and the
                    # conv output clipped to the range minus 1.5 to 1.5.
                    x = self.conv1(i0)
                    o0 = torch.max(x, i0)
                    o1 = torch.clip(x, -1.5, 1.5)
                    return o0, o1

            i0 = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
            mod = MyModule()
            out = mod(i0)

            exported = torch.jit.trace(mod, [i0])
            exported = torch.jit.optimize_for_inference(exported)

            eout = exported(i0)
            self.assertTrue(all(torch.allclose(x, y) for x, y in zip(out, eout)))

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_incompatible_perf_formats(self):
        # [FUNCTION DOC] Checks that the mkldnn conversion handles a conv followed by a max pool
        # and a sum of both outputs, where the layouts are incompatible. Freezes, converts, and
        # compares against eager. Skipped without MKL-DNN.
        with set_default_dtype(torch.float):

            class Mod(nn.Module):
                # [CLASS DOC] HELPER: Module with a convolution and a very large max pool whose
                # outputs are added.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates a Conv2d (3 to 64 channels) and a MaxPool2d
                    # with kernel 111.
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 64, 3, 2)
                    self.max_pool = torch.nn.MaxPool2d(111, 111)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns the conv output plus its max pooled version.
                    a = self.conv(x)
                    b = self.max_pool(a)
                    return a + b

            model = Mod()
            model.eval()
            mod = torch.jit.freeze(torch.jit.script(model))
            (
                N,
                C,
                H,
                W,
            ) = (
                10,
                3,
                224,
                224,
            )
            inp = torch.randn(N, C, H, W)
            self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
            self.assertEqual(model(inp), mod(inp))

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_pool2d_batchnorm(self):
        # [FUNCTION DOC] Checks mkldnn conversion of a conv, relu, 2d pooling or batch norm,
        # hardswish stack for each of four layer choices. Removes exception nodes from the graph
        # and runs dead code elimination so the size check is gone, converts, and checks the graph
        # ends with a single conversion to dense just before return, with outputs equal to eager.
        # Skipped without MKL-DNN.
        with set_default_dtype(torch.float):
            pooling_layers = [
                torch.nn.AdaptiveAvgPool2d(4),
                # torch.nn.AdaptiveMaxPool2d(4), # return tuples
                torch.nn.MaxPool2d(4),
                torch.nn.AvgPool2d(4),
                torch.nn.BatchNorm2d(64).eval(),
            ]

            for pl in pooling_layers:
                sub_model = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 2, 2),
                    torch.nn.ReLU(),
                    pl,
                    torch.nn.Hardswish(),
                )
                sub_model.eval()
                mod = torch.jit.freeze(torch.jit.script(sub_model))
                (
                    N,
                    C,
                    H,
                    W,
                ) = (
                    10,
                    3,
                    224,
                    224,
                )
                inp = torch.randn(N, C, H, W)
                # these two passes needed to remove
                # a size check in BatchNorm2d
                removeExceptions(mod.graph)
                self.run_pass("dce", mod.graph)
                self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
                FileCheck().check("aten::to_dense").check_next("return").run(mod.graph)
                self.assertEqual(sub_model(inp), mod(inp))

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_pool3d_batchnorm(self):
        # [FUNCTION DOC] 3d analogue of the previous test using conv3d with max pool, average pool
        # or batch norm. Same graph cleanup and checks: one conversion back to dense right before
        # the return, and outputs match eager. Skipped without MKL-DNN.
        with set_default_dtype(torch.float):
            pooling_layers = [
                torch.nn.MaxPool3d(4),
                # torch.nn.AdaptiveAvgPool3d(4), # no ideep bindings
                # torch.nn.AdaptiveMaxPool3d(4), # return tuples
                torch.nn.AvgPool3d(4),
                torch.nn.BatchNorm3d(64).eval(),
            ]

            for pl in pooling_layers:
                sub_model = torch.nn.Sequential(
                    torch.nn.Conv3d(3, 64, 2, 2),
                    torch.nn.ReLU(),
                    pl,
                    torch.nn.Hardswish(),
                )
                sub_model.eval()
                mod = torch.jit.freeze(torch.jit.script(sub_model))
                N, C, H, W, D = 10, 3, 64, 64, 64
                inp = torch.randn(N, C, D, H, W)
                # these two passes needed to remove
                # a size check in BatchNorm2d
                removeExceptions(mod.graph)
                self.run_pass("dce", mod.graph)
                self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
                FileCheck().check("aten::to_dense").check_next("return").run(mod.graph)
                self.assertEqual(sub_model(inp), mod(inp))

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    @skipIfNoTorchVision
    def test_conv_hardswish(self):
        # [FUNCTION DOC] Runs the first resnet18 conv followed by each of several activations (hard
        # swish, hard sigmoid, relu6, tanh, hardtanh variants, gelu and clamps via a helper
        # module). For each, freezes, converts to mkldnn, checks there is exactly one conversion to
        # dense, and compares to eager. Requires MKL-DNN and torchvision.
        with set_default_dtype(torch.float):

            class Clamp(torch.nn.Module):
                # [CLASS DOC] HELPER: Module that clamps its input to a fixed minimum and maximum.
                def __init__(self, min_val, max_val, **kwargs):
                    # [FUNCTION DOC] HELPER: Stores the clamp bounds; extra keyword arguments are
                    # accepted and ignored.
                    super().__init__()
                    self.min_val = min_val
                    self.max_val = max_val

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns the input clamped to the stored bounds.
                    return torch.clamp(x, self.min_val, self.max_val)

            (
                N,
                C,
                H,
                W,
            ) = (
                10,
                3,
                224,
                224,
            )
            activations = [
                torch.nn.Hardswish(),
                torch.nn.Hardsigmoid(),
                torch.nn.ReLU6(),
                torch.nn.Tanh(),
                torch.nn.Hardtanh(0.0, 6.0),
                torch.nn.Hardtanh(1.0, 100.0),
                torch.nn.Hardtanh(-100.0, -1.0),
                torch.nn.GELU(),
                Clamp(-100.0, -1.0),
                Clamp(1.0, 100.0),
                Clamp(0.0, 6.0),
                Clamp(-1.0, 0.0),
            ]

            model = torchvision.models.resnet18()
            for activation in activations:
                sub_model = torch.nn.Sequential(model.conv1, activation)
                sub_model.eval()
                mod = torch.jit.freeze(torch.jit.script(sub_model))
                inp = torch.randn(N, C, H, W)
                self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
                FileCheck().check_count("aten::to_dense", 1, exactly=True).run(
                    mod.graph
                )
                self.assertEqual(sub_model(inp), mod(inp))

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_hardswish_hardsigmoid(self):
        # [FUNCTION DOC] Builds small IR graphs that convert a tensor to mkldnn and apply the
        # mkldnn hardswish or hardsigmoid op (in-place or not), turns each into a function, and
        # checks the result converted back to dense equals the regular functional op for several
        # input sizes. Skipped without MKL-DNN.
        with set_default_dtype(torch.float):
            op_map = {
                "prim::MKLDNNHardSwish": F.hardswish,
                "prim::MKLDNNHardSigmoid": F.hardsigmoid,
            }

            input_sizes = ([0], [1], [3], [1, 3, 8, 8])
            for mkldnn_opname, aten_op in op_map.items():
                for size in input_sizes:
                    for inplace in (True, False):
                        inplace_str = "_" if inplace else ""
                        inplace_tgt = "%34" if inplace else "%35"
                        graph_str = f"""graph(%input.1 : Tensor):
                            %33 : None = prim::Constant()
                            %34 : Tensor = aten::to_mkldnn(%input.1, %33)
                            %35 : Tensor = {mkldnn_opname}{inplace_str}(%34)
                            return ({inplace_tgt})
                        """
                        g = torch._C.parse_ir(graph_str)
                        m = self.createFunctionFromGraph(g)
                        x = torch.rand(size)
                        # `inplace=False` is intentional, otherwise we modify the input
                        # and we aren't testing aten impls anyways
                        self.assertEqual(aten_op(x, inplace=False), m(x).to_dense())

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_scalar_mul(self):
        # [FUNCTION DOC] Freezes a conv followed by scalar multiplications, optimizes for
        # inference, and checks the graph has one scalar multiply that is not in place followed by
        # one that is in place, and that outputs match eager. Skipped without MKL-DNN.
        with set_default_dtype(torch.float):

            class Mod(nn.Module):
                # [CLASS DOC] HELPER: Module with a convolution followed by repeated
                # scalar multiplications.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates a Conv2d with 8 channels, kernel 1 and
                    # padding 1.
                    super().__init__()
                    self.mod = nn.Conv2d(8, 8, 1, padding=1)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Multiplies the conv output by 4, then returns that
                    # times 4 plus the same value times 5.0.
                    a1 = self.mod(x) * 4
                    return a1 * 4 + a1 * 5.0

            mod = Mod().eval()
            scripted = torch.jit.freeze(torch.jit.script(mod))
            optimized = torch.jit.optimize_for_inference(scripted)
            inp = torch.rand([1, 8, 8, 8])
            # a1 can't be inplaced for first use, can for second
            FileCheck().check("ScalarMul(").check("ScalarMul_").run(optimized.graph)
            self.assertEqual(optimized(inp), mod(inp))

    def test_remove_detach(self):
        # [FUNCTION DOC] Checks that freezing removes a detach call whose result is only
        # multiplied, and that outputs match eager.
        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Module that detaches its input and multiplies it by itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the product of the detached input with itself.
                y = x.detach()
                return y * y

        mod = Mod().eval()
        frozen_mod = torch.jit.freeze(torch.jit.script(mod))
        inp = torch.randn((2, 2))
        FileCheck().check_not("aten::detach").run(frozen_mod.graph)
        self.assertEqual(frozen_mod(inp), mod(inp))

    def test_remove_detach_not_applied(self):
        # [FUNCTION DOC] Checks that freezing keeps the detach op when its result is compared
        # by identity with the input (so removing it would change behavior), and that outputs
        # match eager.
        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Module that detaches its input and returns whether the result is
            # the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns whether the detached tensor is the same object as
                # the input.
                y = x.detach()
                return x is y

        mod = Mod().eval()
        frozen_mod = torch.jit.freeze(torch.jit.script(mod))
        inp = torch.randn((2, 2))
        FileCheck().check("aten::detach").run(frozen_mod.graph)
        self.assertEqual(frozen_mod(inp), mod(inp))


@skipIfTorchDynamo("somehow causing hanging during python shutdown")
@unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled")
class TestMKLDNNReinplacing(JitTestCase):
    # [CLASS DOC] Tests for the mkldnn in-place reinplacing pass after freezing and converting.
    # Skipped without MKL-DNN or under TorchDynamo. The default dtype is switched to float for
    # each test.
    def setUp(self):
        # [FUNCTION DOC] Saves the default dtype and sets the default dtype to float.
        super().setUp()
        self.default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float)

    def tearDown(self):
        # [FUNCTION DOC] Runs the base tearDown, then restores the saved default dtype.
        super().tearDown()
        torch.set_default_dtype(self.default_dtype)

    def getConv(self):
        # [FUNCTION DOC] HELPER: Returns a new eval-mode Conv2d with 3 input and 32 output
        # channels, kernel 3 and stride 2.
        return nn.Conv2d(3, 32, kernel_size=3, stride=2).eval()

    def getInput(self):
        # [FUNCTION DOC] HELPER: Returns a random 4x3x4x4 input tensor.
        return torch.rand([4, 3, 4, 4])

    def freezeAndConvert(self, mod):
        # [FUNCTION DOC] HELPER: Scripts the module in eval mode, freezes it, runs the mkldnn
        # conversion pass on its graph, and returns the frozen module.
        mod = torch.jit.freeze(torch.jit.script(mod.eval()))
        self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
        return mod

    def checkResults(self, mod1, mod2):
        # [FUNCTION DOC] HELPER: Runs both modules on a fresh random input and asserts
        # equal outputs.
        inp = self.getInput()
        self.assertEqual(mod1(inp), mod2(inp))

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/91489")
    def test_successful(self):
        # [FUNCTION DOC] Checks that conv followed by hardswish and relu becomes an mkldnn
        # convolution followed by in-place hardswish and in-place relu, with equal outputs. Skipped
        # on Linux because of a linked issue.
        # simple conv-relu

        mod_eager = nn.Sequential(self.getConv(), nn.Hardswish(), nn.ReLU())
        mod = self.freezeAndConvert(mod_eager)
        FileCheck().check("mkldnn_convolution").check_next(
            "prim::MKLDNNHardSwish_"
        ).check_next("aten::relu_").run(mod.graph)
        self.checkResults(mod_eager, mod)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/91481")
    def test_merge_liveness(self):
        # [FUNCTION DOC] Checks liveness handling: the multiply in the module following the
        # conv can be in place, while the add of a value that stays alive until return must not
        # be. Verifies via graph checks and output equality. Skipped on Linux because of a
        # linked issue.
        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Module multiplying its input by a tensor and returning a sum and
            # the temporary.
            def __init__(self, tensor):
                # [FUNCTION DOC] HELPER: Stores the tensor operand.
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiplies by the tensor and returns the temporary doubled
                # together with the temporary itself.
                # this mul can be inplaced since x is dead after this use
                temporary = x * self.tensor
                # temporary livespan is the return node,
                # add can not be inplaced
                return temporary + temporary, temporary

        mod_eager = nn.Sequential(self.getConv(), Mod(torch.rand([4, 32, 1, 1])))
        mod = self.freezeAndConvert(mod_eager)
        FileCheck().check("aten::mul_").check_not("aten::add_").run(mod.graph)
        self.checkResults(mod_eager, mod)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/91486")
    def test_always_alive_values(self):
        # [FUNCTION DOC] Checks that the reinplacing pass does not modify values that are always
        # alive (graph returns or constants). First module returns its input as well as a product,
        # so no in-place multiply appears; second module applies a shared conv twice and no
        # in-place add appears for the add of the graph input. Skipped on Linux because of a
        # linked issue.
        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Module returning the product of the input and a tensor, and
            # the input.
            def __init__(self, tensor):
                # [FUNCTION DOC] HELPER: Stores the tensor operand.
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times the tensor, and the input itself.
                # x can't be inplaced because its a return value,
                # check that the inplacing pass doesn't try to inplace
                # self.tensor because its always alive
                return x * self.tensor, x

        mod_eager = nn.Sequential(self.getConv(), Mod(torch.rand([4, 32, 1, 1])))
        mod = self.freezeAndConvert(mod_eager)
        FileCheck().check_not("aten::mul_").run(mod.graph)
        self.checkResults(mod_eager, mod)

        conv = self.getConv()

        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Module applying a shared conv to the input and to the input added
            # to itself.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a random tensor and keeps the shared conv from
                # the test.
                super().__init__()
                self.tensor = torch.rand([4, 32, 1, 1])
                self.conv = conv

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the conv of the input and the conv of the input
                # plus itself.
                # the shapes dont add up on this just testing a particular pattern
                conv_output = self.conv(x)
                return conv_output, self.conv(torch.add(x, x))

        mod = self.freezeAndConvert(Mod())
        # x is an input to the graph, and so it should not be inplaced
        # in the torch.add(x, x) call
        FileCheck().check_not("aten::add_").run(mod.graph)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/91488")
    def test_switch_inputs_to_inplace(self):
        # [FUNCTION DOC] Checks that when one add operand cannot be modified in place, the
        # pass swaps operands so the dead one (the conv output) is updated in place. Verifies
        # an in-place add in the graph and output equality. Skipped on Linux because of a
        # linked issue.
        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Module returning a stored tensor plus its input (operands in
            # swapped order).
            def __init__(self, tensor):
                # [FUNCTION DOC] HELPER: Stores the tensor operand.
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the stored tensor plus the input.
                # self.tensor cannot be inplaced, however x can,
                # and bc add is commutative we can reverse inputs to add_
                return self.tensor + x

        mod_eager = nn.Sequential(self.getConv(), Mod(torch.rand([4, 32, 1, 1])))
        mod = self.freezeAndConvert(mod_eager)
        FileCheck().check("aten::add_").run(mod.graph)
        self.checkResults(mod_eager, mod)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
